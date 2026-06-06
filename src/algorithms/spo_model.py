"""
SPO agent for current SVO-RL codebase.

Background
----------
SPO (Simple Policy Optimization) is proposed by Xie et al. (ICML 2025).
Compared to PPO, SPO replaces the heuristic ratio-clipping technique with a
quadratic penalty on (ratio - 1), which is theoretically ε-aligned and
empirically more effective at constraining ratio deviations during training.

Reference repository:
    https://github.com/MyRepositories-hub/Simple-Policy-Optimization
The core SPO objective (mujoco/trainer.py L106-109) is:
    policy_loss = -(advantages * ratios
                    - |advantages| * (ratios - 1)^2 / (2*eps)).mean()

Project-specific extensions (论文 novelty)
------------------------------------------
1) **CVaR integration** via Lagrangian combined advantage
       A_eff = A_R - lambda * cvar_cost_coef * A_C
   The single SPO penalty is applied to A_eff, sharing one trust region for
   the reward and cost paths instead of two inconsistent ratio-clippings.

2) **SVO-guided adaptive ε**
       ε_t = clip(ε_base * exp(-alpha * ρ_t), ε_min, ε_max)
   where ρ_t is the per-step SVO interaction urgency from the rollout buffer
   (already used by the CVaR module as svo_budget_term). High urgency →
   smaller ε → tighter trust region in risky interactions.

This implementation mirrors DAPOCarlaAgent: subclass PPOAgent and only
override update(). All other interfaces (select_action, save/load, SVO/SLT
hooks, rollout buffer) stay identical to PPOAgent so train.py/test.py do
not need to special-case SPO beyond the agent dispatch.
"""

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from src.algorithms.ppo_model import PPOAgent


class SPOCarlaAgent(PPOAgent):
    """
    SPO variant based on PPOAgent.

    Why subclass PPOAgent:
    - Keeps the project's PPO interfaces identical (select_action returns
      7 items, buffer schema with SVO fields, save/load metadata, SLT hooks).
    - Minimizes integration risk in train.py/test.py.
    - Only the policy loss in update() differs from PPO.
    """

    def __init__(self, config):
        super().__init__(config)
        ppo_cfg = self.config.ppo
        # SPO hyperparameters (read from config; defaults set in PPOConfig)
        self.spo_eps_base = float(ppo_cfg.spo_epsilon_base)
        self.spo_eps_min = float(ppo_cfg.spo_epsilon_min)
        self.spo_eps_max = float(ppo_cfg.spo_epsilon_max)
        self.spo_alpha = float(ppo_cfg.spo_risk_alpha)
        self.spo_adaptive_enabled = bool(ppo_cfg.use_svo_adaptive_spo)
        self.config.ppo.use_spo = True

        _mode = "SVO-Adaptive" if self.spo_adaptive_enabled else "Fixed"
        print(
            f"[SPO] Simple Policy Optimization enabled ({_mode}): "
            f"ε_base={self.spo_eps_base:.3f}, "
            f"ε_range=[{self.spo_eps_min:.3f}, {self.spo_eps_max:.3f}], "
            f"α={self.spo_alpha:.2f}"
        )

    # ================================================================== #
    #  Helper: per-sample ε for SPO                                       #
    # ================================================================== #

    def _compute_spo_epsilon(self, data: dict, batch_idx: np.ndarray):
        """
        Compute the SPO trust-region width ε for the current mini-batch.

        Returns
        -------
        eps_t : torch.Tensor (B,) or float
            Per-sample ε (adaptive mode) or scalar (fixed mode).
        rho_mean, rho_max : float
            Stats of the SVO urgency batch (0 in fixed mode, for logging).
        """
        adaptive = (
            self.spo_adaptive_enabled
            and self.use_svo_game
            and 'svo_budget_terms' in data
        )
        if adaptive:
            rho = data['svo_budget_terms'][batch_idx].clamp(0.0, 1.0)   # (B,)
            eps_t = self.spo_eps_base * torch.exp(-self.spo_alpha * rho)
            eps_t = eps_t.clamp(self.spo_eps_min, self.spo_eps_max).clamp(min=1e-6)
            return eps_t, rho.mean().item(), rho.max().item()
        # fixed-ε fallback
        return float(self.spo_eps_base), 0.0, 0.0

    # ================================================================== #
    #  SPO update (mirrors PPOAgent.update structure, swaps actor loss)   #
    # ================================================================== #

    def update(self, last_state: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        SPO policy update.

        Structure mirrors PPOAgent.update. The only changes are:
          (a) actor loss uses SPO quadratic penalty on Lagrangian-combined
              advantage A_eff = A_R - λ·k_c·A_C  (instead of PPO clipping)
          (b) per-sample ε from SVO posterior (when adaptive)
          (c) cost_actor_loss is folded into A_eff and not added separately
          (d) extra logging for ratio_deviation / ε / SVO risk
        """
        ppo_cfg = self.config.ppo

        if self.buffer.ptr <= 0:
            return {
                "actor_loss": 0.0,
                "critic_loss": 0.0,
                "entropy": 0.0,
                "lr": self.ppo_optimizer.param_groups[0]["lr"],
            }

        # ---- Bootstrap value for GAE ----
        last_idx = self.buffer.ptr - 1
        last_done = bool(self.buffer.dones[last_idx] > 0.5)
        last_cost_value = 0.0
        if last_done:
            last_value = 0.0
        else:
            with torch.no_grad():
                if last_state is not None:
                    bootstrap_state = torch.FloatTensor(last_state).unsqueeze(0).to(self.device)
                    bootstrap_features = self._encode(bootstrap_state)
                else:
                    bootstrap_state = torch.FloatTensor(
                        self.buffer.states[last_idx]
                    ).unsqueeze(0).to(self.device)
                    if self.use_svo_game:
                        bootstrap_features = self._encode(
                            bootstrap_state,
                            torch.FloatTensor(self.buffer.svo_mu[last_idx]).unsqueeze(0).to(self.device),
                            torch.FloatTensor(self.buffer.svo_sigma[last_idx]).unsqueeze(0).to(self.device),
                            torch.FloatTensor(self.buffer.pred_trajs[last_idx]).unsqueeze(0).to(self.device),
                            torch.BoolTensor(self.buffer.interact_mask[last_idx]).unsqueeze(0).to(self.device),
                        )
                    else:
                        bootstrap_features = self._encode(bootstrap_state)

                last_value = self.critic_head(bootstrap_features).cpu().item()
                if self.use_cvar:
                    last_cost_value = self.cost_critic_head(bootstrap_features).cpu().item()

        # ---- GAE ----
        returns, advantages = self.buffer.compute_returns(
            last_value, ppo_cfg.gamma, ppo_cfg.gae_lambda
        )
        cost_returns = cost_advantages = None
        cvar_stats = {}
        if self.use_cvar:
            cost_returns, cost_advantages = self.buffer.compute_cost_returns(
                last_cost_value, ppo_cfg.cost_gamma, ppo_cfg.gae_lambda
            )
            cvar_stats = self._cvar_dual_update()

        data = self.buffer.get_data(returns, advantages, cost_returns, cost_advantages)

        if ppo_cfg.normalize_advantage:
            data["advantages"] = (
                (data["advantages"] - data["advantages"].mean())
                / (data["advantages"].std() + 1e-8)
            )
            if self.use_cvar:
                data["cost_advantages"] = (
                    (data["cost_advantages"] - data["cost_advantages"].mean())
                    / (data["cost_advantages"].std() + 1e-8)
                )

        # ---- Mini-batch loop ----
        n_samples = len(data["states"])
        batch_size = ppo_cfg.mini_batch_size
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_cost_critic_loss = 0.0
        total_entropy = 0.0
        n_updates = 0
        # SPO 监测 (论文 Figure 6 / Table 1 的核心指标)
        total_ratio_dev_mean = 0.0
        total_ratio_dev_max = 0.0
        total_eps_sum = 0.0
        total_eps_min = float("inf")
        total_eps_max = -float("inf")
        total_rho_sum = 0.0
        total_rho_max = 0.0

        for _ in range(ppo_cfg.ppo_epochs):
            indices = np.random.permutation(n_samples)

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_idx = indices[start:end]

                batch_states = data["states"][batch_idx]
                batch_actions = data["actions"][batch_idx]
                batch_old_log_probs = data["log_probs"][batch_idx]
                batch_returns = data["returns"][batch_idx]
                batch_advantages = data["advantages"][batch_idx]

                # Single shared encoder forward
                if self.use_svo_game:
                    features = self._encode(
                        batch_states,
                        data["svo_mu"][batch_idx],
                        data["svo_sigma"][batch_idx],
                        data["pred_trajs"][batch_idx],
                        data["interact_mask"][batch_idx],
                    )
                else:
                    features = self._encode(batch_states)

                new_log_probs, entropy = self.actor_head.evaluate(features, batch_actions)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # === Per-sample ε (SVO-adaptive or fixed) ===
                eps_t, rho_mean, rho_max = self._compute_spo_epsilon(data, batch_idx)

                # === Lagrangian combined advantage (CVaR integration) ===
                if self.use_cvar:
                    batch_cost_adv = data["cost_advantages"][batch_idx]
                    A_eff = batch_advantages - (
                        self.cvar_lambda * ppo_cfg.cvar_cost_coef * batch_cost_adv
                    )
                else:
                    A_eff = batch_advantages

                # === SPO objective ===
                # 公式 1:1 复刻自官方仓库 mujoco/trainer.py L106-109:
                #     policy_loss = -(adv * ratios - |adv| * (ratios-1)^2 / (2*eps)).mean()
                # 项目扩展两点 (论文 novelty):
                #   (1) 用 A_eff (Lagrangian 合成) 替代 adv      → CVaR 集成
                #   (2) 用 eps_t (per-sample tensor) 替代 标量    → SVO-adaptive trust region
                spo_objective = (
                    ratio * A_eff
                    - A_eff.abs() * (ratio - 1.0).pow(2) / (2.0 * eps_t)
                )
                actor_loss = -spo_objective.mean()

                # === Cost critic loss (与 trust region 无关, 与 PPO 一致) ===
                cost_critic_loss = torch.zeros((), device=self.device)
                if self.use_cvar:
                    new_cost_values = self.cost_critic_head(features)
                    batch_old_cost_values = data["cost_values"][batch_idx]
                    batch_cost_returns = data["cost_returns"][batch_idx]
                    cv_clipped = batch_old_cost_values + torch.clamp(
                        new_cost_values - batch_old_cost_values,
                        -ppo_cfg.clip_epsilon, ppo_cfg.clip_epsilon,
                    )
                    cost_critic_loss = torch.max(
                        (new_cost_values - batch_cost_returns) ** 2,
                        (cv_clipped - batch_cost_returns) ** 2,
                    ).mean()

                # === Reward critic loss (Clipped, 与 PPO 完全一致) ===
                new_values = self.critic_head(features)
                batch_old_values = data["values"][batch_idx]
                value_clipped = batch_old_values + torch.clamp(
                    new_values - batch_old_values,
                    -ppo_cfg.clip_epsilon, ppo_cfg.clip_epsilon,
                )
                critic_loss = torch.max(
                    (new_values - batch_returns) ** 2,
                    (value_clipped - batch_returns) ** 2,
                ).mean()

                # === Total loss (cost_actor_loss 已折叠进 A_eff, 不再独立加) ===
                loss = (
                    actor_loss
                    + ppo_cfg.value_coef * critic_loss
                    - ppo_cfg.entropy_coef * entropy.mean()
                )
                if self.use_cvar:
                    loss = loss + ppo_cfg.cost_loss_coef * cost_critic_loss

                self.ppo_optimizer.zero_grad()
                loss.backward()
                _grad_params = (
                    list(self.encoder.parameters())
                    + list(self.actor_head.parameters())
                    + list(self.critic_head.parameters())
                )
                if self.use_cvar:
                    _grad_params += list(self.cost_critic_head.parameters())
                nn.utils.clip_grad_norm_(_grad_params, ppo_cfg.max_grad_norm)
                self.ppo_optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                if self.use_cvar:
                    total_cost_critic_loss += cost_critic_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

                # SPO 监测 (no_grad block)
                with torch.no_grad():
                    ratio_dev = (ratio - 1.0).abs()
                    total_ratio_dev_mean += ratio_dev.mean().item()
                    total_ratio_dev_max = max(total_ratio_dev_max, ratio_dev.max().item())
                    if torch.is_tensor(eps_t):
                        total_eps_sum += eps_t.mean().item()
                        total_eps_min = min(total_eps_min, eps_t.min().item())
                        total_eps_max = max(total_eps_max, eps_t.max().item())
                    else:
                        _e = float(eps_t)
                        total_eps_sum += _e
                        total_eps_min = min(total_eps_min, _e)
                        total_eps_max = max(total_eps_max, _e)
                    if self.spo_adaptive_enabled and self.use_svo_game:
                        total_rho_sum += rho_mean
                        total_rho_max = max(total_rho_max, rho_max)

            # Early stopping (与 PPO 一致, 防御性保留)
            if ppo_cfg.target_kl is not None:
                with torch.no_grad():
                    if self.use_svo_game:
                        all_features = self._encode(
                            data["states"],
                            data["svo_mu"],
                            data["svo_sigma"],
                            data["pred_trajs"],
                            data["interact_mask"],
                        )
                    else:
                        all_features = self._encode(data["states"])
                    new_lp, _ = self.actor_head.evaluate(all_features, data["actions"])
                    kl = (data["log_probs"] - new_lp).mean().item()
                    if kl > ppo_cfg.target_kl:
                        break

        # ---- LR scheduler ----
        if self.ppo_scheduler:
            self.ppo_scheduler.step()

        self.buffer.reset()

        # ---- Stats ----
        spo_stats = {
            "actor_loss": total_actor_loss / max(n_updates, 1),
            "critic_loss": total_critic_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
            "lr": self.ppo_optimizer.param_groups[0]["lr"],
            # SPO 论文 Figure 6 / Table 1 的核心验证指标
            "ratio_deviation_mean": total_ratio_dev_mean / max(n_updates, 1),
            "ratio_deviation_max": total_ratio_dev_max,
            "spo_epsilon_mean": total_eps_sum / max(n_updates, 1),
            "spo_epsilon_min": (
                total_eps_min if total_eps_min != float("inf") else 0.0
            ),
            "spo_epsilon_max": (
                total_eps_max if total_eps_max != -float("inf") else 0.0
            ),
        }
        if self.spo_adaptive_enabled and self.use_svo_game:
            spo_stats["svo_risk_mean"] = total_rho_sum / max(n_updates, 1)
            spo_stats["svo_risk_max"] = total_rho_max
        if self.use_cvar:
            spo_stats["cost_critic_loss"] = total_cost_critic_loss / max(n_updates, 1)
            spo_stats.update(cvar_stats)

        # ---- SLT auxiliary representation learning (与 PPO 一致) ----
        if self.use_slt and len(self.seq_queue) >= self.config.slt.batch_size:
            slt_stats = self._slt_update()
            spo_stats.update(slt_stats)

        return spo_stats