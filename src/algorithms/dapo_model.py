"""
DAPO agent adapter for current SVO-RL codebase.

This implementation intentionally reuses PPOAgent's full data flow
(encoder, SVO path, rollout buffer, save/load/select_action interface)
and only swaps the policy clipping rule in update():
    PPO:  clip(ratio, 1-eps,       1+eps)
    DAPO: clip(ratio, 1-eps_low,   1+eps_high)
"""

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from src.algorithms.ppo_model import PPOAgent


class DAPOCarlaAgent(PPOAgent):
    """
    DAPO variant based on PPOAgent.

    Why subclass PPOAgent:
    - Keeps current project interfaces identical to PPO
      (select_action returns 7 items, buffer schema with SVO fields,
      save/load metadata, reset_svo_state, SLT hooks).
    - Minimizes integration risk in train.py/test.py.
    """

    def __init__(self, config):
        super().__init__(config)
        ppo_cfg = self.config.ppo
        # Use explicit DAPO clipping if provided, otherwise fall back to PPO epsilon.
        self.clip_eps_low = float(getattr(ppo_cfg, "clip_eps_low", ppo_cfg.clip_epsilon))
        self.clip_eps_high = float(getattr(ppo_cfg, "clip_eps_high", ppo_cfg.clip_epsilon))
        self.config.ppo.use_dapo = True

        print(
            f"[DAPO] Decoupled clipping enabled: "
            f"low={self.clip_eps_low:.3f}, high={self.clip_eps_high:.3f}"
        )

    def update(self, last_state: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        DAPO policy update with decoupled clipping.
        All other training details stay aligned with PPOAgent.update.
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
        last_cost_value = 0.0   # [CVaR] 默认值, 仅 cvar_enabled 且非 done 时被覆盖
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
                # [CVaR] cost critic bootstrap
                if self.use_cvar:
                    last_cost_value = self.cost_critic_head(bootstrap_features).cpu().item()

        returns, advantages = self.buffer.compute_returns(
            last_value, ppo_cfg.gamma, ppo_cfg.gae_lambda
        )
        # [CVaR] cost path
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

        n_samples = len(data["states"])
        batch_size = ppo_cfg.mini_batch_size
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_cost_critic_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

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

                # DAPO: decoupled clipping window.
                clipped_ratio = torch.clamp(
                    ratio,
                    1.0 - self.clip_eps_low,
                    1.0 + self.clip_eps_high,
                )
                surr1 = ratio * batch_advantages
                surr2 = clipped_ratio * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # [CVaR] cost surrogate (DAPO 同样的 decoupled clipping)
                cost_actor_loss = torch.zeros((), device=self.device)
                cost_critic_loss = torch.zeros((), device=self.device)
                if self.use_cvar:
                    batch_cost_adv = data["cost_advantages"][batch_idx]
                    c_clipped_ratio = torch.clamp(
                        ratio,
                        1.0 - self.clip_eps_low,
                        1.0 + self.clip_eps_high,
                    )
                    c_surr1 = ratio * batch_cost_adv
                    c_surr2 = c_clipped_ratio * batch_cost_adv
                    cost_actor_loss = torch.max(c_surr1, c_surr2).mean()

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

                # Keep the same value clipping strategy as PPOAgent.
                new_values = self.critic_head(features)
                batch_old_values = data["values"][batch_idx]
                value_clipped = batch_old_values + torch.clamp(
                    new_values - batch_old_values,
                    -ppo_cfg.clip_epsilon,
                    ppo_cfg.clip_epsilon,
                )
                critic_loss = torch.max(
                    (new_values - batch_returns) ** 2,
                    (value_clipped - batch_returns) ** 2,
                ).mean()

                loss = (
                    actor_loss
                    + ppo_cfg.value_coef * critic_loss
                    - ppo_cfg.entropy_coef * entropy.mean()
                )
                if self.use_cvar:
                    loss = (loss
                            + self.cvar_lambda * ppo_cfg.cvar_cost_coef * cost_actor_loss
                            + ppo_cfg.cost_loss_coef * cost_critic_loss)

                self.ppo_optimizer.zero_grad()
                loss.backward()
                _grad_params = (list(self.encoder.parameters())
                                + list(self.actor_head.parameters())
                                + list(self.critic_head.parameters()))
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

            # Keep PPO-style early stopping (slightly relaxed to match prior DAPO behavior).
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
                    if kl > ppo_cfg.target_kl * 1.5:
                        break

        if self.ppo_scheduler:
            self.ppo_scheduler.step()

        self.buffer.reset()

        dapo_stats = {
            "actor_loss": total_actor_loss / max(n_updates, 1),
            "critic_loss": total_critic_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
            "lr": self.ppo_optimizer.param_groups[0]["lr"],
        }
        if self.use_cvar:
            dapo_stats["cost_critic_loss"] = total_cost_critic_loss / max(n_updates, 1)
            dapo_stats.update(cvar_stats)

        if self.use_slt and len(self.seq_queue) >= self.config.slt.batch_size:
            slt_stats = self._slt_update()
            dapo_stats.update(slt_stats)

        return dapo_stats
