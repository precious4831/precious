"""
ppo_model.py -- PPO Agent with SVO-Game Integration

========================================================================
在完整训练流程中的位置:
  本文件定义PPOAgent, 被 train.py 和 test.py 调用.
  训练: python train.py --stage l1 / svo_only / l2
  验证: python test.py  --l1_model ... --svo_model ... --l2_model ...
========================================================================

架构:
    shared_encoder → GameAwareCrossAttention → actor_head / critic_head
    svo_birl (SVO推断), klevel (K-Level预测)

核心设计:
  1. Actor/Critic 共享 Encoder (减半参数, 一致表征)
  2. 单PPO优化器 + 合并损失 (解决共享Encoder梯度冲突)
  3. SVO/K-Level数据缓存在Buffer中 (update时不重复计算)
  4. config.svo.enabled 控制开关 (False时退化为纯PPO)
  5. finetune_svo() 方法用于Stage 2联合训练

梯度冲突修复说明:
  原版: Actor有Encoder_A, Critic有Encoder_B (各自独立)
  新版: Actor和Critic共享一个Encoder
  问题: actor_loss.backward() 释放计算图 → critic_loss.backward() 崩溃
  解决: loss = actor_loss + value_coef*critic_loss - entropy_coef*entropy
        单次 loss.backward() + 单次 optimizer.step()
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from typing import Tuple, Dict, Optional

from src.config import Config
from src.models.encoder import HierarchicalTransformerEncoder, parse_observation
from src.models.svo_model import SVOVariationalBIRL
from src.models.slt import SequentialLatentTransformer, SequentialQueue


# ======================================================================== #
#  ActorHead: MLP-only (无内部Encoder)                                      #
# ======================================================================== #
# 
class ActorHead(nn.Module):
    """
    Actor策略头.

    输入: features (B, input_dim)  — 来自共享Encoder+aux拼接
    输出: mean (B, action_dim), std (B, action_dim)

    注意: 不包含Encoder, Encoder在PPOAgent层面共享.
    """

    def __init__(self, input_dim: int, action_dim: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 32),        nn.ReLU(),
        )
        self.mean_layer = nn.Linear(32, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self._init_weights()

    def _init_weights(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        nn.init.orthogonal_(self.mean_layer.weight, gain=0.01)
        nn.init.constant_(self.mean_layer.bias, 0)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.head(features)
        mean = self.mean_layer(x)
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std

    def get_action(self, features: torch.Tensor,
                   deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """采样动作 (tanh squashed Gaussian)."""
        mean, std = self.forward(features)

        if deterministic:
            action = torch.tanh(mean)
            log_prob = torch.zeros(features.shape[0], device=features.device)
        else:
            dist = Normal(mean, std)
            action_raw = dist.rsample()
            action = torch.tanh(action_raw)

            # log_prob with tanh correction
            log_prob = dist.log_prob(action_raw).sum(dim=-1)
            log_prob -= (2 * (np.log(2) - action_raw
                         - nn.functional.softplus(-2 * action_raw))).sum(dim=-1)

        return action, log_prob

    def evaluate(self, features: torch.Tensor,
                 action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """评估已有动作的log_prob和entropy."""
        mean, std = self.forward(features)
        dist = Normal(mean, std)

        action_raw = torch.atanh(torch.clamp(action, -0.999, 0.999))
        log_prob = dist.log_prob(action_raw).sum(dim=-1)
        log_prob -= (2 * (np.log(2) - action_raw
                     - nn.functional.softplus(-2 * action_raw))).sum(dim=-1)

        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


# ======================================================================== #
#  CriticHead: MLP-only                                                     #
# ======================================================================== #

class CriticHead(nn.Module):
    """
    Critic价值头.

    输入: features (B, input_dim)
    输出: value (B,)
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 32),        nn.ReLU(),
            nn.Linear(32, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features).squeeze(-1)


# ======================================================================== #
#  RolloutBuffer (扩展版, 缓存SVO/K-Level数据)                              #
# ======================================================================== #

class RolloutBuffer:
    """
    PPO经验缓冲区, 扩展支持SVO/K-Level数据缓存.

    SVO/K-Level数据在rollout阶段计算一次, 缓存在buffer中,
    PPO update时直接读取, 避免重复计算
    (SVO推断: GPU前向; K-Level: CEM优化 → 每步重算代价极大).

    内存开销: ~260 floats/step × 2048 steps ≈ 2MB (可忽略)
    """

    def __init__(self, buffer_size: int, state_dim: int, action_dim: int,
                 device: torch.device, config: Config = None,
                 use_svo_game: bool = True):
        self.buffer_size = buffer_size
        self.device = device
        self.ptr = 0
        self.full = False
        self.use_svo = use_svo_game
        # [CVaR] 主开关 (从 config 读, 全程不变)
        self.use_cvar = bool(config is not None and config.ppo.cvar_enabled)

        # 标准PPO数据
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)

        # SVO-Game扩展数据
        if use_svo_game and config is not None:
            N = config.encoder.num_neighbours
            H = config.klevel.prediction_horizon
            prior_mu = config.svo.prior_mu       # [Fix Bug2] 从config取, 不硬编码
            prior_sigma = config.svo.prior_sigma
            self.svo_mu = np.full((buffer_size, N), prior_mu, dtype=np.float32)
            self.svo_sigma = np.full((buffer_size, N), prior_sigma, dtype=np.float32)
            self.pred_trajs = np.zeros((buffer_size, N, H, 5), dtype=np.float32)
            self.interact_mask = np.zeros((buffer_size, N), dtype=bool)
            # [SPO/CVaR] step-wise SVO 紧迫度 u_t
            #   u_t = mean over active i of (w_mu*(1-mu_i/90) + w_sigma*sigma_i/sigma_prior)
            #   双重用途:
            #     - CVaR: episode-level budget d_e = d0 * exp(-beta * mean(u_t))
            #     - SPO:  per-sample 自适应 ε_t = ε_base * exp(-alpha * u_t)
            self.svo_budget_terms = np.zeros(buffer_size, dtype=np.float32)

        # [CVaR] 扩展数据
        if self.use_cvar:
            self.costs = np.zeros(buffer_size, dtype=np.float32)
            self.cost_values = np.zeros(buffer_size, dtype=np.float32)
            self.episode_ids = np.zeros(buffer_size, dtype=np.int64)
            # [SPO 改造] svo_budget_terms 已挪到 SVO 块; 但若 SVO 关闭 + CVaR 开启,
            # _cvar_dual_update 仍要访问该数组 → 兜底分配为全 0 (等价于 d_e = d0)
            if not (use_svo_game and config is not None):
                self.svo_budget_terms = np.zeros(buffer_size, dtype=np.float32)

    def add(self, state, action, reward, value, log_prob, done,
            svo_mu=None, svo_sigma=None, pred_trajs=None, interact_mask=None,
            cost=0.0, cost_value=0.0, episode_id=0, svo_budget_term=0.0):
        """添加一步经验."""
        i = self.ptr
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.values[i] = value
        self.log_probs[i] = log_prob
        self.dones[i] = done

        if self.use_svo and svo_mu is not None:
            self.svo_mu[i] = svo_mu
            self.svo_sigma[i] = svo_sigma
            self.pred_trajs[i] = pred_trajs
            self.interact_mask[i] = interact_mask
            # [SPO 改造] svo_budget_term 跟随 SVO 而不是 CVaR
            self.svo_budget_terms[i] = svo_budget_term

        if self.use_cvar:
            self.costs[i] = cost
            self.cost_values[i] = cost_value
            self.episode_ids[i] = episode_id
            # [SPO 改造] svo_budget_terms 已挪到 SVO 块, 这里不再写入

        self.ptr += 1
        if self.ptr >= self.buffer_size:
            self.full = True

    def compute_returns(self, last_value: float, gamma: float,
                        gae_lambda: float) -> Tuple[np.ndarray, np.ndarray]:
        """GAE计算returns和advantages (reward 路径)."""
        size = self.ptr
        advantages = np.zeros(size, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(size)):
            if t == size - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]
            non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + self.values[:size]
        return returns, advantages

    def compute_cost_returns(self, last_cost_value: float, cost_gamma: float,
                             gae_lambda: float) -> Tuple[np.ndarray, np.ndarray]:
        """[CVaR] GAE 计算 cost returns/advantages (cost 路径).

        和 compute_returns 完全同构, 只是用 self.costs/self.cost_values.
        """
        size = self.ptr
        advantages = np.zeros(size, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(size)):
            next_v = last_cost_value if t == size - 1 else self.cost_values[t + 1]
            non_terminal = 1.0 - self.dones[t]
            delta = self.costs[t] + cost_gamma * next_v * non_terminal - self.cost_values[t]
            last_gae = delta + cost_gamma * gae_lambda * non_terminal * last_gae
            advantages[t] = last_gae
        returns = advantages + self.cost_values[:size]
        return returns, advantages

    def get_data(self, returns: np.ndarray,
                 advantages: np.ndarray,
                 cost_returns: Optional[np.ndarray] = None,
                 cost_advantages: Optional[np.ndarray] = None) -> Dict[str, torch.Tensor]:
        """转换为torch tensors."""
        n = self.ptr
        data = {
            'states':     torch.FloatTensor(self.states[:n]).to(self.device),
            'actions':    torch.FloatTensor(self.actions[:n]).to(self.device),
            'log_probs':  torch.FloatTensor(self.log_probs[:n]).to(self.device),
            'returns':    torch.FloatTensor(returns).to(self.device),
            'advantages': torch.FloatTensor(advantages).to(self.device),
            'values':     torch.FloatTensor(self.values[:n]).to(self.device),  # [Fix] value clipping用
        }
        if self.use_svo:
            data['svo_mu']        = torch.FloatTensor(self.svo_mu[:n]).to(self.device)
            data['svo_sigma']     = torch.FloatTensor(self.svo_sigma[:n]).to(self.device)
            data['pred_trajs']    = torch.FloatTensor(self.pred_trajs[:n]).to(self.device)
            data['interact_mask'] = torch.BoolTensor(self.interact_mask[:n]).to(self.device)
            # [SPO] 给 SVO-adaptive ε 用; 标量 u_t per step ∈ [0, ~2]
            data['svo_budget_terms'] = torch.FloatTensor(self.svo_budget_terms[:n]).to(self.device)
        if self.use_cvar and cost_returns is not None:
            data['cost_returns']    = torch.FloatTensor(cost_returns).to(self.device)
            data['cost_advantages'] = torch.FloatTensor(cost_advantages).to(self.device)
            data['cost_values']     = torch.FloatTensor(self.cost_values[:n]).to(self.device)
        return data

    def reset(self):
        self.ptr = 0
        self.full = False


# ======================================================================== #
#  PPOAgent                                                                 #
# ======================================================================== #

class PPOAgent:
    """
    PPO智能体 (SVO-Game集成版)

    架构:
      shared_encoder (HierarchicalTransformerEncoder)
          ↓
      [GameAwareCrossAttention] ← svo_mu, svo_sigma, pred_trajs
          ↓
      actor_head (ActorHead) → action
      critic_head (CriticHead) → value

    优化器:
      ppo_optimizer: encoder + actor_head + critic_head (合并loss, 单次backward)
      svo_optimizer: svo_birl参数 (Stage 2微调, 独立)
    """

    def __init__(self, config: Config):
        self.config = config
        ppo_cfg = config.ppo
        enc_cfg = config.encoder

        # ---- 设备 ----
        if config.train.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.train.device)
        print(f"使用设备: {self.device}")

        # ---- 维度 ----
        state_dim = enc_cfg.total_obs_dim
        action_dim = config.action.action_dim
        aux_dim = enc_cfg.aux_state_dim

        # ---- [Fix Bug1] 使用config.svo.enabled而非hasattr ----
        # hasattr(config, 'svo')永远True (Config.__init__总创建svo),
        # 消融实验时无法关闭. 改用显式enabled字段.
        self.use_svo_game = config.svo.enabled

        # ---- 共享Encoder ----
        self.encoder = HierarchicalTransformerEncoder(config).to(self.device)
        enc_out_dim = self.encoder.output_dim  # 128

        # ---- Actor/Critic Heads (MLP-only, 无内部Encoder) ----
        head_input_dim = enc_out_dim + aux_dim  # 128 + 10 = 138
        self.actor_head = ActorHead(head_input_dim, action_dim).to(self.device)
        self.critic_head = CriticHead(head_input_dim).to(self.device)

        # ---- [CVaR] Cost critic 头 + Lagrangian 乘子 ----
        # 复用 CriticHead 结构 (相同的 input_dim, scalar 输出)
        self.use_cvar = bool(ppo_cfg.cvar_enabled)
        if self.use_cvar:
            self.cost_critic_head = CriticHead(head_input_dim).to(self.device)
            # cvar_lambda: 用 Python float, 不进梯度图; 通过 Lagrangian dual update 维护
            self.cvar_lambda = float(ppo_cfg.cvar_lambda_init)
            print(f"  [CVaR] enabled: alpha={ppo_cfg.cvar_alpha}, "
                  f"d0={ppo_cfg.cvar_budget_base}, "
                  f"lambda_lr={ppo_cfg.cvar_lambda_lr}")
        else:
            self.cost_critic_head = None
            self.cvar_lambda = 0.0

        # ---- SVO-Game 模块 (可选) ----
        if self.use_svo_game:
            self.svo_birl = SVOVariationalBIRL(config).to(self.device)
            # [Level-k v2] KLevelPredictor已移除, BV控制由LevelKController处理
            # SVO推断结果仅用于L2奖励计算, 不再做CEM轨迹预测
            self.num_neighbours = enc_cfg.num_neighbours

            self._svo_prior_mu = config.svo.prior_mu
            self._svo_prior_sigma = config.svo.prior_sigma

            # [RSSM v6] 每辆NPC维护独立的时序状态 (h_t, s_t)
            #   h_t: (1, hidden_dim)  确定性状态 (GRU隐状态)
            #   s_t: (1, 5)           意图概率向量 π (Categorical分布)
            #   推理时只走先验分支, s_t = π_prior传入下步GRU
            self._use_rssm = True   # v6: RSSM始终启用, rssm_enabled flag已废弃
            # dict[npc_idx] → (h_tensor(1,hdim), s_tensor(1,5))
            self._svo_temporal_h = {}
            print(f"  [RSSM v6] Categorical意图RSSM已启用, "
                  f"num_intents=5, anchors=[15,30,45,60,75]°")
        else:
            self._use_rssm = False

        # ---- PPO优化器 (共享Encoder的梯度冲突修复) ----
        # 为何单优化器:
        #   共享Encoder → actor_loss和critic_loss都要反向传播经过同一个encoder
        #   两次分开backward()会导致第一次释放计算图, 第二次崩溃
        #   解决: 合并loss → 单次backward() → 单优化器step()
        _opt_groups = [
            {'params': self.encoder.parameters(),     'lr': ppo_cfg.actor_lr},
            {'params': self.actor_head.parameters(),  'lr': ppo_cfg.actor_lr},
            {'params': self.critic_head.parameters(), 'lr': ppo_cfg.critic_lr},
        ]
        if self.use_cvar:
            _opt_groups.append(
                {'params': self.cost_critic_head.parameters(), 'lr': ppo_cfg.critic_lr}
            )
        self.ppo_optimizer = optim.Adam(_opt_groups)

        # [Fix Bug3] SVO优化器 (Stage 2微调用)
        if self.use_svo_game:
            self.svo_optimizer = optim.Adam(
                self.svo_birl.parameters(), lr=config.svo.finetune_lr
            )
        else:
            self.svo_optimizer = None

        # ---- 学习率调度 ----
        # [Fix] ExponentialLR每次update都step，250k步后LR≈0，策略冻结
        # 改为线性衰减到10%，确保全程有学习能力
        if ppo_cfg.lr_decay:
            total_updates = max(1, int(config.train.max_timesteps / ppo_cfg.rollout_steps))
            self.ppo_scheduler = optim.lr_scheduler.LinearLR(
                self.ppo_optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=total_updates,
            )
        else:
            self.ppo_scheduler = None

        # ---- Buffer ----
        self.buffer = RolloutBuffer(
            ppo_cfg.rollout_steps, state_dim, action_dim, self.device,
            config=config, use_svo_game=self.use_svo_game,
        )
        self.total_timesteps = 0
        self.episode_count = 0

        # ---- [SLT] Sequential Latent Transformer (辅助表示学习) ----
        self.use_slt = getattr(config, 'slt', None) is not None and config.slt.enabled
        if self.use_slt:
            # SLT的latent_dim应该和encoder输出一致
            config.slt.latent_dim = enc_out_dim  # 128
            self.slt = SequentialLatentTransformer(config).to(self.device)

            # SLT独立optimizer (方案C: encoder用更低lr)
            slt_cfg = config.slt
            self.slt_optimizer = optim.Adam([
                {'params': self.encoder.parameters(),
                 'lr': slt_cfg.lr * slt_cfg.encoder_lr_scale},
                {'params': self.slt.parameters(),
                 'lr': slt_cfg.lr},
            ])

            # 连续序列队列
            self.seq_queue = SequentialQueue(
                future_horizon=slt_cfg.future_horizon,
                obs_dim=state_dim,
                action_dim=action_dim,
                max_size=slt_cfg.queue_max_size,
            )
            print(f"  [SLT] 已启用, T_f={slt_cfg.future_horizon}, "
                  f"lr={slt_cfg.lr}, encoder_lr_scale={slt_cfg.encoder_lr_scale}")

    # ================================================================== #
    #  内部方法                                                            #
    # ================================================================== #

    def _compute_svo_klevel(self, trajs_tensor: torch.Tensor):
        """
        SVO推断 + L5特征构建 (RSSM v6: Categorical意图分布).

        推断流程:
          1. 对每辆有效NPC调用 svo_birl.infer_step_prior()
             → 只走先验分支, 后验网络推理时不参与
          2. 时序状态 _svo_temporal_h[i] = (h_t, s_t)
             h_t: (1, hidden_dim)  确定性GRU状态
             s_t: (1, 5)           意图概率向量 π (=π_prior, 传入下步GRU)
          3. θ = π @ μ_anchors  连续SVO角度
          4. σ = 意图分布的加权标准差 (表征不确定性)

        推断用于:
          1. L2奖励: cos(θ)·R_ego + sin(θ)·R_others
          2. L5 GameAwareCrossAttention: svo_mu, svo_sigma输入

        trajs_tensor: (1, 1+N, T, 5) on device
        Returns: svo_mu (N,), svo_sigma (N,), pred_trajs (N, H, 5), mask (N,)
        """
        from src.models.svo_model import uniform_intent

        N        = self.num_neighbours
        H        = self.config.svo.prediction_horizon
        T        = trajs_tensor.shape[2]
        ego_traj = trajs_tensor[:, 0]   # (1, T, 5)

        # 默认值: 均匀意图 → θ=45°(中性), σ=最大不确定
        svo_mu        = np.full(N, 45.0, dtype=np.float32)
        svo_sigma     = np.full(N, 21.21, dtype=np.float32)  # 均匀分布标准差
        pred_trajs    = np.zeros((N, H, 5), dtype=np.float32)
        interact_mask = np.zeros(N, dtype=bool)

        for i in range(N):
            npc_traj_i = trajs_tensor[:, 1 + i]   # (1, T, 5)

            # 空槽位: NPC不存在 → 清理状态, 跳过
            if npc_traj_i.abs().max().item() < 1e-6:
                if i in self._svo_temporal_h:
                    del self._svo_temporal_h[i]
                continue

            # ---- 取或初始化 RSSM 状态 ----
            if i in self._svo_temporal_h:
                h_prev, s_prev = self._svo_temporal_h[i]
            else:
                # 新出现的NPC: h_0=0, s_0=均匀意图
                h_prev = self.svo_birl.init_temporal_state(1, self.device)
                s_prev = uniform_intent(1, self.device)

            # ---- 推理: 只走先验分支 ----
            # infer_step_prior 返回:
            #   theta_val : float      θ = π_prior @ μ_anchors
            #   pi_prior  : (1, 5)    先验意图概率向量
            #   h_new     : (1, H)    新确定性状态
            #   s_new     : (1, 5)    = pi_prior, 传入下步GRU
            theta_val, pi_prior, h_new, s_new = self.svo_birl.infer_step_prior(
                npc_traj_i, ego_traj, h_prev, s_prev
            )

            # ---- 更新时序状态: 存 (h_t, π_t) ----
            # 关键变化: 旧版存(h, mu标量), 新版存(h, π向量)
            # π向量作为s_{t}传入下步GRU, 携带完整意图不确定性
            self._svo_temporal_h[i] = (h_new, s_new)

            # ---- 计算σ: 意图分布的加权标准差 ----
            mu_anchors = self.svo_birl.mu_anchors          # (5,) on device
            var_i = (pi_prior * (
                mu_anchors.unsqueeze(0) - theta_val
            ).pow(2)).sum(dim=-1)
            sigma_val = var_i.sqrt().item()

            svo_mu[i]    = theta_val
            svo_sigma[i] = sigma_val

            # ---- 标记交互NPC + 填充历史轨迹 (供L5 GRU编码) ----
            interact_mask[i] = True
            npc_np = npc_traj_i[0].cpu().numpy()   # (T, 5)
            if T >= H:
                pred_trajs[i] = npc_np[-H:]
            else:
                pred_trajs[i, H - T:] = npc_np

        return svo_mu, svo_sigma, pred_trajs, interact_mask

    def reset_svo_state(self):
        """
        重置RSSM时序状态 (episode边界调用).

        在 train.py / test.py 中每次 env.reset() 后调用.
        清空所有NPC的 (h_t, s_t=π_t), 下一步自动从 h_0=0, s_0=均匀分布 重新开始.
        """
        if self.use_svo_game:
            self._svo_temporal_h.clear()

    def _encode(self, states: torch.Tensor,
                svo_mu: torch.Tensor = None, svo_sigma: torch.Tensor = None,
                pred_trajs: torch.Tensor = None,
                interact_mask: torch.Tensor = None) -> torch.Tensor:
        """
        解析观测 → Encoder前向 → 拼接aux.

        states: (B, total_obs_dim)
        SVO参数: 全部(B, ...)的tensor, 或None

        Returns: (B, head_input_dim)  即 (B, 138)
        """
        # [Level-k v9] parse_observation 现在返回 4 个值 (新增 bv_actions)
        trajs, map_wps, aux, bv_actions = parse_observation(states, self.config)

        if self.use_svo_game and svo_mu is not None:
            enc_feat = self.encoder(trajs, map_wps, svo_mu, svo_sigma,
                                    pred_trajs, interact_mask,
                                    bv_actions=bv_actions)
        else:
            enc_feat = self.encoder(trajs, map_wps, bv_actions=bv_actions)

        return torch.cat([enc_feat, aux], dim=-1)

    def _encode_raw(self, states: torch.Tensor) -> torch.Tensor:
        """
        Encoder前向, 只返回encoder的raw output (不拼aux).

        用于SLT: SLT只需要学习encoder输出的表示,
        aux是手工特征(速度/偏移等), 不需要SLT来学预测.

        Args:
            states: (B, total_obs_dim)
        Returns: (B, enc_out_dim)  即 (B, 128)
        """
        # [Level-k v9] parse_observation 4 返回值
        trajs, map_wps, _, bv_actions = parse_observation(states, self.config)
        enc_feat = self.encoder(trajs, map_wps, bv_actions=bv_actions)
        return enc_feat  # (B, 128), 不拼aux

    def _slt_update(self) -> Dict[str, float]:
        """
        SLT辅助表示学习更新 (方案C: PPO update之后调用).

        流程:
          1. 从seq_queue采样batch的连续(state, action)序列
          2. 用encoder编码所有T_f步state → latent序列 (用_encode_raw, 不含aux)
          3. SLT计算预测loss (cosine similarity)
          4. 反传更新 encoder + SLT (独立optimizer)

        Returns:
            metrics: dict with slt_loss, slt_cosine_sim
        """
        slt_cfg = self.config.slt

        # 1. 采样连续序列
        s_batch, a_batch = self.seq_queue.sample_batch(slt_cfg.batch_size)
        # s_batch: (B, T_f, obs_dim),  a_batch: (B, T_f, action_dim)

        states = torch.FloatTensor(s_batch).to(self.device)
        actions = torch.FloatTensor(a_batch).to(self.device)

        B, Tf, obs_dim = states.shape

        # 2. 编码所有T_f步: 展平 → encoder → 恢复shape
        states_flat = states.reshape(B * Tf, obs_dim)          # (B*Tf, obs_dim)
        latent_flat = self._encode_raw(states_flat)             # (B*Tf, latent_dim)
        latent_seq = latent_flat.reshape(B, Tf, -1)             # (B, Tf, latent_dim)

        # 3. SLT loss
        loss, metrics = self.slt.compute_loss(latent_seq, actions)

        # 4. 反传 (SLT独立optimizer)
        self.slt_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.slt.parameters()),
            slt_cfg.max_grad_norm,
        )
        self.slt_optimizer.step()

        return metrics

    # ================================================================== #
    #  select_action (rollout阶段)                                        #
    # ================================================================== #

    def select_action(self, state: np.ndarray,
                      deterministic: bool = False) -> tuple:
        """
        选择动作.

        Returns:
            action     : np.ndarray (action_dim,)
            log_prob   : float
            value      : float
            svo_mu     : np.ndarray (N,) or None
            svo_sigma  : np.ndarray (N,) or None
            pred_trajs : np.ndarray (N, H, 5) or None
            interact_mask : np.ndarray (N,) or None
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        svo_mu = svo_sigma = pred_trajs = interact_mask = None

        with torch.no_grad():
            if self.use_svo_game:
                # 1. 解析观测提取轨迹
                # [Level-k v9] parse_observation 4 返回值, 这里只用 trajs
                trajs, _, _, _ = parse_observation(state_tensor, self.config)

                # 2. SVO推断 + K-Level预测
                svo_mu, svo_sigma, pred_trajs, interact_mask = \
                    self._compute_svo_klevel(trajs)

                # 3. Encode with game-aware info
                features = self._encode(
                    state_tensor,
                    torch.FloatTensor(svo_mu).unsqueeze(0).to(self.device),
                    torch.FloatTensor(svo_sigma).unsqueeze(0).to(self.device),
                    torch.FloatTensor(pred_trajs).unsqueeze(0).to(self.device),
                    torch.BoolTensor(interact_mask).unsqueeze(0).to(self.device),
                )
            else:
                features = self._encode(state_tensor)

            # 4. Actor/Critic前向
            action, log_prob = self.actor_head.get_action(features, deterministic)
            value = self.critic_head(features)

        return (
            action.cpu().numpy().squeeze(),
            log_prob.cpu().item(),
            value.cpu().item(),
            svo_mu, svo_sigma, pred_trajs, interact_mask,
        )

    # ================================================================== #
    #  [CVaR] cost value 单独查询 (rollout 时存 buffer 用)                   #
    # ================================================================== #

    def get_cost_value(self, state: np.ndarray,
                       svo_mu=None, svo_sigma=None,
                       pred_trajs=None, interact_mask=None) -> float:
        """[CVaR] 给定 state 返回 cost critic 的 value 估计.

        Args 接收和 select_action 一致的 SVO 字段 (复用其推断结果, 避免重复 SVO 前向).
        若 cvar 未启用, 直接返回 0.0.
        """
        if not self.use_cvar:
            return 0.0
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            if self.use_svo_game and svo_mu is not None:
                features = self._encode(
                    state_tensor,
                    torch.FloatTensor(svo_mu).unsqueeze(0).to(self.device),
                    torch.FloatTensor(svo_sigma).unsqueeze(0).to(self.device),
                    torch.FloatTensor(pred_trajs).unsqueeze(0).to(self.device),
                    torch.BoolTensor(interact_mask).unsqueeze(0).to(self.device),
                )
            else:
                features = self._encode(state_tensor)
            return float(self.cost_critic_head(features).cpu().item())

    # ================================================================== #
    #  [Level-k] 批量推理 (BV策略控制用)                                   #
    # ================================================================== #

    def batch_select_action(self, obs_batch: np.ndarray,
                            deterministic: bool = True) -> np.ndarray:
        """
        批量推理: 一次前向处理N辆BV的观测.

        用于LevelKController: 多辆BV的obs堆成batch, 一次GPU前向.
        L1策略无SVO, 直接: obs → encoder → actor → action.

        Args:
            obs_batch: (N, obs_dim) np.ndarray
            deterministic: bool, True=确定性策略 (BV控制推荐)

        Returns:
            actions: (N, action_dim) np.ndarray
        """
        import torch
        with torch.no_grad():
            states = torch.FloatTensor(obs_batch).to(self.device)
            # L1策略不用SVO, 直接encode
            features = self._encode(states)
            actions, _ = self.actor_head.get_action(features, deterministic)
            return actions.cpu().numpy()

    # ================================================================== #
    #  [CVaR] 辅助函数: episode-level CVaR + dual update                    #
    # ================================================================== #

    def _cvar_dual_update(self) -> Dict[str, float]:
        """[CVaR] 在 buffer 内做 episode 切分 → 计算 C_e / U_e / d_e (per-episode normalized)
        → empirical CVaR_hat_norm → Lagrangian dual update.

        约束写法 (per-episode normalized):
            CVaR_alpha( {C_e / d_e} ) <= 1
            lambda <- max(0, lambda + lr * (CVaR_hat_norm - 1.0))
        """
        ppo_cfg = self.config.ppo
        n = self.buffer.ptr

        # 用 cost_gamma 折扣的 trajectory cost: C_e = sum_t gamma^t * c_t (在 episode 内)
        episode_ids = self.buffer.episode_ids[:n]
        costs = self.buffer.costs[:n]
        u_steps = self.buffer.svo_budget_terms[:n]
        cost_gamma = float(ppo_cfg.cost_gamma)

        ep_C = {}     # ep_id -> trajectory cost
        ep_U_sum = {} # ep_id -> sum of u_t
        ep_U_cnt = {} # ep_id -> count
        ep_t = {}     # ep_id -> current step index inside episode

        for i in range(n):
            eid = int(episode_ids[i])
            t = ep_t.get(eid, 0)
            ep_C[eid] = ep_C.get(eid, 0.0) + (cost_gamma ** t) * float(costs[i])
            ep_U_sum[eid] = ep_U_sum.get(eid, 0.0) + float(u_steps[i])
            ep_U_cnt[eid] = ep_U_cnt.get(eid, 0) + 1
            ep_t[eid] = t + 1

        ep_keys = list(ep_C.keys())
        if len(ep_keys) == 0:
            return {'cvar_hat_norm': 0.0, 'cvar_lambda': self.cvar_lambda,
                    'avg_budget': float(ppo_cfg.cvar_budget_base),
                    'avg_episode_cost': 0.0, 'avg_svo_budget_term': 0.0,
                    'cvar_hat_unnorm': 0.0}

        d0 = float(ppo_cfg.cvar_budget_base)
        beta = float(ppo_cfg.svo_budget_beta)
        d_min = d0 * float(ppo_cfg.cvar_budget_min_ratio)

        C_arr = np.array([ep_C[k] for k in ep_keys], dtype=np.float64)
        U_arr = np.array([ep_U_sum[k] / max(1, ep_U_cnt[k]) for k in ep_keys],
                         dtype=np.float64)
        d_arr = np.maximum(d0 * np.exp(-beta * U_arr), d_min)
        Cn_arr = C_arr / d_arr  # per-episode normalized cost

        # empirical CVaR (Acerbi 2002): nu = quantile_{1-alpha}(Cn), CVaR = nu + (1/(alpha*M)) sum relu(Cn - nu)
        alpha = float(ppo_cfg.cvar_alpha)
        M = len(Cn_arr)
        if M == 1:
            # 至少 1 条 tail 样本: 直接用 max
            cvar_hat_norm = float(Cn_arr.max())
            cvar_hat_unnorm = float(C_arr.max())
        else:
            nu = float(np.quantile(Cn_arr, 1.0 - alpha))
            tail = np.maximum(Cn_arr - nu, 0.0)
            cvar_hat_norm = nu + tail.sum() / max(alpha * M, 1e-8)
            nu_un = float(np.quantile(C_arr, 1.0 - alpha))
            tail_un = np.maximum(C_arr - nu_un, 0.0)
            cvar_hat_unnorm = nu_un + tail_un.sum() / max(alpha * M, 1e-8)

        # Dual update: 约束目标常量 = 1 (per-episode normalized)
        violation = cvar_hat_norm - 1.0
        new_lambda = max(0.0, self.cvar_lambda + float(ppo_cfg.cvar_lambda_lr) * violation)
        _lambda_max = getattr(ppo_cfg, "cvar_lambda_max", None)
        if _lambda_max is not None:
            new_lambda = min(new_lambda, float(_lambda_max))
        self.cvar_lambda = new_lambda

        return {
            'cvar_hat_norm':       float(cvar_hat_norm),
            'cvar_hat_unnorm':     float(cvar_hat_unnorm),
            'cvar_lambda':         float(self.cvar_lambda),
            'avg_budget':          float(d_arr.mean()),
            'avg_episode_cost':    float(C_arr.mean()),
            'avg_svo_budget_term': float(U_arr.mean()),
            'n_episodes_in_rollout': int(M),
        }

    # ================================================================== #
    #  PPO update                                                         #
    # ================================================================== #

    def update(self, last_state: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        PPO策略更新.

        使用合并loss + 单次backward解决共享Encoder梯度冲突.
        SVO/K-Level数据从buffer缓存中读取, 不重复计算.
        """
        ppo_cfg = self.config.ppo

        if self.buffer.ptr <= 0:
            return {
                'actor_loss': 0.0,
                'critic_loss': 0.0,
                'entropy': 0.0,
                'lr': self.ppo_optimizer.param_groups[0]['lr'],
            }

        # ---- 计算最后一步的value (for GAE) ----
        last_idx = self.buffer.ptr - 1
        last_done = bool(self.buffer.dones[last_idx] > 0.5)
        last_cost_value = 0.0   # [CVaR] 默认值, 仅 cvar_enabled 且非 done 时被覆盖
        if last_done:
            last_value = 0.0
        else:
            with torch.no_grad():
                # [Fix] 优先使用rollout尾部next_state做bootstrap，避免用s_t近似s_{t+1}
                if last_state is not None:
                    bootstrap_state = torch.FloatTensor(last_state).unsqueeze(0).to(self.device)
                    # next_state没有同步缓存的SVO特征，bootstrap时回退到基础encode
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
                # [CVaR] cost critic 的 bootstrap value (复用同一份 features)
                if self.use_cvar:
                    last_cost_value = self.cost_critic_head(bootstrap_features).cpu().item()

        # ---- GAE ----
        returns, advantages = self.buffer.compute_returns(
            last_value, ppo_cfg.gamma, ppo_cfg.gae_lambda
        )

        # ---- [CVaR] cost GAE + episode-level CVaR + dual update ----
        cost_returns = cost_advantages = None
        cvar_stats = {}
        if self.use_cvar:
            cost_returns, cost_advantages = self.buffer.compute_cost_returns(
                last_cost_value, ppo_cfg.cost_gamma, ppo_cfg.gae_lambda
            )
            cvar_stats = self._cvar_dual_update()

        data = self.buffer.get_data(returns, advantages, cost_returns, cost_advantages)

        if ppo_cfg.normalize_advantage:
            data['advantages'] = (
                (data['advantages'] - data['advantages'].mean())
                / (data['advantages'].std() + 1e-8)
            )
            # [CVaR] cost advantages 也做归一化, 尺度与 reward advantage 一致
            if self.use_cvar:
                data['cost_advantages'] = (
                    (data['cost_advantages'] - data['cost_advantages'].mean())
                    / (data['cost_advantages'].std() + 1e-8)
                )

        # ---- Mini-batch PPO更新 ----
        n_samples = len(data['states'])
        batch_size = ppo_cfg.mini_batch_size
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_cost_critic_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for epoch in range(ppo_cfg.ppo_epochs):
            indices = np.random.permutation(n_samples)

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_idx = indices[start:end]

                # 取mini-batch数据
                batch_states = data['states'][batch_idx]
                batch_actions = data['actions'][batch_idx]
                batch_old_log_probs = data['log_probs'][batch_idx]
                batch_returns = data['returns'][batch_idx]
                batch_advantages = data['advantages'][batch_idx]

                # 单次Encoder前向 (共享)
                if self.use_svo_game:
                    features = self._encode(
                        batch_states,
                        data['svo_mu'][batch_idx],
                        data['svo_sigma'][batch_idx],
                        data['pred_trajs'][batch_idx],
                        data['interact_mask'][batch_idx],
                    )
                else:
                    features = self._encode(batch_states)

                # Actor loss (PPO clipped objective)
                new_log_probs, entropy = self.actor_head.evaluate(
                    features, batch_actions
                )
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio, 1 - ppo_cfg.clip_epsilon, 1 + ppo_cfg.clip_epsilon
                ) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # [CVaR] cost surrogate (PPO-style clipped, 但加号: 要"压低" cost)
                # actor_total = actor_loss + lambda * cost_coef * cost_actor_loss
                cost_actor_loss = torch.zeros((), device=self.device)
                cost_critic_loss = torch.zeros((), device=self.device)
                if self.use_cvar:
                    batch_cost_adv = data['cost_advantages'][batch_idx]
                    c_surr1 = ratio * batch_cost_adv
                    c_surr2 = torch.clamp(
                        ratio, 1 - ppo_cfg.clip_epsilon, 1 + ppo_cfg.clip_epsilon
                    ) * batch_cost_adv
                    # 注意符号: cost 越小越好 → 最小化 max(surr1, surr2)
                    cost_actor_loss = torch.max(c_surr1, c_surr2).mean()

                    # cost critic loss (clipped, 同 reward critic)
                    new_cost_values = self.cost_critic_head(features)
                    batch_old_cost_values = data['cost_values'][batch_idx]
                    batch_cost_returns = data['cost_returns'][batch_idx]
                    cv_clipped = batch_old_cost_values + torch.clamp(
                        new_cost_values - batch_old_cost_values,
                        -ppo_cfg.clip_epsilon, ppo_cfg.clip_epsilon
                    )
                    cost_critic_loss = torch.max(
                        (new_cost_values - batch_cost_returns) ** 2,
                        (cv_clipped - batch_cost_returns) ** 2,
                    ).mean()

                # Critic loss — [Fix v2] Clipped value loss, 防止value function更新过大
                new_values = self.critic_head(features)
                batch_old_values = data['values'][batch_idx]
                value_clipped = batch_old_values + torch.clamp(
                    new_values - batch_old_values,
                    -ppo_cfg.clip_epsilon, ppo_cfg.clip_epsilon
                )
                critic_loss = torch.max(
                    (new_values - batch_returns) ** 2,
                    (value_clipped - batch_returns) ** 2,
                ).mean()

                # 合并loss → 单次backward (共享Encoder梯度冲突修复)
                loss = (actor_loss
                        + ppo_cfg.value_coef * critic_loss
                        - ppo_cfg.entropy_coef * entropy.mean())
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

            # Early stopping (KL散度约束)
            if ppo_cfg.target_kl is not None:
                with torch.no_grad():
                    if self.use_svo_game:
                        all_features = self._encode(
                            data['states'], data['svo_mu'],
                            data['svo_sigma'], data['pred_trajs'],
                            data['interact_mask'],
                        )
                    else:
                        all_features = self._encode(data['states'])
                    new_lp, _ = self.actor_head.evaluate(
                        all_features, data['actions']
                    )
                    kl = (data['log_probs'] - new_lp).mean().item()
                    if kl > ppo_cfg.target_kl:
                        break

        # ---- LR调度 ----
        if self.ppo_scheduler:
            self.ppo_scheduler.step()

        self.buffer.reset()

        ppo_stats = {
            'actor_loss': total_actor_loss / max(n_updates, 1),
            'critic_loss': total_critic_loss / max(n_updates, 1),
            'entropy': total_entropy / max(n_updates, 1),
            'lr': self.ppo_optimizer.param_groups[0]['lr'],
        }
        if self.use_cvar:
            ppo_stats['cost_critic_loss'] = total_cost_critic_loss / max(n_updates, 1)
            ppo_stats.update(cvar_stats)

        # ---- [SLT] Step 2: 辅助表示学习更新 (方案C: PPO之后) ----
        if self.use_slt and len(self.seq_queue) >= self.config.slt.batch_size:
            slt_stats = self._slt_update()
            ppo_stats.update(slt_stats)

        return ppo_stats

    # ================================================================== #
    #  [SVO-Game] Stage 2 SVO微调                                         #
    # ================================================================== #

    def finetune_svo(self, npc_trajs: torch.Tensor, ego_trajs: torch.Tensor,
                     npc_futures: torch.Tensor) -> Dict[str, float]:
        """
        Stage 2: 用最近rollout数据微调SVO BIRL.

        在PPO update之外单独调用. train.py负责:
          1. rollout阶段收集NPC的past+future轨迹
          2. 调用 agent.finetune_svo(npc_past, ego_past, npc_future)
          3. 调用 agent.update() 做PPO更新

        npc_trajs  : (B, T_past, 5)   NPC历史轨迹
        ego_trajs  : (B, T_past, 5)   Ego历史轨迹
        npc_futures: (B, T_future, 5)  NPC真实未来轨迹

        Returns: metrics dict (elbo, recon_loss, kl_loss, svo_mu_mean, svo_sigma_mean)
        """
        if not self.use_svo_game:
            return {}

        self.svo_birl.train()
        loss, metrics = self.svo_birl.compute_elbo(npc_trajs, ego_trajs, npc_futures)

        self.svo_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.svo_birl.parameters(), self.config.ppo.max_grad_norm
        )
        self.svo_optimizer.step()

        return metrics

    # ================================================================== #
    #  Save / Load                                                        #
    # ================================================================== #

    def save(self, path: str):
        """保存checkpoint (含所有组件状态)."""
        checkpoint = {
            'encoder': self.encoder.state_dict(),
            'actor_head': self.actor_head.state_dict(),
            'critic_head': self.critic_head.state_dict(),
            'ppo_optimizer': self.ppo_optimizer.state_dict(),
            'total_timesteps': self.total_timesteps,
            'episode_count': self.episode_count,
        }

        # [Fix Bug4] 保存scheduler状态
        if self.ppo_scheduler:
            checkpoint['ppo_scheduler'] = self.ppo_scheduler.state_dict()

        # SVO模块
        if self.use_svo_game:
            checkpoint['svo_birl'] = self.svo_birl.state_dict()
            checkpoint['svo_optimizer'] = self.svo_optimizer.state_dict()

        # SLT模块
        if self.use_slt:
            checkpoint['slt'] = self.slt.state_dict()
            checkpoint['slt_optimizer'] = self.slt_optimizer.state_dict()

        # [CVaR] cost critic + lagrangian 乘子
        if self.use_cvar:
            checkpoint['cost_critic_head'] = self.cost_critic_head.state_dict()
            checkpoint['cvar_lambda'] = float(self.cvar_lambda)

        torch.save(checkpoint, path)
        print(f"模型已保存: {path}")

    def load(self, path: str):
        """加载checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        encoder_result = self.encoder.load_state_dict(checkpoint['encoder'], strict=False)
        missing_encoder_keys = list(encoder_result.missing_keys)
        unexpected_encoder_keys = list(encoder_result.unexpected_keys)
        allowed_missing = [
            key for key in missing_encoder_keys
            if key.startswith("bv_action_proj.")
        ]
        disallowed_missing = [
            key for key in missing_encoder_keys
            if key not in allowed_missing
        ]
        if disallowed_missing or unexpected_encoder_keys:
            raise RuntimeError(
                "Encoder checkpoint is incompatible. "
                f"missing={disallowed_missing}, unexpected={unexpected_encoder_keys}"
            )
        if allowed_missing:
            print(
                "[Compat][Info] checkpoint encoder has no bv_action_proj; "
                "keeping the new layer randomly initialized. "
                f"missing_keys={len(allowed_missing)}"
            )
        self.actor_head.load_state_dict(checkpoint['actor_head'])
        self.critic_head.load_state_dict(checkpoint['critic_head'])
        self.total_timesteps = checkpoint.get('total_timesteps', 0)
        self.episode_count = checkpoint.get('episode_count', 0)

        # [CVaR] 兼容旧 checkpoint: 缺 cost_critic_head 时不报错, 用新初始化
        if self.use_cvar:
            if 'cost_critic_head' in checkpoint:
                self.cost_critic_head.load_state_dict(checkpoint['cost_critic_head'])
            else:
                print("[CVaR][Info] 旧 checkpoint 不含 cost_critic_head, 保持新初始化权重")
            self.cvar_lambda = float(checkpoint.get('cvar_lambda', self.cvar_lambda))
            if 'cvar_lambda' not in checkpoint:
                print(f"[CVaR][Info] 旧 checkpoint 不含 cvar_lambda, 保持初始值 {self.cvar_lambda}")

        # ppo_optimizer 加载: 旧 checkpoint 参数组数 < 新 (无 cost_critic), 跳过避免崩溃
        try:
            self.ppo_optimizer.load_state_dict(checkpoint['ppo_optimizer'])
        except (ValueError, KeyError) as e:
            print(f"[CVaR][Info] ppo_optimizer state_dict 不兼容 (可能是无 cost_critic 的旧 ckpt), "
                  f"重新初始化: {type(e).__name__}")

        # [Fix Bug4] 恢复scheduler状态
        if self.ppo_scheduler and 'ppo_scheduler' in checkpoint:
            self.ppo_scheduler.load_state_dict(checkpoint['ppo_scheduler'])

        # SVO模块
        if self.use_svo_game and 'svo_birl' in checkpoint:
            self.svo_birl.load_state_dict(checkpoint['svo_birl'])
        if self.svo_optimizer and 'svo_optimizer' in checkpoint:
            self.svo_optimizer.load_state_dict(checkpoint['svo_optimizer'])

        # SLT模块
        if self.use_slt and 'slt' in checkpoint:
            self.slt.load_state_dict(checkpoint['slt'])
        if self.use_slt and 'slt_optimizer' in checkpoint:
            try:
                self.slt_optimizer.load_state_dict(checkpoint['slt_optimizer'])
            except (ValueError, KeyError) as e:
                print(f"[SLT][Info] slt_optimizer state_dict is incompatible; "
                      f"keeping it reinitialized: {type(e).__name__}")

        print(f"模型已加载: {path}")

    def load_svo_pretrained(self, svo_path: str):
        """
        加载Stage 1预训练的SVO权重.
        在Stage 2开始前调用.
        """
        if not self.use_svo_game:
            print("SVO-Game未启用, 跳过加载")
            return
        ckpt = torch.load(svo_path, map_location=self.device)
        if 'svo_birl' in ckpt:
            sd = ckpt['svo_birl']
        elif 'model_state_dict' in ckpt:
            sd = ckpt['model_state_dict']
        else:
            sd = ckpt

        missing, unexpected = self.svo_birl.load_state_dict(sd, strict=False)
        if missing:
            print(f"  [Info] 缺失keys (新模块, 从头训练): {len(missing)}")
        if unexpected:
            print(f"  [Info] 多余keys (已忽略): {len(unexpected)}")
        print(f"SVO预训练权重已加载: {svo_path}")


# ======================================================================== #
#  测试                                                                     #
# ======================================================================== #

if __name__ == "__main__":
    print("测试PPO模型 (SVO-Game集成版)...")

    config = Config()
    agent = PPOAgent(config)

    state_dim = config.encoder.total_obs_dim
    dummy_state = np.random.randn(state_dim).astype(np.float32)

    action, log_prob, value, sm, ss, pt, mk = agent.select_action(dummy_state)
    print(f"状态维度: {state_dim}")
    print(f"动作: {action}")
    print(f"Log prob: {log_prob:.4f}")
    print(f"Value: {value:.4f}")
    print(f"SVO-Game: {agent.use_svo_game}")

    if sm is not None:
        print(f"SVO mu: {sm[:3]}...")
        print(f"SVO sigma: {ss[:3]}...")
        print(f"Interactive mask: {mk}")

    ep = sum(p.numel() for p in agent.encoder.parameters())
    ap = sum(p.numel() for p in agent.actor_head.parameters())
    cp = sum(p.numel() for p in agent.critic_head.parameters())
    print(f"\nEncoder: {ep:,} | Actor: {ap:,} | Critic: {cp:,} | Total: {ep+ap+cp:,}")
    print(f"(原版双Encoder约: {ep*2+ap+cp:,} — 减少{ep:,}参数)")
    print("\n测试完成!")
