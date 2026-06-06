"""
slt_model.py -- Sequential Latent Transformer (SLT)

========================================================================
在完整训练流程中的位置:
  本文件定义SLT辅助表示学习模块, 被 ppo_model.py 在PPO update后调用.
  训练时自动启用 (config.slt.enabled=True), 推理时零额外开销.
========================================================================

训练时辅助任务: 将未来预测信息蒸馏到当前latent表示中,
减少RL探索空间, 加速收敛.

结构 (参考 Liu et al. 2024, Scene-Rep Transformer, Fig.5-6):
  TransitionModel T: Transformer decoder (自回归预测未来latent)
  Projector Θ: MLP (将latent映射到投影空间)
  Predictor P: MLP (Siamese网络非对称分支, 防表示坍缩)

Loss:
  Lglb = -cosine_similarity(z_target, z_pred).mean()
  z_target = sg(Θ(h_future))    目标 (stop gradient)
  z_pred   = P(Θ(ĥ_future))    预测

推理时完全不使用, 零额外开销.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================== #
#  SequentialLatentTransformer                                              #
# ======================================================================== #

class SequentialLatentTransformer(nn.Module):
    """
    SLT: 用连续T_f步的(latent, action)序列,
    通过Transformer decoder自回归预测未来latent,
    和真实未来latent做cosine similarity loss.
    """

    def __init__(self, config):
        super().__init__()
        slt_cfg = config.slt
        latent_dim = slt_cfg.latent_dim
        action_dim = config.action.action_dim

        self.latent_dim = latent_dim
        self.future_horizon = slt_cfg.future_horizon

        # === Action Embedding ===
        self.action_embed = nn.Sequential(
            nn.Linear(action_dim, slt_cfg.action_embed_dim),
            nn.ReLU(),
        )

        # === Transition Model T (Transformer Decoder) ===
        input_dim = latent_dim + slt_cfg.action_embed_dim

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=input_dim,
            nhead=slt_cfg.num_heads,
            dim_feedforward=input_dim * 2,
            dropout=0.1,
            batch_first=True,
        )
        self.transition_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=slt_cfg.num_layers,
        )

        # 输出投影: decoder输出 → latent_dim
        self.output_proj = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )

        # === Projector Θ (MLP) ===
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, slt_cfg.projector_dim),
            nn.ReLU(),
            nn.Linear(slt_cfg.projector_dim, slt_cfg.projector_dim),
        )

        # === Predictor P (非对称分支, 只在预测侧) ===
        self.predictor = nn.Sequential(
            nn.Linear(slt_cfg.projector_dim, slt_cfg.predictor_dim),
            nn.ReLU(),
            nn.Linear(slt_cfg.predictor_dim, slt_cfg.projector_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, latent_seq, action_seq):
        """
        自回归预测未来latent.

        Args:
            latent_seq: (B, T_f, latent_dim)  连续T_f步的encoder输出
            action_seq: (B, T_f, action_dim)  对应的动作

        Returns:
            predicted_latents: (B, T_f-1, latent_dim)
            预测h[1], h[2], ..., h[T_f-1] (由h[0]~h[T_f-2]和对应action预测)
        """
        B, T, _ = latent_seq.shape

        # 拼接 latent + action_embedding
        action_emb = self.action_embed(action_seq)          # (B, T, action_embed_dim)
        x = torch.cat([latent_seq, action_emb], dim=-1)     # (B, T, input_dim)

        # Causal mask (自回归: 每步只能看到自己和之前的步)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=x.device
        )

        # Transformer decoder (self-attention only, 无cross-attention)
        decoded = self.transition_decoder(
            x, x,
            tgt_mask=causal_mask,
        )  # (B, T, input_dim)

        # 投影到latent空间
        predicted = self.output_proj(decoded)  # (B, T, latent_dim)

        # decoded[t] 预测的是 latent[t+1]
        return predicted[:, :-1, :]  # (B, T-1, latent_dim)

    def compute_loss(self, latent_seq, action_seq):
        """
        计算SLT Siamese cosine similarity loss.

        Args:
            latent_seq: (B, T_f, latent_dim)  真实的连续latent序列
            action_seq: (B, T_f, action_dim)  对应动作序列

        Returns:
            loss: scalar, Lglb
            metrics: dict

        参考论文Fig.5:
          目标: z = sg(Θ(h[1:]))      真实未来latent, stop gradient
          预测: ẑ = P(Θ(ĥ[:-1→]))    Transition Model预测的未来latent
          Lglb = -cosine_sim(z, ẑ).mean()
        """
        # 1. Transition Model预测
        predicted_latents = self.forward(latent_seq, action_seq)
        # (B, T-1, latent_dim)

        # 2. 目标: 真实未来latent (stop gradient!)
        target_latents = latent_seq[:, 1:, :].detach()  # (B, T-1, latent_dim)

        # 3. 投影 + 预测
        # 目标分支: sg(Θ(h'))
        z_target = self.projector(target_latents).detach()  # stop gradient

        # 预测分支: P(Θ(ĥ'))
        z_pred = self.projector(predicted_latents)
        z_pred = self.predictor(z_pred)

        # 4. Cosine similarity loss
        z_target = F.normalize(z_target, dim=-1)
        z_pred = F.normalize(z_pred, dim=-1)

        cosine_sim = (z_target * z_pred).sum(dim=-1)  # (B, T-1)
        loss = -cosine_sim.mean()

        metrics = {
            'slt_loss': loss.item(),
            'slt_cosine_sim': cosine_sim.mean().item(),
        }
        return loss, metrics


# ======================================================================== #
#  SequentialQueue: 收集连续序列                                             #
# ======================================================================== #

class SequentialQueue:
    """
    收集同一episode内连续T_f步的(state, action)序列.

    PPO buffer中的数据在episode边界处断裂,
    不能直接取连续T_f步. 此队列用滑动窗口
    从episode内提取连续序列.

    参考论文 Algorithm 1 第8-11行:
      Store τ in deque Df
      if Tepi > Tf then
          Store in buffer D with (τ, s, a) ~ Df
    """

    def __init__(self, future_horizon, obs_dim, action_dim, max_size=10000):
        self.Tf = future_horizon
        self.max_size = max_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # episode内的临时队列
        self._episode_states = []
        self._episode_actions = []

        # 正式存储: 用预分配numpy数组, 比list更高效
        self.states = np.zeros((max_size, future_horizon, obs_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, future_horizon, action_dim), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add_step(self, state, action):
        """每步rollout时调用."""
        self._episode_states.append(state.copy())
        self._episode_actions.append(action.copy())

        # 每积累够T_f步, 用滑动窗口存入正式buffer
        if len(self._episode_states) >= self.Tf:
            s_seq = np.stack(self._episode_states[-self.Tf:])   # (Tf, obs_dim)
            a_seq = np.stack(self._episode_actions[-self.Tf:])  # (Tf, act_dim)

            self.states[self.ptr] = s_seq
            self.actions[self.ptr] = a_seq
            self.ptr = (self.ptr + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)

    def on_episode_end(self):
        """episode结束时调用, 清空临时队列."""
        self._episode_states.clear()
        self._episode_actions.clear()

    def sample_batch(self, batch_size):
        """
        采样一个batch的连续序列.

        Returns:
            states:  (B, T_f, obs_dim) np.ndarray
            actions: (B, T_f, action_dim) np.ndarray
        """
        actual_size = min(batch_size, self.size)
        indices = np.random.choice(self.size, actual_size, replace=False)
        return self.states[indices], self.actions[indices]

    def __len__(self):
        return self.size