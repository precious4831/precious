"""
svo_model.py -- RSSM Categorical SVO Inference (v6)

设计:
  h_t = GRUCell([e_t, s_{t-1}], h_{t-1})   确定性状态
  p(s_t|h_t)     先验  → 推理时用
  q(s_t|h_t,e_t) 后验  → 训练时用
  s_t = π (5维Categorical意图概率向量)
  θ   = π @ μ_anchors  (连续SVO角度, 可导)
  KL  = Categorical KL(π_post || π_prior)

训练: forward_sequence() — 完整窗口输入TrajEncoder, RSSM单步, Recon+KL+Anchor
推理: infer_step_prior() — 只走先验, 完整窗口输入, 与训练坐标系一致
兼容: compute_elbo() / infer_step() / infer()
"""

import math
import random as pyrandom
import torch
import torch.nn as nn
import torch.nn.functional as F

INTENT_ANCHORS_DEG = [15.0, 30.0, 45.0, 60.0, 75.0]
NUM_INTENTS = 5


def uniform_intent(batch_size, device):
    return torch.full((batch_size, NUM_INTENTS), 1.0 / NUM_INTENTS, device=device)


# ── TrajEncoder ──────────────────────────────────────────────────────────

class TrajEncoder(nn.Module):
    def __init__(self, traj_dim=5, hidden=64):
        super().__init__()
        self.hidden_dim = hidden
        self.npc_gru = nn.GRU(traj_dim, hidden, batch_first=True)
        self.ego_gru = nn.GRU(traj_dim, hidden, batch_first=True)
        self.fusion  = nn.Sequential(
            nn.Linear(hidden * 2 + 4, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _ego_frame(traj, rx, ry, ryaw, cy, sy):
        dx = traj[:, :, 0:1] - rx
        dy = traj[:, :, 1:2] - ry
        return torch.cat([
            dx*cy + dy*sy, -dx*sy + dy*cy,
            traj[:, :, 2:3] - ryaw,
            traj[:, :, 3:5],
        ], dim=-1)

    def forward(self, npc_traj, ego_traj):
        ref  = ego_traj[:, -1]
        rx   = ref[:, 0:1].unsqueeze(1)
        ry   = ref[:, 1:2].unsqueeze(1)
        ryaw = ref[:, 2:3].unsqueeze(1)
        cy, sy = torch.cos(ryaw), torch.sin(ryaw)

        nr = self._ego_frame(npc_traj, rx, ry, ryaw, cy, sy)
        er = self._ego_frame(ego_traj, rx, ry, ryaw, cy, sy)

        _, hn = self.npc_gru(nr); hn = hn.squeeze(0)
        _, he = self.ego_gru(er); he = he.squeeze(0)

        rp = nr[:, -1, :2] - er[:, -1, :2]
        rv = nr[:, -1, 3:5] - er[:, -1, 3:5]
        return self.fusion(torch.cat([hn, he, rp, rv], dim=-1))


# ── SVORSSMCell ───────────────────────────────────────────────────────────

class SVORSSMCell(nn.Module):
    def __init__(self, hidden_dim=64, num_intents=NUM_INTENTS):
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.num_intents = num_intents
        self.det_cell  = nn.GRUCell(hidden_dim + num_intents, hidden_dim)
        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, num_intents),
        )
        self.post_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, num_intents),
        )
        self._init()

    def _init(self):
        for net in [self.prior_net, self.post_net]:
            nn.init.zeros_(net[-1].bias)
            nn.init.xavier_uniform_(net[-1].weight, gain=0.1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.bias is not None and m.bias.abs().sum() == 0 and m.weight.abs().max() > 0.2:
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

    def init_h(self, B, device):
        return torch.zeros(B, self.hidden_dim, device=device)

    def forward(self, e_t, h_prev, s_prev):
        """
        Returns: h_t (B,H), pi_prior (B,5), pi_post (B,5)
        """
        h_t      = self.det_cell(torch.cat([e_t, s_prev], dim=-1), h_prev)
        pi_prior = F.softmax(self.prior_net(h_t), dim=-1)
        pi_post  = F.softmax(self.post_net(torch.cat([h_t, e_t], dim=-1)), dim=-1)
        return h_t, pi_prior, pi_post


# ── TrajectoryDecoder ─────────────────────────────────────────────────────

class TrajectoryDecoder(nn.Module):
    def __init__(self, traj_dim=5, hidden=64, pred_steps=10):
        super().__init__()
        self.pred_steps = pred_steps
        self.svo_dim    = 8
        self.svo_proj   = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, self.svo_dim))
        self.h0_proj    = nn.Linear(hidden, hidden)
        self.gru_cell   = nn.GRUCell(traj_dim + self.svo_dim, hidden)
        self.out_mu     = nn.Linear(hidden, traj_dim)
        self.out_lv     = nn.Linear(hidden, traj_dim)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, theta, history_enc, last_state, future_gt=None, tf_ratio=0.0):
        tr  = theta * (math.pi / 180.0)
        emb = self.svo_proj(torch.stack([torch.cos(tr), torch.sin(tr)], dim=-1))
        h   = self.h0_proj(history_enc)
        inp = last_state
        mus, lvs = [], []
        for t in range(self.pred_steps):
            h   = self.gru_cell(torch.cat([inp, emb], dim=-1), h)
            mu  = self.out_mu(h)
            lv  = self.out_lv(h).clamp(-6., 2.)
            mus.append(mu); lvs.append(lv)
            use_gt = future_gt is not None and self.training and pyrandom.random() < tf_ratio
            inp = future_gt[:, t] if use_gt else mu
        return torch.stack(mus, dim=1), torch.stack(lvs, dim=1)


# ── SVOVariationalBIRL ───────────────────────────────────────────────────

class SVOVariationalBIRL(nn.Module):
    def __init__(self, config):
        super().__init__()
        hdim   = config.svo.hidden_dim
        pred_h = config.svo.prediction_horizon
        self.beta          = config.svo.beta
        self.tf_ratio      = config.svo.teacher_forcing_ratio
        self.anchor_weight = config.svo.anchor_loss_weight

        self.traj_encoder = TrajEncoder(traj_dim=5, hidden=hdim)
        self.rssm_cell    = SVORSSMCell(hidden_dim=hdim, num_intents=NUM_INTENTS)
        self.decoder      = TrajectoryDecoder(traj_dim=5, hidden=hdim, pred_steps=pred_h)

        self.register_buffer(
            'mu_anchors', torch.tensor(INTENT_ANCHORS_DEG, dtype=torch.float32))

    # ── 初始化 ──────────────────────────────────────────────────────────

    def init_states(self, B, device):
        return self.rssm_cell.init_h(B, device), uniform_intent(B, device)

    def init_temporal_state(self, B, device):
        h, _ = self.init_states(B, device)
        return h

    # ── ego-centric工具 ─────────────────────────────────────────────────

    @staticmethod
    def _ego_refs(ego_past):
        r   = ego_past[:, -1]
        rx  = r[:, 0:1].unsqueeze(1)
        ry  = r[:, 1:2].unsqueeze(1)
        ry2 = r[:, 2:3].unsqueeze(1)
        return rx, ry, ry2, torch.cos(ry2), torch.sin(ry2)

    @staticmethod
    def _to_rel(traj, rx, ry, ryaw, cy, sy):
        dx = traj[:, :, 0:1] - rx
        dy = traj[:, :, 1:2] - ry
        return torch.cat([dx*cy+dy*sy, -dx*sy+dy*cy,
                          traj[:, :, 2:3]-ryaw, traj[:, :, 3:5]], dim=-1)

    # ── 训练: 单步（完整窗口输入）─────────────────────────────────────

    def forward_sequence(self, npc_past, ego_past, npc_future,
                         target_mu=None, beta=None):
        """
        训练单步：TrajEncoder 处理完整 T 帧窗口，RSSM 更新一步。

        坐标系统一规则（训练和推理完全一致）：
          参考系 = ego_past[:, -1]（当前窗口ego末帧）
          TrajEncoder 内部用此参考系做 ego-centric 变换
          Decoder 的输入/目标也用同一参考系

        Args:
            npc_past  : (B, T, 5)   NPC历史轨迹窗口
            ego_past  : (B, T, 5)   Ego历史轨迹窗口
            npc_future: (B, H, 5)   NPC未来轨迹（Recon监督信号）
            target_mu : (B,)        弱监督锚点角度（可选）
            beta      : float       KL权重

        Returns:
            loss   : scalar
            metrics: dict
        """
        _beta  = beta if beta is not None else self.beta
        B      = npc_past.shape[0]
        device = npc_past.device

        h, s = self.init_states(B, device)

        # ── 统一参考系：ego末帧 ──────────────────────────────────────
        # TrajEncoder 内部也用 ego_traj[:, -1] 作参考，此处提取出来
        # 供 Decoder 的输入/目标使用，保证三者坐标系完全相同
        rx, ry, ryaw, cy, sy = self._ego_refs(ego_past)

        # Decoder 目标：npc_future 转到 ego-centric（同一参考系）
        npc_future_rel = self._to_rel(npc_future, rx, ry, ryaw, cy, sy)
        # Decoder 起点：npc 末帧转到 ego-centric（同一参考系）
        npc_last_rel   = self._to_rel(npc_past[:, -1:], rx, ry, ryaw, cy, sy)[:, 0]

        # ── TrajEncoder：完整 T 帧窗口 → e ──────────────────────────
        # GRU 顺序处理 T=10 帧，输出富含速度趋势/加速度/交互模式的特征
        # 内部参考系 = ego_past[:, -1]，与上方提取的 rx/ry/ryaw 一致
        e = self.traj_encoder(npc_past, ego_past)   # (B, hidden)

        # ── RSSM：单步更新 ───────────────────────────────────────────
        # h_t = GRUCell([e, s_{t-1}], h_{t-1})
        # 先验 p(s_t|h_t)      → 推理时用
        # 后验 q(s_t|h_t, e)   → 训练时用（能"看到"当前轨迹观测）
        h, pi_prior, pi_post = self.rssm_cell(e, h, s)

        # θ = π_post @ μ_anchors（连续可导）
        theta = (pi_post * self.mu_anchors).sum(dim=-1)   # (B,)

        # ── Categorical KL：KL(π_post || π_prior) ───────────────────
        eps     = 1e-8
        kl_loss = (pi_post * (
            torch.log(pi_post + eps) - torch.log(pi_prior + eps)
        )).sum(dim=-1).mean()

        # ── Recon Loss：Decoder 预测 npc_future ─────────────────────
        # history_enc = e（TrajEncoder输出，编码了npc行为特征）
        # last_state  = npc_last_rel（npc末帧，ego-centric，与future同坐标系）
        pm, plv = self.decoder(theta, e, npc_last_rel,
                               npc_future_rel, self.tf_ratio)
        recon_loss = (0.5 * (plv + (npc_future_rel - pm).pow(2) /
                             (plv.exp() + 1e-8))).mean()

        # ── Anchor Loss：弱监督（θ趋向风格标签角度）────────────────
        anchor_loss = torch.zeros(1, device=device).squeeze()
        if target_mu is not None and self.anchor_weight > 0:
            anchor_loss = F.mse_loss(theta, target_mu)

        loss = recon_loss + _beta * kl_loss + self.anchor_weight * anchor_loss

        metrics = {
            'loss':              loss.item(),
            'recon_loss':        recon_loss.item(),
            'kl_loss':           kl_loss.item(),
            'anchor_loss':       anchor_loss.item(),
            'theta_mean':        theta.mean().item(),
            'pi_max_mean':       pi_post.max(dim=-1)[0].mean().item(),
        }
        with torch.no_grad():
            theta_p = (pi_prior * self.mu_anchors).sum(dim=-1)
            metrics['theta_prior_mean']  = theta_p.mean().item()
            metrics['pi_prior_max_mean'] = pi_prior.max(dim=-1)[0].mean().item()

        return loss, metrics

    # ── 向后兼容 ────────────────────────────────────────────────────────

    def compute_elbo(self, npc_traj, ego_traj, npc_future,
                     target_mu=None, beta=None,
                     h_temporal=None, theta_prev=None):
        return self.forward_sequence(npc_traj, ego_traj, npc_future,
                                     target_mu=target_mu, beta=beta)

    # ── 推理: 只用先验 ─────────────────────────────────────────────────

    @torch.no_grad()
    def infer_step_prior(self, npc_traj, ego_traj, h_prev, s_prev):
        """
        推理单步, 只走先验分支.
        Returns: theta(float), pi_prior(1,5), h_new(1,H), s_new(1,5)
        """
        e_t = self.traj_encoder(npc_traj, ego_traj)
        h_new, pi_prior, _ = self.rssm_cell(e_t, h_prev, s_prev)
        theta = (pi_prior * self.mu_anchors).sum(dim=-1).item()
        return theta, pi_prior, h_new, pi_prior

    @torch.no_grad()
    def infer_step(self, npc_traj, ego_traj, h_prev, theta_prev=None):
        """向后兼容ppo_model.py. Returns: mu(B,), sigma(B,), h_new."""
        B, device = h_prev.shape[0], h_prev.device
        e_t = self.traj_encoder(npc_traj, ego_traj)
        h_new, pi_prior, _ = self.rssm_cell(e_t, h_prev, uniform_intent(B, device))
        theta = (pi_prior * self.mu_anchors).sum(dim=-1)
        var   = (pi_prior * (self.mu_anchors.unsqueeze(0) - theta.unsqueeze(-1)).pow(2)).sum(dim=-1)
        return theta, var.sqrt(), h_new

    @torch.no_grad()
    def infer(self, npc_traj, ego_traj):
        """无状态推断 (h=0, s=均匀). Returns: theta(B,), sigma(B,)."""
        B, device = npc_traj.shape[0], npc_traj.device
        h0, s0 = self.init_states(B, device)
        e_t = self.traj_encoder(npc_traj, ego_traj)
        _, pi_prior, _ = self.rssm_cell(e_t, h0, s0)
        theta = (pi_prior * self.mu_anchors).sum(dim=-1)
        var   = (pi_prior * (self.mu_anchors.unsqueeze(0) - theta.unsqueeze(-1)).pow(2)).sum(dim=-1)
        return theta, var.sqrt()

    @torch.no_grad()
    def sample_theta(self, npc_traj, ego_traj, n_samples=1):
        theta, _ = self.infer(npc_traj, ego_traj)
        return theta.unsqueeze(1).expand(-1, n_samples)