"""
encoder.py — Hierarchical Transformer Encoder (PyTorch)

========================================================================
在完整训练流程中的位置:
  本文件定义HierarchicalTransformerEncoder, 被 ppo_model.py 调用.
  训练: python train.py --stage l1 / svo_only / l2
  验证: python test.py  --l1_model ... --svo_model ... --l2_model ...
========================================================================

L1: 时间步 self-attention     (per agent)
L2: Agent relational attention (ego ← neighbours)
L3: Vehicle-Map cross-attention
L4: Goal attention             (multi-mode)
L5: [SVO-Game] Game-Aware Cross-Attention (optional)

修复记录:
  - 原版: attention全mask时NaN → _safe_attention回退
  - 原版: max-pool全mask → 零向量
  - SVO-Game: GameAwareCrossAttention注入SVO后验+K-Level预测
  - Fix: num_modes>1时加断言保护 (GameAwareCrossAttention要求squeeze后的2D输入)
  - Fix: 消除magic number, prior_sigma从config传入
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================== #
#  工具函数                                                                  #
# ======================================================================== #

class Rotater(nn.Module):
    """坐标旋转（无可学习参数），全局→自车局部坐标"""

    def __init__(self, mode='traj'):
        super().__init__()
        self.mode = mode

    def forward(self, states, mask, curr_frame=None):
        if self.mode == 'traj':
            return self._rotate_traj(states, mask, curr_frame)
        else:
            return self._rotate_map(states, mask, curr_frame)

    def _rotate_traj(self, states, mask, curr_frame=None):
        if curr_frame is None:
            ego_mask = mask[:, 0, :]
            ind = ego_mask.sum(dim=-1).long().clamp(min=1) - 1
            bi = torch.arange(states.shape[0], device=states.device)
            curr_frame = states[bi, 0, ind]

        yaw = curr_frame[:, 2]
        c, s = torch.cos(yaw).view(-1, 1, 1), torch.sin(yaw).view(-1, 1, 1)

        x = states[..., 0] - curr_frame[:, 0].view(-1, 1, 1)
        y = states[..., 1] - curr_frame[:, 1].view(-1, 1, 1)

        angle = states[..., 2] - yaw.view(-1, 1, 1)
        angle = (angle + np.pi) % (2 * np.pi) - np.pi

        vx = states[..., 3] - curr_frame[:, 3].view(-1, 1, 1)
        vy = states[..., 4] - curr_frame[:, 4].view(-1, 1, 1)

        rotated = torch.stack([
            c*x + s*y, -s*x + c*y, angle,
            c*vx + s*vy, -s*vx + c*vy
        ], dim=-1)
        return rotated * mask.unsqueeze(-1).float(), curr_frame

    def _rotate_map(self, states, mask, curr_frame):
        yaw = curr_frame[:, 2]
        c, s = torch.cos(yaw).view(-1, 1, 1), torch.sin(yaw).view(-1, 1, 1)
        x = states[..., 0] - curr_frame[:, 0].view(-1, 1, 1)
        y = states[..., 1] - curr_frame[:, 1].view(-1, 1, 1)
        rotated = torch.stack([c*x + s*y, -s*x + c*y], dim=-1)
        return rotated * mask.unsqueeze(-1).float()


def _safe_attention(attn_module, q, k, v, key_padding_mask=None):
    """
    安全的 attention 调用:
    如果某个 batch 的 key 全被 mask，则取消该 batch 的 mask 防止 NaN
    """
    if key_padding_mask is not None:
        all_masked = key_padding_mask.all(dim=-1)           # (B,)
        if all_masked.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_masked] = False
    out, w = attn_module(q, k, v, key_padding_mask=key_padding_mask)
    return out, w


def _masked_max_pool(x, mask):
    """
    带 mask 的 max-pool: (B, T, D) + (B, T) → (B, D)
    全 mask 时返回零向量而非 -1e9
    """
    if mask is not None:
        x_masked = x.masked_fill(~mask.unsqueeze(-1), -1e9)
        pooled = x_masked.max(dim=1)[0]
        all_invalid = ~mask.any(dim=-1)
        if all_invalid.any():
            pooled[all_invalid] = 0.0
        return pooled
    else:
        return x.max(dim=1)[0]


# ======================================================================== #
#  MapEncoder                                                               #
# ======================================================================== #

class MapEncoder(nn.Module):
    """单条 polyline → self-attention → pool → embedding"""

    def __init__(self, input_dim=2, units=128, num_heads=2):
        super().__init__()
        self.proj = nn.Linear(input_dim, units)
        self.attn = nn.MultiheadAttention(units, num_heads, dropout=0, batch_first=True)
        self.norm = nn.LayerNorm(units)
        self.vec_proj = nn.Linear(input_dim, 64)
        self.out_proj = nn.Linear(units + 64, units)

    def forward(self, inputs, mask):
        nodes = self.proj(inputs)
        kpm = ~mask if mask is not None else None
        nodes, _ = _safe_attention(self.attn, nodes, nodes, nodes, key_padding_mask=kpm)
        nodes = self.norm(F.relu(nodes))
        pooled = _masked_max_pool(nodes, mask)
        vec = F.relu(self.vec_proj(inputs[:, 0, :]))
        return F.relu(self.out_proj(torch.cat([pooled, vec], dim=-1)))


# ======================================================================== #
#  [SVO-Game] L5: Game-Aware Cross-Attention                                #
# ======================================================================== #

class GameAwareCrossAttention(nn.Module):
    """
    将SVO后验 + K-Level预测轨迹注入ego编码.

    输入:
        ego_enc      : (B, units)  L4输出的ego特征 (要求2D, 即num_modes=1)
        svo_mu       : (B, N)      SVO后验均值(度)
        svo_sigma    : (B, N)      SVO后验标准差(度)
        pred_trajs   : (B, N, H, 5) K-Level预测轨迹
        interact_mask: (B, N)      交互mask, True=交互车辆

    输出:
        (B, units) 增强后的ego编码

    流程:
        1. SVO编码: [cos(μ), sin(μ), σ/σ_prior, log(σ)] → MLP → (B, N, units)
        2. 轨迹编码: GRU(pred_trajs) → (B, N, units)
        3. 融合: cat([svo_emb, traj_emb]) → Linear → tokens (B, N, units)
        4. 置信度门控: tokens *= sigmoid(k - σ/τ)  低确信时衰减
        5. Cross-attention: ego_enc attend tokens (仅交互车辆)
        6. 残差连接: output = ego_enc + LayerNorm(attn_out)
    """

    # 门控超参 (控制置信度→门控值的映射)
    GATE_BIAS = 2.0     # 基础偏置, 高确信时gate≈sigmoid(2)=0.88
    GATE_TEMP = 10.0    # 温度, sigma/temp控制衰减速率

    def __init__(self, units=128, num_heads=2, prior_sigma=25.0):
        super().__init__()
        self.prior_sigma = prior_sigma  # 用于归一化sigma输入

        # SVO角度 → embedding
        # 输入4维: [cos(mu_rad), sin(mu_rad), sigma/prior_sigma, log(sigma)]
        self.svo_proj = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(),
            nn.Linear(64, units),
        )

        # K-Level预测轨迹 → embedding
        self.traj_gru = nn.GRU(5, units, batch_first=True)

        # 融合 SVO + 轨迹
        self.fusion = nn.Linear(units * 2, units)

        # Cross-attention: ego attend NPC tokens
        self.cross_attn = nn.MultiheadAttention(
            units, num_heads, dropout=0, batch_first=True
        )
        self.norm = nn.LayerNorm(units)

    def forward(self, ego_enc, svo_mu, svo_sigma, pred_trajs, interact_mask):
        """
        ego_enc      : (B, units)
        svo_mu       : (B, N) degrees
        svo_sigma    : (B, N) degrees
        pred_trajs   : (B, N, H, 5)
        interact_mask: (B, N) bool

        Returns: (B, units)
        """
        B, N = svo_mu.shape

        # --- 1. SVO角度编码 ---
        mu_rad = svo_mu * (np.pi / 180.0)
        svo_input = torch.stack([
            torch.cos(mu_rad),
            torch.sin(mu_rad),
            svo_sigma / self.prior_sigma,     # 归一化 (不再硬编码25.0)
            torch.log(svo_sigma + 1e-3),      # log尺度特征
        ], dim=-1)                             # (B, N, 4)
        svo_emb = self.svo_proj(svo_input)    # (B, N, units)

        # --- 2. K-Level轨迹编码 ---
        # reshape (B, N, H, 5) → (B*N, H, 5) 过GRU
        H = pred_trajs.shape[2]
        flat_trajs = pred_trajs.reshape(B * N, H, 5)
        _, h_traj = self.traj_gru(flat_trajs)              # (1, B*N, units)
        traj_emb = h_traj.squeeze(0).reshape(B, N, -1)     # (B, N, units)

        # --- 3. 融合 ---
        tokens = self.fusion(torch.cat([svo_emb, traj_emb], dim=-1))  # (B, N, units)

        # --- 4. 置信度门控 ---
        # 高sigma(不确信) → gate小 → 衰减token影响力
        # sigma=0° → gate≈0.88, sigma=25° → gate≈0.27
        gate = torch.sigmoid(
            self.GATE_BIAS - svo_sigma / self.GATE_TEMP
        ).unsqueeze(-1)                                    # (B, N, 1)
        tokens = tokens * gate

        # --- 5. Cross-attention ---
        q = ego_enc.unsqueeze(1)                           # (B, 1, units)

        # key_padding_mask: True = 忽略 → 非交互车辆忽略
        kpm = ~interact_mask if interact_mask is not None else None
        attn_out, _ = _safe_attention(
            self.cross_attn, q, tokens, tokens, key_padding_mask=kpm
        )                                                  # (B, 1, units)

        # --- 6. 残差连接 ---
        return ego_enc + self.norm(F.relu(attn_out.squeeze(1)))


# ======================================================================== #
#  HierarchicalTransformerEncoder                                           #
# ======================================================================== #

class HierarchicalTransformerEncoder(nn.Module):
    """
    层次化 Transformer 编码器

    L1: 时间步 self-attn → per-agent temporal embedding
    L2: Agent relational cross-attn → ego关系编码
    L3: Vehicle-Map cross-attn → 空间上下文
    L4: Goal attention → 目标意图
    L5: [SVO-Game] Game-Aware Cross-Attention (可选)
    """
#L1输入 trajs: (B, 1+N, T, 5)  levels: (B, 1+N) int 0或1 interactive_mask: (B, N) bool
#L2输入 ego_e: (B, units)  nbr_es: list of (B, units) 关系编码
#L3输入 map_wps: (B, M, L, 2)  map_embs: (B, M, units)
#L4输入 rel: (B, units)  ego_map: (B, ep, units)  ep=ego_paths
#L5输入 feat: (B, units)  svo_mu: (B, N)  svo_sigma: (B, N)  pred_trajs: (B, N, H, 5)  interact_mask: (B, N)
    def __init__(self, config):
        super().__init__()
        enc   = config.encoder
        units = enc.hidden_units
        heads = enc.num_heads

        self.neighbours    = enc.num_neighbours
        self.num_modes     = enc.num_modes
        self.make_rotation = enc.make_rotation
        self.ego_paths     = enc.ego_paths

        if self.make_rotation:
            self.traj_rot = Rotater('traj')
            self.map_rot  = Rotater('map')

        # Level-1: 时间步编码
        self.time_proj = nn.Linear(5, units)
        self.time_attn = nn.MultiheadAttention(units, heads, dropout=0, batch_first=True)
        self.time_norm = nn.LayerNorm(units)

        # Level-2: Agent 关系
        self.rel_attn = nn.MultiheadAttention(units, heads, dropout=0, batch_first=True)
        self.rel_norm = nn.LayerNorm(units)

        # Level-3: 地图编码
        self.map_enc = MapEncoder(2, units, heads)
        self.mv_attn = nn.MultiheadAttention(units, heads, dropout=0, batch_first=True)
        self.mv_norm = nn.LayerNorm(units)

        # Level-4: Goal
        self.goal_attns = nn.ModuleList([
            nn.MultiheadAttention(units, enc.goal_head_num, dropout=0, batch_first=True)
            for _ in range(self.num_modes)
        ])
        self.goal_norm = nn.LayerNorm(units)

        # Level-5: [SVO-Game] Game-Aware Cross-Attention
        prior_sigma = config.svo.prior_sigma if hasattr(config, 'svo') else 25.0
        self.game_cross_attn = GameAwareCrossAttention(units, heads, prior_sigma)

        # ============================================================== #
        # [Level-k v9] BV 动作编码分支                                     #
        # ============================================================== #
        # 输入: (B, N, 2) — 每个邻居的 [steer, throttle_brake]
        # 输出: (B, N, units) — embedding 后融合到 neighbor 特征
        # 用一个小 MLP, 不加复杂结构 (a_BV 只是 2 维数据, 简单投影即可)
        self.use_bv_actions = enc.use_bv_actions_in_obs
        if self.use_bv_actions:
            bv_act_dim = enc.bv_action_feat_dim - 1   # 去掉 valid_mask 那一维, 只投影 [steer, tb]
            self.bv_action_proj = nn.Sequential(
                nn.Linear(bv_act_dim, units // 2),
                nn.ReLU(),
                nn.Linear(units // 2, units),
            )
            # 融合方式: 加到 nbr 时序编码的输出上 (按 valid_mask 加权)
            # 这样 valid=0 时 BV 特征不参与, 等价于不拼接

        self.output_dim = units
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _time_encode(self, states, mask):
        """(B, T, 5) → self-attention → max-pool → (B, units)"""
        x = self.time_proj(states)
        kpm = ~mask if mask is not None else None
        x, _ = _safe_attention(self.time_attn, x, x, x, key_padding_mask=kpm)
        x = self.time_norm(F.relu(x))
        return _masked_max_pool(x, mask)

    def _mv_rel(self, v_emb, nbr_map, nbr_mm, i):
        """第i个邻车 attend 其对应2条路径"""
        idx = i * 2
        use_map = nbr_map[:, idx:idx+2, :]
        q = v_emb.unsqueeze(1)
        kv = torch.cat([q, use_map], dim=1)                # (B, 3, units)
        ones = torch.ones(v_emb.shape[0], 1, device=v_emb.device, dtype=torch.bool)
        m = nbr_mm[:, idx:idx+2] if nbr_mm is not None \
            else torch.ones(v_emb.shape[0], 2, device=v_emb.device, dtype=torch.bool)
        kpm = ~torch.cat([ones, m], dim=1)                  # (B, 3)
        out, _ = _safe_attention(self.mv_attn, q, kv, kv, key_padding_mask=kpm)
        return self.mv_norm(F.relu(out.squeeze(1)))

    def _build_traj_mask(self, trajs):
        """
        构建轨迹 mask，确保 ego 当前帧始终有效
        trajs: (B, 1+N, T, 5)
        Returns: (B, 1+N, T) bool
        """
        mask = (trajs.abs() > 1e-6).any(dim=-1)
        mask[:, 0, -1] = True
        return mask

    def forward(self, trajs, map_wps,
                svo_mu=None, svo_sigma=None,
                pred_trajs=None, interact_mask=None,
                bv_actions=None):
        """
        trajs   : (B, 1+N, T, 5)   全局[x, y, yaw, vx, vy]
        map_wps : (B, M, L, 2)     全局[x, y]

        [SVO-Game 可选参数] — 全部提供才启用L5:
        svo_mu       : (B, N)       SVO后验均值(度)
        svo_sigma    : (B, N)       SVO后验标准差(度)
        pred_trajs   : (B, N, H, 5) K-Level预测轨迹
        interact_mask: (B, N)       交互mask

        [Level-k v9 可选参数]:
        bv_actions   : (B, N, 3)    每个邻居 [steer, throttle_brake, valid_mask]
                                    L1 BV 训练时全 0; L2 时只有被 levelk 接管的邻居非 0

        Returns: (B, units)
        """
        t_mask = self._build_traj_mask(trajs)
        m_mask = (map_wps.abs() > 1e-6).any(dim=-1)        # (B, M, L)
        m_exist = m_mask.any(dim=-1)                         # (B, M)

        # ---- 1. 坐标旋转 ----
        if self.make_rotation:
            trajs, curr = self.traj_rot(trajs, t_mask)
            map_wps = self.map_rot(map_wps, m_mask, curr)

        # ---- 2. 分离 ego / neighbours ----
        ego_s, nbr_s = trajs[:, 0], trajs[:, 1:]
        ego_m, nbr_m = t_mask[:, 0], t_mask[:, 1:]

        # ---- 3. Level-1: 时间步编码 ----
        ego_e = self._time_encode(ego_s, ego_m)
        nbr_es = [self._time_encode(nbr_s[:, i], nbr_m[:, i])
                  for i in range(self.neighbours)]

        # ---- 3b. [Level-k v9] 融合 a_BV 到邻居特征 ----
        # 对每个邻居 i: nbr_es[i] += bv_action_proj(a_BV_i) * valid_mask_i
        # 这样 valid=0 (TM 控制 / 不存在 / L1 训练) 时, BV 特征不参与
        if self.use_bv_actions and bv_actions is not None:
            # bv_actions: (B, N, 3)
            bv_act_xy = bv_actions[..., :2]          # (B, N, 2) - [steer, tb]
            bv_valid  = bv_actions[..., 2:3]         # (B, N, 1) - valid_mask
            bv_emb_all = self.bv_action_proj(bv_act_xy)  # (B, N, units)
            bv_emb_all = bv_emb_all * bv_valid       # 无效位置归零
            # 加到对应邻居的时序特征上
            for i in range(self.neighbours):
                nbr_es[i] = nbr_es[i] + bv_emb_all[:, i, :]

        # ---- 4. Level-3: Map 编码 ----
        map_embs = torch.stack([
            self.map_enc(map_wps[:, i], m_mask[:, i])
            for i in range(map_wps.shape[1])
        ], dim=1)                                           # (B, M, units)

        ep = self.ego_paths
        ego_map, nbr_map = map_embs[:, :ep], map_embs[:, ep:]
        ego_mm, nbr_mm   = m_exist[:, :ep],  m_exist[:, ep:]

        # ---- 5. Vehicle-Map attention ----
        nbr_rel = [self._mv_rel(nbr_es[i], nbr_map, nbr_mm, i)
                   for i in range(self.neighbours)]

        # ---- 6. Level-2: Relational attention ----
        actor_exist = torch.cat([
            torch.ones(ego_e.shape[0], 1, device=ego_e.device, dtype=torch.bool),
            (nbr_s.abs() > 1e-6).any(dim=-1).any(dim=-1)   # (B, N)
        ], dim=1)

        actor = torch.cat([ego_e.unsqueeze(1), torch.stack(nbr_rel, dim=1)], dim=1)
        rel, _ = _safe_attention(self.rel_attn, ego_e.unsqueeze(1), actor, actor,
                                 key_padding_mask=~actor_exist)
        rel = self.rel_norm(F.relu(rel.squeeze(1)))

        # ---- 7. Level-4: Goal attention ----
        goals = []
        for attn in self.goal_attns:
            g, _ = _safe_attention(attn, rel.unsqueeze(1), ego_map, ego_map,
                                   key_padding_mask=~ego_mm)
            goals.append(g.squeeze(1))
        goals = self.goal_norm(F.relu(torch.stack(goals, dim=1)))

        # ---- 8. 残差融合 ----
        feat = goals + rel.unsqueeze(1).expand_as(goals)
        if self.num_modes == 1:
            feat = feat.squeeze(1)                          # (B, units)

        # ---- 9. [SVO-Game] Level-5: Game-Aware Cross-Attention ----
        if svo_mu is not None and pred_trajs is not None:
            assert feat.dim() == 2, (
                f"GameAwareCrossAttention requires num_modes=1 (feat 2D), "
                f"got feat shape {feat.shape} with num_modes={self.num_modes}"
            )
            feat = self.game_cross_attn(
                feat, svo_mu, svo_sigma, pred_trajs, interact_mask
            )

        return feat


# ======================================================================== #
#  Observation Parser                                                       #
# ======================================================================== #

def parse_observation(flat_obs, config):
    """
    flat vector → 结构化张量

    flat_obs: (B, total_obs_dim) or (total_obs_dim,)
    Returns: trajs (B, 1+N, T, 5), map_wps (B, M, L, 2), aux (B, aux_dim),
             bv_actions (B, N, 3) or None  [Level-k v9 新增]

    Layout:
        [trajs (1+N)*T*5] [map_wps M*L*2] [aux 10] [bv_actions N*3]
    """
    enc = config.encoder
    if flat_obs.dim() == 1:
        flat_obs = flat_obs.unsqueeze(0)
    ts = enc.trajs_flat_dim
    ms = enc.map_flat_dim
    ax = enc.aux_state_dim
    bv = enc.bv_actions_flat_dim   # 0 if disabled, N*3 if enabled

    trajs   = flat_obs[:, :ts].reshape(-1, 1+enc.num_neighbours, enc.history_steps, 5)
    map_wps = flat_obs[:, ts:ts+ms].reshape(-1, enc.total_map_polylines, enc.path_length, 2)
    aux     = flat_obs[:, ts+ms:ts+ms+ax]

    if bv > 0:
        bv_actions = flat_obs[:, ts+ms+ax:ts+ms+ax+bv].reshape(
            -1, enc.num_neighbours, enc.bv_action_feat_dim
        )
    else:
        bv_actions = None

    return trajs, map_wps, aux, bv_actions