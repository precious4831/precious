"""
CARLA PPO自动驾驶项目 - 配置文件 (RSSM-SVO增强版)

========================================================================
完整训练流程 (按顺序执行):
  Step 0: python data_collector.py --episodes 300 --num_npc 200 --output svo_dataset.npz
  Step 1: python pretrain_svo.py --dataset svo_dataset.npz --output svo_pretrained.pt
  Step 2: python train.py --stage l1
  Step 3: python train.py --stage svo_only --svo_pretrained svo_pretrained.pt
  Step 4: python train.py --stage l2 --svo_pretrained svo_pretrained.pt --level1_path checkpoints/level1_policy.pt
  验证:   python test.py --l1_model ... --svo_model ... --l2_model ... --level1_path ...
========================================================================

CARLA Version: 0.9.14
Python Version: 3.9
"""

import os
import json
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from datetime import datetime


@dataclass
class CarlaConfig:
    host: str = "localhost"
    port: int = 2000
    timeout: float = 120.0
    town: str = "Town04"
    weather_preset: str = "ClearNoon"
    ego_vehicle_filter: str = "vehicle.tesla.model3"
    npc_vehicle_filter: str = "vehicle.*"
    synchronous_mode: bool = True
    fixed_delta_seconds: float = 0.05


@dataclass
class ObservationConfig:
    ego_state_dim: int = 10
    max_num_neighbors: int = 10
    per_vehicle_dim: int = 7
    detection_radius: float = 60.0
    max_speed: float = 60.0
    max_acceleration: float = 10.0
    lateral_detection_range: float = 3.5
    add_obs_noise: bool = True
    pos_noise_std: float = 0.2
    vel_noise_std: float = 0.1
    yaw_noise_std: float = 0.05

    # [MandLC] 关键交互车(目标左车道最近NPC)专用观测维度
    # 7维: [exists, rel_x_norm, rel_y_norm, rel_vx_norm, rel_vy_norm, theta_cos, theta_sin]
    key_interactor_dim: int = 7

    @property
    def total_state_dim(self) -> int:
        return self.ego_state_dim + (self.max_num_neighbors * self.per_vehicle_dim)


@dataclass
class EncoderConfig:
    num_neighbours: int = 5
    history_steps: int = 10
    hidden_units: int = 128
    num_heads: int = 2
    num_modes: int = 1
    goal_head_num: int = 2
    make_rotation: bool = True
    ego_paths: int = 3
    neighbor_paths: int = 2
    path_length: int = 10
    path_spacing: float = 5.0
    # [v8] 回归 10 维 (原版), 去掉 MandLC 时代扩展的 key_interactor 7 维
    aux_state_dim: int = 10

    # [Level-k v9] 新增: 每个邻居 BV 的当前动作 [steer, throttle_brake, valid_mask]
    # 仅 L2 训练/测试时启用; L1 BV 训练时为 0 维
    bv_action_feat_dim: int = 3
    # 启用开关: 由 KLevelConfig.enable_bv_action_obs 决定
    # 但 obs 维度本身在 config 层面就要确定, 否则 encoder 维度不匹配
    # 解决: 用一个 flag, total_obs_dim 总是含 bv_actions, 但 L1 阶段填 0
    use_bv_actions_in_obs: bool = True

    @property
    def total_map_polylines(self):
        return self.ego_paths + self.num_neighbours * self.neighbor_paths

    @property
    def trajs_flat_dim(self):
        return (1 + self.num_neighbours) * self.history_steps * 5

    @property
    def map_flat_dim(self):
        return self.total_map_polylines * self.path_length * 2

    @property
    def bv_actions_flat_dim(self):
        """L2 时拼接的 BV 动作: (num_neighbours, bv_action_feat_dim)"""
        if self.use_bv_actions_in_obs:
            return self.num_neighbours * self.bv_action_feat_dim
        return 0

    @property
    def total_obs_dim(self):
        return (
            self.trajs_flat_dim
            + self.map_flat_dim
            + self.aux_state_dim
            + self.bv_actions_flat_dim
        )


@dataclass
class SLTConfig:
    """Sequential Latent Transformer 配置

    训练时辅助任务: 用连续T_f步的(latent, action)对,
    通过Transformer decoder预测未来latent,
    和真实未来latent做cosine similarity loss.
    推理时不使用 (零额外开销).

    参考: Liu et al. (2024) Scene-Rep Transformer
    """
    enabled: bool = True
    future_horizon: int = 5            # T_f: 预测未来步数
    latent_dim: int = 128              # encoder输出维度 (= encoder.hidden_units)
    action_embed_dim: int = 64         # action嵌入维度
    num_heads: int = 4                 # Transformer decoder注意力头数
    num_layers: int = 2                # Transformer decoder层数
    projector_dim: int = 128           # Projector Θ 隐层维度
    predictor_dim: int = 64            # Predictor P 隐层维度
    lr: float = 1e-4                   # SLT optimizer学习率
    encoder_lr_scale: float = 0.1      # encoder在SLT optimizer中的lr倍率
    batch_size: int = 64               # SLT更新batch大小
    queue_max_size: int = 10000        # 连续序列队列最大容量
    max_grad_norm: float = 1.0         # SLT梯度裁剪


@dataclass
class SVOConfig:
    """Variational BIRL SVO Inference Configuration"""
    enabled: bool = True
    hidden_dim: int = 64
    prediction_horizon: int = 10
    prior_mu: float = 45.0
    prior_sigma: float = 25.0
    beta: float = 0.05
    pretrain_lr: float = 1e-3
    pretrain_epochs: int = 100
    pretrain_batch_size: int = 128
    finetune_lr: float = 1e-5
    data_collect_episodes: int = 200
    teacher_forcing_ratio: float = 0.5
    # === RSSM Temporal Prior ===
    rssm_enabled: bool = True
    rssm_hidden_dim: int = 64
    # === Weak Supervision ===
    anchor_loss_weight: float = 3.0    # [Fix v5] anchor回到3.0, encoder相对化后应能区分


@dataclass
class KLevelConfig:
    """Level-k 博弈推理配置 (重构版)

    新的训练流程 (参考 Bouton 2020 / CHARMS 2024):
      L0 = CARLA TM (规则策略)
      L1 = "背景车策略", 任务=巡航/不撞 (无变道目标), 对手=L0
      L2 = "ego策略",    任务=变道/避障/到达,        对手=L1
                          ego 观测里显式拼 a_BV (Level-k best-response)

    关键变化:
      旧版: L1 训练用 ego 任务 reward, L2 时直接复用 L1 当 BV → BV 会去执行变道
      新版: L1 单独训练 (BV reward), L2 时 ego obs 拼 a_BV → 真博弈
    """
    # === IDM参数 (R_others计算用, L2 reward) ===
    prediction_horizon: int = 10
    idm_desired_speed: float = 25.0       # 期望速度 (m/s)
    idm_time_headway: float = 1.5         # 安全时距 (s)
    idm_max_accel: float = 2.0            # 最大加速度 (m/s²)
    idm_comfort_decel: float = 3.0        # 舒适减速度 (m/s²)
    idm_min_gap: float = 2.0              # 最小净距 (m)
    idm_accel_exponent: float = 4.0       # 加速度指数

    # === Level-k 训练配置 ===
    # L1 BV 训练 (新流程: 不带 SVO, 不带变道任务)
    level1_total_steps: int = 500_000     # L1 BV 训练总步数 (CHARMS 标准)
    level1_save_path: str = "checkpoints/level1_bv_policy.pt"

    # L2 ego 训练 (带 SVO + a_BV 拼接)
    level2_total_steps: int = 1_500_000   # L2 ego 训练总步数
    level2_save_path: str = "checkpoints/level2_ego_policy.pt"

    # === BV 策略控制 (L2训练/测试时) ===
    bv_control_mode: str = "tm"
    #   "tm"     : 所有BV用CARLA TM (L1 BV训练时, baseline测试时)
    #   "level1" : 检测范围内BV用L1策略 (L2训练时)
    bv_policy_path: str = ""              # L1 BV策略权重路径 (L2时需要)
    bv_control_radius: float = 60.0       # BV策略控制范围(米), 超出用TM
    bv_batch_inference: bool = True       # 批量推理 (多辆BV堆成batch)

    # === a_BV 观测拼接 (Level-k best-response 接口, L2 时启用) ===
    # 在 ego obs 的 aux 末尾拼 num_neighbours × bv_action_feat_dim 维
    # 每个邻居的当前动作 [steer, throttle_brake, valid_mask]
    # 对应 LevelKController 接管的 BV 用策略输出, 其他用 vehicle.get_control()
    enable_bv_action_obs: bool = True
    bv_action_feat_dim: int = 3           # [steer, throttle_brake, valid_mask]

    # === 奖励模式 ===
    reward_mode: str = "l2"
    #   "l1_bv" : R = BV 巡航 reward (新增, L1 BV 训练用)
    #   "l1"    : R = R_ego (旧版 ego baseline, 已废弃但保留)
    #   "l2"    : R = cos(θ)*R_ego + sin(θ)*R_others (L2 ego 训练用)


@dataclass
class BVRewardConfig:
    """L1 BV 训练专用 reward 配置 (新增)

    设计原则 (参考 Bouton 2020 keep-lane agent + CHARMS L1):
      - 鼓励正常巡航速度
      - 严厉惩罚碰撞
      - 鼓励待在车道里
      - 轻微惩罚变道 (允许但不奖励)
      - 鼓励平稳转向
      - 不含变道目标 / 不含路径终点 / 不含障碍避障

    这样训出来的 BV 默认会"自然往前开"，必要时博弈让行/抢道，
    不会主动去执行 ego 那种"必须变道到目标车道"的任务。
    """
    # === 巡航速度 ===
    target_speed: float = 60.0          # km/h, 鼓励的巡航速度
    min_speed: float = 5.0              # km/h, 低于此值惩罚 (避免 BV 学会停车)
    max_speed: float = 80.0             # km/h, 上限
    speed_reward_weight: float = 1.0    # 速度奖励权重
    low_speed_penalty: float = -0.3     # 速度过低惩罚

    # === 安全 ===
    collision_penalty: float = -200.0   # 碰撞 (与 ego reward 一致)
    min_safe_distance: float = 6.0      # 最小安全距离 (m)
    near_front_penalty: float = -0.1    # 太靠近前车的惩罚

    # === 车道保持 (核心: 让 BV 默认沿车道开) ===
    lane_keeping_weight: float = 0.8    # 高于 ego 的 0.5, 强约束
    max_lateral_offset: float = 1.0     # 横向容忍 (m)
    terminal_lateral_offset: float = 4.0  # 偏离 4m 触发终止 (和 ego 一致)
    terminal_lane_violation_penalty: float = -10.0

    # === 变道惩罚 (允许但不鼓励) ===
    # 这是和 ego reward 最大的区别: ego 鼓励变道, BV 惩罚变道
    lane_change_penalty: float = -0.5   # 每次车道 ID 变化触发
    lane_change_detection_dist: float = 2.5  # 横向位移超过此值算变道

    # === 平滑性 ===
    steering_penalty_weight: float = 0.15  # 比 ego 高一点, 鼓励平稳
    steering_smoothness_weight: float = 0.05  # 转向角变化率惩罚

    # === 时间 / 进度 ===
    # BV 不需要"到达终点"奖励, 但每步给微小时间奖励避免静止
    time_alive_reward: float = 0.02     # 每步活着的小奖励
    progress_reward: float = 0.0        # 不奖励进度 (BV 不赶路)


@dataclass
class ActionConfig:
    action_type: str = "continuous"
    action_dim: int = 2

    # 第一阶段：先保证能完成单次相邻车道变道
    max_steer: float = 0.30
    max_throttle: float = 0.70
    max_brake: float = 0.80
    throttle_brake_scale: float = 1.0

    # 放宽一点转向变化限制，减少“想变但转不过去”
    steer_rate_limit: float = 0.10       # [Fix] 0.08→0.10
    steer_smooth_factor: float = 0.45    # [Fix] 0.55→0.45

    speed_governor_margin_kmh: float = 2.0
    speed_governor_brake_kmh: float = 6.0


@dataclass
class ScenarioConfig:
    scenario_type: str = "highway"  # "highway" / "unprotected_left_turn"
    use_fixed_spawn: bool = True
    spawn_point_index: int = 105 #105
    spawn_location: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    spawn_rotation_yaw: float = 0.0
    route_length: float = 150.0
    enable_obstacles: bool = True
    # BV 专用训练可开启：保证自车同车道也有一定密度车流
    ensure_ego_lane_traffic: bool = False
    min_ego_lane_npc: int = 0
    # 无保护左转场景参数
    left_turn_npc_per_direction: int = 4
    left_turn_spawn_radius_min: float = 18.0
    left_turn_spawn_radius_max: float = 85.0
    left_turn_junction_search_dist: float = 90.0
    obstacle_vehicle_filter: str = "vehicle.audi.a2"
    obstacle_distance_range: Tuple[float, float] = (28.0, 40.0)
    waypoint_spacing: float = 2.0


@dataclass
class RewardConfig:
    """奖励函数配置 — v8 回归到旧版(验证有效)的值.

    注意: 这些参数是你旧版 config 里验证过有效的(能训出变道绕障).
    新训练直接复用, 别再动.
    """
    target_speed: float = 60.0
    min_speed: float = 10.0
    max_speed: float = 80.0

    speed_reward_weight: float = 1.5   # 速度奖励权重 (v8: 1.0 → 1.5, 旧版; 提高速度奖励权重, 鼓励更积极的巡航)
    low_speed_penalty: float = -1.0    # stronger low-speed penalty to discourage standing still

    # 车道相关
    lane_keeping_weight: float = 0.6   # v8: 0.6 → 0.5 (旧版)
    max_lateral_offset: float = 1.0     # v8: 1.5 → 1.0 (旧版: 超过1米开始惩罚, 4米触发终止; 回归后超过1米就惩罚, 超过4米终止)  
    terminal_lateral_offset: float = 4.0   # v8: 5.0 → 4.0 (旧版: 偏4米以上触发终止)
    terminal_cross_two_lanes_penalty: float = -120.0 # v8: -100 → -120 (旧版: 直接惩罚跨两条车道, 避免 BV 那种"先变道到旁边车道, 再变回目标车道"的怪异行为)
    terminal_lane_violation_penalty: float = -10.0  # 惩罚终止时的车道违规 (旧版没有, 新增: 让 agent 知道自己是因为车道违规被终止, 而不是单纯的"死了")

    # 安全相关
    safe_distance_weight: float = 0.5   # v8: 0.6 → 0.5 (旧版)
    min_safe_distance: float = 8.0      # v8: 10.0 → 8.0 (旧版)
    desired_headway_time: float = 2.0
    near_front_penalty: float = -0.1    # v8: -0.05 → -0.1 (旧版)
    collision_penalty: float = -200.0   # v8: -100 → -200 (旧版)

    # 平滑/舒适
    steering_penalty_weight: float = 0.2   # v8: 0.12 → 0.1 (旧版)

    # 时间/目标
    progress_reward: float = 0.0       # remove fixed alive bonus while debugging the "ego not moving" issue
    time_penalty: float = -0.05        # v8: -0.1 → -0.05 (旧版)
    reach_goal_reward: float = 100.0

    # SVO风险 (保留, L2路径用)
    svo_risk_weight: float = 0.3 #
    svo_risk_distance: float = 6.0 #

    # ================================================================== #
    # [v8] SVO 相关 (保留 — L2 训练核心)
    # ================================================================== #

    # Oracle / BIRL SVO 切换
    # True (默认, 训练推荐):  θ 来自 NPC 真实 style_label
    #                         (aggressive→15°, semi_aggr→30°, normal→45°,
    #                          semi_cons→60°, conservative→75°)
    # False (消融/测试):       θ 来自 BIRL 推断 (_svo_mu)
    #
    # 可在命令行 --use_oracle_svo / --no_oracle_svo 切换 (见 train.py/test.py)
    use_oracle_svo: bool = False


@dataclass
class TrafficConfig:
    num_npc_vehicles: int = 20
    global_speed_percentage: float = 80.0
    global_distance_percentage: float = 100.0
    ignore_lights_percentage: float = 0.0
    ignore_signs_percentage: float = 0.0
    random_lane_change_percentage: float = 10.0
    aggressive_driving: bool = False


@dataclass
class PPOConfig:
    hidden_sizes: List[int] = field(default_factory=lambda: [256, 256, 128])
    activation: str = "relu"
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    lr_decay: bool = True
    lr_decay_rate: float = 0.9995
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    rollout_steps: int = 2048
    mini_batch_size: int = 64
    ppo_epochs: int = 10
    normalize_advantage: bool = True
    target_kl: Optional[float] = 0.02
    use_dapo: bool = False
    clip_eps_low: float = 0.2
    clip_eps_high: float = 0.28

    # ============================================================== #
    # [SPO] Simple Policy Optimization (Xie et al. ICML 2025)          #
    # ============================================================== #
    # 主开关 (与 use_dapo 互斥, 由 train.py 根据 --algo spo 设置)
    # use_spo=False  → 标准 PPO clipping (现有行为不变)
    # use_spo=True   → SPO quadratic penalty 替换 ratio clipping
    use_spo: bool = False
    # SVO-guided adaptive ε:
    #   False → 固定 ε = spo_epsilon_base (复刻 SPO 论文)
    #   True  → ε_t = clip(ε_base * exp(-α * ρ_t), ε_min, ε_max), 周车交互风险高时收缩
    use_svo_adaptive_spo: bool = True
    spo_epsilon_base: float = 0.2     # 基准 ε (= SPO 论文默认值, 也对应当前 PPO clip_epsilon)
    spo_epsilon_min: float = 0.05     # 自适应 ε 下界 (高交互风险场景)
    spo_epsilon_max: float = 0.2      # 自适应 ε 上界 (= ε_base, 不允许大于基准)
    spo_risk_alpha: float = 1.5       # ρ_t 的指数衰减系数

    # ============================================================== #
    # [CVaR] SVO-aware CVaR-constrained Lagrangian PPO                 #
    # ============================================================== #
    # 主开关 (L1 BV 训练强制关闭, L2/svo_only 默认开启)
    cvar_enabled: bool = True
    cvar_alpha: float = 0.1                  # tail 比例
    cost_gamma: float = 0.99                 # cost return 折扣
    cvar_budget_base: float = 4.0            # d0
    cvar_budget_min_ratio: float = 0.2       # d_e 下限 = d0 * 此值, 防 C/d 爆炸
    cvar_lambda_init: float = 0.0
    cvar_lambda_lr: float = 3e-4         # λ 学习率 (与 critic_lr 同级别, 但通常 PPO 训练初期 λ 更新较慢, 可以适当调高)
    cvar_lambda_max: float = 0.2
    cvar_cost_coef: float = 0.3         
    cost_value_coef: float = 0.5             # 兼容字段, 当前未使用
    cost_loss_coef: float = 1.0
    # SVO -> 紧迫度: u = w_mu*(1 - mu/90) + w_sigma*(sigma/prior_sigma)
    svo_mu_budget_weight: float = 0.5
    svo_sigma_budget_weight: float = 0.5
    svo_budget_beta: float = 1.0             # d_e = d0 * exp(-beta * U_e)
    # step cost 权重
    cost_w_collision: float = 1.0
    cost_w_lane: float = 0.5
    cost_w_safe: float = 0.1
    cost_eps: float = 1e-6


@dataclass
class TrainConfig:
    max_timesteps: int = 1500000       # [Fix] 1.5M→3M 训练更充分
    max_episodes: int = 2000
    max_episode_steps: int = 1000
    max_episode_time: float = 60.0
    save_freq_steps: int = 50000
    save_freq_episodes: int = 100
    eval_freq_steps: int = 20000
    eval_episodes: int = 5
    log_freq_steps: int = 2048
    seed: int = 42
    device: str = "auto"
    base_output_dir: str = "./outputs"


@dataclass
class TestConfig:
    num_episodes: int = 20
    max_episode_steps: int = 200
    deterministic: bool = True
    base_output_dir: str = "./outputs"


@dataclass
class VisualConfig:
    enable: bool = True
    render_waypoints: bool = True
    width: int = 1440
    height: int = 810
    camera_distance: float = 12.0
    camera_height: float = 6.0
    camera_pitch: float = -20.0
    waypoint_color: Tuple[int, int, int] = (0, 255, 0)
    waypoint_size: float = 0.08
    num_waypoints_to_draw: int = 40
    fps: int = 30


class Config:
    def __init__(self):
        self.carla = CarlaConfig()
        self.observation = ObservationConfig()
        self.action = ActionConfig()
        self.scenario = ScenarioConfig()
        self.reward = RewardConfig()
        self.bv_reward = BVRewardConfig()       # [Level-k v9] L1 BV 训练专用 reward
        self.traffic = TrafficConfig()
        self.ppo = PPOConfig()
        self.train = TrainConfig()
        self.test = TestConfig()
        self.visual = VisualConfig()
        self.encoder = EncoderConfig()
        self.svo = SVOConfig()
        self.slt = SLTConfig()
        self.klevel = KLevelConfig()

    def create_experiment_dir(self, mode: str = "train") -> dict:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = self.train.base_output_dir if mode == "train" else self.test.base_output_dir
        experiment_dir = os.path.join(base_dir, f"{mode}_{timestamp}")
        dirs = {
            "experiment": experiment_dir,
            "tensorboard": os.path.join(experiment_dir, "tensorboard"),
            "models": os.path.join(experiment_dir, "models"),
            "logs": os.path.join(experiment_dir, "logs"),
        }
        if mode == "test":
            dirs["videos"] = os.path.join(experiment_dir, "videos")
            dirs["results"] = os.path.join(experiment_dir, "results")
        for path in dirs.values():
            os.makedirs(path, exist_ok=True)
        return dirs

    def save(self, path: str):
        config_dict = {}
        for name in ['carla','observation','action','scenario','reward','bv_reward',
                      'traffic','ppo','train','test','visual','encoder','svo','slt','klevel']:
            section = getattr(self, name)
            config_dict[name] = {k:v for k,v in section.__dict__.items() if not k.startswith('_')}
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=4)

    @classmethod
    def load(cls, path: str) -> 'Config':
        config = cls()
        with open(path, 'r') as f:
            config_dict = json.load(f)
        for section_name, section_dict in config_dict.items():
            if hasattr(config, section_name):
                section = getattr(config, section_name)
                for key, value in section_dict.items():
                    if hasattr(section, key):
                        setattr(section, key, value)
        return config

    def print_config(self):
        print("=" * 60)
        print("配置信息")
        print("=" * 60)
        print(f"观测维度: {self.encoder.total_obs_dim}")
        print(f"Encoder: HierarchicalTransformer units={self.encoder.hidden_units}")
        print(f"动作维度: {self.action.action_dim}")
        print(f"目标速度: {self.reward.target_speed} km/h")
        print(f"NPC数量: {self.traffic.num_npc_vehicles}")
        print(f"SVO-Game: {'启用' if self.svo.enabled else '禁用'} | "
              f"RSSM: {'启用' if self.svo.rssm_enabled else '禁用'} | "
              f"prior=N({self.svo.prior_mu}, {self.svo.prior_sigma}) | "
              f"anchor_w={self.svo.anchor_loss_weight}")
        print(f"Level-k: BV控制={self.klevel.bv_control_mode} | "
              f"奖励模式={self.klevel.reward_mode} | "
              f"控制半径={self.klevel.bv_control_radius}m")
        if self.ppo.cvar_enabled:
            print(f"CVaR: enabled | alpha={self.ppo.cvar_alpha} | "
                  f"d0={self.ppo.cvar_budget_base} | "
                  f"lambda_lr={self.ppo.cvar_lambda_lr} | "
                  f"cost_coef={self.ppo.cvar_cost_coef} | "
                  f"beta_svo={self.ppo.svo_budget_beta}")
        else:
            print("CVaR: disabled")
        # [SPO] 算法状态
        if self.ppo.use_spo:
            _spo_mode = "SVO-Adaptive" if self.ppo.use_svo_adaptive_spo else "Fixed"
            print(f"SPO: enabled ({_spo_mode}) | "
                  f"ε_base={self.ppo.spo_epsilon_base} | "
                  f"ε_range=[{self.ppo.spo_epsilon_min}, {self.ppo.spo_epsilon_max}] | "
                  f"α={self.ppo.spo_risk_alpha}")
        elif self.ppo.use_dapo:
            print(f"Algorithm: DAPO | "
                  f"clip=[{self.ppo.clip_eps_low}, {self.ppo.clip_eps_high}]")
        else:
            print(f"Algorithm: PPO | clip_ε={self.ppo.clip_epsilon}")
        print("=" * 60)


def get_default_config() -> Config:
    return Config()

def get_highway_config() -> Config:
    config = Config()
    config.carla.town = "Town04"
    config.reward.target_speed = 60.0
    config.traffic.num_npc_vehicles = 40
    return config

def get_debug_config() -> Config:
    config = Config()
    config.traffic.num_npc_vehicles = 10
    config.train.max_timesteps = 50000
    config.ppo.rollout_steps = 512
    return config

def get_unprotected_left_turn_config() -> Config:
    """
    一键无保护左转配置:
    - Town03
    - 固定出生点 47
    - 无静态障碍
    - NPC 忽略信号灯，四方向均有车流
    """
    config = Config()
    config.carla.town = "Town03"
    config.scenario.scenario_type = "unprotected_left_turn"
    config.scenario.use_fixed_spawn = True
    config.scenario.spawn_point_index = 47
    config.scenario.enable_obstacles = False
    config.scenario.route_length = 120.0
    config.scenario.left_turn_npc_per_direction = 4
    config.scenario.left_turn_spawn_radius_min = 18.0
    config.scenario.left_turn_spawn_radius_max = 85.0
    config.scenario.left_turn_junction_search_dist = 90.0

    # 四方向总车流规模
    config.traffic.num_npc_vehicles = 16
    # 按你的要求: NPC 忽略灯控
    config.traffic.ignore_lights_percentage = 100.0
    config.traffic.ignore_signs_percentage = 100.0
    # 路口场景先关闭随机变道，稳定车流
    config.traffic.random_lane_change_percentage = 0.0
    return config

if __name__ == "__main__":
    config = Config()
    config.print_config()
