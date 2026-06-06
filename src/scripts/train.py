"""
CARLA 强化学习训练脚本 (Level-k v9 - 真博弈版).

支持 PPO / DAPO + SVO + Level-k 完整三阶段训练.

完整流程
============================================================
Step 0) 收集 SVO 预训练数据 (一次性)
  python data_collector.py --episodes 300 --output svo_dataset.npz

Step 1) SVO 模块预训练 (一次性)
  python pretrain_svo.py --dataset svo_dataset.npz --output svo_pretrained.pt

Step 2) L1 BV 训练 (新流程: 用 BVTrainEnv, BV 任务 reward, 不带 SVO)
  python train.py --stage l1
  → 输出: outputs/L1_BV/.../models/best_model.pth (= L1 BV 策略)

Step 3) L2 ego 训练 (周围 BV 用 Step 2 训出的 L1 策略接管)
  python train.py --stage l2 \\
      --svo_pretrained svo_pretrained.pt \\
      --level1_path outputs/L1_BV/.../models/best_model.pth
  → ego obs 显式拼接 a_BV, 实现真正的 Level-k best-response

消融对比 (可选)
  python train.py --stage svo_only --svo_pretrained svo_pretrained.pt
  ↑ SVO 但 BV=TM (没有 Level-k 真博弈, 用于对照)

恢复训练
  python train.py --resume /path/to/checkpoint.pth --total_timesteps N
============================================================

v9 vs v6 关键变化:
  - 砍掉旧的 --stage l1_bv (合并到 --stage l1)
  - --stage l1 现在是真正的 BV 训练 (BVTrainEnv + BVRewardConfig)
  - --stage l2 默认启用 LevelKController (BV=L1, ego obs 拼 a_BV)
  - obs 维度从 570 → 585 (新增 num_neighbours x 3 = 15 维 a_BV)
"""

import argparse
import os
from pathlib import Path
import random
import re
import signal
import sys
import time
from collections import deque
from datetime import datetime
from typing import Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# [Progress] tqdm 进度条 (可选依赖, 缺失时退化为 no-op)
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False
    print("[Warning] 未安装 tqdm, 进度条功能禁用. 安装: pip install tqdm")

from src.envs.carla_env import CarlaEnv, make_env
from src.config import (
    Config,
    get_debug_config,
    get_default_config,
    get_highway_config,
    get_unprotected_left_turn_config,
)
from src.algorithms.ppo_model import PPOAgent

try:
    from src.algorithms.dapo_model import DAPOCarlaAgent

    DAPO_AVAILABLE = True
except ImportError:
    DAPO_AVAILABLE = False

try:
    from src.algorithms.spo_model import SPOCarlaAgent

    SPO_AVAILABLE = True
except ImportError:
    SPO_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="CARLA PPO 训练脚本")

    # 基础配置
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        choices=["default", "highway", "debug", "unprotected_left_turn"],
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", type=str, default="Town04")

    # 训练参数
    parser.add_argument("--total_timesteps", type=int, default=1_500_000)
    parser.add_argument("--num_npc", type=int, default=20)
    parser.add_argument("--target_speed", type=float, default=60.0)

    # PPO 参数
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--clip_epsilon", type=float, default=0.2)
    parser.add_argument("--rollout_steps", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=64)

    # 系统参数
    parser.add_argument("--seed", type=int, default=517)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, default="./outputs")

    # 频率参数
    parser.add_argument("--save_freq", type=int, default=50_000)
    parser.add_argument("--eval_freq", type=int, default=20_000)
    parser.add_argument("--log_freq", type=int, default=2048)

    # 恢复训练
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的 checkpoint 路径")

    # 算法选择
    parser.add_argument("--algo", type=str, default="spo", choices=["ppo", "dapo", "spo"])
    # [SPO] 是否启用 SVO-guided 自适应 ε (仅 --algo spo 时生效)
    parser.add_argument(
        "--no_svo_adaptive_spo",
        action="store_true",default=True,
        help="--algo spo 时关闭 SVO-adaptive ε, 退化为 fixed SPO (消融实验用)"
    )

    # SVO 相关
    parser.add_argument("--no_svo", action="store_true", default=False, help="禁用 SVO 模块")
    parser.add_argument("--svo_pretrained", type=str, default=r"D:\桌面\投稿\SVO-CVaR\pretrain_svo\svo_pretrained.pt", help="SVO 预训练权重路径")

    # [CVaR] 命令行开关 (默认开, 仅 ablation 时关)
    parser.add_argument("--no_cvar", action="store_true",default=False,
                        help="禁用 CVaR 安全约束 (用于 ablation 实验)")

    # 分阶段训练 [Level-k v9 重构]
    #   l1       = L1 BV 训练 (新流程, 用 bv_env, BV 任务 reward, 不带 SVO)
    #   svo_only = SVO 预训练之后用 ego 任务训练, BV=TM (消融对比, 无 Level-k 博弈)
    #   l2       = L2 ego 训练 (BV=L1, ego obs 拼 a_BV, 真博弈)
    parser.add_argument("--stage", type=str, default="l2", choices=["l1", "svo_only", "l2"])
    parser.add_argument("--level1_path", type=str, default=None, help="L2 阶段所需的 L1_BV 策略路径")

    # [v8] SVO θ 来源: Oracle (NPC 真实风格标签) vs BIRL (预训练推断)
    # 默认 oracle (训练稳定), 可在消融/测试切回 BIRL
    parser.add_argument(
        "--use_oracle_svo",
        action="store_true",
        default=None,
        help="L2 奖励中 SVO θ 用 NPC 真实风格标签 (默认, 推荐训练用)"
    )
    parser.add_argument(
        "--no_oracle_svo",
        action="store_true",
        help="L2 奖励中 SVO θ 用 BIRL 推断 (消融实验/测试用)"
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    """设置随机种子，提升实验可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def create_config_from_args(args: argparse.Namespace) -> Config:
    """根据命令行参数构建配置对象。"""
    if args.config == "highway":
        config = get_highway_config()
    elif args.config == "debug":
        config = get_debug_config()
    elif args.config == "unprotected_left_turn":
        config = get_unprotected_left_turn_config()
    else:
        config = get_default_config()

    # 算法开关 (PPO / DAPO / SPO 三选一, 互斥)
    if args.algo == "dapo":
        config.ppo.use_dapo = True
        config.ppo.use_spo = False
        config.ppo.clip_eps_high = 0.28
    elif args.algo == "spo":
        config.ppo.use_dapo = False
        config.ppo.use_spo = True
        # 默认开 SVO-adaptive, 加 --no_svo_adaptive_spo 退化为 fixed SPO
        if args.no_svo_adaptive_spo:
            config.ppo.use_svo_adaptive_spo = False
    else:
        config.ppo.use_dapo = False
        config.ppo.use_spo = False

    # CARLA 连接参数
    config.carla.host = args.host
    config.carla.port = args.port
    config.carla.town = args.town

    # 训练主参数
    config.train.max_timesteps = args.total_timesteps
    config.traffic.num_npc_vehicles = args.num_npc
    config.reward.target_speed = args.target_speed

    # PPO 超参数
    config.ppo.actor_lr = args.lr
    config.ppo.critic_lr = args.lr
    config.ppo.gamma = args.gamma
    config.ppo.clip_epsilon = args.clip_epsilon
    config.ppo.rollout_steps = args.rollout_steps
    config.ppo.mini_batch_size = args.batch_size

    # 其他系统配置
    config.train.seed = args.seed
    config.train.device = args.device
    stage_dir_map = {
        "l1": "L1_BV",
        "l2": "L2",
        "svo_only": "SVO_ONLY",
    }
    stage_dir = stage_dir_map.get(args.stage, args.stage.upper())
    config.train.base_output_dir = os.path.join(args.output_dir, stage_dir)
    config.train.save_freq_steps = args.save_freq
    config.train.eval_freq_steps = args.eval_freq
    config.train.log_freq_steps = args.log_freq

    # ============================================================== #
    # [Level-k v9] 按阶段切换策略                                       #
    # ============================================================== #
    if args.stage == "l1":
        # L1 BV 训练: 任务 = "在车流中正常行驶 + 不撞", 不变道目标
        # 使用 BVTrainEnv (继承 CarlaEnv, 强制无障碍 + BV reward)
        # BV 不需要推断别人风格 → 关闭 SVO
        config.svo.enabled = False
        config.klevel.bv_control_mode = "tm"            # L1 训练时周围全 TM
        config.klevel.reward_mode = "l1_bv"             # 用 BVRewardConfig
        config.scenario.enable_obstacles = False        # BV 任务无障碍
        config.train.max_timesteps = config.klevel.level1_total_steps
        # [CVaR] L1 BV 阶段强制禁用 CVaR (BV 没 SVO 信号, CVaR 失去动态调节意义;
        # BV reward 已含强 collision penalty, 不需要额外约束)
        config.ppo.cvar_enabled = False
        print("[Level-k v9] 阶段 L1 BV: 周围 TM, 关闭 SVO, 无障碍, CVaR 禁用, "
              "BV 巡航 reward (target={}km/h)".format(config.bv_reward.target_speed))

    elif args.stage == "svo_only":
        # 消融实验: 用 SVO 但 BV=TM (无 Level-k 真博弈, 用于对照)
        config.svo.enabled = True
        config.klevel.bv_control_mode = "tm"
        config.klevel.reward_mode = "l2"
        config.train.max_timesteps = config.klevel.level2_total_steps
        print("[Level-k v9] 阶段 SVO-only: 启用 SVO, BV=TM (消融, 无 Level-k 博弈)")

    elif args.stage == "l2":
        # L2 ego 训练: 周围 BV 用 L1 BV 策略接管 + ego obs 拼 a_BV → 真博弈
        config.svo.enabled = True
        config.klevel.reward_mode = "l2"
        config.train.max_timesteps = config.klevel.level2_total_steps

        if args.level1_path:
            # 启用 L1 BV 接管 (真 Level-k 博弈)
            config.klevel.bv_control_mode = "level1"
            config.klevel.bv_policy_path = args.level1_path
            print(f"[Level-k v9] 阶段 L2: 启用 SVO + L1 BV 接管, BV 策略={args.level1_path}")
            print(f"[Level-k v9]   ego obs 末尾会拼接 a_BV (num_neighbours x 3 = "
                  f"{config.encoder.bv_actions_flat_dim} 维)")
        else:
            # 未提供 L1 BV 路径: 退化到 BV=TM (相当于 svo_only)
            config.klevel.bv_control_mode = "tm"
            print("[Level-k v9] 阶段 L2: 启用 SVO, 但未提供 --level1_path, BV=TM (退化为 svo_only)")
            print("[Level-k v9]   提示: 真 Level-k 博弈需要先用 --stage l1 训练 L1 BV")

    # 手动关闭 SVO 的最高优先级开关
    if args.no_svo:
        config.svo.enabled = False
        print("[SVO] 已通过 --no_svo 禁用")

    # [CVaR] 手动关闭 CVaR (ablation 实验用)
    if args.no_cvar:
        config.ppo.cvar_enabled = False
        print("[CVaR] 已通过 --no_cvar 禁用")

    # [v8] Oracle / BIRL SVO θ 开关 (仅 L2 阶段生效)
    if args.no_oracle_svo:
        config.reward.use_oracle_svo = False
        print("[SVO θ] 已通过 --no_oracle_svo 切换到 BIRL 推断模式")
    elif args.use_oracle_svo:
        config.reward.use_oracle_svo = True
        print("[SVO θ] 已通过 --use_oracle_svo 启用 Oracle 模式 (默认)")

    # === 最终 SVO 状态打印 ===
    if config.klevel.reward_mode == "l2":
        use_oracle = getattr(config.reward, 'use_oracle_svo', True)
        print(f"\n[SVO θ] === L2 SVO 模式 ===")
        if use_oracle:
            print(f"  Oracle: θ 来自 NPC 真实风格标签")
            print(f"    aggressive      → 15°")
            print(f"    semi_aggressive → 30°")
            print(f"    normal          → 45°")
            print(f"    semi_conservative → 60°")
            print(f"    conservative    → 75°")
        else:
            print(f"  BIRL: θ 来自 svo_birl 预训练模型的推断平均值")
        print(f"  公式: R = cos(θ)·r_ego + sin(θ)·r_others·2.0  (Toghi 2022)\n")

    return config


def evaluate(env: CarlaEnv, agent: PPOAgent, n_episodes: int = 5) -> Dict[str, float]:
    """评估当前策略, 返回奖励/速度/碰撞率/成功率等指标."""
    rewards = []
    lengths = []
    speeds = []
    collision_count = 0
    success_count = 0

    for _ in range(n_episodes):
        obs = env.reset()
        if hasattr(agent, "reset_svo_state"):
            agent.reset_svo_state()
        done = False
        ep_reward = 0.0
        ep_length = 0
        ep_speeds = []
        last_info = {}

        while not done:
            result = agent.select_action(obs, deterministic=True)
            action = result[0]
            # 评估阶段也同步 SVO 信息, 保持与训练路径一致
            env.set_svo_info(result[3], result[4], result[5], result[6])
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            ep_length += 1
            ep_speeds.append(info.get("speed_kmh", 0.0))
            last_info = info

        rewards.append(ep_reward)
        lengths.append(ep_length)
        speeds.append(float(np.mean(ep_speeds)) if ep_speeds else 0.0)
        if last_info.get('collision', False):
            collision_count += 1
        if last_info.get('success', False):
            success_count += 1

    return {
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "std_reward": float(np.std(rewards)) if rewards else 0.0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "mean_speed": float(np.mean(speeds)) if speeds else 0.0,
        "collision_rate": collision_count / max(1, n_episodes),
        "success_rate_eval": success_count / max(1, n_episodes),
    }


class InterruptHandler:
    """[Progress] 安全 Ctrl+C 退出处理器.

    第一次 Ctrl+C: 标记 should_exit=True, 让训练循环在下个 step 边界优雅退出 (保存 ckpt).
    第二次 Ctrl+C: 立即抛 KeyboardInterrupt (强退).
    跨平台 (Windows + Linux + Mac), 用 signal.signal 标准库.
    """

    def __init__(self):
        self.should_exit = False
        self._interrupt_count = 0
        self._original_handler = None

    def __enter__(self):
        self._original_handler = signal.signal(signal.SIGINT, self._handle)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 恢复原 handler
        if self._original_handler is not None:
            try:
                signal.signal(signal.SIGINT, self._original_handler)
            except Exception:
                pass

    def _handle(self, signum, frame):
        self._interrupt_count += 1
        if self._interrupt_count == 1:
            self.should_exit = True
            # 不在 handler 内做 print (可能不安全), 让主循环看到 should_exit 后处理
            sys.stderr.write(
                "\n[Interrupt] 收到 Ctrl+C, 训练将在当前 step 完成后保存退出. "
                "再按一次 Ctrl+C 强制退出.\n"
            )
            sys.stderr.flush()
        else:
            sys.stderr.write("\n[Interrupt] 强制退出.\n")
            sys.stderr.flush()
            # 恢复默认行为, 让本次 KeyboardInterrupt 直接传播
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            raise KeyboardInterrupt


def _compute_svo_budget_term(svo_mu, svo_sigma, interact_mask, ppo_cfg, prior_sigma):
    """[CVaR] step-wise SVO 紧迫度: u_t = mean over active i of
    (w_mu * (1 - mu_i/90) + w_sigma * sigma_i/prior_sigma).
    无 active neighbor 时返回 0.0. clip 到 [0, ~2].
    """
    if svo_mu is None or interact_mask is None:
        return 0.0
    mask = np.asarray(interact_mask, dtype=bool)
    if not mask.any():
        return 0.0
    mu = np.clip(np.asarray(svo_mu, dtype=np.float32)[mask] / 90.0, 0.0, 1.0)
    sg = np.asarray(svo_sigma, dtype=np.float32)[mask] / max(prior_sigma, 1e-6)
    u = float((ppo_cfg.svo_mu_budget_weight * (1.0 - mu)
               + ppo_cfg.svo_sigma_budget_weight * sg).mean())
    return max(0.0, u)


def resolve_train_dirs(config: Config, resume_path: Optional[str] = None) -> Tuple[dict, bool]:
    """
    解析训练输出目录。

    返回:
    - dirs: 目录字典（experiment/tensorboard/models/logs）
    - is_resuming_same_run: 是否在“同一实验目录”上续训
    """
    if resume_path and os.path.exists(resume_path):
        resume_abs = os.path.abspath(resume_path)
        models_dir = os.path.dirname(resume_abs)
        experiment_dir = os.path.dirname(models_dir)

        # 仅当 checkpoint 位于 .../models/ 下时，才判定为复用原实验目录
        if os.path.basename(models_dir).lower() == "models":
            dirs = {
                "experiment": experiment_dir,
                "tensorboard": os.path.join(experiment_dir, "tensorboard"),
                "models": models_dir,
                "logs": os.path.join(experiment_dir, "logs"),
            }
            for path in dirs.values():
                os.makedirs(path, exist_ok=True)
            return dirs, True

        print(f"[Resume][Warning] checkpoint 不在 models 目录下: {resume_abs}")
        print("[Resume][Warning] 将回退到新建实验目录。")

    return config.create_experiment_dir("train"), False


def maybe_infer_step_from_ckpt_name(ckpt_path: str) -> int:
    """兼容老 checkpoint：尝试从文件名 checkpoint_700000.pth 推断步数。"""
    ckpt_name = os.path.basename(ckpt_path)
    match = re.search(r"checkpoint_(\d+)\.pth$", ckpt_name)
    return int(match.group(1)) if match else 0


def train(config: Config, resume_path: Optional[str] = None, svo_pretrained: Optional[str] = None):
    """主训练流程。"""
    set_seed(config.train.seed)

    if resume_path and not os.path.exists(resume_path):
        raise FileNotFoundError(f"恢复训练失败，找不到 checkpoint: {resume_path}")

    # 目录策略：恢复训练时优先复用原实验目录
    dirs, is_resuming_same_run = resolve_train_dirs(config, resume_path)
    if is_resuming_same_run:
        print(f"[Resume] 复用实验目录: {dirs['experiment']}")
    else:
        print(f"实验目录: {dirs['experiment']}")

    # 保存配置：续训时不覆盖原 config.json，而是另存一份 resume 配置快照
    config_path = os.path.join(dirs["logs"], "config.json")
    if is_resuming_same_run and os.path.exists(config_path):
        resume_cfg_path = os.path.join(
            dirs["logs"], f"config_resume_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        config.save(resume_cfg_path)
        print(f"[Resume] 当前启动配置已另存: {resume_cfg_path}")
    else:
        config.save(config_path)

    config.print_config()

    print("\n创建 CARLA 环境...")
    # [Level-k v9] L1 BV 训练用 BVTrainEnv (BV reward, 无障碍, 不带 SVO 评估)
    # 其他阶段 (svo_only / l2) 仍用普通 CarlaEnv (ego 任务)
    if config.klevel.reward_mode == "l1_bv":
        from src.envs.bv_env import make_bv_env
        env = make_bv_env(config, config.train.seed)
        print("[Level-k v9] 已加载 BVTrainEnv (L1 BV 训练专用)")
    else:
        env = make_env(config, config.train.seed)
        print("[Level-k v9] 已加载 CarlaEnv (ego 任务)")

    # 初始化智能体 (PPO / DAPO / SPO 三选一, 三态互斥)
    if config.ppo.use_dapo and DAPO_AVAILABLE:
        print("\n[System] 初始化 DAPO...")
        agent = DAPOCarlaAgent(config)
    elif config.ppo.use_spo and SPO_AVAILABLE:
        print("\n[System] 初始化 SPO...")
        agent = SPOCarlaAgent(config)
    else:
        if config.ppo.use_dapo and not DAPO_AVAILABLE:
            print("[Warning] 未找到 dapo_model.py，回退到 PPO。")
        if config.ppo.use_spo and not SPO_AVAILABLE:
            print("[Warning] 未找到 spo_model.py，回退到 PPO。")
        algo_name = "PPO + SVO-Game" if config.svo.enabled else "PPO"
        print(f"\n[System] 初始化 {algo_name}...")
        agent = PPOAgent(config)

    # 加载 SVO 预训练权重（仅在启用 SVO 且传了路径时）
    if svo_pretrained and config.svo.enabled:
        agent.load_svo_pretrained(svo_pretrained)
    elif config.svo.enabled and not svo_pretrained:
        print("[Warning] SVO 已启用但未提供 --svo_pretrained，SVO 将随机初始化。")

    # Level-k: 可选启用 L1 策略控制 BV（当前默认 BV=TM）
    if config.klevel.bv_control_mode == "level1":
        from src.algorithms.klevel import LevelKController

        l1_path = config.klevel.bv_policy_path
        if not l1_path or not os.path.exists(l1_path):
            raise FileNotFoundError(
                "L2 训练需要有效的 L1_BV 权重路径，请传 --level1_path 或在配置里设置 klevel.bv_policy_path"
            )
        bv_controller = LevelKController(config, device=agent.device)
        bv_controller.load_policy(l1_path, config)
        env.set_bv_control(bv_controller)
        print(f"[Level-k] BV 控制已启用，L1_BV 权重: {l1_path}")
    else:
        bv_controller = None

    # 恢复训练：加载网络、优化器与计数状态
    if resume_path:
        agent.load(resume_path)
        print(f"[Resume] 从 {resume_path} 恢复训练")
        if getattr(agent, "total_timesteps", 0) <= 0:
            inferred_step = maybe_infer_step_from_ckpt_name(resume_path)
            if inferred_step > 0:
                agent.total_timesteps = inferred_step
                print(f"[Resume] 从文件名推断已训练步数: {agent.total_timesteps}")

    total_timesteps = int(config.train.max_timesteps)
    rollout_steps = int(config.ppo.rollout_steps)
    current_timestep = int(getattr(agent, "total_timesteps", 0))
    episode_count = int(getattr(agent, "episode_count", 0))
    agent.total_timesteps = current_timestep
    agent.episode_count = episode_count

    if current_timestep >= total_timesteps:
        print(
            f"[Info] 当前步数 {current_timestep} 已达到/超过目标总步数 {total_timesteps}，"
            "将直接保存并结束。"
        )

    # TensorBoard：续训时在原目录续写，并从当前 step 继续
    if is_resuming_same_run:
        writer = SummaryWriter(log_dir=dirs["tensorboard"], purge_step=current_timestep)
        print(f"[Resume] TensorBoard 续写: {dirs['tensorboard']}")
    else:
        writer = SummaryWriter(dirs["tensorboard"])

    episode_reward = 0.0
    episode_length = 0
    best_reward = -float("inf")
    episode_rewards = []
    success_history = deque(maxlen=100)
    # episode 终止原因分布 (滑动窗口, 保留用于调试)
    termination_reasons = deque(maxlen=100)

    # [v8] L2 SVO 统计 (代替 MandLC 专用指标)
    #   - 记录每 episode 的平均 θ (oracle/BIRL 都适用)
    #   - 用于 tensorboard 绘制 L2/avg_theta_deg 曲线
    l2_stage = (config.klevel.reward_mode == "l2")
    ep_theta_sum = 0.0
    ep_theta_count = 0

    print(f"\n开始训练, 总步数: {total_timesteps}")
    print(f"当前进度: step={current_timestep}/{total_timesteps}, episode={episode_count}")
    print("=" * 60)

    # 让环境侧 Episode 打印也从恢复编号继续
    if hasattr(env, "episode_count"):
        env.episode_count = episode_count

    obs = env.reset()
    wall_start = time.time()
    run_start_step = current_timestep

    # [Progress] 计算 stage 名 (用于进度条 description)
    _stage_name = {"l1_bv": "L1-BV", "l2": "L2", "l1": "L1"}.get(
        config.klevel.reward_mode, config.klevel.reward_mode.upper()
    )

    reached_target_steps = False
    interrupt_handler = InterruptHandler()
    pbar = None
    try:
        interrupt_handler.__enter__()  # 注册 SIGINT handler

        # [Progress] tqdm 进度条 (按 step 粒度, total = total_timesteps)
        if _HAS_TQDM:
            pbar = tqdm(
                total=total_timesteps,
                initial=current_timestep,
                desc=f"[{_stage_name}]",
                unit="step",
                dynamic_ncols=True,
                ascii=True,        # Windows cmd 不支持 unicode 块字符
                file=sys.stdout,
                leave=True,
                miniters=max(1, rollout_steps // 50),  # 每 ~2% 刷新一次, 避免 CARLA 慢时刷屏
            )

        while current_timestep < total_timesteps:
            # [Progress] Ctrl+C 检查 (在 rollout 之前就检查, 避免再花一整个 rollout)
            if interrupt_handler.should_exit:
                break

            # Rollout 采样阶段
            for _ in range(rollout_steps):
                action, log_prob, value, svo_mu, svo_sigma, pred_trajs, interact_mask = agent.select_action(obs)
                env.set_svo_info(svo_mu, svo_sigma, pred_trajs, interact_mask)

                # [CVaR] 取 cost value (单独前向, 复用 SVO 推断结果). cvar 关闭时返回 0.0.
                cost_value = agent.get_cost_value(
                    obs, svo_mu, svo_sigma, pred_trajs, interact_mask
                )

                next_obs, reward, done, info = env.step(action)

                # [CVaR] step-wise svo 紧迫度: 用 prev step 的 svo 后验
                u_t = _compute_svo_budget_term(
                    svo_mu, svo_sigma, interact_mask,
                    config.ppo, config.svo.prior_sigma,
                )

                agent.buffer.add(
                    obs,
                    action,
                    reward,
                    value,
                    log_prob,
                    done,
                    svo_mu=svo_mu,
                    svo_sigma=svo_sigma,
                    pred_trajs=pred_trajs,
                    interact_mask=interact_mask,
                    cost=float(info.get('cost_step', 0.0)),
                    cost_value=cost_value,
                    episode_id=episode_count,
                    svo_budget_term=u_t,
                )

                if getattr(agent, "use_slt", False):
                    agent.seq_queue.add_step(obs, action)

                episode_reward += reward
                episode_length += 1
                current_timestep += 1
                agent.total_timesteps = current_timestep

                # [Progress] 推进进度条 (静默地, 不破坏 print)
                if pbar is not None:
                    pbar.update(1)

                # [v8] L2 SVO 统计: 记录当前步的 θ
                if l2_stage and 'theta_deg' in info:
                    ep_theta_sum += float(info['theta_deg'])
                    ep_theta_count += 1
                if done:
                    episode_count += 1
                    agent.episode_count = episode_count

                    episode_rewards.append(episode_reward)
                    is_success = 1.0 if info.get("success", False) else 0.0
                    success_history.append(is_success)

                    smooth_success_rate = float(np.mean(success_history)) if success_history else 0.0
                    smooth_avg_reward = float(np.mean(episode_rewards[-100:])) if episode_rewards else 0.0

                    # Episode 级别日志
                    writer.add_scalar("episode/reward", episode_reward, episode_count)
                    writer.add_scalar("episode/length", episode_length, episode_count)
                    writer.add_scalar("episode/speed_kmh", info.get("speed_kmh", 0.0), episode_count)
                    writer.add_scalar("episode/is_collision", 1.0 if info.get("collision", False) else 0.0, episode_count)
                    writer.add_scalar("episode/is_success", 1.0 if info.get("success", False) else 0.0, episode_count)
                    writer.add_scalar("episode/is_timeout", 1.0 if info.get("timeout", False) else 0.0, episode_count)
                    writer.add_scalar(
                        "episode/is_stuck",
                        1.0 if info.get("termination_reason", "") == "stuck" else 0.0,
                        episode_count,
                    )
                    writer.add_scalar("metric/success_rate", smooth_success_rate, episode_count)
                    writer.add_scalar("metric/avg_reward_100ep", smooth_avg_reward, episode_count)

                    if svo_mu is not None:
                        writer.add_scalar("svo/risk_penalty", getattr(env, "_svo_risk_penalty", 0.0), episode_count)

                    if bv_controller is not None:
                        writer.add_scalar("levelk/num_controlled_bvs", bv_controller.num_controlled, episode_count)

                    # ==================================================
                    # [v8] L2 SVO 统计日志 (取代原 MandLC 大量日志)
                    # ==================================================
                    # 终止原因分布
                    term_reason = str(info.get('termination_reason', 'unknown'))
                    termination_reasons.append(term_reason)

                    if l2_stage and ep_theta_count > 0:
                        avg_theta_ep = ep_theta_sum / ep_theta_count
                        writer.add_scalar("L2/avg_theta_deg", avg_theta_ep, episode_count)

                    # 终止原因分布 (近100 episodes)
                    if termination_reasons:
                        from collections import Counter
                        reason_counts = Counter(termination_reasons)
                        n = len(termination_reasons)
                        for r in ['reached_goal', 'collision', 'max_steps',
                                  'cross_two_lanes', 'non_driving_lane']:
                            writer.add_scalar(
                                f"termreason/{r}",
                                reason_counts.get(r, 0) / n,
                                episode_count,
                            )
                        # out_of_lane 是前缀匹配
                        out_count = sum(1 for tr in termination_reasons
                                        if tr.startswith('out_of_lane'))
                        writer.add_scalar(
                            "termreason/out_of_lane",
                            out_count / n, episode_count,
                        )

                    print(
                        f"Episode {episode_count}: reward={episode_reward:.1f}, "
                        f"length={episode_length}, avg_reward={smooth_avg_reward:.1f}, "
                        f"reason={info.get('termination_reason', 'unknown')}"
                    )

                    episode_reward = 0.0
                    episode_length = 0

                    # [v8] 重置 L2 SVO 统计
                    ep_theta_sum = 0.0
                    ep_theta_count = 0

                    obs = env.reset()

                    if hasattr(agent, "reset_svo_state"):
                        agent.reset_svo_state()
                    if bv_controller is not None:
                        bv_controller.reset()
                    if getattr(agent, "use_slt", False):
                        agent.seq_queue.on_episode_end()
                else:
                    obs = next_obs

                if config.visual.enable:
                    env.render()

                if current_timestep >= total_timesteps:
                    break

                # [Progress] Ctrl+C 在 episode 边界优雅退出 (避免半截 episode 进 buffer)
                if interrupt_handler.should_exit and done:
                    break

            # [Progress] 如果 inner loop 因为 Ctrl+C 提前退出, 跳过 PPO update (buffer 还没满)
            if interrupt_handler.should_exit:
                break

            # PPO 参数更新阶段
            if isinstance(agent, PPOAgent):
                update_stats = agent.update(last_state=obs)
            else:
                update_stats = agent.update()

            writer.add_scalar("train/actor_loss", update_stats["actor_loss"], current_timestep)
            writer.add_scalar("train/critic_loss", update_stats["critic_loss"], current_timestep)
            writer.add_scalar("train/entropy", update_stats["entropy"], current_timestep)
            writer.add_scalar("train/learning_rate", update_stats["lr"], current_timestep)

            # [SPO] ratio_deviation 是 SPO 论文 Figure 6 / Table 1 的核心验证指标
            #       仅 SPOCarlaAgent.update() 会写入这些字段, PPO/DAPO 不会
            if "ratio_deviation_mean" in update_stats:
                writer.add_scalar("policy/ratio_deviation_mean",
                                  update_stats["ratio_deviation_mean"], current_timestep)
                writer.add_scalar("policy/ratio_deviation_max",
                                  update_stats["ratio_deviation_max"], current_timestep)
            # [SPO] ε 监测 (仅 --algo spo)
            if "spo_epsilon_mean" in update_stats:
                writer.add_scalar("policy/spo_epsilon_mean",
                                  update_stats["spo_epsilon_mean"], current_timestep)
                writer.add_scalar("policy/spo_epsilon_min",
                                  update_stats["spo_epsilon_min"], current_timestep)
                writer.add_scalar("policy/spo_epsilon_max",
                                  update_stats["spo_epsilon_max"], current_timestep)
            # [SPO] SVO 紧迫度 (仅 --algo spo + 自适应模式)
            if "svo_risk_mean" in update_stats:
                writer.add_scalar("policy/svo_risk_mean",
                                  update_stats["svo_risk_mean"], current_timestep)
                writer.add_scalar("policy/svo_risk_max",
                                  update_stats["svo_risk_max"], current_timestep)

            # [CVaR] 日志: cvar_hat / lambda / budget / episode cost
            if "cvar_hat_norm" in update_stats:
                writer.add_scalar("cvar/cvar_hat_norm",     update_stats["cvar_hat_norm"],     current_timestep)
                writer.add_scalar("cvar/cvar_hat_unnorm",   update_stats["cvar_hat_unnorm"],   current_timestep)
                writer.add_scalar("cvar/lambda",            update_stats["cvar_lambda"],       current_timestep)
                writer.add_scalar("cvar/avg_budget",        update_stats["avg_budget"],        current_timestep)
                writer.add_scalar("cvar/avg_episode_cost",  update_stats["avg_episode_cost"],  current_timestep)
                writer.add_scalar("cvar/avg_svo_budget_term", update_stats["avg_svo_budget_term"], current_timestep)
                writer.add_scalar("cvar/cost_critic_loss",  update_stats.get("cost_critic_loss", 0.0), current_timestep)
                writer.add_scalar("cvar/n_episodes_in_rollout", update_stats.get("n_episodes_in_rollout", 0), current_timestep)

            if "slt_loss" in update_stats:
                writer.add_scalar("slt/loss", update_stats["slt_loss"], current_timestep)
                writer.add_scalar("slt/cosine_sim", update_stats["slt_cosine_sim"], current_timestep)

            elapsed = time.time() - wall_start
            train_steps = max(1, current_timestep - run_start_step)
            fps = train_steps / max(elapsed, 1e-6)
            remaining = (total_timesteps - current_timestep) / max(fps, 1e-6)

            _msg = (
                f"[{current_timestep}/{total_timesteps}] "
                f"actor_loss={update_stats['actor_loss']:.4f}, "
                f"critic_loss={update_stats['critic_loss']:.4f}, "
            )
            if "cvar_hat_norm" in update_stats:
                _msg += (f"cvar_hat={update_stats['cvar_hat_norm']:.3f}, "
                         f"lambda={update_stats['cvar_lambda']:.4f}, "
                         f"d_avg={update_stats['avg_budget']:.3f}, ")
            _msg += f"FPS={fps:.0f}, remaining={remaining / 60:.1f}min"

            # [Progress] tqdm.write 不破坏进度条, 同时更新进度条尾巴的紧凑指标
            if pbar is not None:
                pbar.write(_msg)
                _postfix = {
                    "loss": f"{update_stats['actor_loss']:.3f}",
                    "FPS": f"{fps:.0f}",
                }
                if "cvar_hat_norm" in update_stats:
                    _postfix["cvar"] = f"{update_stats['cvar_hat_norm']:.2f}"
                    _postfix["λ"] = f"{update_stats['cvar_lambda']:.3f}"
                pbar.set_postfix(_postfix, refresh=False)
            else:
                print(_msg)

            # 定期评估
            if current_timestep % config.train.eval_freq_steps < rollout_steps:
                print("\n评估中...")
                eval_results = evaluate(env, agent, config.train.eval_episodes)

                writer.add_scalar("eval/mean_reward", eval_results["mean_reward"], current_timestep)
                writer.add_scalar("eval/std_reward", eval_results["std_reward"], current_timestep)
                writer.add_scalar("eval/mean_length", eval_results["mean_length"], current_timestep)
                writer.add_scalar("eval/mean_speed", eval_results["mean_speed"], current_timestep)
                writer.add_scalar("eval/collision_rate", eval_results["collision_rate"], current_timestep)
                writer.add_scalar("eval/success_rate", eval_results["success_rate_eval"], current_timestep)

                print(
                    f"评估结果: reward={eval_results['mean_reward']:.1f}±{eval_results['std_reward']:.1f}, "
                    f"speed={eval_results['mean_speed']:.1f} km/h, "
                    f"succ={eval_results['success_rate_eval']:.2f}, "
                    f"coll={eval_results['collision_rate']:.2f}"
                )

                if eval_results["mean_reward"] > best_reward:
                    best_reward = eval_results["mean_reward"]
                    best_path = os.path.join(dirs["models"], "best_model.pth")
                    agent.total_timesteps = current_timestep
                    agent.episode_count = episode_count
                    agent.save(best_path)
                    print(f"新最佳模型: reward={best_reward:.1f}")

                # [Fix] 评估会重置同一个环境，评估后需要重新对齐训练观测
                obs = env.reset()
                episode_reward = 0.0
                episode_length = 0
                if hasattr(agent, "reset_svo_state"):
                    agent.reset_svo_state()
                if bv_controller is not None:
                    bv_controller.reset()
                if getattr(agent, "use_slt", False):
                    agent.seq_queue.on_episode_end()
                print()

            # 定期保存 checkpoint
            if current_timestep % config.train.save_freq_steps < rollout_steps:
                ckpt_path = os.path.join(dirs["models"], f"checkpoint_{current_timestep}.pth")
                agent.total_timesteps = current_timestep
                agent.episode_count = episode_count
                agent.save(ckpt_path)

        reached_target_steps = current_timestep >= total_timesteps

    finally:
        # [Progress] 关闭进度条 (在所有 print 之前, 让控制台干净)
        if pbar is not None:
            pbar.close()

        # [Progress] 恢复 SIGINT 默认 handler
        try:
            interrupt_handler.__exit__(None, None, None)
        except Exception:
            pass

        # 区分三种结束情况:
        #   1) 正常训完 (reached_target_steps): 保存 final_model
        #   2) Ctrl+C 中断 (interrupt_handler.should_exit): 保存 interrupted_step_N.pth + 打印 resume 命令
        #   3) 其他异常: 也保存 final_model (向后兼容旧行为)
        is_user_interrupt = interrupt_handler.should_exit and not reached_target_steps
        is_exception = (sys.exc_info()[0] is not None) and not is_user_interrupt

        agent.total_timesteps = current_timestep
        agent.episode_count = episode_count

        if is_user_interrupt:
            # 用户主动 Ctrl+C: 保存专门的 interrupted ckpt
            interrupted_path = os.path.join(
                dirs["models"], f"interrupted_step{current_timestep}.pth"
            )
            agent.save(interrupted_path)
        else:
            # 正常完成或异常: 保存 final_model (旧行为)
            final_path = os.path.join(dirs["models"], "final_model.pth")
            agent.save(final_path)

        total_time = time.time() - wall_start
        print("\n" + "=" * 60)
        if is_user_interrupt:
            print("⚠ 训练已被 Ctrl+C 中断")
        elif is_exception:
            print("训练异常退出")
        else:
            print("训练完成")
        print(f"总用时: {total_time / 3600:.2f} h")
        print(f"总步数: {current_timestep} / {total_timesteps}")
        print(f"总 Episode: {episode_count}")
        print(f"最佳评估奖励: {best_reward:.1f}")
        print(f"模型目录: {dirs['models']}")
        print(f"TensorBoard 目录: {dirs['tensorboard']}")
        print("=" * 60)

        # [Progress] Ctrl+C 中断时打印 resume 命令, 复制粘贴即可恢复训练
        if is_user_interrupt:
            print("\n恢复训练命令:")
            _resume_args = [f"python train.py --stage {os.environ.get('_TRAIN_STAGE', 'l2')}"]
            _resume_args.append(f"--resume {interrupted_path}")
            if config.svo.enabled and os.environ.get("_SVO_PRETRAINED"):
                _resume_args.append(f"--svo_pretrained {os.environ['_SVO_PRETRAINED']}")
            if config.klevel.bv_control_mode == "level1" and config.klevel.bv_policy_path:
                _resume_args.append(f"--level1_path {config.klevel.bv_policy_path}")
            print("    " + " \\\n    ".join(_resume_args))
            print()

        try:
            env.close()
        except Exception as e:
            print(f"[Warning] env.close() 失败: {e}")
        try:
            writer.close()
        except Exception as e:
            print(f"[Warning] writer.close() 失败: {e}")

    return agent, dirs


def main() -> None:
    args = parse_args()
    config = create_config_from_args(args)
    # [Progress] 把 stage 和 svo_pretrained 路径暴露给 train(), 用于打印恢复命令
    os.environ["_TRAIN_STAGE"] = args.stage
    if args.svo_pretrained:
        os.environ["_SVO_PRETRAINED"] = args.svo_pretrained
    train(config, args.resume, args.svo_pretrained)


if __name__ == "__main__":
    main()
