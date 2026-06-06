"""
test.py -- 批量多场景测试框架 (Level-k v9 - 真博弈版)

========================================================================
在完整训练流程中的位置:
  Step 0: python data_collector.py  (收集NPC轨迹数据)
  Step 1: python pretrain_svo.py    (预训练SVO BIRL模型)
  Step 2: python train.py --stage l1 (L1 BV 训练 - 新流程)
  Step 3: python train.py --stage svo_only --svo_pretrained ... (SVO-only消融)
  Step 4: python train.py --stage l2 --svo_pretrained ... --level1_path ...
  验证:   python test.py            ← 本文件 (消融对比测试)
========================================================================

[Level-k v9 重要变化]
  Step 2 现在训练的是 BV 专用策略 (BVTrainEnv + BVRewardConfig).
  这个策略的任务是"在车流中正常行驶 + 不撞", 不是 ego 任务.
  --l1_model 和 --level1_path 现在都指向同一个 BV 策略文件.

消融实验组别:
  1. L1 (BV baseline):        测试 BV 自己当 ego 的表现
                              (它是 BV 策略, 当 ego 用本来就不太好, 仅供对照)
  2. SVO-only (no K-Level):   BV=TM, 有SVO, SVO加权奖励但 BV 仍 TM
  3. Full L2 (SVO+K-Level):   BV=L1 BV 策略, ego obs 拼 a_BV, 真博弈

使用方法:
    # 完整 3 组消融对比
    python test.py --l1_model checkpoints/level1_bv_policy.pt \\
                   --svo_model checkpoints/svo_only_best.pth \\
                   --l2_model checkpoints/level2_best.pth \\
                   --level1_path checkpoints/level1_bv_policy.pt

    # 只测 L2 (推荐, 这是 v9 的主结果)
    python test.py --l2_model checkpoints/level2_best.pth \\
                   --level1_path checkpoints/level1_bv_policy.pt

    # 录制 GIF 动图 (论文/汇报)
    python test.py --l2_model l2.pt --level1_path l1_bv.pt \\
                   --save_gif --gif_episodes 3 --gif_fps 15 --gif_scale 0.5
"""

import os
import sys
import argparse
import time
import random
import json
import math
from pathlib import Path
import numpy as np
import torch
from datetime import datetime
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config, get_default_config
from src.envs.carla_env import CarlaEnv, make_env
from src.algorithms.ppo_model import PPOAgent

# ======================================================================== #
#  场景定义                                                                  #
# ======================================================================== #

SCENARIOS = {
    'dense_mixed': {
        'name': 'Dense Mixed Traffic',
        'desc': '120 NPC, 混合风格, 强制变道',
        'num_npc': 120,
        'force_style': None,
        'enable_obstacles': True,
    },
    'aggressive': {
        'name': 'All Aggressive NPC',
        'desc': '120 NPC, 全部aggressive, 强制变道',
        'num_npc': 120,
        'force_style': 'aggressive',
        'enable_obstacles': True,
    },
    'conservative': {
        'name': 'All Conservative NPC',
        'desc': '120 NPC, 全部conservative, 强制变道',
        'num_npc': 120,
        'force_style': 'conservative',
        'enable_obstacles': True,
    },
    'normal': {
        'name': 'All Normal NPC',
        'desc': '120 NPC, 全部normal, 强制变道',
        'num_npc': 120,
        'force_style': 'normal',
        'enable_obstacles': True,
    },
}


# ======================================================================== #
#  参数解析                                                                  #
# ======================================================================== #

def parse_args():
    parser = argparse.ArgumentParser(description='CARLA 多场景测试 (3组消融对比)')

    # === 3组消融模型路径 ===
    parser.add_argument('--l1_model', type=str, default=None,
                        help='L1 (Pure PPO) 模型路径: BV=TM, 无SVO')
    parser.add_argument('--svo_model', type=str, default=None,
                        help='SVO-only (no K-Level) 模型路径: BV=TM, 有SVO')
    parser.add_argument('--l2_model', type=str, default=r"D:\桌面\train_20260507_201802\models\best_model.pth",
                        help='Full L2 (SVO+K-Level) 模型路径: BV=L1, 有SVO')

    # === L1策略路径 (L2测试必需) ===
    parser.add_argument('--level1_path', type=str, default=r"D:\桌面\训练集\K-level\Level-1\train_20260330_124638\models\best_model.pth",
                        help='L1策略权重路径 (L2测试时用于控制BV)')

    # === SVO预训练权重 (SVO-only和L2测试需要) ===
    parser.add_argument('--svo_pretrained', type=str, default=r"D:\桌面\毕设代码\carla_PPO_highway 3.18改（Claude）-k-level改进 （tits）\pretrain_svo\svo_pretrained.pt",
                        help='SVO BIRL预训练权重路径 (可选, 若模型checkpoint已包含则不需要)')

    # CARLA
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--town', type=str, default='Town04')

    # 测试参数
    parser.add_argument('--num_episodes', type=int, default=20,
                        help='每个场景的测试episodes数')
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)

    # 场景选择
    parser.add_argument('--scenarios', nargs='+',
                        default=['dense_mixed', 'aggressive', 'conservative', 'normal'],
                        choices=list(SCENARIOS.keys()),
                        help='要测试的场景列表')

    # 输出
    parser.add_argument('--output_dir', type=str, default='./test_outputs')
    parser.add_argument('--no_plot', action='store_true',
                        help='不生成可视化图片')
    parser.add_argument('--no_render', action='store_true',
                        help='关闭pygame窗口 (GIF录制仍可用)')

    # GIF录制
    parser.add_argument('--save_gif', action='store_true',
                        help='保存测试过程GIF动图 (每个模型×场景保存前N个episode)')
    parser.add_argument('--gif_episodes', type=int, default=1,
                        help='每个(模型, 场景)组合保存几个episode的GIF (默认1)')
    parser.add_argument('--gif_fps', type=int, default=15,
                        help='GIF帧率 (默认15, 越低文件越小)')
    parser.add_argument('--gif_scale', type=float, default=0.5,
                        help='GIF缩放比例 (默认0.5, 降低分辨率减小文件体积)')

    # [v8] SVO θ 来源: Oracle vs BIRL (训练时默认 oracle; 测试时可切换对比)
    parser.add_argument('--use_oracle_svo', action='store_true', default=None,
                        help='L2 测试用 Oracle SVO (NPC 真实风格标签映射 θ)')
    parser.add_argument('--no_oracle_svo', action='store_true',
                        help='L2 测试用 BIRL 推断 (验证推断模型质量)')

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


# ======================================================================== #
#  模型加载: 根据消融组别配置                                                  #
# ======================================================================== #

def load_models(args):
    """
    根据命令行参数加载各组消融模型.

    返回: OrderedDict {name: {agent, uses_svo, needs_bv_controller}}
    按固定顺序: L1 → SVO-only → Full L2
    """
    from collections import OrderedDict
    models = OrderedDict()

    # --- 组1: L1 (Pure PPO) ---
    if args.l1_model:
        print(f"\n加载 L1 (Pure PPO) 模型: {args.l1_model}")
        l1_config = get_default_config()
        l1_config.svo.enabled = False  # 无SVO
        l1_agent = PPOAgent(l1_config)
        l1_agent.load(args.l1_model)
        models['L1 (Pure PPO)'] = {
            'agent': l1_agent,
            'uses_svo': False,
            'needs_bv_controller': False,
        }

    # --- 组2: SVO-only (no K-Level) ---
    if args.svo_model:
        print(f"\n加载 SVO-only 模型: {args.svo_model}")
        svo_config = get_default_config()
        svo_config.svo.enabled = True  # 有SVO
        svo_agent = PPOAgent(svo_config)
        svo_agent.load(args.svo_model)
        models['SVO-only'] = {
            'agent': svo_agent,
            'uses_svo': True,
            'needs_bv_controller': False,  # BV仍由TM控制
        }

    # --- 组3: Full L2 (SVO+K-Level) ---
    if args.l2_model:
        if args.level1_path is None:
            print("[Error] L2模型测试需要 --level1_path 指定L1策略路径 (用于控制BV)")
            print("        请提供: --level1_path checkpoints/level1_policy.pt")
            sys.exit(1)
        if not os.path.exists(args.level1_path):
            print(f"[Error] L1策略文件不存在: {args.level1_path}")
            sys.exit(1)

        print(f"\n加载 Full L2 (SVO+K-Level) 模型: {args.l2_model}")
        print(f"  BV策略: {args.level1_path}")
        l2_config = get_default_config()
        l2_config.svo.enabled = True
        l2_config.klevel.bv_control_mode = "level1"
        l2_config.klevel.bv_policy_path = args.level1_path
        l2_agent = PPOAgent(l2_config)
        l2_agent.load(args.l2_model)
        models['Full L2 (SVO+K-Level)'] = {
            'agent': l2_agent,
            'uses_svo': True,
            'needs_bv_controller': True,
        }

    if not models:
        print("错误: 至少需要指定 --l1_model, --svo_model 或 --l2_model 之一")
        sys.exit(1)

    return models


def setup_bv_controller(config, device, level1_path):
    """
    创建并加载L1策略的BV控制器.

    L2测试时, 检测范围内的NPC由L1策略控制,
    超出范围的NPC自动交还TM, 与L2训练时完全一致.
    """
    from src.algorithms.klevel import LevelKController

    controller = LevelKController(config, device=device)
    controller.load_policy(level1_path, config)
    return controller


# ======================================================================== #
#  单Episode测试 (记录详细轨迹)                                              #
# ======================================================================== #

def run_episode(env, agent, max_steps, deterministic=True, record_traj=True,
                save_frames=False):
    """
    运行单个episode, 收集完整数据.

    Args:
        save_frames: True时从CARLA摄像头采集每帧图像 (用于GIF生成)

    返回:
        result: dict, 包含指标和轨迹数据 (save_frames时额外含'frames'字段)
    """
    obs = env.reset()
    agent.reset_svo_state()

    done = False
    step = 0
    total_reward = 0.0
    collision = False
    success = False
    termination = 'unknown'

    # 帧采集 (GIF用)
    frames = [] if save_frames else None

    # 轨迹记录
    traj_ego = []
    traj_speeds = []
    traj_steers = []
    traj_throttles = []
    traj_accels = []
    traj_front_dists = []
    traj_svo_mus = []
    traj_svo_risk = []
    traj_npcs = []
    traj_obstacle = []
    ego_spawn_xy = None

    min_front_dist = 100.0
    obstacle_passed = False
    prev_speed = 0.0

    # [v8] L2 SVO 统计
    l2_thetas = []   # 每步 SVO θ (度), 用于计算 episode 平均θ

    while not done and step < max_steps:
        result = agent.select_action(obs, deterministic=deterministic)
        action = result[0]
        svo_mu, svo_sigma, pred_trajs, interact_mask = result[3], result[4], result[5], result[6]

        # 传递SVO信息
        env.set_svo_info(svo_mu, svo_sigma, pred_trajs, interact_mask)

        obs, reward, done, info = env.step(action)
        total_reward += reward
        step += 1

        # 记录数据
        speed = info.get('speed_kmh', 0)
        front_dist = info.get('front_distance', 100)
        min_front_dist = min(min_front_dist, front_dist)

        if record_traj and env.ego_vehicle is not None:
            loc = env.ego_vehicle.get_location()
            traj_ego.append((loc.x, loc.y))
            traj_speeds.append(speed)
            traj_steers.append(float(action[0]))
            traj_throttles.append(float(action[1]))
            acc = speed - prev_speed
            traj_accels.append(acc)
            traj_front_dists.append(front_dist)
            prev_speed = speed

            if ego_spawn_xy is None:
                ego_spawn_xy = (loc.x, loc.y)

            # [Fix] SVO索引对齐: svo_mu 的顺序与 _get_observation 的 selected 一致
            # (障碍车 + NPC 按距离排序后取前N), 不是 npc_vehicles 的原顺序.
            # 这里用与 _get_key_svo_theta 相同的逻辑构建索引.
            det_r = env.config.observation.detection_radius
            selected_ids_and_dist = []
            if env.obstacle_vehicle and env.obstacle_vehicle.is_alive:
                d_obs = env.obstacle_vehicle.get_location().distance(loc)
                selected_ids_and_dist.append((env.obstacle_vehicle.id, d_obs))
            for npc in env.npc_vehicles:
                if not npc.is_alive:
                    continue
                d = npc.get_location().distance(loc)
                if d < det_r:
                    selected_ids_and_dist.append((npc.id, d))
            selected_ids_and_dist.sort(key=lambda x: x[1])
            N_neigh = env.config.encoder.num_neighbours
            selected_ids = [x[0] for x in selected_ids_and_dist[:N_neigh]]

            # 记录NPC位置 + SVO角度 (按正确索引)
            step_npcs = []
            for npc in env.npc_vehicles:
                if not npc.is_alive:
                    continue
                npc_loc = npc.get_location()
                d = npc_loc.distance(loc)
                if d < det_r:
                    if npc.id in selected_ids and svo_mu is not None:
                        idx = selected_ids.index(npc.id)
                        if idx < len(svo_mu):
                            mu_j = float(svo_mu[idx])
                        else:
                            mu_j = 45.0
                    else:
                        mu_j = 45.0
                    step_npcs.append((npc_loc.x, npc_loc.y, mu_j))
            traj_npcs.append(step_npcs)

            # 记录障碍车位置
            if env.obstacle_vehicle and env.obstacle_vehicle.is_alive:
                obs_loc = env.obstacle_vehicle.get_location()
                traj_obstacle.append((obs_loc.x, obs_loc.y))
            elif traj_obstacle:
                traj_obstacle.append(traj_obstacle[-1])
            else:
                traj_obstacle.append((0, 0))

            # SVO信息 (全局均值, 向后兼容现有可视化)
            if svo_mu is not None and interact_mask is not None:
                active = svo_mu[interact_mask] if interact_mask.any() else []
                mean_mu = float(np.mean(active)) if len(active) > 0 else -1
                traj_svo_mus.append(mean_mu)
                traj_svo_risk.append(getattr(env, '_svo_risk_penalty', 0.0))
            else:
                traj_svo_mus.append(-1)
                traj_svo_risk.append(0.0)

        # [v8] L2 SVO θ 累计
        if 'theta_deg' in info:
            l2_thetas.append(float(info['theta_deg']))

        if info.get('collision', False):
            collision = True
        if info.get('success', False):
            success = True
        termination = info.get('termination_reason', 'unknown')

        if env.config.visual.enable:
            env.render()

        # 采集帧 (GIF用): 从CARLA摄像头获取原始RGB图像
        if save_frames and env.camera_image is not None:
            frames.append(env.camera_image.copy())

    # 障碍车是否被绕过
    obstacle_passed = getattr(env, '_obstacle_passed', False)

    # 舒适性指标
    steer_smoothness = float(np.mean(np.abs(np.diff(traj_steers)))) if len(traj_steers) > 1 else 0
    accel_smoothness = float(np.std(traj_accels)) if len(traj_accels) > 1 else 0

    out = {
        'reward': total_reward,
        'length': step,
        'avg_speed': float(np.mean(traj_speeds)) if traj_speeds else 0,
        'max_speed': float(np.max(traj_speeds)) if traj_speeds else 0,
        'collision': collision,
        'success': success,
        'termination': termination,
        'obstacle_passed': obstacle_passed,
        'min_front_dist': min_front_dist,
        'steer_smoothness': steer_smoothness,
        'accel_smoothness': accel_smoothness,
        # [v8] L2 SVO: 本 episode 平均 θ (用于测试时对比不同NPC风格下的行为)
        'avg_theta_deg': float(np.mean(l2_thetas)) if l2_thetas else 45.0,
        # 轨迹数据 (可视化用)
        'traj_ego': traj_ego,
        'traj_speeds': traj_speeds,
        'traj_steers': traj_steers,
        'traj_throttles': traj_throttles,
        'traj_accels': traj_accels,
        'traj_front_dists': traj_front_dists,
        'traj_svo_mus': traj_svo_mus,
        'traj_svo_risk': traj_svo_risk,
        'traj_npcs': traj_npcs,
        'traj_obstacle': traj_obstacle,
        'ego_spawn_xy': ego_spawn_xy,
        'frames': frames,
    }
    return out


# ======================================================================== #
#  GIF录制                                                                   #
# ======================================================================== #

def save_episode_gif(frames, output_path, fps=15, scale=0.5):
    """
    将采集的帧序列保存为GIF动图.

    Args:
        frames: list of (H, W, 3) np.ndarray (RGB uint8)
        output_path: 输出路径 (.gif)
        fps: 帧率 (越低文件越小)
        scale: 缩放比例 (0.5 = 宽高各缩一半, 文件体积减至~1/4)
    """
    if not frames:
        return

    try:
        from PIL import Image
    except ImportError:
        print("  [Warning] PIL不可用, 无法生成GIF (pip install Pillow)")
        return

    pil_frames = []
    for f in frames:
        img = Image.fromarray(f)
        if scale != 1.0:
            new_w = int(img.width * scale)
            new_h = int(img.height * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)
        pil_frames.append(img)

    if not pil_frames:
        return

    # duration = 每帧毫秒数
    duration = max(int(1000 / fps), 20)

    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=0,  # 0 = 无限循环
        optimize=True,
    )


# ======================================================================== #
#  场景测试                                                                  #
# ======================================================================== #

def test_scenario(env, agent, scenario_cfg, num_episodes, max_steps, model_name,
                  save_gif=False, gif_episodes=1, gif_fps=15, gif_scale=0.5,
                  gif_dir=None):
    """
    在指定场景下测试多个episodes.

    scenario_cfg: SCENARIOS字典中的一项
    model_name: 消融组别名称
    save_gif: 是否保存GIF动图
    gif_episodes: 保存前几个episode的GIF
    gif_dir: GIF输出目录
    """
    # 配置场景参数
    env.config.traffic.num_npc_vehicles = scenario_cfg['num_npc']
    env.config.scenario.enable_obstacles = scenario_cfg['enable_obstacles']

    # 强制NPC风格
    force_style = scenario_cfg.get('force_style', None)
    env._force_npc_style = force_style

    print(f"\n{'='*60}")
    print(f"场景: {scenario_cfg['name']} | 模型: {model_name}")
    print(f"描述: {scenario_cfg['desc']}")
    print(f"Episodes: {num_episodes}")
    if save_gif:
        print(f"GIF录制: 前{gif_episodes}个episode, {gif_fps}fps, 缩放{gif_scale}")
    print(f"{'='*60}")

    all_results = []
    for ep in range(num_episodes):
        # 前gif_episodes个episode采集帧
        need_frames = save_gif and ep < gif_episodes
        result = run_episode(env, agent, max_steps, deterministic=True,
                             save_frames=need_frames)

        # 保存GIF
        if need_frames and result['frames'] and gif_dir:
            gif_name = (f"{model_name}_{scenario_cfg['name']}_ep{ep+1}"
                        .replace(' ', '_').replace('(', '').replace(')', ''))
            gif_path = os.path.join(gif_dir, f"{gif_name}.gif")
            save_episode_gif(result['frames'], gif_path, gif_fps, gif_scale)
            status_tag = "碰撞" if result['collision'] else (
                "成功" if result['success'] else "其他")
            print(f"    GIF已保存: {gif_path} "
                  f"({len(result['frames'])}帧, {status_tag})")

        # 释放帧数据 (可能很大)
        result['frames'] = None

        all_results.append(result)

        status = "碰撞" if result['collision'] else (
            "成功" if result['success'] else result['termination'])
        lane_change = "✓" if result['obstacle_passed'] else "✗"
        # [v8] 追加 episode 平均 θ (用于验证 SVO 差异化)
        theta_str = f", avg_θ={result.get('avg_theta_deg', 45.0):.1f}°"
        print(f"  Ep {ep+1:3d}: reward={result['reward']:7.1f}, "
              f"speed={result['avg_speed']:5.1f}km/h, "
              f"变道={lane_change}, status={status}{theta_str}")

    # 汇总
    summary = compute_summary(all_results)
    summary['scenario'] = scenario_cfg['name']
    summary['model'] = model_name

    print(f"\n  --- 汇总 ---")
    print(f"  碰撞率: {summary['collision_rate']:.1%}")
    print(f"  成功率: {summary['success_rate']:.1%}")
    print(f"  变道率: {summary['lane_change_rate']:.1%}")
    print(f"  平均速度: {summary['avg_speed']:.1f} ± {summary['std_speed']:.1f} km/h")
    print(f"  平均奖励: {summary['avg_reward']:.1f} ± {summary['std_reward']:.1f}")
    print(f"  安全距离: {summary['avg_min_front_dist']:.1f} m")

    # [v8] L2 SVO 差异化 — 显示平均 θ (场景风格 vs 推断 θ 对照)
    if 'avg_theta_deg' in summary:
        force_style = scenario_cfg.get('force_style', 'mixed')
        print(f"  平均SVO θ: {summary['avg_theta_deg']:.1f}°"
              f"  (场景风格={force_style})")

    return all_results, summary


def compute_summary(results):
    """计算汇总指标"""
    n = len(results)
    summary = {
        'n_episodes': n,
        'collision_rate': sum(r['collision'] for r in results) / n,
        'success_rate': sum(r['success'] for r in results) / n,
        'lane_change_rate': sum(r['obstacle_passed'] for r in results) / n,
        'avg_reward': float(np.mean([r['reward'] for r in results])),
        'std_reward': float(np.std([r['reward'] for r in results])),
        'avg_speed': float(np.mean([r['avg_speed'] for r in results])),
        'std_speed': float(np.std([r['avg_speed'] for r in results])),
        'avg_max_speed': float(np.mean([r['max_speed'] for r in results])),
        'avg_length': float(np.mean([r['length'] for r in results])),
        'avg_min_front_dist': float(np.mean([r['min_front_dist'] for r in results])),
        'avg_steer_smooth': float(np.mean([r['steer_smoothness'] for r in results])),
        'avg_accel_smooth': float(np.mean([r['accel_smoothness'] for r in results])),
    }
    # [v8] L2 SVO: 如果 result 有 avg_theta_deg, 汇总平均
    if results and 'avg_theta_deg' in results[0]:
        summary['avg_theta_deg'] = float(np.mean(
            [r.get('avg_theta_deg', 45.0) for r in results]))
    return summary


# ======================================================================== #
#  可视化: 轨迹图 + 控制量时序图                                              #
# ======================================================================== #

def plot_trajectory_comparison(results_dict, scenario_name, output_dir):
    """
    轨迹对比图: 最多3组消融, 转换到道路坐标系

    results_dict: {model_name: results_list} — 最多3组
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        try:
            matplotlib.rcParams['font.family'] = 'Times New Roman'
        except Exception:
            pass
        matplotlib.rcParams['font.size'] = 12
    except Exception:
        print("  [Warning] matplotlib不可用, 跳过轨迹图")
        return

    def to_road_frame(traj_xy):
        """全局XY → 道路坐标 (纵向/横向)"""
        pts = np.array(traj_xy)
        if len(pts) < 3:
            return pts[:, 0], pts[:, 1], 0, 0, 0
        n_dir = min(10, len(pts) - 1)
        dx = pts[n_dir, 0] - pts[0, 0]
        dy = pts[n_dir, 1] - pts[0, 1]
        road_angle = np.arctan2(dy, dx)
        cos_a = np.cos(-road_angle)
        sin_a = np.sin(-road_angle)
        ox, oy = pts[0, 0], pts[0, 1]
        rx = (pts[:, 0] - ox) * cos_a - (pts[:, 1] - oy) * sin_a
        ry = (pts[:, 0] - ox) * sin_a + (pts[:, 1] - oy) * cos_a
        return rx, ry, road_angle, ox, oy

    def transform_point(x, y, road_angle, ox, oy):
        cos_a = np.cos(-road_angle)
        sin_a = np.sin(-road_angle)
        rx = (x - ox) * cos_a - (y - oy) * sin_a
        ry = (x - ox) * sin_a + (y - oy) * cos_a
        return rx, ry

    n_models = len(results_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 6))
    if n_models == 1:
        axes = [axes]

    colors_map = {
        'L1 (Pure PPO)': '#f57c00',
        'SVO-only': '#4caf50',
        'Full L2 (SVO+K-Level)': '#1976d2',
    }

    for ax, (model_name, results) in zip(axes, results_dict.items()):
        if results is None or len(results) == 0:
            ax.set_title(f'{model_name} (N/A)')
            ax.set_xlabel('Longitudinal (m)')
            ax.set_ylabel('Lateral (m)')
            continue

        ref_result = results[0]
        if len(ref_result['traj_ego']) < 3:
            continue
        _, _, road_angle, ox, oy = to_road_frame(ref_result['traj_ego'])

        color = colors_map.get(model_name, 'gray')

        # 画ego轨迹
        for i, r in enumerate(results[:10]):
            traj = r['traj_ego']
            if len(traj) < 2:
                continue
            pts = np.array(traj)
            rxs, rys = [], []
            for px, py in pts:
                rx, ry = transform_point(px, py, road_angle, ox, oy)
                rxs.append(rx)
                rys.append(ry)
            alpha = 0.8 if i == 0 else 0.3
            lw = 2.0 if i == 0 else 0.8
            label = model_name if i == 0 else None
            ax.plot(rxs, rys, color=color, alpha=alpha, linewidth=lw, label=label)

        # 画障碍车位置
        if ref_result['traj_obstacle']:
            obs_xy = ref_result['traj_obstacle'][0]
            if obs_xy != (0, 0):
                orx, ory = transform_point(obs_xy[0], obs_xy[1], road_angle, ox, oy)
                ax.scatter(orx, ory, marker='X', s=200, c='red', zorder=10, label='Obstacle')

        # 画车道线 (近似)
        lane_w = 3.5
        for lane_i in range(-2, 3):
            ax.axhline(y=lane_i * lane_w, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)

        ax.set_xlabel('Longitudinal (m)')
        ax.set_ylabel('Lateral (m)')
        ax.set_title(model_name)
        ax.legend(fontsize=9, loc='upper left')
        ax.set_ylim(-14, 14)
        ax.grid(True, alpha=0.15)

    fig.suptitle(f'Trajectory Comparison — {scenario_name}', fontsize=14)
    plt.tight_layout()
    safe_name = scenario_name.replace(' ', '_')
    path = os.path.join(output_dir, f'trajectory_{safe_name}.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  轨迹对比图: {path}")


def plot_control_timeseries(results, scenario_name, model_name, output_dir):
    """控制量时序图 (取第一个episode)"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        try:
            matplotlib.rcParams['font.family'] = 'Times New Roman'
        except Exception:
            pass
        matplotlib.rcParams['font.size'] = 11
    except Exception:
        return

    if not results:
        return
    r = results[0]

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    steps = range(len(r['traj_speeds']))

    # 速度
    axes[0].plot(steps, r['traj_speeds'], color='#1976d2', linewidth=1)
    axes[0].axhline(y=60, color='green', linestyle='--', alpha=0.5, label='Target')
    axes[0].set_ylabel('Speed (km/h)')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.2)

    # 转向
    axes[1].plot(steps, r['traj_steers'], color='#d32f2f', linewidth=1)
    axes[1].set_ylabel('Steering')
    axes[1].grid(True, alpha=0.2)

    # 前车距离
    axes[2].plot(steps, r['traj_front_dists'], color='#388e3c', linewidth=1)
    axes[2].axhline(y=8, color='red', linestyle='--', alpha=0.5, label='Min Safe')
    axes[2].set_ylabel('Front Dist (m)')
    axes[2].set_ylim(0, min(100, max(r['traj_front_dists']) + 5))
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.2)

    # SVO
    axes[3].plot(steps, r['traj_svo_mus'], color='#7b1fa2', linewidth=1)
    axes[3].set_ylabel('SVO μ (°)')
    axes[3].set_xlabel('Step')
    axes[3].set_ylim(-5, 95)
    axes[3].grid(True, alpha=0.2)

    fig.suptitle(f'{model_name} — {scenario_name}', fontsize=13)
    plt.tight_layout()
    safe_name = f"control_{model_name.replace(' ', '_').replace('(', '').replace(')', '')}_{scenario_name.replace(' ', '_')}"
    path = os.path.join(output_dir, f'{safe_name}.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  控制量图: {path}")


def plot_ablation_comparison(all_summaries, output_dir):
    """3组消融对比柱状图: 各场景各指标"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        try:
            matplotlib.rcParams['font.family'] = 'Times New Roman'
        except Exception:
            pass
        matplotlib.rcParams['font.size'] = 12
    except Exception:
        return

    order = ['L1 (Pure PPO)', 'SVO-only', 'Full L2 (SVO+K-Level)']
    models = sorted(set(s['model'] for s in all_summaries),
                    key=lambda x: order.index(x) if x in order else 99)
    scenarios = sorted(set(s['scenario'] for s in all_summaries))

    metrics = [
        ('collision_rate', 'Collision Rate ↓', True),
        ('success_rate', 'Success Rate ↑', False),
        ('avg_speed', 'Avg Speed (km/h)', False),
        ('lane_change_rate', 'Lane Change Rate ↑', False),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    x = np.arange(len(scenarios))
    n_models = len(models)
    width = 0.8 / max(n_models, 1)
    colors = {
        'L1 (Pure PPO)': '#f57c00',
        'SVO-only': '#4caf50',
        'Full L2 (SVO+K-Level)': '#1976d2',
    }

    for ax, (metric_key, metric_name, lower_better) in zip(axes, metrics):
        for i, model in enumerate(models):
            vals = []
            for sc in scenarios:
                match = [s for s in all_summaries
                         if s['model'] == model and s['scenario'] == sc]
                vals.append(match[0][metric_key] if match else 0)
            offset = (i - n_models / 2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=model,
                   color=colors.get(model, 'gray'), alpha=0.85)

        ax.set_ylabel(metric_name)
        ax.set_xticks(x)
        sc_short = [s.replace('All ', '').replace(' NPC', '').replace('Dense ', '')
                     for s in scenarios]
        ax.set_xticklabels(sc_short, fontsize=9, rotation=15)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2, axis='y')

    fig.suptitle('Ablation Comparison Across Scenarios', fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, 'ablation_comparison.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  消融对比图: {path}")


# ======================================================================== #
#  结果输出: LaTeX + JSON                                                   #
# ======================================================================== #

def generate_latex_table(all_summaries):
    """生成LaTeX表格代码 (支持3组消融)"""
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Ablation comparison across driving scenarios}")
    lines.append(r"\label{tab:ablation}")
    lines.append(r"\begin{tabular}{llcccccc}")
    lines.append(r"\toprule")
    lines.append(r"Scenario & Model & Collision$\downarrow$ & Success$\uparrow$ & "
                 r"Lane Change$\uparrow$ & Speed & Min Dist & Steer Smooth$\downarrow$ \\")
    lines.append(r"\midrule")

    scenarios = sorted(set(s['scenario'] for s in all_summaries))
    order = ['L1 (Pure PPO)', 'SVO-only', 'Full L2 (SVO+K-Level)']
    for sc in scenarios:
        rows = [s for s in all_summaries if s['scenario'] == sc]
        rows.sort(key=lambda r: order.index(r['model']) if r['model'] in order else 99)
        for i, s in enumerate(rows):
            sc_col = sc if i == 0 else ""
            lines.append(
                f"  {sc_col} & {s['model']} & "
                f"{s['collision_rate']:.1%} & {s['success_rate']:.1%} & "
                f"{s['lane_change_rate']:.1%} & "
                f"{s['avg_speed']:.1f} & {s['avg_min_front_dist']:.1f} & "
                f"{s['avg_steer_smooth']:.4f} \\\\"
            )
        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ======================================================================== #
#  主函数                                                                    #
# ======================================================================== #

def main():
    args = parse_args()
    set_seed(args.seed)

    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"test_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    fig_dir = os.path.join(output_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    # ============================================================
    #  加载模型
    # ============================================================
    models = load_models(args)

    # ============================================================
    #  创建环境 (所有模型共用同一个env实例, 通过配置切换)
    # ============================================================
    config = get_default_config()
    config.carla.host = args.host
    config.carla.port = args.port
    config.carla.town = args.town
    config.test.max_episode_steps = args.max_steps
    config.visual.enable = not args.no_render

    # [v8] 测试时应用 SVO Oracle/BIRL 开关
    if args.no_oracle_svo:
        config.reward.use_oracle_svo = False
        print("[SVO θ] 测试使用 BIRL 推断模式 (--no_oracle_svo)")
    elif args.use_oracle_svo:
        config.reward.use_oracle_svo = True
        print("[SVO θ] 测试使用 Oracle 模式 (--use_oracle_svo)")
    else:
        # 默认沿用训练时的 config 默认值 (True, Oracle)
        print(f"[SVO θ] 测试使用默认模式 (use_oracle_svo={config.reward.use_oracle_svo})")

    # --save_gif 需要CARLA摄像头, 即使不开pygame窗口也要启用visual
    if args.save_gif and not config.visual.enable:
        config.visual.enable = True
        print("[GIF] 自动启用摄像头 (--save_gif 需要CARLA camera sensor)")

    print("\n创建CARLA环境...")
    env = make_env(config, args.seed)

    # GIF输出目录
    gif_dir = None
    if args.save_gif:
        gif_dir = os.path.join(output_dir, 'gifs')
        os.makedirs(gif_dir, exist_ok=True)

    # ============================================================
    #  准备BV控制器 (L2测试用)
    # ============================================================
    bv_controller = None
    if args.level1_path and os.path.exists(args.level1_path):
        l2_info = models.get('Full L2 (SVO+K-Level)', None)
        if l2_info:
            bv_controller = setup_bv_controller(
                l2_info['agent'].config,
                l2_info['agent'].device,
                args.level1_path,
            )

    # ============================================================
    #  批量测试
    # ============================================================
    all_summaries = []
    all_detailed = {}  # {(scenario_name, model_name): results}

    for scenario_key in args.scenarios:
        scenario_cfg = SCENARIOS[scenario_key]

        for model_name, model_info in models.items():
            agent = model_info['agent']
            needs_bv = model_info['needs_bv_controller']

            set_seed(args.seed)  # 每个组合重置种子, 保证公平对比

            # --- 切换BV控制模式 ---
            if needs_bv and bv_controller is not None:
                env.set_bv_control(bv_controller)
                print(f"  [Level-k] BV策略控制已启用 (控制半径={bv_controller.control_radius}m)")
            else:
                # 确保BV由TM控制 (L1和SVO-only测试)
                env.set_bv_control(None)

            results, summary = test_scenario(
                env, agent, scenario_cfg,
                args.num_episodes, args.max_steps, model_name,
                save_gif=args.save_gif, gif_episodes=args.gif_episodes,
                gif_fps=args.gif_fps, gif_scale=args.gif_scale,
                gif_dir=gif_dir,
            )
            all_summaries.append(summary)
            all_detailed[(scenario_cfg['name'], model_name)] = results

            # --- 重置BV控制器状态 ---
            if bv_controller is not None:
                bv_controller.reset()

    # ============================================================
    #  输出结果
    # ============================================================
    print("\n" + "=" * 70)
    print("测试完成! 生成结果...")
    print("=" * 70)

    # 1. JSON
    json_path = os.path.join(output_dir, 'results.json')
    json_data = {
        'timestamp': timestamp,
        'args': vars(args),
        'summaries': all_summaries,
    }
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2, default=str)
    print(f"\nJSON: {json_path}")

    # 2. LaTeX
    latex_code = generate_latex_table(all_summaries)
    latex_path = os.path.join(output_dir, 'table.tex')
    with open(latex_path, 'w') as f:
        f.write(latex_code)
    print(f"LaTeX: {latex_path}")
    print("\n--- LaTeX代码 ---")
    print(latex_code)

    # 3. 终端汇总表
    print("\n--- 汇总表 ---")
    header = f"{'Scenario':<25} {'Model':<25} {'Coll':>6} {'Succ':>6} {'LnChg':>6} {'Speed':>8} {'MinD':>6}"
    print(header)
    print("-" * len(header))
    for s in all_summaries:
        print(f"{s['scenario']:<25} {s['model']:<25} "
              f"{s['collision_rate']:>5.1%} {s['success_rate']:>5.1%} "
              f"{s['lane_change_rate']:>5.1%} "
              f"{s['avg_speed']:>7.1f} {s['avg_min_front_dist']:>5.1f}")

    # 4. 可视化
    if not args.no_plot:
        print("\n生成可视化图表...")

        # 轨迹对比图 (每个场景, 所有模型并排)
        for scenario_key in args.scenarios:
            sc_name = SCENARIOS[scenario_key]['name']
            sc_results = {}
            for model_name in models.keys():
                res = all_detailed.get((sc_name, model_name), None)
                if res:
                    sc_results[model_name] = res
            if sc_results:
                plot_trajectory_comparison(sc_results, sc_name, fig_dir)

        # 控制量时序图 (每个模型×每个场景)
        for (sc_name, model_name), results in all_detailed.items():
            plot_control_timeseries(results, sc_name, model_name, fig_dir)

        # 消融对比柱状图
        if len(models) > 1:
            plot_ablation_comparison(all_summaries, fig_dir)

    print(f"\n所有结果保存在: {output_dir}")

    # 清理
    env.close()


if __name__ == "__main__":
    main()
