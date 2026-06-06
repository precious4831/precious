"""
data_collector.py -- Stage 1 SVO预训练数据收集

在CARLA中运行episodes收集 (ego_past, npc_past, npc_future) 轨迹三元组.

与训练环境的关键区别:
  1. 无障碍车 — 只有ego + NPC自然行驶
  2. ego也由Traffic Manager控制 (autopilot) — 正常驾驶, 不是随机动作
  3. 不走env.step() — 避免_apply_action覆盖autopilot
  4. 无碰撞终止 — 每个episode跑满max_steps, 最大化数据量
  5. NPC行为多样化 — TM随机分配激进/正常/保守风格

数据格式:
    每条样本: ego_past (T_past, 5), npc_past (T_past, 5), npc_future (T_future, 5)
    5维 = [x, y, psi, vx, vy]

使用方法:
    python data_collector.py --episodes 200 --output svo_dataset.npz
    python data_collector.py --episodes 500 --max_steps 800 --num_npc 30
"""

from pathlib import Path
import sys
import os
import argparse
import time
import random
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config, get_default_config
from src.envs.carla_env import CarlaEnv, make_env


def parse_args():
    parser = argparse.ArgumentParser(description='SVO预训练数据收集')

    parser.add_argument('--episodes', type=int, default=300,
                        help='收集的episode数')
    parser.add_argument('--max_steps', type=int, default=500,
                        help='每个episode步数 (全部跑满, 不提前终止)')
    parser.add_argument('--output', type=str, default='svo_dataset.npz',
                        help='输出文件路径')

    # CARLA
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--town', type=str, default='Town04')
    parser.add_argument('--num_npc', type=int, default=200)
    parser.add_argument('--seed', type=int, default=41, help='随机种子 (影响NPC生成和行为)')

    # 轨迹窗口参数
    parser.add_argument('--past_steps', type=int, default=10,
                        help='历史窗口长度 (= config.encoder.history_steps)')
    parser.add_argument('--future_steps', type=int, default=10,
                        help='预测窗口长度 (= config.svo.prediction_horizon)')
    parser.add_argument('--stride', type=int, default=5,
                        help='滑动窗口步长 (越小样本越多)')
    parser.add_argument('--min_traj_length', type=int, default=25,
                        help='NPC最小观测长度 (past + future + 余量)')

    return parser.parse_args()


def extract_samples(pairs, past_steps, future_steps, stride):
    """
    从episode轨迹对中用滑动窗口切割 (past, future) 样本.

    pairs: list of {'ego': (T, 5), 'npc': (T, 5)}
    返回: list of (ego_past, npc_past, npc_future, style) tuples
    """
    samples = []
    window = past_steps + future_steps

    for pair in pairs:
        ego_traj = pair['ego']
        npc_traj = pair['npc']
        style = pair.get('style', 'unknown')
        T = min(len(ego_traj), len(npc_traj))

        if T < window:
            continue

        for start in range(0, T - window + 1, stride):
            t_split = start + past_steps

            ego_past = ego_traj[start:t_split]
            npc_past = npc_traj[start:t_split]
            npc_future = npc_traj[t_split:t_split + future_steps]

            # 跳过静止NPC (均速 < 0.5 m/s)
            npc_speed = np.sqrt(npc_past[:, 3]**2 + npc_past[:, 4]**2)
            if npc_speed.mean() < 0.5:
                continue

            samples.append((
                ego_past.astype(np.float32),
                npc_past.astype(np.float32),
                npc_future.astype(np.float32),
                style,
            ))

    return samples


def run_collection_episode(env, max_steps):
    """
    运行一个数据收集episode.

    与训练不同:
      1. env.reset() 生成场景 (无障碍车)
      2. ego设为autopilot (TM控制)
      3. 循环中只 world.tick() + _get_observation() (不调env.step())
      4. 不检查碰撞/出界, 跑满max_steps
    """
    # 1. reset环境 (生成ego + NPC, 无障碍车)
    env.reset()

    # 2. ego设为autopilot — TM控制, 正常驾驶
    #    必须在reset之后设, 因为reset会重新生成ego
    tm_port = env.traffic_manager.get_port()
    env.ego_vehicle.set_autopilot(True, tm_port)

    # 给ego一个合理的TM配置 (正常偏保守, 减少碰撞)
    tm = env.traffic_manager
    tm.vehicle_percentage_speed_difference(env.ego_vehicle, 10.0)  # 略低于限速
    tm.distance_to_leading_vehicle(env.ego_vehicle, 5.0)
    tm.auto_lane_change(env.ego_vehicle, True)

    # 3. 收集循环: 只tick + 记录轨迹, 不控制ego
    for step in range(max_steps):
        env.world.tick()

        # BehaviorAgent模式: 每步驱动NPC决策
        env.tick_npc_agents()

        # 轻量级记录: 只查ego/NPC位置, 不构建完整观测
        env.record_step()

        # 检查ego是否还活着 (极罕见情况)
        if env.ego_vehicle is None or not env.ego_vehicle.is_alive:
            print(f"    [Warning] ego在step {step}被销毁, 提前结束")
            break

    # 4. 提取轨迹对
    n_ego = len(env._episode_ego_states)
    n_npc_recorded = len(env._episode_npc_states)
    pairs = env.get_episode_trajectories(min_length=20)
    return pairs, n_ego, n_npc_recorded


def collect_data(config, args):
    """主数据收集循环."""
    print("=" * 60)
    print("SVO预训练数据收集")
    print("=" * 60)
    print(f"Episodes: {args.episodes}")
    print(f"每episode步数: {args.max_steps} (全部跑满)")
    print(f"Past/Future窗口: {args.past_steps}/{args.future_steps} steps")
    print(f"滑动步长: {args.stride}")
    print(f"NPC数量: {args.num_npc} (自动多样化)")
    print(f"Ego: autopilot (TM控制)")
    print(f"障碍车: 禁用")
    print(f"输出: {args.output}")
    print()

    # 创建环境
    env = make_env(config, args.seed)
    env.enable_trajectory_recording = True  # 开启全量轨迹记录
    env.use_behavior_agents = True          # 用BehaviorAgent精确控制NPC风格

    all_samples = []
    total_pairs = 0
    start_time = time.time()

    for ep in range(args.episodes):
        # 运行一个收集episode
        pairs, n_ego, n_npc_recorded = run_collection_episode(env, args.max_steps)
        total_pairs += len(pairs)

        # 滑动窗口切割为训练样本
        ep_samples = extract_samples(
            pairs, args.past_steps, args.future_steps, args.stride
        )
        all_samples.extend(ep_samples)

        # 进度 (含诊断: ego记录步数, NPC检测数)
        elapsed = time.time() - start_time
        eps = (ep + 1) / elapsed if elapsed > 0 else 0
        print(f"Episode {ep+1:4d}/{args.episodes}: "
              f"ego_steps={n_ego:4d}, npc_detected={n_npc_recorded:3d}, "
              f"pairs={len(pairs):3d}, samples={len(ep_samples):4d}, "
              f"total={len(all_samples):6d}, speed={eps:.2f} ep/s")

    # 保存数据集
    if len(all_samples) == 0:
        print("\n" + "=" * 60)
        print("[Error] 未收集到任何有效样本!")
        print(f"  总轨迹对: {total_pairs}")
        print(f"  可能原因:")
        print(f"    - NPC太少 (当前: {config.traffic.num_npc_vehicles})")
        print(f"    - max_steps太短 (当前: {args.max_steps})")
        print(f"    - 检测范围太小 → config.observation.detection_radius")
        print(f"    - min_traj_length太大 (当前: {args.min_traj_length})")
        print("=" * 60)
        return

    ego_past = np.stack([s[0] for s in all_samples])
    npc_past = np.stack([s[1] for s in all_samples])
    npc_future = np.stack([s[2] for s in all_samples])
    styles = np.array([s[3] for s in all_samples])   # ['aggressive', 'normal', ...]

    np.savez_compressed(
        args.output,
        ego_past=ego_past,
        npc_past=npc_past,
        npc_future=npc_future,
        styles=styles,
    )

    # 风格分布统计
    from collections import Counter
    style_counts = Counter(styles)

    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print("数据收集完成!")
    print(f"总Episodes: {args.episodes}")
    print(f"总轨迹对: {total_pairs}")
    print(f"总训练样本: {len(all_samples)}")
    print(f"  ego_past:   {ego_past.shape}")
    print(f"  npc_past:   {npc_past.shape}")
    print(f"  npc_future: {npc_future.shape}")
    print(f"  风格分布: {dict(style_counts)}")
    print(f"文件大小: {os.path.getsize(args.output) / 1024 / 1024:.1f} MB")
    print(f"用时: {elapsed / 60:.1f} 分钟")
    print(f"保存至: {args.output}")
    print("=" * 60)


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # 输出路径转为绝对路径
    args.output = os.path.abspath(args.output)

    # 创建配置
    config = get_default_config()
    config.carla.host = args.host
    config.carla.port = args.port
    config.carla.town = args.town
    config.traffic.num_npc_vehicles = args.num_npc

    # === 数据收集专用配置 ===
    config.scenario.enable_obstacles = False  # 关闭障碍车
    config.visual.enable = False              # 关闭可视化
    config.train.max_episode_steps = args.max_steps + 100  # 放宽, 不让env提前终止

    collect_data(config, args)


if __name__ == "__main__":
    main()
