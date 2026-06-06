"""
Simple L1 evaluation script.

Usage:
  python test_l1_simple.py --model outputs/.../best_model.pth --episodes 20 --render
  python test_l1_simple.py --model outputs/.../best_model.pth --episodes 10 --save_gif
"""

from pathlib import Path
import sys
import argparse
import json
import os
import random
from collections import Counter
from datetime import datetime

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_default_config, get_highway_config, get_debug_config
from src.envs.carla_env import make_env
from src.algorithms.ppo_model import PPOAgent


def parse_args():
    parser = argparse.ArgumentParser(description="Simple L1 policy evaluation")
    parser.add_argument("--model", type=str, required=True, help="Path to L1 checkpoint (.pth)")
    parser.add_argument("--config", type=str, default="default", choices=["default", "highway", "debug"])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic policy instead of deterministic")
    parser.add_argument("--render", action="store_true", help="Enable pygame render")
    parser.add_argument("--save_gif", action="store_true", help="Save episode GIF(s)")
    parser.add_argument("--gif_episodes", type=int, default=1, help="How many first episodes to record")
    parser.add_argument("--gif_fps", type=int, default=15, help="GIF FPS")
    parser.add_argument("--gif_scale", type=float, default=0.5, help="GIF resize scale")
    parser.add_argument("--gif_stride", type=int, default=2, help="Capture every N steps for GIF")
    parser.add_argument("--gif_dir", type=str, default="", help="Optional gif output directory")

    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", type=str, default="Town04")
    parser.add_argument("--num_npc", type=int, default=20)
    parser.add_argument("--target_speed", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--save_json", type=str, default="", help="Optional output json path")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def build_config(args):
    if args.config == "highway":
        cfg = get_highway_config()
    elif args.config == "debug":
        cfg = get_debug_config()
    else:
        cfg = get_default_config()

    cfg.carla.host = args.host
    cfg.carla.port = args.port
    cfg.carla.town = args.town
    cfg.traffic.num_npc_vehicles = args.num_npc
    cfg.reward.target_speed = args.target_speed
    cfg.train.max_episode_steps = args.max_steps
    # Camera sensor is needed for GIF recording even when pygame window is disabled.
    cfg.visual.enable = bool(args.render or args.save_gif)

    # Force pure L1 evaluation setting.
    cfg.svo.enabled = False
    cfg.klevel.bv_control_mode = "tm"
    cfg.klevel.reward_mode = "l1"
    return cfg


def save_episode_gif(frames, output_path, fps=15, scale=0.5):
    if not frames:
        return False, "no_frames"

    try:
        from PIL import Image
    except ImportError:
        return False, "Pillow not installed (pip install Pillow)"

    pil_frames = []
    for frame in frames:
        img = Image.fromarray(frame)
        if scale != 1.0:
            new_w = max(2, int(img.width * scale))
            new_h = max(2, int(img.height * scale))
            img = img.resize((new_w, new_h), Image.LANCZOS)
        pil_frames.append(img)

    duration = max(int(1000 / max(fps, 1)), 20)
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=0,
        optimize=True,
    )
    return True, ""


def run_eval(args):
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")

    set_seed(args.seed)
    cfg = build_config(args)
    env = make_env(cfg, seed=args.seed)
    agent = PPOAgent(cfg)
    agent.load(args.model)

    results = []
    reason_counter = Counter()
    gif_paths = []
    gif_dir = ""
    if args.save_gif:
        gif_dir = args.gif_dir.strip() if args.gif_dir else os.path.join(os.getcwd(), "l1_gifs")
        os.makedirs(gif_dir, exist_ok=True)
        print(f"GIF recording enabled: first {args.gif_episodes} episodes -> {gif_dir}")

    try:
        for ep in range(1, args.episodes + 1):
            obs = env.reset()
            done = False
            ep_reward = 0.0
            ep_len = 0
            speeds = []
            frames = []
            capture_frames = args.save_gif and ep <= args.gif_episodes

            last_info = {}
            while not done and ep_len < args.max_steps:
                out = agent.select_action(obs, deterministic=not args.stochastic)
                action = out[0]
                obs, reward, done, info = env.step(action)

                ep_reward += float(reward)
                ep_len += 1
                speeds.append(float(info.get("speed_kmh", 0.0)))
                last_info = info

                if args.render:
                    env.render()
                if capture_frames and env.camera_image is not None and (ep_len % max(args.gif_stride, 1) == 0):
                    frames.append(env.camera_image.copy())

            reason = last_info.get("termination_reason", "unknown")
            success = bool(last_info.get("success", False))
            collision = bool(last_info.get("collision", False))
            mean_speed = float(np.mean(speeds)) if speeds else 0.0

            reason_counter[reason] += 1
            row = {
                "episode": ep,
                "reward": ep_reward,
                "length": ep_len,
                "mean_speed_kmh": mean_speed,
                "success": success,
                "collision": collision,
                "termination_reason": reason,
            }
            results.append(row)

            print(
                f"Ep {ep:03d} | reward={ep_reward:8.1f} | len={ep_len:4d} | "
                f"speed={mean_speed:5.1f} | success={int(success)} | "
                f"collision={int(collision)} | reason={reason}"
            )

            if capture_frames:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                gif_name = f"l1_ep{ep:03d}_{reason}_{ts}.gif"
                gif_path = os.path.join(gif_dir, gif_name)
                ok, msg = save_episode_gif(frames, gif_path, fps=args.gif_fps, scale=args.gif_scale)
                if ok:
                    gif_paths.append(gif_path)
                    print(f"  GIF saved: {gif_path} (frames={len(frames)})")
                else:
                    print(f"  GIF skipped: {msg}")
    finally:
        env.close()

    rewards = np.array([r["reward"] for r in results], dtype=np.float32)
    lengths = np.array([r["length"] for r in results], dtype=np.float32)
    speeds = np.array([r["mean_speed_kmh"] for r in results], dtype=np.float32)
    success_rate = float(np.mean([r["success"] for r in results])) if results else 0.0
    collision_rate = float(np.mean([r["collision"] for r in results])) if results else 0.0

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": os.path.abspath(args.model),
        "episodes": args.episodes,
        "avg_reward": float(rewards.mean()) if len(rewards) else 0.0,
        "std_reward": float(rewards.std()) if len(rewards) else 0.0,
        "avg_length": float(lengths.mean()) if len(lengths) else 0.0,
        "avg_speed_kmh": float(speeds.mean()) if len(speeds) else 0.0,
        "success_rate": success_rate,
        "collision_rate": collision_rate,
        "termination_reason_counts": dict(reason_counter),
        "gif_paths": gif_paths,
        "per_episode": results,
    }

    print("\n" + "=" * 64)
    print("L1 Evaluation Summary")
    print("=" * 64)
    print(f"episodes           : {summary['episodes']}")
    print(f"avg_reward         : {summary['avg_reward']:.2f} +- {summary['std_reward']:.2f}")
    print(f"avg_length         : {summary['avg_length']:.2f}")
    print(f"avg_speed_kmh      : {summary['avg_speed_kmh']:.2f}")
    print(f"success_rate       : {summary['success_rate']:.3f}")
    print(f"collision_rate     : {summary['collision_rate']:.3f}")
    print(f"termination_counts : {summary['termination_reason_counts']}")

    if args.save_json:
        out_path = os.path.abspath(args.save_json)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\nSaved summary json: {out_path}")


if __name__ == "__main__":
    run_eval(parse_args())
