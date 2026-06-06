"""
plot_ablation.py — 读 ablation 实验 TensorBoard 日志, 出论文级图表

核心思路: 自动扫 outputs/ 下所有训练目录, 从每个目录的 logs/config.json
读 (use_spo, use_svo_adaptive_spo, use_dapo, cvar_enabled, svo.enabled) 自动
识别属于哪个 ablation 组, 不需要手动登记.

用法
----
1) 完全自动 (推荐):
   python plot_ablation.py --scan outputs/ --out_dir figures/

2) 手动指定 (混用不同 outputs):
   python plot_ablation.py \\
       --runs g6_spo_adaptive=outputs/train_xxx,outputs/train_yyy \\
       --runs g4_ppo_svo_cvar=outputs/L2/train_zzz \\
       --out_dir figures/

3) 过滤 (只画 SPO vs PPO+SVO+CVaR):
   python plot_ablation.py --scan outputs/ --groups "4 5 6" --out_dir figures/

输出 (figures/ 下)
-----------------
  fig_train_reward.pdf         主训练曲线 reward
  fig_train_cost.pdf           CVaR cost 曲线
  fig_ratio_deviation.pdf      ★ 复刻 SPO 论文 Figure 6, 验证 trust region 控制 ★
  fig_spo_epsilon.pdf          自适应 ε 时序曲线
  fig_eps_vs_rho_scatter.pdf   ε vs ρ 散点 (验证自适应机制)
  fig_main_table.csv           主结果表 (各 metric mean ± std)
  fig_main_table.tex           LaTeX 主结果表 (booktabs 格式)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("[Error] 需要 tensorboard: pip install tensorboard", file=sys.stderr)
    sys.exit(1)


# ===================================================================== #
#  Group 定义 (画图属性 + 自动识别规则)                                       #
# ===================================================================== #

# 6 组的 (label, color, linestyle, identification rule)
# rule 是一个 lambda(ppo_cfg, svo_cfg) -> bool
GROUPS = [
    {
        "id": "g1_ppo_baseline",
        "label": "PPO",
        "color": "#888888", "ls": "--",
        "rule": lambda p, s: (
            not p.get("use_dapo", False)
            and not bool(p.get("use_spo", False))
            and not s.get("enabled", True)
            and not bool(p.get("cvar_enabled", False))
        ),
    },
    {
        "id": "g2_ppo_cvar",
        "label": "PPO + CVaR",
        "color": "#4477AA", "ls": "--",
        "rule": lambda p, s: (
            not p.get("use_dapo", False)
            and not bool(p.get("use_spo", False))
            and not s.get("enabled", True)
            and bool(p.get("cvar_enabled", False))
        ),
    },
    {
        "id": "g3_ppo_svo",
        "label": "PPO + SVO",
        "color": "#66CCEE", "ls": "--",
        "rule": lambda p, s: (
            not p.get("use_dapo", False)
            and not bool(p.get("use_spo", False))
            and s.get("enabled", True)
            and not bool(p.get("cvar_enabled", False))
        ),
    },
    {
        "id": "g4_ppo_svo_cvar",
        "label": "PPO + SVO + CVaR",
        "color": "#228833", "ls": "-",
        "rule": lambda p, s: (
            not p.get("use_dapo", False)
            and not bool(p.get("use_spo", False))
            and s.get("enabled", True)
            and bool(p.get("cvar_enabled", False))
        ),
    },
    {
        "id": "g5_spo_fixed",
        "label": "Fixed SPO + SVO + CVaR",
        "color": "#EE6677", "ls": "-",
        "rule": lambda p, s: (
            bool(p.get("use_spo", False))
            and not bool(p.get("use_svo_adaptive_spo", False))
        ),
    },
    {
        "id": "g6_spo_adaptive",
        "label": "Adaptive SPO (Ours)",
        "color": "#AA3377", "ls": "-",
        "rule": lambda p, s: (
            bool(p.get("use_spo", False))
            and bool(p.get("use_svo_adaptive_spo", False))
        ),
    },
]

GROUP_BY_ID = {g["id"]: g for g in GROUPS}
GROUP_ORDER = [g["id"] for g in GROUPS]


# ===================================================================== #
#  TensorBoard / config 读取                                              #
# ===================================================================== #

def _find_tb_dir(exp_dir: str) -> Optional[str]:
    candidates = [os.path.join(exp_dir, "tensorboard"), exp_dir]
    for c in candidates:
        if os.path.isdir(c) and glob(os.path.join(c, "events.out.tfevents.*")):
            return c
    return None


def load_scalars(exp_dir: str, tags: List[str]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    tb_dir = _find_tb_dir(exp_dir)
    if tb_dir is None:
        return {}
    ea = EventAccumulator(tb_dir, size_guidance={"scalars": 0})
    try:
        ea.Reload()
    except Exception as e:
        print(f"  [Warn] {tb_dir} 加载失败: {e}")
        return {}
    available = set(ea.Tags().get("scalars", []))
    out = {}
    for tag in tags:
        if tag not in available:
            continue
        events = ea.Scalars(tag)
        steps = np.array([e.step for e in events])
        vals = np.array([e.value for e in events])
        if len(steps) > 0:
            out[tag] = (steps, vals)
    return out


def load_config(exp_dir: str) -> Optional[dict]:
    cfg_path = os.path.join(exp_dir, "logs", "config.json")
    if not os.path.exists(cfg_path):
        return None
    try:
        with open(cfg_path) as f:
            return json.load(f)
    except Exception as e:
        print(f"  [Warn] {cfg_path} 解析失败: {e}")
        return None


def identify_group(cfg: dict) -> Optional[str]:
    if cfg is None:
        return None
    p = cfg.get("ppo", {})
    s = cfg.get("svo", {})
    for g in GROUPS:
        try:
            if g["rule"](p, s):
                return g["id"]
        except Exception:
            continue
    return None


# ===================================================================== #
#  自动扫描 outputs/                                                       #
# ===================================================================== #

def scan_outputs(scan_root: str) -> Dict[str, List[str]]:
    """
    递归扫 scan_root, 找所有同时包含 logs/config.json 和 tensorboard/ 的目录,
    根据 config.json 识别所属 group.
    """
    found = defaultdict(list)
    skipped_unidentified = []
    skipped_no_tb = []

    for cfg_path in glob(os.path.join(scan_root, "**", "logs", "config.json"),
                         recursive=True):
        exp_dir = os.path.dirname(os.path.dirname(cfg_path))
        if _find_tb_dir(exp_dir) is None:
            skipped_no_tb.append(exp_dir)
            continue
        cfg = load_config(exp_dir)
        gid = identify_group(cfg)
        if gid is None:
            skipped_unidentified.append(exp_dir)
            continue
        found[gid].append(exp_dir)

    print(f"[Scan] 在 {scan_root} 找到 "
          f"{sum(len(v) for v in found.values())} 个可识别实验:")
    for gid in GROUP_ORDER:
        if gid in found:
            print(f"  {GROUP_BY_ID[gid]['label']:30s} ({gid}): "
                  f"{len(found[gid])} runs")
            for d in found[gid]:
                print(f"    - {d}")
    if skipped_unidentified:
        print(f"\n[Scan] 跳过 {len(skipped_unidentified)} 个无法识别 group 的目录:")
        for d in skipped_unidentified[:5]:
            print(f"  - {d}")
        if len(skipped_unidentified) > 5:
            print(f"  ... (共 {len(skipped_unidentified)} 个)")
    if skipped_no_tb:
        print(f"[Scan] 跳过 {len(skipped_no_tb)} 个无 TensorBoard 数据的目录")

    return dict(found)


def parse_runs_arg(runs_args: List[str]) -> Dict[str, List[str]]:
    """
    解析 --runs 参数 (可多次):
      --runs g6_spo_adaptive=outputs/train_xxx,outputs/train_yyy
    """
    runs = defaultdict(list)
    for spec in runs_args:
        if "=" not in spec:
            print(f"[Warn] --runs 格式应为 'group_id=path1,path2', 忽略: {spec}")
            continue
        gid, paths = spec.split("=", 1)
        gid = gid.strip()
        if gid not in GROUP_BY_ID:
            print(f"[Warn] 未知 group_id '{gid}', 合法值: {list(GROUP_BY_ID)}")
            continue
        for p in paths.split(","):
            p = p.strip()
            if not p:
                continue
            if not os.path.isdir(p):
                print(f"[Warn] 不是目录, 跳过: {p}")
                continue
            runs[gid].append(p)
    return dict(runs)


# ===================================================================== #
#  曲线聚合                                                                 #
# ===================================================================== #

def ema_smooth(values: np.ndarray, alpha: float = 0.9) -> np.ndarray:
    if len(values) == 0:
        return values
    out = np.zeros_like(values, dtype=float)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * out[i - 1] + (1 - alpha) * values[i]
    return out


def aggregate_seeds(
    seed_curves: List[Tuple[np.ndarray, np.ndarray]],
    smooth_alpha: float = 0.9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not seed_curves:
        return np.array([]), np.array([]), np.array([])
    max_idx = int(np.argmax([len(s) for s, _ in seed_curves]))
    common_steps = seed_curves[max_idx][0]
    aligned = []
    for steps, vals in seed_curves:
        aligned.append(np.interp(common_steps, steps,
                                 ema_smooth(vals, alpha=smooth_alpha)))
    arr = np.stack(aligned, axis=0)
    return common_steps, arr.mean(axis=0), arr.std(axis=0)


# ===================================================================== #
#  绘图原语                                                                 #
# ===================================================================== #

def setup_matplotlib():
    plt.rcParams.update({
        "font.size": 11,
        "font.family": "serif",
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 100,
    })


def _format_xticks(ax):
    def fmt(x, _pos):
        if abs(x) >= 1e6:
            return f"{x / 1e6:.1f}M"
        if abs(x) >= 1e3:
            return f"{x / 1e3:.0f}k"
        return f"{int(x)}"
    ax.xaxis.set_major_formatter(plt.FuncFormatter(fmt))


def plot_curve(ax, runs, tag, smooth_alpha=0.9, show_std=True) -> bool:
    plotted = False
    for gid in GROUP_ORDER:
        if gid not in runs:
            continue
        g = GROUP_BY_ID[gid]
        seed_curves = []
        for exp_dir in runs[gid]:
            sc = load_scalars(exp_dir, [tag])
            if tag in sc:
                seed_curves.append(sc[tag])
        if not seed_curves:
            continue
        steps, mean, std = aggregate_seeds(seed_curves, smooth_alpha=smooth_alpha)
        n = len(seed_curves)
        ax.plot(steps, mean, color=g["color"], linewidth=2, linestyle=g["ls"],
                label=f"{g['label']} (n={n})")
        if show_std and n > 1:
            ax.fill_between(steps, mean - std, mean + std,
                            color=g["color"], alpha=0.15)
        plotted = True
    _format_xticks(ax)
    ax.set_xlabel("Training Steps")
    return plotted


# ===================================================================== #
#  6 个具体的图                                                              #
# ===================================================================== #

def fig_train_reward(runs, out_path, smooth_alpha):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if not plot_curve(ax, runs, "episode/reward", smooth_alpha):
        print("  [Skip] episode/reward 没数据")
        plt.close(fig); return
    ax.set_ylabel("Episode Reward")
    ax.set_title("Training Reward (mean ± std over seeds)")
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)
    print(f"  Saved: {out_path}")


def fig_train_cost(runs, out_path, smooth_alpha):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plotted = False
    for tag in ["episode/cost", "episode/episode_cost", "cvar/avg_episode_cost"]:
        if plot_curve(ax, runs, tag, smooth_alpha):
            plotted = True; break
    if not plotted:
        print("  [Skip] episode/cost 没数据")
        plt.close(fig); return
    ax.set_ylabel("Episode Cost")
    ax.set_title("Episode Cost (lower is safer)")
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)
    print(f"  Saved: {out_path}")


def fig_ratio_deviation(runs, out_path, smooth_alpha):
    """★ 论文核心图 ★ 复刻 SPO 论文 Figure 6 (PPO 没这个 tag, 只显示 SPO 系)"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, tag, ylab, title in [
        (axes[0], "policy/ratio_deviation_mean",
         r"$\mathbb{E}[\,|\,r_t(\theta)-1\,|\,]$", "Mean Ratio Deviation"),
        (axes[1], "policy/ratio_deviation_max",
         r"$\max_t |\,r_t(\theta)-1\,|$", "Max Ratio Deviation"),
    ]:
        plot_curve(ax, runs, tag, smooth_alpha)
        ax.set_ylabel(ylab)
        ax.set_yscale("log")
        ax.axhline(0.2, color="red", linestyle=":", alpha=0.5,
                   label=r"$\epsilon = 0.2$")
        ax.set_title(title)
        ax.legend(loc="best", fontsize=9, framealpha=0.9)
    fig.suptitle("Trust-Region Control (lower is tighter, like SPO Figure 6)",
                 y=1.02, fontsize=12)
    fig.tight_layout(); fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out_path}")


def fig_spo_epsilon(runs, out_path, smooth_alpha):
    """SPO 自适应 ε 的时序变化"""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plotted = False
    for gid in ["g5_spo_fixed", "g6_spo_adaptive"]:
        if gid not in runs:
            continue
        g = GROUP_BY_ID[gid]
        sc_mean, sc_min, sc_max = [], [], []
        for exp_dir in runs[gid]:
            sc = load_scalars(exp_dir, [
                "policy/spo_epsilon_mean",
                "policy/spo_epsilon_min",
                "policy/spo_epsilon_max",
            ])
            if "policy/spo_epsilon_mean" in sc: sc_mean.append(sc["policy/spo_epsilon_mean"])
            if "policy/spo_epsilon_min" in sc:  sc_min.append(sc["policy/spo_epsilon_min"])
            if "policy/spo_epsilon_max" in sc:  sc_max.append(sc["policy/spo_epsilon_max"])
        if not sc_mean:
            continue
        steps, mean_, _ = aggregate_seeds(sc_mean, smooth_alpha=smooth_alpha)
        ax.plot(steps, mean_, color=g["color"], linewidth=2,
                label=f"{g['label']} mean ε")
        if sc_min and sc_max:
            _, mn, _ = aggregate_seeds(sc_min, smooth_alpha=smooth_alpha)
            _, mx, _ = aggregate_seeds(sc_max, smooth_alpha=smooth_alpha)
            ax.fill_between(steps, mn, mx, color=g["color"], alpha=0.15,
                            label=f"{g['label']} [min, max]")
        plotted = True

    if not plotted:
        print("  [Skip] policy/spo_epsilon_* 没数据 (没跑 SPO 组)")
        plt.close(fig); return

    ax.axhline(0.2, color="gray", linestyle=":", alpha=0.5,
               label=r"$\epsilon_{base}=0.2$")
    ax.axhline(0.05, color="gray", linestyle=":", alpha=0.5,
               label=r"$\epsilon_{min}=0.05$")
    _format_xticks(ax)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel(r"Trust-Region Width $\epsilon_t$")
    ax.set_title("SPO Trust-Region Width Over Training")
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)
    print(f"  Saved: {out_path}")


def fig_eps_vs_rho(runs, out_path, alpha_param=1.5,
                   eps_base=0.2, eps_min=0.05, eps_max=0.2):
    """ε vs ρ 散点 + 理论曲线"""
    fig, ax = plt.subplots(figsize=(6.5, 5))
    if "g6_spo_adaptive" not in runs:
        print("  [Skip] 没有 g6_spo_adaptive 组")
        plt.close(fig); return

    all_rho, all_eps, all_steps = [], [], []
    for exp_dir in runs["g6_spo_adaptive"]:
        sc = load_scalars(exp_dir, [
            "policy/svo_risk_mean", "policy/spo_epsilon_mean",
        ])
        if ("policy/svo_risk_mean" in sc
                and "policy/spo_epsilon_mean" in sc):
            rs, rv = sc["policy/svo_risk_mean"]
            es, ev = sc["policy/spo_epsilon_mean"]
            ev_aligned = np.interp(rs, es, ev)
            all_rho.extend(rv); all_eps.extend(ev_aligned); all_steps.extend(rs)

    if not all_rho:
        print("  [Skip] policy/svo_risk_mean 没数据")
        plt.close(fig); return

    all_rho = np.array(all_rho); all_eps = np.array(all_eps); all_steps = np.array(all_steps)
    sc = ax.scatter(all_rho, all_eps, c=all_steps, cmap="viridis",
                    s=14, alpha=0.55, edgecolors="none")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, _: f"{x / 1e6:.1f}M" if abs(x) >= 1e6 else f"{x / 1e3:.0f}k"
    ))
    cbar.set_label("Training Step")

    rho_grid = np.linspace(0, 1, 100)
    eps_theory = np.clip(eps_base * np.exp(-alpha_param * rho_grid),
                         eps_min, eps_max)
    ax.plot(rho_grid, eps_theory, "r--", linewidth=2, alpha=0.85,
            label=fr"$\epsilon_t = \mathrm{{clip}}({eps_base}\,e^{{-{alpha_param}\rho}}, "
                  fr"{eps_min}, {eps_max})$")

    ax.set_xlabel(r"SVO Interaction Urgency $\rho_t$")
    ax.set_ylabel(r"Trust-Region Width $\epsilon_t$")
    ax.set_title("Adaptive ε responds to SVO risk")
    ax.set_xlim(0, 1); ax.set_ylim(0, eps_max + 0.02)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)
    print(f"  Saved: {out_path}")


def make_main_table(runs, out_csv, out_tex, last_pct: float = 0.1):
    metrics = [
        ("Reward",       "episode/reward",                "↑"),
        ("Cost",         "episode/cost",                  "↓"),
        ("RatioDevMean", "policy/ratio_deviation_mean",   "↓"),
        ("RatioDevMax",  "policy/ratio_deviation_max",    "↓"),
    ]
    rows = []
    for gid in GROUP_ORDER:
        if gid not in runs:
            continue
        g = GROUP_BY_ID[gid]
        row = {"Group": g["label"], "n_seeds": len(runs[gid])}
        for name, tag, _ in metrics:
            seed_finals = []
            for exp_dir in runs[gid]:
                sc = load_scalars(exp_dir, [tag])
                if tag not in sc:
                    continue
                _, vals = sc[tag]
                if len(vals) == 0:
                    continue
                cutoff = int(len(vals) * (1 - last_pct))
                seed_finals.append(float(np.mean(vals[cutoff:])))
            if seed_finals:
                row[f"{name}_mean"] = float(np.mean(seed_finals))
                row[f"{name}_std"] = float(np.std(seed_finals))
            else:
                row[f"{name}_mean"] = float("nan")
                row[f"{name}_std"] = float("nan")
        rows.append(row)

    if not rows:
        print("  [Skip] 没有可输出的行"); return

    # CSV
    fieldnames = ["Group", "n_seeds"] + [f"{n}_{m}" for n, _, _ in metrics for m in ("mean", "std")]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(rows)
    print(f"  Saved: {out_csv}")

    # LaTeX
    with open(out_tex, "w") as f:
        f.write("% Auto-generated by plot_ablation.py\n")
        f.write("% \\usepackage{booktabs} required\n")
        f.write("\\begin{tabular}{l c " + "r " * len(metrics) + "}\n")
        f.write("\\toprule\n")
        cols = ["Method", "n"] + [f"{n} {a}" for n, _, a in metrics]
        f.write(" & ".join(cols) + " \\\\\n\\midrule\n")
        for row in rows:
            cells = [row["Group"], str(row["n_seeds"])]
            for n, _, _ in metrics:
                m, s = row[f"{n}_mean"], row[f"{n}_std"]
                cells.append("---" if np.isnan(m) else f"${m:.3f}\\!\\pm\\!{s:.3f}$")
            f.write(" & ".join(cells) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"  Saved: {out_tex}")

    # 终端打印
    print()
    print("  ── Main Results ─────────────────────────────────────────")
    fmt = "  {:<28} {:>4} | {:>14} | {:>14} | {:>14}"
    print(fmt.format("Group", "n", "Reward ↑", "Cost ↓", "RatioDevMean ↓"))
    print("  " + "─" * 80)
    for row in rows:
        def s(name):
            m, sd = row.get(f"{name}_mean"), row.get(f"{name}_std")
            return "    N/A    " if (m is None or np.isnan(m)) else f"{m:8.3f}±{sd:5.3f}"
        print(fmt.format(row["Group"][:28], row["n_seeds"],
                         s("Reward"), s("Cost"), s("RatioDevMean")))


# ===================================================================== #
#  Main                                                                    #
# ===================================================================== #

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--scan", type=str,
                     help="自动扫描根目录 (一般是 outputs/)")
    src.add_argument("--runs", action="append", default=[],
                     help='手动指定 (可多次): "group_id=path1,path2"')

    ap.add_argument("--out_dir", type=str, default="figures",
                    help="图表输出目录")
    ap.add_argument("--groups", type=str, default="",
                    help="只画指定组, 空格分隔, 如 '4 5 6' 或 'g4 g5 g6'")
    ap.add_argument("--smooth", type=float, default=0.9,
                    help="EMA 平滑 (0=不平滑, 0.99=极平滑)")
    ap.add_argument("--last_pct", type=float, default=0.1,
                    help="主表统计最后 X%% 训练步")
    ap.add_argument("--alpha", type=float, default=1.5,
                    help="ε vs ρ 散点图的理论曲线 α 参数 (= spo_risk_alpha)")
    ap.add_argument("--eps_base", type=float, default=0.2)
    ap.add_argument("--eps_min", type=float, default=0.05)
    ap.add_argument("--eps_max", type=float, default=0.2)
    args = ap.parse_args()

    setup_matplotlib()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # 加载 runs
    if args.scan:
        runs = scan_outputs(args.scan)
    else:
        runs = parse_runs_arg(args.runs)

    if not runs:
        print("[Error] 没有任何可用 run, 请检查 --scan 或 --runs 参数")
        sys.exit(1)

    # 过滤 groups
    if args.groups:
        sel_prefixes = []
        for g in args.groups.split():
            g = g.strip().lstrip("g")
            sel_prefixes.append(f"g{g}_")
        runs = {k: v for k, v in runs.items()
                if any(k.startswith(p) for p in sel_prefixes)}
        if not runs:
            print(f"[Error] --groups '{args.groups}' 过滤后为空")
            sys.exit(1)

    print()
    print(f"[Plot] 生成图表 ({len(runs)} 组, "
          f"共 {sum(len(v) for v in runs.values())} runs) ...")
    fig_train_reward(runs,    out / "fig_train_reward.pdf",    args.smooth)
    fig_train_cost(runs,      out / "fig_train_cost.pdf",      args.smooth)
    fig_ratio_deviation(runs, out / "fig_ratio_deviation.pdf", args.smooth)
    fig_spo_epsilon(runs,     out / "fig_spo_epsilon.pdf",     args.smooth)
    fig_eps_vs_rho(runs,      out / "fig_eps_vs_rho_scatter.pdf",
                   alpha_param=args.alpha,
                   eps_base=args.eps_base, eps_min=args.eps_min, eps_max=args.eps_max)
    print()
    print("[Table] 生成主结果表 ...")
    make_main_table(runs,
                    out / "fig_main_table.csv",
                    out / "fig_main_table.tex",
                    last_pct=args.last_pct)
    print()
    print(f"[Done] 全部输出到: {out.resolve()}")


if __name__ == "__main__":
    main()