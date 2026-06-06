"""
visualize_svo_tits_v3.py -- TITS/IEEE风格的SVO可视化脚本（平滑KDE + SVG导出版）

改进点:
1. 保留完整坐标轴方框
2. SVO五类标签改为：Competitive / Egotistical / Normal / Prosocial / Altruistic
3. 图中不显示样本数 n
4. KDE曲线进一步平滑，避免锯齿感
5. 删除KDE图中的竖向虚线
6. 同时导出 PNG + PDF + SVG，其中会生成三张 SVG 图

标签映射（按SVO由低到高解释）:
    aggressive         -> Competitive
    semi_aggressive    -> Egotistical
    normal             -> Normal
    semi_conservative  -> Prosocial
    conservative       -> Altruistic

用法:
    python visualize_svo_tits_v3.py --model svo_pretrained.pt --dataset svo_dataset.npz
"""

import os
import argparse
import warnings
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

try:
    from scipy.stats import gaussian_kde
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy导入失败，将使用替代KDE绘图")

from src.config import get_default_config
from src.models.svo_model import SVOVariationalBIRL


STYLE_ORDER = [
    "aggressive",
    "semi_aggressive",
    "normal",
    "semi_conservative",
    "conservative",
]

COLOR_MAP = {
    "aggressive": "#C44E52",
    "semi_aggressive": "#DD8452",
    "normal": "#4C72B0",
    "semi_conservative": "#55A868",
    "conservative": "#8172B3",
}

LABEL_MAP = {
    "aggressive": "Competitive",
    "semi_aggressive": "Egotistical",
    "normal": "Normal",
    "semi_conservative": "Prosocial",
    "conservative": "Altruistic",
}

ONE_COL = (3.5, 2.8)
TWO_COL = (7.0, 3.5)


def parse_args():
    parser = argparse.ArgumentParser(description="TITS风格SVO预训练可视化（平滑KDE + SVG导出版）")
    parser.add_argument("--model", type=str, required=True,default=r"D:\桌面\毕设代码\SVO-CVaR\svo_pretrained.pt", help="预训练模型路径")
    parser.add_argument("--dataset", type=str, required=True,default=r"D:\桌面\毕设代码\SVO-CVaR\pretrain_svo\svo_dataset 3.11 300ep.npz", help="数据集路径")
    parser.add_argument("--output_dir", type=str, default="figures_tits", help="输出目录")
    parser.add_argument("--device", type=str, default="auto", help="设备")
    parser.add_argument("--max_samples", type=int, default=10000, help="最大推断样本数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--kde_bw", type=float, default=1.5, help="KDE带宽，越大越平滑")
    parser.add_argument("--smooth_window", type=int, default=13, help="额外平滑窗口大小，建议奇数")
    parser.add_argument("--smooth_sigma", type=float, default=2.2, help="额外高斯平滑sigma")
    return parser.parse_args()



def set_tits_style():
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.5,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.8,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.spines.top": True,
        "axes.spines.right": True,
    })


@torch.no_grad()
def infer_svo(model, ego_past, npc_past, device, batch_size=256):
    model.eval()
    all_mu, all_sigma = [], []
    for i in range(0, len(ego_past), batch_size):
        ego_b = torch.as_tensor(ego_past[i:i + batch_size], dtype=torch.float32, device=device)
        npc_b = torch.as_tensor(npc_past[i:i + batch_size], dtype=torch.float32, device=device)
        mu, sigma = model.infer(npc_b, ego_b)
        all_mu.append(mu.detach().cpu().numpy())
        all_sigma.append(sigma.detach().cpu().numpy())
    return np.concatenate(all_mu), np.concatenate(all_sigma)



def style_axis(ax, xlim=(0, 90), ylim=None, xlabel=None, ylabel=None, x_major=15, y_major=None):
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.xaxis.set_major_locator(MultipleLocator(x_major))
    if y_major is not None:
        ax.yaxis.set_major_locator(MultipleLocator(y_major))

    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.25)
    ax.grid(False, axis="x")

    for spine in ["left", "bottom", "top", "right"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(0.8)
        ax.spines[spine].set_color("#333333")



def gaussian_kernel_1d(window_size, sigma):
    window_size = max(int(window_size), 3)
    if window_size % 2 == 0:
        window_size += 1
    radius = window_size // 2
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / max(float(sigma), 1e-6)) ** 2)
    kernel /= kernel.sum()
    return kernel



def smooth_density(density, window_size=13, sigma=2.2):
    kernel = gaussian_kernel_1d(window_size, sigma)
    return np.convolve(density, kernel, mode="same")



def kde_simple(data, x_grid, bw_factor=0.32):
    n = len(data)
    if n == 0:
        return np.zeros_like(x_grid)
    std = np.std(data)
    bw = max(std * bw_factor * n ** (-1 / 5), 1e-3)
    density = np.zeros_like(x_grid)
    for i, x in enumerate(x_grid):
        density[i] = np.sum(np.exp(-0.5 * ((x - data) / bw) ** 2)) / (
            n * bw * np.sqrt(2 * np.pi)
        )
    return density



def save_figure(fig, output_path_no_ext):
    png_path = f"{output_path_no_ext}.png"
    pdf_path = f"{output_path_no_ext}.pdf"
    svg_path = f"{output_path_no_ext}.svg"
    fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"  已保存: {png_path}")
    print(f"  已保存: {pdf_path}")
    print(f"  已保存: {svg_path}")



def plot_svo_kde(mu_by_style, output_path_no_ext, kde_bw=0.32, smooth_window=13, smooth_sigma=2.2):
    fig, ax = plt.subplots(figsize=TWO_COL)
    x_grid = np.linspace(0, 90, 700)
    legend_handles = []

    for style in STYLE_ORDER:
        data = np.asarray(mu_by_style.get(style, []), dtype=float)
        if len(data) == 0:
            continue

        if SCIPY_AVAILABLE:
            kde = gaussian_kde(data, bw_method=kde_bw)
            density = kde(x_grid)
        else:
            density = kde_simple(data, x_grid, bw_factor=kde_bw)

        density = smooth_density(density, window_size=smooth_window, sigma=smooth_sigma)
        density = np.clip(density, 0, None)

        color = COLOR_MAP[style]
        ax.plot(
            x_grid,
            density,
            color=color,
            linewidth=1.9,
            solid_capstyle="round",
            solid_joinstyle="round",
            antialiased=True,
        )
        ax.fill_between(x_grid, density, 0, color=color, alpha=0.10)

        legend_handles.append(
            Line2D([0], [0], color=color, lw=1.9, label=LABEL_MAP[style])
        )

    style_axis(
        ax,
        xlim=(0, 90),
        ylim=(0, None),
        xlabel="Inferred SVO angle (deg)",
        ylabel="Density",
        x_major=15,
    )
    ax.legend(handles=legend_handles, loc="upper right", frameon=False, handlelength=2.2)
    save_figure(fig, output_path_no_ext)



def plot_svo_boxplot(mu_by_style, output_path_no_ext):
    fig, ax = plt.subplots(figsize=ONE_COL)

    data, labels, colors = [], [], []
    for style in STYLE_ORDER:
        arr = np.asarray(mu_by_style.get(style, []), dtype=float)
        if len(arr) == 0:
            continue
        data.append(arr)
        labels.append(LABEL_MAP[style])
        colors.append(COLOR_MAP[style])

    bp = ax.boxplot(
        data,
        patch_artist=True,
        widths=0.55,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.2),
        whiskerprops=dict(color="#555555", linewidth=0.8),
        capprops=dict(color="#555555", linewidth=0.8),
        boxprops=dict(linewidth=0.8, color="#555555"),
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)

    rng = np.random.default_rng(42)
    for i, (arr, color) in enumerate(zip(data, colors), start=1):
        n_show = min(len(arr), 180)
        sample = rng.choice(arr, size=n_show, replace=False) if len(arr) > n_show else arr
        jitter = rng.normal(0, 0.045, size=len(sample))
        ax.scatter(
            np.full_like(sample, i, dtype=float) + jitter,
            sample,
            s=6,
            alpha=0.18,
            color=color,
            edgecolors="none",
            rasterized=True,
        )

    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=18, ha="right")
    style_axis(
        ax,
        xlim=(0.5, len(labels) + 0.5),
        ylim=(0, 90),
        xlabel=None,
        ylabel="Inferred SVO angle (deg)",
        x_major=1,
        y_major=15,
    )
    save_figure(fig, output_path_no_ext)



def plot_mu_sigma_scatter(mu_by_style, sigma_by_style, output_path_no_ext):
    fig, ax = plt.subplots(figsize=TWO_COL)

    for style in STYLE_ORDER:
        mu = np.asarray(mu_by_style.get(style, []), dtype=float)
        sigma = np.asarray(sigma_by_style.get(style, []), dtype=float)
        if len(mu) == 0:
            continue

        ax.scatter(
            mu,
            sigma,
            s=8,
            alpha=0.22,
            color=COLOR_MAP[style],
            edgecolors="none",
            label=LABEL_MAP[style],
            rasterized=True,
        )

        ax.scatter(
            [mu.mean()],
            [sigma.mean()],
            s=34,
            color=COLOR_MAP[style],
            edgecolors="black",
            linewidths=0.4,
            zorder=4,
        )

    style_axis(
        ax,
        xlim=(0, 90),
        ylim=(0, None),
        xlabel="Posterior mean $\\mu$ (deg)",
        ylabel="Posterior uncertainty $\\sigma$ (deg)",
        x_major=15,
    )
    ax.legend(loc="upper right", frameon=False, handletextpad=0.4)
    save_figure(fig, output_path_no_ext)



def summarize_statistics(mu_by_style, sigma_by_style):
    print("\n" + "=" * 72)
    print("Summary statistics for paper/report")
    print(f"{'Internal style':<22}{'Display label':<16}{'n':>8}{'mu mean':>12}{'mu std':>12}{'sigma mean':>14}")
    print("-" * 72)
    for style in STYLE_ORDER:
        if style not in mu_by_style:
            continue
        mu = np.asarray(mu_by_style[style], dtype=float)
        sigma = np.asarray(sigma_by_style[style], dtype=float)
        if len(mu) == 0:
            continue
        print(
            f"{style:<22}{LABEL_MAP[style]:<16}{len(mu):>8}{mu.mean():>11.2f}°{mu.std():>11.2f}°{sigma.mean():>13.2f}°"
        )
    print("=" * 72)



def main():
    args = parse_args()
    np.random.seed(args.seed)
    set_tits_style()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"加载数据集: {args.dataset}")
    data = np.load(args.dataset, allow_pickle=True)
    ego_past = data["ego_past"]
    npc_past = data["npc_past"]

    if "styles" in data:
        styles = data["styles"]
        print(f"  样本数: {len(ego_past)}, 含风格标签")
    else:
        styles = np.array(["unknown"] * len(ego_past))
        print(f"  样本数: {len(ego_past)}, 无风格标签")

    if len(ego_past) > args.max_samples:
        idx = np.random.choice(len(ego_past), args.max_samples, replace=False)
        ego_past = ego_past[idx]
        npc_past = npc_past[idx]
        styles = styles[idx]
        print(f"  采样至 {args.max_samples} 条")

    print(f"加载模型: {args.model}")
    config = get_default_config()
    model = SVOVariationalBIRL(config).to(device)
    ckpt = torch.load(args.model, map_location=device)
    if isinstance(ckpt, dict) and "svo_birl" in ckpt:
        model.load_state_dict(ckpt["svo_birl"])
    else:
        model.load_state_dict(ckpt)

    print("运行SVO推断...")
    mu_all, sigma_all = infer_svo(model, ego_past, npc_past, device)

    mu_by_style = {}
    sigma_by_style = {}
    for style in np.unique(styles):
        mask = styles == style
        mu_by_style[style] = mu_all[mask]
        sigma_by_style[style] = sigma_all[mask]
        print(
            f"  {style} -> {LABEL_MAP.get(style, style)}: n={mask.sum()}, "
            f"mu={mu_all[mask].mean():.2f}°±{mu_all[mask].std():.2f}°, "
            f"sigma={sigma_all[mask].mean():.2f}°"
        )

    print("\n生成图表...")
    plot_svo_kde(
        mu_by_style,
        os.path.join(args.output_dir, "fig_svo_kde"),
        kde_bw=args.kde_bw,
        smooth_window=args.smooth_window,
        smooth_sigma=args.smooth_sigma,
    )
    plot_svo_boxplot(mu_by_style, os.path.join(args.output_dir, "fig_svo_boxplot"))
    plot_mu_sigma_scatter(mu_by_style, sigma_by_style, os.path.join(args.output_dir, "fig_svo_mu_sigma"))

    summarize_statistics(mu_by_style, sigma_by_style)
    print(f"\n输出目录: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
