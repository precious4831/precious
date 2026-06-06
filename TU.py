"""
draw_trajectory.py
------------------
运行方式：
    python draw_trajectory.py

输出：
    traj_aggressive.svg   —— 激进型NPC场景对比图
    traj_conservative.svg —— 保守型NPC场景对比图

依赖：numpy, matplotlib
    pip install numpy matplotlib
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.transforms import Affine2D

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['axes.linewidth'] = 0.7

# ══════════════════════════════════════════════════════
#  可调参数（在这里修改布局与样式）
# ══════════════════════════════════════════════════════

LANE_Y_EGO = 0.0    # 自车车道中心 y
LANE_Y_NPC = 1.0    # NPC车道中心 y
LANE_TOP   = 1.22   # 上边界
LANE_BOT   = -0.22  # 下边界

CAR_W = 0.046       # 车辆快照纵向长度
CAR_H = 0.082       # 车辆快照横向高度

OBS_X = 0.42        # 障碍物 x 位置（0~1归一化）

FIG_W  = 13.0       # 图宽（英寸）
FIG_H  = 3.8        # 图高（英寸）

EGO_COLOR_OURS = '#2980b9'   # Ours 自车颜色
EGO_COLOR_PPO  = '#c0392b'   # PPO  自车颜色（碰撞）
OBS_COLOR      = '#FFD700'   # 障碍物颜色
ROAD_COLOR     = '#f4f6f7'
LANE_COLOR     = '#d7dce0'

ANNOTATION_FONTSIZE = 7.8
TITLE_FONTSIZE      = 10.5
LEGEND_FONTSIZE     = 8.5

OUTPUT_FORMAT = 'svg'   # 改为 'png' 则输出 PNG，DPI 由下方控制
OUTPUT_DPI    = 200     # 仅 PNG 有效

SAVE_GIF = True
GIF_DURATION = 6.0
GIF_FPS = 10
GIF_DPI = 135


# ══════════════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════════════

def min_jerk(t):
    """Minimum-jerk easing: zero slope and acceleration at both ends."""
    t = np.clip(t, 0.0, 1.0)
    return 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5


def smooth_lc(lc_start, lc_end, y_from, y_to, n=900):
    """
    生成平滑 S 形变道曲线。
    lc_start / lc_end : 变道开始/结束的归一化 x（0~1）
    y_from / y_to     : 起始/目标车道 y 坐标
    返回 (x_array, y_array)
    """
    x = np.linspace(0.0, 1.0, n)
    t = (x - lc_start) / (lc_end - lc_start)
    y = y_from + (y_to - y_from) * min_jerk(t)
    return x, y


def make_npc_xs(decel_start=None, decel_factor=0.42, n=900):
    """
    生成 NPC 的纵向 x 序列。
    若 decel_start 不为 None，则在该位置之后减速，
    模拟保守型 NPC 让行（x 推进变慢）。
    """
    raw = np.linspace(0.0, 1.0, n)
    if decel_start is None:
        return raw
    xs = raw.copy()
    after = raw > decel_start
    if np.any(after):
        t = (raw[after] - decel_start) / (1.0 - decel_start)
        local_speed = 1.0 - (1.0 - decel_factor) * min_jerk(t)
        ds = np.gradient(raw[after]) * local_speed
        progressed = np.cumsum(ds)
        xs[after] = decel_start + progressed - progressed[0]
    return xs


def heading_at(xs, ys, idx):
    """Estimate local heading angle in degrees for a trajectory point."""
    i0 = max(idx - 3, 0)
    i1 = min(idx + 3, len(xs) - 1)
    dx = xs[i1] - xs[i0]
    dy = ys[i1] - ys[i0]
    return np.degrees(np.arctan2(dy, dx))


def draw_car(ax, cx, cy, color, alpha=1.0, zorder=5, angle=0.0):
    """在 (cx, cy) 为中心画一个圆角车辆方块。"""
    rect = mpatches.FancyBboxPatch(
        (cx - CAR_W / 2, cy - CAR_H / 2), CAR_W, CAR_H,
        boxstyle="round,pad=0.006,rounding_size=0.010",
        fc=color, ec='white', lw=0.65,
        alpha=alpha, zorder=zorder)
    rect.set_transform(
        Affine2D().rotate_deg_around(cx, cy, angle) + ax.transData
    )
    rect.set_path_effects([
        pe.withStroke(linewidth=1.25, foreground='white', alpha=0.88),
        pe.SimplePatchShadow(offset=(0.45, -0.45), alpha=0.10),
        pe.Normal(),
    ])
    ax.add_patch(rect)


def draw_obstacle(ax, cx, cy):
    """画障碍物（黄色方块 + × 号）。"""
    rect = mpatches.FancyBboxPatch(
        (cx - CAR_W / 2, cy - CAR_H / 2), CAR_W, CAR_H,
        boxstyle="round,pad=0.008,rounding_size=0.009",
        fc=OBS_COLOR, ec='#555555', lw=1.0, zorder=7)
    ax.add_patch(rect)
    ax.text(cx, cy, 'x', ha='center', va='center',
            fontsize=6.5, color='#333', fontweight='bold', zorder=8)


def snap_along(ax, xs, ys, color, positions, zorder=5):
    """
    在轨迹 (xs, ys) 上，按 positions 列表中的归一化 x 位置
    画车辆快照方块，透明度从 0.35 渐变到 1.0。
    """
    n = len(positions)
    for k, xp in enumerate(positions):
        alpha = 0.18 + 0.66 * (k / max(n - 1, 1))
        idx = int(np.argmin(np.abs(xs - xp)))
        draw_car(ax, xs[idx], ys[idx], color, alpha=alpha,
                 zorder=zorder, angle=heading_at(xs, ys, idx))


def plot_trajectory(ax, xs, ys, color, lw=2.15, alpha=1.0,
                    zorder=3, dashed=False):
    """Draw a publication-style smooth trajectory with subtle halo and arrows."""
    if len(xs) < 2:
        return

    ax.plot(xs, ys, color='white', lw=lw + 2.6, alpha=0.86,
            solid_capstyle='round', zorder=zorder - 0.1)

    points = np.column_stack([xs, ys]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(
        segments,
        colors=[color],
        linewidths=lw,
        alpha=alpha,
        linestyles='--' if dashed else 'solid',
        capstyle='round',
        joinstyle='round',
        zorder=zorder,
    )
    ax.add_collection(lc)

    arrow_idx = np.linspace(int(0.28 * len(xs)), int(0.82 * len(xs)), 3).astype(int)
    for idx in arrow_idx:
        idx = int(np.clip(idx, 1, len(xs) - 2))
        ax.annotate(
            '',
            xy=(xs[idx + 1], ys[idx + 1]),
            xytext=(xs[idx - 1], ys[idx - 1]),
            arrowprops=dict(arrowstyle='-|>', color=color, lw=0,
                            mutation_scale=8.5, alpha=0.82),
            zorder=zorder + 0.2,
        )


def annotate(ax, text, xy, xytext, color):
    """带箭头的文字标注，白底圆角框。"""
    ax.annotate(
        text, xy=xy, xytext=xytext,
        fontsize=ANNOTATION_FONTSIZE, color=color, ha='center',
        arrowprops=dict(arrowstyle='->', color=color, lw=0.9),
        bbox=dict(fc='white', ec='#aaaaaa', pad=2.5,
                  boxstyle='round,pad=0.3'),
        zorder=10)


def setup_ax(ax, title):
    """设置坐标轴、车道背景与边框。"""
    ax.set_facecolor('white')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(LANE_BOT, LANE_TOP + 0.18)
    ax.axis('off')

    # 路面填充
    ax.fill_between(
        [-0.02, 1.02],
        [LANE_BOT] * 2,
        [LANE_TOP] * 2,
        color=ROAD_COLOR, zorder=0)
    ax.fill_between(
        [-0.02, 1.02],
        [LANE_Y_NPC - 0.36] * 2,
        [LANE_Y_NPC + 0.36] * 2,
        color='#eef2f4', zorder=0.2)

    # 上下边界实线
    for y in [LANE_BOT, LANE_TOP]:
        ax.plot([-0.02, 1.02], [y, y], color='#b6bdc3', lw=1.1, zorder=1)

    # 车道中间虚线
    mid_y = (LANE_Y_EGO + LANE_Y_NPC) / 2
    ax.plot([-0.02, 1.02], [mid_y, mid_y],
            color=LANE_COLOR, lw=0.9, ls=(0, (12, 9)), zorder=1)

    for y in [LANE_Y_EGO, LANE_Y_NPC]:
        ax.plot([-0.02, 1.02], [y, y],
                color='#c9ced3', lw=0.45, ls=(0, (2, 12)),
                alpha=0.42, zorder=1)

    ax.set_title(title, fontsize=TITLE_FONTSIZE,
                 fontweight='bold', pad=6)


# ══════════════════════════════════════════════════════
#  主绘图函数
# ══════════════════════════════════════════════════════

def draw_scenario(scenario='aggressive'):
    """
    scenario : 'aggressive' 或 'conservative'
    """
    assert scenario in ('aggressive', 'conservative'), \
        "scenario 须为 'aggressive' 或 'conservative'"

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H), sharey=True)
    plt.subplots_adjust(wspace=0.04)

    # ── 场景参数 ──────────────────────────────────────
    if scenario == 'aggressive':
        fig.suptitle('Ego Decision vs Aggressive NPC  (θ ≈ 15°)',
                     fontsize=12, fontweight='bold')
        npc_color  = '#e74c3c'
        npc_label  = 'Aggressive NPC (θ≈15°)'
        npc_xs     = make_npc_xs(decel_start=None)        # 匀速
        npc_ys     = np.full_like(npc_xs, LANE_Y_NPC)

        # Ours：自车等待 NPC 通过后变道（变道晚，在 NPC 后方）
        ours_lc     = (0.52, 0.72)
        ours_snaps  = [0.08, 0.30, 0.62, 0.86]
        ours_note   = ('Ego waits,\nmerges behind NPC',
                       (0.70, LANE_Y_NPC + 0.04),
                       (0.60, LANE_TOP - 0.05),
                       EGO_COLOR_OURS)

        # PPO：过早切入，截断
        ppo_lc      = (0.26, 0.42)
        ppo_cut     = 0.49
        ppo_snaps   = [0.08, 0.26, 0.40]
        ppo_note    = ('Collision',
                       None, None, EGO_COLOR_PPO)
        extra_note  = None

    else:  # conservative
        fig.suptitle('Ego Decision vs Conservative NPC  (θ ≈ 75°)',
                     fontsize=12, fontweight='bold')
        npc_color  = '#27ae60'
        npc_label  = 'Conservative NPC (θ≈75°)'
        npc_xs     = make_npc_xs(decel_start=0.22, decel_factor=0.40)
        npc_ys     = np.full(len(npc_xs), LANE_Y_NPC)

        # Ours：识别让行，提前切入，在 NPC 前方
        ours_lc     = (0.20, 0.38)
        ours_snaps  = [0.06, 0.28, 0.55, 0.86]
        ours_note   = ('Ego merges\nahead of NPC',
                       (0.36, LANE_Y_NPC - 0.05),
                       (0.20, LANE_TOP - 0.05),
                       EGO_COLOR_OURS)

        # PPO：迟疑，碰障碍物，截断
        ppo_lc      = (0.54, 0.68)
        ppo_cut     = 0.53
        ppo_snaps   = [0.08, 0.30, 0.44]
        ppo_note    = ('Hesitation\n→ Collision',
                       None, None, EGO_COLOR_PPO)
        extra_note  = ('NPC decelerates\nto yield',
                       (npc_xs[int(0.30 * len(npc_xs))], LANE_Y_NPC),
                       (0.42, LANE_TOP - 0.05),
                       '#1e8449')

    # ── 逐列绘制 ──────────────────────────────────────
    for col, (ax, col_title) in enumerate(
            zip(axes, ['SVO+K-Level (Ours)', 'Pure PPO'])):

        setup_ax(ax, col_title)
        draw_obstacle(ax, OBS_X, LANE_Y_EGO)

        # NPC 轨迹
        plot_trajectory(ax, npc_xs, npc_ys, npc_color,
                        lw=1.35, alpha=0.46, zorder=2, dashed=True)
        npc_snap_pos = [0.10, 0.34, 0.60]
        snap_along(ax, npc_xs, npc_ys, npc_color,
                   npc_snap_pos, zorder=4)

        if col == 0:
            # ── Ours ──
            ex, ey = smooth_lc(*ours_lc, LANE_Y_EGO, LANE_Y_NPC)
            plot_trajectory(ax, ex, ey, EGO_COLOR_OURS,
                            lw=2.2, alpha=0.95, zorder=3)
            snap_along(ax, ex, ey, EGO_COLOR_OURS,
                       ours_snaps, zorder=6)
            # 主标注
            txt, xy, xyt, c = ours_note
            annotate(ax, txt, xy, xyt, c)
            # 额外标注（仅保守型）
            if extra_note is not None:
                txt2, xy2, xyt2, c2 = extra_note
                annotate(ax, txt2, xy2, xyt2, c2)

        else:
            # ── PPO（碰撞截断）──
            ex, ey = smooth_lc(*ppo_lc, LANE_Y_EGO, LANE_Y_NPC)
            cut = int(ppo_cut * len(ex))
            ex, ey = ex[:cut], ey[:cut]
            plot_trajectory(ax, ex, ey, EGO_COLOR_PPO,
                            lw=2.2, alpha=0.95, zorder=3)
            snap_along(ax, ex, ey, EGO_COLOR_PPO,
                       ppo_snaps, zorder=6)
            # 碰撞标记
            ax.plot(ex[-1], ey[-1], 'x',
                    color=EGO_COLOR_PPO, ms=9, mew=2.2, zorder=11)
            txt, _, _, c = ppo_note
            annotate(ax,
                     txt,
                     (ex[-1], ey[-1]),
                     (ex[-1] + 0.14, ey[-1] + 0.30),
                     c)

    # ── 图例 ──────────────────────────────────────────
    legend_els = [
        mpatches.Patch(fc=EGO_COLOR_OURS, ec='#333',
                       label='Ego – Ours (success)'),
        mpatches.Patch(fc=EGO_COLOR_PPO,  ec='#333',
                       label='Ego – PPO (collision)'),
        mpatches.Patch(fc=npc_color,       ec='#333',
                       label=npc_label),
        mpatches.Patch(fc=OBS_COLOR,       ec='#555',
                       label='Obstacle'),
    ]
    fig.legend(handles=legend_els, loc='lower center', ncol=4,
               fontsize=LEGEND_FONTSIZE,
               bbox_to_anchor=(0.5, -0.04),
               frameon=True, edgecolor='#cccccc')
    plt.tight_layout(rect=[0, 0.10, 1, 1])
    return fig


# ══════════════════════════════════════════════════════
#  入口
# ══════════════════════════════════════════════════════

X_MIN, X_MAX = -408.0, -178.0
Y_MIN, Y_MAX = -5.8, 11.5
TIME_SLICES = [0, 2, 4, 6]
VEHICLE_L = 12.0
VEHICLE_W = 2.2


def metric_lane_y(name):
    lanes = {
        'lane0': 8.2,
        'lane1': 4.4,
        'lane2': 0.6,
        'lane3': -3.2,
    }
    return lanes[name]


def lane_change_y(t, start, end, y0, y1):
    return y0 + (y1 - y0) * min_jerk((t - start) / (end - start))


def metric_positions(scenario, t):
    passing_lane = metric_lane_y('lane0')
    blocked_lane = metric_lane_y('lane1')
    lane2 = metric_lane_y('lane2')
    lane3 = metric_lane_y('lane3')

    if scenario == 'aggressive':
        # Aggressive NPC occupies the passing lane, so ego delays the lane change
        # and merges behind it after the obstacle. The NPC keeps speed while
        # the ego eases off before committing to the lane change.
        ego_x = (-385.0 + 24.0 * t
                 - 10.0 * min_jerk(t / 3.8)
                 + 8.0 * min_jerk((t - 4.0) / 2.0))
        ego_y = lane_change_y(t, 3.7, 5.9, blocked_lane, passing_lane)
        npc_x = -365.0 + 25.0 * t
        npc_y = passing_lane
        veh2_x = -398.0 + 21.0 * t
        veh2_y = lane2
    else:
        # Conservative NPC yields in the upper lane. The ego first accelerates
        # past the NPC in its original lane, then starts the lane change only
        # after it has a clear longitudinal lead.
        ego_x = -385.0 + 34.0 * t - 9.0 * min_jerk(t / GIF_DURATION)
        ego_y = lane_change_y(t, 2.55, 4.30, blocked_lane, passing_lane)
        npc_x = -342.0 + 16.0 * t - 20.0 * min_jerk(t / GIF_DURATION)
        npc_y = passing_lane
        veh2_x = -400.0 + 20.2 * t
        veh2_y = lane3

    return {
        'ego': (ego_x, ego_y),
        'npc': (npc_x, npc_y),
        'veh2': (veh2_x, veh2_y),
        'obstacle': (-246.0, blocked_lane),
    }


def metric_path(scenario, key, t_now):
    ts = np.linspace(0.0, float(t_now), 120)
    xs, ys = [], []
    for t in ts:
        x, y = metric_positions(scenario, t)[key]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def format_time_label(t):
    text = f'{float(t):.1f}'.rstrip('0').rstrip('.')
    return f't = {text} s'


def draw_metric_vehicle(ax, x, y, color, label, alpha=0.88,
                        edge='#555555', zorder=5):
    rect = mpatches.FancyBboxPatch(
        (x - VEHICLE_L / 2, y - VEHICLE_W / 2),
        VEHICLE_L, VEHICLE_W,
        boxstyle="round,pad=0.04,rounding_size=0.25",
        fc=color, ec=edge, lw=0.8, alpha=alpha, zorder=zorder)
    ax.add_patch(rect)
    if label:
        ax.text(x, y, label, ha='center', va='center',
                fontsize=9.5, color='#1f1f1f', zorder=zorder + 1)


def draw_other_npcs(ax, scenario, t):
    lane0 = metric_lane_y('lane0')
    lane2 = metric_lane_y('lane2')
    lane3 = metric_lane_y('lane3')
    if scenario == 'aggressive':
        npc_specs = [
            (-360.0, 18.0, lane2, '#7fc6a4', '4'),
            (-300.0, 19.2, lane3, '#b39ddb', '5'),
        ]
    else:
        npc_specs = [
            (-358.0, 18.6, lane2, '#7fc6a4', '4'),
            (-312.0, 20.0, lane3, '#b39ddb', '5'),
        ]

    for x0, v, y, color, label in npc_specs:
        x = x0 + v * t
        if X_MIN + 4 < x < X_MAX - 4:
            draw_metric_vehicle(
                ax, x, y, color, label, alpha=0.70,
                edge='#667078', zorder=2)


def setup_metric_ax(ax, row_idx):
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_facecolor('white')
    ax.set_yticks([-5, 0, 5, 10])
    ax.set_ylabel('y [m]', fontsize=10)
    ax.tick_params(axis='both', labelsize=9, length=2.8, width=0.6)
    ax.grid(axis='x', color='#dddddd', lw=0.55, alpha=0.8)

    for spine in ax.spines.values():
        spine.set_linewidth(0.55)
        spine.set_color('#444444')

    for y in [metric_lane_y('lane3'), metric_lane_y('lane2'),
              metric_lane_y('lane1'), metric_lane_y('lane0')]:
        ax.plot([X_MIN, X_MAX], [y, y],
                color='#d6d6d6', lw=0.55, ls=(0, (2, 9)), zorder=0)

    for y in [6.3, 2.5, -1.3]:
        ax.plot([X_MIN, X_MAX], [y, y],
                color='#bcbcbc', lw=1.0, ls=(0, (6, 6)), zorder=0)

    ax.plot([X_MIN, X_MAX], [10.1, 10.1],
            color='black', lw=1.45, zorder=1)
    ax.plot([X_MIN, X_MAX], [-5.1, -5.1],
            color='black', lw=1.45, zorder=1)

    if row_idx < len(TIME_SLICES) - 1:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel('x [m]', fontsize=11)
        ax.set_xticks([-400, -350, -300, -250, -200])


def draw_time_panel(ax, scenario, row_idx, t, emphasize_ego=False):
    setup_metric_ax(ax, row_idx)
    draw_other_npcs(ax, scenario, t)
    data = metric_positions(scenario, t)

    colors = {
        'ego': '#ec407a',
        'npc': '#45a9df',
        'veh2': '#f29b72',
        'obstacle': '#ffd22e',
    }
    labels = {'ego': '0', 'npc': '1', 'veh2': '2', 'obstacle': 'x'}

    if t > 0:
        for key in ['ego', 'npc', 'veh2']:
            color = colors[key]
            xs, ys = metric_path(scenario, key, t)
            is_ego = key == 'ego'
            lw = 2.1 if emphasize_ego and is_ego else 1.2
            alpha = 0.92 if emphasize_ego and is_ego else 0.58
            ls = 'solid' if emphasize_ego and is_ego else (0, (4, 4))
            zorder = 4 if emphasize_ego and is_ego else 3
            ax.plot(xs, ys, color=color, lw=lw, ls=ls,
                    alpha=alpha, zorder=zorder)

    for key in ['obstacle', 'veh2', 'npc', 'ego']:
        x, y = data[key]
        draw_metric_vehicle(ax, x, y, colors[key], labels[key],
                            alpha=0.82, edge='#666666', zorder=5)

    ax.text(
        X_MAX - 36, Y_MIN + 1.45, format_time_label(t),
        fontsize=11, ha='left', va='center',
        bbox=dict(fc='white', ec='#bdbdbd', lw=0.8, pad=3.0),
        zorder=10)


def draw_scenario(scenario='aggressive'):
    """
    scenario : 'aggressive' or 'conservative'
    """
    assert scenario in ('aggressive', 'conservative'), \
        "scenario must be 'aggressive' or 'conservative'"

    fig, axes = plt.subplots(
        len(TIME_SLICES), 1,
        figsize=(7.2, 4.95),
        sharex=True,
        constrained_layout=False)

    title = {
        'aggressive': r'Aggressive NPC interaction ($\theta \approx 15^\circ$)',
        'conservative': r'Conservative NPC interaction ($\theta \approx 75^\circ$)',
    }[scenario]
    fig.suptitle(title, fontsize=11.5, fontweight='bold', y=0.99)

    for row_idx, (ax, t) in enumerate(zip(axes, TIME_SLICES)):
        draw_time_panel(ax, scenario, row_idx, t)

    legend_els = [
        mpatches.Patch(fc='#ec407a', ec='#666666', label='0: ego vehicle'),
        mpatches.Patch(fc='#45a9df', ec='#666666', label='1: decision NPC'),
        mpatches.Patch(fc='#f29b72', ec='#666666', label='2: surrounding NPC'),
        mpatches.Patch(fc='#ffd22e', ec='#666666', label='obstacle'),
    ]
    fig.legend(handles=legend_els, loc='lower center', ncol=4,
               fontsize=8.6, bbox_to_anchor=(0.5, -0.01),
               frameon=False)
    plt.subplots_adjust(left=0.10, right=0.985, top=0.93,
                        bottom=0.12, hspace=0.20)
    return fig


def draw_scenario_animation(scenario='aggressive'):
    """
    Build a continuous GIF-ready animation for PPT: the ego trajectory is
    emphasized while NPC traces remain visible as context.
    """
    assert scenario in ('aggressive', 'conservative'), \
        "scenario must be 'aggressive' or 'conservative'"

    fig, ax = plt.subplots(figsize=(7.6, 2.75), constrained_layout=False)
    title = {
        'aggressive': r'Aggressive NPC interaction ($\theta \approx 15^\circ$)',
        'conservative': r'Conservative NPC interaction ($\theta \approx 75^\circ$)',
    }[scenario]
    fig.suptitle(title, fontsize=11.5, fontweight='bold', y=0.98)

    legend_els = [
        mpatches.Patch(fc='#ec407a', ec='#666666', label='0: ego trajectory'),
        mpatches.Patch(fc='#45a9df', ec='#666666', label='1: decision NPC'),
        mpatches.Patch(fc='#f29b72', ec='#666666', label='2: surrounding NPC'),
        mpatches.Patch(fc='#ffd22e', ec='#666666', label='obstacle'),
    ]
    fig.legend(handles=legend_els, loc='lower center', ncol=4,
               fontsize=8.4, bbox_to_anchor=(0.5, -0.01),
               frameon=False)
    plt.subplots_adjust(left=0.09, right=0.985, top=0.86, bottom=0.24)

    times = np.linspace(0.0, GIF_DURATION, int(GIF_DURATION * GIF_FPS) + 1)

    def update(t):
        ax.clear()
        draw_time_panel(
            ax, scenario, len(TIME_SLICES) - 1, float(t),
            emphasize_ego=True)
        return []

    anim = FuncAnimation(
        fig, update, frames=times, interval=1000 / GIF_FPS,
        blit=False, repeat=True)
    return fig, anim


def save_animation_gif(scenario, fname):
    if not PillowWriter.isAvailable():
        raise RuntimeError(
            'GIF output needs Pillow. Install it with: pip install pillow')

    fig, anim = draw_scenario_animation(scenario)
    writer = PillowWriter(fps=GIF_FPS)
    anim.save(fname, writer=writer, dpi=GIF_DPI)
    plt.close(fig)


if __name__ == '__main__':
    for sc in ['aggressive', 'conservative']:
        fig = draw_scenario(sc)
        fname = f'traj_{sc}.{OUTPUT_FORMAT}'
        if OUTPUT_FORMAT == 'svg':
            fig.savefig(fname, format='svg',
                        bbox_inches='tight', facecolor='white')
        else:
            fig.savefig(fname, dpi=OUTPUT_DPI,
                        bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f'Saved: {fname}')

        if SAVE_GIF:
            gif_name = f'traj_{sc}.gif'
            save_animation_gif(sc, gif_name)
            print(f'Saved: {gif_name}')
