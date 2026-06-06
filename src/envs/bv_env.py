"""
bv_env.py -- L1 BV 训练专用环境

========================================================================
在完整训练流程中的位置:
  Stage 1 (本文件): L1 BV 训练
    场景: 直道车流 (无障碍车, 无变道任务)
    Ego  = 正在学习的 L1 BV (会被部署当作 BV)
    BV   = TM 控制 (= L0)
    Reward: BV 巡航奖励 (速度+不撞+车道保持-变道惩罚-转向不平稳)

  Stage 2 (svo_pretrain): SVO 预训练 (不变)

  Stage 3 (train.py --stage l2): L2 ego 训练
    场景: ego 任务 (变道/避障/到达)
    Ego  = 正在学习的 L2 ego
    BV   = 加载 Stage 1 训出的 L1 BV 权重
    obs  = 含 a_BV 拼接 (Level-k best-response 接口)
========================================================================

设计要点 (参考 Bouton 2020 keep-lane agent + CHARMS L1):
  1. 任务和 ego 完全不同: BV 不知道目标终点, 不需要变道避障
  2. Reward 鼓励: 巡航速度 + 车道保持 + 转向平稳
  3. Reward 惩罚: 碰撞 (重) + 变道 (轻微) + 偏离车道 (终止级)
  4. 终止条件: 碰撞 / 偏离车道 / 走到路尽头
  5. 不需要 SVO (BV 不推断别人的风格)

继承 CarlaEnv, 只覆盖 reset/step 中和"任务"绑定的部分,
观测构建 (_get_observation) 完全复用 — 这样训出来的策略
和 L2 阶段当 BV 用时, 看到的观测格式完全一致.

使用方法 (在 train.py --stage l1 里):
    env = make_bv_env(config, seed=42)
    obs = env.reset()
    for _ in range(N):
        action = agent.select_action(obs)
        next_obs, reward, done, info = env.step(action)
        ...
"""

import math
import random
from typing import Dict, Tuple, Any

import numpy as np

try:
    import carla
except ImportError:
    pass

from src.config import Config, get_default_config
from src.envs.carla_env import CarlaEnv


# ============================================================================
# BV 训练专用环境
# ============================================================================

class BVTrainEnv(CarlaEnv):
    """L1 BV 训练环境.

    继承 CarlaEnv, 主要差异:
      - 强制关闭障碍车 (BV 不学避障)
      - reset 时不规划"目标变道路径", 直接用当前车道前向作为参考
      - reward 用 BVRewardConfig (BVRewardConfig 在 config.py 里)
      - 终止条件: 碰撞 / 严重偏离车道 / 路径走完 / 超时
      - 不调 SVO 模块
    """

    def __init__(self, config: Config = None):
        if config is None:
            config = get_default_config()

        # 强制覆盖几个关键 flag - BV 训练专用
        config.scenario.enable_obstacles = False        # 不放障碍车
        config.scenario.scenario_type = "highway"       # 直道场景
        config.svo.enabled = False                      # 不用 SVO
        config.klevel.bv_control_mode = "tm"            # 周围全部 TM
        config.reward.use_oracle_svo = False            # 用不到

        # 关键: BV 训练时 obs 不应含 a_BV 拼接 (BV 自己不需要看别的 BV 的动作)
        # 但 obs 维度必须和 L2 时一致 (因为加载到 LevelKController 时网络维度要匹配)
        # 解决: obs 里的 bv_actions 段在 L1 训练时全填 0, 网络见到全 0 自然不依赖它
        # 所以 use_bv_actions_in_obs 保持 True, _get_observation 里检测当前是否为 BV 训练
        # 通过 self._is_bv_training_mode flag 控制是否填 0
        self._is_bv_training_mode = True

        super().__init__(config)

        # 进度跟踪 (用于 reset 时判断"走到路尽头")
        self._bv_total_distance = 0.0
        self._bv_last_loc = None

        # 变道检测
        self._bv_last_lane_id = None
        self._bv_lane_change_count = 0

        # 上一步转向 (用于平滑性 reward)
        self._bv_prev_steer = 0.0

        print("[BV-Env] L1 BV 训练环境初始化完成")
        print(f"  场景: {config.scenario.scenario_type} (直道)")
        print(f"  障碍车: 已禁用")
        print(f"  SVO: 已禁用")
        print(f"  BV 模式: {config.klevel.bv_control_mode} (周围全部 TM)")
        print(f"  Reward: BVRewardConfig (target_speed={config.bv_reward.target_speed} km/h)")

    # ====================================================================== #
    #  Reset                                                                  #
    # ====================================================================== #

    def reset(self) -> np.ndarray:
        """BV 训练 reset.

        和原 CarlaEnv.reset 的差异:
          1. 不调用 _plan_route_unprotected_left_turn (BV 不做左转)
          2. _plan_route 走的是当前车道直行 (本来就是这样, 兼容)
          3. 不生成障碍车 (在 __init__ 已强制 enable_obstacles=False)
          4. 重置 BV 专用统计量

        其他逻辑 (spawn ego, spawn TM NPCs, setup sensors) 完全复用父类.
        """
        obs = super().reset()

        # 重置 BV 专用统计
        self._bv_total_distance = 0.0
        self._bv_last_loc = None
        self._bv_lane_change_count = 0
        self._bv_prev_steer = 0.0

        if self.ego_vehicle is not None:
            self._bv_last_loc = self.ego_vehicle.get_location()
            wp = self.map.get_waypoint(self._bv_last_loc, lane_type=carla.LaneType.Driving)
            if wp is not None:
                self._bv_last_lane_id = wp.lane_id

        return obs

    # ====================================================================== #
    #  Step (覆盖 reward 计算 + 终止逻辑)                                       #
    # ====================================================================== #

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """BV 训练 step.

        和原 CarlaEnv.step 的差异只有两处:
          - reward 用 _calculate_bv_reward
          - done 用 _check_bv_done

        其他 (apply_action, world.tick, get_observation) 完全复用父类.
        """
        # 1) 应用动作 (复用父类的连续动作映射)
        self._apply_action(action)

        # 2) 仿真 tick
        self.world.tick()

        # 3) 更新 NPC behavior agent (TM 模式下 noop)
        self.tick_npc_agents()

        # 4) 更新 waypoint 索引 (用于 _calculate_bv_reward 里的 lateral_offset)
        self._update_waypoint_index()

        # 5) 构建观测
        obs = self._get_observation()

        # 6) BV 专用 reward
        reward = self._calculate_bv_reward(action)

        # 7) BV 专用终止
        done, info = self._check_bv_done()

        # 8) 累积统计
        self.current_step += 1
        self.episode_reward = float(getattr(self, 'episode_reward', 0.0)) + reward

        # 9) 渲染 (与父类一致)
        if self.config.visual.enable:
            self.render()

        # 10) 更新 prev_steer (供下一步平滑性计算)
        try:
            self._bv_prev_steer = float(self.ego_vehicle.get_control().steer)
        except Exception:
            pass

        info['episode_reward'] = self.episode_reward
        info['step'] = self.current_step
        info['lane_change_count'] = self._bv_lane_change_count
        # [CVaR] BV 训练阶段 cvar_enabled=False, 这里仅占位保证 train.py 不 KeyError
        info.setdefault('cost_step', 0.0)
        info.setdefault('collision_step', 1 if info.get('collision', False) else 0)
        info.setdefault('lane_violation_step', 1 if info.get('lane_violation', False) else 0)

        return obs, float(reward), done, info

    # ====================================================================== #
    #  BV 专用 Reward                                                          #
    # ====================================================================== #

    def _calculate_bv_reward(self, action: np.ndarray) -> float:
        """L1 BV 训练 reward (和 ego 任务 reward 完全不同).

        组成:
          + 巡航速度 (鼓励 target_speed 附近)
          + 车道保持 (在 max_lateral_offset 内)
          + 时间存活
          - 速度过低
          - 碰撞 (重)
          - 偏离车道 (重)
          - 变道事件 (轻微, 允许但不鼓励)
          - 转向不平稳

        关键: 不含变道目标 reward, 不含路径终点 reward, 不含障碍避让 reward.
        """
        bvr = self.config.bv_reward
        reward = 0.0

        if self.ego_vehicle is None:
            return -1.0  # ego 没了, 给个小惩罚

        # ---------- 1) 速度 ----------
        velocity = self.ego_vehicle.get_velocity()
        speed_kmh = math.sqrt(velocity.x ** 2 + velocity.y ** 2) * 3.6

        # 巡航奖励: |speed - target| / target, 越接近越高
        speed_diff_norm = abs(speed_kmh - bvr.target_speed) / max(bvr.target_speed, 1e-6)
        reward += max(0.0, 1.0 - speed_diff_norm) * bvr.speed_reward_weight

        # 速度过低惩罚
        if speed_kmh < bvr.min_speed:
            reward += bvr.low_speed_penalty

        # ---------- 2) 车道保持 ----------
        ego_loc = self.ego_vehicle.get_location()
        target_wp = self._get_target_waypoint()
        if target_wp is None:
            target_wp = self.map.get_waypoint(ego_loc, lane_type=carla.LaneType.Driving)

        lateral_offset = 0.0
        if target_wp is not None:
            lateral_offset = self._calc_lateral_offset(ego_loc, target_wp)
            abs_lat = abs(lateral_offset)
            if abs_lat < bvr.max_lateral_offset:
                reward += (1.0 - abs_lat / bvr.max_lateral_offset) * bvr.lane_keeping_weight
            else:
                # 软惩罚: 偏出 max_lateral_offset 但还没到 terminal_lateral_offset
                # 这一段允许 BV 短暂偏离 (比如博弈避让), 但不鼓励
                reward -= (abs_lat - bvr.max_lateral_offset) * 0.2

        # ---------- 3) 安全 (前车距离) ----------
        front_dist = self._get_front_distance()
        if front_dist < bvr.min_safe_distance:
            reward -= (bvr.min_safe_distance - front_dist) / max(bvr.min_safe_distance, 1e-6) * 0.5
        elif front_dist < bvr.min_safe_distance * 2:
            reward += bvr.near_front_penalty

        # ---------- 4) 变道惩罚 (核心: 和 ego reward 的最大区别) ----------
        # 检测车道变化: lane_id 变了 = 变道
        try:
            current_wp = self.map.get_waypoint(ego_loc, lane_type=carla.LaneType.Driving)
            if current_wp is not None:
                current_lane_id = current_wp.lane_id
                if (self._bv_last_lane_id is not None
                        and current_lane_id != self._bv_last_lane_id):
                    reward += bvr.lane_change_penalty
                    self._bv_lane_change_count += 1
                self._bv_last_lane_id = current_lane_id
        except Exception:
            pass

        # ---------- 5) 转向平滑 ----------
        try:
            current_steer = float(self.ego_vehicle.get_control().steer)
            steer_delta = abs(current_steer - self._bv_prev_steer)
            reward -= steer_delta * bvr.steering_smoothness_weight
            # 大幅转向直接惩罚 (BV 不应做激烈机动)
            reward -= abs(current_steer) * bvr.steering_penalty_weight * 0.1
        except Exception:
            pass

        # ---------- 6) 碰撞 (终止级) ----------
        if self.collision_history:
            reward += bvr.collision_penalty

        # ---------- 7) 时间存活 ----------
        reward += bvr.time_alive_reward

        return reward

    # ====================================================================== #
    #  BV 专用终止                                                              #
    # ====================================================================== #

    def _check_bv_done(self) -> Tuple[bool, Dict[str, Any]]:
        """BV 训练终止条件.

        终止类型:
          1. 碰撞 (collision)
          2. 严重偏离车道 (lane_violation, 偏离 > terminal_lateral_offset)
          3. 路径走完 (route_finished, 接近 route 末尾)
          4. 超时 (timeout, 超过 max_episode_steps)
        """
        bvr = self.config.bv_reward
        info = {
            'collision': False,
            'lane_violation': False,
            'route_finished': False,
            'timeout': False,
            'reached_goal': False,  # BV 没有目标, 永远 False, 保留兼容性
            'max_steps': False,
        }

        if self.ego_vehicle is None:
            info['collision'] = True
            return True, info

        # 1) 碰撞
        if self.collision_history:
            info['collision'] = True
            return True, info

        # 2) 严重偏离车道
        ego_loc = self.ego_vehicle.get_location()
        target_wp = self._get_target_waypoint()
        if target_wp is None:
            target_wp = self.map.get_waypoint(ego_loc, lane_type=carla.LaneType.Driving)
        if target_wp is not None:
            lat = abs(self._calc_lateral_offset(ego_loc, target_wp))
            if lat > bvr.terminal_lateral_offset:
                info['lane_violation'] = True
                return True, info

        # 3) 路径走完 (接近 route 末尾)
        if self.route_waypoints and len(self.route_waypoints) > 0:
            if self.current_waypoint_idx >= len(self.route_waypoints) - 3:
                info['route_finished'] = True
                return True, info

        # 4) 超时
        if self.current_step >= self.config.train.max_episode_steps:
            info['timeout'] = True
            info['max_steps'] = True
            return True, info

        return False, info

    # ====================================================================== #
    #  覆盖 _get_observation 中的 bv_actions 段 (L1 训练时填 0)                  #
    # ====================================================================== #

    # 注意: _get_observation 在父类 carla_env.py 里, 我们通过 _is_bv_training_mode
    # flag 让父类判断当前是否需要填 0. 父类那边的实现会在第 2 次 patch 里处理.


# ============================================================================
# 工厂函数
# ============================================================================

def make_bv_env(
    config: Config = None,
    seed: int = None,
    town: str = None,
    spawn_point_index: int = None,
    num_npc: int = None,
    target_speed_kmh: float = None,
) -> BVTrainEnv:
    """L1 BV 训练环境工厂函数.

    Args:
        config: 配置对象, 若 None 则用 get_default_config()
        seed:   随机种子
        town:   CARLA town 名 (如 'Town04', 'Town06')
        spawn_point_index:  spawn 点索引 (从 world.get_map().get_spawn_points() 选)
        num_npc: TM 控制的 NPC 数量
        target_speed_kmh:  BV 目标巡航速度

    使用示例:
        # 默认参数
        env = make_bv_env(seed=42)

        # 自定义场景
        env = make_bv_env(
            seed=42,
            town='Town04',
            spawn_point_index=105,
            num_npc=30,
            target_speed_kmh=70.0,
        )
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    if config is None:
        config = get_default_config()

    # 应用覆盖参数
    if town is not None:
        config.carla.town = town
    if spawn_point_index is not None:
        config.scenario.spawn_point_index = spawn_point_index
        config.scenario.use_fixed_spawn = True
    if num_npc is not None:
        config.traffic.num_npc_vehicles = num_npc
    if target_speed_kmh is not None:
        config.bv_reward.target_speed = target_speed_kmh
        # 同时更新 reward.target_speed (避免父类某些路径用错)
        config.reward.target_speed = target_speed_kmh

    return BVTrainEnv(config)
