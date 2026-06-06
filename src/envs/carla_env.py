"""
CARLA PPO自动驾驶项目 - 环境代码

========================================================================
在完整训练流程中的位置:
  本文件定义CarlaEnv, 被 train.py, test.py, data_collector.py 调用.
  训练: python train.py --stage l1 / svo_only / l2
  验证: python test.py  --l1_model ... --svo_model ... --l2_model ...
  数据: python data_collector.py --episodes 300
========================================================================

特性:
- 静止车辆作为障碍物，放在当前车道正前方
- 参考路径保持直行（不做换道规划）
- RL自主学习换道避障
- 偏离5米以上才终止，支持短暂换道
CARLA Version: 0.9.14
"""

import sys
import os
import glob
import time
import random
import math
import numpy as np
from typing import Dict, Tuple, List, Any
from collections import deque
import gym
from gym import spaces

try:
    sys.path.append(glob.glob('/opt/carla-simulator/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

try:
    import pygame
    from pygame.locals import K_ESCAPE, K_p
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

from src.config import Config


class CarlaEnv(gym.Env):
    """CARLA强化学习环境 - 换道避障场景"""

    def __init__(self, config: Config = None):
        super(CarlaEnv, self).__init__()
        self.config = config if config else Config()

        # CARLA
        self.client = None
        self.world = None
        self.map = None
        self.traffic_manager = None
        self.debug = None

        # Actors
        self.ego_vehicle = None
        self.npc_vehicles = []
        self.obstacle_vehicle = None  # 静止障碍物车辆
        self.sensors = {}
        self.actor_list = []

        # 路径
        self.route_waypoints = []
        self.current_waypoint_idx = 0
        self.spawn_transform = None

        # 状态
        self.collision_history = []
        self.current_step = 0
        self.episode_start_time = 0
        self.total_reward = 0
        self.prev_steer = 0
        self.episode_count = 0

        # 可视化
        self.display = None
        self.clock = None
        self.camera_sensor = None
        self.camera_image = None
        self.font = None
        self.font_large = None
        self.font_small = None

        # 统计
        self.reward_history = deque(maxlen=100)
        self.speed_history = deque(maxlen=200)
        self.episode_rewards = deque(maxlen=50)
        self.collision_count = 0
        self.current_action = np.array([0.0, 0.0])
        self.paused = False

        # ===== Reward显示相关（瞬时/当前回合均值/上一回合均值）=====
        self.last_reward = 0.0
        self.current_episode_reward_sum = 0.0
        self.current_episode_steps = 0
        self.prev_episode_avg_reward = 0.0

        # ===== 成功/近失统计（按“终止原因”定义近失）=====
        self.success_count = 0
        self.near_miss_count = 0  # 非碰撞、非成功的所有终止都计为近失
        self._prev_ego_wp = None
        self._prev_lane_id = None
        self._ep_lane_change_left = 0
        self._ep_lane_change_right = 0
        self._ep_lane_change_unknown = 0
        self._step_waypoint_progress = 0
        self._last_applied_steer = 0.0

        # ===== 历史轨迹缓冲区 (Transformer编码器需要) =====
        self._history_T = self.config.encoder.history_steps
        self.ego_history = deque(maxlen=self._history_T)
        self.npc_histories = {}
        self.obstacle_history = deque(maxlen=self._history_T)

        # ===== [SVO-Game] 全量轨迹记录 (data_collector用) =====
        self.enable_trajectory_recording = False  # 默认关闭, data_collector开启
        self._episode_ego_states = []             # 每步ego状态 [(5,), ...]
        self._episode_npc_states = {}             # {npc_id: [(5,), ...]} 每步NPC状态
        self._npc_style_labels = {}               # {npc_id: 'aggressive'/.../'conservative'}

        # [SVO-Game] BehaviorAgent模式 (data_collector启用, 训练时关闭)
        self.use_behavior_agents = False          # 默认关闭, 用TM
        self._npc_agents = {}                     # {npc_id: BehaviorAgent}
        self._force_npc_style = None              # [NEW] 强制NPC风格 (test.py用)

        # [Level-k] BV策略控制器 (L2训练/测试时启用)
        self.klevel_controller = None             # LevelKController实例
        self._bv_control_mode = "tm"              # "tm" / "level1"

        # ================================================================== #
        # [v8] MandLC 相关状态已全部删除. 仅保留 SVO 推断相关变量.
        # ================================================================== #

        self._setup_spaces()
        self._connect_carla()

        if self.config.visual.enable and PYGAME_AVAILABLE:
            self._init_pygame()

    def _setup_spaces(self):
        obs_dim = self.config.encoder.total_obs_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

    def _connect_carla(self):
        cfg = self.config.carla
        print(f"连接CARLA {cfg.host}:{cfg.port}...")
        self.client = carla.Client(cfg.host, cfg.port)
        self.client.set_timeout(cfg.timeout)

        self.world = self.client.load_world(cfg.town)
        self.map = self.world.get_map()
        self.debug = self.world.debug

        settings = self.world.get_settings()
        settings.synchronous_mode = cfg.synchronous_mode
        settings.fixed_delta_seconds = cfg.fixed_delta_seconds
        self.world.apply_settings(settings)

        self.traffic_manager = self.client.get_trafficmanager()
        self.traffic_manager.set_synchronous_mode(cfg.synchronous_mode)

        self.world.set_weather(carla.WeatherParameters.ClearNoon)
        print(f"已连接, 地图: {cfg.town}")

        spawn_points = self.map.get_spawn_points()
        print(f"可用spawn points: {len(spawn_points)}")

    def _init_pygame(self):
        pygame.init()
        pygame.font.init()
        self.display = pygame.display.set_mode(
            (self.config.visual.width, self.config.visual.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        pygame.display.set_caption("CARLA PPO - Lane Change Learning")
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)
        print("Pygame可视化已启用 (P-暂停, ESC-退出)")

    def _get_spawn_transform(self) -> carla.Transform:
        """获取出生点"""
        scenario_cfg = self.config.scenario
        spawn_points = self.map.get_spawn_points()
        if not spawn_points:
            raise RuntimeError("地图没有可用的spawn points")

        if scenario_cfg.use_fixed_spawn and scenario_cfg.spawn_point_index >= 0: # 使用固定出生点
            idx = scenario_cfg.spawn_point_index % len(spawn_points)
            transform = spawn_points[idx]
        else:
            transform = random.choice(spawn_points)
        return transform

    def _spawn_ego_vehicle(self) -> bool:
        """生成主车 - 红色，出生点在spawn_transform"""
        bp_lib = self.world.get_blueprint_library()
        vehicle_bp = bp_lib.filter(self.config.carla.ego_vehicle_filter)[0]

        if vehicle_bp.has_attribute('color'):
            vehicle_bp.set_attribute('color', '255,0,0')

        self.spawn_transform = self._get_spawn_transform()

       
        original_loc = self.spawn_transform.location#获取原始出生位置
        wp = self.map.get_waypoint(original_loc) #获取原始出生位置的waypoint
        
        

        for attempt in range(5):
            try:
                self.ego_vehicle = self.world.spawn_actor(vehicle_bp, self.spawn_transform)
                self.actor_list.append(self.ego_vehicle)
                loc = self.spawn_transform.location
                
                # [NEW] 记录出生车道ID, 用于限制变道范围
                spawn_wp = self.map.get_waypoint(loc)
                self._spawn_lane_id = spawn_wp.lane_id if spawn_wp else 0
                print(f"主车(红色)生成于: ({loc.x:.1f}, {loc.y:.1f}), lane_id={self._spawn_lane_id}")
                if spawn_wp is not None:
                    left_wp = spawn_wp.get_left_lane()
                    right_wp = spawn_wp.get_right_lane()
                    left_id = left_wp.lane_id if (left_wp and left_wp.lane_type == carla.LaneType.Driving) else None
                    right_id = right_wp.lane_id if (right_wp and right_wp.lane_type == carla.LaneType.Driving) else None
                    print(f"[LaneCheck] left_driving_lane={left_id}, right_driving_lane={right_id}")
                return True
            except Exception:
                self.spawn_transform.location.z += 0.5
                time.sleep(0.1)

        return False

    def _plan_route(self):
        """规划路径 - 沿当前车道直行，不换道""" #路径用于计算 奖励函数
        scenario_cfg = self.config.scenario
        start_wp = self.map.get_waypoint(self.spawn_transform.location)
        if start_wp is None:
            print("警告: 无法获取起始waypoint")
            return

        spacing = scenario_cfg.waypoint_spacing
        self.route_waypoints = [start_wp]
        current_wp = start_wp

        num_waypoints = int(scenario_cfg.route_length / spacing)
        for _ in range(num_waypoints):
            next_wps = current_wp.next(spacing)
            if not next_wps:
                break
            current_wp = next_wps[0]
            self.route_waypoints.append(current_wp)

        self.current_waypoint_idx = 0
        print(f"路径规划完成: {len(self.route_waypoints)} waypoints (直行)")

    def _plan_route_unprotected_left_turn(self):
        """
        无保护左转路线:
        - 以出生点车道为起点
        - 在遇到分支时优先选择“向左”转角最大的分支
        """
        scenario_cfg = self.config.scenario
        start_wp = self.map.get_waypoint(self.spawn_transform.location, lane_type=carla.LaneType.Driving)
        if start_wp is None:
            print("[LeftTurn] 警告: 无法获取起始waypoint，回退直行路线")
            self._plan_route()
            return

        spacing = scenario_cfg.waypoint_spacing
        self.route_waypoints = [start_wp]
        current_wp = start_wp
        num_waypoints = int(scenario_cfg.route_length / spacing)

        left_turn_applied = False
        for _ in range(num_waypoints):
            next_wps = current_wp.next(spacing)
            if not next_wps:
                break

            chosen_wp = next_wps[0]
            if (not left_turn_applied) and len(next_wps) > 1:
                curr_yaw = current_wp.transform.rotation.yaw
                best_delta = -1e9
                for cand in next_wps:
                    delta = self._normalize_angle(cand.transform.rotation.yaw - curr_yaw)
                    if delta > best_delta:
                        best_delta = delta
                        chosen_wp = cand
                if best_delta > 5.0:
                    left_turn_applied = True

            current_wp = chosen_wp
            self.route_waypoints.append(current_wp)

        self.current_waypoint_idx = 0
        flag = "左转分支已命中" if left_turn_applied else "未检测到明显左转分支(按主车道推进)"
        print(f"[LeftTurn] 路径规划完成: {len(self.route_waypoints)} waypoints, {flag}")

    def _find_active_junction_center(self, start_wp):
        """
        沿前向搜索最近路口中心点.
        返回 carla.Location 或 None.
        """
        if start_wp is None:
            return None

        cfg = self.config.scenario
        step = max(1.0, float(cfg.waypoint_spacing))
        max_dist = float(getattr(cfg, "left_turn_junction_search_dist", 90.0))

        wp = start_wp
        traveled = 0.0
        while wp is not None and traveled <= max_dist:
            if wp.is_junction:
                try:
                    junc = wp.get_junction()
                    return junc.bounding_box.location
                except Exception:
                    return wp.transform.location
            next_wps = wp.next(step)
            if not next_wps:
                break
            wp = next_wps[0]
            traveled += step
        return None

    @staticmethod
    def _direction_key_from_yaw(yaw_deg: float) -> str:
        """根据车道朝向粗分四个方向(N/S/E/W)."""
        yaw_rad = math.radians(yaw_deg)
        vx = math.cos(yaw_rad)
        vy = math.sin(yaw_rad)
        if abs(vx) >= abs(vy):
            return 'E' if vx >= 0 else 'W'
        return 'N' if vy >= 0 else 'S'

    def _spawn_npc_vehicles_unprotected_left_turn(self):
        """
        无保护左转场景NPC生成:
        - 以目标路口中心为基准
        - 在四个方向都生成来车
        - 默认忽略信号灯, 禁止随机变道
        """
        if self.config.traffic.num_npc_vehicles <= 0:
            return

        ego_loc = self.ego_vehicle.get_location()
        ego_wp = self.map.get_waypoint(ego_loc, lane_type=carla.LaneType.Driving)
        if ego_wp is None:
            print("[LeftTurn] 无法获取ego waypoint，回退普通NPC生成")
            self._spawn_npc_vehicles()
            return

        junc_center = self._find_active_junction_center(ego_wp)
        if junc_center is None:
            print("[LeftTurn] 未找到目标路口，回退普通NPC生成")
            self._spawn_npc_vehicles()
            return

        cfg = self.config.scenario
        r_min = float(getattr(cfg, "left_turn_spawn_radius_min", 18.0))
        r_max = float(getattr(cfg, "left_turn_spawn_radius_max", 85.0))
        per_dir = int(getattr(cfg, "left_turn_npc_per_direction", 0))
        target_total = int(self.config.traffic.num_npc_vehicles)
        if per_dir <= 0:
            per_dir = max(1, target_total // 4)
        target_total = min(target_total, per_dir * 4)

        bp_lib = self.world.get_blueprint_library()
        vehicle_bps = [
            bp for bp in bp_lib.filter(self.config.carla.npc_vehicle_filter)
            if int(bp.get_attribute('number_of_wheels')) >= 4
        ]

        bins = {'N': [], 'S': [], 'E': [], 'W': []}
        for sp in self.map.get_spawn_points():
            d = sp.location.distance(junc_center)
            if d < r_min or d > r_max:
                continue
            if sp.location.distance(ego_loc) < 14.0:
                continue

            wp = self.map.get_waypoint(sp.location, lane_type=carla.LaneType.Driving)
            if wp is None:
                continue
            key = self._direction_key_from_yaw(wp.transform.rotation.yaw)
            bins[key].append(sp)

        for k in bins:
            random.shuffle(bins[k])

        STYLE_DEFS = [
            ('aggressive', 0.15),
            ('semi_aggressive', 0.20),
            ('normal', 0.30),
            ('semi_conservative', 0.20),
            ('conservative', 0.15),
        ]
        style_names = [s[0] for s in STYLE_DEFS]
        style_probs = [s[1] for s in STYLE_DEFS]

        force_style = getattr(self, '_force_npc_style', None)
        count = 0
        used_locs = []

        def _spawn_one(sp_transform) -> bool:
            nonlocal count
            for loc in used_locs:
                if sp_transform.location.distance(loc) < 8.0:
                    return False

            bp = random.choice(vehicle_bps)
            if bp.has_attribute('color'):
                bp.set_attribute('color', random.choice(bp.get_attribute('color').recommended_values))

            try:
                t = carla.Transform(sp_transform.location, sp_transform.rotation)
                t.location.z += 0.5
                npc = self.world.spawn_actor(bp, t)
                self.npc_vehicles.append(npc)
                self.actor_list.append(npc)
                used_locs.append(sp_transform.location)
                count += 1

                if force_style is not None:
                    style_label = force_style
                else:
                    style_label = np.random.choice(style_names, p=style_probs)
                self._npc_style_labels[npc.id] = style_label

                npc.set_autopilot(True, self.traffic_manager.get_port())
                self._apply_tm_style(npc, style_label, allow_lane_change=False)
                return True
            except Exception:
                return False

        # 先保证四个方向都有车
        for direction in ['N', 'S', 'E', 'W']:
            spawned_dir = 0
            for sp in bins[direction]:
                if count >= target_total or spawned_dir >= per_dir:
                    break
                if _spawn_one(sp):
                    spawned_dir += 1

        # 若某些方向候选不足, 从剩余池补齐总数
        if count < target_total:
            remain = []
            for direction in ['N', 'S', 'E', 'W']:
                remain.extend(bins[direction])
            random.shuffle(remain)
            for sp in remain:
                if count >= target_total:
                    break
                _spawn_one(sp)

        print(
            f"[LeftTurn] 生成 {count}/{target_total} 辆NPC "
            f"(四方向目标每向{per_dir}辆, 忽略灯控={self.config.traffic.ignore_lights_percentage:.0f}%)"
        )

    def _spawn_obstacle_vehicle(self):
        """生成静止障碍物车辆 - 距离在 [30, 80] 米间均匀分布"""
        if not self.config.scenario.enable_obstacles:
            return

        bp_lib = self.world.get_blueprint_library()
        obstacle_filter = self.config.scenario.obstacle_vehicle_filter
        obstacle_bps = bp_lib.filter(obstacle_filter)
        if not obstacle_bps: obstacle_bps = bp_lib.filter('vehicle.*')

        obstacle_bp = obstacle_bps[0]
        if obstacle_bp.has_attribute('color'):
            obstacle_bp.set_attribute('color', '255,255,0')

        # === 核心修改：动态采样障碍物距离 ===
        dist_min, dist_max = self.config.scenario.obstacle_distance_range
        obstacle_dist = random.uniform(dist_min, dist_max)
        
        accumulated_dist = 0
        obstacle_wp = None

        for i in range(1, len(self.route_waypoints)):
            prev_wp = self.route_waypoints[i - 1]
            curr_wp = self.route_waypoints[i]
            dist = curr_wp.transform.location.distance(prev_wp.transform.location)
            accumulated_dist += dist
            if accumulated_dist >= obstacle_dist:
                obstacle_wp = curr_wp
                break

        if obstacle_wp is None:
            return

        obstacle_transform = obstacle_wp.transform
        obstacle_transform.location.z += 0.5

        try:
            self.obstacle_vehicle = self.world.spawn_actor(obstacle_bp, obstacle_transform)
            self.actor_list.append(self.obstacle_vehicle)
            self.obstacle_vehicle.set_autopilot(False)
            self.obstacle_vehicle.apply_control(carla.VehicleControl(
                throttle=0.0, steer=0.0, brake=1.0, hand_brake=True
            ))
            print(f"障碍物车辆(黄色)生成, 动态距离: {obstacle_dist:.1f}m")
        except Exception as e:
            pass

    def _setup_sensors(self):
        """设置传感器"""
        bp_lib = self.world.get_blueprint_library()

        collision_bp = bp_lib.find('sensor.other.collision')
        self.sensors['collision'] = self.world.spawn_actor(
            collision_bp, carla.Transform(), attach_to=self.ego_vehicle
        )
        self.sensors['collision'].listen(self._on_collision)
        self.actor_list.append(self.sensors['collision'])

        if self.config.visual.enable and PYGAME_AVAILABLE:
            camera_bp = bp_lib.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', str(self.config.visual.width))
            camera_bp.set_attribute('image_size_y', str(self.config.visual.height))
            camera_bp.set_attribute('fov', '100')

            camera_transform = carla.Transform(
                carla.Location(x=-self.config.visual.camera_distance, z=self.config.visual.camera_height),
                carla.Rotation(pitch=self.config.visual.camera_pitch)
            )
            self.camera_sensor = self.world.spawn_actor(
                camera_bp, camera_transform, attach_to=self.ego_vehicle
            )
            self.camera_sensor.listen(self._on_camera_image)
            self.actor_list.append(self.camera_sensor)

    def _on_collision(self, event):
        self.collision_history.append(event)
        self.collision_count += 1

    def _on_camera_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        rgb = array[:, :, :3][:, :, ::-1]
        self.camera_image = rgb

    def _spawn_npc_vehicles(self):
        """生成NPC — 限制在ego所在高速路段附近
 
        改进: 不再使用全图spawn点, 而是沿ego所在道路的前后方向,
        在当前车道和相邻车道上生成NPC, 确保每辆都在交互范围内.
 
        Town04高速公路通常是3-4车道, NPC只生成在这些车道上,
        不会跑到匝道、辅路、其他方向的道路上.
 
        生成策略:
          1. 从ego位置沿道路前方和后方各取waypoint
          2. 对每个waypoint, 获取所有平行车道的位置
          3. 在这些位置上生成NPC (跳过离ego/障碍车太近的)
          4. 数量控制在 num_npc_vehicles 以内
        """
        if self.config.traffic.num_npc_vehicles <= 0:
            return
 
        bp_lib = self.world.get_blueprint_library()
        vehicle_bps = [bp for bp in bp_lib.filter(self.config.carla.npc_vehicle_filter)
                       if int(bp.get_attribute('number_of_wheels')) >= 4]
 
        ego_loc = self.ego_vehicle.get_location()
        ego_wp = self.map.get_waypoint(ego_loc, lane_type=carla.LaneType.Driving)
        obstacle_loc = self.obstacle_vehicle.get_location() if self.obstacle_vehicle else ego_loc
 
        if ego_wp is None:
            print("[Warning] 无法获取ego waypoint, 回退到全图spawn")
            self._spawn_npc_vehicles_fallback()
            return
 
        # ---- 1. 沿道路收集候选生成位置 ----
        candidate_transforms = []
        same_lane_candidates = []
        ensure_ego_lane_traffic = bool(getattr(self.config.scenario, "ensure_ego_lane_traffic", False))
        min_ego_lane_npc = max(0, int(getattr(self.config.scenario, "min_ego_lane_npc", 0)))
 
        # 前方: 从ego前方40m开始, 每隔spawn_gap生成, 直到max_forward
        # 后方: 从ego后方20m开始, 每隔spawn_gap生成, 直到max_backward
        spawn_gap = 7.0        # NPC间距 (米), 避免太密
        max_forward = 50.0     # 前方最远距离
        max_backward = 100.0    # 后方最远距离
        min_dist_ego = 12.0     # 离ego最近距离 (避免出生就碰撞)
        min_dist_obstacle = 10.0  # 离障碍车最近距离
 
        # 前方waypoints
        wp = ego_wp
        dist_traveled = 0.0
        while dist_traveled < max_forward:
            next_wps = wp.next(spawn_gap)
            if not next_wps:
                break
            wp = next_wps[0]
            dist_traveled += spawn_gap

            if ensure_ego_lane_traffic:
                # wp 沿 ego 同车道推进，单独收集“同车道”候选，供优先生成
                self._try_add_candidate(
                    wp, same_lane_candidates,
                    ego_loc, obstacle_loc,
                    min_dist_ego, min_dist_obstacle
                )

            # 收集该位置所有平行车道
            self._collect_lane_positions(wp, candidate_transforms,
                                         ego_loc, obstacle_loc,
                                         min_dist_ego, min_dist_obstacle)
 
        # 后方waypoints
        wp = ego_wp
        dist_traveled = 0.0
        while dist_traveled < max_backward:
            prev_wps = wp.previous(spawn_gap)
            if not prev_wps:
                break
            wp = prev_wps[0]
            dist_traveled += spawn_gap

            if ensure_ego_lane_traffic:
                self._try_add_candidate(
                    wp, same_lane_candidates,
                    ego_loc, obstacle_loc,
                    min_dist_ego, min_dist_obstacle
                )

            self._collect_lane_positions(wp, candidate_transforms,
                                         ego_loc, obstacle_loc,
                                         min_dist_ego, min_dist_obstacle)

        # 打乱顺序增加随机性
        random.shuffle(candidate_transforms)
        random.shuffle(same_lane_candidates)

        if ensure_ego_lane_traffic and same_lane_candidates:
            # 先保证一定数量的同车道车辆，再混入相邻车道车辆
            keep_same_lane = min(min_ego_lane_npc, len(same_lane_candidates))
            fixed_same_lane = same_lane_candidates[:keep_same_lane]
            mixed_pool = same_lane_candidates[keep_same_lane:] + candidate_transforms
            random.shuffle(mixed_pool)
            spawn_candidates = fixed_same_lane + mixed_pool
        else:
            spawn_candidates = candidate_transforms

        # ---- 2. 生成NPC ----
        # 5种驾驶风格及其概率
        STYLE_DEFS = [
            ('aggressive',       0.15),
            ('semi_aggressive',  0.20),
            ('normal',           0.30),
            ('semi_conservative',0.20),
            ('conservative',     0.15),
        ]
        style_names = [s[0] for s in STYLE_DEFS]
        style_probs = [s[1] for s in STYLE_DEFS]
        force_style = getattr(self, '_force_npc_style', None)
 
        count = 0
        target_count = self.config.traffic.num_npc_vehicles
 
        for sp_transform in spawn_candidates:
            if count >= target_count:
                break
 
            bp = random.choice(vehicle_bps)
            if bp.has_attribute('color'):
                bp.set_attribute('color', random.choice(
                    bp.get_attribute('color').recommended_values))
 
            try:
                # 稍微抬高生成位置避免和地面碰撞
                sp_transform.location.z += 0.5
                npc = self.world.spawn_actor(bp, sp_transform)
                self.npc_vehicles.append(npc)
                self.actor_list.append(npc)
                count += 1
 
                # 分配风格
                if force_style is not None:
                    style_label = force_style
                else:
                    style_label = np.random.choice(style_names, p=style_probs)
                self._npc_style_labels[npc.id] = style_label
 
                if self.use_behavior_agents:
                    self._setup_behavior_agent(npc, style_label)
                else:
                    npc.set_autopilot(True, self.traffic_manager.get_port())
                    self._apply_tm_style(npc, style_label)
 
            except Exception:
                continue
 
        mode = "BehaviorAgent" if self.use_behavior_agents else "TrafficManager"
        print(f"生成 {count}/{target_count} 辆NPC ({mode}, "
              f"路段限定, 前{max_forward}m后{max_backward}m)")
 
    def _collect_lane_positions(self, center_wp, candidates,
                                ego_loc, obstacle_loc,
                                min_dist_ego, min_dist_obstacle):
        """
        从一个waypoint出发, 收集它和所有平行车道的生成位置.
 
        遍历左/右车道, 只保留Driving类型的车道.
        跳过离ego或障碍车太近的位置.
        """
        # 当前车道
        """
        self._try_add_candidate(center_wp, candidates,
                                ego_loc, obstacle_loc,
                                min_dist_ego, min_dist_obstacle)
        """
        def _is_same_direction(adj_wp, ref_wp):
            if adj_wp is None:
                return False
            if adj_wp.lane_type != carla.LaneType.Driving:
                return False

            yaw_diff = abs(
                (adj_wp.transform.rotation.yaw - ref_wp.transform.rotation.yaw + 180) % 360 - 180
            )
            return yaw_diff < 90.0

        # 左相邻车道
        left_wp = center_wp.get_left_lane()
        if _is_same_direction(left_wp, center_wp):
            self._try_add_candidate(
                left_wp, candidates,
                ego_loc, obstacle_loc,
                min_dist_ego, min_dist_obstacle
            )

        # 右相邻车道
        right_wp = center_wp.get_right_lane()
        if _is_same_direction(right_wp, center_wp):
            self._try_add_candidate(
                right_wp, candidates,
                ego_loc, obstacle_loc,
                min_dist_ego, min_dist_obstacle
            )
        
        
 
    def _try_add_candidate(self, wp, candidates, ego_loc, obstacle_loc,
                        min_dist_ego, min_dist_obstacle):
        """检查距离约束后添加候选位置.
        
        改进: 用纵向距离过滤, 不用直线距离.
        这样相邻车道上和ego平齐的NPC不会被错误过滤.
        """
        loc = wp.transform.location
        
        # 纵向距离 (沿道路方向)
        dx = loc.x - ego_loc.x
        dy = loc.y - ego_loc.y
        longitudinal_dist = abs(dx * math.cos(math.radians(wp.transform.rotation.yaw)) 
                            + dy * math.sin(math.radians(wp.transform.rotation.yaw)))
        
        # 只过滤纵向太近的 (同车道前后太近会碰撞)
        # 横向不过滤 (相邻车道的车就是要近)
        if longitudinal_dist < min_dist_ego:
            return
        if loc.distance(obstacle_loc) < min_dist_obstacle:
            return
        # 候选点之间不要重叠
        for existing in candidates:
            if loc.distance(existing.location) < 8.0:
                return
        candidates.append(wp.transform)
 
    def _spawn_npc_vehicles_fallback(self):
        """回退方案: 全图spawn (原始逻辑)"""
        bp_lib = self.world.get_blueprint_library()
        vehicle_bps = [bp for bp in bp_lib.filter(self.config.carla.npc_vehicle_filter)
                       if int(bp.get_attribute('number_of_wheels')) >= 4]
        spawn_points = self.map.get_spawn_points()
        ego_loc = self.ego_vehicle.get_location()
        random.shuffle(spawn_points)
        count = 0
        for sp in spawn_points[:self.config.traffic.num_npc_vehicles]:
            if sp.location.distance(ego_loc) < 30:
                continue
            bp = random.choice(vehicle_bps)
            try:
                npc = self.world.spawn_actor(bp, sp)
                self.npc_vehicles.append(npc)
                self.actor_list.append(npc)
                npc.set_autopilot(True, self.traffic_manager.get_port())
                count += 1
            except Exception:
                continue
        print(f"[Fallback] 生成 {count} 辆NPC (全图spawn)")

    def _apply_tm_style(self, npc, style_label, allow_lane_change=True):
        """用Traffic Manager参数模拟驾驶风格 (训练用, 高效)"""
        tm = self.traffic_manager
        # 全局灯控/标志忽略开关
        try:
            tm.ignore_lights_percentage(npc, float(self.config.traffic.ignore_lights_percentage))
        except Exception:
            pass
        try:
            tm.ignore_signs_percentage(npc, float(self.config.traffic.ignore_signs_percentage))
        except Exception:
            pass

        if style_label == 'aggressive':
            tm.vehicle_percentage_speed_difference(npc, random.uniform(-20, -5))
            tm.distance_to_leading_vehicle(npc, random.uniform(2.5, 4.0))
            tm.auto_lane_change(npc, True if allow_lane_change else False)
            tm.random_left_lanechange_percentage(npc, 10 if allow_lane_change else 0)
            tm.random_right_lanechange_percentage(npc, 10 if allow_lane_change else 0)
        elif style_label == 'semi_aggressive':
            tm.vehicle_percentage_speed_difference(npc, random.uniform(-10, 5))
            tm.distance_to_leading_vehicle(npc, random.uniform(3.5, 5.0))
            tm.auto_lane_change(npc, True if allow_lane_change else False)
            tm.random_left_lanechange_percentage(npc, 5 if allow_lane_change else 0)
            tm.random_right_lanechange_percentage(npc, 5 if allow_lane_change else 0)
        elif style_label == 'normal':
            tm.vehicle_percentage_speed_difference(npc, random.uniform(-5, 10))
            tm.distance_to_leading_vehicle(npc, random.uniform(5.0, 7.0))
            tm.auto_lane_change(npc, (random.random() < 0.3) if allow_lane_change else False)
        elif style_label == 'semi_conservative':
            tm.vehicle_percentage_speed_difference(npc, random.uniform(10, 20))
            tm.distance_to_leading_vehicle(npc, random.uniform(7.0, 10.0))
            tm.auto_lane_change(npc, False)
        elif style_label == 'conservative':
            tm.vehicle_percentage_speed_difference(npc, random.uniform(20, 35))
            tm.distance_to_leading_vehicle(npc, random.uniform(10.0, 14.0))
            tm.auto_lane_change(npc, False)

    def _setup_behavior_agent(self, npc, style_label):
        """
        用CARLA官方BehaviorAgent控制NPC (数据收集用, 精确).

        5种风格基于CARLA官方3种 (Aggressive/Normal/Cautious) 扩展:
          aggressive:       官方Aggressive, max_speed调高
          semi_aggressive:  基于Aggressive, 略保守
          normal:           官方Normal
          semi_conservative:基于Cautious, 略激进
          conservative:     官方Cautious, max_speed调低
        """
        try:
            from src.agents.navigation.behavior_agent import BehaviorAgent
        except ImportError:
            # 如果CARLA agents包不可用, 回退到TM
            npc.set_autopilot(True, self.traffic_manager.get_port())
            self._apply_tm_style(npc, style_label)
            return

        # 映射到CARLA官方3种基础风格
        if style_label in ('aggressive', 'semi_aggressive'):
            base_behavior = 'aggressive'
        elif style_label in ('conservative', 'semi_conservative'):
            base_behavior = 'cautious'
        else:
            base_behavior = 'normal'

        agent = BehaviorAgent(npc, behavior=base_behavior)

        # 在官方基础上微调参数
        if style_label == 'aggressive':
            agent.behavior.max_speed = 100 # 更高的最高速度
            agent.behavior.safety_time = 1 # 更短的安全时间
            agent.behavior.min_proximity_threshold = 6 # 近车道更近
            agent.behavior.braking_distance = 3 # 更短的刹车距离
            agent.behavior.tailgate_counter = -1 # 更激进的跟车 (不保持安全距离)
        elif style_label == 'semi_aggressive':
            agent.behavior.max_speed = 80
            agent.behavior.safety_time = 2.5
            agent.behavior.min_proximity_threshold = 8 
            agent.behavior.braking_distance = 4
            agent.behavior.tailgate_counter = -1
        elif style_label == 'normal':
            agent.behavior.max_speed = 60
            agent.behavior.safety_time = 3
            agent.behavior.min_proximity_threshold = 10
            agent.behavior.braking_distance = 5
        elif style_label == 'semi_conservative':
            agent.behavior.max_speed = 45
            agent.behavior.safety_time = 4
            agent.behavior.min_proximity_threshold = 14
            agent.behavior.braking_distance = 7
        elif style_label == 'conservative':
            agent.behavior.max_speed = 30
            agent.behavior.safety_time = 5
            agent.behavior.min_proximity_threshold = 18
            agent.behavior.braking_distance = 9

        # 设置随机目的地
        agent.set_destination(random.choice(self.map.get_spawn_points()).location)
        self._npc_agents[npc.id] = agent

    def tick_npc_agents(self):
        """
        每步调用: 让BehaviorAgent控制的NPC执行一步决策.
        data_collector在world.tick()后调用此方法.
        训练时(use_behavior_agents=False)不需要调用.
        """
        if not self.use_behavior_agents:
            return

        dead_ids = []
        for npc_id, agent in self._npc_agents.items():
            npc = agent._vehicle
            if not npc.is_alive:
                dead_ids.append(npc_id)
                continue
            try:
                # BehaviorAgent到达目的地后重新设一个
                if agent.done():
                    agent.set_destination(random.choice(self.map.get_spawn_points()).location)
                control = agent.run_step()
                npc.apply_control(control)
            except Exception:
                dead_ids.append(npc_id)

        for npc_id in dead_ids:
            self._npc_agents.pop(npc_id, None)

    def _draw_waypoints(self):
        """在CARLA世界中绘制waypoints"""
        if not self.config.visual.render_waypoints or not self.route_waypoints:
            return

        visual_cfg = self.config.visual
        start_idx = max(0, self.current_waypoint_idx)
        end_idx = min(len(self.route_waypoints), start_idx + visual_cfg.num_waypoints_to_draw)

        for i in range(start_idx, end_idx):
            wp = self.route_waypoints[i]
            if i == self.current_waypoint_idx:
                color = carla.Color(r=255, g=255, b=0)
                size = 0.15
            else:
                color = carla.Color(r=0, g=255, b=0)
                size = 0.08

            self.debug.draw_point(
                wp.transform.location + carla.Location(z=0.3),
                size=size, color=color, life_time=0.1
            )

    def _update_waypoint_index(self):
        """更新当前waypoint索引"""
        if not self.route_waypoints or self.ego_vehicle is None:
            return

        ego_loc = self.ego_vehicle.get_location()
        min_dist = float('inf')
        closest_idx = self.current_waypoint_idx

        search_start = max(0, self.current_waypoint_idx - 2)
        search_end = min(len(self.route_waypoints), self.current_waypoint_idx + 15)

        for i in range(search_start, search_end):
            wp = self.route_waypoints[i]
            dist = ego_loc.distance(wp.transform.location)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        if closest_idx >= self.current_waypoint_idx - 1:
            self.current_waypoint_idx = closest_idx

    def _get_target_waypoint(self) -> carla.Waypoint:
        """获取当前目标waypoint"""
        if not self.route_waypoints:
            return self.map.get_waypoint(self.ego_vehicle.get_location())
        lookahead = 3
        target_idx = min(self.current_waypoint_idx + lookahead, len(self.route_waypoints) - 1)
        return self.route_waypoints[target_idx]

    def _find_lane_by_id(self, base_wp, target_lane_id):
        """
        从当前waypoint出发，在同方向相邻车道中寻找指定lane_id。
        只搜索少量左右相邻车道，避免跳到对向或无关车道。
        """
        if base_wp is None:
            return None

        if base_wp.lane_id == target_lane_id:
            return base_wp

        # 先向左搜
        wp = base_wp
        for _ in range(3):
            wp = wp.get_left_lane()
            if wp is None or wp.lane_type != carla.LaneType.Driving:
                break
            if wp.lane_id == target_lane_id:
                return wp

        # 再向右搜
        wp = base_wp
        for _ in range(3):
            wp = wp.get_right_lane()
            if wp is None or wp.lane_type != carla.LaneType.Driving:
                break
            if wp.lane_id == target_lane_id:
                return wp

        return None
    # ==================================================================
    #  观测: 历史轨迹(1+N,T,5) + 地图路径(M,L,2) → flat vector
    #  坐标旋转在 Encoder 内部完成，这里输出全局坐标
    #  奖励函数、终止判断等完全不受影响
    # ==================================================================

    def _get_actor_state_raw(self, actor) -> List[float]:
        """actor的5维全局状态: 注入高斯噪声防过拟合"""
        loc = actor.get_location()
        vel = actor.get_velocity()
        yaw_rad = np.radians(actor.get_transform().rotation.yaw)
        yaw_rad = (yaw_rad + np.pi) % (2 * np.pi) - np.pi
        
        # 提取真实值
        state = [loc.x, loc.y, yaw_rad, vel.x, vel.y]
        
        # === 核心修改：注入观测噪声 ===
        if self.config.observation.add_obs_noise:
            cfg = self.config.observation
            state[0] += random.gauss(0, cfg.pos_noise_std) # X
            state[1] += random.gauss(0, cfg.pos_noise_std) # Y
            state[2] += random.gauss(0, cfg.yaw_noise_std) # Yaw
            state[3] += random.gauss(0, cfg.vel_noise_std) # Vx
            state[4] += random.gauss(0, cfg.vel_noise_std) # Vy
            
        return state

    def _get_actor_state_from_cache(self, loc, vel, yaw_deg) -> List[float]:
        """从已有数据构建5维状态，注入高斯噪声"""
        yaw_rad = np.radians(yaw_deg)
        yaw_rad = (yaw_rad + np.pi) % (2 * np.pi) - np.pi
        
        state = [loc.x, loc.y, yaw_rad, vel.x, vel.y]
        
        if self.config.observation.add_obs_noise:
            cfg = self.config.observation
            state[0] += random.gauss(0, cfg.pos_noise_std)
            state[1] += random.gauss(0, cfg.pos_noise_std)
            state[2] += random.gauss(0, cfg.yaw_noise_std)
            state[3] += random.gauss(0, cfg.vel_noise_std)
            state[4] += random.gauss(0, cfg.vel_noise_std)
            
        return state

    def _get_lane_waypoints_xy(self, start_wp, num_points, spacing) -> np.ndarray:
        """沿车道前方采样路径点, 返回(num_points, 2) [x,y]全局坐标"""
        pts = np.zeros((num_points, 2), dtype=np.float32)
        cur = start_wp
        for i in range(num_points):
            pts[i] = [cur.transform.location.x, cur.transform.location.y]
            nxt = cur.next(spacing)
            if nxt:
                cur = nxt[0]
            else:
                pts[i+1:] = pts[i]
                break
        return pts

    def _get_observation(self) -> np.ndarray:
        """
        构建观测 flat vector = [trajs | map_wps]
        优化: 每辆车只做1次API查询，结果复用于记录历史+选邻车+构建观测
        """
        enc = self.config.encoder
        N, T = enc.num_neighbours, enc.history_steps
        M, L, sp = enc.total_map_polylines, enc.path_length, enc.path_spacing

        if self.ego_vehicle is None:
            return np.zeros(enc.total_obs_dim, dtype=np.float32)

        # ---- 0. 一次性获取 ego 信息并记录历史 ----
        ego_loc = self.ego_vehicle.get_location()
        ego_vel = self.ego_vehicle.get_velocity()
        ego_yaw = self.ego_vehicle.get_transform().rotation.yaw
        ego_state = self._get_actor_state_from_cache(ego_loc, ego_vel, ego_yaw)
        self.ego_history.append(ego_state)

        # [SVO-Game] 记录全量ego轨迹 (data_collector用)
        if self.enable_trajectory_recording:
            self._episode_ego_states.append(ego_state.copy())

        # ---- 1. 障碍物（只1次API调用） ----
        if self.obstacle_vehicle and self.obstacle_vehicle.is_alive:
            obs_state = self._get_actor_state_raw(self.obstacle_vehicle)
            self.obstacle_history.append(obs_state)

        # ---- 2. 只对检测范围内的NPC做API调用，同时记录历史 ----
        det_r = self.config.observation.detection_radius
        cands = []

        if self.obstacle_vehicle and self.obstacle_vehicle.is_alive:
            obs_loc = self.obstacle_vehicle.get_location()
            cands.append((self.obstacle_vehicle, obs_loc.distance(ego_loc)))

        for npc in self.npc_vehicles:
            if not npc.is_alive:
                continue
            # Important for Level-k: when building observation for a BV,
            # self.ego_vehicle is temporarily switched to that BV. Exclude it
            # from neighbor candidates to avoid "self as nearest NPC" artifacts.
            if self.ego_vehicle is not None and npc.id == self.ego_vehicle.id:
                continue
            npc_loc = npc.get_location()
            d = npc_loc.distance(ego_loc)
            if d < det_r:
                # 只对范围内的车做完整查询并记录历史
                npc_vel = npc.get_velocity()
                npc_yaw = npc.get_transform().rotation.yaw
                state = self._get_actor_state_from_cache(npc_loc, npc_vel, npc_yaw)
                if npc.id not in self.npc_histories:
                    self.npc_histories[npc.id] = deque(maxlen=self._history_T)
                self.npc_histories[npc.id].append(state)

                # [SVO-Game] 记录全量NPC轨迹 (data_collector用)
                if self.enable_trajectory_recording:
                    if npc.id not in self._episode_npc_states:
                        self._episode_npc_states[npc.id] = []
                    self._episode_npc_states[npc.id].append(state.copy())

                cands.append((npc, d))

        cands.sort(key=lambda x: x[1])
        selected = [c[0] for c in cands[:N]]

        # ---- 3. 构建轨迹 (1+N, T, 5) ----
        trajs = np.zeros((1 + N, T, 5), dtype=np.float32)
        eh = list(self.ego_history)
        off = T - len(eh)
        for i, s in enumerate(eh):
            trajs[0, off + i] = s

        for idx, npc in enumerate(selected):
            hist = list(self.npc_histories.get(npc.id, []))
            if not hist and self.obstacle_vehicle and npc.id == self.obstacle_vehicle.id:
                hist = list(self.obstacle_history)
            off = T - len(hist)
            for i, s in enumerate(hist):
                trajs[1 + idx, off + i] = s

        # ---- 4. 构建地图路径 (M, L, 2) ----
        map_wps = np.zeros((M, L, 2), dtype=np.float32)
        ego_wp = self.map.get_waypoint(ego_loc)
        if ego_wp is not None:
            map_wps[0] = self._get_lane_waypoints_xy(ego_wp, L, sp)
            left = ego_wp.get_left_lane()
            if left and left.lane_type == carla.LaneType.Driving:
                map_wps[1] = self._get_lane_waypoints_xy(left, L, sp)
            right = ego_wp.get_right_lane()
            if right and right.lane_type == carla.LaneType.Driving:
                map_wps[2] = self._get_lane_waypoints_xy(right, L, sp)

            for idx, npc in enumerate(selected):
                base = enc.ego_paths + idx * enc.neighbor_paths
                npc_wp = self.map.get_waypoint(npc.get_location())
                if npc_wp is not None:
                    map_wps[base] = self._get_lane_waypoints_xy(npc_wp, L, sp)
                    adj = npc_wp.get_left_lane()
                    if adj is None or adj.lane_type != carla.LaneType.Driving:
                        adj = npc_wp.get_right_lane()
                    if adj and adj.lane_type == carla.LaneType.Driving:
                        map_wps[base + 1] = self._get_lane_waypoints_xy(adj, L, sp)

        # ---- 5. 辅助控制状态 — 不进Encoder，直接拼到输出后 ----
        # 原版10维 + [MandLC]新增7维 = 17维 (若mandatory_lc关闭, 新增7维全填0)
        obs_cfg = self.config.observation
        velocity = self.ego_vehicle.get_velocity()
        speed_ms = math.sqrt(velocity.x ** 2 + velocity.y ** 2)
        speed_norm = np.clip(speed_ms / obs_cfg.max_speed, 0, 1)

        acc = self.ego_vehicle.get_acceleration()
        acc_val = math.sqrt(acc.x ** 2 + acc.y ** 2)
        acc_norm = np.clip(acc_val / obs_cfg.max_acceleration, -1, 1)

        target_wp = self._get_target_waypoint()
        heading_error = self._normalize_angle(ego_yaw - target_wp.transform.rotation.yaw)
        heading_error_norm = np.clip(np.radians(heading_error) / np.pi, -1, 1)

        lateral_offset = self._calc_lateral_offset(ego_loc, target_wp)
        lateral_norm = np.clip(lateral_offset / obs_cfg.lateral_detection_range, -1, 1)

        control = self.ego_vehicle.get_control()

        speed_limit = self.ego_vehicle.get_speed_limit() / 3.6
        speed_limit_norm = np.clip(speed_limit / obs_cfg.max_speed, 0, 1) if speed_limit > 0 else 1.0

        # 前方最近车辆距离（归一化）
        front_dist = self._get_front_distance()
        front_dist_norm = np.clip(front_dist / 100.0, 0, 1)

        # === 原10维 ===
        aux_base = [
            speed_norm,           # 0: 当前速度
            acc_norm,             # 1: 加速度
            heading_error_norm,   # 2: 航向角误差
            lateral_norm,         # 3: 横向偏移 (相对目标waypoint)
            control.steer,        # 4: 当前转向角
            control.throttle,     # 5: 当前油门
            control.brake,        # 6: 当前刹车
            speed_limit_norm,     # 7: 速度限制
            front_dist_norm,      # 8: 前方车距（归一化）
            float(len(selected)), # 9: 附近车辆数（信息密度）
        ]

        # [v8 简化] aux_state 回到 10 维 (不再有 MandLC 7 维扩展)
        # key_interactor 相关特征已随 MandLC 代码移除
        aux = np.array(aux_base, dtype=np.float32)

        # ============================================================== #
        # [Level-k v9] 拼接 a_BV: 每个邻居的当前 BV 动作                  #
        # ============================================================== #
        # 维度: (num_neighbours, bv_action_feat_dim=3) → flatten
        # 每行 = [steer, throttle_brake, valid_mask]
        #   - L1 BV 接管的邻居: 从 levelk_controller.last_actions 读, valid=1
        #   - TM 控制的邻居 / 不存在的邻居 / L1 BV 训练模式: 全 0, valid=0
        # selected 的索引和 trajs[1:] 完全对齐 (按距离排序)
        bv_actions = self._collect_bv_actions(selected, N)

        return np.concatenate([
            trajs.flatten(),
            map_wps.flatten(),
            aux,
            bv_actions.flatten(),
        ]).astype(np.float32)

    def _collect_bv_actions(self, selected_neighbors, N):
        """[Level-k v9] 收集邻居的当前 BV 动作.

        Args:
            selected_neighbors: list of carla.Actor, 按距离排序 (和 trajs[1:] 对齐)
            N: num_neighbours, 输出第一维大小

        Returns:
            np.ndarray (N, 3) — 每行 [steer, throttle_brake, valid_mask]
              - 全 0 行: 该位置无邻居 / TM 控制 / L1 BV 训练模式
              - 非 0 行: 该位置由 LevelKController 接管 (valid=1)
        """
        feat_dim = self.config.encoder.bv_action_feat_dim  # 3
        bv_actions = np.zeros((N, feat_dim), dtype=np.float32)

        # 检查是否处于 L1 BV 训练模式 (BVTrainEnv 设置此 flag)
        # 训练 L1 BV 时, obs 里的 a_BV 段必须全填 0 (BV 不该 condition 在自己的对手动作上)
        if getattr(self, '_is_bv_training_mode', False):
            return bv_actions

        # 检查是否启用了 a_BV 拼接 (config 层面 disable 时也填 0)
        if not self.config.klevel.enable_bv_action_obs:
            return bv_actions

        # 没有 LevelKController (例如 baseline 测试): 全填 0, valid=0
        if self.klevel_controller is None:
            return bv_actions

        # 遍历邻居, 从 controller 缓存读 a_BV
        for i, npc in enumerate(selected_neighbors):
            if i >= N:
                break
            if npc is None or not npc.is_alive:
                continue

            # 跳过障碍车 (它不是 BV, 是静态干扰物)
            if (self.obstacle_vehicle is not None
                    and npc.id == self.obstacle_vehicle.id):
                continue

            action = self.klevel_controller.get_action_for_npc(npc.id)
            if action is not None:
                # 该邻居被 L1 BV 策略接管, 填动作 + valid=1
                bv_actions[i, 0] = float(action[0])  # steer
                bv_actions[i, 1] = float(action[1])  # throttle_brake
                bv_actions[i, 2] = 1.0               # valid_mask
            # else: TM 控制 / 未接管, 保持 [0, 0, 0]

        return bv_actions

    # ==================================================================
    #  以下方法供奖励函数/终止判断使用，完全不变
    # ==================================================================

    def _calc_lateral_offset(self, location, waypoint) -> float: #计算横向偏移（正右负左）
        wp_loc = waypoint.transform.location
        wp_fwd = waypoint.transform.get_forward_vector()
        vec = np.array([location.x - wp_loc.x, location.y - wp_loc.y])
        wp_right = np.array([wp_fwd.y, -wp_fwd.x])
        return np.dot(vec, wp_right)

    def _normalize_angle(self, angle: float) -> float: #将角度归一化到[-180, 180]范围
        while angle > 180: #
            angle -= 360
        while angle < -180:
            angle += 360
        return angle

    def _apply_action(self, action: np.ndarray): #应用动作 - 将[-1, 1]范围的动作转换为实际控制输入
        self.current_action = action.copy()
        action_cfg = self.config.action

        steer = float(np.clip(action[0], -1.0, 1.0)) * action_cfg.max_steer #转向动作 [-1, 1]映射到[-max_steer, max_steer]
        throttle_brake = float(np.clip(action[1], -1.0, 1.0)) * action_cfg.throttle_brake_scale

        if throttle_brake >= 0:
            throttle = min(throttle_brake, action_cfg.max_throttle)
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(-throttle_brake, action_cfg.max_brake)

        # 限速治理: 超过目标速度后压油门并逐步制动
        velocity = self.ego_vehicle.get_velocity()
        speed_kmh = math.sqrt(velocity.x ** 2 + velocity.y ** 2) * 3.6
        target_kmh = float(self.config.reward.target_speed)
        margin = float(getattr(action_cfg, "speed_governor_margin_kmh", 2.0))
        brake_trigger = float(getattr(action_cfg, "speed_governor_brake_kmh", 6.0))
        if speed_kmh > target_kmh:
            excess = speed_kmh - target_kmh
            throttle_cap = max(0.0, action_cfg.max_throttle * (1.0 - min(excess / 20.0, 0.9)))
            throttle = min(throttle, throttle_cap)
            if excess > margin:
                brake = max(brake, min(action_cfg.max_brake, 0.05 + 0.03 * (excess - margin)))
            if excess > brake_trigger:
                brake = max(brake, min(action_cfg.max_brake, 0.18 + 0.02 * (excess - brake_trigger)))

        # 转向防抖: 速率限制 + 一阶平滑
        max_delta = float(getattr(action_cfg, "steer_rate_limit", 0.06))
        alpha = float(np.clip(getattr(action_cfg, "steer_smooth_factor", 0.65), 0.0, 0.95))
        steer = float(np.clip(steer, self._last_applied_steer - max_delta, self._last_applied_steer + max_delta))
        steer = float(alpha * self._last_applied_steer + (1.0 - alpha) * steer)
        self._last_applied_steer = steer

        self.ego_vehicle.apply_control(carla.VehicleControl(
            throttle=throttle, steer=steer, brake=brake
        ))

    def _calculate_reward(self) -> float:
        """计算奖励
        === 公式 ===
        L1 模式 (reward_mode="l1"):
            R = r_ego

        L2 模式 (reward_mode="l2"):
            R = cos(θ) · r_ego + sin(θ) · r_others_norm · 2.0    (Toghi 2022)
            θ ∈ [0, π/2]:
              θ→0°  (激进 NPC): cos≈1 → r_ego 主导,  ego 不太管 NPC
                                sin≈0 → r_others 权重低
              θ→90° (保守 NPC): cos≈0 → r_ego 权重低
                                sin≈1 → r_others 主导, ego 照顾 NPC

            r_others 用所有后方NPC的 IDM Δa 平均,归一化到[-1,1],再放大2倍.

        === θ 来源: 两种模式, 通过 config.reward.use_oracle_svo 切换 ===
          use_oracle_svo=True:  用 NPC 真实 style 标签 (aggressive→15°... conservative→75°)
                                映射到平均 θ. 训练阶段推荐, 稳定可靠.
          use_oracle_svo=False: 用 BIRL 预训练模型推断的 svo_mu 平均值.
                                测试/消融实验用, 验证 BIRL 推断质量.
        """
        reward_mode = getattr(self.config.klevel, 'reward_mode', 'l2')
        r_ego = self._calculate_ego_reward()

        # L1 模式直接返回
        if reward_mode == "l1":
            self.reward_history.append(r_ego)
            return r_ego

        # [Level-k v9] L1_BV 模式: 不应走到这里, BVTrainEnv 直接调 _calculate_bv_reward
        # 防御性兜底: 如果有人误用此分支, 退化到 r_ego
        if reward_mode == "l1_bv":
            self.reward_history.append(r_ego)
            return r_ego

        # L2 模式: Toghi 2022 公式
        theta_rad = self._get_svo_theta()   # 根据 use_oracle_svo 自动选 oracle / BIRL
        r_others = self._compute_others_reward()
        r_others_norm = float(np.clip(r_others / 6.0, -1.0, 1.0))

        reward = (math.cos(theta_rad) * r_ego
                  + math.sin(theta_rad) * r_others_norm * 2.0)

        # 调试用 reward components (供 render/tensorboard 读取)
        self._last_reward_components = {
            'r_ego': float(r_ego),
            'r_others': float(r_others),
            'r_others_norm': float(r_others_norm),
            'theta_deg': float(math.degrees(theta_rad)),
            'r_total': float(reward),
        }

        self.reward_history.append(reward)
        return reward

    def _get_svo_theta(self):
        """获取 SVO θ (弧度), 根据 use_oracle_svo 开关在 Oracle / BIRL 间切换.

        Oracle 模式 (use_oracle_svo=True):
            用场景内所有交互NPC的真实风格标签映射到 θ, 取平均.
            style → θ 映射:
              aggressive          → 15°
              semi_aggressive     → 30°
              normal              → 45°
              semi_conservative   → 60°
              conservative        → 75°
            优点: 稳定, 不受 BIRL 推断误差影响
            用途: 训练阶段, 消融实验的"上限"baseline

        BIRL 模式 (use_oracle_svo=False):
            用 self._svo_mu (ppo_model._infer_svo_batch 计算的推断结果) 平均值.
            只取 interact_mask=True 的NPC, 且排除默认先验值(θ≈45°)
            优点: 测试泛化, 验证 BIRL 质量
            用途: 测试阶段, 对比 Oracle baseline
        """
        use_oracle = bool(getattr(self.config.reward, 'use_oracle_svo', True))

        if use_oracle:
            # === Oracle 模式 ===
            style_map = {
                'aggressive': 15.0,
                'semi_aggressive': 30.0,
                'normal': 45.0,
                'semi_conservative': 60.0,
                'conservative': 75.0,
            }
            if not self._npc_style_labels:
                return math.radians(45.0)

            # 只统计检测半径内的NPC (和 r_others 的范围保持一致)
            ego_loc = self.ego_vehicle.get_location() if self.ego_vehicle else None
            if ego_loc is None:
                return math.radians(45.0)
            det_r = self.config.observation.detection_radius

            thetas = []
            for npc in self.npc_vehicles:
                if not npc.is_alive:
                    continue
                if npc.id not in self._npc_style_labels:
                    continue
                d = npc.get_location().distance(ego_loc)
                if d > det_r:
                    continue
                style = self._npc_style_labels[npc.id]
                thetas.append(style_map.get(style, 45.0))

            if not thetas:
                return math.radians(45.0)
            return math.radians(float(np.mean(thetas)))
        else:
            # === BIRL 模式 === (复用原有 _get_mean_svo_theta 逻辑)
            return self._get_mean_svo_theta()

    def _get_mean_svo_theta(self):
        """获取BIRL推断的平均SVO角度(弧度), 用于L2奖励计算.
        [DEPRECATED for MandLC] 保留以向后兼容, 强制变道场景应用 _get_key_svo_theta"""
        svo_mu = getattr(self, '_svo_mu', None)
        if svo_mu is None:
            return math.radians(45.0)  # 默认先验
        interact_mask = getattr(self, '_svo_interact_mask', None)
        if interact_mask is not None and len(interact_mask) == len(svo_mu):
            cand = svo_mu[interact_mask]
        else:
            cand = svo_mu
        valid = cand[np.abs(cand - 45.0) > 0.5]
        if len(valid) == 0:
            return math.radians(45.0)
        return math.radians(float(np.mean(valid)))


    def _calculate_ego_reward(self) -> float:
        """纯 ego 奖励

        包含 5 项:
          1) 速度奖励: 越接近 target_speed 越高
          2) 车道保持: 在 max_lateral_offset 内正奖励, 超出线性惩罚
          3) 前车距离: 小于 min_safe_distance 时线性惩罚
          4) 转向平滑: |Δsteer| 惩罚
          5) 终止类: 碰撞一次性惩罚, 每步时间惩罚

        """
        rcfg = self.config.reward
        reward = 0.0

        # 速度
        velocity = self.ego_vehicle.get_velocity()
        speed_kmh = math.sqrt(velocity.x ** 2 + velocity.y ** 2) * 3.6
        self.speed_history.append(speed_kmh)

        # 参考 waypoint
        transform = self.ego_vehicle.get_transform()
        ego_loc = transform.location
        target_wp = self._get_target_waypoint()
        if target_wp is None:
            target_wp = self.map.get_waypoint(ego_loc, lane_type=carla.LaneType.Driving)

        # 1) 速度奖励
        speed_diff = abs(speed_kmh - rcfg.target_speed) / max(rcfg.target_speed, 1e-6) 
        reward += max(0.0, 1.0 - speed_diff) * rcfg.speed_reward_weight
        if speed_kmh < rcfg.min_speed:
            reward += rcfg.low_speed_penalty

        # 2) 车道保持
        if target_wp is not None:
            lateral_offset = self._calc_lateral_offset(ego_loc, target_wp)
            abs_lateral = abs(lateral_offset)
            if abs_lateral < rcfg.max_lateral_offset:
                reward += (1.0 - abs_lateral / rcfg.max_lateral_offset) * rcfg.lane_keeping_weight
            else:
                reward -= (abs_lateral - rcfg.max_lateral_offset) * 0.1

        # 3) 前车距离
        front_dist = self._get_front_distance()
        if front_dist < rcfg.min_safe_distance:
            reward -= (rcfg.min_safe_distance - front_dist) / max(rcfg.min_safe_distance, 1e-6) * rcfg.safe_distance_weight
        elif front_dist < rcfg.min_safe_distance * 2:
            reward += rcfg.near_front_penalty

        # 4) 转向平滑
        control = self.ego_vehicle.get_control()
        reward -= abs(control.steer - self.prev_steer) * rcfg.steering_penalty_weight

        # 5) 碰撞 + 固定进度 + 时间
        if self.collision_history:
            reward += rcfg.collision_penalty
        reward += rcfg.progress_reward
        reward += rcfg.time_penalty

        return reward

    def _get_front_distance(self) -> float:
        """获取前方最近车辆距离 (v8 回归旧版: 用 ego_fwd 作前向).

        旧版本逻辑: 用 ego 车头朝向作为"前方"方向, 横向阈值 2.5m.
        ego 打方向时, 坐标系会跟着转, 但这对 lane_keeping+safety 的正常训练没问题
        (你旧版就是这么训出能正常变道绕障的).
        """
        ego_transform = self.ego_vehicle.get_transform()
        ego_loc = ego_transform.location
        ego_fwd = ego_transform.get_forward_vector()

        min_dist = 100.0
        fwd_2d = np.array([ego_fwd.x, ego_fwd.y])
        fwd_2d = fwd_2d / (np.linalg.norm(fwd_2d) + 1e-6)

        all_vehicles = self.npc_vehicles.copy()
        if self.obstacle_vehicle is not None and self.obstacle_vehicle.is_alive:
            all_vehicles.append(self.obstacle_vehicle)

        for vehicle in all_vehicles:
            if vehicle is None or not vehicle.is_alive:
                continue
            veh_loc = vehicle.get_location()
            rel_pos = np.array([veh_loc.x - ego_loc.x, veh_loc.y - ego_loc.y])
            long = np.dot(rel_pos, fwd_2d)
            lat = abs(np.dot(rel_pos, np.array([fwd_2d[1], -fwd_2d[0]])))
            if long > 0 and lat < 2.5:
                min_dist = min(min_dist, long)

        return min_dist

    def _check_done(self) -> Tuple[bool, Dict[str, Any]]:
        """检查 episode 结束 (v8 恢复原版: 不再有 MandLC 特殊分支).

        终止条件:
          1. 碰撞 → 'collision'
          2. 无可行驶车道 → 'no_driving_lane'
          3. 跨越两条车道以上 → 'cross_two_lanes'
          4. 非驾驶车道 → 'non_driving_lane'
          5. 超出 terminal_lateral_offset 横向偏移 → 'out_of_lane'
          6. 超时 → 'max_steps'
          7. 到达路径终点 → 'reached_goal' (success=True)
        """
        info = {
            'collision': False,
            'out_of_lane': False,
            'timeout': False,
            'success': False,
            'total_reward': self.total_reward,
            'episode_length': self.current_step
        }
        rcfg = self.config.reward

        # 1. 碰撞
        if self.collision_history:
            info['collision'] = True
            info['termination_reason'] = 'collision'
            return True, info

        # 2. 无可行驶车道
        transform = self.ego_vehicle.get_transform()
        ego_wp = self.map.get_waypoint(
            transform.location, lane_type=carla.LaneType.Driving
        )
        if ego_wp is None:
            info['out_of_lane'] = True
            info['termination_reason'] = 'no_driving_lane'
            return True, info

        current_lane_id = ego_wp.lane_id if ego_wp else 0
        spawn_lane_id = int(getattr(self, '_spawn_lane_id', 0))

        # 3. 跨越两条车道以上
        if abs(current_lane_id - spawn_lane_id) > 1:
            info['out_of_lane'] = True
            info['termination_reason'] = 'cross_two_lanes'
            return True, info

        # 4. 非驾驶车道
        if ego_wp.lane_type != carla.LaneType.Driving:
            info['out_of_lane'] = True
            info['termination_reason'] = 'non_driving_lane'
            return True, info

        # 5. 横向偏移过大
        lateral_offset = abs(self._calc_lateral_offset(transform.location, ego_wp))
        terminal_threshold = float(rcfg.terminal_lateral_offset)
        if lateral_offset > terminal_threshold:
            info['out_of_lane'] = True
            info['termination_reason'] = f'out_of_lane (>{terminal_threshold}m)'
            return True, info

        # 6. 超时
        if self.current_step >= self.config.train.max_episode_steps:
            info['timeout'] = True
            info['termination_reason'] = 'max_steps'
            return True, info

        # 7. 到达路径终点
        if len(self.route_waypoints) > 0 and self.current_waypoint_idx >= len(self.route_waypoints) - 5:
            info['success'] = True
            info['termination_reason'] = 'reached_goal'
            return True, info

        return False, info

    # ================================================================== #
    #  [Level-k] BV策略控制接口                                           #
    # ================================================================== #

    def set_bv_control(self, controller):
        """
        设置或清除BV策略控制器.

        Args:
            controller: LevelKController实例 (已加载策略), 或None (恢复TM控制)
        """
        self.klevel_controller = controller
        if controller is not None:
            self._bv_control_mode = "level1"
            print(f"[Level-k] BV策略控制已启用, 控制范围={controller.control_radius}m")
        else:
            self._bv_control_mode = "tm"

    def build_observation_for(self, center_vehicle):
        """
        以任意车辆为中心构建观测 (与_get_observation格式完全一致).

        这是Level-k策略控制BV的前提:
          策略网络输入格式 = [trajs | map_wps | aux]
          现在以NPC_i为中心, 周围车(包括真ego)成为它的neighbors.

        实现: "临时切换ego" 策略
          1. 保存真实ego → 临时切换 self.ego_vehicle = center_vehicle
          2. 调用 _get_observation() (它以self.ego_vehicle为中心)
          3. 恢复真实ego

        参考: CHARMS abstract.py 第288行
          self.observation_type.observer_vehicle = vehicle
          obs = self.observation_type.observe()

        Args:
            center_vehicle: carla.Vehicle, 作为"ego"的车辆

        Returns:
            obs: np.ndarray (total_obs_dim,), 与_get_observation()格式一致
        """
        # 保存真实状态
        real_ego = self.ego_vehicle
        real_route = self.route_waypoints
        real_wp_idx = self.current_waypoint_idx
        real_ego_history = self.ego_history
        real_npc_histories = {
            npc_id: deque(
                hist,
                maxlen=hist.maxlen if hist.maxlen is not None else self._history_T
            )
            for npc_id, hist in self.npc_histories.items()
        }
        real_obstacle_history = deque(
            self.obstacle_history,
            maxlen=self.obstacle_history.maxlen if self.obstacle_history.maxlen is not None else self._history_T
        )
        real_recording = self.enable_trajectory_recording
        real_npc_vehicles = self.npc_vehicles

        # 切换ego
        self.ego_vehicle = center_vehicle
        self.enable_trajectory_recording = False

        # [Level-k v9] 关键: 给 BV 构建观测时, 强制 bv_actions 段填 0
        # 否则会出现"BV condition 在另一个 BV 的动作上"的无穷递归
        # 这和 L1 BV 训练时的 obs 格式完全一致, 推理时无 distribution shift
        real_bv_training_flag = getattr(self, '_is_bv_training_mode', False)
        self._is_bv_training_mode = True

        try:
            # 为center_vehicle构建临时路径 (直行参考线)
            try:
                center_loc = center_vehicle.get_location()
                center_wp = self.map.get_waypoint(center_loc, lane_type=carla.LaneType.Driving)
                if center_wp is not None:
                    temp_route = [center_wp]
                    wp = center_wp
                    for _ in range(50):
                        next_wps = wp.next(self.config.scenario.waypoint_spacing)
                        if next_wps:
                            wp = next_wps[0]
                            temp_route.append(wp)
                        else:
                            break
                    self.route_waypoints = temp_route
                else:
                    self.route_waypoints = real_route  # 退回原路径
                self.current_waypoint_idx = 0
            except Exception:
                self.route_waypoints = real_route
                self.current_waypoint_idx = 0

            # 临时ego历史 (用NPC的历史, 如果有的话)
            npc_id = center_vehicle.id
            if npc_id in self.npc_histories and len(self.npc_histories[npc_id]) > 0:
                self.ego_history = deque(self.npc_histories[npc_id], maxlen=self._history_T)
            else:
                # 没有历史, 用当前状态填充
                self.ego_history = deque(maxlen=self._history_T)
                try:
                    vel = center_vehicle.get_velocity()
                    yaw = center_vehicle.get_transform().rotation.yaw
                    state = self._get_actor_state_from_cache(center_loc, vel, yaw)
                    self.ego_history.append(state)
                except Exception:
                    self.ego_history.append(np.zeros(5, dtype=np.float32))

            # 关键修复:
            # 1) BV观测时显式注入真实ego，确保BV能感知并对ego做跟驰/让行决策
            # 2) 仅临时注入，结束后恢复
            if (
                real_ego is not None
                and real_ego.is_alive
                and real_ego.id != center_vehicle.id
                and not any(v.id == real_ego.id for v in self.npc_vehicles if v is not None and v.is_alive)
            ):
                self.npc_vehicles = list(self.npc_vehicles) + [real_ego]

            # 构建观测 (复用现有逻辑)
            obs = self._get_observation()
        except Exception:
            obs = np.zeros(self.config.encoder.total_obs_dim, dtype=np.float32)
        finally:
            # 恢复所有状态，保证build_observation_for无副作用
            self.ego_vehicle = real_ego
            self.route_waypoints = real_route
            self.current_waypoint_idx = real_wp_idx
            self.ego_history = real_ego_history
            self.npc_histories = real_npc_histories
            self.obstacle_history = real_obstacle_history
            self.enable_trajectory_recording = real_recording
            self.npc_vehicles = real_npc_vehicles
            # [Level-k v9] 恢复 BV 训练模式 flag
            self._is_bv_training_mode = real_bv_training_flag

        return obs

    # ================================================================== #
    #  [Level-k] R_others: 他车收益奖励 (L2模式用)                        #
    # ================================================================== #

    def _compute_others_reward(self):
        """
        他车收益奖励 (只有L2奖励模式使用).

        参考CHARMS公式(7):
          r_others = clip(Δa_rear_current, -3, 3) + clip(Δa_rear_target, -3, 3)

        Δa = ego执行动作后, 后车的IDM期望加速度变化.
          正值 = ego的动作让后车可以加速 (友好行为)
          负值 = ego的动作迫使后车减速 (自私行为)
        """
        r_others = 0.0
        kl_cfg = self.config.klevel
        ego_loc = self.ego_vehicle.get_location()
        ego_vel = self.ego_vehicle.get_velocity()
        ego_speed = math.sqrt(ego_vel.x**2 + ego_vel.y**2)
        ego_fwd = self.ego_vehicle.get_transform().get_forward_vector()
        fwd_2d = np.array([ego_fwd.x, ego_fwd.y])
        fwd_2d = fwd_2d / (np.linalg.norm(fwd_2d) + 1e-6)

        # 查找当前车道和目标车道的后车
        for npc in self.npc_vehicles:
            if not npc.is_alive:
                continue
            npc_loc = npc.get_location()
            rel = np.array([npc_loc.x - ego_loc.x, npc_loc.y - ego_loc.y])
            long_dist = np.dot(rel, fwd_2d)       # 正=前方, 负=后方
            lat_dist = np.dot(rel, np.array([fwd_2d[1], -fwd_2d[0]]))

            # 只看后方车辆 (long_dist < 0) 且在相邻车道内
            if long_dist >= 0 or abs(long_dist) > 50.0:
                continue
            if abs(lat_dist) > 5.0:
                continue

            npc_vel = npc.get_velocity()
            npc_speed = math.sqrt(npc_vel.x**2 + npc_vel.y**2)

            # IDM加速度: ego作为前车
            gap_to_ego = abs(long_dist) - 4.5  # 减去车长
            gap_to_ego = max(gap_to_ego, 0.1)
            delta_v = npc_speed - ego_speed  # 接近速度

            a_with_ego = self._idm_acceleration(
                npc_speed, gap_to_ego, delta_v, kl_cfg
            )
            # IDM自由流加速度 (无前车)
            a_free = kl_cfg.idm_max_accel * (
                1.0 - (npc_speed / max(kl_cfg.idm_desired_speed, 0.01)) ** kl_cfg.idm_accel_exponent
            )
            # Δa = 有ego时的加速度 - 无ego时的加速度
            delta_a = a_with_ego - a_free
            r_others += np.clip(delta_a, -3.0, 3.0)

        return r_others

    @staticmethod
    def _idm_acceleration(v, s_gap, delta_v, kl_cfg):
        """IDM加速度公式."""
        v = max(v, 0.0)
        s_star = kl_cfg.idm_min_gap + max(0.0,
            v * kl_cfg.idm_time_headway
            + v * delta_v / (2.0 * math.sqrt(kl_cfg.idm_max_accel * kl_cfg.idm_comfort_decel)))
        a = kl_cfg.idm_max_accel * (
            1.0 - (v / max(kl_cfg.idm_desired_speed, 0.01)) ** kl_cfg.idm_accel_exponent
            - (s_star / max(s_gap, 0.01)) ** 2
        )
        return float(np.clip(a, -kl_cfg.idm_comfort_decel, kl_cfg.idm_max_accel))

    # ================================================================== #
    #  [NEW] SVO风险感知 + 可视化接口                                       #
    # ================================================================== #

    def set_svo_info(self, svo_mu, svo_sigma, pred_trajs, interact_mask):
        """
        接收SVO推断结果, 用于:
          1. 存储SVO后验, 供L2奖励读取
          2. 存储可视化数据 (pygame面板显示)
        说明:
          pred_trajs 当前并非严格未来预测轨迹, 暂不用于风险惩罚计算。

        由train.py在每步select_action后、env.step前调用.
        """
        if svo_mu is None:
            self._svo_mu = None
            self._svo_sigma = None
            self._svo_interact_mask = None
            self._svo_risk_penalty = 0.0
            self._svo_display_info = {}
            return

        # [Fix] 同步保存SVO后验, 供L2奖励读取
        self._svo_mu = svo_mu.copy()
        self._svo_sigma = svo_sigma.copy()
        self._svo_interact_mask = interact_mask.copy() if interact_mask is not None else None

        self._svo_display_info = {
            'svo_mu': svo_mu.copy(),
            'svo_sigma': svo_sigma.copy(),
            'interact_mask': interact_mask.copy() if interact_mask is not None else None,
        }

        # 先关闭，等待future prediction模块就绪后再恢复
        self._svo_risk_penalty = 0.0
    # ================================================================== #
    #  [SVO-Game] 轨迹数据提取 (data_collector用)                         #
    # ================================================================== #

    def get_episode_trajectories(self, min_length=20):
        """
        提取当前episode的(ego, npc)轨迹对, 用于SVO预训练数据集.

        只返回观测窗口>=min_length步的NPC (太短无法构建past+future对).

        Returns: list of dict, 每个dict包含:
            'ego':  np.ndarray (T_total, 5)
            'npc':  np.ndarray (T_total, 5)  与ego等长(短的前面补零)
            'npc_id': int
        """
        if not self.enable_trajectory_recording:
            return []

        ego_arr = np.array(self._episode_ego_states, dtype=np.float32)
        T_ego = len(ego_arr)
        if T_ego < min_length:
            return []

        pairs = []
        for npc_id, states in self._episode_npc_states.items():
            if len(states) < min_length:
                continue
            npc_arr = np.array(states, dtype=np.float32)
            # 对齐: NPC可能在episode中途才出现在检测范围内
            # 做法: 取NPC实际记录的长度, ego从末尾对齐截取同样长度
            T_npc = len(npc_arr)
            T_use = min(T_ego, T_npc)
            pairs.append({
                'ego': ego_arr[-T_use:],
                'npc': npc_arr[-T_use:],
                'npc_id': npc_id,
                'style': self._npc_style_labels.get(npc_id, 'unknown'),
            })

        return pairs

    def record_step(self):
        """
        [SVO-Game] 轻量级轨迹记录 — 仅记录ego和NPC的位置/速度.

        与_get_observation()的区别:
          - 不构建完整570维观测向量
          - 不查询地图路径
          - 不计算辅助状态 (heading error, lateral offset等)
          - 不依赖route_waypoints或current_waypoint_idx

        只做: 查ego状态 + 遍历检测范围内NPC + append到记录列表.
        专为data_collector.py设计, 正常训练不使用.
        """
        if self.ego_vehicle is None or not self.ego_vehicle.is_alive:
            return

        # Ego状态
        ego_loc = self.ego_vehicle.get_location()
        ego_vel = self.ego_vehicle.get_velocity()
        ego_yaw = self.ego_vehicle.get_transform().rotation.yaw
        ego_state = self._get_actor_state_from_cache(ego_loc, ego_vel, ego_yaw)

        if self.enable_trajectory_recording:
            self._episode_ego_states.append(ego_state.copy())

        # NPC状态 (检测范围内)
        det_r = self.config.observation.detection_radius
        for npc in self.npc_vehicles:
            if not npc.is_alive:
                continue
            npc_loc = npc.get_location()
            d = npc_loc.distance(ego_loc)
            if d < det_r:
                npc_vel = npc.get_velocity()
                npc_yaw = npc.get_transform().rotation.yaw
                state = self._get_actor_state_from_cache(npc_loc, npc_vel, npc_yaw)

                if self.enable_trajectory_recording:
                    if npc.id not in self._episode_npc_states:
                        self._episode_npc_states[npc.id] = []
                    self._episode_npc_states[npc.id].append(state.copy())

    def reset(self) -> np.ndarray:
        # 记录上一回合平均reward
        if self.episode_count > 0 and self.current_step > 0:
            self.episode_rewards.append(self.total_reward)
            self.prev_episode_avg_reward = float(self.total_reward) / float(self.current_step)
        elif self.episode_count > 0:
            self.prev_episode_avg_reward = 0.0

        self._cleanup()

        self.current_step = 0
        self.total_reward = 0
        self.collision_history = []
        self.prev_steer = 0
        self._last_applied_steer = 0.0
        self.episode_count += 1
        self.current_waypoint_idx = 0

        # 重置当前回合reward统计
        self.last_reward = 0.0
        self.current_episode_reward_sum = 0.0
        self.current_episode_steps = 0

        # 清空历史轨迹缓冲区
        self.ego_history.clear()
        self.npc_histories.clear()
        self.obstacle_history.clear()

        # [SVO-Game] 清空全量轨迹记录
        self._episode_ego_states = []
        self._episode_npc_states = {}
        self._npc_style_labels = {}
        self._npc_agents = {}

        # [Fix] 变道/SVO状态重置
        self._obstacle_passed = False
        self._returned_to_lane = False  # [Fix] 回原车道奖励只给一次
        self._svo_mu = None
        self._svo_sigma = None
        self._svo_interact_mask = None
        self._svo_risk_penalty = 0.0
        self._svo_display_info = {}  # [NEW] SVO可视化信息

        # [Level-k] 重置BV策略控制状态
        if self.klevel_controller is not None:
            self.klevel_controller.reset()

        if not self._spawn_ego_vehicle():
            raise RuntimeError("无法生成主车")

        # 重置奖励组件 (供调试/tensorboard 读取)
        self._last_reward_components = {}

        scenario_type = getattr(self.config.scenario, "scenario_type", "highway")
        if scenario_type == "unprotected_left_turn":
            self._plan_route_unprotected_left_turn()
        else:
            self._plan_route()

        if scenario_type != "unprotected_left_turn":
            self._spawn_obstacle_vehicle()

        self._setup_sensors()
        for _ in range(5):
            self.world.tick()
        if scenario_type == "unprotected_left_turn":
            self._spawn_npc_vehicles_unprotected_left_turn()
        else:
            self._spawn_npc_vehicles()
        for _ in range(15):
            self.world.tick()

        # Lane-change statistics for this episode.
        self._ep_lane_change_left = 0
        self._ep_lane_change_right = 0
        self._ep_lane_change_unknown = 0
        self._step_waypoint_progress = 0
        self._prev_ego_wp = self.map.get_waypoint(
            self.ego_vehicle.get_transform().location, lane_type=carla.LaneType.Driving
        )
        self._prev_lane_id = self._prev_ego_wp.lane_id if self._prev_ego_wp else None

        self.episode_start_time = time.time()
        print(f"Episode {self.episode_count} 开始")
        return self._get_observation()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        self._apply_action(action)

        # [Level-k] BV策略控制: 在ego动作之后、world.tick之前施加BV动作
        if (self.klevel_controller is not None
                and self._bv_control_mode != "tm"):
            controlled_ids = self.klevel_controller.step(
                self.ego_vehicle,
                self.npc_vehicles,
                self.world,
                self,  # 传env自身, 用于build_observation_for()
            )
        else:
            controlled_ids = []

        self.world.tick()

        self.current_step += 1
        prev_wp_idx = self.current_waypoint_idx
        self._update_waypoint_index()
        self._draw_waypoints()
        self._step_waypoint_progress = max(0, self.current_waypoint_idx - prev_wp_idx)

        curr_wp = self.map.get_waypoint(
            self.ego_vehicle.get_transform().location, lane_type=carla.LaneType.Driving
        )
        curr_lane_id = curr_wp.lane_id if curr_wp else None

        # Track left/right lane-change direction for debugging policy bias.
        if self._prev_ego_wp is not None and curr_wp is not None and curr_lane_id != self._prev_ego_wp.lane_id:
            left_wp = self._prev_ego_wp.get_left_lane()
            right_wp = self._prev_ego_wp.get_right_lane()

            if (left_wp is not None
                    and left_wp.lane_type == carla.LaneType.Driving
                    and left_wp.lane_id == curr_lane_id):
                self._ep_lane_change_left += 1
            elif (right_wp is not None
                    and right_wp.lane_type == carla.LaneType.Driving
                    and right_wp.lane_id == curr_lane_id):
                self._ep_lane_change_right += 1
            else:
                self._ep_lane_change_unknown += 1

        self._prev_ego_wp = curr_wp
        self._prev_lane_id = curr_lane_id

        obs = self._get_observation()

        reward = self._calculate_reward()
        self.last_reward = float(reward)
        self.total_reward += reward
        # 为下一步计算转向平滑项保留“上一时刻转向”
        self.prev_steer = self.ego_vehicle.get_control().steer

        self.current_episode_reward_sum += reward
        self.current_episode_steps += 1

        done, info = self._check_done()

        terminal_adjustment = 0.0
        if done:
            term_reason = info.get('termination_reason', '')
            rcfg = self.config.reward

            if info.get('success', False):
                terminal_adjustment = float(rcfg.reach_goal_reward)
            elif term_reason == 'cross_two_lanes':
                terminal_adjustment = float(rcfg.terminal_cross_two_lanes_penalty)
            elif term_reason in ('no_driving_lane', 'non_driving_lane', 'no_reference_lane') or str(term_reason).startswith('out_of_lane'):
                terminal_adjustment = float(rcfg.terminal_lane_violation_penalty)
            # 注意: 碰撞惩罚 collision_penalty 已经在 _calculate_ego_reward 里加过,
            # 这里不重复给. 超出起始车道/MandLC 相关终止已取消.

            if terminal_adjustment != 0.0:
                reward += terminal_adjustment
                self.last_reward = float(reward)
                self.total_reward += terminal_adjustment
                self.current_episode_reward_sum += terminal_adjustment

        # ===== 近失定义：非碰撞、非成功的任何终止，都算近失 =====
        if done:
            if info.get('success', False):
                self.success_count += 1
            elif info.get('collision', False):
                pass
            else:
                self.near_miss_count += 1

        velocity = self.ego_vehicle.get_velocity()
        info['speed_kmh'] = math.sqrt(velocity.x ** 2 + velocity.y ** 2) * 3.6
        info['front_distance'] = self._get_front_distance()
        info['safe_distance_target'] = max(
            self.config.reward.min_safe_distance,
            (info['speed_kmh'] / 3.6) * self.config.reward.desired_headway_time,
        )

        # ============================================================== #
        # [CVaR] step-level safety cost                                    #
        # ============================================================== #
        # collision_step / lane_violation_step 仅在终止那一步打 1 (sparse)
        # cost_step = w_col*coll + w_lane*lane + w_safe * shortfall_ratio
        ppo_cfg = self.config.ppo
        info['collision_step'] = 1 if info.get('collision', False) else 0
        _term = str(info.get('termination_reason', '') or '')
        _is_lane_violation = (
            _term in ('cross_two_lanes', 'no_driving_lane',
                      'non_driving_lane', 'no_reference_lane')
            or _term.startswith('out_of_lane')
        )
        info['lane_violation_step'] = 1 if _is_lane_violation else 0
        _safe_target = float(info['safe_distance_target'])
        _front_d = float(info['front_distance'])
        _shortfall = max(0.0, _safe_target - _front_d) / (_safe_target + ppo_cfg.cost_eps)
        info['cost_step'] = float(
            ppo_cfg.cost_w_collision * info['collision_step']
            + ppo_cfg.cost_w_lane * info['lane_violation_step']
            + ppo_cfg.cost_w_safe * _shortfall
        )

        info['svo_risk_penalty'] = float(getattr(self, '_svo_risk_penalty', 0.0))
        info['waypoint_progress'] = f"{self.current_waypoint_idx}/{len(self.route_waypoints)}"

        target_wp = self._get_target_waypoint()
        ego_wp = self.map.get_waypoint(
            self.ego_vehicle.get_transform().location, lane_type=carla.LaneType.Driving
        )
        # Keep lateral offset in info aligned with termination check (route-based reference).
        lat_ref_wp = target_wp if target_wp is not None else ego_wp
        info['lateral_offset'] = self._calc_lateral_offset(
            self.ego_vehicle.get_transform().location, lat_ref_wp
        ) if lat_ref_wp is not None else 0.0

        info['reward_step'] = self.last_reward
        info['ep_avg_reward'] = (self.current_episode_reward_sum / max(1, self.current_episode_steps))
        info['prev_ep_avg_reward'] = self.prev_episode_avg_reward
        info['terminal_adjustment'] = terminal_adjustment
        info['terminal_bonus'] = max(terminal_adjustment, 0.0)
        info['terminal_penalty'] = min(terminal_adjustment, 0.0)
        info['step_waypoint_progress'] = int(getattr(self, '_step_waypoint_progress', 0))
        info['total_reward'] = self.total_reward
        info['current_lane_id'] = curr_lane_id if curr_lane_id is not None else 0
        info['spawn_lane_id'] = int(getattr(self, '_spawn_lane_id', 0))
        info['lane_diff'] = abs(info['current_lane_id'] - info['spawn_lane_id'])
        info['ep_lane_change_left'] = int(self._ep_lane_change_left)
        info['ep_lane_change_right'] = int(self._ep_lane_change_right)
        info['ep_lane_change_unknown'] = int(self._ep_lane_change_unknown)
        info['success_count'] = self.success_count
        info['near_miss_count'] = self.near_miss_count

        # === SVO debug info (v8) ===
        comp = getattr(self, '_last_reward_components', None)
        if comp:
            info['r_ego'] = float(comp.get('r_ego', 0.0))
            info['r_others'] = float(comp.get('r_others', 0.0))
            info['r_others_norm'] = float(comp.get('r_others_norm', 0.0))
            info['theta_deg'] = float(comp.get('theta_deg', 45.0))
            info['r_total'] = float(comp.get('r_total', 0.0))
        return obs, reward, done, info

    def render(self, mode='human'):
        if not self.config.visual.enable or not PYGAME_AVAILABLE or self.display is None:
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == K_ESCAPE:
                    self.close()
                    return
                elif event.key == K_p:
                    self.paused = not self.paused

        if self.paused:
            self._draw_pause_screen()
            pygame.display.flip()
            self.clock.tick(10)
            return

        self.display.fill((30, 30, 30))

        if self.camera_image is not None:
            camera_rect = pygame.Rect(10, 10, self.config.visual.width - 320, self.config.visual.height - 20)
            camera_surface = pygame.surfarray.make_surface(self.camera_image.swapaxes(0, 1))
            scaled_surface = pygame.transform.scale(camera_surface, (camera_rect.width, camera_rect.height))
            self.display.blit(scaled_surface, camera_rect)
            pygame.draw.rect(self.display, (100, 100, 100), camera_rect, 2)

        self._draw_info_panel()
        pygame.display.flip()
        self.clock.tick(self.config.visual.fps)

    def _draw_info_panel(self): #在屏幕右侧绘制信息面板，显示当前状态、奖励、统计数据等
        panel_x = self.config.visual.width - 300
        panel_y = 10
        panel_width = 280
        panel_height = self.config.visual.height - 20

        pygame.draw.rect(self.display, (40, 40, 50), (panel_x, panel_y, panel_width, panel_height))
        pygame.draw.rect(self.display, (80, 80, 100), (panel_x, panel_y, panel_width, panel_height), 2)

        y = panel_y + 10
        title = self.font_large.render("Monitor", True, (255, 255, 255))
        self.display.blit(title, (panel_x + 10, y))
        y += 50

        if self.ego_vehicle is None:
            return

        velocity = self.ego_vehicle.get_velocity()
        speed_kmh = math.sqrt(velocity.x ** 2 + velocity.y ** 2) * 3.6
        front_dist = self._get_front_distance()
        target_wp = self._get_target_waypoint()
        lateral_offset = self._calc_lateral_offset(self.ego_vehicle.get_transform().location, target_wp)

        ep_avg = (self.current_episode_reward_sum / max(1, self.current_episode_steps))

        items = [
            ("Episode", f"{self.episode_count}", (100, 200, 255)),
            ("Step", f"{self.current_step}", (200, 200, 200)),
            ("WP", f"{self.current_waypoint_idx}/{len(self.route_waypoints)}", (100, 255, 100)),
            ("", "", None),

            ("Speed", f"{speed_kmh:.1f} km/h", (0, 255, 0) if speed_kmh > 30 else (255, 100, 100)),
            ("Target", f"{self.config.reward.target_speed:.0f} km/h", (150, 150, 150)),
            ("Front", f"{front_dist:.1f} m",
             (0, 255, 0) if front_dist > 15 else ((255, 255, 0) if front_dist > 8 else (255, 0, 0))),
            ("Lateral", f"{lateral_offset:.2f} m",
             (0, 255, 0) if abs(lateral_offset) < 2 else (
                 (255, 255, 0) if abs(lateral_offset) < 4 else (255, 100, 100))),
            ("", "", None),

            ("Reward", f"{self.last_reward:.3f}", (200, 220, 255)),
            ("Ep Avg Rwd", f"{ep_avg:.3f}", (200, 220, 255)),
            ("Prev Ep Avg", f"{self.prev_episode_avg_reward:.3f}", (160, 160, 200)),
            ("", "", None),

            ("Total Reward", f"{self.total_reward:.1f}", (0, 255, 0) if self.total_reward > 0 else (255, 100, 100)),
            ("Collisions", f"{self.collision_count}", (0, 255, 0) if self.collision_count == 0 else (255, 100, 100)),
            ("Successes", f"{self.success_count}", (0, 255, 180)),
            ("Near Misses", f"{self.near_miss_count}", (255, 200, 0)),
            ("", "", None),

            ("Throt/Brk", f"{self.current_action[1]:.2f}",
             (100, 255, 100) if self.current_action[1] >= 0 else (255, 100, 100)),
            ("Steer", f"{self.current_action[0] * self.config.action.max_steer:.3f}", (255, 200, 100)),
        ]

        for label, value, color in items:
            if label == "":
                y += 10
                continue
            label_surf = self.font_small.render(f"{label}:", True, (180, 180, 180))
            self.display.blit(label_surf, (panel_x + 10, y + 3))
            value_surf = self.font.render(value, True, color if color else (255, 255, 255))
            self.display.blit(value_surf, (panel_x + 150, y))
            y += 28

        y += 10
        self._draw_steering_bar(panel_x + 10, y, panel_width - 20, self.current_action[0])

        # ============================================================
        # [NEW] SVO推断信息面板
        # ============================================================
        y += 30
        svo_info = getattr(self, '_svo_display_info', {})
        if svo_info:
            svo_mu = svo_info.get('svo_mu', [])
            svo_sigma = svo_info.get('svo_sigma', [])
            interact_mask = svo_info.get('interact_mask', [])

            svo_title = self.font_small.render("--- SVO Inference ---", True, (180, 220, 255))
            self.display.blit(svo_title, (panel_x + 10, y))
            y += 22

            n_interactive = sum(interact_mask) if len(interact_mask) > 0 else 0
            risk_val = getattr(self, '_svo_risk_penalty', 0.0)
            
            info_items = [
                ("Interactive", f"{n_interactive}", (255, 200, 100)),
                ("Risk", f"{risk_val:.2f}", (255, 100, 100) if risk_val > 0.3 else (100, 255, 100)),
            ]
            for label, value, color in info_items:
                lbl = self.font_small.render(f"{label}:", True, (160, 160, 160))
                val = self.font_small.render(value, True, color)
                self.display.blit(lbl, (panel_x + 10, y))
                self.display.blit(val, (panel_x + 120, y))
                y += 20

            # 显示每辆交互车的SVO (新增: 每辆的相对位置, 方便核对)
            # 为此需要重建 selected 列表, 与 _get_observation 中的顺序一致
            # [v8] 不再用 _start_lane_forward/right (已删), 改用 ego 自身朝向
            ego_loc_now = self.ego_vehicle.get_location() if self.ego_vehicle else None
            det_r = self.config.observation.detection_radius
            selected_info = []  # [(rel_long, rel_lat, dist, lane_id, is_obs), ...]
            if ego_loc_now is not None:
                ego_fwd_vec = self.ego_vehicle.get_transform().get_forward_vector()
                fnorm = math.sqrt(ego_fwd_vec.x ** 2 + ego_fwd_vec.y ** 2) + 1e-8
                fx, fy = ego_fwd_vec.x / fnorm, ego_fwd_vec.y / fnorm
                rx_s, ry_s = fy, -fx   # 右向 = 前向顺时针转 90°

                cands_dbg = []
                if self.obstacle_vehicle and self.obstacle_vehicle.is_alive:
                    d_obs = self.obstacle_vehicle.get_location().distance(ego_loc_now)
                    cands_dbg.append((self.obstacle_vehicle, d_obs, True))
                for npc in self.npc_vehicles:
                    if not npc.is_alive:
                        continue
                    d = npc.get_location().distance(ego_loc_now)
                    if d < det_r:
                        cands_dbg.append((npc, d, False))
                cands_dbg.sort(key=lambda x: x[1])
                for veh, d, is_obs in cands_dbg[:5]:
                    vl = veh.get_location()
                    dx = vl.x - ego_loc_now.x
                    dy = vl.y - ego_loc_now.y
                    rel_long = dx * fx + dy * fy
                    rel_lat = dx * rx_s + dy * ry_s
                    try:
                        wp = self.map.get_waypoint(vl)
                        lid = wp.lane_id if wp else 0
                    except Exception:
                        lid = 0
                    selected_info.append((rel_long, rel_lat, d, lid, is_obs))

            # 显示每辆交互车的SVO
            for i in range(min(len(svo_mu), 5)):
                if np.abs(svo_mu[i] - 45.0) < 0.1 and not interact_mask[i]:
                    continue  # 跳过默认先验值(无真实NPC)
                mu_val = svo_mu[i]
                # SVO角度→颜色: 红(0°)→黄(45°)→绿(90°)
                r = int(max(0, min(255, 255 - mu_val * 255 / 90)))
                g = int(max(0, min(255, mu_val * 255 / 90)))
                color = (r, g, 80)
                tag = "L1" if interact_mask[i] else "L0"
                # SVO角度→风格标签
                if mu_val < 25:
                    style = "Aggr"
                elif mu_val < 40:
                    style = "S-A"
                elif mu_val < 55:
                    style = "Norm"
                elif mu_val < 70:
                    style = "S-C"
                else:
                    style = "Cons"

                # [新] 附加每辆NPC的位置信息, 方便核对可视化
                pos_str = ""
                if i < len(selected_info):
                    rel_long, rel_lat, d, lid, is_obs = selected_info[i]
                    fb = "前" if rel_long > 0 else "后"
                    lr = "右" if rel_lat > 0 else "左"
                    obs_tag = "[OBS]" if is_obs else ""
                    pos_str = f" {fb}{abs(rel_long):.0f}m{lr}{abs(rel_lat):.1f}m L{lid}{obs_tag}"

                txt = f"N{i}:{mu_val:.0f}°({style})[{tag}]{pos_str}"
                surf = self.font_small.render(txt, True, color)
                self.display.blit(surf, (panel_x + 10, y))
                y += 18

    def _draw_steering_bar(self, x, y, width, steer):
        pygame.draw.rect(self.display, (60, 60, 70), (x, y, width, 20))
        center_x = x + width // 2
        pygame.draw.line(self.display, (150, 150, 150), (center_x, y), (center_x, y + 20), 2)

        steer_x = center_x + int(steer * width / 2)
        bar_color = (255, 200, 100) if abs(steer) < 0.5 else (255, 100, 100)

        pygame.draw.rect(self.display, bar_color, (min(center_x, steer_x), y + 4, abs(steer_x - center_x), 12))
        pygame.draw.circle(self.display, (255, 255, 255), (steer_x, y + 10), 6)

    def _draw_pause_screen(self):
        overlay = pygame.Surface((self.config.visual.width, self.config.visual.height))
        overlay.set_alpha(128)
        overlay.fill((0, 0, 0))
        self.display.blit(overlay, (0, 0))
        text = self.font_large.render("PAUSED - Press P", True, (255, 255, 255))
        rect = text.get_rect(center=(self.config.visual.width // 2, self.config.visual.height // 2))
        self.display.blit(text, rect)

    def _cleanup(self):
        for sensor in self.sensors.values():
            if sensor is not None and sensor.is_alive:
                sensor.stop()
                sensor.destroy()
        self.sensors.clear()

        if self.camera_sensor is not None and self.camera_sensor.is_alive:
            self.camera_sensor.stop()
            self.camera_sensor.destroy()
        self.camera_sensor = None

        for actor in self.actor_list:
            if actor is not None and actor.is_alive:
                actor.destroy()
        self.actor_list.clear()

        self.npc_vehicles.clear()
        self.obstacle_vehicle = None
        self.route_waypoints.clear()
        self.ego_vehicle = None

    def close(self):
        self._cleanup()
        if self.world is not None:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
        if self.display is not None:
            pygame.quit()
            self.display = None
        print("环境已关闭")


def make_env(config: Config = None, seed: int = None) -> CarlaEnv: #环境工厂函数，支持传入配置和随机种子
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    return CarlaEnv(config)
