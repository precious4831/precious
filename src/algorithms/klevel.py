"""
klevel.py -- Level-k 策略控制器 (v9 - 真博弈版)

========================================================================
v9 vs v6 的核心变化:
  v6: BV 用 ego 的 L1 PPO 策略, 因为"BV 不该变道", 所以纵横解耦 (steer 用 PID)
  v9: BV 用 BV 专属的 L1 策略 (来自 train.py --stage l1), 直接执行其完整动作
      → BV 任务和 ego 任务不同, 不会乱变道, 不需要 PID 兜底
      → 同时缓存最新动作供 ego 观测 (a_BV) 拼接, 实现真正的 Level-k best-response

========================================================================
在完整训练流程中的位置:
  L1 BV 训练 (train.py --stage l1):  BV 由 TM 控制 (= L0), 训练 BV 策略
                                      不使用 LevelKController
  L2 ego 训练 (train.py --stage l2):  BV 由 LevelKController 控制 (= 加载好的 L1 BV)
                                      ego obs 拼接 a_BV → 真博弈
  测试    (test.py):                   BV 由 LevelKController 控制
========================================================================

设计 (大幅简化, 对标开源 Stackelberg 实现):
  1. 每个 step:
     a. 筛选 ego 附近 control_radius 内的 NPC
     b. 为这些 NPC 构建观测 (env.build_observation_for(npc))
     c. 用 L1 BV 策略批量推理出 [steer, throttle_brake]
     d. 直接 apply 完整动作 (不再 PID 解耦)
     e. 缓存 last_actions[npc.id] 供 env 在构建 ego obs 时拼接 a_BV
  2. 不再有预热 (L1 BV 训练时已经见过"周围全 TM"的初始分布, 接管即可工作)
  3. 不再有低速兜底 (L1 BV 自己学会了维持速度)
  4. 不再有 PID (L1 BV 自己学会了车道保持)

参考: Bouton 2020 / CHARMS 2024 (BV 是和 ego 对称的、独立训练的策略)
"""

import math
import numpy as np

try:
    import carla
except ImportError:
    pass


class LevelKController:
    """
    Level-k BV 策略控制器 (v9 简化版).

    职责:
      1. 加载 L1 BV 策略权重 (来自 train.py --stage l1 输出)
      2. 每 step 控制 ego 附近的 NPC (用 L1 BV 推理替代 TM)
      3. 缓存每辆 BV 的最新动作, 供 env 在构建 ego obs 时拼接 a_BV
      4. 自动管理 NPC 接管 / 释放回 TM 的生命周期
    """

    def __init__(self, config, device=None):
        self.config = config
        self.device = device
        self.control_radius = config.klevel.bv_control_radius
        self.batch_inference = config.klevel.bv_batch_inference

        self.policy = None
        self._controlled_npcs = {}  # {npc_id: True}
        self._tm_port = None

        # === a_BV 缓存 (供 env._get_observation 拼接到 ego obs) ===
        # {npc_id: np.array([steer, throttle_brake], dtype=float32)}
        # 每个 step 推理后更新; npc 离开 control_radius 时移除
        self.last_actions = {}

    def load_policy(self, policy_path, config=None):
        """加载 L1 BV 策略权重.

        Args:
            policy_path: train.py --stage l1 训出的权重文件
            config: 可选, 用于构造 PPOAgent 的 config (默认用 self.config)

        重要: 加载的策略必须是 BV 任务训出来的 (svo.enabled=False),
        和 ego L1 baseline 不同. 用 train.py --stage l1 训出来的就是这个.
        """
        from src.algorithms.ppo_model import PPOAgent
        import torch

        cfg = config or self.config

        # BV 不带 SVO (BV 自己不推断别人风格)
        from copy import deepcopy
        l1_config = deepcopy(cfg)
        l1_config.svo.enabled = False
        # BV 训练时 obs 里的 bv_actions 段是全 0 的, 加载推理时也保持 True
        # 这样网络结构和 L2 时一致, 加载权重不会报维度不匹配

        self.policy = PPOAgent(l1_config)
        self.policy.load(policy_path)

        # 冻结网络 + eval 模式
        for param in self.policy.encoder.parameters():
            param.requires_grad = False
        for param in self.policy.actor_head.parameters():
            param.requires_grad = False
        self.policy.encoder.eval()
        self.policy.actor_head.eval()

        print(f"[LevelK-v9] L1 BV 策略已加载: {policy_path}")
        print(f"[LevelK-v9] 控制范围: {self.control_radius}m, 批量推理: {self.batch_inference}")
        print(f"[LevelK-v9] 模式: 完整动作执行 (无 PID 解耦, 无预热)")

    # ================================================================== #
    #  主入口: 每 step 调用                                                  #
    # ================================================================== #

    def step(self, ego_vehicle, npc_vehicles, world, env):
        """
        每 step 调用一次, 在 env.step() 内的 world.tick() 之前.

        流程:
          1. 筛选 ego 附近 control_radius 内的 NPC
          2. 为它们构建观测, 批量推理 L1 BV 动作
          3. 直接 apply (不 PID 解耦)
          4. 更新 last_actions 缓存
          5. 释放离开范围的 NPC 回 TM

        Returns:
            list of npc_ids: 当前 step 接管控制的 NPC ID 列表
        """
        if self.policy is None:
            return []

        if self._tm_port is None and hasattr(env, 'traffic_manager') and env.traffic_manager is not None:
            self._tm_port = env.traffic_manager.get_port()

        ego_loc = ego_vehicle.get_location()

        # 1) 筛选范围内的 NPC
        nearby_npcs = []
        for npc in npc_vehicles:
            if not npc.is_alive:
                continue
            d = ego_loc.distance(npc.get_location())
            if d <= self.control_radius:
                nearby_npcs.append(npc)

        if len(nearby_npcs) == 0:
            self._release_all_to_tm(npc_vehicles)
            return []

        # 2) 为每辆 NPC 构建观测
        obs_list = []
        valid_npcs = []
        for npc in nearby_npcs:
            try:
                obs_i = env.build_observation_for(npc)
                obs_list.append(obs_i)
                valid_npcs.append(npc)
            except Exception:
                continue

        if len(valid_npcs) == 0:
            return []

        # 3) 批量推理
        obs_batch = np.stack(obs_list, axis=0)
        actions = self.policy.batch_select_action(obs_batch, deterministic=True)
        # actions: (N, 2) → [steer_norm, throttle_brake_norm]

        # 4) 接管控制 + 应用完整动作 + 更新 last_actions
        controlled_ids = []
        current_ids = set()
        for i, npc in enumerate(valid_npcs):
            current_ids.add(npc.id)

            # 首次接管: 关闭 TM
            if npc.id not in self._controlled_npcs:
                try:
                    npc.set_autopilot(False)
                except RuntimeError:
                    continue
                self._controlled_npcs[npc.id] = True

            # 应用完整动作 (不 PID 解耦, 不低速兜底)
            self._apply_full_action(npc, actions[i])

            # 更新 a_BV 缓存
            self.last_actions[npc.id] = np.array(
                [float(actions[i, 0]), float(actions[i, 1])],
                dtype=np.float32,
            )

            controlled_ids.append(npc.id)

        # 5) 释放离开范围的 NPC 回 TM
        for npc_id in list(self._controlled_npcs.keys()):
            if npc_id not in current_ids:
                self._release_npc_to_tm(npc_id, npc_vehicles)
                # 同时清掉 a_BV 缓存
                self.last_actions.pop(npc_id, None)

        return controlled_ids

    # ================================================================== #
    #  控制应用                                                              #
    # ================================================================== #

    def _apply_full_action(self, npc, action):
        """直接应用 L1 BV 输出的完整动作 [steer, throttle_brake].

        v9 vs v6:
          v6: 只用 throttle, steer 用 PID 替换
          v9: steer 和 throttle 都用 L1 BV 输出
              因为 L1 BV 是用 BV reward 训出来的, 不会乱变道

        Args:
            npc: carla.Vehicle
            action: np.ndarray (2,) — [steer, throttle_brake], 范围 [-1, 1]
        """
        action_cfg = self.config.action

        # 解码 [-1, 1] → CARLA 控制
        steer = float(np.clip(action[0], -1.0, 1.0)) * getattr(action_cfg, 'steer_scale', 1.0)
        steer = float(np.clip(steer, -getattr(action_cfg, 'max_steer', 1.0),
                                       getattr(action_cfg, 'max_steer', 1.0)))

        throttle_brake = float(np.clip(action[1], -1.0, 1.0)) * action_cfg.throttle_brake_scale

        control = carla.VehicleControl()
        control.steer = steer

        if throttle_brake >= 0:
            control.throttle = min(throttle_brake, action_cfg.max_throttle)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(-throttle_brake, action_cfg.max_brake)

        control.hand_brake = False
        control.manual_gear_shift = False

        try:
            npc.apply_control(control)
        except RuntimeError:
            pass

    # ================================================================== #
    #  生命周期管理                                                          #
    # ================================================================== #

    def _release_npc_to_tm(self, npc_id, npc_vehicles):
        """将 NPC 交还 Traffic Manager."""
        for npc in npc_vehicles:
            if npc.id == npc_id and npc.is_alive:
                try:
                    if self._tm_port is not None:
                        npc.set_autopilot(True, self._tm_port)
                    else:
                        npc.set_autopilot(True)
                except RuntimeError:
                    pass
                break
        self._controlled_npcs.pop(npc_id, None)

    def _release_all_to_tm(self, npc_vehicles):
        """将所有策略控制的 NPC 交还 TM."""
        for npc_id in list(self._controlled_npcs.keys()):
            self._release_npc_to_tm(npc_id, npc_vehicles)
        self.last_actions.clear()

    def reset(self):
        """Episode 重置: 清空所有缓存."""
        self._controlled_npcs.clear()
        self.last_actions.clear()

    # ================================================================== #
    #  外部查询接口                                                          #
    # ================================================================== #

    def get_action_for_npc(self, npc_id):
        """获取某个 NPC 上一步的 BV 动作 (供 env 拼接到 ego obs).

        Args:
            npc_id: int, CARLA actor id

        Returns:
            np.ndarray (2,) [steer, throttle_brake] in [-1, 1]
            None 如果该 NPC 当前不被 LevelKController 控制
        """
        return self.last_actions.get(npc_id, None)

    def is_controlling(self, npc_id):
        """该 NPC 是否被 LevelKController 接管 (vs TM 控制)."""
        return npc_id in self._controlled_npcs

    @property
    def num_controlled(self):
        return len(self._controlled_npcs)
