# 奖励函数 / CVaR / SPO 调试索引

适用项目：`SVO-CVaR`

用途：记录本项目中所有和奖励函数、CVaR 约束、SPO 动态调节有关的代码位置、公式、默认参数和调试方向。

---

## 1. 总览

| 模块 | 作用 | 主要代码位置 | 主要配置 |
|---|---|---|---|
| Ego / L2 reward 总入口 | 根据 `reward_mode` 选择 `l1` / `l2` 奖励 | `src/envs/carla_env.py::_calculate_reward` | `config.klevel.reward_mode` |
| Ego reward | 速度、车道保持、安全距离、转向平滑、碰撞、时间/进度 | `src/envs/carla_env.py::_calculate_ego_reward` | `config.reward` |
| L2 SVO reward | `cos(theta) * r_ego + sin(theta) * r_others_norm * 2.0` | `src/envs/carla_env.py::_calculate_reward` | `config.reward.use_oracle_svo`, `config.klevel` |
| R_others | 用 IDM 计算 ego 行为对后车收益的影响 | `src/envs/carla_env.py::_compute_others_reward` | `config.klevel.idm_*` |
| L1 BV reward | 背景车训练专用巡航 reward | `src/envs/bv_env.py::_calculate_bv_reward` | `config.bv_reward` |
| Terminal adjustment | 成功奖励、跨两车道/车道违规终止惩罚 | `src/envs/carla_env.py::step` | `config.reward.reach_goal_reward` 等 |
| CVaR step cost | 安全 cost，不直接加到 reward，而是进 buffer | `src/envs/carla_env.py::step` | `config.ppo.cost_w_*` |
| CVaR dual update | episode cost / budget / CVaR / lambda 更新 | `src/algorithms/ppo_model.py::_cvar_dual_update` | `config.ppo.cvar_*`, `svo_budget_*` |
| SPO epsilon | 固定或 SVO-adaptive trust region 宽度 | `src/algorithms/spo_model.py::_compute_spo_epsilon` | `config.ppo.spo_*` |
| SPO policy loss | 用二次惩罚替换 PPO ratio clipping | `src/algorithms/spo_model.py::update` | `config.ppo.use_spo` |

---

## 2. 训练阶段和奖励模式

| 阶段 / 命令 | reward_mode | SVO | CVaR | 环境 | 奖励函数 |
|---|---:|---:|---:|---|---|
| `python train.py --stage l1` | `l1_bv` | 关闭 | 强制关闭 | `BVTrainEnv` | `_calculate_bv_reward` |
| `python train.py --stage svo_only` | `l2` | 开启 | 默认开启，除非 `--no_cvar` | `CarlaEnv` | `_calculate_reward` -> L2 |
| `python train.py --stage l2` | `l2` | 开启 | 默认开启，除非 `--no_cvar` | `CarlaEnv` | `_calculate_reward` -> L2 |
| 旧 ego baseline | `l1` | 可关 | 看配置 | `CarlaEnv` | `_calculate_reward` -> `_calculate_ego_reward` |

代码入口：`src/scripts/train.py`

关键开关：

| 参数 | 作用 |
|---|---|
| `--algo ppo` | 标准 PPO |
| `--algo spo` | 启用 SPO |
| `--algo dapo` | 启用 DAPO |
| `--no_svo_adaptive_spo` | `--algo spo` 时关闭 SVO-adaptive epsilon，退化为 fixed SPO |
| `--no_cvar` | 手动关闭 CVaR |
| `--no_svo` | 关闭 SVO 模块 |
| `--use_oracle_svo` | L2 reward 的 theta 使用 NPC 真实风格标签 |
| `--no_oracle_svo` | L2 reward 的 theta 使用 BIRL 推断 `svo_mu` |

---

## 3. Ego Reward

代码位置：`src/envs/carla_env.py::_calculate_ego_reward`

配置位置：`src/config.py::RewardConfig`

公式结构：

```text
r_ego =
  speed_reward
  + lane_keeping_reward_or_penalty
  + front_distance_penalty
  + steering_smooth_penalty
  + collision_penalty_if_collision
  + progress_reward
  + time_penalty
```

| 项 | 当前默认值 | 配置字段 | 代码逻辑 | 调大后效果 | 调小时效果 |
|---|---:|---|---|---|---|
| 目标速度 | `60.0 km/h` | `reward.target_speed` | 越接近目标速度奖励越高 | 更偏向高速巡航 | 更容易慢速保守 |
| 最低速度阈值 | `10.0 km/h` | `reward.min_speed` | 低于阈值加低速惩罚 | 防止龟速更强 | 容忍低速 |
| 速度奖励权重 | `1.5` | `reward.speed_reward_weight` | `max(0, 1 - speed_diff) * weight` | 更追求速度 | 安全/车道项相对更重要 |
| 低速惩罚 | `-0.3` | `reward.low_speed_penalty` | `speed < min_speed` 时触发 | 更不愿停车 | 更容易停/慢行 |
| 车道保持权重 | `0.5` | `reward.lane_keeping_weight` | 横向偏移小于阈值时给正奖励 | 更贴近参考车道 | 更愿意偏移/变道 |
| 横向容忍 | `1.0 m` | `reward.max_lateral_offset` | 超出后线性惩罚 | 容忍偏移范围变大 | 车道约束更紧 |
| 终止横向偏移 | `4.0 m` | `reward.terminal_lateral_offset` | 超出触发 out_of_lane | 更不容易车道违规终止 | 更容易终止 |
| 安全距离权重 | `0.5` | `reward.safe_distance_weight` | 前车距离不足时线性惩罚 | 跟车更保守 | 跟车更激进 |
| 最小安全距离 | `8.0 m` | `reward.min_safe_distance` | 小于该距离开始惩罚 | 更早保持距离 | 更贴近前车 |
| 时距目标 | `2.0 s` | `reward.desired_headway_time` | 用于 `safe_distance_target` 和 CVaR cost | 高速时更保守 | 高速时更贴近 |
| 近前车惩罚 | `-0.1` | `reward.near_front_penalty` | 距离在 `[min_safe, 2*min_safe)` 时触发 | 更讨厌近距离跟车 | 更容忍跟车 |
| 碰撞惩罚 | `-200.0` | `reward.collision_penalty` | 发生碰撞时一次性加入 reward | 更强避碰 | 可能更激进 |
| 转向平滑权重 | `0.1` | `reward.steering_penalty_weight` | 惩罚 `abs(steer - prev_steer)` | 转向更平滑 | 转向更灵活 |
| 每步进度奖励 | `0.05` | `reward.progress_reward` | 每步固定小奖励 | 更愿意活着/前进 | 减弱拖时间倾向 |
| 每步时间惩罚 | `-0.05` | `reward.time_penalty` | 每步固定惩罚 | 更快完成任务 | 更不急 |
| 成功奖励 | `100.0` | `reward.reach_goal_reward` | done 且 success 时终端奖励 | 更追求到达终点 | 成功诱导弱 |
| 跨两车道终止惩罚 | `-120.0` | `reward.terminal_cross_two_lanes_penalty` | `termination_reason == cross_two_lanes` | 更避免大幅跨车道 | 更容忍跨车道 |
| 车道违规终止惩罚 | `-10.0` | `reward.terminal_lane_violation_penalty` | 车道违规终止时触发 | 更避免出界 | 车道违规学习信号弱 |

调试建议：

| 现象 | 优先检查 / 调整 |
|---|---|
| 车速上不去 | `speed_reward_weight`、`target_speed`、`low_speed_penalty`、动作限速器 |
| 频繁撞车 | `collision_penalty`、`safe_distance_weight`、`min_safe_distance`、CVaR 的 `cost_w_collision` |
| 跟车太近 | `min_safe_distance`、`desired_headway_time`、`safe_distance_weight`、`cost_w_safe` |
| 不愿意变道 / 动作太保守 | 降低 `lane_keeping_weight`、降低 `steering_penalty_weight`、检查 CVaR/SPO 是否过紧 |
| 乱打方向 | 提高 `steering_penalty_weight`，或检查 action smooth 参数 |

---

## 4. L2 SVO Reward

代码位置：`src/envs/carla_env.py::_calculate_reward`

公式：

```text
R_L2 = cos(theta) * r_ego + sin(theta) * r_others_norm * 2.0
r_others_norm = clip(r_others / 6.0, -1.0, 1.0)
```

theta 来源：

| 模式 | 配置 | 来源 | 说明 |
|---|---|---|---|
| Oracle | `reward.use_oracle_svo=True` | NPC 真实 style label | 训练推荐，稳定 |
| BIRL | `reward.use_oracle_svo=False` | `self._svo_mu` 推断均值 | 测试/消融，验证 SVO 推断质量 |

Oracle style -> theta 映射：

| NPC 风格 | theta |
|---|---:|
| `aggressive` | `15 deg` |
| `semi_aggressive` | `30 deg` |
| `normal` | `45 deg` |
| `semi_conservative` | `60 deg` |
| `conservative` | `75 deg` |

调试含义：

| theta 趋势 | reward 权重变化 | 行为倾向 |
|---|---|---|
| theta 越小 | `cos(theta)` 大，`sin(theta)` 小 | ego reward 主导，更自利/激进 |
| theta 越大 | `cos(theta)` 小，`sin(theta)` 大 | others reward 主导，更照顾 NPC |

L2 debug info 会写入 `info`：

| 字段 | 含义 |
|---|---|
| `r_ego` | ego reward 原始值 |
| `r_others` | 他车收益原始值 |
| `r_others_norm` | clip 后的他车收益 |
| `theta_deg` | 当前使用的 SVO 角度 |
| `r_total` | L2 总 reward |

---

## 5. R_others / IDM 社会项

代码位置：`src/envs/carla_env.py::_compute_others_reward` 和 `_idm_acceleration`

作用：衡量 ego 当前行为对后车的影响。正值表示 ego 让后车更容易行驶，负值表示 ego 迫使后车减速。

关键配置：`src/config.py::KLevelConfig`

| 参数 | 默认值 | 含义 | 调试方向 |
|---|---:|---|---|
| `idm_desired_speed` | `25.0 m/s` | IDM 期望速度 | 高则后车更想快走 |
| `idm_time_headway` | `1.5 s` | 安全时距 | 高则后车更保守，ego 插入更容易被判负面 |
| `idm_max_accel` | `2.0 m/s^2` | 最大加速度 | 高则后车加速能力强 |
| `idm_comfort_decel` | `3.0 m/s^2` | 舒适减速度 | 影响急减速惩罚 |
| `idm_min_gap` | `2.0 m` | 最小间距 | 高则社会项更保守 |
| `idm_accel_exponent` | `4.0` | IDM 加速指数 | 一般不优先调 |

---

## 6. L1 BV Reward

代码位置：`src/envs/bv_env.py::_calculate_bv_reward`

配置位置：`src/config.py::BVRewardConfig`

公式结构：

```text
R_BV =
  speed_cruise_reward
  + lane_keeping_reward_or_penalty
  + front_distance_penalty
  + lane_change_penalty
  + steering_smoothness_penalty
  + steering_magnitude_penalty
  + collision_penalty_if_collision
  + time_alive_reward
```

| 项 | 默认值 | 配置字段 | 代码逻辑 | 调试方向 |
|---|---:|---|---|---|
| 目标速度 | `60.0 km/h` | `bv_reward.target_speed` | 接近目标速度奖励高 | 控制 BV 巡航速度 |
| 最低速度 | `5.0 km/h` | `bv_reward.min_speed` | 低于触发惩罚 | 防止 BV 学会停车 |
| 速度奖励权重 | `1.0` | `bv_reward.speed_reward_weight` | 速度奖励尺度 | 高则 BV 更积极巡航 |
| 低速惩罚 | `-0.3` | `bv_reward.low_speed_penalty` | 低速惩罚 | 更负则更不愿慢行 |
| 碰撞惩罚 | `-200.0` | `bv_reward.collision_penalty` | 碰撞终止级惩罚 | 更负则更强避碰 |
| 最小安全距离 | `6.0 m` | `bv_reward.min_safe_distance` | 前车距离不足惩罚 | 高则 BV 更保守 |
| 近前车惩罚 | `-0.1` | `bv_reward.near_front_penalty` | 近距离但未过近 | 更负则跟车更远 |
| 车道保持权重 | `0.8` | `bv_reward.lane_keeping_weight` | 车道内正奖励 | 高则 BV 更爱保持车道 |
| 横向容忍 | `1.0 m` | `bv_reward.max_lateral_offset` | 超出后软惩罚 | 小则更严格 |
| 终止横向偏移 | `4.0 m` | `bv_reward.terminal_lateral_offset` | 严重偏离终止 | 小则更容易终止 |
| 车道违规终止惩罚 | `-10.0` | `bv_reward.terminal_lane_violation_penalty` | 配置存在，但当前 BV done 里主要返回 info | 如需终端惩罚需确认是否接入 reward |
| 变道惩罚 | `-0.5` | `bv_reward.lane_change_penalty` | lane id 变化时触发 | 更负则 BV 更少变道 |
| 变道检测距离 | `2.5 m` | `bv_reward.lane_change_detection_dist` | 配置存在，当前代码主要用 lane id 检测 | 当前不是主要生效项 |
| 转向幅度惩罚 | `0.15` | `bv_reward.steering_penalty_weight` | `abs(steer) * weight * 0.1` | 高则 BV 少大转向 |
| 转向变化惩罚 | `0.05` | `bv_reward.steering_smoothness_weight` | `abs(steer - prev_steer) * weight` | 高则动作更平滑 |
| 存活奖励 | `0.02` | `bv_reward.time_alive_reward` | 每步小奖励 | 高则更鼓励持续巡航 |
| 进度奖励 | `0.0` | `bv_reward.progress_reward` | 当前配置存在，reward 中未实际使用 | 一般不用调 |

---

## 7. CVaR Step Cost

代码位置：`src/envs/carla_env.py::step`

注意：CVaR 用的是 `cost_step`，不是环境 reward。它会通过 `agent.buffer.add(... cost=info['cost_step'] ...)` 进入 buffer。

公式：

```text
safe_target = max(
  reward.min_safe_distance,
  speed_kmh / 3.6 * reward.desired_headway_time
)

shortfall = max(0, safe_target - front_distance) / (safe_target + cost_eps)

cost_step =
  cost_w_collision * collision_step
  + cost_w_lane * lane_violation_step
  + cost_w_safe * shortfall
```

配置位置：`src/config.py::PPOConfig`

| 参数 | 默认值 | 作用 | 调大后效果 |
|---|---:|---|---|
| `cost_w_collision` | `1.0` | 碰撞 cost 权重 | CVaR 更强压制碰撞 |
| `cost_w_lane` | `0.5` | 车道违规 cost 权重 | 更强压制出界/跨两车道 |
| `cost_w_safe` | `0.2` | 安全距离不足 cost 权重 | 更强压制近距离跟车 |
| `cost_eps` | `1e-6` | 防除零 | 通常不调 |

---

## 8. CVaR Episode Constraint

代码位置：`src/algorithms/ppo_model.py::_cvar_dual_update`

核心流程：

```text
C_e = sum_t cost_gamma^t * cost_step_t
U_e = mean_t u_t
d_e = max(cvar_budget_base * exp(-svo_budget_beta * U_e),
          cvar_budget_base * cvar_budget_min_ratio)
normalized_cost = C_e / d_e

CVaR_alpha(normalized_cost) <= 1
lambda <- clamp(lambda + cvar_lambda_lr * (cvar_hat_norm - 1), 0, cvar_lambda_max)
```

| 参数 | 默认值 | 含义 | 调大后效果 | 调小时效果 |
|---|---:|---|---|---|
| `cvar_enabled` | `True` | CVaR 总开关 | 开启安全约束 | 关闭后只剩 reward 学习 |
| `cvar_alpha` | `0.1` | tail 比例 | 关注更大尾部集合，约束相对平均一些 | 更关注极端最坏样本 |
| `cost_gamma` | `0.99` | cost return 折扣 | 长期风险更重要 | 更看重短期风险 |
| `cvar_budget_base` | `2.0` | 基础预算 `d0` | 约束更宽松 | 约束更严格 |
| `cvar_budget_min_ratio` | `0.2` | `d_e` 下限比例 | 防止预算过小 | 过小可能导致 `C/d` 爆炸 |
| `cvar_lambda_init` | `0.0` | 初始 lambda | 初期更快压 cost | 初期更自由 |
| `cvar_lambda_lr` | `1e-3` | lambda 更新步长 | 约束反应更快，但可能振荡 | 约束反应慢 |
| `cvar_lambda_max` | `0.5` | lambda 上限 | 允许更强安全惩罚 | 限制 CVaR 干预 |
| `cvar_cost_coef` | `1.0` | cost actor 项权重 | 更强降低 cost | CVaR 影响变弱 |
| `cost_loss_coef` | `1.0` | cost critic loss 权重 | cost value 学得更重 | cost critic 影响弱 |

TensorBoard 指标：

| 指标 | 含义 |
|---|---|
| `cvar/cvar_hat_norm` | 归一化 CVaR，目标是小于等于 `1` |
| `cvar/cvar_hat_unnorm` | 未归一化 episode cost 的 CVaR |
| `cvar/lambda` | Lagrange 乘子，越大代表安全约束压力越强 |
| `cvar/avg_budget` | SVO-adjusted 平均安全预算 |
| `cvar/avg_episode_cost` | 平均 episode cost |
| `cvar/avg_svo_budget_term` | 平均 SVO 紧迫度 |
| `cvar/cost_critic_loss` | cost critic 训练损失 |
| `cvar/n_episodes_in_rollout` | 当前 rollout 内 episode 数 |

调试建议：

| 现象 | 优先调整 |
|---|---|
| `cvar_hat_norm` 长期大于 1 且事故多 | 提高 `cvar_lambda_lr` 或 `cvar_lambda_max`，提高对应 `cost_w_*` |
| lambda 很快顶到上限，策略太保守 | 提高 `cvar_budget_base`，降低 `cvar_cost_coef`，或降低 `cost_w_safe` |
| lambda 基本为 0 但仍不安全 | 检查 `cost_step` 是否真的非零，尤其 `collision_step`、`lane_violation_step`、`front_distance` |
| cost critic loss 很大 | 降低 `cost_loss_coef` 或检查 cost 尺度是否过大 |

---

## 9. SVO Urgency / Budget Term

代码位置：`src/scripts/train.py::_compute_svo_budget_term`

用途：同一个 `u_t` 同时给 CVaR 和 SPO 用。

```text
u_t = mean_active_neighbors(
  svo_mu_budget_weight * (1 - mu_i / 90)
  + svo_sigma_budget_weight * (sigma_i / prior_sigma)
)
```

| 用途 | 使用方式 |
|---|---|
| CVaR | `U_e = mean_t u_t`，再用 `d_e = d0 * exp(-beta * U_e)` 动态收缩安全预算 |
| SPO | `rho_t = clamp(u_t, 0, 1)`，再用 `epsilon_t = epsilon_base * exp(-alpha * rho_t)` 动态收缩 trust region |

配置：

| 参数 | 默认值 | 作用 | 调大后效果 |
|---|---:|---|---|
| `svo_mu_budget_weight` | `0.5` | 对低 SVO 均值，也就是更激进 NPC 的敏感度 | 激进交互时更快收紧 |
| `svo_sigma_budget_weight` | `0.5` | 对 SVO 不确定性的敏感度 | 不确定时更快收紧 |
| `svo_budget_beta` | `1.0` | CVaR budget 对 `U_e` 的指数收缩强度 | 高风险 episode 预算更小，约束更强 |
| `svo.prior_sigma` | `25.0` | sigma 归一化分母 | 越小则同样 sigma 产生更大 `u_t` |

---

## 10. SPO 动态可调部分

代码位置：

- `src/algorithms/spo_model.py::_compute_spo_epsilon`
- `src/algorithms/spo_model.py::update`
- 配置：`src/config.py::PPOConfig`

SPO 开关：

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `use_spo` | `False` | 由 `--algo spo` 设置为 True |
| `use_svo_adaptive_spo` | `True` | 是否启用 SVO-guided adaptive epsilon |

SPO epsilon：

```text
fixed mode:
  epsilon = spo_epsilon_base

adaptive mode:
  rho_t = clamp(svo_budget_term, 0, 1)
  epsilon_t = clip(
    spo_epsilon_base * exp(-spo_risk_alpha * rho_t),
    spo_epsilon_min,
    spo_epsilon_max
  )
```

| 参数 | 默认值 | 含义 | 调大后效果 | 调小时效果 |
|---|---:|---|---|---|
| `spo_epsilon_base` | `0.2` | 基础 trust region 宽度 | 策略更新更大胆 | 更新更保守 |
| `spo_epsilon_min` | `0.05` | adaptive 模式最小 epsilon | 高风险时仍允许一定更新 | 高风险时约束更紧 |
| `spo_epsilon_max` | `0.2` | adaptive 模式最大 epsilon | 低风险时允许更大更新 | 整体更保守 |
| `spo_risk_alpha` | `1.5` | `rho_t` 指数衰减强度 | 高风险时 epsilon 更快变小 | adaptive 效果更弱 |
| `use_svo_adaptive_spo` | `True` | 是否根据 SVO 风险调 epsilon | 高风险动态收紧 | 退化为 fixed SPO |

SPO policy loss：

```text
A_eff = A_R - lambda * cvar_cost_coef * A_C     # 如果 CVaR 开启
A_eff = A_R                                    # 如果 CVaR 关闭

objective =
  ratio * A_eff
  - abs(A_eff) * (ratio - 1)^2 / (2 * epsilon_t)

actor_loss = -mean(objective)
```

说明：

| 部分 | 作用 |
|---|---|
| `ratio` | 新旧策略概率比 |
| `epsilon_t` | trust region 宽度，越小越保守 |
| `A_eff` | reward advantage 和 CVaR cost advantage 的合成优势 |
| `lambda * cvar_cost_coef * A_C` | CVaR 对策略更新方向的安全约束 |

TensorBoard 指标：

| 指标 | 含义 |
|---|---|
| `policy/ratio_deviation_mean` | 平均 `abs(ratio - 1)` |
| `policy/ratio_deviation_max` | 最大 `abs(ratio - 1)` |
| `policy/spo_epsilon_mean` | 当前平均 epsilon |
| `policy/spo_epsilon_min` | 当前最小 epsilon |
| `policy/spo_epsilon_max` | 当前最大 epsilon |
| `policy/svo_risk_mean` | adaptive SPO 下平均 SVO 紧迫度 |
| `policy/svo_risk_max` | adaptive SPO 下最大 SVO 紧迫度 |

调试建议：

| 现象 | 优先调整 |
|---|---|
| ratio deviation 过大，训练不稳 | 降低 `spo_epsilon_base` / `spo_epsilon_max`，提高 `spo_risk_alpha` |
| 策略几乎不更新 | 提高 `spo_epsilon_min` 或 `spo_epsilon_base`，降低 `spo_risk_alpha` |
| 高风险场景仍太激进 | 降低 `spo_epsilon_min`，提高 `spo_risk_alpha`，提高 `svo_mu_budget_weight` / `svo_sigma_budget_weight` |
| adaptive 和 fixed 差别不明显 | 检查 `avg_svo_budget_term` / `policy/svo_risk_mean` 是否接近 0；若是，SVO 紧迫度没打起来 |

---

## 11. PPO / SPO / CVaR 共用训练参数

配置位置：`src/config.py::PPOConfig`

| 参数 | 默认值 | 作用 | 调试方向 |
|---|---:|---|---|
| `actor_lr` | `3e-4` | actor 学习率 | 不稳则降，学慢则升 |
| `critic_lr` | `3e-4` | critic / cost critic 学习率 | value loss 大可适当调 |
| `gamma` | `0.99` | reward return 折扣 | 长期任务通常保持 |
| `gae_lambda` | `0.95` | GAE 平滑 | 高更平滑，低更偏短期 |
| `clip_epsilon` | `0.2` | PPO clipping / value clipping / cost critic clipping | SPO 中仍用于 critic clipping |
| `entropy_coef` | `0.01` | 探索强度 | 策略早熟可提高 |
| `value_coef` | `0.5` | reward critic loss 权重 | critic 影响强弱 |
| `max_grad_norm` | `0.5` | 梯度裁剪 | 不稳可降低 |
| `rollout_steps` | `2048` | 每次更新采样步数 | 大更稳定，小更新更频繁 |
| `mini_batch_size` | `64` | mini-batch | 大更稳定，小噪声更大 |
| `ppo_epochs` | `10` | 每轮 rollout 训练 epoch | 高利用率但更易过拟合旧数据 |
| `normalize_advantage` | `True` | advantage 归一化 | 通常保持 |
| `target_kl` | `0.02` | KL early stop | 过小会频繁早停 |

---

## 12. 快速定位表

| 想调什么 | 改哪里 | 看什么指标 |
|---|---|---|
| ego 速度 / 巡航积极性 | `RewardConfig.target_speed`, `speed_reward_weight`, `low_speed_penalty` | `episode/reward`, `speed_kmh` |
| ego 安全距离 | `RewardConfig.min_safe_distance`, `desired_headway_time`, `safe_distance_weight` | `front_distance`, `cost_step`, `cvar/avg_episode_cost` |
| 碰撞惩罚 | `RewardConfig.collision_penalty`, `PPOConfig.cost_w_collision` | collision rate, `cvar/cvar_hat_norm` |
| 车道违规 | `terminal_lateral_offset`, `terminal_cross_two_lanes_penalty`, `cost_w_lane` | lane violation, `terminal_penalty` |
| BV 是否乱变道 | `BVRewardConfig.lane_change_penalty`, `lane_keeping_weight` | `lane_change_count` |
| L2 社会性强弱 | `reward.use_oracle_svo`, SVO theta 分布, `KLevelConfig.idm_*` | `theta_deg`, `r_others_norm` |
| CVaR 约束强弱 | `cvar_budget_base`, `cvar_lambda_lr`, `cvar_lambda_max`, `cvar_cost_coef` | `cvar/lambda`, `cvar/cvar_hat_norm` |
| CVaR 对 SVO 风险敏感度 | `svo_mu_budget_weight`, `svo_sigma_budget_weight`, `svo_budget_beta` | `cvar/avg_svo_budget_term`, `cvar/avg_budget` |
| SPO 更新幅度 | `spo_epsilon_base`, `spo_epsilon_min`, `spo_epsilon_max` | `policy/ratio_deviation_*`, `policy/spo_epsilon_*` |
| SPO 高风险收紧强度 | `spo_risk_alpha`, `use_svo_adaptive_spo` | `policy/svo_risk_mean`, `policy/spo_epsilon_mean` |

---

## 13. 推荐调试顺序

1. 先确认奖励模式：`reward_mode` 是否符合阶段。
2. 看环境侧 reward 分解：`r_ego`、`r_others`、`theta_deg`、`r_total`。
3. 看安全 cost 是否生效：`cost_step`、`collision_step`、`lane_violation_step`。
4. 看 CVaR 是否真正约束：`cvar_hat_norm` 是否围绕 1，`lambda` 是否有变化。
5. 如果用 SPO，看 `ratio_deviation_*` 和 `spo_epsilon_*`，判断 trust region 是否过松/过紧。
6. 最后才大改 reward 权重；优先小步调单个参数，避免多个信号同时变化导致难定位。

