"""
utils.py
通用工具：滑动平均、计时器、观测归一化器（RunningMeanStd）、随机种子设置。
"""

import time
import random

import numpy as np


class MovingAverage:
    """滑动窗口均值与标准差"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.values: list = []

    def add(self, value: float):
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.pop(0)

    def mean(self) -> float:
        return float(np.mean(self.values)) if self.values else 0.0

    def std(self) -> float:
        return float(np.std(self.values)) if len(self.values) > 1 else 0.0


class Timer:
    """训练计时器"""

    def __init__(self):
        self.start_time: float = 0.0
        self.elapsed: float = 0.0

    def start(self):
        self.start_time = time.time()

    def stop(self) -> float:
        if self.start_time > 0:
            self.elapsed = time.time() - self.start_time
            self.start_time = 0.0
        return self.elapsed

    def get_elapsed(self) -> float:
        if self.start_time > 0:
            return time.time() - self.start_time
        return self.elapsed


class RunningMeanStd:
    """在线计算均值与方差的Welford增量算法

    用于PPO观测归一化。normalize()内置NaN/Inf保护。
    """

    def __init__(self, shape: tuple, epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray):
        """用单条或一批观测更新统计量"""
        x = np.asarray(x, dtype=np.float64)
        # 跳过含NaN的观测
        if np.isnan(x).any():
            return
        if x.ndim == 1:
            batch_mean = x
            batch_var = np.zeros_like(x)
            batch_count = 1
        else:
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * (
            self.count * batch_count / total_count
        )
        self.mean = new_mean
        self.var = m2 / total_count
        self.count = total_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """零均值单位方差归一化，内置NaN/Inf保护"""
        std = np.sqrt(self.var + 1e-8)
        result = ((x - self.mean) / std).astype(np.float32)
        # 将任何NaN/Inf替换为0
        result = np.nan_to_num(result, nan=0.0, posinf=5.0, neginf=-5.0)
        return np.clip(result, -10.0, 10.0)

    def state_dict(self) -> dict:
        return {
            "mean": self.mean.copy(),
            "var": self.var.copy(),
            "count": self.count,
        }

    def load_state_dict(self, state: dict):
        self.mean = state["mean"].copy()
        self.var = state["var"].copy()
        self.count = state["count"]


def format_time(seconds: float) -> str:
    """将秒数格式化为 HH:MM:SS"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def set_random_seed(seed: int):
    """设置全局随机种子（Python, NumPy, PyTorch）"""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

class Rotator:
    """计算相对位置信息"""
    def __init__(self, ego_x,ego_y,ego_yaw_deg):
        self.ego_x = ego_x
        self.ego_y = ego_y
        #carla中航向角是以度为单位，且0度指向正北，顺时针增加，因此需要转换为弧度并调整为以正东为0度，逆时针增加
        self.ego_yaw_rad = np.radians(ego_yaw_deg)

        #计算旋转矩阵（逆时针旋转-ego_yaw_rad）
        # x' = dx*cos + dy*sin
        # y' = -dx*sin + dy*cos
        self.cos_a = np.cos(self.ego_yaw_rad)
        self.sin_a = np.sin(self.ego_yaw_rad)

    def transform_point(self, x_global, y_global):
        """
        转换位置坐标
        Global (世界坐标) -> Local (自车坐标)
        """
        # 1. 平移
        dx = x_global - self.ego_x
        dy = y_global - self.ego_y
        
        # 2. 旋转
        # x' = dx * cos + dy * sin
        # y' = -dx * sin + dy * cos
        x_local = dx * self.cos_a + dy * self.sin_a
        y_local = -dx * self.sin_a + dy * self.cos_a
        return x_local, y_local

    def transform_vector(self, vx_global, vy_global):
        """
        转换速度向量
        Global (世界速度) -> Local (相对自车速度方向)
        """
        vx_local = vx_global * self.cos_a + vy_global * self.sin_a
        vy_local = -vx_global * self.sin_a + vy_global * self.cos_a
        return vx_local, vy_local