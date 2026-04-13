"""
modules - 核心算法模块包
各模块由对应分工成员实现，通过 interfaces.py 中定义的数据结构通信。

模块说明：
- detector      : [B] one-class 螺丝检测器（YOLO + SAHI）
- registration  : [A] 锚帧几何配准（AKAZE + Homography）
- dedup         : [A] 全局去重聚类（投影坐标 + DBSCAN）
- classifier    : [C] 5 类螺丝分类器（Lab2 迁移 + fine-tune）
"""

from .detector import Detector
from .registration import FrameRegistration
from .dedup import GlobalDedup
from .classifier import ScrewClassifier

__all__ = [
    "Detector",
    "FrameRegistration",
    "GlobalDedup",
    "ScrewClassifier",
]
