"""
interfaces.py - 团队协作数据接口定义
所有模块通过此文件中的数据结构进行通信，禁止各模块私自定义等价结构。

接口 Owner: D（工程封装）
修改需通知全体成员。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


# ---------------------------------------------------------------------------
# 检测器输出（B 产出，A/C/D 消费）
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """
    单次螺丝检测结果。

    由 detector.py (B) 产生，随后传递给：
    - registration.py (A)：做坐标投影
    - dedup.py (A)：做全局去重聚类
    - classifier.py (C)：用 crop 做分类
    """

    frame_id: int
    """视频帧编号，从 0 开始，对应 VideoCapture 读取顺序。"""

    bbox: np.ndarray
    """
    检测框，形状 (4,)，格式 [x1, y1, x2, y2]，单位：像素。
    坐标基于**原始分辨率**（未缩放），与 crop 保持一致。
    """

    confidence: float
    """检测置信度，范围 [0, 1]。"""

    crop: np.ndarray
    """
    裁切的原图区域，形状 (H, W, 3)，BGR 色彩空间，uint8。
    由 detector.py 按 bbox 从原始分辨率帧上裁切，不做缩放。
    """

    track_id: int = -1
    """
    ByteTrack 分配的跨帧跟踪 ID。
    -1 表示该检测未经过跟踪（如仅做单帧推理时）。
    同一物理螺丝在相邻帧的 track_id 应相同。
    """

    def center(self) -> np.ndarray:
        """返回 bbox 中心点坐标 [cx, cy]（float32）。"""
        return np.array(
            [(self.bbox[0] + self.bbox[2]) / 2.0,
             (self.bbox[1] + self.bbox[3]) / 2.0],
            dtype=np.float32,
        )

    def area(self) -> float:
        """返回 bbox 面积（像素²）。"""
        return float(
            max(0.0, self.bbox[2] - self.bbox[0])
            * max(0.0, self.bbox[3] - self.bbox[1])
        )

    def __repr__(self) -> str:
        return (
            f"Detection(frame={self.frame_id}, "
            f"bbox=[{self.bbox[0]:.1f},{self.bbox[1]:.1f},"
            f"{self.bbox[2]:.1f},{self.bbox[3]:.1f}], "
            f"conf={self.confidence:.3f}, track_id={self.track_id})"
        )


# ---------------------------------------------------------------------------
# 配准模块输出（A 产出，A/D 消费）
# ---------------------------------------------------------------------------

@dataclass
class Registration:
    """
    单帧几何配准结果。

    由 registration.py (A) 产生，描述「当前帧 → 参考帧」的投影变换。
    D 在 pipeline.py 中用此结果将 Detection.bbox 投影到参考坐标系。
    """

    frame_id: int
    """对应的视频帧编号。"""

    H_to_ref: np.ndarray
    """
    3×3 Homography 矩阵（float64），将当前帧像素坐标变换到参考帧像素坐标。
    用法：
        pt_ref = H_to_ref @ [x, y, 1]^T  （需做齐次归一化）
    """

    valid: bool
    """
    配准是否有效。
    False 表示特征点太少或内点率过低，此帧应降级处理（如用 tracker 关联）。
    """

    inlier_ratio: float
    """
    RANSAC 内点比例，范围 [0, 1]。
    建议阈值：< 0.3 时将 valid 置为 False。
    """

    def project_point(self, pt: np.ndarray) -> Optional[np.ndarray]:
        """
        将单个点 [x, y] 从当前帧坐标投影到参考帧坐标。

        Parameters
        ----------
        pt : np.ndarray, shape (2,)

        Returns
        -------
        np.ndarray, shape (2,)，若 valid=False 则返回 None。
        """
        if not self.valid:
            return None
        p_h = np.array([pt[0], pt[1], 1.0], dtype=np.float64)
        q_h = self.H_to_ref @ p_h
        if abs(q_h[2]) < 1e-8:
            return None
        return (q_h[:2] / q_h[2]).astype(np.float32)

    def project_bbox(self, bbox: np.ndarray) -> Optional[np.ndarray]:
        """
        将 bbox [x1, y1, x2, y2] 的四个角点投影到参考帧，
        返回轴对齐的外接矩形 [x1', y1', x2', y2']。

        若 valid=False 返回 None。
        """
        if not self.valid:
            return None
        corners = np.array([
            [bbox[0], bbox[1]],
            [bbox[2], bbox[1]],
            [bbox[2], bbox[3]],
            [bbox[0], bbox[3]],
        ], dtype=np.float64)
        proj = []
        for c in corners:
            p = self.project_point(c)
            if p is None:
                return None
            proj.append(p)
        proj = np.array(proj)
        return np.array([
            proj[:, 0].min(),
            proj[:, 1].min(),
            proj[:, 0].max(),
            proj[:, 1].max(),
        ], dtype=np.float32)

    def __repr__(self) -> str:
        return (
            f"Registration(frame={self.frame_id}, valid={self.valid}, "
            f"inlier_ratio={self.inlier_ratio:.3f})"
        )


# ---------------------------------------------------------------------------
# 去重聚类输出（A 产出，C/D 消费）
# ---------------------------------------------------------------------------

@dataclass
class Cluster:
    """
    全局去重后的单个唯一螺丝实例。

    由 dedup.py (A) 产生，一个 Cluster 对应视频中的一颗真实螺丝。
    C 负责填充 class_probs 和 pred_class；D 用于计数和 mask 生成。
    """

    cluster_id: int
    """聚类编号，从 0 开始，视频内唯一。"""

    observations: List[Detection]
    """该螺丝在所有关键帧中的所有 Detection 观测（可能跨多帧）。"""

    best_crop: np.ndarray
    """
    从所有 observations 中选出的最佳 crop（清晰度最高、遮挡最少）。
    形状 (H, W, 3)，BGR，uint8。
    由 A 选取，供 C 的分类器使用。
    """

    class_probs: np.ndarray = field(
        default_factory=lambda: np.ones(5, dtype=np.float32) / 5.0
    )
    """
    长度为 5 的一维数组，表示该螺丝属于 Type_1~Type_5 的累积概率。
    初始值为均匀分布；C 的分类投票完成后应更新此字段。
    索引 0 → Type_1，…，索引 4 → Type_5。
    """

    pred_class: int = -1
    """
    最终预测类别，0-indexed（0=Type_1, …, 4=Type_5）。
    由 C 在完成 cluster 级投票后写入。
    -1 表示尚未分类。
    """

    ref_center: Optional[np.ndarray] = None
    """
    该螺丝在参考帧坐标系中的中心点 [x, y]（float32）。
    由 A 在聚类时写入，用于 mask 生成和可视化。
    """

    ref_bbox: Optional[np.ndarray] = None
    """
    该螺丝在参考帧坐标系中的代表性 bbox [x1, y1, x2, y2]（float32）。
    由 A 选取（通常取 best observation 对应的 projected bbox）。
    """

    @property
    def type_label(self) -> str:
        """返回人类可读的类别标签，如 'Type_1'。pred_class=-1 时返回 'Unknown'。"""
        if self.pred_class < 0:
            return "Unknown"
        return f"Type_{self.pred_class + 1}"

    @property
    def n_observations(self) -> int:
        """该螺丝被检测到的总帧次数。"""
        return len(self.observations)

    def __repr__(self) -> str:
        return (
            f"Cluster(id={self.cluster_id}, "
            f"class={self.type_label}, "
            f"obs={self.n_observations}, "
            f"best_prob={self.class_probs.max():.3f})"
        )


# ---------------------------------------------------------------------------
# 最终计数结果（D 产出，用于生成 result.npy）
# ---------------------------------------------------------------------------

@dataclass
class VideoResult:
    """
    单段视频的最终输出结果。

    由 pipeline.py (D) 汇总产生，最终序列化为 result.npy。
    """

    video_name: str
    """视频文件名，不含后缀（例如 'IMG_2374'）。"""

    counts: List[int]
    """
    长度为 5 的列表，按 [Type_1, Type_2, Type_3, Type_4, Type_5] 顺序记录各类螺丝总数。
    """

    clusters: List[Cluster] = field(default_factory=list)
    """所有去重后的螺丝 Cluster 列表，保留用于调试和可视化。"""

    processing_time: float = 0.0
    """处理该视频的总耗时（秒）。"""

    mask_frame_id: int = -1
    """mask 图像对应的帧编号（取视频中间帧）。"""

    def to_dict_entry(self):
        """
        返回用于写入 result.npy 字典的键值对。

        Returns
        -------
        (str, list) - (video_name, counts)
        """
        return self.video_name, self.counts

    def __repr__(self) -> str:
        return (
            f"VideoResult(video='{self.video_name}', "
            f"counts={self.counts}, "
            f"time={self.processing_time:.2f}s)"
        )
