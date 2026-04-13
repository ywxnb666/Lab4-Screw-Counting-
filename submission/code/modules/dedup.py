"""
modules/dedup.py - 全局去重聚类模块
Owner: A（几何配准与去重）

职责：
  - 接收所有关键帧的检测结果（List[Detection]）及对应的配准矩阵（List[Registration]）
  - 将各帧检测框投影到参考坐标系
  - 使用增量聚类（或 DBSCAN）将投影坐标相近的检测合并为同一颗螺丝
  - 为每个 Cluster 选取最佳 crop（清晰度最高的观测）
  - 输出 List[Cluster]，每个 Cluster 对应视频中的一颗唯一螺丝

TODO (A)：
  [ ] 实现基于投影坐标的增量聚类（推荐 DBSCAN 或简单 NMS 风格合并）
  [ ] 调整聚类半径 CLUSTER_DIST_THRESH（像素，参考坐标系）
  [ ] 处理配准 valid=False 的帧（tracker 辅助短时关联）
  [ ] 实现 best_crop 选取策略（拉普拉斯方差最大的观测）
  [ ] 填充 Cluster.ref_center 和 Cluster.ref_bbox
  [ ] 在 3 段开发视频上验证去重后计数与真实值的误差

依赖：numpy, scipy（DBSCAN 可用 sklearn），opencv-python
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from interfaces import Cluster, Detection, Registration

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 超参数（A 负责调优）
# ---------------------------------------------------------------------------

CLUSTER_DIST_THRESH: float = 40.0
"""
同一颗螺丝的两次检测在参考坐标系中的最大距离阈值（像素）。
超过此距离则视为不同螺丝。
建议根据螺丝在视频中的实际像素大小调整（通常为螺丝直径的 0.5~1 倍）。
"""

MIN_OBSERVATIONS: int = 1
"""
一个 Cluster 至少需要的观测次数，低于此值的 Cluster 将被过滤。
设为 1 时不过滤（保留所有检测）。
TODO (A)：适当提高此阈值可过滤误检（如设为 2）。
"""

INVALID_REG_FALLBACK: str = "skip"
"""
配准失败（Registration.valid=False）时的处理策略：
- "skip"   : 跳过该帧的检测（最保守，可能漏计）
- "identity": 使用单位矩阵（假设无运动，可能重复计数）
- "tracker": 利用 track_id 与已有 Cluster 关联（TODO: A 实现）
"""

BEST_CROP_METRIC: str = "sharpness"
"""
选取 Cluster 最佳 crop 的评估指标：
- "sharpness" : 拉普拉斯方差（越高越清晰，推荐）
- "area"      : bbox 面积（越大越好）
- "confidence": 检测置信度（越高越好）
"""


# ---------------------------------------------------------------------------
# 辅助工具
# ---------------------------------------------------------------------------

def _project_center(detection: Detection, registration: Registration) -> Optional[np.ndarray]:
    """
    将 Detection 的 bbox 中心投影到参考坐标系。

    Parameters
    ----------
    detection : Detection
        待投影的检测结果。
    registration : Registration
        对应帧的配准矩阵。

    Returns
    -------
    np.ndarray, shape (2,) 或 None（配准无效时）。
    """
    if not registration.valid:
        return None
    center = detection.center()
    return registration.project_point(center)


def _project_bbox(detection: Detection, registration: Registration) -> Optional[np.ndarray]:
    """
    将 Detection 的 bbox 投影到参考坐标系（轴对齐外接矩形）。

    Parameters
    ----------
    detection : Detection
    registration : Registration

    Returns
    -------
    np.ndarray, shape (4,) [x1, y1, x2, y2] 或 None。
    """
    if not registration.valid:
        return None
    return registration.project_bbox(detection.bbox)


def _compute_sharpness(crop: np.ndarray) -> float:
    """
    计算 crop 的清晰度（拉普拉斯方差）。

    值越大，图像越清晰。
    """
    if crop is None or crop.size == 0:
        return 0.0
    if crop.ndim == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _select_best_crop(
    observations: List[Detection],
    metric: str = BEST_CROP_METRIC,
) -> np.ndarray:
    """
    从一组观测中选取最佳 crop。

    Parameters
    ----------
    observations : List[Detection]
        同一 Cluster 的所有检测观测。
    metric : str
        评估指标（见 BEST_CROP_METRIC 说明）。

    Returns
    -------
    np.ndarray : 最佳 crop 图像（H×W×3 BGR）。
    """
    if not observations:
        return np.zeros((8, 8, 3), dtype=np.uint8)

    if metric == "sharpness":
        scores = [_compute_sharpness(d.crop) for d in observations]
    elif metric == "area":
        scores = [d.area() for d in observations]
    elif metric == "confidence":
        scores = [d.confidence for d in observations]
    else:
        logger.warning("未知 best_crop 指标 '%s'，默认使用 confidence。", metric)
        scores = [d.confidence for d in observations]

    best_idx = int(np.argmax(scores))
    return observations[best_idx].crop.copy()


# ---------------------------------------------------------------------------
# 增量聚类器（A 负责实现核心逻辑）
# ---------------------------------------------------------------------------

class _IncrementalClusterer:
    """
    基于投影坐标的增量聚类器。

    算法思路（贪心最近邻）：
    1. 维护一个已有 Cluster 中心列表
    2. 对每个新的投影检测，计算其到所有已有 Cluster 中心的距离
    3. 若最近距离 < CLUSTER_DIST_THRESH，则将该检测加入最近 Cluster
    4. 否则，创建新 Cluster

    优点：O(N×K)，K 为当前 Cluster 数，实时更新，内存占用小
    缺点：对检测顺序敏感，可能受配准噪声影响

    TODO (A)：
    - 考虑用 DBSCAN（sklearn.cluster.DBSCAN）替代，效果更稳定
    - 在聚类后增加「孤立点过滤」（观测次数 < MIN_OBSERVATIONS 的丢弃）
    - 考虑用 track_id 辅助关联（相同 track_id 的检测优先合并）
    """

    def __init__(self, dist_thresh: float = CLUSTER_DIST_THRESH) -> None:
        self.dist_thresh = dist_thresh
        self._clusters: List[_ClusterBuilder] = []

    def add(
        self,
        detection: Detection,
        ref_center: np.ndarray,
        ref_bbox: Optional[np.ndarray],
    ) -> int:
        """
        将一个已投影的检测结果加入聚类。

        Parameters
        ----------
        detection : Detection
            原始检测结果（含 crop）。
        ref_center : np.ndarray
            该检测在参考坐标系中的中心点 [x, y]。
        ref_bbox : np.ndarray | None
            该检测在参考坐标系中的 bbox（可为 None）。

        Returns
        -------
        int : 该检测被分配到的 Cluster ID。
        """
        if not self._clusters:
            cid = self._new_cluster(detection, ref_center, ref_bbox)
            return cid

        # 计算到所有已有 Cluster 中心的欧氏距离
        centers = np.array([c.center for c in self._clusters])  # (K, 2)
        dists = np.linalg.norm(centers - ref_center[np.newaxis, :], axis=1)  # (K,)
        min_dist_idx = int(np.argmin(dists))
        min_dist = float(dists[min_dist_idx])

        if min_dist <= self.dist_thresh:
            # 加入最近的已有 Cluster
            self._clusters[min_dist_idx].add_observation(detection, ref_center, ref_bbox)
            return self._clusters[min_dist_idx].cluster_id
        else:
            # 创建新 Cluster
            cid = self._new_cluster(detection, ref_center, ref_bbox)
            return cid

    def _new_cluster(
        self,
        detection: Detection,
        ref_center: np.ndarray,
        ref_bbox: Optional[np.ndarray],
    ) -> int:
        cid = len(self._clusters)
        cb = _ClusterBuilder(cluster_id=cid)
        cb.add_observation(detection, ref_center, ref_bbox)
        self._clusters.append(cb)
        return cid

    def build(self, min_observations: int = MIN_OBSERVATIONS) -> List[Cluster]:
        """
        完成聚类，返回 List[Cluster]。

        Parameters
        ----------
        min_observations : int
            观测次数低于此值的 Cluster 将被过滤（设 1 则不过滤）。

        Returns
        -------
        List[Cluster]
        """
        clusters = []
        for cb in self._clusters:
            if cb.n_observations < min_observations:
                logger.debug(
                    "Cluster #%d 观测次数 %d < %d，已过滤。",
                    cb.cluster_id, cb.n_observations, min_observations,
                )
                continue
            clusters.append(cb.to_cluster())
        return clusters


class _ClusterBuilder:
    """单个 Cluster 的构建器（内部类，不对外暴露）。"""

    def __init__(self, cluster_id: int) -> None:
        self.cluster_id = cluster_id
        self.observations: List[Detection] = []
        self._ref_centers: List[np.ndarray] = []
        self._ref_bboxes: List[Optional[np.ndarray]] = []

    @property
    def n_observations(self) -> int:
        return len(self.observations)

    @property
    def center(self) -> np.ndarray:
        """当前所有投影中心点的均值（作为 Cluster 代表中心）。"""
        if not self._ref_centers:
            return np.zeros(2, dtype=np.float32)
        return np.mean(self._ref_centers, axis=0).astype(np.float32)

    def add_observation(
        self,
        detection: Detection,
        ref_center: np.ndarray,
        ref_bbox: Optional[np.ndarray],
    ) -> None:
        self.observations.append(detection)
        self._ref_centers.append(ref_center)
        self._ref_bboxes.append(ref_bbox)

    def _best_ref_bbox(self) -> Optional[np.ndarray]:
        """选取面积最大的投影 bbox 作为代表。"""
        valid_bboxes = [b for b in self._ref_bboxes if b is not None]
        if not valid_bboxes:
            return None
        areas = [
            max(0, b[2] - b[0]) * max(0, b[3] - b[1])
            for b in valid_bboxes
        ]
        best_idx = int(np.argmax(areas))
        return valid_bboxes[best_idx].copy()

    def to_cluster(self) -> Cluster:
        """将 _ClusterBuilder 转换为正式的 Cluster 对象。"""
        best_crop = _select_best_crop(self.observations)
        return Cluster(
            cluster_id=self.cluster_id,
            observations=list(self.observations),
            best_crop=best_crop,
            class_probs=np.ones(5, dtype=np.float32) / 5.0,  # 均匀先验，由 C 更新
            pred_class=-1,                                     # 由 C 填充
            ref_center=self.center,
            ref_bbox=self._best_ref_bbox(),
        )


# ---------------------------------------------------------------------------
# DBSCAN 聚类器（A 可选实现，效果更稳定）
# ---------------------------------------------------------------------------

class _DBSCANClusterer:
    """
    基于 DBSCAN 的全局聚类器（离线聚类，需要收集完所有检测后执行）。

    相比增量聚类的优点：
    - 不受检测顺序影响
    - 自动识别离群点（eps=CLUSTER_DIST_THRESH 范围内无邻居的检测视为噪声）
    - 效果更稳定

    缺点：
    - 需要等所有帧处理完毕后才能聚类（不支持在线更新）

    TODO (A)：
    - 选择使用此类还是 _IncrementalClusterer
    - 调整 eps（即 CLUSTER_DIST_THRESH）和 min_samples
    """

    def __init__(
        self,
        eps: float = CLUSTER_DIST_THRESH,
        min_samples: int = MIN_OBSERVATIONS,
    ) -> None:
        self.eps = eps
        self.min_samples = min_samples

    def cluster(
        self,
        detections: List[Detection],
        ref_centers: List[np.ndarray],
        ref_bboxes: List[Optional[np.ndarray]],
    ) -> List[Cluster]:
        """
        对所有已投影检测执行 DBSCAN 聚类。

        Parameters
        ----------
        detections : List[Detection]
            所有有效的检测结果（已过滤配准无效帧）。
        ref_centers : List[np.ndarray]
            对应的参考坐标系中心点列表。
        ref_bboxes : List[Optional[np.ndarray]]
            对应的参考坐标系 bbox 列表。

        Returns
        -------
        List[Cluster]
        """
        if not detections:
            return []

        try:
            from sklearn.cluster import DBSCAN
        except ImportError:
            logger.warning(
                "scikit-learn 未安装，DBSCAN 不可用。"
                "回退到增量聚类。运行: pip install scikit-learn"
            )
            return self._fallback_incremental(detections, ref_centers, ref_bboxes)

        centers_arr = np.array(ref_centers, dtype=np.float32)  # (N, 2)
        labels = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit_predict(centers_arr)

        # 将 labels 相同的检测分组
        cluster_dict: Dict[int, _ClusterBuilder] = {}
        for i, label in enumerate(labels):
            if label == -1:
                # DBSCAN 噪声点：单独成一个 Cluster（或直接丢弃）
                # 当前策略：若 MIN_OBSERVATIONS=1，则保留；否则丢弃
                if self.min_samples <= 1:
                    noise_id = -(i + 1000)  # 临时负 ID，后续重新编号
                    cb = cluster_dict.setdefault(noise_id, _ClusterBuilder(noise_id))
                    cb.add_observation(detections[i], ref_centers[i], ref_bboxes[i])
            else:
                cb = cluster_dict.setdefault(label, _ClusterBuilder(label))
                cb.add_observation(detections[i], ref_centers[i], ref_bboxes[i])

        # 重新编号 Cluster
        clusters = []
        for new_id, (_, cb) in enumerate(sorted(cluster_dict.items())):
            cb.cluster_id = new_id
            clusters.append(cb.to_cluster())

        logger.debug("DBSCAN 聚类完成: %d 个检测 → %d 个螺丝", len(detections), len(clusters))
        return clusters

    def _fallback_incremental(
        self,
        detections: List[Detection],
        ref_centers: List[np.ndarray],
        ref_bboxes: List[Optional[np.ndarray]],
    ) -> List[Cluster]:
        """DBSCAN 不可用时，回退到增量聚类。"""
        clusterer = _IncrementalClusterer(dist_thresh=self.eps)
        for det, center, bbox in zip(detections, ref_centers, ref_bboxes):
            clusterer.add(det, center, bbox)
        return clusterer.build(min_observations=self.min_samples)


# ---------------------------------------------------------------------------
# 门面类（统一入口，D 在 pipeline.py 中调用此类）
# ---------------------------------------------------------------------------

class GlobalDedup:
    """
    全局去重聚类门面类。

    接收所有关键帧的检测结果和配准结果，
    将投影到参考坐标系后距离相近的检测合并为唯一螺丝实例（Cluster），
    输出 List[Cluster]。

    D 在 pipeline.py 中只调用此类，不直接使用内部聚类器。

    用法
    ----
    >>> dedup = GlobalDedup()
    >>> clusters = dedup.run(all_detections, all_registrations)
    >>> print(f"共检测到 {len(clusters)} 颗螺丝")
    """

    def __init__(
        self,
        dist_thresh: float = CLUSTER_DIST_THRESH,
        min_observations: int = MIN_OBSERVATIONS,
        use_dbscan: bool = True,
        invalid_reg_fallback: str = INVALID_REG_FALLBACK,
    ) -> None:
        """
        Parameters
        ----------
        dist_thresh : float
            聚类距离阈值（参考坐标系，像素）。
        min_observations : int
            Cluster 最少观测次数（低于此值的 Cluster 被过滤）。
        use_dbscan : bool
            True: 使用 DBSCAN（离线，更稳定）；
            False: 使用增量聚类（在线，更快）。
        invalid_reg_fallback : str
            配准失败时的处理策略（见 INVALID_REG_FALLBACK 说明）。
        """
        self.dist_thresh = dist_thresh
        self.min_observations = min_observations
        self.use_dbscan = use_dbscan
        self.invalid_reg_fallback = invalid_reg_fallback

        logger.info(
            "GlobalDedup 初始化: dist_thresh=%.1f, min_obs=%d, "
            "use_dbscan=%s, invalid_reg='%s'",
            dist_thresh, min_observations, use_dbscan, invalid_reg_fallback,
        )

    def run(
        self,
        all_detections: List[List[Detection]],
        all_registrations: List[Registration],
    ) -> List[Cluster]:
        """
        执行全局去重聚类。

        Parameters
        ----------
        all_detections : List[List[Detection]]
            每个关键帧的检测结果列表。
            all_detections[i] 对应 all_registrations[i] 所在帧的检测。
            长度必须与 all_registrations 相同。
        all_registrations : List[Registration]
            每个关键帧的配准结果列表（与 all_detections 一一对应）。

        Returns
        -------
        List[Cluster]
            去重后的唯一螺丝实例列表。每个 Cluster 的 pred_class 初始为 -1，
            由后续的 ScrewClassifier (C) 填充。

        Raises
        ------
        ValueError
            若 all_detections 和 all_registrations 长度不一致。
        """
        if len(all_detections) != len(all_registrations):
            raise ValueError(
                f"all_detections 长度 ({len(all_detections)}) "
                f"与 all_registrations 长度 ({len(all_registrations)}) 不一致。"
            )

        # ---- Step 1：投影所有检测到参考坐标系 ----
        valid_detections: List[Detection] = []
        valid_centers: List[np.ndarray] = []
        valid_bboxes: List[Optional[np.ndarray]] = []

        n_skipped = 0
        for frame_dets, reg in zip(all_detections, all_registrations):
            if not reg.valid:
                n_skipped += len(frame_dets)
                if self.invalid_reg_fallback == "skip":
                    logger.debug(
                        "配准失败 (frame=%d, inlier_ratio=%.3f)，跳过该帧 %d 个检测。",
                        reg.frame_id, reg.inlier_ratio, len(frame_dets),
                    )
                    continue
                elif self.invalid_reg_fallback == "identity":
                    # 使用单位矩阵（当前帧坐标 ≈ 参考坐标）
                    identity_reg = Registration(
                        frame_id=reg.frame_id,
                        H_to_ref=np.eye(3, dtype=np.float64),
                        valid=True,
                        inlier_ratio=0.0,
                    )
                    reg = identity_reg
                    logger.debug(
                        "配准失败 (frame=%d)，使用单位矩阵（identity fallback）。",
                        reg.frame_id,
                    )
                else:
                    # TODO (A)：实现 tracker 辅助关联
                    logger.debug(
                        "配准失败 (frame=%d)，tracker 关联未实现，跳过。",
                        reg.frame_id,
                    )
                    continue

            for det in frame_dets:
                center = _project_center(det, reg)
                bbox = _project_bbox(det, reg)
                if center is None:
                    n_skipped += 1
                    continue
                valid_detections.append(det)
                valid_centers.append(center)
                valid_bboxes.append(bbox)

        logger.info(
            "投影完成: %d 个检测参与聚类，%d 个因配准失败被跳过。",
            len(valid_detections), n_skipped,
        )

        if not valid_detections:
            logger.warning("没有有效的检测结果，返回空 Cluster 列表。")
            return []

        # ---- Step 2：聚类 ----
        if self.use_dbscan:
            clusterer = _DBSCANClusterer(
                eps=self.dist_thresh,
                min_samples=self.min_observations,
            )
            clusters = clusterer.cluster(valid_detections, valid_centers, valid_bboxes)
        else:
            inc_clusterer = _IncrementalClusterer(dist_thresh=self.dist_thresh)
            for det, center, bbox in zip(valid_detections, valid_centers, valid_bboxes):
                inc_clusterer.add(det, center, bbox)
            clusters = inc_clusterer.build(min_observations=self.min_observations)

        logger.info(
            "去重聚类完成: %d 个投影检测 → %d 个唯一螺丝 Cluster。",
            len(valid_detections), len(clusters),
        )

        return clusters

    # ------------------------------------------------------------------
    # 调试工具
    # ------------------------------------------------------------------

    @staticmethod
    def summarize(clusters: List[Cluster]) -> str:
        """
        返回 Cluster 列表的摘要字符串（用于日志输出）。

        Parameters
        ----------
        clusters : List[Cluster]

        Returns
        -------
        str
        """
        if not clusters:
            return "Cluster 列表为空"

        lines = [f"共 {len(clusters)} 个 Cluster:"]
        for c in clusters:
            lines.append(
                f"  #{c.cluster_id:3d}  obs={c.n_observations:3d}  "
                f"class={c.type_label}  "
                f"center=({c.ref_center[0]:.1f}, {c.ref_center[1]:.1f})"
                if c.ref_center is not None
                else f"  #{c.cluster_id:3d}  obs={c.n_observations:3d}  class={c.type_label}"
            )

        # 计数汇总
        from collections import Counter
        class_counts = Counter(c.pred_class for c in clusters)
        count_str = ", ".join(
            f"Type_{k + 1}={v}" for k, v in sorted(class_counts.items()) if k >= 0
        )
        unclassified = class_counts.get(-1, 0)
        lines.append(f"\n计数: [{count_str}]  未分类: {unclassified}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 便捷函数
# ---------------------------------------------------------------------------

def run_dedup(
    all_detections: List[List[Detection]],
    all_registrations: List[Registration],
    dist_thresh: float = CLUSTER_DIST_THRESH,
    min_observations: int = MIN_OBSERVATIONS,
    use_dbscan: bool = True,
) -> List[Cluster]:
    """
    全局去重聚类便捷函数（无需手动实例化 GlobalDedup）。

    Parameters
    ----------
    all_detections : List[List[Detection]]
        每帧的检测结果列表。
    all_registrations : List[Registration]
        每帧的配准结果列表。
    dist_thresh : float
        聚类距离阈值（像素）。
    min_observations : int
        Cluster 最少观测次数。
    use_dbscan : bool
        是否使用 DBSCAN。

    Returns
    -------
    List[Cluster]
    """
    dedup = GlobalDedup(
        dist_thresh=dist_thresh,
        min_observations=min_observations,
        use_dbscan=use_dbscan,
    )
    return dedup.run(all_detections, all_registrations)
