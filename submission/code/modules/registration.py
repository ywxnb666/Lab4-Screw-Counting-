"""
modules/registration.py - 锚帧几何配准模块
Owner: A（几何配准与去重）

职责：
  - 从关键帧序列中选取锚帧（anchor frames）
  - 使用 AKAZE / ORB 特征点匹配计算 Homography
  - 将任意关键帧配准到参考坐标系（第一个锚帧或指定参考帧）
  - 监控配准质量，低质量帧标记为 invalid 并触发 tracker 降级路线

TODO (A)：
  [ ] 调整 AKAZE / ORB 参数至最优
  [ ] 实现锚帧自动选取策略（均匀 + 重叠率验证）
  [ ] 验证在 3 段开发视频上的配准成功率 ≥ 90%
  [ ] 调整 INLIER_RATIO_THRESHOLD（过低会丢帧，过高会漏掉难帧）
  [ ] 为连续配准失败段（≥3 帧）添加 tracker 降级通知接口

依赖：opencv-python, numpy
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from interfaces import Registration

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 超参数（A 负责调优）
# ---------------------------------------------------------------------------

# 特征检测器选择：'AKAZE' | 'ORB'
# AKAZE 精度更高但更慢；ORB 更快但低纹理场景下可能不稳定
FEATURE_TYPE: str = "AKAZE"

# AKAZE 参数
AKAZE_DESCRIPTOR_TYPE = cv2.AKAZE_DESCRIPTOR_MLDB
AKAZE_DESCRIPTOR_SIZE = 0        # 0 = 完整描述子
AKAZE_DESCRIPTOR_CHANNELS = 3
AKAZE_THRESHOLD = 0.001          # 响应阈值，越小检测到越多关键点（更慢）
AKAZE_NOCTAVES = 4
AKAZE_NOCTAVE_LAYERS = 4

# ORB 参数（AKAZE 不可用时的备选）
ORB_N_FEATURES = 5000
ORB_SCALE_FACTOR = 1.2
ORB_N_LEVELS = 8
ORB_EDGE_THRESHOLD = 15

# RANSAC 参数
RANSAC_REPROJ_THRESH = 4.0       # 重投影误差阈值（像素），越小越严格
RANSAC_MAX_ITERS = 2000          # RANSAC 最大迭代次数
RANSAC_CONFIDENCE = 0.995        # RANSAC 置信度

# 配准质量阈值
INLIER_RATIO_THRESHOLD: float = 0.25  # 内点率低于此值时将 valid 置为 False
MIN_MATCH_COUNT: int = 15             # 最少特征匹配数，不足则配准失败

# 低分辨率流配准参数（配准在低分辨率上计算，结果再映射到原始分辨率）
# 与 utils/video_io.py 中的 LOW_RES_LONG_EDGE 保持一致
LOW_RES_LONG_EDGE: int = 960

# 参考帧选取策略
# 'first'  : 使用序列第一个关键帧作为参考帧（最简单）
# 'middle' : 使用序列中间帧（减少边缘帧的变形累积）
ANCHOR_STRATEGY: str = "first"


# ---------------------------------------------------------------------------
# 特征检测器工厂
# ---------------------------------------------------------------------------

def _build_feature_detector(feature_type: str = FEATURE_TYPE):
    """
    构建特征检测器和描述子提取器。

    Parameters
    ----------
    feature_type : str
        'AKAZE' 或 'ORB'。

    Returns
    -------
    detector : cv2.Feature2D
        特征检测 + 描述子提取器（AKAZE / ORB）。
    matcher : cv2.DescriptorMatcher
        对应的特征匹配器。
    """
    if feature_type == "AKAZE":
        detector = cv2.AKAZE_create(
            descriptor_type=AKAZE_DESCRIPTOR_TYPE,
            descriptor_size=AKAZE_DESCRIPTOR_SIZE,
            descriptor_channels=AKAZE_DESCRIPTOR_CHANNELS,
            threshold=AKAZE_THRESHOLD,
            nOctaves=AKAZE_NOCTAVES,
            nOctaveLayers=AKAZE_NOCTAVE_LAYERS,
        )
        # AKAZE 使用二进制描述子，Hamming 距离
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        logger.debug("特征检测器: AKAZE")

    elif feature_type == "ORB":
        detector = cv2.ORB_create(
            nfeatures=ORB_N_FEATURES,
            scaleFactor=ORB_SCALE_FACTOR,
            nlevels=ORB_N_LEVELS,
            edgeThreshold=ORB_EDGE_THRESHOLD,
        )
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        logger.debug("特征检测器: ORB (nfeatures=%d)", ORB_N_FEATURES)

    else:
        raise ValueError(f"不支持的特征类型: {feature_type!r}，请选择 'AKAZE' 或 'ORB'。")

    return detector, matcher


# ---------------------------------------------------------------------------
# 特征提取与匹配工具函数
# ---------------------------------------------------------------------------

def _detect_and_compute(
    detector,
    image: np.ndarray,
) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
    """
    检测关键点并计算描述子。

    Parameters
    ----------
    detector : cv2.Feature2D
        特征检测器（AKAZE / ORB）。
    image : np.ndarray
        输入图像（BGR 或 灰度）。

    Returns
    -------
    keypoints : List[cv2.KeyPoint]
    descriptors : np.ndarray | None
        若未检测到关键点则返回 None。
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    kpts, descs = detector.detectAndCompute(gray, None)

    if kpts is None or len(kpts) == 0:
        return [], None

    return kpts, descs


def _ratio_test_matches(
    matcher,
    descs1: np.ndarray,
    descs2: np.ndarray,
    ratio: float = 0.75,
) -> List[cv2.DMatch]:
    """
    使用 Lowe's ratio test 过滤特征匹配。

    Parameters
    ----------
    matcher : cv2.DescriptorMatcher
    descs1 : np.ndarray
        查询图像的描述子。
    descs2 : np.ndarray
        训练图像（参考帧）的描述子。
    ratio : float
        Lowe's ratio test 阈值（默认 0.75）。
        越小越严格（保留匹配少但质量高），越大越宽松（匹配多但噪声多）。

    Returns
    -------
    List[cv2.DMatch] : 通过 ratio test 的优质匹配列表。
    """
    try:
        raw_matches = matcher.knnMatch(descs1, descs2, k=2)
    except cv2.error as e:
        logger.warning("特征匹配出错: %s", e)
        return []

    good_matches = []
    for pair in raw_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good_matches.append(m)

    return good_matches


def _compute_homography(
    kpts_src: List[cv2.KeyPoint],
    kpts_dst: List[cv2.KeyPoint],
    matches: List[cv2.DMatch],
) -> Tuple[Optional[np.ndarray], float]:
    """
    使用 RANSAC 从特征匹配中估计 Homography。

    Parameters
    ----------
    kpts_src : List[cv2.KeyPoint]
        源图像关键点（当前帧）。
    kpts_dst : List[cv2.KeyPoint]
        目标图像关键点（参考帧）。
    matches : List[cv2.DMatch]
        优质匹配列表。

    Returns
    -------
    H : np.ndarray | None
        3×3 Homography 矩阵；若计算失败则返回 None。
    inlier_ratio : float
        RANSAC 内点比例 [0, 1]。
    """
    if len(matches) < MIN_MATCH_COUNT:
        logger.debug("匹配数 %d < 最小要求 %d，配准失败。", len(matches), MIN_MATCH_COUNT)
        return None, 0.0

    src_pts = np.float32([kpts_src[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kpts_dst[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    try:
        H, mask = cv2.findHomography(
            src_pts,
            dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=RANSAC_REPROJ_THRESH,
            maxIters=RANSAC_MAX_ITERS,
            confidence=RANSAC_CONFIDENCE,
        )
    except cv2.error as e:
        logger.warning("findHomography 失败: %s", e)
        return None, 0.0

    if H is None or mask is None:
        return None, 0.0

    inlier_ratio = float(mask.ravel().sum()) / len(matches)
    return H, inlier_ratio


def _scale_homography(
    H: np.ndarray,
    scale_src: float,
    scale_dst: float,
) -> np.ndarray:
    """
    将低分辨率坐标系下的 Homography 转换到原始分辨率坐标系。

    配准在低分辨率图像上计算（速度快），检测在原始分辨率上进行。
    此函数将 H_lowres 转换为 H_fullres，使其可用于投影原始分辨率坐标。

    Parameters
    ----------
    H : np.ndarray
        低分辨率坐标系下的 3×3 Homography。
    scale_src : float
        源图像的缩放比例（low_res_width / full_res_width）。
    scale_dst : float
        目标图像（参考帧）的缩放比例。

    Returns
    -------
    np.ndarray : 原始分辨率坐标系下的 3×3 Homography。
    """
    # H_full = S_dst^{-1} @ H_low @ S_src
    # 其中 S = [[scale, 0, 0], [0, scale, 0], [0, 0, 1]]
    S_src = np.diag([scale_src, scale_src, 1.0])
    S_dst_inv = np.diag([1.0 / scale_dst, 1.0 / scale_dst, 1.0])
    return (S_dst_inv @ H @ S_src).astype(np.float64)


# ---------------------------------------------------------------------------
# 帧特征缓存（避免参考帧反复提取特征）
# ---------------------------------------------------------------------------

class _FrameFeatureCache:
    """
    缓存单帧的特征点和描述子，避免重复计算。

    用于参考帧（anchor frame）的特征缓存。
    """

    def __init__(self) -> None:
        self.frame_id: int = -1
        self.keypoints: Optional[List[cv2.KeyPoint]] = None
        self.descriptors: Optional[np.ndarray] = None
        self.image_shape: Optional[Tuple[int, int]] = None  # (H, W)

    def is_valid(self) -> bool:
        return (
            self.frame_id >= 0
            and self.keypoints is not None
            and self.descriptors is not None
        )

    def set(
        self,
        frame_id: int,
        keypoints: List[cv2.KeyPoint],
        descriptors: np.ndarray,
        image_shape: Tuple[int, int],
    ) -> None:
        self.frame_id = frame_id
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.image_shape = image_shape

    def clear(self) -> None:
        self.frame_id = -1
        self.keypoints = None
        self.descriptors = None
        self.image_shape = None


# ---------------------------------------------------------------------------
# 主配准类（A 负责实现）
# ---------------------------------------------------------------------------

class FrameRegistration:
    """
    锚帧几何配准器。

    对视频关键帧序列进行几何配准，将所有关键帧的坐标统一到参考坐标系。

    核心策略（替代逐帧累积以减少误差累积）：
    - 从关键帧序列中均匀选取 4-6 个锚帧
    - 每个关键帧**直接**与最近的锚帧配准（不做链式累积）
    - 锚帧之间相互配准，建立全局坐标系

    TODO (A)：
    1. 调整超参数（AKAZE_THRESHOLD / RANSAC_REPROJ_THRESH 等）
    2. 实现 select_anchors() 中的重叠率验证逻辑
    3. 实现 register_sequence() 中的全局坐标系建立逻辑
    4. 在 3 段开发视频上测试，确保 valid 帧占比 ≥ 90%
    5. 添加 tracker 降级通知接口（当连续 ≥3 帧配准失败时）

    用法
    ----
    >>> registrar = FrameRegistration()
    >>> ref_frame_lr = video_reader.read_frame(0, low_res=True)
    >>> registrar.set_reference(ref_frame_lr, frame_id=0, full_res_scale=0.5)
    >>> reg = registrar.register(query_frame_lr, frame_id=42, full_res_scale=0.5)
    >>> if reg.valid:
    ...     proj_bbox = reg.project_bbox(detection.bbox)
    """

    def __init__(
        self,
        feature_type: str = FEATURE_TYPE,
        inlier_ratio_threshold: float = INLIER_RATIO_THRESHOLD,
        min_match_count: int = MIN_MATCH_COUNT,
    ) -> None:
        """
        Parameters
        ----------
        feature_type : str
            使用的特征检测器类型：'AKAZE' 或 'ORB'。
        inlier_ratio_threshold : float
            配准有效的最低内点率阈值。
        min_match_count : int
            配准所需的最少匹配数。
        """
        self.feature_type = feature_type
        self.inlier_ratio_threshold = inlier_ratio_threshold
        self.min_match_count = min_match_count

        self._detector, self._matcher = _build_feature_detector(feature_type)
        self._ref_cache = _FrameFeatureCache()

        # 配准质量统计（用于 benchmark 和报告）
        self._stats: Dict[str, int] = {
            "total": 0,
            "valid": 0,
            "invalid_low_inlier": 0,
            "invalid_few_matches": 0,
            "invalid_exception": 0,
        }

        logger.info(
            "FrameRegistration 初始化完成 "
            "(feature=%s, inlier_thresh=%.2f, min_matches=%d)",
            feature_type, inlier_ratio_threshold, min_match_count,
        )

    # ------------------------------------------------------------------
    # 参考帧设置
    # ------------------------------------------------------------------

    def set_reference(
        self,
        ref_frame: np.ndarray,
        frame_id: int = 0,
        full_res_scale: float = 1.0,
    ) -> bool:
        """
        设置参考帧（anchor frame），提取并缓存其特征点。

        后续所有 register() 调用都将以此帧作为目标坐标系。

        Parameters
        ----------
        ref_frame : np.ndarray
            参考帧图像（低分辨率，H×W×3 BGR）。
        frame_id : int
            参考帧的原始帧编号（用于日志和调试）。
        full_res_scale : float
            低分辨率相对于原始分辨率的缩放比例
            （low_res_side / full_res_side）。
            例如：原始 3840×2160，低分辨率 960×540，则 scale=0.25。

        Returns
        -------
        bool : 特征提取是否成功（关键点数 >= MIN_MATCH_COUNT）。
        """
        kpts, descs = _detect_and_compute(self._detector, ref_frame)

        if descs is None or len(kpts) < self.min_match_count:
            logger.warning(
                "参考帧 (frame_id=%d) 特征点不足 (%d < %d)，配准可能全部失败。",
                frame_id, len(kpts), self.min_match_count,
            )
            self._ref_cache.clear()
            return False

        self._ref_cache.set(
            frame_id=frame_id,
            keypoints=kpts,
            descriptors=descs,
            image_shape=ref_frame.shape[:2],
        )
        self._full_res_scale_ref = full_res_scale

        logger.info(
            "参考帧已设置: frame_id=%d, 关键点数=%d, 低分辨率缩放=%.4f",
            frame_id, len(kpts), full_res_scale,
        )
        return True

    def has_reference(self) -> bool:
        """是否已设置参考帧。"""
        return self._ref_cache.is_valid()

    # ------------------------------------------------------------------
    # 单帧配准（主接口）
    # ------------------------------------------------------------------

    def register(
        self,
        query_frame: np.ndarray,
        frame_id: int,
        full_res_scale: float = 1.0,
    ) -> Registration:
        """
        将查询帧配准到参考帧坐标系，返回 Registration 对象。

        Parameters
        ----------
        query_frame : np.ndarray
            查询帧（低分辨率，H×W×3 BGR），与参考帧相同分辨率。
        frame_id : int
            查询帧的原始帧编号。
        full_res_scale : float
            查询帧低分辨率相对于原始分辨率的缩放比例。
            （通常与参考帧相同，除非两帧分辨率不同）

        Returns
        -------
        Registration :
            H_to_ref 为将原始分辨率坐标从当前帧投影到参考帧的变换矩阵。
            valid=False 时表示配准失败，外部应降级处理。
        """
        self._stats["total"] += 1

        # 未设置参考帧：返回恒等变换（仅在第一帧时可能发生）
        if not self.has_reference():
            logger.warning(
                "尚未设置参考帧，frame=%d 返回恒等变换。", frame_id
            )
            return Registration(
                frame_id=frame_id,
                H_to_ref=np.eye(3, dtype=np.float64),
                valid=True,
                inlier_ratio=1.0,
            )

        # 参考帧配准到自身：恒等变换
        if frame_id == self._ref_cache.frame_id:
            self._stats["valid"] += 1
            return Registration(
                frame_id=frame_id,
                H_to_ref=np.eye(3, dtype=np.float64),
                valid=True,
                inlier_ratio=1.0,
            )

        try:
            return self._register_impl(query_frame, frame_id, full_res_scale)
        except Exception as e:
            logger.error("配准异常 (frame=%d): %s", frame_id, e, exc_info=True)
            self._stats["invalid_exception"] += 1
            return Registration(
                frame_id=frame_id,
                H_to_ref=np.eye(3, dtype=np.float64),
                valid=False,
                inlier_ratio=0.0,
            )

    def _register_impl(
        self,
        query_frame: np.ndarray,
        frame_id: int,
        full_res_scale: float,
    ) -> Registration:
        """配准的内部实现（不含异常捕获）。"""
        # 1. 提取查询帧特征
        kpts_q, descs_q = _detect_and_compute(self._detector, query_frame)

        if descs_q is None or len(kpts_q) < self.min_match_count:
            logger.debug(
                "查询帧 frame=%d 特征点不足 (%d < %d)，配准失败。",
                frame_id, len(kpts_q) if kpts_q else 0, self.min_match_count,
            )
            self._stats["invalid_few_matches"] += 1
            return Registration(
                frame_id=frame_id,
                H_to_ref=np.eye(3, dtype=np.float64),
                valid=False,
                inlier_ratio=0.0,
            )

        # 2. 特征匹配（ratio test）
        good_matches = _ratio_test_matches(
            self._matcher,
            descs_q,
            self._ref_cache.descriptors,
            ratio=0.75,
        )

        if len(good_matches) < self.min_match_count:
            logger.debug(
                "frame=%d 优质匹配数 %d < %d，配准失败。",
                frame_id, len(good_matches), self.min_match_count,
            )
            self._stats["invalid_few_matches"] += 1
            return Registration(
                frame_id=frame_id,
                H_to_ref=np.eye(3, dtype=np.float64),
                valid=False,
                inlier_ratio=0.0,
            )

        # 3. RANSAC 估计 Homography（在低分辨率坐标系下）
        H_lr, inlier_ratio = _compute_homography(
            kpts_q,
            self._ref_cache.keypoints,
            good_matches,
        )

        if H_lr is None:
            self._stats["invalid_few_matches"] += 1
            return Registration(
                frame_id=frame_id,
                H_to_ref=np.eye(3, dtype=np.float64),
                valid=False,
                inlier_ratio=0.0,
            )

        # 4. 质量检查
        is_valid = inlier_ratio >= self.inlier_ratio_threshold
        if not is_valid:
            logger.debug(
                "frame=%d 内点率 %.3f < %.3f，配准标记为 invalid。",
                frame_id, inlier_ratio, self.inlier_ratio_threshold,
            )
            self._stats["invalid_low_inlier"] += 1

        # 5. 将低分辨率 Homography 转换到原始分辨率坐标系
        #    H_full: full_res_query -> full_res_reference
        H_full = _scale_homography(
            H_lr,
            scale_src=full_res_scale,
            scale_dst=self._full_res_scale_ref,
        )

        if is_valid:
            self._stats["valid"] += 1

        logger.debug(
            "frame=%d  matches=%d  inlier_ratio=%.3f  valid=%s",
            frame_id, len(good_matches), inlier_ratio, is_valid,
        )

        return Registration(
            frame_id=frame_id,
            H_to_ref=H_full,
            valid=is_valid,
            inlier_ratio=inlier_ratio,
        )

    # ------------------------------------------------------------------
    # 关键帧序列批量配准
    # ------------------------------------------------------------------

    def register_sequence(
        self,
        keyframe_images: List[np.ndarray],
        keyframe_ids: List[int],
        full_res_scales: Optional[List[float]] = None,
    ) -> List[Registration]:
        """
        对整段视频的关键帧序列进行批量配准。

        自动设置参考帧（按 ANCHOR_STRATEGY 策略选取），
        并对序列中的每一帧进行配准。

        Parameters
        ----------
        keyframe_images : List[np.ndarray]
            关键帧低分辨率图像列表（按时间顺序）。
        keyframe_ids : List[int]
            对应的原始帧编号列表。
        full_res_scales : List[float] | None
            每帧的低分辨率缩放比例列表；None 时假设所有帧比例相同（使用最后一次 set_reference 的比例）。

        Returns
        -------
        List[Registration] : 与输入列表等长，每个元素对应一帧的配准结果。

        TODO (A)：
        - 当前实现为简单的「全部对第一帧」策略
        - 可优化为：选取 4-6 个锚帧 → 每帧对最近锚帧配准 → 合并变换链到第一个锚帧
        - 需验证合并变换是否引入额外误差
        """
        if not keyframe_images:
            return []

        if full_res_scales is None:
            # 尝试从已设置的参考帧推断
            try:
                default_scale = self._full_res_scale_ref
            except AttributeError:
                default_scale = 1.0
            full_res_scales = [default_scale] * len(keyframe_images)

        # 选取参考帧
        ref_idx = self._select_reference_index(len(keyframe_images))
        ref_img = keyframe_images[ref_idx]
        ref_id = keyframe_ids[ref_idx]
        ref_scale = full_res_scales[ref_idx]

        success = self.set_reference(ref_img, frame_id=ref_id, full_res_scale=ref_scale)
        if not success:
            logger.warning(
                "参考帧 (idx=%d, frame_id=%d) 特征提取失败，"
                "所有帧将返回恒等变换（valid=False）。",
                ref_idx, ref_id,
            )
            return [
                Registration(
                    frame_id=fid,
                    H_to_ref=np.eye(3, dtype=np.float64),
                    valid=False,
                    inlier_ratio=0.0,
                )
                for fid in keyframe_ids
            ]

        # 逐帧配准
        registrations: List[Registration] = []
        consecutive_failures = 0

        for img, fid, scale in zip(keyframe_images, keyframe_ids, full_res_scales):
            reg = self.register(img, fid, full_res_scale=scale)
            registrations.append(reg)

            if not reg.valid:
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    logger.warning(
                        "连续 %d 帧配准失败（从 frame=%d 开始），"
                        "建议 A 检查该时段的视频质量或调整超参数。",
                        consecutive_failures,
                        fid - consecutive_failures + 1,
                    )
            else:
                consecutive_failures = 0

        logger.info(
            "批量配准完成: 共 %d 帧，有效 %d 帧 (%.1f%%)，无效 %d 帧。",
            len(registrations),
            sum(1 for r in registrations if r.valid),
            sum(1 for r in registrations if r.valid) / len(registrations) * 100,
            sum(1 for r in registrations if not r.valid),
        )

        return registrations

    def _select_reference_index(self, n_frames: int) -> int:
        """
        根据 ANCHOR_STRATEGY 选取参考帧在关键帧序列中的索引。

        Parameters
        ----------
        n_frames : int
            关键帧总数。

        Returns
        -------
        int : 参考帧在列表中的索引。
        """
        if ANCHOR_STRATEGY == "first":
            return 0
        elif ANCHOR_STRATEGY == "middle":
            return n_frames // 2
        else:
            logger.warning("未知的 ANCHOR_STRATEGY: %s，回退到 'first'。", ANCHOR_STRATEGY)
            return 0

    # ------------------------------------------------------------------
    # 统计与报告
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, int]:
        """
        返回配准质量统计信息。

        Returns
        -------
        Dict[str, int] :
            total                : 总配准次数
            valid                : 有效配准次数
            invalid_low_inlier   : 因内点率过低无效的次数
            invalid_few_matches  : 因匹配数不足无效的次数
            invalid_exception    : 因异常无效的次数
        """
        return dict(self._stats)

    def print_stats(self) -> None:
        """打印配准质量统计报告（用于 benchmark 和调试）。"""
        stats = self.get_stats()
        total = max(stats["total"], 1)
        print("=" * 50)
        print(f"[FrameRegistration] 配准质量统计")
        print(f"  特征类型       : {self.feature_type}")
        print(f"  总配准次数     : {stats['total']}")
        print(f"  有效           : {stats['valid']}  ({stats['valid'] / total * 100:.1f}%)")
        print(f"  无效（内点率低）: {stats['invalid_low_inlier']}")
        print(f"  无效（匹配少） : {stats['invalid_few_matches']}")
        print(f"  无效（异常）   : {stats['invalid_exception']}")
        print("=" * 50)

    def reset_stats(self) -> None:
        """重置统计计数器。"""
        for k in self._stats:
            self._stats[k] = 0

    # ------------------------------------------------------------------
    # 可视化辅助（调试用）
    # ------------------------------------------------------------------

    def visualize_matches(
        self,
        query_frame: np.ndarray,
        max_matches: int = 50,
    ) -> Optional[np.ndarray]:
        """
        可视化查询帧与参考帧之间的特征匹配（调试用）。

        Parameters
        ----------
        query_frame : np.ndarray
            查询帧低分辨率图像（BGR）。
        max_matches : int
            最多显示的匹配数量（防止图像过于拥挤）。

        Returns
        -------
        np.ndarray | None : 匹配可视化图像；若未设置参考帧则返回 None。
        """
        if not self.has_reference():
            logger.warning("尚未设置参考帧，无法可视化匹配。")
            return None

        kpts_q, descs_q = _detect_and_compute(self._detector, query_frame)
        if descs_q is None:
            return None

        good_matches = _ratio_test_matches(
            self._matcher,
            descs_q,
            self._ref_cache.descriptors,
        )
        # 只显示前 max_matches 个匹配
        display_matches = good_matches[:max_matches]

        # 由于参考帧图像未直接存储（节省内存），构造一张灰色占位图
        ref_h, ref_w = self._ref_cache.image_shape
        ref_placeholder = np.full((ref_h, ref_w, 3), 128, dtype=np.uint8)
        cv2.putText(
            ref_placeholder,
            "Reference Frame",
            (ref_w // 4, ref_h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )

        vis = cv2.drawMatches(
            query_frame,
            kpts_q,
            ref_placeholder,
            self._ref_cache.keypoints,
            display_matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        return vis
