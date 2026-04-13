"""
pipeline.py - 视频处理主流程编排
Owner: D（工程封装）

职责：
  - 协调 A/B/C 各模块，完成单段视频的端到端处理
  - 关键帧抽取（motion-based keyframe selection）
  - 调用 detector（B）进行逐帧检测
  - 调用 registration（A）进行关键帧几何配准
  - 调用 dedup（A）进行全局去重聚类
  - 调用 classifier（C）进行 Cluster 级分类投票
  - 生成 mask 叠加图（中间帧）
  - 返回 VideoResult（计数 + mask + 耗时）

依赖：
  - modules/detector.py      [B]
  - modules/registration.py  [A]
  - modules/dedup.py         [A]
  - modules/classifier.py    [C]
  - utils/video_io.py        [D]
  - utils/visualizer.py      [D]
  - interfaces.py            [D]
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from interfaces import Cluster, Detection, Registration, VideoResult
from modules.detector import Detector
from modules.registration import FrameRegistration
from modules.dedup import GlobalDedup
from modules.classifier import ScrewClassifier
from utils.video_io import VideoReader, list_videos, get_video_name
from utils.visualizer import Visualizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 关键帧提取超参数（D 负责调优）
# ---------------------------------------------------------------------------

# 关键帧选取策略：基于相邻帧特征点位移
KF_MOTION_THRESH_RATIO: float = 0.15
"""
触发关键帧的累积位移阈值（相对于帧宽度的比例）。
例如 0.15 表示当特征点累积位移超过帧宽 15% 时，选为关键帧。
"""

KF_MAX_INTERVAL: int = 15
"""
最多每 KF_MAX_INTERVAL 帧强制取一个关键帧（防止漏帧）。
"""

KF_MIN_INTERVAL: int = 5
"""
最少每 KF_MIN_INTERVAL 帧才允许取一个关键帧（防止过密采样）。
"""

KF_MIN_COUNT: int = 8
"""
一段视频至少提取的关键帧数量（视频过短时的保底）。
"""

KF_MAX_COUNT: int = 60
"""
一段视频最多提取的关键帧数量（防止过多导致速度超限）。
"""

# ORB 特征点数量（用于关键帧选取中的位移估计，不是检测用的 AKAZE）
KF_ORB_N_FEATURES: int = 200


# ---------------------------------------------------------------------------
# 关键帧提取
# ---------------------------------------------------------------------------

def _extract_keyframes_motion(
    reader: VideoReader,
    motion_thresh_ratio: float = KF_MOTION_THRESH_RATIO,
    max_interval: int = KF_MAX_INTERVAL,
    min_interval: int = KF_MIN_INTERVAL,
    min_count: int = KF_MIN_COUNT,
    max_count: int = KF_MAX_COUNT,
) -> List[int]:
    """
    基于相邻帧间特征点位移的关键帧选取算法。

    算法逻辑：
    1. 逐帧读取低分辨率流，用 ORB 检测特征点
    2. 计算相邻帧特征点的光流位移均值
    3. 当累积位移超过阈值，或距上次关键帧超过 max_interval 时，选为关键帧
    4. 距上次关键帧不足 min_interval 时，禁止选取（防止过密）

    Parameters
    ----------
    reader : VideoReader
        已打开的视频读取器。
    motion_thresh_ratio : float
        触发关键帧的累积位移比例（相对帧宽）。
    max_interval : int
        强制关键帧的最大间隔（帧数）。
    min_interval : int
        关键帧的最小间隔（帧数）。
    min_count : int
        最少关键帧数量。
    max_count : int
        最多关键帧数量。

    Returns
    -------
    List[int] : 选取的关键帧编号列表（升序）。
    """
    meta = reader.meta
    total_frames = meta.frame_count
    frame_w = meta.width

    if total_frames <= 0:
        logger.warning("视频帧数为 0，无法提取关键帧。")
        return []

    motion_thresh_px = frame_w * motion_thresh_ratio

    # ORB 检测器（轻量，专用于位移估计）
    orb = cv2.ORB_create(nfeatures=KF_ORB_N_FEATURES)
    bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    keyframe_ids: List[int] = []
    last_kf_id: int = -min_interval   # 初始化为负，让第一帧可被选取
    cumulative_displacement: float = 0.0

    prev_gray: Optional[np.ndarray] = None
    prev_kpts = None
    prev_descs = None

    logger.debug(
        "关键帧提取: 总帧数=%d, 位移阈值=%.1fpx (%.0f%% of %dpx宽), "
        "min_interval=%d, max_interval=%d",
        total_frames, motion_thresh_px,
        motion_thresh_ratio * 100, frame_w,
        min_interval, max_interval,
    )

    for frame_id, _frame_hr, frame_lr in reader.iter_frames(step=1, yield_low_res=True):
        if frame_lr is None:
            continue

        gray = cv2.cvtColor(frame_lr, cv2.COLOR_BGR2GRAY)
        kpts, descs = orb.detectAndCompute(gray, None)

        # 计算与上一帧的位移
        if (prev_gray is not None
                and prev_descs is not None
                and descs is not None
                and len(kpts) > 5
                and len(prev_kpts) > 5):
            try:
                matches = bf_matcher.match(descs, prev_descs)
                if matches:
                    # 取前一半质量最好的匹配
                    matches = sorted(matches, key=lambda m: m.distance)
                    good_matches = matches[: max(1, len(matches) // 2)]

                    displacements = []
                    for m in good_matches:
                        pt1 = np.array(kpts[m.queryIdx].pt)
                        pt2 = np.array(prev_kpts[m.trainIdx].pt)
                        displacements.append(np.linalg.norm(pt1 - pt2))

                    # 去除离群位移（> 中位数 3 倍的视为噪声）
                    disps = np.array(displacements)
                    median_disp = np.median(disps)
                    valid_disps = disps[disps < max(1.0, median_disp * 3.0)]
                    mean_disp = float(np.mean(valid_disps)) if len(valid_disps) > 0 else 0.0

                    # 将低分辨率位移还原到原始分辨率
                    mean_disp_full = mean_disp / meta.low_res_scale
                    cumulative_displacement += mean_disp_full
            except cv2.error:
                pass

        prev_gray = gray
        prev_kpts = kpts
        prev_descs = descs

        frames_since_last_kf = frame_id - last_kf_id

        # 判断是否选为关键帧
        should_select = False

        if frame_id == 0:
            # 第一帧必选
            should_select = True
        elif frames_since_last_kf < min_interval:
            # 距上次关键帧太近，跳过
            should_select = False
        elif frames_since_last_kf >= max_interval:
            # 强制选取（防止长时间无关键帧）
            should_select = True
        elif cumulative_displacement >= motion_thresh_px:
            # 累积位移超过阈值
            should_select = True

        if should_select:
            keyframe_ids.append(frame_id)
            last_kf_id = frame_id
            cumulative_displacement = 0.0
            logger.debug("关键帧: frame_id=%d", frame_id)

            if len(keyframe_ids) >= max_count:
                logger.info("关键帧数量已达上限 %d，停止采样。", max_count)
                break

    # 确保最后一帧也被包含
    last_frame_id = total_frames - 1
    if keyframe_ids and keyframe_ids[-1] != last_frame_id:
        if last_frame_id - keyframe_ids[-1] >= min_interval:
            keyframe_ids.append(last_frame_id)

    # 保底：若关键帧数量不足，均匀补充
    if len(keyframe_ids) < min_count:
        extra_needed = min_count - len(keyframe_ids)
        existing_set = set(keyframe_ids)
        step = max(1, total_frames // (extra_needed + len(keyframe_ids) + 1))
        for fid in range(0, total_frames, step):
            if fid not in existing_set:
                keyframe_ids.append(fid)
                existing_set.add(fid)
                if len(keyframe_ids) >= min_count:
                    break
        keyframe_ids = sorted(set(keyframe_ids))

    # 限制最大数量（保底补充后可能超限）
    if len(keyframe_ids) > max_count:
        # 均匀抽样
        indices = np.linspace(0, len(keyframe_ids) - 1, max_count, dtype=int)
        keyframe_ids = [keyframe_ids[i] for i in indices]

    logger.info(
        "关键帧提取完成: %d 帧 / %d 总帧 (%.1f%%)，帧编号范围 [%d, %d]",
        len(keyframe_ids), total_frames,
        len(keyframe_ids) / max(total_frames, 1) * 100,
        keyframe_ids[0] if keyframe_ids else -1,
        keyframe_ids[-1] if keyframe_ids else -1,
    )

    return keyframe_ids


def extract_keyframes_uniform(
    reader: VideoReader,
    target_count: int = 30,
) -> List[int]:
    """
    均匀采样关键帧（当 motion-based 提取失败或作为对比基线时使用）。

    Parameters
    ----------
    reader : VideoReader
        已打开的视频读取器。
    target_count : int
        目标关键帧数量。

    Returns
    -------
    List[int] : 均匀分布的关键帧编号列表。
    """
    total = reader.meta.frame_count
    if total <= 0:
        return []
    count = min(target_count, total)
    indices = np.linspace(0, total - 1, count, dtype=int)
    return sorted(set(indices.tolist()))


# ---------------------------------------------------------------------------
# 单视频处理主函数
# ---------------------------------------------------------------------------

def process_video(
    video_path: Path,
    detector: Detector,
    registrar: FrameRegistration,
    deduper: GlobalDedup,
    classifier: ScrewClassifier,
    visualizer: Visualizer,
    keyframe_strategy: str = "motion",
) -> VideoResult:
    """
    处理单段视频，完成从帧读取到计数输出的完整流程。

    Parameters
    ----------
    video_path : Path
        视频文件路径。
    detector : Detector
        螺丝检测器实例（B 的模块）。
    registrar : FrameRegistration
        帧配准器实例（A 的模块）。
    deduper : GlobalDedup
        全局去重聚类器实例（A 的模块）。
    classifier : ScrewClassifier
        螺丝分类器实例（C 的模块）。
    visualizer : Visualizer
        可视化器实例（D 的工具）。
    keyframe_strategy : str
        关键帧提取策略：'motion'（基于位移）或 'uniform'（均匀采样）。

    Returns
    -------
    VideoResult : 包含计数结果、掩膜图像（通过 mask_frame_id 确定）和耗时。

    Raises
    ------
    FileNotFoundError
        若视频文件不存在。
    RuntimeError
        若视频无法打开。
    """
    t_start = time.perf_counter()
    video_name = get_video_name(video_path)
    logger.info("=" * 60)
    logger.info("开始处理视频: %s", video_name)

    # ================================================================
    # Step 0: 打开视频，读取元数据
    # ================================================================
    with VideoReader(video_path) as reader:
        meta = reader.meta
        logger.info("视频元数据: %s", meta)

        mid_frame_id = meta.mid_frame_id
        full_res_scale = meta.low_res_scale   # low_res / full_res

        # ============================================================
        # Step 1: 关键帧提取
        # ============================================================
        t1 = time.perf_counter()

        if keyframe_strategy == "motion":
            try:
                keyframe_ids = _extract_keyframes_motion(reader)
            except Exception as e:
                logger.warning("Motion-based 关键帧提取失败 (%s)，回退到均匀采样。", e)
                keyframe_ids = extract_keyframes_uniform(reader, target_count=30)
        else:
            keyframe_ids = extract_keyframes_uniform(reader, target_count=30)

        if not keyframe_ids:
            logger.error("未能提取任何关键帧，视频 '%s' 计数为全 0。", video_name)
            return VideoResult(
                video_name=video_name,
                counts=[0] * 5,
                clusters=[],
                processing_time=time.perf_counter() - t_start,
                mask_frame_id=mid_frame_id,
            )

        t2 = time.perf_counter()
        logger.info(
            "[Step 1] 关键帧提取: %d 帧，耗时 %.3fs",
            len(keyframe_ids), t2 - t1,
        )

        # ============================================================
        # Step 2: 读取关键帧图像（双分辨率）
        # ============================================================
        t1 = time.perf_counter()

        # 重置 reader 位置并读取关键帧
        kf_images_hr: List[np.ndarray] = []   # 原始分辨率（用于检测）
        kf_images_lr: List[np.ndarray] = []   # 低分辨率（用于配准）
        valid_kf_ids: List[int] = []           # 成功读取的关键帧编号

        for fid, frame_hr, frame_lr in reader.iter_frames_at(
            keyframe_ids, yield_low_res=True
        ):
            if frame_hr is None or frame_lr is None:
                logger.warning("帧 %d 读取失败，跳过。", fid)
                continue
            kf_images_hr.append(frame_hr)
            kf_images_lr.append(frame_lr)
            valid_kf_ids.append(fid)

        if not valid_kf_ids:
            logger.error("所有关键帧读取失败，视频 '%s' 计数为全 0。", video_name)
            return VideoResult(
                video_name=video_name,
                counts=[0] * 5,
                clusters=[],
                processing_time=time.perf_counter() - t_start,
                mask_frame_id=mid_frame_id,
            )

        t2 = time.perf_counter()
        logger.info(
            "[Step 2] 关键帧读取: %d 帧成功 / %d 帧请求，耗时 %.3fs",
            len(valid_kf_ids), len(keyframe_ids), t2 - t1,
        )

        # ============================================================
        # Step 3: 批量检测（B 的模块）
        # ============================================================
        t1 = time.perf_counter()

        all_detections: List[List[Detection]] = detector.detect_batch(
            kf_images_hr, valid_kf_ids
        )

        total_detections = sum(len(d) for d in all_detections)
        t2 = time.perf_counter()
        logger.info(
            "[Step 3] 螺丝检测: %d 帧共检测到 %d 个螺丝，耗时 %.3fs",
            len(valid_kf_ids), total_detections, t2 - t1,
        )

        # ============================================================
        # Step 4: 批量配准（A 的模块）
        # ============================================================
        t1 = time.perf_counter()

        all_registrations: List[Registration] = registrar.register_sequence(
            keyframe_images=kf_images_lr,
            keyframe_ids=valid_kf_ids,
            full_res_scales=[full_res_scale] * len(valid_kf_ids),
        )

        valid_reg_count = sum(1 for r in all_registrations if r.valid)
        t2 = time.perf_counter()
        logger.info(
            "[Step 4] 几何配准: %d / %d 帧有效 (%.1f%%)，耗时 %.3fs",
            valid_reg_count, len(all_registrations),
            valid_reg_count / max(len(all_registrations), 1) * 100,
            t2 - t1,
        )

        # ============================================================
        # Step 5: 全局去重聚类（A 的模块）
        # ============================================================
        t1 = time.perf_counter()

        clusters: List[Cluster] = deduper.run(all_detections, all_registrations)

        t2 = time.perf_counter()
        logger.info(
            "[Step 5] 去重聚类: %d 个检测 → %d 个唯一螺丝，耗时 %.3fs",
            total_detections, len(clusters), t2 - t1,
        )

        # ============================================================
        # Step 6: Cluster 级分类投票（C 的模块）
        # ============================================================
        t1 = time.perf_counter()

        classified_clusters, counts = classifier.classify_and_count(clusters)

        t2 = time.perf_counter()
        logger.info(
            "[Step 6] 螺丝分类完成，耗时 %.3fs\n"
            "  计数结果: Type_1=%d, Type_2=%d, Type_3=%d, Type_4=%d, Type_5=%d\n"
            "  总计: %d 颗螺丝",
            t2 - t1,
            counts[0], counts[1], counts[2], counts[3], counts[4],
            sum(counts),
        )

        # ============================================================
        # Step 7: 读取中间帧（用于 mask 生成）
        # ============================================================
        t1 = time.perf_counter()

        mid_frame = reader.read_frame(mid_frame_id, low_res=False)
        if mid_frame is None:
            logger.warning(
                "中间帧 (frame_id=%d) 读取失败，使用第一个关键帧代替。",
                mid_frame_id,
            )
            mid_frame = kf_images_hr[0] if kf_images_hr else None
            mid_frame_id = valid_kf_ids[0] if valid_kf_ids else 0

        t2 = time.perf_counter()
        logger.info("[Step 7] 中间帧读取耗时 %.3fs", t2 - t1)

    # 退出 with 块，VideoReader 已关闭

    # ================================================================
    # Step 8: 生成 mask 叠加图（D 的工具）
    # ================================================================
    t1 = time.perf_counter()

    mask_image: Optional[np.ndarray] = None
    if mid_frame is not None:
        # 若 Cluster 没有 ref_bbox（配准全失败时），尝试用最近关键帧的检测框代替
        _ensure_clusters_have_bbox(classified_clusters, all_detections, valid_kf_ids, mid_frame_id)

        mask_image = visualizer.draw_clusters(
            mid_frame,
            classified_clusters,
            draw_bbox=True,
            draw_mask=True,
            draw_id=False,
        )

        # 在 mask 图左上角写入计数摘要
        count_lines = [
            f"Video: {video_name}",
            f"Total: {sum(counts)} screws",
        ] + [f"Type_{i + 1}: {counts[i]}" for i in range(5)]
        visualizer.add_text_banner(mask_image, count_lines, position="top-left")

    t2 = time.perf_counter()
    logger.info("[Step 8] Mask 生成耗时 %.3fs", t2 - t1)

    # ================================================================
    # 汇总
    # ================================================================
    total_time = time.perf_counter() - t_start

    result = VideoResult(
        video_name=video_name,
        counts=counts,
        clusters=classified_clusters,
        processing_time=total_time,
        mask_frame_id=mid_frame_id,
    )

    logger.info(
        "视频 '%s' 处理完成，总耗时 %.2fs\n"
        "  计数: %s",
        video_name, total_time, counts,
    )

    # 将 mask_image 附加到 result 上（pipeline 外部保存）
    result._mask_image = mask_image   # type: ignore[attr-defined]

    return result


# ---------------------------------------------------------------------------
# Cluster bbox 补全工具（当配准全部失败时的降级处理）
# ---------------------------------------------------------------------------

def _ensure_clusters_have_bbox(
    clusters: List[Cluster],
    all_detections: List[List[Detection]],
    kf_ids: List[int],
    target_frame_id: int,
) -> None:
    """
    为没有 ref_bbox 的 Cluster 补全一个近似 bbox。

    当配准全部失败时，Cluster.ref_bbox 可能为 None（无法投影）。
    此函数尝试用 Cluster 中距离 target_frame_id 最近的观测 bbox 作为近似。

    参数和返回值省略（原地修改 clusters）。
    """
    for cluster in clusters:
        if cluster.ref_bbox is not None:
            continue

        # 找距离 target_frame_id 最近的观测
        if not cluster.observations:
            continue

        best_obs = min(
            cluster.observations,
            key=lambda d: abs(d.frame_id - target_frame_id),
        )
        cluster.ref_bbox = best_obs.bbox.copy()
        cluster.ref_center = best_obs.center()

    logger.debug(
        "_ensure_clusters_have_bbox: 检查了 %d 个 Cluster 的 bbox 完整性。",
        len(clusters),
    )


# ---------------------------------------------------------------------------
# 多视频批处理（run.py 调用此函数）
# ---------------------------------------------------------------------------

class VideoPipeline:
    """
    多视频批处理流水线。

    封装所有模块的初始化（避免重复加载模型），
    提供 process_folder() 方法处理整个视频文件夹。

    用法
    ----
    >>> pipeline = VideoPipeline()
    >>> results, masks = pipeline.process_folder("/path/to/videos")
    >>> pipeline.print_summary(results)
    """

    def __init__(
        self,
        detector_weights: Optional[Path] = None,
        classifier_weights: Optional[Path] = None,
        use_fp16: bool = True,
        device: str = "",
        keyframe_strategy: str = "motion",
        dist_thresh: float = 40.0,
        min_observations: int = 1,
        use_dbscan: bool = True,
        mask_alpha: float = 0.40,
    ) -> None:
        """
        初始化所有模块（模型只加载一次）。

        Parameters
        ----------
        detector_weights : Path | None
            检测器权重路径；None 则使用默认路径。
        classifier_weights : Path | None
            分类器权重路径；None 则使用默认路径。
        use_fp16 : bool
            是否使用 FP16 推理。
        device : str
            推理设备（'' 为自动选择）。
        keyframe_strategy : str
            关键帧提取策略：'motion' 或 'uniform'。
        dist_thresh : float
            去重聚类距离阈值（像素，参考坐标系）。
        min_observations : int
            Cluster 最少观测次数（低于此值被过滤）。
        use_dbscan : bool
            是否使用 DBSCAN 聚类。
        mask_alpha : float
            mask 叠加透明度。
        """
        self.keyframe_strategy = keyframe_strategy

        logger.info("初始化 VideoPipeline...")

        # 初始化各模块（模型只加载一次，供所有视频复用）
        detector_kwargs = {}
        if detector_weights is not None:
            detector_kwargs["weights_path"] = detector_weights

        classifier_kwargs = {}
        if classifier_weights is not None:
            classifier_kwargs["weights_path"] = classifier_weights

        logger.info("正在加载检测器（B 的模块）...")
        self.detector = Detector(
            use_fp16=use_fp16,
            device=device,
            **detector_kwargs,
        )

        logger.info("正在初始化配准器（A 的模块）...")
        self.registrar = FrameRegistration()

        logger.info("正在初始化去重聚类器（A 的模块）...")
        self.deduper = GlobalDedup(
            dist_thresh=dist_thresh,
            min_observations=min_observations,
            use_dbscan=use_dbscan,
        )

        logger.info("正在加载分类器（C 的模块）...")
        self.classifier = ScrewClassifier(
            use_fp16=use_fp16,
            device=device,
            **classifier_kwargs,
        )

        self.visualizer = Visualizer(
            mask_alpha=mask_alpha,
            show_label=True,
            use_circle_mask=True,
        )

        logger.info("VideoPipeline 初始化完成。")
        self._print_mode_summary()

    def _print_mode_summary(self) -> None:
        """打印各模块的运行模式摘要。"""
        det_mode = "YOLO" if self.detector.is_yolo_mode else "兜底(OpenCV)"
        clf_mode = "PyTorch" if self.classifier.is_torch_mode else "兜底(随机)"
        logger.info(
            "运行模式摘要:\n"
            "  检测器: %s\n"
            "  分类器: %s\n"
            "  关键帧策略: %s",
            det_mode, clf_mode, self.keyframe_strategy,
        )
        if not self.detector.is_yolo_mode:
            logger.warning("⚠️  检测器处于兜底模式，计数精度很低！请提供 models/detector.pt")
        if not self.classifier.is_torch_mode:
            logger.warning("⚠️  分类器处于兜底模式，分类结果随机！请提供 models/classifier.pt")

    def process_video(self, video_path: Path) -> VideoResult:
        """
        处理单段视频。

        Parameters
        ----------
        video_path : Path
            视频文件路径。

        Returns
        -------
        VideoResult : 包含计数、耗时和附加的 mask 图像（result._mask_image）。
        """
        # 每个视频处理前重置配准器的统计信息（不重置模型）
        self.registrar.reset_stats()

        return process_video(
            video_path=video_path,
            detector=self.detector,
            registrar=self.registrar,
            deduper=self.deduper,
            classifier=self.classifier,
            visualizer=self.visualizer,
            keyframe_strategy=self.keyframe_strategy,
        )

    def process_folder(
        self,
        data_dir: Path,
    ) -> Tuple[Dict[str, List[int]], Dict[str, Optional[np.ndarray]]]:
        """
        处理视频文件夹中的所有视频。

        Parameters
        ----------
        data_dir : Path
            包含视频文件的文件夹路径。

        Returns
        -------
        results : Dict[str, List[int]]
            {video_name: [count_type1, …, count_type5]}
        masks : Dict[str, Optional[np.ndarray]]
            {video_name: overlay_image}（overlay_image 可能为 None）
        """
        videos = list_videos(data_dir)
        if not videos:
            logger.error("在 '%s' 中未找到视频文件。", data_dir)
            return {}, {}

        results: Dict[str, List[int]] = {}
        masks: Dict[str, Optional[np.ndarray]] = {}

        logger.info("开始处理 %d 段视频...", len(videos))

        for i, video_path in enumerate(videos):
            video_name = get_video_name(video_path)
            logger.info(
                "[%d/%d] 处理: %s",
                i + 1, len(videos), video_path.name,
            )

            try:
                result = self.process_video(video_path)
                results[result.video_name] = result.counts
                masks[result.video_name] = getattr(result, "_mask_image", None)

            except FileNotFoundError as e:
                logger.error("视频文件不存在: %s", e)
                results[video_name] = [0] * 5
                masks[video_name] = None
            except Exception as e:
                logger.error(
                    "处理视频 '%s' 时发生意外错误: %s",
                    video_name, e, exc_info=True,
                )
                results[video_name] = [0] * 5
                masks[video_name] = None

        logger.info("=" * 60)
        logger.info("所有视频处理完成，结果汇总:")
        for vname, counts in results.items():
            logger.info("  %s: %s (总计=%d)", vname, counts, sum(counts))

        return results, masks

    @staticmethod
    def print_summary(
        results: Dict[str, List[int]],
        elapsed_seconds: float,
    ) -> None:
        """
        打印处理结果摘要表格。

        Parameters
        ----------
        results : Dict[str, List[int]]
            各视频的计数结果字典。
        elapsed_seconds : float
            总耗时（秒）。
        """
        print("\n" + "=" * 70)
        print(f"{'视频名称':30s}  {'T1':>4s}  {'T2':>4s}  {'T3':>4s}  {'T4':>4s}  {'T5':>4s}  {'总计':>4s}")
        print("-" * 70)

        for vname, counts in results.items():
            print(
                f"{vname:30s}  "
                f"{counts[0]:>4d}  {counts[1]:>4d}  "
                f"{counts[2]:>4d}  {counts[3]:>4d}  "
                f"{counts[4]:>4d}  {sum(counts):>4d}"
            )

        print("=" * 70)
        print(f"总耗时: {elapsed_seconds:.2f}s")
        if len(results) > 0:
            avg_time = elapsed_seconds / len(results)
            print(f"平均每视频: {avg_time:.2f}s")
        print()
