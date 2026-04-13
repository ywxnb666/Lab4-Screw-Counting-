#!/usr/bin/env python3
"""
tools/export_crops.py - 检测 Crop 导出工具
Owner: D（工程封装）

用途：
  从已标注的视频帧（或检测结果）中裁切螺丝区域（crop），
  导出为独立图像文件，供以下用途：
  1. C 进行 5 类分类标签的人工分拣（将 crop 按 Type_1~Type_5 手动归类）
  2. C 训练/fine-tune 分类器的数据集准备
  3. 可视化检测结果质量（调试用）

工作流程（对应 plan.md 6.2 节）：
  Step 1：从视频关键帧中运行检测器（或加载已有标注），得到 bbox 列表
  Step 2：按 bbox 裁切原图，加可选边距
  Step 3：导出为 {video_name}_frame{frame_id:06d}_crop{crop_id:04d}.jpg
  Step 4：（可选）生成 HTML 预览页，方便人工分拣

使用示例：
  # 从 YOLO 标注文件批量导出 crop（标注已知，不需要跑检测）
  python tools/export_crops.py \\
      --mode from_labels \\
      --video_dir ../../vedio_exp/ \\
      --labels_dir annotations/yolo_labels/ \\
      --frames_dir frames/ \\
      --output crops/

  # 从视频直接运行检测器导出 crop（需要 models/detector.pt）
  python tools/export_crops.py \\
      --mode from_detector \\
      --video_dir ../../vedio_exp/ \\
      --output crops/ \\
      --conf 0.35

  # 导出后生成 HTML 预览
  python tools/export_crops.py \\
      --mode from_labels \\
      --video_dir ../../vedio_exp/ \\
      --labels_dir annotations/yolo_labels/ \\
      --frames_dir frames/ \\
      --output crops/ \\
      --html_preview

  # 按置信度过滤，只导出高质量 crop
  python tools/export_crops.py \\
      --mode from_detector \\
      --video_dir ../../vedio_exp/ \\
      --output crops/ \\
      --conf 0.50 --min_size 32

依赖：opencv-python, numpy
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("export_crops")

# ---------------------------------------------------------------------------
# 支持的视频/图像格式
# ---------------------------------------------------------------------------

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv", ".flv", ".webm"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# ---------------------------------------------------------------------------
# 默认参数
# ---------------------------------------------------------------------------

DEFAULT_PADDING: float = 0.10        # crop 边距（相对于 bbox 短边的比例）
DEFAULT_MIN_SIZE: int = 20           # crop 最小边长（像素），小于此值的跳过
DEFAULT_TARGET_SIZE: int = 224       # 导出 crop 的目标边长（0=不缩放）
DEFAULT_JPEG_QUALITY: int = 92       # JPEG 压缩质量
DEFAULT_CONF_THRESHOLD: float = 0.35 # 检测置信度阈值（from_detector 模式）
DEFAULT_MAX_KEYFRAMES: int = 40      # 每视频最多处理的关键帧数


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class CropRecord:
    """单个 crop 的元数据记录。"""
    video_name: str
    frame_id: int
    crop_id: int
    bbox_abs: Tuple[float, float, float, float]   # (x1, y1, x2, y2) 绝对坐标
    bbox_norm: Tuple[float, float, float, float]  # (cx_n, cy_n, w_n, h_n) 归一化
    confidence: float
    class_id: int        # -1 表示未知（from_detector 模式）
    class_name: str      # "unknown" 表示未知
    save_path: str       # 相对于 output_dir 的路径
    sharpness: float     # 拉普拉斯方差（清晰度估计）
    frame_w: int
    frame_h: int

    def to_dict(self) -> dict:
        return {
            "video_name": self.video_name,
            "frame_id": self.frame_id,
            "crop_id": self.crop_id,
            "bbox_x1": self.bbox_abs[0],
            "bbox_y1": self.bbox_abs[1],
            "bbox_x2": self.bbox_abs[2],
            "bbox_y2": self.bbox_abs[3],
            "bbox_cx_norm": self.bbox_norm[0],
            "bbox_cy_norm": self.bbox_norm[1],
            "bbox_w_norm": self.bbox_norm[2],
            "bbox_h_norm": self.bbox_norm[3],
            "confidence": round(self.confidence, 4),
            "class_id": self.class_id,
            "class_name": self.class_name,
            "save_path": self.save_path,
            "sharpness": round(self.sharpness, 2),
            "frame_w": self.frame_w,
            "frame_h": self.frame_h,
        }


@dataclass
class ExportStats:
    """导出统计信息。"""
    total_videos: int = 0
    total_frames: int = 0
    total_crops: int = 0
    skipped_too_small: int = 0
    skipped_low_conf: int = 0
    failed_frames: int = 0
    elapsed_s: float = 0.0
    per_video: Dict[str, int] = field(default_factory=dict)

    def print(self) -> None:
        """打印导出统计报告。"""
        print("\n" + "=" * 55)
        print("Crop 导出统计")
        print("=" * 55)
        print(f"  处理视频数  : {self.total_videos}")
        print(f"  处理帧数    : {self.total_frames}")
        print(f"  导出 crop 数: {self.total_crops}")
        print(f"  跳过（太小）: {self.skipped_too_small}")
        print(f"  跳过（低置信度）: {self.skipped_low_conf}")
        print(f"  帧读取失败  : {self.failed_frames}")
        print(f"  总耗时      : {self.elapsed_s:.2f}s")
        print()
        print("  各视频导出数:")
        for vname, count in sorted(self.per_video.items()):
            print(f"    {vname:30s}: {count:4d} crops")
        print("=" * 55 + "\n")


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _get_rotation(cap: cv2.VideoCapture) -> int:
    """读取视频旋转角度。"""
    try:
        rot = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))
        return rot if rot in (0, 90, 180, 270) else 0
    except Exception:
        return 0


def _apply_rotation(frame: np.ndarray, rotation: int) -> np.ndarray:
    """应用旋转校正。"""
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def _estimate_sharpness(crop: np.ndarray) -> float:
    """
    估计 crop 的清晰度（拉普拉斯方差，越大越清晰）。

    Parameters
    ----------
    crop : np.ndarray
        输入图像（H×W×3 BGR）。

    Returns
    -------
    float
    """
    if crop is None or crop.size == 0:
        return 0.0
    if crop.ndim == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop
    if gray.size < 4:
        return 0.0
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _crop_with_padding(
    frame: np.ndarray,
    bbox: Tuple[float, float, float, float],
    padding: float = DEFAULT_PADDING,
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """
    从帧中按 bbox 裁切 crop，并额外扩展 padding 比例的边距。

    Parameters
    ----------
    frame : np.ndarray
        源帧（H×W×3 BGR）。
    bbox : Tuple[float, float, float, float]
        (x1, y1, x2, y2) 绝对坐标。
    padding : float
        相对于 bbox 短边的扩展比例。

    Returns
    -------
    crop : np.ndarray
        裁切后的图像（不越界）。
    actual_bbox : Tuple[float, float, float, float]
        实际裁切区域的坐标（含 padding，不越界）。
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox

    if padding > 0:
        bw = x2 - x1
        bh = y2 - y1
        pad_px = min(bw, bh) * padding
        x1 -= pad_px
        y1 -= pad_px
        x2 += pad_px
        y2 += pad_px

    # 裁剪到图像范围
    x1 = max(0.0, x1)
    y1 = max(0.0, y1)
    x2 = min(float(w), x2)
    y2 = min(float(h), y2)

    ix1, iy1 = int(round(x1)), int(round(y1))
    ix2, iy2 = int(round(x2)), int(round(y2))

    if ix2 <= ix1 or iy2 <= iy1:
        return np.zeros((1, 1, 3), dtype=np.uint8), (x1, y1, x2, y2)

    crop = frame[iy1:iy2, ix1:ix2].copy()
    return crop, (x1, y1, x2, y2)


def _resize_square(
    image: np.ndarray,
    size: int,
    pad_color: Tuple[int, int, int] = (114, 114, 114),
) -> np.ndarray:
    """
    等比缩放 + Letterbox 填充到 size×size 正方形。

    Parameters
    ----------
    image : np.ndarray
    size : int
    pad_color : Tuple[int, int, int]

    Returns
    -------
    np.ndarray : size×size×3
    """
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return np.full((size, size, 3), pad_color, dtype=np.uint8)

    scale = size / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((size, size, 3), pad_color, dtype=np.uint8)
    top = (size - new_h) // 2
    left = (size - new_w) // 2
    canvas[top: top + new_h, left: left + new_w] = resized
    return canvas


def _bbox_abs_to_norm(
    bbox: Tuple[float, float, float, float],
    img_w: int,
    img_h: int,
) -> Tuple[float, float, float, float]:
    """
    将绝对坐标 (x1, y1, x2, y2) 转换为归一化 YOLO 格式 (cx, cy, w, h)。

    Parameters
    ----------
    bbox : Tuple
    img_w : int
    img_h : int

    Returns
    -------
    Tuple[float, float, float, float]
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    return cx, cy, bw, bh


# ---------------------------------------------------------------------------
# 从 YOLO 标注文件解析 bbox
# ---------------------------------------------------------------------------

def _load_yolo_bboxes(
    label_path: Path,
    img_w: int,
    img_h: int,
) -> List[Tuple[int, float, float, float, float, float]]:
    """
    解析单个 YOLO 标注文件，返回 bbox 列表。

    Parameters
    ----------
    label_path : Path
        .txt 标注文件路径。
    img_w : int
        图像宽度（像素）。
    img_h : int
        图像高度（像素）。

    Returns
    -------
    List of (class_id, x1, y1, x2, y2, confidence=1.0)
    """
    if not label_path.exists():
        return []

    bboxes = []
    try:
        with open(label_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                class_id = int(parts[0])
                cx_n, cy_n, w_n, h_n = map(float, parts[1:5])

                # 归一化坐标合法性检查
                if not (0.0 < w_n <= 1.0 and 0.0 < h_n <= 1.0):
                    continue

                cx = cx_n * img_w
                cy = cy_n * img_h
                bw = w_n * img_w
                bh = h_n * img_h
                x1 = cx - bw / 2.0
                y1 = cy - bh / 2.0
                x2 = cx + bw / 2.0
                y2 = cy + bh / 2.0

                # 置信度字段（可选，YOLO 标注中通常没有）
                conf = float(parts[5]) if len(parts) > 5 else 1.0
                bboxes.append((class_id, x1, y1, x2, y2, conf))

    except Exception as e:
        logger.warning("解析标注文件失败 %s: %s", label_path, e)

    return bboxes


# ---------------------------------------------------------------------------
# 核心导出器
# ---------------------------------------------------------------------------

class CropExporter:
    """
    螺丝 Crop 导出器。

    支持两种来源：
    1. from_labels  : 从已有 YOLO 标注文件 + 帧图像中裁切
    2. from_detector: 实时运行检测器（需要 models/detector.pt）

    用法
    ----
    >>> exporter = CropExporter(output_dir="crops/", padding=0.10)
    >>> records = exporter.export_from_labels(
    ...     frames_dir="frames/",
    ...     labels_dir="annotations/yolo_labels/",
    ...     class_names=["Type_1", "Type_2", "Type_3", "Type_4", "Type_5"],
    ... )
    >>> exporter.save_manifest(records)
    """

    def __init__(
        self,
        output_dir: str | Path,
        padding: float = DEFAULT_PADDING,
        min_size: int = DEFAULT_MIN_SIZE,
        target_size: int = DEFAULT_TARGET_SIZE,
        jpeg_quality: int = DEFAULT_JPEG_QUALITY,
        organize_by_class: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        output_dir : str | Path
            crop 图像的输出根目录。
        padding : float
            crop 边距（相对于 bbox 短边的比例，默认 0.10 即扩展 10%）。
        min_size : int
            crop 最小边长（像素），小于此值的 crop 将被跳过。
        target_size : int
            导出 crop 的目标边长（正方形，0 表示不缩放保留原始 crop 尺寸）。
        jpeg_quality : int
            JPEG 压缩质量（0-100）。
        organize_by_class : bool
            True: 按类别名称创建子目录（方便人工分拣）；
            False: 所有 crop 放在同一目录下。
        """
        self.output_dir = Path(output_dir)
        self.padding = padding
        self.min_size = min_size
        self.target_size = target_size
        self.jpeg_quality = jpeg_quality
        self.organize_by_class = organize_by_class

        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "CropExporter 初始化: output=%s, padding=%.2f, "
            "min_size=%d, target_size=%d",
            self.output_dir, padding, min_size, target_size,
        )

    # ------------------------------------------------------------------
    # 模式 1：从 YOLO 标注文件导出（标注已知）
    # ------------------------------------------------------------------

    def export_from_labels(
        self,
        frames_dir: str | Path,
        labels_dir: str | Path,
        class_names: Optional[List[str]] = None,
        video_name: Optional[str] = None,
    ) -> List[CropRecord]:
        """
        从帧图像 + YOLO 标注文件中裁切并导出 crop。

        文件名对应规则：
          frames_dir/{stem}.jpg  ↔  labels_dir/{stem}.txt

        Parameters
        ----------
        frames_dir : str | Path
            关键帧图像所在目录（由 extract_keyframes.py 导出）。
        labels_dir : str | Path
            YOLO 格式标注文件所在目录。
        class_names : List[str] | None
            类别名称列表（顺序对应 class_id）；
            None 时尝试从 labels_dir/classes.txt 读取，找不到则使用数字 ID。
        video_name : str | None
            视频名称（用于 crop 文件命名）；
            None 时从 frames_dir 名称推断。

        Returns
        -------
        List[CropRecord] : 所有成功导出的 crop 记录。
        """
        frames_dir = Path(frames_dir)
        labels_dir = Path(labels_dir)

        if not frames_dir.is_dir():
            raise NotADirectoryError(f"帧图像目录不存在: {frames_dir}")
        if not labels_dir.is_dir():
            raise NotADirectoryError(f"标注目录不存在: {labels_dir}")

        # 推断类别名称
        if class_names is None:
            classes_txt = labels_dir / "classes.txt"
            if classes_txt.exists():
                class_names = classes_txt.read_text(encoding="utf-8").strip().splitlines()
                logger.info("从 classes.txt 读取类别: %s", class_names)
            else:
                class_names = []
                logger.warning("未提供类别名称，将使用数字 ID 作为类别标识。")

        # 收集所有帧图像
        image_files = sorted(
            p for p in frames_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

        if not image_files:
            logger.warning("帧图像目录为空: %s", frames_dir)
            return []

        logger.info("找到 %d 张帧图像，开始导出 crop...", len(image_files))

        if video_name is None:
            video_name = frames_dir.name

        records: List[CropRecord] = []
        global_crop_id = 0

        for img_path in image_files:
            label_path = labels_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                logger.debug("无对应标注文件，跳过: %s", img_path.name)
                continue

            frame = cv2.imread(str(img_path))
            if frame is None:
                logger.warning("读取帧图像失败: %s", img_path)
                continue

            h, w = frame.shape[:2]

            # 解析帧编号（从文件名提取）
            frame_id = self._parse_frame_id(img_path.stem)

            bboxes = _load_yolo_bboxes(label_path, w, h)
            if not bboxes:
                continue

            for class_id, x1, y1, x2, y2, conf in bboxes:
                # bbox 尺寸检查
                bw = x2 - x1
                bh = y2 - y1
                if min(bw, bh) < self.min_size:
                    logger.debug(
                        "crop 太小 (%.1fx%.1f < %d)，跳过。",
                        bw, bh, self.min_size,
                    )
                    continue

                crop, actual_bbox = _crop_with_padding(
                    frame, (x1, y1, x2, y2), self.padding
                )

                if crop is None or min(crop.shape[:2]) < self.min_size:
                    continue

                # 可选缩放
                if self.target_size > 0:
                    crop = _resize_square(crop, self.target_size)

                class_name = (
                    class_names[class_id]
                    if 0 <= class_id < len(class_names)
                    else f"class_{class_id}"
                )

                # 确定保存路径
                save_subdir = self._get_save_subdir(video_name, class_name)
                filename = (
                    f"{video_name}_frame{frame_id:06d}"
                    f"_crop{global_crop_id:04d}.jpg"
                )
                save_path = save_subdir / filename
                save_path.parent.mkdir(parents=True, exist_ok=True)

                # 保存 crop
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
                success = cv2.imwrite(str(save_path), crop, encode_params)

                if not success:
                    logger.warning("crop 保存失败: %s", save_path)
                    continue

                sharpness = _estimate_sharpness(crop)
                bbox_norm = _bbox_abs_to_norm(actual_bbox, w, h)

                record = CropRecord(
                    video_name=video_name,
                    frame_id=frame_id,
                    crop_id=global_crop_id,
                    bbox_abs=actual_bbox,
                    bbox_norm=bbox_norm,
                    confidence=conf,
                    class_id=class_id,
                    class_name=class_name,
                    save_path=str(save_path.relative_to(self.output_dir)),
                    sharpness=sharpness,
                    frame_w=w,
                    frame_h=h,
                )
                records.append(record)
                global_crop_id += 1

        logger.info(
            "from_labels 导出完成: %d 张帧 → %d 个 crop",
            len(image_files), len(records),
        )
        return records

    # ------------------------------------------------------------------
    # 模式 2：从视频实时检测导出
    # ------------------------------------------------------------------

    def export_from_detector(
        self,
        video_path: str | Path,
        detector=None,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        max_keyframes: int = DEFAULT_MAX_KEYFRAMES,
    ) -> Tuple[List[CropRecord], ExportStats]:
        """
        实时运行检测器，从视频中裁切并导出 crop。

        Parameters
        ----------
        video_path : str | Path
            视频文件路径。
        detector : modules.detector.Detector | None
            已初始化的检测器实例；None 时自动初始化（需要 models/detector.pt）。
        conf_threshold : float
            检测置信度阈值（低于此值的检测被丢弃）。
        max_keyframes : int
            每视频最多处理的关键帧数（均匀采样）。

        Returns
        -------
        records : List[CropRecord]
            所有导出的 crop 记录。
        stats : ExportStats
            导出统计信息。
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        video_name = video_path.stem
        stats = ExportStats(total_videos=1)
        t_start = time.perf_counter()

        # 初始化检测器（若未提供）
        if detector is None:
            try:
                _root = Path(__file__).parent.parent
                if str(_root) not in sys.path:
                    sys.path.insert(0, str(_root))
                from modules.detector import Detector
                detector = Detector(conf_threshold=conf_threshold)
            except Exception as e:
                logger.error("检测器初始化失败: %s", e)
                stats.elapsed_s = time.perf_counter() - t_start
                return [], stats

        # 打开视频
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error("无法打开视频: %s", video_path)
            stats.elapsed_s = time.perf_counter() - t_start
            return [], stats

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                total_frames = 300  # 兜底估计

            rotation = _get_rotation(cap)

            # 均匀采样关键帧
            n_kf = min(max_keyframes, total_frames)
            kf_ids = sorted(set(
                int(np.round(i * (total_frames - 1) / max(n_kf - 1, 1)))
                for i in range(n_kf)
            ))

            records: List[CropRecord] = []
            global_crop_id = 0

            for target_id in kf_ids:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_id)
                ret, frame = cap.read()
                if not ret or frame is None:
                    stats.failed_frames += 1
                    continue

                frame = _apply_rotation(frame, rotation)
                h, w = frame.shape[:2]
                stats.total_frames += 1

                # 运行检测器
                try:
                    detections = detector.detect(frame, frame_id=target_id)
                except Exception as e:
                    logger.warning("检测失败 (frame=%d): %s", target_id, e)
                    stats.failed_frames += 1
                    continue

                for det in detections:
                    if det.confidence < conf_threshold:
                        stats.skipped_low_conf += 1
                        continue

                    x1, y1, x2, y2 = det.bbox[:4]
                    bw = x2 - x1
                    bh = y2 - y1
                    if min(bw, bh) < self.min_size:
                        stats.skipped_too_small += 1
                        continue

                    crop, actual_bbox = _crop_with_padding(
                        frame, (x1, y1, x2, y2), self.padding
                    )
                    if crop is None or min(crop.shape[:2]) < self.min_size:
                        stats.skipped_too_small += 1
                        continue

                    if self.target_size > 0:
                        crop = _resize_square(crop, self.target_size)

                    # 检测器输出类别未知，统一放到 "unknown" 目录
                    class_name = "unknown"
                    class_id = -1

                    save_subdir = self._get_save_subdir(video_name, class_name)
                    filename = (
                        f"{video_name}_frame{target_id:06d}"
                        f"_crop{global_crop_id:04d}.jpg"
                    )
                    save_path = save_subdir / filename
                    save_path.parent.mkdir(parents=True, exist_ok=True)

                    encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
                    success = cv2.imwrite(str(save_path), crop, encode_params)
                    if not success:
                        continue

                    sharpness = _estimate_sharpness(crop)
                    bbox_norm = _bbox_abs_to_norm(actual_bbox, w, h)

                    record = CropRecord(
                        video_name=video_name,
                        frame_id=target_id,
                        crop_id=global_crop_id,
                        bbox_abs=actual_bbox,
                        bbox_norm=bbox_norm,
                        confidence=det.confidence,
                        class_id=class_id,
                        class_name=class_name,
                        save_path=str(save_path.relative_to(self.output_dir)),
                        sharpness=sharpness,
                        frame_w=w,
                        frame_h=h,
                    )
                    records.append(record)
                    global_crop_id += 1
                    stats.total_crops += 1

        finally:
            cap.release()

        stats.elapsed_s = time.perf_counter() - t_start
        stats.total_crops = len(records)
        stats.per_video[video_name] = len(records)

        logger.info(
            "from_detector 导出完成 (%s): %d 帧 → %d 个 crop (%.2fs)",
            video_name, stats.total_frames, len(records), stats.elapsed_s,
        )
        return records, stats

    # ------------------------------------------------------------------
    # 批量处理（文件夹）
    # ------------------------------------------------------------------

    def export_folder_from_labels(
        self,
        frames_root: str | Path,
        labels_dir: str | Path,
        class_names: Optional[List[str]] = None,
    ) -> Tuple[List[CropRecord], ExportStats]:
        """
        批量从文件夹处理多个视频的帧。

        约定：frames_root/{video_name}/ 为各视频的帧目录。

        Parameters
        ----------
        frames_root : str | Path
            帧图像根目录（每个视频一个子目录）。
        labels_dir : str | Path
            YOLO 标注文件目录（支持与帧同目录或独立目录两种结构）。
        class_names : List[str] | None
            类别名称列表。

        Returns
        -------
        all_records : List[CropRecord]
        stats : ExportStats
        """
        frames_root = Path(frames_root)
        labels_dir = Path(labels_dir)
        t_start = time.perf_counter()

        all_records: List[CropRecord] = []
        stats = ExportStats()

        # 枚举所有视频子目录
        video_dirs = sorted(
            d for d in frames_root.iterdir() if d.is_dir()
        )

        # 若无子目录，将 frames_root 本身视为单个视频目录
        if not video_dirs:
            video_dirs = [frames_root]

        stats.total_videos = len(video_dirs)

        for video_dir in video_dirs:
            video_name = video_dir.name

            # 标注目录：优先同名子目录，其次使用 labels_dir 本身
            if (labels_dir / video_name).is_dir():
                vl_dir = labels_dir / video_name
            else:
                vl_dir = labels_dir

            try:
                records = self.export_from_labels(
                    frames_dir=video_dir,
                    labels_dir=vl_dir,
                    class_names=class_names,
                    video_name=video_name,
                )
                all_records.extend(records)
                stats.total_crops += len(records)
                stats.per_video[video_name] = len(records)
            except Exception as e:
                logger.error("处理视频 '%s' 失败: %s", video_name, e)
                stats.per_video[video_name] = 0

        stats.elapsed_s = time.perf_counter() - t_start
        return all_records, stats

    def export_folder_from_detector(
        self,
        video_dir: str | Path,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        max_keyframes: int = DEFAULT_MAX_KEYFRAMES,
    ) -> Tuple[List[CropRecord], ExportStats]:
        """
        批量对文件夹中的所有视频运行检测器并导出 crop。

        Parameters
        ----------
        video_dir : str | Path
            包含视频文件的目录。
        conf_threshold : float
            检测置信度阈值。
        max_keyframes : int
            每视频最多处理的关键帧数。

        Returns
        -------
        all_records : List[CropRecord]
        stats : ExportStats
        """
        video_dir = Path(video_dir)
        videos = sorted(
            p for p in video_dir.iterdir()
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        )

        if not videos:
            logger.error("未找到视频文件: %s", video_dir)
            return [], ExportStats()

        # 初始化检测器（只初始化一次，供所有视频复用）
        try:
            _root = Path(__file__).parent.parent
            if str(_root) not in sys.path:
                sys.path.insert(0, str(_root))
            from modules.detector import Detector
            detector = Detector(conf_threshold=conf_threshold)
        except Exception as e:
            logger.error("检测器初始化失败: %s", e)
            return [], ExportStats()

        all_records: List[CropRecord] = []
        agg_stats = ExportStats(total_videos=len(videos))

        for vp in videos:
            records, vstats = self.export_from_detector(
                video_path=vp,
                detector=detector,
                conf_threshold=conf_threshold,
                max_keyframes=max_keyframes,
            )
            all_records.extend(records)
            agg_stats.total_frames += vstats.total_frames
            agg_stats.total_crops += vstats.total_crops
            agg_stats.skipped_too_small += vstats.skipped_too_small
            agg_stats.skipped_low_conf += vstats.skipped_low_conf
            agg_stats.failed_frames += vstats.failed_frames
            agg_stats.per_video.update(vstats.per_video)

        return all_records, agg_stats

    # ------------------------------------------------------------------
    # 导出清单
    # ------------------------------------------------------------------

    def save_manifest(
        self,
        records: List[CropRecord],
        filename: str = "crop_manifest.json",
    ) -> Path:
        """
        将所有 crop 记录保存为 JSON 清单文件。

        清单文件供 C 用于：
        - 了解每个 crop 的来源帧和 bbox
        - 按清晰度（sharpness）筛选高质量 crop
        - 按类别（class_name）统计数量

        Parameters
        ----------
        records : List[CropRecord]
            所有 crop 记录。
        filename : str
            清单文件名（保存在 output_dir 下）。

        Returns
        -------
        Path : 清单文件路径。
        """
        manifest_path = self.output_dir / filename

        # 统计信息
        class_counts: Dict[str, int] = {}
        for r in records:
            class_counts[r.class_name] = class_counts.get(r.class_name, 0) + 1

        data = {
            "meta": {
                "total_crops": len(records),
                "class_counts": class_counts,
                "output_dir": str(self.output_dir),
            },
            "crops": [r.to_dict() for r in records],
        }

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(
            "清单已保存: %s  (%d 条记录，类别分布: %s)",
            manifest_path, len(records), class_counts,
        )
        return manifest_path

    def generate_html_preview(
        self,
        records: List[CropRecord],
        filename: str = "crop_preview.html",
        max_per_class: int = 200,
    ) -> Path:
        """
        生成 HTML 预览页，方便人工分拣 crop 类别。

        HTML 页面按类别分组显示所有 crop 缩略图，
        点击图片可在新标签页查看原图。

        Parameters
        ----------
        records : List[CropRecord]
            所有 crop 记录。
        filename : str
            HTML 文件名（保存在 output_dir 下）。
        max_per_class : int
            每类最多显示的 crop 数（防止页面太大）。

        Returns
        -------
        Path : HTML 文件路径。
        """
        # 按 class_name 分组
        by_class: Dict[str, List[CropRecord]] = {}
        for r in records:
            by_class.setdefault(r.class_name, []).append(r)

        # 按清晰度降序排列（最清晰的排在前面）
        for cls_records in by_class.values():
            cls_records.sort(key=lambda r: r.sharpness, reverse=True)

        lines = [
            "<!DOCTYPE html>",
            '<html lang="zh">',
            "<head>",
            '<meta charset="UTF-8">',
            "<title>螺丝 Crop 预览 - Lab4 团队作业</title>",
            "<style>",
            "  body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }",
            "  h1 { color: #333; }",
            "  h2 { color: #555; border-bottom: 2px solid #aaa; padding-bottom: 5px; }",
            "  .class-section { margin-bottom: 40px; background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }",
            "  .crop-grid { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }",
            "  .crop-item { text-align: center; cursor: pointer; }",
            "  .crop-item img { width: 100px; height: 100px; object-fit: contain; border: 2px solid #ddd; border-radius: 4px; background: #eee; }",
            "  .crop-item img:hover { border-color: #4a90e2; transform: scale(1.05); transition: all 0.2s; }",
            "  .crop-label { font-size: 10px; color: #777; max-width: 100px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }",
            "  .stats { background: #e8f4e8; padding: 10px; border-radius: 5px; margin-bottom: 20px; }",
            "</style>",
            "</head>",
            "<body>",
            "<h1>螺丝 Crop 预览</h1>",
            '<div class="stats">',
            f"<strong>总计：</strong>{len(records)} 个 crop，"
            f"{len(by_class)} 个类别",
            "</div>",
        ]

        for class_name in sorted(by_class.keys()):
            class_records = by_class[class_name][:max_per_class]
            lines.extend([
                '<div class="class-section">',
                f"<h2>{class_name} ({len(by_class[class_name])} 个"
                + (f"，显示前 {max_per_class} 个" if len(by_class[class_name]) > max_per_class else "")
                + ")</h2>",
                '<div class="crop-grid">',
            ])

            for r in class_records:
                # 使用相对路径（相对于 HTML 文件所在目录 = output_dir）
                img_rel = r.save_path.replace("\\", "/")
                label = f"f{r.frame_id} c{r.crop_id} sh={r.sharpness:.0f}"
                lines.extend([
                    '<div class="crop-item">',
                    f'  <a href="{img_rel}" target="_blank">',
                    f'    <img src="{img_rel}" alt="{r.class_name}" '
                    f'title="frame={r.frame_id} conf={r.confidence:.3f} sharpness={r.sharpness:.1f}">',
                    "  </a>",
                    f'  <div class="crop-label">{label}</div>',
                    "</div>",
                ])

            lines.extend(["</div>", "</div>"])

        lines.extend(["</body>", "</html>"])

        html_path = self.output_dir / filename
        html_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("HTML 预览已生成: %s", html_path)
        return html_path

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _get_save_subdir(self, video_name: str, class_name: str) -> Path:
        """
        根据 organize_by_class 决定 crop 的保存子目录。

        按类别组织：output_dir/{class_name}/{video_name}/
        不按类别：  output_dir/{video_name}/
        """
        if self.organize_by_class:
            return self.output_dir / class_name / video_name
        else:
            return self.output_dir / video_name

    @staticmethod
    def _parse_frame_id(stem: str) -> int:
        """
        从文件名 stem 中提取帧编号。

        约定文件名格式：{video_name}_frame{frame_id:06d}（由 extract_keyframes.py 生成）。

        Parameters
        ----------
        stem : str
            文件名（不含后缀）。

        Returns
        -------
        int : 帧编号；若无法提取则返回 0。
        """
        import re
        match = re.search(r"frame(\d+)", stem)
        if match:
            return int(match.group(1))
        # 尝试提取文件名末尾的纯数字
        match2 = re.search(r"(\d+)$", stem)
        if match2:
            return int(match2.group(1))
        return 0


# ---------------------------------------------------------------------------
# 清晰度过滤工具
# ---------------------------------------------------------------------------

def filter_by_sharpness(
    records: List[CropRecord],
    min_sharpness: float = 50.0,
) -> Tuple[List[CropRecord], List[CropRecord]]:
    """
    按清晰度过滤 crop 记录。

    Parameters
    ----------
    records : List[CropRecord]
        所有 crop 记录。
    min_sharpness : float
        最低清晰度阈值（拉普拉斯方差，默认 50.0）。

    Returns
    -------
    good : List[CropRecord]
        清晰度 >= min_sharpness 的记录。
    blurry : List[CropRecord]
        清晰度 < min_sharpness 的记录（模糊 crop）。
    """
    good = [r for r in records if r.sharpness >= min_sharpness]
    blurry = [r for r in records if r.sharpness < min_sharpness]
    logger.info(
        "清晰度过滤 (threshold=%.1f): %d 合格 / %d 模糊（共 %d）",
        min_sharpness, len(good), len(blurry), len(records),
    )
    return good, blurry


def print_class_distribution(records: List[CropRecord]) -> None:
    """
    打印 crop 的类别分布统计。

    Parameters
    ----------
    records : List[CropRecord]
    """
    if not records:
        print("没有 crop 记录。")
        return

    counts: Dict[str, int] = {}
    for r in records:
        counts[r.class_name] = counts.get(r.class_name, 0) + 1

    total = len(records)
    print("\n[类别分布]")
    for cls_name, cnt in sorted(counts.items()):
        pct = cnt / total * 100
        bar = "█" * max(1, int(pct / 2))
        print(f"  {cls_name:20s}: {cnt:5d}  ({pct:5.1f}%)  {bar}")
    print(f"  {'合计':20s}: {total:5d}")
    print()


# ---------------------------------------------------------------------------
# 命令行接口
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        prog="export_crops.py",
        description="螺丝 Crop 导出工具 (Owner: D) — 用于分类训练数据准备",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用模式：
  from_labels   : 从已有 YOLO 标注 + 帧图像裁切（推荐，标注质量可控）
  from_detector : 实时运行检测器从视频裁切（无需预先标注，但精度依赖检测器）

典型工作流（from_labels）：
  1. 先用 extract_keyframes.py 抽帧：
     python tools/extract_keyframes.py --input vedio_exp/ --output frames/

  2. 在 CVAT 上标注帧（one-class bbox），导出 YOLO 格式到 annotations/yolo_labels/

  3. 用本工具导出 crop：
     python tools/export_crops.py \\
         --mode from_labels \\
         --frames_dir frames/ \\
         --labels_dir annotations/yolo_labels/ \\
         --output crops/ \\
         --class_names screw

  4. C 手动将 crops/unknown/ 中的图片分拣到 Type_1~Type_5 子目录

示例（from_detector）：
  python tools/export_crops.py \\
      --mode from_detector \\
      --video_dir ../../vedio_exp/ \\
      --output crops/ \\
      --conf 0.40 \\
      --html_preview
        """,
    )

    # ---- 模式 ----
    parser.add_argument(
        "--mode",
        required=True,
        choices=["from_labels", "from_detector"],
        help="导出模式：from_labels（从标注文件）或 from_detector（实时检测）。",
    )

    # ---- 输入路径 ----
    input_group = parser.add_argument_group("输入路径")
    input_group.add_argument(
        "--video_dir",
        type=str,
        default=None,
        help="视频文件夹路径（from_detector 模式使用）。",
    )
    input_group.add_argument(
        "--frames_dir",
        type=str,
        default=None,
        help="关键帧图像目录（from_labels 模式使用）。"
             "若包含子目录（每视频一个），则批量处理所有子目录。",
    )
    input_group.add_argument(
        "--labels_dir",
        type=str,
        default=None,
        help="YOLO 格式标注文件目录（from_labels 模式使用）。",
    )
    input_group.add_argument(
        "--class_names",
        nargs="+",
        default=None,
        help="类别名称列表（空格分隔）。例如：--class_names screw"
             " 或 --class_names Type_1 Type_2 Type_3 Type_4 Type_5",
    )

    # ---- 输出 ----
    output_group = parser.add_argument_group("输出")
    output_group.add_argument(
        "--output", "-o",
        required=True,
        help="crop 图像输出根目录。",
    )
    output_group.add_argument(
        "--no_class_subdir",
        action="store_true",
        default=False,
        help="不按类别创建子目录（所有 crop 放在同一目录下）。",
    )
    output_group.add_argument(
        "--html_preview",
        action="store_true",
        default=False,
        help="生成 HTML 预览页（方便浏览器中查看和分拣 crop）。",
    )

    # ---- Crop 参数 ----
    crop_group = parser.add_argument_group("Crop 参数")
    crop_group.add_argument(
        "--padding",
        type=float,
        default=DEFAULT_PADDING,
        help=f"crop 边距，相对于 bbox 短边的比例（默认 {DEFAULT_PADDING}，即 10%%）。",
    )
    crop_group.add_argument(
        "--min_size",
        type=int,
        default=DEFAULT_MIN_SIZE,
        help=f"crop 最小边长（像素，默认 {DEFAULT_MIN_SIZE}）。",
    )
    crop_group.add_argument(
        "--target_size",
        type=int,
        default=DEFAULT_TARGET_SIZE,
        help=f"导出 crop 的目标边长（正方形，默认 {DEFAULT_TARGET_SIZE}；0=保留原始尺寸）。",
    )
    crop_group.add_argument(
        "--quality",
        type=int,
        default=DEFAULT_JPEG_QUALITY,
        help=f"JPEG 压缩质量（默认 {DEFAULT_JPEG_QUALITY}）。",
    )

    # ---- 检测器参数（from_detector 模式）----
    det_group = parser.add_argument_group("检测器参数（from_detector 模式）")
    det_group.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONF_THRESHOLD,
        help=f"检测置信度阈值（默认 {DEFAULT_CONF_THRESHOLD}）。",
    )
    det_group.add_argument(
        "--max_keyframes",
        type=int,
        default=DEFAULT_MAX_KEYFRAMES,
        help=f"每视频最多处理的关键帧数（默认 {DEFAULT_MAX_KEYFRAMES}）。",
    )

    # ---- 其他 ----
    parser.add_argument(
        "--min_sharpness",
        type=float,
        default=0.0,
        help="导出后按清晰度过滤，低于此值的 crop 被剔除（0=不过滤）。",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="输出详细日志。",
    )

    return parser.parse_args()


def main() -> int:
    """
    Crop 导出工具主函数。

    Returns
    -------
    int : 退出码（0=成功，非 0=失败）。
    """
    args = _parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    output_dir = Path(args.output)

    # ---- 初始化导出器 ----
    exporter = CropExporter(
        output_dir=output_dir,
        padding=args.padding,
        min_size=args.min_size,
        target_size=args.target_size,
        jpeg_quality=args.quality,
        organize_by_class=not args.no_class_subdir,
    )

    t_start = time.perf_counter()
    all_records: List[CropRecord] = []
    stats = ExportStats()

    # ================================================================
    # 模式 1：from_labels（从标注文件导出）
    # ================================================================
    if args.mode == "from_labels":
        if not args.frames_dir:
            logger.error("from_labels 模式下必须提供 --frames_dir。")
            return 1
        if not args.labels_dir:
            logger.error("from_labels 模式下必须提供 --labels_dir。")
            return 1

        frames_dir = Path(args.frames_dir)
        labels_dir = Path(args.labels_dir)

        if not frames_dir.exists():
            logger.error("--frames_dir 不存在: %s", frames_dir)
            return 1
        if not labels_dir.exists():
            logger.error("--labels_dir 不存在: %s", labels_dir)
            return 1

        # 判断是单目录还是多视频根目录
        has_subdirs = any(d.is_dir() for d in frames_dir.iterdir())

        if has_subdirs:
            logger.info("检测到多视频子目录结构，批量处理...")
            all_records, stats = exporter.export_folder_from_labels(
                frames_root=frames_dir,
                labels_dir=labels_dir,
                class_names=args.class_names,
            )
        else:
            logger.info("单视频帧目录，直接处理...")
            all_records = exporter.export_from_labels(
                frames_dir=frames_dir,
                labels_dir=labels_dir,
                class_names=args.class_names,
            )
            stats.total_crops = len(all_records)
            stats.total_videos = 1

    # ================================================================
    # 模式 2：from_detector（实时检测导出）
    # ================================================================
    elif args.mode == "from_detector":
        if not args.video_dir:
            logger.error("from_detector 模式下必须提供 --video_dir。")
            return 1

        video_dir = Path(args.video_dir)
        if not video_dir.exists():
            logger.error("--video_dir 不存在: %s", video_dir)
            return 1

        if video_dir.is_file():
            # 单个视频文件
            all_records, stats = exporter.export_from_detector(
                video_path=video_dir,
                conf_threshold=args.conf,
                max_keyframes=args.max_keyframes,
            )
        else:
            # 视频文件夹
            all_records, stats = exporter.export_folder_from_detector(
                video_dir=video_dir,
                conf_threshold=args.conf,
                max_keyframes=args.max_keyframes,
            )

    stats.elapsed_s = time.perf_counter() - t_start

    if not all_records:
        logger.warning("未导出任何 crop，请检查输入路径和参数。")
        return 1

    # ---- 可选：按清晰度过滤 ----
    if args.min_sharpness > 0:
        all_records, blurry = filter_by_sharpness(all_records, args.min_sharpness)
        stats.total_crops = len(all_records)
        logger.info("清晰度过滤后剩余 %d 个 crop（丢弃 %d 个模糊 crop）。",
                    len(all_records), len(blurry))

    # ---- 打印类别分布 ----
    print_class_distribution(all_records)

    # ---- 打印统计 ----
    stats.total_crops = len(all_records)
    stats.print()

    # ---- 保存清单 ----
    exporter.save_manifest(all_records)

    # ---- 生成 HTML 预览（可选）----
    if args.html_preview:
        exporter.generate_html_preview(all_records)

    logger.info("✅ Crop 导出完成！输出目录: %s", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
