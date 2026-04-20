"""
modules/detector.py - 螺丝检测器模块
Owner: B（检测数据与 detector）

职责：
  - 加载训练好的 YOLO 螺丝检测器（当前以 5 类 detector 为主）
  - 支持可选 SAHI（Slicing Aided Hyper Inference）切片推理
  - 对单帧或批量帧进行推理，返回 List[Detection]
  - 兜底：若模型权重不存在，自动切换到基于 OpenCV 的简单检测器

TODO (B)：
  [ ] 继续在真实视频关键帧上校验 5 类 detector 的召回和误检
  [ ] 视速度预算决定是否重新启用 SAHI
  [ ] 验证 FP16 推理在目标 GPU 上是否正常
  [ ] 在 3 段开发视频上测速并记录到 benchmark 报告

依赖：ultralytics, sahi（可选）, opencv-python, numpy
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import torch
except ImportError:
    torch = None  # type: ignore[misc, assignment]

# 将项目根目录加入 sys.path（确保 interfaces 可导入）
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from interfaces import Detection

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 超参数（B 负责调优）
# ---------------------------------------------------------------------------

CONF_THRESHOLD: float = 0.10    # 候选框收集阈值；最终保留由分类别阈值再过滤
IOU_THRESHOLD: float = 0.60     # 与离线评测一致的 YOLO NMS IoU 阈值
IMG_SIZE: int = 1280            # 与 v1 + per-class conf 离线评测一致的推理尺寸
USE_FP16: bool = True           # 是否使用 FP16 半精度推理（需要 GPU）
USE_SAHI: bool = False          # 默认关闭，优先对齐离线最佳直接推理工作点
INFER_BATCH_MAX: int = 16       # predict(list) 时的 batch 上限（与逐帧同超参）
SAHI_SLICE_H: int = 1280        # SAHI 切片高度
SAHI_SLICE_W: int = 1280        # SAHI 切片宽度
SAHI_OVERLAP: float = 0.20      # SAHI 切片重叠比例

# 模型权重路径（相对于项目根目录）
DEFAULT_WEIGHTS = Path(__file__).parent.parent / "models" / "detector.pt"
DEFAULT_CLASS_CONF_JSON = Path(__file__).parent.parent / "configs" / "b_detector_thresholds.json"


# ---------------------------------------------------------------------------
# 辅助工具
# ---------------------------------------------------------------------------

def _bbox_crop(frame: np.ndarray, bbox: np.ndarray, pad_ratio: float = 0.05) -> np.ndarray:
    """
    从帧中按 bbox 裁切 crop，并额外扩展 pad_ratio 比例的边距。

    Parameters
    ----------
    frame : np.ndarray
        原始帧（H×W×3 BGR）。
    bbox : np.ndarray
        [x1, y1, x2, y2] 格式，像素坐标。
    pad_ratio : float
        相对于 bbox 短边的扩展比例。

    Returns
    -------
    np.ndarray : 裁切后的图像区域（uint8 BGR）。
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox[:4].astype(float)

    bw = x2 - x1
    bh = y2 - y1
    pad = min(bw, bh) * pad_ratio

    x1 = int(max(0, x1 - pad))
    y1 = int(max(0, y1 - pad))
    x2 = int(min(w, x2 + pad))
    y2 = int(min(h, y2 + pad))

    if x2 <= x1 or y2 <= y1:
        return np.zeros((8, 8, 3), dtype=np.uint8)

    return frame[y1:y2, x1:x2].copy()


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> List[int]:
    """
    手动实现 Non-Maximum Suppression（仅在 SAHI 兜底场景使用）。

    Parameters
    ----------
    boxes : np.ndarray
        (N, 4)，[x1, y1, x2, y2]。
    scores : np.ndarray
        (N,) 置信度。
    iou_thresh : float
        IoU 阈值。

    Returns
    -------
    List[int] : 保留的框的索引。
    """
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1 + 1) * np.maximum(0, yy2 - yy1 + 1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]

    return keep


def _normalize_class_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


# ---------------------------------------------------------------------------
# OpenCV 兜底检测器（B 不需要修改，仅在 YOLO 权重缺失时使用）
# ---------------------------------------------------------------------------

class _FallbackDetector:
    """
    基于 OpenCV Hough 圆检测 + 轮廓分析的兜底检测器。

    精度较低，仅用于：
    1. 首次运行时 YOLO 权重尚未训练完成的场景
    2. 快速验证 run.py 流程的端到端连通性

    B 无需修改此类，专注于实现 YOLODetector。
    """

    def __init__(
        self,
        conf_threshold: float = CONF_THRESHOLD,
        emit_warning: bool = True,
    ) -> None:
        self.conf_threshold = conf_threshold
        if emit_warning:
            logger.warning(
                "⚠️  使用 OpenCV 兜底检测器（Hough + 轮廓）。"
                "精度较低，请尽快提供 models/detector.pt。"
            )

    def detect(self, frame: np.ndarray, frame_id: int) -> List[Detection]:
        """
        对单帧进行检测，返回 Detection 列表。

        Parameters
        ----------
        frame : np.ndarray
            输入帧（H×W×3 BGR），原始分辨率。
        frame_id : int
            帧编号。

        Returns
        -------
        List[Detection]
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ---- 方法 1：自适应阈值 + 轮廓 ----
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=21, C=8,
        )
        # 形态学闭运算去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections: List[Detection] = []
        h, w = frame.shape[:2]
        min_area = (min(h, w) * 0.01) ** 2   # 最小面积：图像短边的 1%²
        max_area = (min(h, w) * 0.25) ** 2   # 最大面积：图像短边的 25%²

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            # 圆形度过滤（螺丝大致为圆形）
            perimeter = cv2.arcLength(cnt, True)
            if perimeter < 1e-6:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < 0.35:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            bbox = np.array([x, y, x + bw, y + bh], dtype=np.float32)
            crop = _bbox_crop(frame, bbox)

            # 伪置信度：用圆形度作为置信度（仅用于排序，无绝对意义）
            conf = float(min(circularity, 1.0))

            detections.append(Detection(
                frame_id=frame_id,
                bbox=bbox,
                confidence=conf,
                crop=crop,
            ))

        # 对兜底结果做 NMS，防止同一螺丝被重复检测
        if detections:
            boxes = np.array([d.bbox for d in detections])
            scores = np.array([d.confidence for d in detections])
            keep = _nms(boxes, scores, IOU_THRESHOLD)
            detections = [detections[i] for i in keep]

        logger.debug(
            "[兜底检测] frame=%d  检测到 %d 个螺丝",
            frame_id, len(detections),
        )
        return detections


# ---------------------------------------------------------------------------
# YOLO 检测器（B 负责实现）
# ---------------------------------------------------------------------------

class YOLODetector:
    """
    基于 Ultralytics YOLO 的螺丝检测器。

    TODO (B)：
    1. 确保 models/detector.pt 与 submission 默认配置一致
    2. 根据实际 GPU 显存调整 IMG_SIZE 和 SAHI 参数
    3. 在 3 段开发视频上验证召回率 ≥ 90%
    4. 启用 model.track() 以获得 ByteTrack track_id（用于 A 的短时关联）

    Parameters
    ----------
    weights_path : str | Path
        YOLO 权重文件路径（.pt 格式）。
    conf_threshold : float
        检测置信度阈值。
    iou_threshold : float
        NMS IoU 阈值。
    use_fp16 : bool
        是否使用 FP16 推理（需要 CUDA GPU）。
    use_sahi : bool
        是否启用 SAHI 切片推理。
    device : str
        推理设备：'cuda:0' / 'cpu' / '' (自动选择)。
    """

    def __init__(
        self,
        weights_path: str | Path = DEFAULT_WEIGHTS,
        class_conf_json_path: Optional[str | Path] = DEFAULT_CLASS_CONF_JSON,
        conf_threshold: float = CONF_THRESHOLD,
        iou_threshold: float = IOU_THRESHOLD,
        use_fp16: bool = USE_FP16,
        use_sahi: bool = USE_SAHI,
        device: str = "",
    ) -> None:
        self.weights_path = Path(weights_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.use_fp16 = use_fp16
        self.use_sahi = use_sahi
        self.device = device
        self.class_conf_json_path = Path(class_conf_json_path) if class_conf_json_path else None

        self._model = None
        self._sahi_model = None
        self._loaded = False
        self._model_names: Dict[int, str] = {}
        self._class_conf_map_norm: Dict[str, float] = {}

        self._load_model()

    def _load_class_conf_map(self) -> None:
        self._class_conf_map_norm = {}
        if self.class_conf_json_path is None:
            return
        if not self.class_conf_json_path.exists():
            logger.info("类别阈值配置不存在，使用统一 conf_threshold=%.2f", self.conf_threshold)
            return
        try:
            payload = json.loads(self.class_conf_json_path.read_text(encoding="utf-8"))
            raw_map = payload.get("class_conf_thresholds", payload)
            if not isinstance(raw_map, dict):
                logger.warning("类别阈值配置格式错误，忽略: %s", self.class_conf_json_path)
                return
            loaded: Dict[str, float] = {}
            for k, v in raw_map.items():
                if not isinstance(k, str):
                    continue
                try:
                    loaded[_normalize_class_name(k)] = float(v)
                except (TypeError, ValueError):
                    continue
            if not loaded:
                logger.warning("类别阈值配置为空，忽略: %s", self.class_conf_json_path)
                return
            self._class_conf_map_norm = loaded
        except Exception as e:
            logger.warning("读取类别阈值配置失败，忽略 %s: %s", self.class_conf_json_path, e)

    def _effective_predict_conf(self) -> float:
        """
        返回送入 YOLO 的候选框收集阈值。

        离线最佳配置使用 `imgsz=1280 + min_conf=0.10` 收集候选框，
        然后再用分类别阈值二次过滤；这里保持同样的调用方式。
        """
        if not self._class_conf_map_norm:
            return self.conf_threshold
        return min(self.conf_threshold, min(self._class_conf_map_norm.values()))

    def _validate_class_conf_map(self) -> None:
        if not self._class_conf_map_norm:
            return
        model_keys = {_normalize_class_name(v) for v in self._model_names.values()}
        matched = model_keys & set(self._class_conf_map_norm.keys())
        if not matched:
            logger.warning(
                "类别阈值配置与模型类别无交集，自动禁用配置: %s",
                self.class_conf_json_path,
            )
            self._class_conf_map_norm = {}
            return
        logger.info(
            "已加载类别阈值配置（匹配类别 %d 个）: %s",
            len(matched),
            self.class_conf_json_path,
        )

    def _pass_class_conf(self, class_name: str, conf: float) -> bool:
        if not self._class_conf_map_norm:
            return conf >= self.conf_threshold
        key = _normalize_class_name(class_name)
        thr = self._class_conf_map_norm.get(key)
        if thr is None:
            # 若配置存在，默认仅保留显式配置过的类别。
            return False
        return conf >= thr

    def _load_model(self) -> None:
        """
        加载 YOLO 模型。若权重文件不存在则跳过（由 Detector 门面类兜底）。

        TODO (B)：
        - 将训练好的权重保存至 models/detector.pt
        - 若需要 SAHI，额外调用 _load_sahi_model()
        """
        if not self.weights_path.exists():
            logger.warning(
                "YOLO 权重文件不存在: %s\n"
                "请将训练好的 detector.pt 放至此路径。",
                self.weights_path,
            )
            return

        try:
            from ultralytics import YOLO
            self._model = YOLO(str(self.weights_path))
            raw_names = self._model.names
            if isinstance(raw_names, dict):
                self._model_names = {int(k): str(v) for k, v in raw_names.items()}
            else:
                self._model_names = {i: str(n) for i, n in enumerate(raw_names)}

            self._load_class_conf_map()
            self._validate_class_conf_map()

            # 推理预热（避免首帧耗时过长）
            dummy = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            if torch is not None:
                with torch.inference_mode():
                    self._model.predict(
                        dummy,
                        conf=self._effective_predict_conf(),
                        iou=self.iou_threshold,
                        imgsz=IMG_SIZE,
                        half=self.use_fp16,
                        device=self.device or None,
                        verbose=False,
                    )
            else:
                self._model.predict(
                    dummy,
                    conf=self._effective_predict_conf(),
                    iou=self.iou_threshold,
                    imgsz=IMG_SIZE,
                    half=self.use_fp16,
                    device=self.device or None,
                    verbose=False,
                )

            logger.info("✅ YOLO 模型加载完成: %s", self.weights_path)
            logger.info(
                "YOLO detector 默认工作点: imgsz=%d, collect_conf=%.2f, iou=%.2f, use_sahi=%s",
                IMG_SIZE,
                self._effective_predict_conf(),
                self.iou_threshold,
                self.use_sahi,
            )
            self._loaded = True

            # SAHI 初始化
            if self.use_sahi:
                self._load_sahi_model()

        except ImportError:
            logger.error("ultralytics 未安装，请运行: pip install ultralytics")
        except Exception as e:
            logger.error("YOLO 模型加载失败: %s", e)

    def _load_sahi_model(self) -> None:
        """
        初始化 SAHI 推理器（用于高分辨率 4K 视频的切片检测）。

        TODO (B)：
        - 调整 SAHI_SLICE_H / SAHI_SLICE_W / SAHI_OVERLAP 至最优
        - 在 4K 帧上验证小目标召回率的提升效果
        """
        try:
            from sahi import AutoDetectionModel
            self._sahi_model = AutoDetectionModel.from_pretrained(
                model_type="yolov8",
                model_path=str(self.weights_path),
                confidence_threshold=self._effective_predict_conf(),
                device=self.device or "cpu",
            )
            logger.info("✅ SAHI 模型加载完成（切片: %dx%d, 重叠: %.0f%%）",
                        SAHI_SLICE_H, SAHI_SLICE_W, SAHI_OVERLAP * 100)
        except ImportError:
            logger.warning("sahi 未安装，SAHI 切片推理不可用。运行: pip install sahi")
            self._sahi_model = None
        except Exception as e:
            logger.warning("SAHI 初始化失败: %s", e)
            self._sahi_model = None

    # ------------------------------------------------------------------
    # 单帧检测（主接口）
    # ------------------------------------------------------------------

    def detect(
        self,
        frame: np.ndarray,
        frame_id: int,
        enable_tracking: bool = False,
    ) -> List[Detection]:
        """
        对单帧进行 YOLO 推理，返回 Detection 列表。

        Parameters
        ----------
        frame : np.ndarray
            输入帧（H×W×3 BGR），原始分辨率。
        frame_id : int
            帧编号（0-indexed）。
        enable_tracking : bool
            若为 True，调用 model.track() 以获取 ByteTrack track_id。
            （通常仅在连续帧推理时启用）

        Returns
        -------
        List[Detection]

        TODO (B)：
        - 验证 enable_tracking=True 时 track_id 是否正确填充
        - 在 SAHI 模式下 track_id 暂不可用（SAHI 不支持跟踪），此时 track_id=-1
        """
        if not self._loaded or self._model is None:
            return []

        try:
            # 高分辨率帧优先使用 SAHI
            h, w = frame.shape[:2]
            long_edge = max(h, w)
            if self.use_sahi and self._sahi_model is not None and long_edge > 1280:
                return self._detect_with_sahi(frame, frame_id)
            else:
                return self._detect_direct(frame, frame_id, enable_tracking)

        except Exception as e:
            logger.error("检测推理出错 (frame=%d): %s", frame_id, e)
            return []

    def _parse_single_result(
        self,
        r,
        frame: np.ndarray,
        frame_id: int,
        enable_tracking: bool = False,
        generate_masks: bool = True,
    ) -> List[Detection]:
        """解析单帧 YOLO Result，与原先 _detect_direct 内逻辑一致。

        Parameters
        ----------
        generate_masks : bool
            是否提取并保存实例分割掩膜（seg_mask）。
            对于仅用于去重聚类的关键帧检测，可设为 False 以节省内存和 CPU。
        """
        detections: List[Detection] = []
        if r is None or r.boxes is None:
            return detections

        boxes_xyxy = r.boxes.xyxy.cpu().numpy()
        confidences = r.boxes.conf.cpu().numpy()
        class_ids = (
            r.boxes.cls.cpu().numpy().astype(int)
            if r.boxes.cls is not None
            else np.zeros(len(boxes_xyxy), dtype=int)
        )
        # 仅在需要时提取分割掩膜，避免对不需要可视化的帧（关键帧）做高分辨率 resize
        seg_masks = (
            self._extract_seg_masks(r, frame.shape[:2])
            if generate_masks
            else [None] * len(boxes_xyxy)
        )
        track_ids_arr = (
            r.boxes.id.cpu().numpy().astype(int)
            if (enable_tracking and r.boxes.id is not None)
            else np.full(len(boxes_xyxy), -1, dtype=int)
        )

        for i, (bbox, conf, tid, cid) in enumerate(
            zip(boxes_xyxy, confidences, track_ids_arr, class_ids)
        ):
            cls_name = self._model_names.get(int(cid), str(int(cid)))
            if not self._pass_class_conf(cls_name, float(conf)):
                continue
            crop = _bbox_crop(frame, bbox)
            sm = seg_masks[i] if i < len(seg_masks) else None
            detections.append(Detection(
                frame_id=frame_id,
                bbox=bbox.astype(np.float32),
                confidence=float(conf),
                crop=crop,
                track_id=int(tid),
                class_id=int(cid),
                class_name=cls_name,
                seg_mask=sm,
            ))

        return detections

    def _detect_direct(
        self,
        frame: np.ndarray,
        frame_id: int,
        enable_tracking: bool,
    ) -> List[Detection]:
        """直接对整帧进行 YOLO 推理（适合 1080p 及以下）。"""
        kwargs = dict(
            conf=self._effective_predict_conf(),
            iou=self.iou_threshold,
            imgsz=IMG_SIZE,
            half=self.use_fp16,
            device=self.device or None,
            verbose=False,
        )

        cm = torch.inference_mode if torch is not None else None
        if enable_tracking:
            if cm is not None:
                with cm():
                    results = self._model.track(frame, persist=True, **kwargs)
            else:
                results = self._model.track(frame, persist=True, **kwargs)
        else:
            if cm is not None:
                with cm():
                    results = self._model.predict(frame, **kwargs)
            else:
                results = self._model.predict(frame, **kwargs)

        if not results:
            return []

        detections = self._parse_single_result(
            results[0], frame, frame_id, enable_tracking,
        )

        logger.debug(
            "[YOLO 直接推理] frame=%d  检测到 %d 个螺丝",
            frame_id, len(detections),
        )
        return detections

    @staticmethod
    def _extract_seg_masks(result, frame_shape: Tuple[int, int]) -> List[Optional[np.ndarray]]:
        """
        从 Ultralytics 单帧结果中提取并对齐实例分割掩膜。

        返回与 boxes 一一对应的 mask 列表；若该实例无 mask 则为 None。
        """
        if result is None or result.boxes is None:
            return []

        n = len(result.boxes)
        masks: List[Optional[np.ndarray]] = [None] * n
        masks_obj = getattr(result, "masks", None)
        if masks_obj is None or getattr(masks_obj, "data", None) is None:
            return masks

        try:
            raw = masks_obj.data.cpu().numpy()  # (N, Hm, Wm)
        except Exception:
            return masks

        if raw.ndim != 3:
            return masks

        h, w = frame_shape
        count = min(n, raw.shape[0])
        for i in range(count):
            m = (raw[i] > 0.5).astype(np.uint8)
            if m.shape != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                m = (m > 0).astype(np.uint8)
            masks[i] = m
        return masks

    def _detect_with_sahi(
        self,
        frame: np.ndarray,
        frame_id: int,
    ) -> List[Detection]:
        """
        使用 SAHI 切片推理对高分辨率帧进行检测。

        注意：SAHI 推理结果不包含 track_id，track_id 统一为 -1。

        TODO (B)：
        - 调整 postprocess_type 和 postprocess_match_threshold
        - 测试 postprocess_type="GREEDYNMM" vs "NMS" 的效果差异
        """
        try:
            from sahi.predict import get_sliced_prediction
            import PIL.Image

            # OpenCV BGR → RGB PIL Image
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = PIL.Image.fromarray(rgb)

            result = get_sliced_prediction(
                pil_img,
                self._sahi_model,
                slice_height=SAHI_SLICE_H,
                slice_width=SAHI_SLICE_W,
                overlap_height_ratio=SAHI_OVERLAP,
                overlap_width_ratio=SAHI_OVERLAP,
                postprocess_type="GREEDYNMM",
                postprocess_match_metric="IOU",
                postprocess_match_threshold=self.iou_threshold,
                verbose=0,
            )

            detections: List[Detection] = []
            for obj_pred in result.object_prediction_list:
                bbox_sahi = obj_pred.bbox
                bbox = np.array([
                    bbox_sahi.minx, bbox_sahi.miny,
                    bbox_sahi.maxx, bbox_sahi.maxy,
                ], dtype=np.float32)
                conf = float(obj_pred.score.value)
                class_name = ""
                cat = getattr(obj_pred, "category", None)
                if cat is not None and hasattr(cat, "name") and cat.name is not None:
                    class_name = str(cat.name)
                if not class_name:
                    class_name = "0"
                if not self._pass_class_conf(class_name, conf):
                    continue
                crop = _bbox_crop(frame, bbox)
                detections.append(Detection(
                    frame_id=frame_id,
                    bbox=bbox,
                    confidence=conf,
                    crop=crop,
                    track_id=-1,  # SAHI 不支持跟踪
                    class_id=-1,
                    class_name=class_name,
                ))

            logger.debug(
                "[SAHI 推理] frame=%d  检测到 %d 个螺丝",
                frame_id, len(detections),
            )
            return detections

        except Exception as e:
            logger.warning("SAHI 推理出错，回退到直接推理: %s", e)
            return self._detect_direct(frame, frame_id, enable_tracking=False)

    # ------------------------------------------------------------------
    # 批量推理（效率优化，B 可选实现）
    # ------------------------------------------------------------------

    def detect_batch(
        self,
        frames: List[np.ndarray],
        frame_ids: List[int],
        generate_masks: bool = True,
        per_frame_masks: Optional[List[bool]] = None,
    ) -> List[List[Detection]]:
        """
        批量对多帧推理（batch inference），效率高于逐帧调用 detect()。

        TODO (B)：
        - 实现真正的 batch 推理（将多帧一次送入 YOLO）
        - 注意：SAHI 模式下 batch 推理不可用，需退化到逐帧

        Parameters
        ----------
        frames : List[np.ndarray]
            帧列表，每帧 (H×W×3 BGR)。
        frame_ids : List[int]
            对应的帧编号列表。
        generate_masks : bool
            全局 seg_mask 开关。per_frame_masks 未设置时对所有帧生效。
        per_frame_masks : List[bool] | None
            逐帧 seg_mask 开关（长度需与 frames 一致）。若提供，则覆盖
            generate_masks，允许仅为特定帧（如中间帧）生成 seg_mask，
            跳过关键帧的高分辨率 seg_mask resize，显著节省 CPU 时间。

        Returns
        -------
        List[List[Detection]] : 每帧对应的 Detection 列表。
        """
        if not self._loaded or self._model is None:
            return [[] for _ in frames]

        if self.use_sahi and self._sahi_model is not None:
            return [
                self.detect(f, fid, enable_tracking=False)
                for f, fid in zip(frames, frame_ids)
            ]

        try:
            bs = min(INFER_BATCH_MAX, len(frames))
            kwargs = dict(
                conf=self._effective_predict_conf(),
                iou=self.iou_threshold,
                imgsz=IMG_SIZE,
                half=self.use_fp16,
                device=self.device or None,
                verbose=False,
                batch=bs,
            )
            cm = torch.inference_mode if torch is not None else None
            if cm is not None:
                with cm():
                    all_results = self._model.predict(frames, **kwargs)
            else:
                all_results = self._model.predict(frames, **kwargs)
            out: List[List[Detection]] = []
            for idx, (r, fid, frame) in enumerate(zip(all_results, frame_ids, frames)):
                if per_frame_masks is not None and idx < len(per_frame_masks):
                    gen_mask = per_frame_masks[idx]
                else:
                    gen_mask = generate_masks
                out.append(
                    self._parse_single_result(
                        r, frame, fid,
                        enable_tracking=False,
                        generate_masks=gen_mask,
                    ),
                )
            logger.debug(
                "[YOLO 批量推理] %d 帧 batch=%d imgsz=%d",
                len(frames), bs, IMG_SIZE,
            )
            return out
        except Exception as e:
            logger.warning("批量推理失败，回退逐帧: %s", e)
            return [
                self.detect(f, fid, enable_tracking=False)
                for f, fid in zip(frames, frame_ids)
            ]


# ---------------------------------------------------------------------------
# 门面类（统一入口，D 在 pipeline.py 中调用此类）
# ---------------------------------------------------------------------------

class Detector:
    """
    检测器门面类（Facade Pattern）。

    自动选择最优的检测后端：
    - 优先使用 YOLODetector（若权重存在）
    - 权重不存在时自动回退到 _FallbackDetector

    D 在 pipeline.py 中应只调用此类，不直接使用 YOLODetector 或 _FallbackDetector。

    用法
    ----
    >>> detector = Detector()
    >>> detections = detector.detect(frame, frame_id=0)
    >>> batch_results = detector.detect_batch(frames, frame_ids)
    """

    def __init__(
        self,
        weights_path: str | Path = DEFAULT_WEIGHTS,
        class_conf_json_path: Optional[str | Path] = DEFAULT_CLASS_CONF_JSON,
        conf_threshold: float = CONF_THRESHOLD,
        iou_threshold: float = IOU_THRESHOLD,
        use_fp16: bool = USE_FP16,
        use_sahi: bool = USE_SAHI,
        device: str = "",
    ) -> None:
        self._yolo = YOLODetector(
            weights_path=weights_path,
            class_conf_json_path=class_conf_json_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            use_fp16=use_fp16,
            use_sahi=use_sahi,
            device=device,
        )
        # 若 YOLO 加载失败，使用兜底检测器
        self._fallback = _FallbackDetector(
            conf_threshold=conf_threshold,
            emit_warning=False,
        )
        self._use_fallback = not self._yolo._loaded

        if self._use_fallback:
            logger.warning(
                "⚠️  使用 OpenCV 兜底检测器（Hough + 轮廓）。"
                "精度较低，请尽快提供 models/detector.pt。"
            )
            logger.warning(
                "Detector 已切换到兜底模式（OpenCV）。"
                "最终提交前请确保 YOLO 模型可用！"
            )
        else:
            logger.info("Detector 初始化完成（YOLO 模式）。")

    @property
    def is_yolo_mode(self) -> bool:
        """是否处于 YOLO 推理模式（否则为兜底模式）。"""
        return not self._use_fallback

    def detect(
        self,
        frame: np.ndarray,
        frame_id: int,
        enable_tracking: bool = False,
    ) -> List[Detection]:
        """
        对单帧进行检测，自动选择后端。

        Parameters
        ----------
        frame : np.ndarray
            输入帧（H×W×3 BGR），原始分辨率。
        frame_id : int
            帧编号（0-indexed）。
        enable_tracking : bool
            是否启用 ByteTrack 跟踪（仅 YOLO 模式有效）。

        Returns
        -------
        List[Detection]
        """
        if self._use_fallback:
            return self._fallback.detect(frame, frame_id)
        return self._yolo.detect(frame, frame_id, enable_tracking=enable_tracking)

    def detect_batch(
        self,
        frames: List[np.ndarray],
        frame_ids: List[int],
        generate_masks: bool = True,
        per_frame_masks: Optional[List[bool]] = None,
    ) -> List[List[Detection]]:
        """
        批量对多帧检测，自动选择后端。

        Parameters
        ----------
        frames : List[np.ndarray]
            帧列表。
        frame_ids : List[int]
            对应帧编号列表。
        generate_masks : bool
            全局 seg_mask 开关（per_frame_masks 未设置时生效）。
        per_frame_masks : List[bool] | None
            逐帧 seg_mask 开关，允许仅为特定帧（如中间帧）生成掩膜。

        Returns
        -------
        List[List[Detection]]
        """
        if self._use_fallback:
            return [self._fallback.detect(f, fid) for f, fid in zip(frames, frame_ids)]
        return self._yolo.detect_batch(
            frames, frame_ids,
            generate_masks=generate_masks,
            per_frame_masks=per_frame_masks,
        )
