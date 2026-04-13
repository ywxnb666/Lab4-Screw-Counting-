"""
utils/visualizer.py - 掩膜叠加与可视化工具
Owner: D（工程封装）

职责：
  - 在帧图像上绘制检测框（bbox）和掩膜
  - 生成最终提交用的掩膜叠加图（mask_output_path/{video_name}_mask.png）
  - 提供调试用的可视化（cluster 中心、类别标签、轨迹等）

依赖：opencv-python, numpy
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from interfaces import Cluster, Detection

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 颜色方案：Type_1~Type_5，BGR 格式
# ---------------------------------------------------------------------------

# 每种螺丝类型对应的 BGR 颜色（醒目且互相区分）
CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (255,  80,  80),   # Type_1 - 蓝色
    1: ( 80, 200,  80),   # Type_2 - 绿色
    2: ( 80,  80, 255),   # Type_3 - 红色
    3: (255, 200,  80),   # Type_4 - 青色
    4: (200,  80, 255),   # Type_5 - 紫色
}

# 未分类（pred_class=-1）使用白色
UNKNOWN_COLOR: Tuple[int, int, int] = (220, 220, 220)

# 类别标签文本
CLASS_LABELS: Dict[int, str] = {
    0: "Type_1",
    1: "Type_2",
    2: "Type_3",
    3: "Type_4",
    4: "Type_5",
}


def _get_color(pred_class: int) -> Tuple[int, int, int]:
    """根据预测类别返回对应颜色。"""
    return CLASS_COLORS.get(pred_class, UNKNOWN_COLOR)


# ---------------------------------------------------------------------------
# 基础绘制函数
# ---------------------------------------------------------------------------

def draw_bbox(
    canvas: np.ndarray,
    bbox: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    label: Optional[str] = None,
    label_bg: bool = True,
    font_scale: float = 0.5,
) -> np.ndarray:
    """
    在 canvas 上绘制一个 bbox 矩形及可选标签。

    Parameters
    ----------
    canvas : np.ndarray
        目标图像（会被原地修改后返回）。
    bbox : np.ndarray
        [x1, y1, x2, y2] 格式的检测框（像素坐标）。
    color : Tuple[int, int, int]
        BGR 颜色。
    thickness : int
        线宽。
    label : str | None
        要在框左上角显示的文本，None 则不显示。
    label_bg : bool
        是否绘制标签背景矩形（提高可读性）。
    font_scale : float
        字体大小。

    Returns
    -------
    np.ndarray : 绘制后的图像（与输入为同一对象）。
    """
    x1, y1, x2, y2 = (int(round(v)) for v in bbox[:4])
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, 1)
        tx = x1
        ty = max(y1 - 4, th + 4)

        if label_bg:
            cv2.rectangle(
                canvas,
                (tx, ty - th - baseline),
                (tx + tw, ty + baseline),
                color,
                cv2.FILLED,
            )
            # 根据背景亮度决定文字颜色（黑/白）
            lum = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
            text_color = (0, 0, 0) if lum > 128 else (255, 255, 255)
        else:
            text_color = color

        cv2.putText(canvas, label, (tx, ty), font, font_scale, text_color, 1, cv2.LINE_AA)

    return canvas


def draw_filled_bbox(
    canvas: np.ndarray,
    bbox: np.ndarray,
    color: Tuple[int, int, int],
    alpha: float = 0.4,
) -> np.ndarray:
    """
    在 canvas 上绘制半透明填充矩形（用于掩膜效果）。

    Parameters
    ----------
    canvas : np.ndarray
        目标图像（原地修改）。
    bbox : np.ndarray
        [x1, y1, x2, y2]。
    color : Tuple[int, int, int]
        BGR 填充颜色。
    alpha : float
        填充透明度（0=完全透明，1=不透明），默认 0.4。

    Returns
    -------
    np.ndarray : 绘制后的图像。
    """
    x1, y1, x2, y2 = (int(round(v)) for v in bbox[:4])
    h, w = canvas.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return canvas

    overlay = canvas.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, cv2.FILLED)
    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)
    return canvas


def draw_circle_mask(
    canvas: np.ndarray,
    center: np.ndarray,
    radius: int,
    color: Tuple[int, int, int],
    alpha: float = 0.45,
) -> np.ndarray:
    """
    在 canvas 上绘制半透明实心圆（适合螺丝这类圆形零件）。

    Parameters
    ----------
    canvas : np.ndarray
        目标图像（原地修改）。
    center : np.ndarray
        圆心坐标 [cx, cy]。
    radius : int
        半径（像素）。
    color : Tuple[int, int, int]
        BGR 颜色。
    alpha : float
        透明度（默认 0.45）。

    Returns
    -------
    np.ndarray : 绘制后的图像。
    """
    cx, cy = int(round(center[0])), int(round(center[1]))
    overlay = canvas.copy()
    cv2.circle(overlay, (cx, cy), radius, color, cv2.FILLED)
    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)
    # 再画一圈实线边框提高清晰度
    cv2.circle(canvas, (cx, cy), radius, color, 2, cv2.LINE_AA)
    return canvas


# ---------------------------------------------------------------------------
# 主可视化类
# ---------------------------------------------------------------------------

class Visualizer:
    """
    提供掩膜叠加与可视化功能的工具类。

    典型用法
    --------
    >>> vis = Visualizer()
    >>> mask_img = vis.draw_clusters(frame, clusters)
    >>> vis.save_mask(mask_img, output_dir, video_name)
    """

    def __init__(
        self,
        mask_alpha: float = 0.40,
        bbox_thickness: int = 2,
        font_scale: float = 0.55,
        show_label: bool = True,
        show_confidence: bool = False,
        use_circle_mask: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        mask_alpha : float
            掩膜半透明度（默认 0.40）。
        bbox_thickness : int
            检测框线宽。
        font_scale : float
            标签字体大小。
        show_label : bool
            是否在框上显示类别标签。
        show_confidence : bool
            是否在标签中包含置信度数值。
        use_circle_mask : bool
            True: 用圆形掩膜（更适合螺丝）；False: 用矩形掩膜。
        """
        self.mask_alpha = mask_alpha
        self.bbox_thickness = bbox_thickness
        self.font_scale = font_scale
        self.show_label = show_label
        self.show_confidence = show_confidence
        self.use_circle_mask = use_circle_mask

    # ------------------------------------------------------------------
    # Cluster 级可视化（最终提交用）
    # ------------------------------------------------------------------

    def draw_clusters(
        self,
        frame: np.ndarray,
        clusters: List[Cluster],
        draw_bbox: bool = True,
        draw_mask: bool = True,
        draw_id: bool = False,
    ) -> np.ndarray:
        """
        在帧上绘制所有去重后的 Cluster（最终提交用掩膜图）。

        按照作业要求：掩膜图像为「原图和掩膜叠加后的结果，清晰标注检测到的螺丝位置」。

        Parameters
        ----------
        frame : np.ndarray
            原始帧图像（H×W×3 BGR），不会被修改（内部复制）。
        clusters : List[Cluster]
            去重并分类后的螺丝实例列表。
        draw_bbox : bool
            是否绘制检测框边线。
        draw_mask : bool
            是否绘制半透明填充掩膜。
        draw_id : bool
            是否在框上显示 cluster_id（调试用）。

        Returns
        -------
        np.ndarray : 绘制完成的掩膜叠加图，与输入 frame 等大。
        """
        canvas = frame.copy()

        for cluster in clusters:
            color = _get_color(cluster.pred_class)

            # 优先使用 ref_bbox，其次从 observations 中取最新一个
            bbox = cluster.ref_bbox
            if bbox is None and cluster.observations:
                bbox = cluster.observations[-1].bbox

            if bbox is None:
                continue

            # 掩膜填充
            if draw_mask:
                if self.use_circle_mask:
                    cx = (bbox[0] + bbox[2]) / 2.0
                    cy = (bbox[1] + bbox[3]) / 2.0
                    radius = int(round(max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2.0))
                    draw_circle_mask(canvas, np.array([cx, cy]), radius, color, self.mask_alpha)
                else:
                    draw_filled_bbox(canvas, bbox, color, self.mask_alpha)

            # 边框
            if draw_bbox:
                label = None
                if self.show_label:
                    label = cluster.type_label
                    if draw_id:
                        label = f"#{cluster.cluster_id} {label}"
                    if self.show_confidence:
                        label += f" {cluster.class_probs.max():.2f}"
                draw_bbox_fn = draw_bbox if callable(draw_bbox) else _draw_bbox_simple
                _draw_bbox_simple(canvas, bbox, color, self.bbox_thickness, label, self.font_scale)

        # 在右下角绘制图例
        self._draw_legend(canvas, clusters)

        return canvas

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        color: Tuple[int, int, int] = (0, 255, 128),
    ) -> np.ndarray:
        """
        在帧上绘制原始检测框（调试用，未去重）。

        Parameters
        ----------
        frame : np.ndarray
            原始帧。
        detections : List[Detection]
            该帧的检测结果。
        color : Tuple[int, int, int]
            框颜色（默认绿色）。

        Returns
        -------
        np.ndarray : 绘制结果（复制）。
        """
        canvas = frame.copy()
        for det in detections:
            label = f"{det.confidence:.2f}" if self.show_confidence else None
            if det.track_id >= 0:
                label = f"T{det.track_id}" + (f" {det.confidence:.2f}" if self.show_confidence else "")
            _draw_bbox_simple(canvas, det.bbox, color, self.bbox_thickness, label, self.font_scale)
        return canvas

    # ------------------------------------------------------------------
    # 纯掩膜图（二值或彩色，用于调试）
    # ------------------------------------------------------------------

    def make_binary_mask(
        self,
        frame_shape: Tuple[int, int],
        clusters: List[Cluster],
    ) -> np.ndarray:
        """
        生成二值掩膜图（白色=螺丝区域，黑色=背景）。

        Parameters
        ----------
        frame_shape : Tuple[int, int]
            (H, W) 掩膜图大小。
        clusters : List[Cluster]
            螺丝实例列表。

        Returns
        -------
        np.ndarray : H×W 的 uint8 二值图（0 或 255）。
        """
        h, w = frame_shape
        mask = np.zeros((h, w), dtype=np.uint8)
        for cluster in clusters:
            bbox = cluster.ref_bbox
            if bbox is None and cluster.observations:
                bbox = cluster.observations[-1].bbox
            if bbox is None:
                continue
            x1, y1, x2, y2 = (int(round(v)) for v in bbox[:4])
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            if self.use_circle_mask:
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                radius = max(1, max(x2 - x1, y2 - y1) // 2)
                cv2.circle(mask, (cx, cy), radius, 255, cv2.FILLED)
            else:
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, cv2.FILLED)
        return mask

    def make_color_mask(
        self,
        frame_shape: Tuple[int, int],
        clusters: List[Cluster],
    ) -> np.ndarray:
        """
        生成彩色掩膜图（不同类别使用不同颜色）。

        Parameters
        ----------
        frame_shape : Tuple[int, int]
            (H, W) 掩膜图大小。
        clusters : List[Cluster]
            螺丝实例列表。

        Returns
        -------
        np.ndarray : H×W×3 的 uint8 BGR 彩色掩膜图。
        """
        h, w = frame_shape
        mask = np.zeros((h, w, 3), dtype=np.uint8)
        for cluster in clusters:
            color = _get_color(cluster.pred_class)
            bbox = cluster.ref_bbox
            if bbox is None and cluster.observations:
                bbox = cluster.observations[-1].bbox
            if bbox is None:
                continue
            x1, y1, x2, y2 = (int(round(v)) for v in bbox[:4])
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            if self.use_circle_mask:
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                radius = max(1, max(x2 - x1, y2 - y1) // 2)
                cv2.circle(mask, (cx, cy), radius, color, cv2.FILLED)
            else:
                cv2.rectangle(mask, (x1, y1), (x2, y2), color, cv2.FILLED)
        return mask

    # ------------------------------------------------------------------
    # 图例
    # ------------------------------------------------------------------

    def _draw_legend(
        self,
        canvas: np.ndarray,
        clusters: List[Cluster],
    ) -> None:
        """
        在图像右下角绘制类别图例（各类型螺丝数量汇总）。
        """
        # 统计各类数量
        type_counts: Dict[int, int] = {}
        for c in clusters:
            k = c.pred_class
            type_counts[k] = type_counts.get(k, 0) + 1

        if not type_counts:
            return

        h, w = canvas.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = 0.45
        pad = 6
        line_h = 20
        box_w = 130
        box_h = len(type_counts) * line_h + pad * 2

        # 图例背景
        x0 = w - box_w - pad
        y0 = h - box_h - pad
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x0, y0), (w - pad, h - pad), (30, 30, 30), cv2.FILLED)
        cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0, canvas)

        # 逐行写类别+数量
        sorted_classes = sorted(type_counts.keys())
        for i, cls_id in enumerate(sorted_classes):
            color = _get_color(cls_id)
            label = CLASS_LABELS.get(cls_id, "Unknown")
            text = f"{label}: {type_counts[cls_id]}"
            yt = y0 + pad + (i + 1) * line_h - 4
            # 小色块
            cv2.rectangle(canvas, (x0 + pad, yt - 12), (x0 + pad + 12, yt), color, cv2.FILLED)
            cv2.putText(canvas, text, (x0 + pad + 16, yt), font, fs, (230, 230, 230), 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # 保存接口
    # ------------------------------------------------------------------

    def save_mask(
        self,
        mask_image: np.ndarray,
        output_dir: str | Path,
        video_name: str,
    ) -> Path:
        """
        将掩膜叠加图保存为 PNG 文件。

        文件命名格式：{video_name}_mask.png（符合作业要求）。

        Parameters
        ----------
        mask_image : np.ndarray
            已绘制完成的掩膜叠加图（H×W×3 BGR）。
        output_dir : str | Path
            输出文件夹路径（不存在时自动创建）。
        video_name : str
            视频名称（不含后缀），用于构建文件名。

        Returns
        -------
        Path : 保存的文件路径。
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        out_path = output_dir / f"{video_name}_mask.png"
        success = cv2.imwrite(str(out_path), mask_image)
        if success:
            logger.info("掩膜图已保存: %s", out_path)
        else:
            logger.error("掩膜图保存失败: %s", out_path)
        return out_path

    @staticmethod
    def make_side_by_side(
        original: np.ndarray,
        masked: np.ndarray,
        max_width: int = 1920,
    ) -> np.ndarray:
        """
        将原图与掩膜图左右拼接，方便对比（调试/报告用）。

        Parameters
        ----------
        original : np.ndarray
            原始帧（H×W×3）。
        masked : np.ndarray
            掩膜叠加图（H×W×3）。
        max_width : int
            拼接后的最大总宽度（超过则等比缩小）。

        Returns
        -------
        np.ndarray : 拼接后的图像。
        """
        h = max(original.shape[0], masked.shape[0])
        # 统一高度
        def _resize_h(img, target_h):
            scale = target_h / img.shape[0]
            return cv2.resize(img, (int(img.shape[1] * scale), target_h))

        orig_r = _resize_h(original, h)
        mask_r = _resize_h(masked, h)
        combined = np.concatenate([orig_r, mask_r], axis=1)

        total_w = combined.shape[1]
        if total_w > max_width:
            scale = max_width / total_w
            combined = cv2.resize(
                combined,
                (max_width, int(combined.shape[0] * scale)),
            )

        return combined

    @staticmethod
    def add_text_banner(
        canvas: np.ndarray,
        lines: List[str],
        position: str = "top-left",
        font_scale: float = 0.6,
        color: Tuple[int, int, int] = (255, 255, 255),
        bg_color: Tuple[int, int, int] = (30, 30, 30),
        bg_alpha: float = 0.6,
    ) -> np.ndarray:
        """
        在图像上添加多行文本横幅（用于报告截图标注）。

        Parameters
        ----------
        canvas : np.ndarray
            目标图像（原地修改）。
        lines : List[str]
            要显示的文本行列表。
        position : str
            文本位置：'top-left' | 'top-right' | 'bottom-left' | 'bottom-right'。
        font_scale : float
            字体大小。
        color : Tuple[int, int, int]
            文字颜色（BGR）。
        bg_color : Tuple[int, int, int]
            背景颜色（BGR）。
        bg_alpha : float
            背景透明度。

        Returns
        -------
        np.ndarray : 绘制后的图像。
        """
        if not lines:
            return canvas

        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1
        pad = 8
        line_h = int(font_scale * 30) + 4

        # 计算文本块尺寸
        max_tw = max(cv2.getTextSize(l, font, font_scale, thickness)[0][0] for l in lines)
        block_w = max_tw + pad * 2
        block_h = line_h * len(lines) + pad * 2

        h, w = canvas.shape[:2]
        if "right" in position:
            x0 = w - block_w - 4
        else:
            x0 = 4
        if "bottom" in position:
            y0 = h - block_h - 4
        else:
            y0 = 4

        # 背景
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + block_w, y0 + block_h), bg_color, cv2.FILLED)
        cv2.addWeighted(overlay, bg_alpha, canvas, 1 - bg_alpha, 0, canvas)

        # 文本
        for i, line in enumerate(lines):
            yt = y0 + pad + (i + 1) * line_h - 4
            cv2.putText(canvas, line, (x0 + pad, yt), font, font_scale, color, thickness, cv2.LINE_AA)

        return canvas


# ---------------------------------------------------------------------------
# 内部辅助（避免与模块级 draw_bbox 函数名冲突）
# ---------------------------------------------------------------------------

def _draw_bbox_simple(
    canvas: np.ndarray,
    bbox: np.ndarray,
    color: Tuple[int, int, int],
    thickness: int,
    label: Optional[str],
    font_scale: float,
) -> None:
    """内部调用的 bbox 绘制函数，直接修改 canvas。"""
    draw_bbox(canvas, bbox, color, thickness, label, label_bg=True, font_scale=font_scale)


# ---------------------------------------------------------------------------
# 便捷函数（无需实例化 Visualizer）
# ---------------------------------------------------------------------------

def quick_visualize(
    frame: np.ndarray,
    clusters: List[Cluster],
    output_path: Optional[str | Path] = None,
) -> np.ndarray:
    """
    快速生成并可选保存掩膜叠加图。

    Parameters
    ----------
    frame : np.ndarray
        原始帧（H×W×3 BGR）。
    clusters : List[Cluster]
        螺丝实例列表。
    output_path : str | Path | None
        若提供，将结果保存到此路径。

    Returns
    -------
    np.ndarray : 掩膜叠加图。
    """
    vis = Visualizer()
    result = vis.draw_clusters(frame, clusters)
    if output_path is not None:
        cv2.imwrite(str(output_path), result)
        logger.info("可视化结果已保存: %s", output_path)
    return result


def colorize_mask_for_display(binary_mask: np.ndarray) -> np.ndarray:
    """
    将二值掩膜转换为伪彩色图（绿色高亮），方便展示。

    Parameters
    ----------
    binary_mask : np.ndarray
        H×W 的 uint8 二值图（0 或 255）。

    Returns
    -------
    np.ndarray : H×W×3 的 BGR 图像。
    """
    colored = np.zeros((*binary_mask.shape, 3), dtype=np.uint8)
    colored[binary_mask > 0] = (80, 200, 80)
    return colored
