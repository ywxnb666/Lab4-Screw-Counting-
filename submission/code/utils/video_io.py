"""
utils/video_io.py - 视频读取与预处理工具
Owner: D（工程封装）

职责：
  - 封装 OpenCV VideoCapture，提供统一的视频读取接口
  - 处理手机视频的旋转元数据（rotation flag）
  - 支持按帧编号随机访问、按时间戳访问
  - 提供双分辨率输出（低分辨率用于配准，原始分辨率用于检测）

依赖：opencv-python, numpy
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Generator, Iterator, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# 视频后缀白名单
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv", ".flv", ".webm"}

# 低分辨率流的长边目标像素数（用于配准，不用于检测）
LOW_RES_LONG_EDGE = 960


# ---------------------------------------------------------------------------
# 元数据
# ---------------------------------------------------------------------------

class VideoMeta:
    """
    视频基础元数据容器。

    Attributes
    ----------
    path : Path
        视频文件路径。
    width : int
        帧宽度（经旋转校正后）。
    height : int
        帧高度（经旋转校正后）。
    fps : float
        帧率。
    frame_count : int
        总帧数（可能与实际不符，以实际读取为准）。
    duration : float
        时长（秒）。
    rotation : int
        原始旋转角度（0 / 90 / 180 / 270）。
    """

    def __init__(
        self,
        path: Path,
        raw_width: int,
        raw_height: int,
        fps: float,
        frame_count: int,
        rotation: int = 0,
    ) -> None:
        self.path = path
        self.fps = fps
        self.frame_count = frame_count
        self.rotation = rotation

        # 旋转 90° 或 270° 时宽高互换
        if rotation in (90, 270):
            self.width = raw_height
            self.height = raw_width
        else:
            self.width = raw_width
            self.height = raw_height

        self.duration = frame_count / fps if fps > 0 else 0.0

    @property
    def mid_frame_id(self) -> int:
        """视频中间帧的编号（整数）。"""
        return max(0, self.frame_count // 2)

    @property
    def low_res_scale(self) -> float:
        """长边缩放比例，使长边 = LOW_RES_LONG_EDGE。"""
        long_edge = max(self.width, self.height)
        if long_edge <= LOW_RES_LONG_EDGE:
            return 1.0
        return LOW_RES_LONG_EDGE / long_edge

    @property
    def low_res_size(self) -> Tuple[int, int]:
        """低分辨率尺寸 (width, height)，用于 cv2.resize。"""
        scale = self.low_res_scale
        return (
            max(1, int(round(self.width * scale))),
            max(1, int(round(self.height * scale))),
        )

    def __repr__(self) -> str:
        return (
            f"VideoMeta('{self.path.name}', "
            f"{self.width}x{self.height}, "
            f"{self.fps:.2f}fps, "
            f"{self.frame_count}frames, "
            f"{self.duration:.2f}s, "
            f"rot={self.rotation}°)"
        )


# ---------------------------------------------------------------------------
# 旋转工具
# ---------------------------------------------------------------------------

def _get_rotation_from_cap(cap: cv2.VideoCapture) -> int:
    """
    从 OpenCV VideoCapture 中读取旋转角度。

    OpenCV 4.x 通过 CAP_PROP_ORIENTATION_META 属性暴露旋转角度。
    部分编解码器或旧版本可能不支持，此时返回 0。

    Returns
    -------
    int : 旋转角度，取值为 0 / 90 / 180 / 270。
    """
    try:
        rotation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))
        if rotation not in (0, 90, 180, 270):
            rotation = 0
        return rotation
    except Exception:
        return 0


def apply_rotation(frame: np.ndarray, rotation: int) -> np.ndarray:
    """
    对帧应用旋转校正。

    Parameters
    ----------
    frame : np.ndarray
        原始帧（H×W×3 BGR）。
    rotation : int
        顺时针旋转角度（0 / 90 / 180 / 270）。
        注意：视频元数据中的 rotation 表示「需要旋转多少度才能正向显示」。
        OpenCV 读取的原始帧尚未旋转，需要手动应用。

    Returns
    -------
    np.ndarray : 旋转后的帧。
    """
    if rotation == 0:
        return frame
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    logger.warning("未知旋转角度 %d，跳过旋转。", rotation)
    return frame


# ---------------------------------------------------------------------------
# 主读取类
# ---------------------------------------------------------------------------

class VideoReader:
    """
    封装视频读取操作，自动处理旋转元数据。

    典型用法
    --------
    >>> reader = VideoReader("path/to/video.mp4")
    >>> print(reader.meta)
    >>> for frame_id, frame_hr, frame_lr in reader.iter_frames(step=5):
    ...     process(frame_id, frame_hr, frame_lr)
    >>> reader.close()

    或使用上下文管理器：
    >>> with VideoReader("path/to/video.mp4") as reader:
    ...     frame = reader.read_frame(42)
    """

    def __init__(self, video_path: str | Path, apply_orient: bool = True) -> None:
        """
        Parameters
        ----------
        video_path : str | Path
            视频文件路径。
        apply_orient : bool
            是否自动应用旋转校正（默认 True）。
            设为 False 可获取原始未旋转的帧（用于调试）。
        """
        self.path = Path(video_path)
        self.apply_orient = apply_orient

        if not self.path.exists():
            raise FileNotFoundError(f"视频文件不存在: {self.path}")
        if self.path.suffix.lower() not in VIDEO_EXTENSIONS:
            logger.warning(
                "文件后缀 '%s' 不在已知视频格式列表中，仍尝试打开。",
                self.path.suffix,
            )

        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise RuntimeError(f"OpenCV 无法打开视频: {self.path}")

        self.meta = self._build_meta()
        logger.info("已打开视频: %s", self.meta)

    def _build_meta(self) -> VideoMeta:
        """读取视频元数据并构建 VideoMeta 对象。"""
        raw_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        raw_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        rotation = _get_rotation_from_cap(self._cap) if self.apply_orient else 0

        # 某些容器（如 MOV）帧数可能为 0，做修正
        if frame_count <= 0:
            logger.warning("视频帧数读取失败，尝试手动计数（较慢）...")
            frame_count = self._count_frames_manually()

        return VideoMeta(
            path=self.path,
            raw_width=raw_w,
            raw_height=raw_h,
            fps=fps,
            frame_count=frame_count,
            rotation=rotation,
        )

    def _count_frames_manually(self) -> int:
        """通过跳到最后一帧来估算帧数（快速）。"""
        self._cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1.0)
        count = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return max(count, 1)

    # ------------------------------------------------------------------
    # 帧读取接口
    # ------------------------------------------------------------------

    def read_frame(
        self,
        frame_id: int,
        low_res: bool = False,
    ) -> Optional[np.ndarray]:
        """
        读取指定帧编号的帧（随机访问）。

        Parameters
        ----------
        frame_id : int
            帧编号（0-indexed）。
        low_res : bool
            若为 True，返回低分辨率版本；否则返回原始分辨率。

        Returns
        -------
        np.ndarray (H×W×3 BGR) 或 None（读取失败时）。
        """
        if frame_id < 0 or frame_id >= self.meta.frame_count:
            logger.warning("帧编号 %d 超出范围 [0, %d)。", frame_id, self.meta.frame_count)
            return None

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = self._cap.read()
        if not ret or frame is None:
            logger.warning("读取第 %d 帧失败。", frame_id)
            return None

        if self.apply_orient:
            frame = apply_rotation(frame, self.meta.rotation)

        if low_res:
            frame = self._to_low_res(frame)

        return frame

    def read_frame_pair(
        self,
        frame_id: int,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        读取指定帧，同时返回原始分辨率和低分辨率版本。

        Returns
        -------
        (frame_hr, frame_lr) : 原始分辨率帧, 低分辨率帧。
        两者均可能为 None（读取失败时）。
        """
        frame_hr = self.read_frame(frame_id, low_res=False)
        if frame_hr is None:
            return None, None
        frame_lr = self._to_low_res(frame_hr)
        return frame_hr, frame_lr

    def read_mid_frame(self) -> Optional[np.ndarray]:
        """读取视频中间帧（原始分辨率），用于生成 mask 图像。"""
        return self.read_frame(self.meta.mid_frame_id, low_res=False)

    def iter_frames(
        self,
        step: int = 1,
        start: int = 0,
        end: Optional[int] = None,
        yield_low_res: bool = True,
    ) -> Generator[Tuple[int, np.ndarray, Optional[np.ndarray]], None, None]:
        """
        顺序迭代视频帧（效率高于随机访问）。

        Parameters
        ----------
        step : int
            采样步长，每隔 step 帧读取一帧（默认 1，即逐帧）。
        start : int
            起始帧编号（含）。
        end : int | None
            终止帧编号（不含）；None 表示读到末尾。
        yield_low_res : bool
            是否同时产出低分辨率帧（默认 True）。

        Yields
        ------
        (frame_id, frame_hr, frame_lr)
            frame_id   : int，帧编号
            frame_hr   : np.ndarray，原始分辨率帧 (H×W×3 BGR)
            frame_lr   : np.ndarray | None，低分辨率帧（yield_low_res=False 时为 None）
        """
        if end is None:
            end = self.meta.frame_count

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        current_id = start

        while current_id < end:
            ret, frame = self._cap.read()
            if not ret or frame is None:
                break

            if self.apply_orient:
                frame = apply_rotation(frame, self.meta.rotation)

            # 步长跳帧：只在步长整除位置 yield
            if (current_id - start) % step == 0:
                frame_lr = self._to_low_res(frame) if yield_low_res else None
                yield current_id, frame, frame_lr

            current_id += 1

    def iter_frames_at(
        self,
        frame_ids: List[int],
        yield_low_res: bool = True,
    ) -> Generator[Tuple[int, np.ndarray, Optional[np.ndarray]], None, None]:
        """
        按指定帧编号列表顺序读取帧（单向顺序访问，效率优于随机 seek）。

        当 frame_ids 已排序且连续时效率最高；若有跳帧会自动跳过中间帧。

        Parameters
        ----------
        frame_ids : List[int]
            需要读取的帧编号列表，**必须升序排列**。
        yield_low_res : bool
            是否同时产出低分辨率帧。

        Yields
        ------
        (frame_id, frame_hr, frame_lr)
        """
        if not frame_ids:
            return

        sorted_ids = sorted(set(frame_ids))
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, sorted_ids[0])
        current_pos = sorted_ids[0]

        for target_id in sorted_ids:
            if target_id >= self.meta.frame_count:
                logger.warning("帧编号 %d 超出总帧数 %d，跳过。", target_id, self.meta.frame_count)
                continue

            # 若需要向前跳帧，使用 seek
            if target_id < current_pos:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, target_id)
                current_pos = target_id

            # 顺序读取直到目标帧
            while current_pos <= target_id:
                ret, frame = self._cap.read()
                if not ret or frame is None:
                    break
                cur_frame = frame
                current_pos += 1

            if not ret or frame is None:
                logger.warning("读取帧 %d 失败。", target_id)
                continue

            if self.apply_orient:
                cur_frame = apply_rotation(cur_frame, self.meta.rotation)

            frame_lr = self._to_low_res(cur_frame) if yield_low_res else None
            yield target_id, cur_frame, frame_lr

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _to_low_res(self, frame: np.ndarray) -> np.ndarray:
        """将帧缩放至低分辨率（用于配准）。"""
        scale = self.meta.low_res_scale
        if scale >= 1.0:
            return frame.copy()
        w, h = self.meta.low_res_size
        return cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)

    # ------------------------------------------------------------------
    # 上下文管理器
    # ------------------------------------------------------------------

    def close(self) -> None:
        """释放 VideoCapture 资源。"""
        if self._cap.isOpened():
            self._cap.release()
            logger.debug("已关闭视频: %s", self.path.name)

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def list_videos(folder: str | Path) -> List[Path]:
    """
    列出文件夹中所有支持的视频文件，按文件名排序。

    Parameters
    ----------
    folder : str | Path
        视频文件夹路径。

    Returns
    -------
    List[Path] : 视频文件路径列表（已排序）。
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"路径不是文件夹: {folder}")

    videos = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )

    if not videos:
        logger.warning("文件夹 '%s' 中未找到任何视频文件。", folder)
    else:
        logger.info("在 '%s' 中找到 %d 段视频。", folder, len(videos))

    return videos


def get_video_name(video_path: str | Path) -> str:
    """
    获取视频名称（不含后缀），用作 result.npy 的 key。

    >>> get_video_name("/data/IMG_2374.MOV")
    'IMG_2374'
    """
    return Path(video_path).stem


def crop_region(
    frame: np.ndarray,
    bbox: np.ndarray,
    padding: float = 0.0,
) -> np.ndarray:
    """
    从帧中裁切 bbox 区域。

    Parameters
    ----------
    frame : np.ndarray
        源帧（H×W×3）。
    bbox : np.ndarray
        [x1, y1, x2, y2] 格式的裁切框（像素坐标）。
    padding : float
        在 bbox 四周额外扩展的比例（相对于 bbox 短边）。
        例如 0.1 表示扩展 bbox 短边的 10%。

    Returns
    -------
    np.ndarray : 裁切后的图像区域，保证不越界。空框时返回 1×1×3 的零数组。
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox[:4].astype(float)

    if padding > 0:
        bw = x2 - x1
        bh = y2 - y1
        pad = min(bw, bh) * padding
        x1 -= pad
        y1 -= pad
        x2 += pad
        y2 += pad

    x1 = int(max(0, round(x1)))
    y1 = int(max(0, round(y1)))
    x2 = int(min(w, round(x2)))
    y2 = int(min(h, round(y2)))

    if x2 <= x1 or y2 <= y1:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    return frame[y1:y2, x1:x2].copy()


def resize_to_square(
    image: np.ndarray,
    size: int = 224,
    pad_color: Tuple[int, int, int] = (114, 114, 114),
) -> np.ndarray:
    """
    将图像等比缩放并填充到 size×size 的正方形，用于分类器输入。

    Parameters
    ----------
    image : np.ndarray
        输入图像（H×W×3）。
    size : int
        目标边长（默认 224）。
    pad_color : Tuple[int, int, int]
        填充颜色（BGR），默认灰色 (114, 114, 114)。

    Returns
    -------
    np.ndarray : size×size×3 的图像。
    """
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return np.full((size, size, 3), pad_color, dtype=np.uint8)

    scale = size / max(h, w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((size, size, 3), pad_color, dtype=np.uint8)
    top = (size - new_h) // 2
    left = (size - new_w) // 2
    canvas[top: top + new_h, left: left + new_w] = resized

    return canvas


def estimate_blur(image: np.ndarray) -> float:
    """
    估算图像模糊程度（拉普拉斯方差，越高越清晰）。

    用于在 Cluster 中选取最佳 crop（清晰度最高的观测）。

    Parameters
    ----------
    image : np.ndarray
        输入图像（H×W×3 或 H×W）。

    Returns
    -------
    float : 拉普拉斯方差，值越大图像越清晰。
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
