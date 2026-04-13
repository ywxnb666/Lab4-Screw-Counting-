#!/usr/bin/env python3
"""
tools/extract_keyframes.py - 关键帧批量抽取工具
Owner: D（工程封装）

用途：
  从视频文件（夹）中抽取关键帧，保存为图像文件，供全员进行 bbox 标注。
  这是数据准备流程的第一步（对应 plan.md 6.1 节"第 1 步"）。

支持的抽取策略：
  motion   : 基于 ORB 特征点位移（与 pipeline.py 保持一致，推荐）
  uniform  : 均匀时间采样
  scene    : 基于帧间差分的场景切换检测

使用示例：
  # 从单个视频抽取关键帧（motion 策略）
  python tools/extract_keyframes.py --input vedio_exp/IMG_2374.MOV --output frames/

  # 从整个文件夹抽取，每视频最多 40 帧
  python tools/extract_keyframes.py --input vedio_exp/ --output frames/ --max_frames 40

  # 均匀采样，每 1 秒取 1 帧
  python tools/extract_keyframes.py --input vedio_exp/ --output frames/ \\
      --strategy uniform --fps 1.0

  # 同时导出帧编号清单（供标注工具批量导入）
  python tools/extract_keyframes.py --input vedio_exp/ --output frames/ \\
      --export_manifest --manifest_path frame_list.txt

依赖：opencv-python, numpy
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
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
logger = logging.getLogger("extract_keyframes")

# ---------------------------------------------------------------------------
# 支持的视频格式
# ---------------------------------------------------------------------------

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv", ".flv", ".webm"}

# ---------------------------------------------------------------------------
# 默认超参数
# ---------------------------------------------------------------------------

DEFAULT_MOTION_THRESH_RATIO: float = 0.15   # 触发关键帧的累积位移比例（相对帧宽）
DEFAULT_MIN_INTERVAL: int = 5               # 关键帧最小间隔（帧数）
DEFAULT_MAX_INTERVAL: int = 15              # 关键帧最大间隔（强制选取）
DEFAULT_MAX_FRAMES: int = 60               # 单视频最多关键帧数
DEFAULT_MIN_FRAMES: int = 8                # 单视频最少关键帧数
DEFAULT_ORB_N_FEATURES: int = 200          # ORB 特征点数（用于位移估计）
DEFAULT_SCENE_THRESH: float = 30.0         # 场景切换检测的帧差均值阈值（像素）
DEFAULT_UNIFORM_FPS: float = 3.0           # 均匀采样时的目标帧率（帧/秒）
DEFAULT_IMAGE_QUALITY: int = 95            # JPEG 压缩质量（0-100）
DEFAULT_MAX_LONG_EDGE: int = 1920          # 输出图像最大长边（0=不缩放）


# ---------------------------------------------------------------------------
# 视频元数据读取
# ---------------------------------------------------------------------------

def _get_rotation(cap: cv2.VideoCapture) -> int:
    """读取视频旋转角度（0/90/180/270）。"""
    try:
        rot = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))
        return rot if rot in (0, 90, 180, 270) else 0
    except Exception:
        return 0


def _apply_rotation(frame: np.ndarray, rotation: int) -> np.ndarray:
    """对帧应用旋转校正。"""
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def _resize_if_needed(
    frame: np.ndarray,
    max_long_edge: int,
) -> np.ndarray:
    """
    若帧的长边超过 max_long_edge，则等比缩小；否则直接返回。

    Parameters
    ----------
    frame : np.ndarray
        输入帧（H×W×3）。
    max_long_edge : int
        最大长边像素数；0 表示不缩放。

    Returns
    -------
    np.ndarray
    """
    if max_long_edge <= 0:
        return frame
    h, w = frame.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_long_edge:
        return frame
    scale = max_long_edge / long_edge
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


# ---------------------------------------------------------------------------
# 抽帧策略实现
# ---------------------------------------------------------------------------

class KeyframeExtractor:
    """
    视频关键帧抽取器。

    支持三种策略：
    - motion   : 基于 ORB 特征点位移触发关键帧（与 pipeline.py 保持一致）
    - uniform  : 按固定时间间隔均匀采样
    - scene    : 基于相邻帧差分均值检测场景切换

    用法
    ----
    >>> extractor = KeyframeExtractor(strategy="motion")
    >>> frame_ids, frames = extractor.extract("video.mp4")
    >>> extractor.save(frames, frame_ids, output_dir="frames/", video_name="IMG_2374")
    """

    def __init__(
        self,
        strategy: str = "motion",
        motion_thresh_ratio: float = DEFAULT_MOTION_THRESH_RATIO,
        min_interval: int = DEFAULT_MIN_INTERVAL,
        max_interval: int = DEFAULT_MAX_INTERVAL,
        max_frames: int = DEFAULT_MAX_FRAMES,
        min_frames: int = DEFAULT_MIN_FRAMES,
        orb_n_features: int = DEFAULT_ORB_N_FEATURES,
        scene_thresh: float = DEFAULT_SCENE_THRESH,
        uniform_fps: float = DEFAULT_UNIFORM_FPS,
        max_long_edge: int = DEFAULT_MAX_LONG_EDGE,
        apply_orient: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        strategy : str
            抽帧策略：'motion' | 'uniform' | 'scene'。
        motion_thresh_ratio : float
            运动策略：触发关键帧的累积位移比例（相对帧宽）。
        min_interval : int
            运动策略：关键帧最小间隔（帧数）。
        max_interval : int
            运动策略：关键帧最大间隔（帧数，强制选取）。
        max_frames : int
            单视频最多关键帧数。
        min_frames : int
            单视频最少关键帧数（保底均匀补充）。
        orb_n_features : int
            运动策略：ORB 特征点数量（越多越精确，但越慢）。
        scene_thresh : float
            场景策略：帧差均值阈值（像素），超过此值视为场景切换。
        uniform_fps : float
            均匀策略：每秒采样帧数（例如 1.0 表示每秒 1 帧）。
        max_long_edge : int
            输出图像最大长边（0=不缩放保留原始分辨率）。
        apply_orient : bool
            是否自动校正手机视频的旋转方向。
        """
        if strategy not in ("motion", "uniform", "scene"):
            raise ValueError(f"不支持的策略: '{strategy}'，请选择 'motion'、'uniform' 或 'scene'。")

        self.strategy = strategy
        self.motion_thresh_ratio = motion_thresh_ratio
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.max_frames = max_frames
        self.min_frames = min_frames
        self.orb_n_features = orb_n_features
        self.scene_thresh = scene_thresh
        self.uniform_fps = uniform_fps
        self.max_long_edge = max_long_edge
        self.apply_orient = apply_orient

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def extract(
        self,
        video_path: str | Path,
    ) -> Tuple[List[int], List[np.ndarray]]:
        """
        从单个视频文件中抽取关键帧。

        Parameters
        ----------
        video_path : str | Path
            视频文件路径。

        Returns
        -------
        frame_ids : List[int]
            关键帧编号列表（0-indexed，升序）。
        frames : List[np.ndarray]
            对应的关键帧图像列表（H×W×3 BGR）。
            若 max_long_edge > 0，图像已缩放至长边不超过 max_long_edge。

        Raises
        ------
        FileNotFoundError
            若视频文件不存在。
        RuntimeError
            若视频无法打开。
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV 无法打开视频: {video_path}")

        try:
            return self._extract_impl(cap, video_path.name)
        finally:
            cap.release()

    def save(
        self,
        frames: List[np.ndarray],
        frame_ids: List[int],
        output_dir: str | Path,
        video_name: str,
        image_format: str = "jpg",
        quality: int = DEFAULT_IMAGE_QUALITY,
        create_subdir: bool = True,
    ) -> List[Path]:
        """
        将抽取的关键帧保存为图像文件。

        命名规则：{output_dir}/{video_name}/{video_name}_frame{frame_id:06d}.{ext}
        若 create_subdir=False：{output_dir}/{video_name}_frame{frame_id:06d}.{ext}

        Parameters
        ----------
        frames : List[np.ndarray]
            关键帧图像列表。
        frame_ids : List[int]
            对应帧编号列表。
        output_dir : str | Path
            输出根目录。
        video_name : str
            视频名称（不含后缀），用于构建子目录和文件名。
        image_format : str
            输出图像格式：'jpg'（默认，较小）或 'png'（无损）。
        quality : int
            JPEG 压缩质量（0-100，仅 jpg 格式有效）。
        create_subdir : bool
            是否为每个视频创建独立子目录（推荐 True，避免不同视频的帧命名冲突）。

        Returns
        -------
        List[Path] : 所有已保存图像的路径列表。
        """
        output_dir = Path(output_dir)
        if create_subdir:
            save_dir = output_dir / video_name
        else:
            save_dir = output_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        ext = image_format.lower().lstrip(".")
        if ext not in ("jpg", "jpeg", "png"):
            logger.warning("不支持的图像格式 '%s'，回退到 'jpg'。", ext)
            ext = "jpg"

        encode_params = []
        if ext in ("jpg", "jpeg"):
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        elif ext == "png":
            encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]  # 适中压缩

        saved_paths: List[Path] = []
        for frame, fid in zip(frames, frame_ids):
            filename = f"{video_name}_frame{fid:06d}.{ext}"
            save_path = save_dir / filename
            success = cv2.imwrite(str(save_path), frame, encode_params)
            if success:
                saved_paths.append(save_path)
                logger.debug("已保存: %s", save_path)
            else:
                logger.warning("保存失败: %s", save_path)

        logger.info(
            "视频 '%s': 已保存 %d / %d 帧至 '%s'",
            video_name, len(saved_paths), len(frames), save_dir,
        )
        return saved_paths

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _extract_impl(
        self,
        cap: cv2.VideoCapture,
        video_name: str,
    ) -> Tuple[List[int], List[np.ndarray]]:
        """选择对应策略并执行抽取。"""
        # 读取元数据
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        rotation = _get_rotation(cap) if self.apply_orient else 0

        # 旋转后的真实宽高
        if rotation in (90, 270):
            frame_w, frame_h = raw_h, raw_w
        else:
            frame_w, frame_h = raw_w, raw_h

        # 若帧数为 0（某些容器格式），手动计数
        if total_frames <= 0:
            logger.warning("视频帧数读取失败，尝试手动统计...")
            cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1.0)
            total_frames = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            total_frames = max(total_frames, 1)

        logger.info(
            "视频: %s  |  %dx%d  |  %.2f fps  |  %d 帧  |  旋转=%d°",
            video_name, frame_w, frame_h, fps, total_frames, rotation,
        )

        if self.strategy == "motion":
            frame_ids = self._strategy_motion(cap, total_frames, fps, frame_w, rotation)
        elif self.strategy == "uniform":
            frame_ids = self._strategy_uniform(total_frames, fps)
        else:  # scene
            frame_ids = self._strategy_scene(cap, total_frames, rotation)

        # 补充保底关键帧（若数量不足）
        frame_ids = self._ensure_min_frames(frame_ids, total_frames)

        # 限制最大帧数
        if len(frame_ids) > self.max_frames:
            indices = np.linspace(0, len(frame_ids) - 1, self.max_frames, dtype=int)
            frame_ids = [frame_ids[i] for i in indices]

        frame_ids = sorted(set(frame_ids))

        # 读取选定帧的实际图像
        frames = self._read_frames(cap, frame_ids, rotation)

        logger.info(
            "策略='%s'  →  选取 %d 帧（ID 范围: %d ~ %d）",
            self.strategy,
            len(frame_ids),
            frame_ids[0] if frame_ids else -1,
            frame_ids[-1] if frame_ids else -1,
        )
        return frame_ids, frames

    def _strategy_motion(
        self,
        cap: cv2.VideoCapture,
        total_frames: int,
        fps: float,
        frame_w: int,
        rotation: int,
    ) -> List[int]:
        """
        基于 ORB 特征点位移的关键帧选取策略。

        与 pipeline.py 中的 _extract_keyframes_motion() 保持一致，
        确保标注数据和推理时选取的帧尽量相同。
        """
        motion_thresh_px = frame_w * self.motion_thresh_ratio
        orb = cv2.ORB_create(nfeatures=self.orb_n_features)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        keyframe_ids: List[int] = []
        last_kf_id = -self.min_interval
        cumulative_disp = 0.0
        prev_kpts = None
        prev_descs = None

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        for frame_id in range(total_frames):
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if self.apply_orient:
                frame = _apply_rotation(frame, rotation)

            # 降采样到 640 宽以加速 ORB
            h, w = frame.shape[:2]
            if w > 640:
                scale = 640 / w
                small = cv2.resize(frame, (640, int(h * scale)))
            else:
                small = frame
                scale = 1.0

            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            kpts, descs = orb.detectAndCompute(gray, None)

            if (prev_descs is not None and descs is not None
                    and len(kpts) > 5 and len(prev_kpts) > 5):
                try:
                    matches = bf.match(descs, prev_descs)
                    if matches:
                        matches_sorted = sorted(matches, key=lambda m: m.distance)
                        good = matches_sorted[: max(1, len(matches_sorted) // 2)]
                        disps = [
                            np.linalg.norm(
                                np.array(kpts[m.queryIdx].pt)
                                - np.array(prev_kpts[m.trainIdx].pt)
                            )
                            for m in good
                        ]
                        disps_arr = np.array(disps)
                        median_d = np.median(disps_arr)
                        valid = disps_arr[disps_arr < max(1.0, median_d * 3.0)]
                        mean_d = float(np.mean(valid)) if len(valid) > 0 else 0.0
                        # 还原到原始分辨率位移
                        cumulative_disp += mean_d / scale
                except cv2.error:
                    pass

            prev_kpts = kpts
            prev_descs = descs

            frames_since = frame_id - last_kf_id
            should_select = False

            if frame_id == 0:
                should_select = True
            elif frames_since < self.min_interval:
                should_select = False
            elif frames_since >= self.max_interval:
                should_select = True
            elif cumulative_disp >= motion_thresh_px:
                should_select = True

            if should_select:
                keyframe_ids.append(frame_id)
                last_kf_id = frame_id
                cumulative_disp = 0.0

                if len(keyframe_ids) >= self.max_frames:
                    break

        # 包含最后一帧
        last_id = total_frames - 1
        if keyframe_ids and keyframe_ids[-1] != last_id:
            if last_id - keyframe_ids[-1] >= self.min_interval:
                keyframe_ids.append(last_id)

        return keyframe_ids

    def _strategy_uniform(
        self,
        total_frames: int,
        fps: float,
    ) -> List[int]:
        """
        均匀时间采样策略：按目标帧率等间隔采样。
        """
        if fps <= 0:
            fps = 30.0
        interval = max(1, int(round(fps / self.uniform_fps)))
        frame_ids = list(range(0, total_frames, interval))

        # 确保最后一帧被包含
        last_id = total_frames - 1
        if frame_ids and frame_ids[-1] != last_id:
            frame_ids.append(last_id)

        logger.debug(
            "均匀采样: 视频帧率=%.2f, 目标帧率=%.2f, 间隔=%d帧, 采样%d帧",
            fps, self.uniform_fps, interval, len(frame_ids),
        )
        return frame_ids

    def _strategy_scene(
        self,
        cap: cv2.VideoCapture,
        total_frames: int,
        rotation: int,
    ) -> List[int]:
        """
        基于帧差分的场景切换检测策略。

        当相邻帧的灰度均值差超过 scene_thresh 时，视为场景切换，选取为关键帧。
        适合拍摄停顿明显的视频。
        """
        keyframe_ids: List[int] = [0]  # 第一帧必选
        prev_gray = None
        last_kf_id = 0

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        for frame_id in range(total_frames):
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if self.apply_orient:
                frame = _apply_rotation(frame, rotation)

            # 降采样加速
            h, w = frame.shape[:2]
            if w > 480:
                scale = 480 / w
                small = cv2.resize(frame, (480, int(h * scale)))
            else:
                small = frame

            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)

            if prev_gray is not None:
                diff = np.abs(gray - prev_gray).mean()
                frames_since = frame_id - last_kf_id

                if frames_since >= self.min_interval and (
                    diff >= self.scene_thresh
                    or frames_since >= self.max_interval
                ):
                    keyframe_ids.append(frame_id)
                    last_kf_id = frame_id

                    if len(keyframe_ids) >= self.max_frames:
                        break

            prev_gray = gray

        # 包含最后一帧
        last_id = total_frames - 1
        if keyframe_ids and keyframe_ids[-1] != last_id:
            if last_id - keyframe_ids[-1] >= self.min_interval:
                keyframe_ids.append(last_id)

        return keyframe_ids

    def _ensure_min_frames(
        self,
        frame_ids: List[int],
        total_frames: int,
    ) -> List[int]:
        """若关键帧数量不足 min_frames，均匀补充直到达标。"""
        if len(frame_ids) >= self.min_frames:
            return frame_ids

        extra_needed = self.min_frames - len(frame_ids)
        existing = set(frame_ids)
        step = max(1, total_frames // (self.min_frames + 1))

        for fid in range(0, total_frames, step):
            if fid not in existing:
                frame_ids.append(fid)
                existing.add(fid)
                extra_needed -= 1
                if extra_needed <= 0:
                    break

        return sorted(frame_ids)

    def _read_frames(
        self,
        cap: cv2.VideoCapture,
        frame_ids: List[int],
        rotation: int,
    ) -> List[np.ndarray]:
        """
        按帧编号顺序读取帧图像（单向读取，效率高于随机 seek）。

        Parameters
        ----------
        cap : cv2.VideoCapture
            已打开的视频捕获对象。
        frame_ids : List[int]
            需要读取的帧编号列表（必须升序）。
        rotation : int
            旋转角度（0/90/180/270）。

        Returns
        -------
        List[np.ndarray] : 读取成功的帧图像列表（长度可能 < len(frame_ids)）。
        """
        if not frame_ids:
            return []

        frames: List[np.ndarray] = []
        sorted_ids = sorted(frame_ids)

        cap.set(cv2.CAP_PROP_POS_FRAMES, sorted_ids[0])
        current_pos = sorted_ids[0]

        for target_id in sorted_ids:
            # 若目标帧在当前位置之前，需要 seek
            if target_id < current_pos:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_id)
                current_pos = target_id

            # 顺序读取到目标帧
            frame = None
            while current_pos <= target_id:
                ret, f = cap.read()
                if not ret or f is None:
                    break
                frame = f
                current_pos += 1

            if frame is None:
                logger.warning("帧 %d 读取失败，跳过。", target_id)
                continue

            if self.apply_orient:
                frame = _apply_rotation(frame, rotation)

            # 按需缩放
            frame = _resize_if_needed(frame, self.max_long_edge)

            frames.append(frame)

        return frames


# ---------------------------------------------------------------------------
# 清单导出（供标注工具批量导入）
# ---------------------------------------------------------------------------

class ManifestExporter:
    """
    关键帧清单导出器。

    生成帧编号清单文件（文本格式或 JSON 格式），
    供标注工具（Roboflow / CVAT）批量导入，或供团队共享抽帧记录。
    """

    @staticmethod
    def export_txt(
        manifest: Dict[str, List[int]],
        output_path: str | Path,
    ) -> None:
        """
        导出为纯文本格式清单。

        格式：
            # video_name  frame_ids（逗号分隔）
            IMG_2374  0,45,90,135,...
            IMG_2375  0,30,60,...

        Parameters
        ----------
        manifest : Dict[str, List[int]]
            {video_name: [frame_id, ...]} 的字典。
        output_path : str | Path
            输出文本文件路径。
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            "# 关键帧清单 - 由 tools/extract_keyframes.py 自动生成",
            "# 格式: 视频名称 <TAB> 帧编号（逗号分隔）",
            "",
        ]
        for video_name in sorted(manifest.keys()):
            ids_str = ",".join(str(fid) for fid in sorted(manifest[video_name]))
            lines.append(f"{video_name}\t{ids_str}")

        output_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("清单已导出（TXT）: %s", output_path)

    @staticmethod
    def export_json(
        manifest: Dict[str, List[int]],
        output_path: str | Path,
        extra_meta: Optional[Dict] = None,
    ) -> None:
        """
        导出为 JSON 格式清单（含元数据）。

        Parameters
        ----------
        manifest : Dict[str, List[int]]
            {video_name: [frame_id, ...]} 的字典。
        output_path : str | Path
            输出 JSON 文件路径。
        extra_meta : Dict | None
            附加元数据（如策略参数），将写入 JSON 的 'meta' 字段。
        """
        import datetime

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "meta": {
                "generated_at": datetime.datetime.now().isoformat(),
                "total_videos": len(manifest),
                "total_frames": sum(len(v) for v in manifest.values()),
                **(extra_meta or {}),
            },
            "manifest": {
                video_name: sorted(ids)
                for video_name, ids in sorted(manifest.items())
            },
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info("清单已导出（JSON）: %s", output_path)

    @staticmethod
    def load_txt(path: str | Path) -> Dict[str, List[int]]:
        """
        从 TXT 清单文件加载帧编号字典。

        Parameters
        ----------
        path : str | Path
            清单文件路径。

        Returns
        -------
        Dict[str, List[int]]
        """
        manifest: Dict[str, List[int]] = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t", 1)
                if len(parts) != 2:
                    continue
                video_name, ids_str = parts
                ids = [int(x) for x in ids_str.split(",") if x.strip().isdigit()]
                manifest[video_name] = ids
        return manifest


# ---------------------------------------------------------------------------
# 统计报告
# ---------------------------------------------------------------------------

def _print_stats(
    all_results: Dict[str, Tuple[List[int], List[np.ndarray]]],
    elapsed: float,
) -> None:
    """
    打印抽帧结果统计报告。

    Parameters
    ----------
    all_results : Dict[str, Tuple[List[int], List[np.ndarray]]]
        {video_name: (frame_ids, frames)} 的字典。
    elapsed : float
        总耗时（秒）。
    """
    print("\n" + "=" * 65)
    print(f"{'视频名称':30s}  {'关键帧数':>6s}  {'时间段':>10s}")
    print("-" * 65)

    total_frames = 0
    for vname, (fids, _) in sorted(all_results.items()):
        n = len(fids)
        total_frames += n
        id_range = f"{fids[0]}~{fids[-1]}" if fids else "N/A"
        print(f"{vname:30s}  {n:>6d}  {id_range:>10s}")

    print("=" * 65)
    print(f"  总计: {len(all_results)} 个视频，{total_frames} 张关键帧")
    print(f"  耗时: {elapsed:.2f} 秒")
    print()


# ---------------------------------------------------------------------------
# 命令行接口
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        prog="extract_keyframes.py",
        description="视频关键帧批量抽取工具 (Owner: D)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 从单个视频抽取关键帧
  python tools/extract_keyframes.py --input vedio_exp/IMG_2374.MOV --output frames/

  # 从整个文件夹批量处理，最多 40 帧/视频
  python tools/extract_keyframes.py --input vedio_exp/ --output frames/ --max_frames 40

  # 均匀采样（每秒 2 帧）
  python tools/extract_keyframes.py --input vedio_exp/ --output frames/ \\
      --strategy uniform --fps 2.0

  # 不缩放，保留原始分辨率（4K 视频，输出 PNG）
  python tools/extract_keyframes.py --input vedio_exp/ --output frames_4k/ \\
      --max_long_edge 0 --format png

  # 同时导出 JSON 清单
  python tools/extract_keyframes.py --input vedio_exp/ --output frames/ \\
      --export_manifest --manifest_path keyframes.json --manifest_format json
        """,
    )

    # ---- 输入/输出 ----
    io_group = parser.add_argument_group("输入/输出")
    io_group.add_argument(
        "--input", "-i",
        required=True,
        help="视频文件路径，或包含多个视频的文件夹路径。",
    )
    io_group.add_argument(
        "--output", "-o",
        required=True,
        help="关键帧图像的输出根目录（每个视频在此目录下创建同名子文件夹）。",
    )
    io_group.add_argument(
        "--format",
        default="jpg",
        choices=["jpg", "png"],
        help="输出图像格式（默认：jpg，文件较小；png 无损但较大）。",
    )
    io_group.add_argument(
        "--quality",
        type=int,
        default=DEFAULT_IMAGE_QUALITY,
        help=f"JPEG 压缩质量 0-100（默认 {DEFAULT_IMAGE_QUALITY}，仅 --format jpg 时有效）。",
    )
    io_group.add_argument(
        "--max_long_edge",
        type=int,
        default=DEFAULT_MAX_LONG_EDGE,
        help=f"输出图像最大长边（像素，默认 {DEFAULT_MAX_LONG_EDGE}；0 表示不缩放保留原始分辨率）。",
    )
    io_group.add_argument(
        "--no_subdir",
        action="store_true",
        default=False,
        help="不为每个视频创建子目录，所有帧直接保存到 --output（多视频时可能命名冲突）。",
    )

    # ---- 抽帧策略 ----
    strategy_group = parser.add_argument_group("抽帧策略")
    strategy_group.add_argument(
        "--strategy", "-s",
        default="motion",
        choices=["motion", "uniform", "scene"],
        help="关键帧选取策略（默认：motion）。"
             "motion=基于运动位移; uniform=均匀时间采样; scene=场景切换检测。",
    )
    strategy_group.add_argument(
        "--max_frames",
        type=int,
        default=DEFAULT_MAX_FRAMES,
        help=f"单视频最多关键帧数（默认 {DEFAULT_MAX_FRAMES}）。",
    )
    strategy_group.add_argument(
        "--min_frames",
        type=int,
        default=DEFAULT_MIN_FRAMES,
        help=f"单视频最少关键帧数（默认 {DEFAULT_MIN_FRAMES}，不足时均匀补充）。",
    )
    strategy_group.add_argument(
        "--motion_thresh",
        type=float,
        default=DEFAULT_MOTION_THRESH_RATIO,
        help=f"motion 策略：触发关键帧的累积位移比例（相对帧宽，默认 {DEFAULT_MOTION_THRESH_RATIO}）。",
    )
    strategy_group.add_argument(
        "--min_interval",
        type=int,
        default=DEFAULT_MIN_INTERVAL,
        help=f"motion/scene 策略：关键帧最小间隔帧数（默认 {DEFAULT_MIN_INTERVAL}）。",
    )
    strategy_group.add_argument(
        "--max_interval",
        type=int,
        default=DEFAULT_MAX_INTERVAL,
        help=f"motion/scene 策略：关键帧最大间隔帧数（默认 {DEFAULT_MAX_INTERVAL}，超过则强制选取）。",
    )
    strategy_group.add_argument(
        "--fps",
        type=float,
        default=DEFAULT_UNIFORM_FPS,
        help=f"uniform 策略：每秒采样帧数（默认 {DEFAULT_UNIFORM_FPS}）。",
    )
    strategy_group.add_argument(
        "--scene_thresh",
        type=float,
        default=DEFAULT_SCENE_THRESH,
        help=f"scene 策略：帧差均值阈值（像素，默认 {DEFAULT_SCENE_THRESH}）。",
    )

    # ---- 清单导出 ----
    manifest_group = parser.add_argument_group("清单导出（可选）")
    manifest_group.add_argument(
        "--export_manifest",
        action="store_true",
        default=False,
        help="是否导出关键帧清单文件（记录各视频选取的帧编号）。",
    )
    manifest_group.add_argument(
        "--manifest_path",
        type=str,
        default=None,
        help="清单文件输出路径（默认：{output}/keyframe_manifest.{format}）。",
    )
    manifest_group.add_argument(
        "--manifest_format",
        default="txt",
        choices=["txt", "json"],
        help="清单文件格式（默认：txt）。",
    )

    # ---- 其他 ----
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="输出详细调试日志。",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        default=False,
        help="仅统计视频和预计关键帧数，不实际保存图像。",
    )

    return parser.parse_args()


def main() -> int:
    """
    关键帧抽取工具主函数。

    Returns
    -------
    int : 退出码（0=成功，非 0=错误）。
    """
    args = _parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    t_start = time.perf_counter()

    # ---- 收集视频文件 ----
    input_path = Path(args.input)
    if input_path.is_file():
        if input_path.suffix.lower() not in VIDEO_EXTENSIONS:
            logger.error("输入文件不是支持的视频格式: %s", input_path)
            return 1
        videos = [input_path]
    elif input_path.is_dir():
        videos = sorted(
            p for p in input_path.iterdir()
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        )
        if not videos:
            logger.error("在 '%s' 中未找到视频文件。", input_path)
            return 1
    else:
        logger.error("--input 路径不存在: %s", input_path)
        return 1

    logger.info("找到 %d 段视频:", len(videos))
    for v in videos:
        logger.info("  %s", v.name)

    # ---- 初始化抽取器 ----
    extractor = KeyframeExtractor(
        strategy=args.strategy,
        motion_thresh_ratio=args.motion_thresh,
        min_interval=args.min_interval,
        max_interval=args.max_interval,
        max_frames=args.max_frames,
        min_frames=args.min_frames,
        uniform_fps=args.fps,
        scene_thresh=args.scene_thresh,
        max_long_edge=args.max_long_edge,
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: Dict[str, Tuple[List[int], List[np.ndarray]]] = {}
    manifest: Dict[str, List[int]] = {}
    failed: List[str] = []

    # ---- 逐视频处理 ----
    for i, video_path in enumerate(videos):
        video_name = video_path.stem
        logger.info("")
        logger.info("[%d/%d] 处理: %s", i + 1, len(videos), video_path.name)

        try:
            frame_ids, frames = extractor.extract(video_path)

            if not frame_ids:
                logger.warning("视频 '%s' 未能提取到关键帧，跳过。", video_name)
                failed.append(video_name)
                continue

            all_results[video_name] = (frame_ids, frames)
            manifest[video_name] = frame_ids

            # 保存帧图像
            if not args.dry_run:
                extractor.save(
                    frames=frames,
                    frame_ids=frame_ids,
                    output_dir=output_dir,
                    video_name=video_name,
                    image_format=args.format,
                    quality=args.quality,
                    create_subdir=not args.no_subdir,
                )
            else:
                logger.info(
                    "  [dry_run] 将保存 %d 帧到 '%s/%s/'",
                    len(frame_ids), output_dir, video_name,
                )

        except FileNotFoundError as e:
            logger.error("文件不存在: %s", e)
            failed.append(video_name)
        except Exception as e:
            logger.error("处理视频 '%s' 失败: %s", video_name, e, exc_info=args.verbose)
            failed.append(video_name)

    # ---- 导出清单 ----
    if args.export_manifest and manifest and not args.dry_run:
        manifest_path = args.manifest_path
        if manifest_path is None:
            manifest_path = output_dir / f"keyframe_manifest.{args.manifest_format}"
        else:
            manifest_path = Path(manifest_path)

        meta = {
            "strategy": args.strategy,
            "max_frames": args.max_frames,
            "min_frames": args.min_frames,
        }
        if args.manifest_format == "json":
            ManifestExporter.export_json(manifest, manifest_path, extra_meta=meta)
        else:
            ManifestExporter.export_txt(manifest, manifest_path)

    # ---- 统计报告 ----
    elapsed = time.perf_counter() - t_start
    _print_stats(all_results, elapsed)

    if failed:
        logger.warning("以下 %d 个视频处理失败: %s", len(failed), failed)
        return 1

    logger.info("✅ 所有视频处理完成！输出目录: %s", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
