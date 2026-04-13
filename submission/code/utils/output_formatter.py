"""
utils/output_formatter.py - 结果序列化与输出封装
Owner: D（工程封装）

职责：
  - 将 VideoResult 列表序列化为 result.npy（符合作业规范的字典格式）
  - 将总耗时写入 time.txt
  - 将掩膜叠加图保存为 {video_name}_mask.png
  - 提供输出目录的自动创建与路径验证

依赖：numpy, opencv-python
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

NUM_CLASSES = 5          # 螺丝类别数（Type_1 ~ Type_5）
MASK_SUFFIX = "_mask.png"


# ---------------------------------------------------------------------------
# 路径工具
# ---------------------------------------------------------------------------

def _ensure_dir(path: Union[str, Path]) -> Path:
    """
    确保目录存在，不存在则自动创建（含多级父目录）。

    Parameters
    ----------
    path : str | Path
        目标目录路径。

    Returns
    -------
    Path : 已确保存在的目录路径。
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _ensure_parent(path: Union[str, Path]) -> Path:
    """
    确保文件的父目录存在。

    Parameters
    ----------
    path : str | Path
        目标文件路径。

    Returns
    -------
    Path : 文件路径（Path 对象）。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# 核心输出封装类
# ---------------------------------------------------------------------------

class OutputFormatter:
    """
    统一输出格式封装器。

    负责将流水线处理结果序列化为作业要求的格式：
    - result.npy  : 包含计数字典的 NumPy 文件
    - time.txt    : 总耗时（秒）
    - mask 图像   : 每段视频一张的掩膜叠加图

    典型用法
    --------
    >>> formatter = OutputFormatter(
    ...     output_path="./result.npy",
    ...     time_path="./time.txt",
    ...     mask_dir="./masks/",
    ... )
    >>> formatter.save_result(results_dict)
    >>> formatter.save_time(total_seconds)
    >>> formatter.save_mask("IMG_2374", overlay_image)
    """

    def __init__(
        self,
        output_path: Union[str, Path],
        time_path: Union[str, Path],
        mask_dir: Union[str, Path],
    ) -> None:
        """
        Parameters
        ----------
        output_path : str | Path
            result.npy 的保存路径（含文件名）。
        time_path : str | Path
            time.txt 的保存路径（含文件名）。
        mask_dir : str | Path
            掩膜图像的保存目录（文件夹路径）。
        """
        self.output_path = _ensure_parent(output_path)
        self.time_path = _ensure_parent(time_path)
        self.mask_dir = _ensure_dir(mask_dir)

        logger.info(
            "OutputFormatter 初始化完成:\n"
            "  result.npy  -> %s\n"
            "  time.txt    -> %s\n"
            "  mask 目录   -> %s",
            self.output_path,
            self.time_path,
            self.mask_dir,
        )

    # ------------------------------------------------------------------
    # result.npy
    # ------------------------------------------------------------------

    def save_result(
        self,
        results: Dict[str, List[int]],
    ) -> None:
        """
        将计数结果字典保存为 result.npy。

        保存格式符合作业规范：
            numpy.load(path, allow_pickle=True).item() -> dict

        Parameters
        ----------
        results : Dict[str, List[int]]
            key   : 视频文件名（不含后缀，例如 'IMG_2374'）
            value : 长度为 5 的列表，按 [Type_1, …, Type_5] 顺序记录数量

        Raises
        ------
        ValueError
            若任意 value 长度不为 5，或包含负数时抛出。
        """
        self._validate_result_dict(results)

        # 转换 value 为 list[int] 保证可序列化
        serializable: Dict[str, List[int]] = {
            k: [int(c) for c in v]
            for k, v in results.items()
        }

        np.save(str(self.output_path), serializable)
        logger.info(
            "已保存 result.npy -> %s\n  内容: %s",
            self.output_path,
            json.dumps(serializable, ensure_ascii=False, indent=2),
        )

    @staticmethod
    def _validate_result_dict(results: Dict[str, List[int]]) -> None:
        """校验 result 字典格式，不合法时抛出 ValueError。"""
        if not isinstance(results, dict):
            raise TypeError(f"results 必须是 dict，实际类型: {type(results)}")

        for video_name, counts in results.items():
            if not isinstance(video_name, str):
                raise ValueError(f"key 必须是 str，实际: {type(video_name)} ({video_name!r})")
            if len(counts) != NUM_CLASSES:
                raise ValueError(
                    f"视频 '{video_name}' 的计数列表长度为 {len(counts)}，期望 {NUM_CLASSES}。"
                )
            for i, c in enumerate(counts):
                if int(c) < 0:
                    raise ValueError(
                        f"视频 '{video_name}' 第 {i + 1} 类计数为负数: {c}。"
                    )

    @staticmethod
    def load_result(path: Union[str, Path]) -> Dict[str, List[int]]:
        """
        从 result.npy 加载计数字典（用于验证和调试）。

        Parameters
        ----------
        path : str | Path
            result.npy 的路径。

        Returns
        -------
        Dict[str, List[int]]
        """
        data = np.load(str(path), allow_pickle=True).item()
        logger.info("已加载 result.npy: %s", path)
        return data

    # ------------------------------------------------------------------
    # time.txt
    # ------------------------------------------------------------------

    def save_time(self, total_seconds: float) -> None:
        """
        将总处理时间（秒）写入 time.txt。

        文件内容为一个浮点数字符串，保留 4 位小数，末尾带换行。

        Parameters
        ----------
        total_seconds : float
            处理所有测试视频的总耗时（秒）。
        """
        if total_seconds < 0:
            logger.warning("总耗时为负数 (%f s)，已修正为 0。", total_seconds)
            total_seconds = 0.0

        content = f"{total_seconds:.4f}\n"
        self.time_path.write_text(content, encoding="utf-8")
        logger.info(
            "已保存 time.txt -> %s  (%.4f s)",
            self.time_path,
            total_seconds,
        )

    @staticmethod
    def load_time(path: Union[str, Path]) -> float:
        """
        从 time.txt 读取总耗时（用于验证）。

        Parameters
        ----------
        path : str | Path

        Returns
        -------
        float : 总耗时（秒）。
        """
        content = Path(path).read_text(encoding="utf-8").strip()
        return float(content)

    # ------------------------------------------------------------------
    # mask 图像
    # ------------------------------------------------------------------

    def save_mask(
        self,
        video_name: str,
        overlay_image: np.ndarray,
    ) -> Path:
        """
        将掩膜叠加图保存为 PNG 文件。

        文件命名格式：{video_name}_mask.png

        Parameters
        ----------
        video_name : str
            视频名称（不含后缀）。
        overlay_image : np.ndarray
            原图与掩膜叠加后的图像（H×W×3，BGR）。

        Returns
        -------
        Path : 保存路径。

        Raises
        ------
        ValueError
            若 overlay_image 不是合法的图像数组。
        """
        if overlay_image is None or overlay_image.size == 0:
            raise ValueError(f"视频 '{video_name}' 的 mask 图像为空，无法保存。")

        if overlay_image.ndim not in (2, 3):
            raise ValueError(
                f"mask 图像维度应为 2 或 3，实际: {overlay_image.ndim}。"
            )

        save_path = self.mask_dir / f"{video_name}{MASK_SUFFIX}"
        success = cv2.imwrite(str(save_path), overlay_image)

        if not success:
            raise RuntimeError(
                f"cv2.imwrite 写入失败: {save_path}。"
                "请检查路径是否可写，以及图像数组格式是否正确。"
            )

        logger.info(
            "已保存 mask -> %s  (shape: %s)",
            save_path,
            overlay_image.shape,
        )
        return save_path

    def mask_path_for(self, video_name: str) -> Path:
        """
        返回指定视频的 mask 图像路径（不执行写入）。

        Parameters
        ----------
        video_name : str

        Returns
        -------
        Path
        """
        return self.mask_dir / f"{video_name}{MASK_SUFFIX}"

    # ------------------------------------------------------------------
    # 批量保存便捷方法
    # ------------------------------------------------------------------

    def save_all(
        self,
        results: Dict[str, List[int]],
        total_seconds: float,
        masks: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        """
        一次性保存所有输出（result.npy + time.txt + 可选的 mask 图像）。

        Parameters
        ----------
        results : Dict[str, List[int]]
            计数字典（符合 result.npy 格式）。
        total_seconds : float
            总耗时（秒）。
        masks : Dict[str, np.ndarray] | None
            video_name -> overlay_image 的字典；为 None 时跳过 mask 保存。
        """
        self.save_result(results)
        self.save_time(total_seconds)

        if masks:
            for video_name, overlay in masks.items():
                try:
                    self.save_mask(video_name, overlay)
                except Exception as e:
                    logger.error("保存 mask '%s' 失败: %s", video_name, e)

    # ------------------------------------------------------------------
    # 输出验证（用于自测）
    # ------------------------------------------------------------------

    def verify_outputs(self, expected_video_names: List[str]) -> bool:
        """
        验证所有输出文件是否存在且格式正确。

        Parameters
        ----------
        expected_video_names : List[str]
            期望出现在 result.npy 中的视频名称列表。

        Returns
        -------
        bool : 所有验证通过时返回 True，否则返回 False（并记录 WARNING）。
        """
        ok = True

        # 1. result.npy
        if not self.output_path.exists():
            logger.warning("❌  result.npy 不存在: %s", self.output_path)
            ok = False
        else:
            try:
                data = self.load_result(self.output_path)
                for name in expected_video_names:
                    if name not in data:
                        logger.warning("❌  result.npy 中缺少视频 '%s'。", name)
                        ok = False
                    elif len(data[name]) != NUM_CLASSES:
                        logger.warning(
                            "❌  视频 '%s' 计数长度 %d ≠ %d。",
                            name, len(data[name]), NUM_CLASSES,
                        )
                        ok = False
                    else:
                        logger.info("✅  视频 '%s' 计数: %s", name, data[name])
            except Exception as e:
                logger.warning("❌  result.npy 加载失败: %s", e)
                ok = False

        # 2. time.txt
        if not self.time_path.exists():
            logger.warning("❌  time.txt 不存在: %s", self.time_path)
            ok = False
        else:
            try:
                t = self.load_time(self.time_path)
                logger.info("✅  time.txt: %.4f s", t)
            except Exception as e:
                logger.warning("❌  time.txt 读取失败: %s", e)
                ok = False

        # 3. mask 图像
        for name in expected_video_names:
            mp = self.mask_path_for(name)
            if not mp.exists():
                logger.warning("❌  mask 图像不存在: %s", mp)
                ok = False
            else:
                img = cv2.imread(str(mp))
                if img is None:
                    logger.warning("❌  mask 图像无法读取: %s", mp)
                    ok = False
                else:
                    logger.info("✅  mask: %s  (shape: %s)", mp.name, img.shape)

        if ok:
            logger.info("🎉 所有输出验证通过！")
        else:
            logger.warning("⚠️  部分输出验证失败，请检查上方日志。")

        return ok


# ---------------------------------------------------------------------------
# 计时上下文管理器（D 的工具，供 run.py 使用）
# ---------------------------------------------------------------------------

class Timer:
    """
    简单的计时上下文管理器，用于记录代码块的执行时间。

    用法
    ----
    >>> with Timer("检测模块") as t:
    ...     run_detector(frame)
    >>> print(f"耗时: {t.elapsed:.3f}s")

    或嵌套追踪：
    >>> timer = Timer("总流程")
    >>> timer.start()
    >>> # ... 处理视频 1
    >>> timer.lap("视频1")
    >>> # ... 处理视频 2
    >>> timer.lap("视频2")
    >>> total = timer.stop()
    >>> print(timer.report())
    """

    def __init__(self, name: str = "Timer") -> None:
        self.name = name
        self.elapsed: float = 0.0
        self._start: Optional[float] = None
        self._laps: List[Tuple[str, float]] = []

    def start(self) -> "Timer":
        self._start = time.perf_counter()
        self._laps.clear()
        return self

    def stop(self) -> float:
        if self._start is None:
            return 0.0
        self.elapsed = time.perf_counter() - self._start
        self._start = None
        return self.elapsed

    def lap(self, label: str) -> float:
        """记录一个阶段耗时（从上次 lap 或 start 开始）。"""
        now = time.perf_counter()
        if self._start is None:
            return 0.0
        last = self._laps[-1][1] if self._laps else self._start
        # 转换 last 为绝对时间
        if self._laps:
            # _laps 存储的是阶段耗时，需要重新计算绝对时间
            abs_last = self._start + sum(d for _, d in self._laps)
        else:
            abs_last = self._start
        dur = now - abs_last
        self._laps.append((label, dur))
        return dur

    def report(self) -> str:
        """返回各阶段耗时的格式化报告字符串。"""
        lines = [f"[{self.name}] 总耗时: {self.elapsed:.4f}s"]
        for label, dur in self._laps:
            pct = (dur / self.elapsed * 100) if self.elapsed > 0 else 0.0
            lines.append(f"  {label:30s} {dur:7.3f}s  ({pct:5.1f}%)")
        return "\n".join(lines)

    def __enter__(self) -> "Timer":
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()

    def __repr__(self) -> str:
        return f"Timer('{self.name}', elapsed={self.elapsed:.4f}s)"
