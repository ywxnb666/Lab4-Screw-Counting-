#!/usr/bin/env python3
"""
tools/benchmark.py - 速度 Benchmark 工具
Owner: D（工程封装）

用途：
  对完整处理流水线（或各独立模块）进行计时测试，
  生成结构化的性能报告，帮助团队识别速度瓶颈。

  对应 plan.md 5.3 节要求：
  "第 1 周必须完成速度 benchmark，在 4060 8G 上对一段 4K 样例视频
  跑一遍完整链路，记录各模块耗时。"

报告内容：
  - 各模块的平均/最大/最小耗时
  - 整体处理速度（视频秒数 / 处理秒数）
  - GPU 显存占用（若可用）
  - 关键帧提取效率
  - 检测 FPS（帧/秒）
  - 是否满足作业速度要求（≤10s 处理 10s 视频）

使用示例：
  # 对开发视频跑完整 benchmark
  python tools/benchmark.py --data_dir vedio_exp/ --runs 3

  # 只测单个视频，输出详细日志
  python tools/benchmark.py --data_dir vedio_exp/IMG_2374.MOV --runs 1 --verbose

  # 指定模型权重，测试 GPU 性能
  python tools/benchmark.py --data_dir vedio_exp/ \\
      --detector_weights models/detector.pt \\
      --classifier_weights models/classifier.pt \\
      --device cuda:0 --runs 3 --output_json benchmark_result.json

依赖：opencv-python, numpy, psutil（可选，用于内存监控）
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 确保项目根目录在 sys.path 中
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("benchmark")


# ---------------------------------------------------------------------------
# 计时工具
# ---------------------------------------------------------------------------

@contextmanager
def _timer(label: str, results: Dict[str, List[float]]) -> Generator[None, None, None]:
    """
    上下文管理器：对代码块计时并将结果追加到 results 字典。

    Parameters
    ----------
    label : str
        计时器标签（用于字典 key）。
    results : Dict[str, List[float]]
        结果字典，每次计时的耗时（秒）会 append 到 results[label]。

    用法
    ----
    >>> results = {}
    >>> with _timer("检测", results):
    ...     run_detector(frame)
    >>> print(results["检测"])  # [0.123, 0.118, ...]
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        results.setdefault(label, []).append(elapsed)


def _stats(values: List[float]) -> Dict[str, float]:
    """
    计算列表的统计摘要（平均值、最小值、最大值、标准差）。

    Parameters
    ----------
    values : List[float]

    Returns
    -------
    Dict[str, float]
    """
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0, "n": 0}
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "std": float(arr.std()),
        "n": len(values),
    }


# ---------------------------------------------------------------------------
# GPU 监控工具（可选）
# ---------------------------------------------------------------------------

def _get_gpu_memory_mb() -> Optional[float]:
    """
    获取当前 CUDA 显存占用（MB）。

    Returns
    -------
    float | None : 显存占用量（MB）；若无 GPU 或 PyTorch 未安装则返回 None。
    """
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024 ** 2
            reserved = torch.cuda.memory_reserved() / 1024 ** 2
            return reserved  # 返回保留显存（比已分配显存更接近实际占用）
    except ImportError:
        pass
    return None


def _get_cpu_memory_mb() -> Optional[float]:
    """
    获取当前进程的 RSS 内存占用（MB）。

    Returns
    -------
    float | None : 内存占用量（MB）；若 psutil 未安装则返回 None。
    """
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 ** 2
    except ImportError:
        return None


def _get_device_info() -> Dict[str, str]:
    """
    收集运行环境信息（GPU 型号、CUDA 版本、CPU 核数等）。

    Returns
    -------
    Dict[str, str]
    """
    info: Dict[str, str] = {}

    # Python 版本
    info["python"] = sys.version.split()[0]

    # OpenCV 版本
    info["opencv"] = cv2.__version__

    # NumPy 版本
    info["numpy"] = np.__version__

    # PyTorch 和 CUDA
    try:
        import torch
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["cuda"] = torch.version.cuda or "unknown"
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = str(torch.cuda.device_count())
            props = torch.cuda.get_device_properties(0)
            info["gpu_vram_gb"] = f"{props.total_memory / 1024**3:.1f}GB"
        else:
            info["cuda"] = "unavailable"
            info["gpu_name"] = "N/A"
    except ImportError:
        info["torch"] = "not installed"
        info["cuda"] = "N/A"
        info["gpu_name"] = "N/A"

    # CPU
    try:
        import psutil
        info["cpu_cores"] = str(psutil.cpu_count(logical=False))
        info["cpu_threads"] = str(psutil.cpu_count(logical=True))
        info["ram_gb"] = f"{psutil.virtual_memory().total / 1024**3:.1f}GB"
    except ImportError:
        info["cpu_cores"] = str(os.cpu_count() or "unknown")

    return info


# ---------------------------------------------------------------------------
# 数据类：各模块计时结果
# ---------------------------------------------------------------------------

@dataclass
class ModuleTimingResult:
    """单个模块的计时统计结果。"""
    module_name: str
    times_seconds: List[float] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return float(np.mean(self.times_seconds)) if self.times_seconds else 0.0

    @property
    def min(self) -> float:
        return float(np.min(self.times_seconds)) if self.times_seconds else 0.0

    @property
    def max(self) -> float:
        return float(np.max(self.times_seconds)) if self.times_seconds else 0.0

    @property
    def std(self) -> float:
        return float(np.std(self.times_seconds)) if self.times_seconds else 0.0

    @property
    def n(self) -> int:
        return len(self.times_seconds)

    def to_dict(self) -> Dict:
        return {
            "module": self.module_name,
            "mean_s": round(self.mean, 4),
            "min_s": round(self.min, 4),
            "max_s": round(self.max, 4),
            "std_s": round(self.std, 4),
            "n_runs": self.n,
        }


@dataclass
class VideoBenchmarkResult:
    """单段视频的完整 benchmark 结果。"""
    video_name: str
    video_duration_s: float          # 视频实际时长（秒）
    video_fps: float                 # 视频帧率
    video_frame_count: int           # 总帧数
    video_resolution: str            # 分辨率字符串（如 "3840x2160"）

    # 各模块耗时（平均值，秒）
    keyframe_extraction_s: float = 0.0
    frame_reading_s: float = 0.0
    detection_s: float = 0.0
    registration_s: float = 0.0
    dedup_s: float = 0.0
    classification_s: float = 0.0
    mask_generation_s: float = 0.0
    total_s: float = 0.0

    # 派生指标
    n_keyframes: int = 0
    n_detections: int = 0
    n_clusters: int = 0
    speedup_ratio: float = 0.0       # video_duration / total_s（>1 表示快于实时）
    detection_fps: float = 0.0       # 检测 FPS（关键帧 / 检测耗时）

    # 内存占用
    peak_gpu_memory_mb: float = 0.0
    peak_cpu_memory_mb: float = 0.0

    # 是否满足作业速度要求
    meets_requirement: bool = False  # total_s ≤ video_duration_s

    def compute_derived(self) -> None:
        """计算所有派生指标。"""
        if self.total_s > 0 and self.video_duration_s > 0:
            self.speedup_ratio = self.video_duration_s / self.total_s
        self.meets_requirement = self.total_s <= self.video_duration_s
        if self.detection_s > 0 and self.n_keyframes > 0:
            self.detection_fps = self.n_keyframes / self.detection_s

    def to_dict(self) -> Dict:
        d = {
            "video_name": self.video_name,
            "video_duration_s": round(self.video_duration_s, 2),
            "video_fps": round(self.video_fps, 2),
            "video_frame_count": self.video_frame_count,
            "video_resolution": self.video_resolution,
            "n_keyframes": self.n_keyframes,
            "n_detections": self.n_detections,
            "n_clusters": self.n_clusters,
            "timings": {
                "keyframe_extraction_s": round(self.keyframe_extraction_s, 4),
                "frame_reading_s": round(self.frame_reading_s, 4),
                "detection_s": round(self.detection_s, 4),
                "registration_s": round(self.registration_s, 4),
                "dedup_s": round(self.dedup_s, 4),
                "classification_s": round(self.classification_s, 4),
                "mask_generation_s": round(self.mask_generation_s, 4),
                "total_s": round(self.total_s, 4),
            },
            "metrics": {
                "speedup_ratio": round(self.speedup_ratio, 3),
                "detection_fps": round(self.detection_fps, 2),
                "meets_requirement": self.meets_requirement,
            },
            "memory": {
                "peak_gpu_mb": round(self.peak_gpu_memory_mb, 1),
                "peak_cpu_mb": round(self.peak_cpu_memory_mb, 1),
            },
        }
        return d


@dataclass
class BenchmarkReport:
    """完整 benchmark 报告（含所有视频的结果和系统信息）。"""
    device_info: Dict[str, str]
    video_results: List[VideoBenchmarkResult]
    detector_mode: str = "unknown"
    classifier_mode: str = "unknown"
    total_elapsed_s: float = 0.0
    n_runs: int = 1

    def to_dict(self) -> Dict:
        return {
            "summary": {
                "total_videos": len(self.video_results),
                "total_elapsed_s": round(self.total_elapsed_s, 4),
                "n_runs": self.n_runs,
                "detector_mode": self.detector_mode,
                "classifier_mode": self.classifier_mode,
                "all_meet_requirement": all(r.meets_requirement for r in self.video_results),
            },
            "device": self.device_info,
            "videos": [r.to_dict() for r in self.video_results],
        }

    def save_json(self, path: str | Path) -> None:
        """将报告保存为 JSON 文件。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info("Benchmark 报告已保存: %s", path)

    def save_markdown(self, path: str | Path) -> None:
        """将报告保存为 Markdown 格式（报告素材）。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        lines = ["# Benchmark 报告\n"]

        # 系统信息
        lines.append("## 系统环境\n")
        lines.append("| 项目 | 值 |")
        lines.append("|------|-----|")
        for k, v in self.device_info.items():
            lines.append(f"| {k} | {v} |")
        lines.append("")

        # 运行配置
        lines.append("## 运行配置\n")
        lines.append(f"- 检测器模式: **{self.detector_mode}**")
        lines.append(f"- 分类器模式: **{self.classifier_mode}**")
        lines.append(f"- 重复次数: {self.n_runs}")
        lines.append("")

        # 各模块耗时（按视频分组）
        lines.append("## 各模块耗时（秒）\n")
        header = (
            "| 视频 | 分辨率 | 视频时长 | 关键帧 | 检测 | 配准 | 去重 | 分类 | "
            "Mask | **总计** | 速比 | ✅ |"
        )
        sep = "|" + "|".join(["---"] * 12) + "|"
        lines.append(header)
        lines.append(sep)

        for r in self.video_results:
            ok = "✅" if r.meets_requirement else "❌"
            lines.append(
                f"| {r.video_name} | {r.video_resolution} | {r.video_duration_s:.1f}s"
                f" | {r.keyframe_extraction_s:.3f}"
                f" | {r.detection_s:.3f}"
                f" | {r.registration_s:.3f}"
                f" | {r.dedup_s:.3f}"
                f" | {r.classification_s:.3f}"
                f" | {r.mask_generation_s:.3f}"
                f" | **{r.total_s:.3f}**"
                f" | {r.speedup_ratio:.2f}x"
                f" | {ok} |"
            )
        lines.append("")

        # 速度评估
        lines.append("## 速度评估\n")
        lines.append(
            "作业要求：10秒视频处理时间 ≤ 10秒（即速比 ≥ 1.0x）"
        )
        lines.append("")
        for r in self.video_results:
            status = "✅ 达标" if r.meets_requirement else "❌ 超时"
            lines.append(
                f"- **{r.video_name}**: {r.video_duration_s:.1f}s 视频 → "
                f"处理 {r.total_s:.2f}s ({r.speedup_ratio:.2f}x)  {status}"
            )
        lines.append("")

        path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Markdown 报告已保存: %s", path)


# ---------------------------------------------------------------------------
# 核心 Benchmark 函数
# ---------------------------------------------------------------------------

def benchmark_video(
    video_path: Path,
    pipeline,
    n_runs: int = 1,
) -> VideoBenchmarkResult:
    """
    对单段视频进行 n_runs 次处理，计算平均各模块耗时。

    Parameters
    ----------
    video_path : Path
        视频文件路径。
    pipeline : VideoPipeline
        已初始化的处理流水线实例。
    n_runs : int
        重复运行次数（取平均值消除偶发延迟）。

    Returns
    -------
    VideoBenchmarkResult
    """
    from utils.video_io import VideoReader

    # 读取视频元数据
    with VideoReader(video_path) as reader:
        meta = reader.meta
        video_name = video_path.stem
        video_duration = meta.duration
        video_fps = meta.fps
        frame_count = meta.frame_count
        resolution = f"{meta.width}x{meta.height}"

    logger.info(
        "Benchmark: %s  [%s, %.1fs, %d帧, %.0ffps]",
        video_name, resolution, video_duration, frame_count, video_fps,
    )

    # 各模块耗时列表（运行 n_runs 次）
    run_results = []

    for run_idx in range(n_runs):
        logger.info("  Run %d/%d ...", run_idx + 1, n_runs)

        # 记录运行前内存
        gpu_mem_before = _get_gpu_memory_mb() or 0.0
        cpu_mem_before = _get_cpu_memory_mb() or 0.0

        t_run_start = time.perf_counter()

        # 调用流水线处理（pipeline 内部已分段计时）
        # 我们通过读取 result.processing_time 获取总时间，
        # 但各模块的细粒度计时需要 monkey-patch 或直接在此手动拆分。
        # 当前实现：运行完整 pipeline，读取总耗时 + 关键数据。
        result = pipeline.process_video(video_path)

        t_run_total = time.perf_counter() - t_run_start

        # 记录运行后内存
        gpu_mem_after = _get_gpu_memory_mb() or 0.0
        cpu_mem_after = _get_cpu_memory_mb() or 0.0

        run_data = {
            "total": t_run_total,
            "n_clusters": len(result.clusters),
            "peak_gpu_mb": max(gpu_mem_before, gpu_mem_after),
            "peak_cpu_mb": max(cpu_mem_before, cpu_mem_after),
        }
        run_results.append(run_data)

        logger.info(
            "  Run %d: total=%.3fs, clusters=%d, GPU=%.0fMB, CPU=%.0fMB",
            run_idx + 1,
            t_run_total,
            run_data["n_clusters"],
            run_data["peak_gpu_mb"],
            run_data["peak_cpu_mb"],
        )

    # 汇总（取各运行的平均值）
    avg_total = float(np.mean([r["total"] for r in run_results]))
    avg_clusters = int(np.mean([r["n_clusters"] for r in run_results]))
    peak_gpu = float(np.max([r["peak_gpu_mb"] for r in run_results]))
    peak_cpu = float(np.max([r["peak_cpu_mb"] for r in run_results]))

    bm = VideoBenchmarkResult(
        video_name=video_name,
        video_duration_s=video_duration,
        video_fps=video_fps,
        video_frame_count=frame_count,
        video_resolution=resolution,
        total_s=avg_total,
        n_clusters=avg_clusters,
        peak_gpu_memory_mb=peak_gpu,
        peak_cpu_memory_mb=peak_cpu,
    )
    bm.compute_derived()
    return bm


def benchmark_modules_detailed(
    video_path: Path,
    pipeline,
) -> VideoBenchmarkResult:
    """
    对单段视频进行细粒度模块计时（每个步骤单独计时）。

    此函数通过将 pipeline 拆开、逐步调用各模块来实现细粒度计时，
    比 benchmark_video() 提供更详细的各模块耗时数据。

    Parameters
    ----------
    video_path : Path
        视频文件路径。
    pipeline : VideoPipeline
        已初始化的处理流水线实例。

    Returns
    -------
    VideoBenchmarkResult : 含各模块耗时的详细结果。
    """
    from utils.video_io import VideoReader, get_video_name
    from pipeline import _extract_keyframes_motion, extract_keyframes_uniform
    from interfaces import Detection, Registration, Cluster

    video_name = get_video_name(video_path)
    timings: Dict[str, float] = {}

    logger.info("细粒度 Benchmark: %s", video_name)

    # ---- 读取元数据 ----
    with VideoReader(video_path) as reader:
        meta = reader.meta
        video_duration = meta.duration
        video_fps = meta.fps
        frame_count = meta.frame_count
        resolution = f"{meta.width}x{meta.height}"
        full_res_scale = meta.low_res_scale

        # Step 1: 关键帧提取
        t0 = time.perf_counter()
        try:
            keyframe_ids = _extract_keyframes_motion(reader)
        except Exception:
            keyframe_ids = extract_keyframes_uniform(reader, target_count=30)
        timings["keyframe_extraction"] = time.perf_counter() - t0

        n_keyframes = len(keyframe_ids)
        logger.info("  关键帧提取: %.3fs  (%d 帧)", timings["keyframe_extraction"], n_keyframes)

        # Step 2: 读取关键帧
        t0 = time.perf_counter()
        kf_images_hr = []
        kf_images_lr = []
        valid_kf_ids = []

        for fid, frame_hr, frame_lr in reader.iter_frames_at(keyframe_ids, yield_low_res=True):
            if frame_hr is not None and frame_lr is not None:
                kf_images_hr.append(frame_hr)
                kf_images_lr.append(frame_lr)
                valid_kf_ids.append(fid)

        timings["frame_reading"] = time.perf_counter() - t0
        logger.info("  帧读取: %.3fs  (%d 帧)", timings["frame_reading"], len(valid_kf_ids))

        # Step 3: 检测
        t0 = time.perf_counter()
        all_detections = pipeline.detector.detect_batch(kf_images_hr, valid_kf_ids)
        timings["detection"] = time.perf_counter() - t0
        n_detections = sum(len(d) for d in all_detections)
        logger.info("  检测: %.3fs  (%d 个螺丝)", timings["detection"], n_detections)

        # Step 4: 配准
        t0 = time.perf_counter()
        pipeline.registrar.reset_stats()
        all_registrations = pipeline.registrar.register_sequence(
            keyframe_images=kf_images_lr,
            keyframe_ids=valid_kf_ids,
            full_res_scales=[full_res_scale] * len(valid_kf_ids),
        )
        timings["registration"] = time.perf_counter() - t0
        valid_reg = sum(1 for r in all_registrations if r.valid)
        logger.info(
            "  配准: %.3fs  (有效 %d/%d 帧)",
            timings["registration"], valid_reg, len(all_registrations),
        )

        # Step 5: 去重聚类
        t0 = time.perf_counter()
        clusters = pipeline.deduper.run(all_detections, all_registrations)
        timings["dedup"] = time.perf_counter() - t0
        logger.info("  去重聚类: %.3fs  (%d 个 Cluster)", timings["dedup"], len(clusters))

        # Step 6: 分类
        t0 = time.perf_counter()
        classified, counts = pipeline.classifier.classify_and_count(clusters)
        timings["classification"] = time.perf_counter() - t0
        logger.info("  分类: %.3fs  计数=%s", timings["classification"], counts)

        # Step 7: 读取中间帧 + 生成 mask
        t0 = time.perf_counter()
        mid_frame = reader.read_frame(meta.mid_frame_id, low_res=False)
        if mid_frame is not None:
            _mask = pipeline.visualizer.draw_clusters(mid_frame, classified)
        timings["mask_generation"] = time.perf_counter() - t0
        logger.info("  Mask 生成: %.3fs", timings["mask_generation"])

    # ---- 汇总 ----
    total = sum(timings.values())
    bm = VideoBenchmarkResult(
        video_name=video_name,
        video_duration_s=video_duration,
        video_fps=video_fps,
        video_frame_count=frame_count,
        video_resolution=resolution,
        keyframe_extraction_s=timings.get("keyframe_extraction", 0.0),
        frame_reading_s=timings.get("frame_reading", 0.0),
        detection_s=timings.get("detection", 0.0),
        registration_s=timings.get("registration", 0.0),
        dedup_s=timings.get("dedup", 0.0),
        classification_s=timings.get("classification", 0.0),
        mask_generation_s=timings.get("mask_generation", 0.0),
        total_s=total,
        n_keyframes=n_keyframes,
        n_detections=n_detections,
        n_clusters=len(clusters),
        peak_gpu_memory_mb=_get_gpu_memory_mb() or 0.0,
        peak_cpu_memory_mb=_get_cpu_memory_mb() or 0.0,
    )
    bm.compute_derived()
    return bm


# ---------------------------------------------------------------------------
# 报告打印
# ---------------------------------------------------------------------------

def _print_report(report: BenchmarkReport) -> None:
    """
    在控制台打印格式化的 benchmark 报告。

    Parameters
    ----------
    report : BenchmarkReport
    """
    print("\n" + "=" * 80)
    print("BENCHMARK 报告".center(80))
    print("=" * 80)

    # 系统信息
    print("\n【系统环境】")
    for k, v in report.device_info.items():
        print(f"  {k:20s}: {v}")

    print(f"\n  检测器模式   : {report.detector_mode}")
    print(f"  分类器模式   : {report.classifier_mode}")
    print(f"  重复次数     : {report.n_runs}")

    # 各模块耗时
    print("\n【各模块耗时（秒）】")
    header = (
        f"  {'视频':18s}  {'分辨率':12s}  {'时长':>6s}  "
        f"{'KF提取':>6s}  {'检测':>6s}  {'配准':>6s}  "
        f"{'去重':>6s}  {'分类':>6s}  {'Mask':>6s}  "
        f"{'总计':>7s}  {'速比':>5s}  {'达标':>4s}"
    )
    print(header)
    print("  " + "-" * 100)

    for r in report.video_results:
        ok = "✅" if r.meets_requirement else "❌"
        print(
            f"  {r.video_name:18s}  {r.video_resolution:12s}  "
            f"{r.video_duration_s:>6.1f}s  "
            f"{r.keyframe_extraction_s:>6.3f}  "
            f"{r.detection_s:>6.3f}  "
            f"{r.registration_s:>6.3f}  "
            f"{r.dedup_s:>6.3f}  "
            f"{r.classification_s:>6.3f}  "
            f"{r.mask_generation_s:>6.3f}  "
            f"{r.total_s:>7.3f}s  "
            f"{r.speedup_ratio:>4.2f}x  "
            f"{ok}"
        )

    # 速度要求评估
    print("\n【作业速度要求评估】")
    print("  要求：10s 视频处理时间 ≤ 10s（速比 ≥ 1.0x），每超 5s 扣 1 分")
    all_ok = all(r.meets_requirement for r in report.video_results)
    for r in report.video_results:
        if r.meets_requirement:
            speed_score = 3
            status = "✅ 满分"
        else:
            overrun = r.total_s - r.video_duration_s
            deducted = min(3, int(overrun / 5) + 1)
            speed_score = max(0, 3 - deducted)
            status = f"❌ 超时 {overrun:.1f}s，扣 {deducted} 分（得 {speed_score}/3 分）"
        print(f"  {r.video_name}: {r.video_duration_s:.1f}s → {r.total_s:.2f}s  {status}")

    print(f"\n  综合评价: {'✅ 所有视频达标！' if all_ok else '❌ 部分视频超时，需优化速度'}")

    # 瓶颈分析
    if report.video_results:
        print("\n【瓶颈分析（平均耗时最高的模块）】")
        avg_timings = {}
        for r in report.video_results:
            for mod, val in [
                ("关键帧提取", r.keyframe_extraction_s),
                ("帧读取", r.frame_reading_s),
                ("螺丝检测", r.detection_s),
                ("几何配准", r.registration_s),
                ("去重聚类", r.dedup_s),
                ("螺丝分类", r.classification_s),
                ("Mask生成", r.mask_generation_s),
            ]:
                avg_timings.setdefault(mod, []).append(val)

        sorted_mods = sorted(
            avg_timings.items(),
            key=lambda x: np.mean(x[1]),
            reverse=True,
        )
        total_avg = sum(np.mean(v) for v in avg_timings.values())
        for mod, vals in sorted_mods:
            avg_val = np.mean(vals)
            pct = avg_val / max(total_avg, 1e-9) * 100
            bar = "█" * max(1, int(pct / 5))
            print(f"  {mod:12s}: {avg_val:6.3f}s  ({pct:5.1f}%)  {bar}")

    print("\n" + "=" * 80)
    print(f"总耗时（含所有视频）: {report.total_elapsed_s:.2f}s")
    print("=" * 80 + "\n")


# ---------------------------------------------------------------------------
# 命令行参数
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        prog="benchmark.py",
        description="视频螺丝计数流水线速度 Benchmark 工具 (Owner: D)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 对开发视频（文件夹）跑 3 次 benchmark
  python tools/benchmark.py --data_dir vedio_exp/ --runs 3

  # 对单个视频做细粒度模块计时
  python tools/benchmark.py --data_dir vedio_exp/IMG_2374.MOV --detailed

  # 使用正式模型权重，在 GPU 上测速，并保存报告
  python tools/benchmark.py --data_dir vedio_exp/ \\
      --detector_weights models/detector.pt \\
      --classifier_weights models/classifier.pt \\
      --device cuda:0 --runs 3 \\
      --output_json reports/benchmark.json \\
      --output_md reports/benchmark.md
        """,
    )

    parser.add_argument(
        "--data_dir", "-d",
        required=True,
        help="视频文件或包含视频文件的文件夹路径。",
    )
    parser.add_argument(
        "--runs", "-n",
        type=int,
        default=1,
        help="每个视频的重复运行次数（默认 1，建议 3 以减少偶发延迟影响）。",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        default=False,
        help="进行细粒度模块计时（每个步骤单独计时，更详细但仅运行 1 次）。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="推理设备（cuda:0 / cpu / 空字符串=自动选择）。",
    )
    parser.add_argument(
        "--no_fp16",
        action="store_true",
        default=False,
        help="禁用 FP16 推理。",
    )
    parser.add_argument(
        "--keyframe_strategy",
        type=str,
        default="motion",
        choices=["motion", "uniform"],
        help="关键帧提取策略（默认 motion）。",
    )
    parser.add_argument(
        "--detector_weights",
        type=str,
        default=None,
        help="YOLO 检测器权重路径。",
    )
    parser.add_argument(
        "--classifier_weights",
        type=str,
        default=None,
        help="分类器权重路径。",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="将 benchmark 报告保存为 JSON 文件的路径。",
    )
    parser.add_argument(
        "--output_md",
        type=str,
        default=None,
        help="将 benchmark 报告保存为 Markdown 文件的路径（报告素材）。",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="输出详细调试日志。",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main() -> int:
    """
    Benchmark 主函数。

    Returns
    -------
    int : 退出码（0=成功）。
    """
    args = _parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ---- 收集视频文件 ----
    from utils.video_io import list_videos

    input_path = Path(args.data_dir)
    if input_path.is_file():
        from utils.video_io import VIDEO_EXTENSIONS
        if input_path.suffix.lower() not in VIDEO_EXTENSIONS:
            logger.error("文件不是支持的视频格式: %s", input_path)
            return 1
        videos = [input_path]
    elif input_path.is_dir():
        videos = list_videos(input_path)
        if not videos:
            logger.error("在 '%s' 中未找到视频文件。", input_path)
            return 1
    else:
        logger.error("路径不存在: %s", input_path)
        return 1

    logger.info("找到 %d 段视频，开始 Benchmark...", len(videos))

    # ---- 收集系统信息 ----
    device_info = _get_device_info()

    # ---- 初始化流水线 ----
    from pipeline import VideoPipeline

    detector_weights = Path(args.detector_weights) if args.detector_weights else None
    classifier_weights = Path(args.classifier_weights) if args.classifier_weights else None

    pipeline = VideoPipeline(
        detector_weights=detector_weights,
        classifier_weights=classifier_weights,
        use_fp16=not args.no_fp16,
        device=args.device,
        keyframe_strategy=args.keyframe_strategy,
    )

    detector_mode = "YOLO" if pipeline.detector.is_yolo_mode else "兜底(OpenCV)"
    classifier_mode = "PyTorch" if pipeline.classifier.is_torch_mode else "兜底(随机)"

    # ---- 执行 Benchmark ----
    t_global_start = time.perf_counter()
    video_results: List[VideoBenchmarkResult] = []

    for i, video_path in enumerate(videos):
        logger.info("\n[%d/%d] Benchmark: %s", i + 1, len(videos), video_path.name)
        try:
            if args.detailed:
                bm = benchmark_modules_detailed(video_path, pipeline)
            else:
                bm = benchmark_video(video_path, pipeline, n_runs=args.runs)
            video_results.append(bm)
        except Exception as e:
            logger.error("Benchmark 失败 (%s): %s", video_path.name, e, exc_info=args.verbose)

    total_elapsed = time.perf_counter() - t_global_start

    # ---- 构建报告 ----
    report = BenchmarkReport(
        device_info=device_info,
        video_results=video_results,
        detector_mode=detector_mode,
        classifier_mode=classifier_mode,
        total_elapsed_s=total_elapsed,
        n_runs=args.runs if not args.detailed else 1,
    )

    # ---- 打印报告 ----
    _print_report(report)

    # ---- 保存报告 ----
    if args.output_json:
        report.save_json(args.output_json)
    if args.output_md:
        report.save_markdown(args.output_md)

    # ---- 若无指定输出路径，自动保存到 reports/ 目录 ----
    if not args.output_json and not args.output_md:
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        reports_dir = _PROJECT_ROOT / "reports"
        reports_dir.mkdir(exist_ok=True)
        report.save_json(reports_dir / f"benchmark_{ts}.json")
        report.save_markdown(reports_dir / f"benchmark_{ts}.md")
        logger.info("报告已自动保存至 %s/", reports_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
