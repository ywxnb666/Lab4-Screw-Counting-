#!/usr/bin/env python3
"""
run.py - 视频螺丝计数主入口
Owner: D（工程封装）

作业规范接口：
    python run.py --data_dir /path/to/test_videos_folder \\
                  --output_path ./result.npy \\
                  --output_time_path ./time.txt \\
                  --mask_output_path ./mask_folder/

输出格式：
    result.npy   : 通过 numpy.load(path, allow_pickle=True).item() 加载后为字典
                   {'video_name': [n_type1, n_type2, n_type3, n_type4, n_type5], ...}
    time.txt     : 单行浮点数，表示所有视频处理总耗时（秒）
    mask_folder/ : 每段视频一张 {video_name}_mask.png 掩膜叠加图
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# 确保项目根目录在 sys.path 中（支持从任意工作目录调用）
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent.resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# 日志配置（在其他模块导入之前配置，以捕获所有初始化日志）
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool = False) -> None:
    """
    配置日志系统。

    Parameters
    ----------
    verbose : bool
        True: 输出 DEBUG 级别日志；False: 仅输出 INFO 及以上。
    """
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%H:%M:%S"

    # 配置根 logger
    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # 降低第三方库的日志级别（避免过多噪声）
    for noisy_lib in ("ultralytics", "sahi", "PIL", "urllib3", "torch"):
        logging.getLogger(noisy_lib).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# 命令行参数解析
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    解析命令行参数。

    必须与作业规范的接口完全一致：
    --data_dir         : 包含视频文件的文件夹路径
    --output_path      : result.npy 输出路径
    --output_time_path : time.txt 输出路径
    --mask_output_path : 掩膜图像输出文件夹路径

    附加参数（方便调试）：
    --verbose          : 输出详细日志
    --device           : 指定推理设备（cuda:0 / cpu）
    --no_fp16          : 禁用 FP16 推理
    --keyframe_strategy: 关键帧提取策略
    --dist_thresh      : 去重聚类距离阈值（像素）
    --min_observations : Cluster 最少观测次数
    --detector_weights : 检测器权重路径（覆盖默认路径）
    --classifier_weights: 分类器权重路径（覆盖默认路径）
    --dry_run          : 仅列出视频，不执行处理
    """
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="视频螺丝计数算法 - Lab4 团队作业",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python run.py --data_dir ./vedio_exp \\
                --output_path ./result.npy \\
                --output_time_path ./time.txt \\
                --mask_output_path ./masks/

  # 使用 CPU 推理（无 GPU 时）：
  python run.py --data_dir ./vedio_exp --device cpu --no_fp16 \\
                --output_path ./result.npy \\
                --output_time_path ./time.txt \\
                --mask_output_path ./masks/

  # 调试模式（输出详细日志）：
  python run.py --data_dir ./vedio_exp --verbose \\
                --output_path ./result.npy \\
                --output_time_path ./time.txt \\
                --mask_output_path ./masks/
        """,
    )

    # ---- 必需参数（作业规范接口）----
    required = parser.add_argument_group("必需参数（作业规范接口）")
    required.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="包含测试视频文件的文件夹路径（支持 .mp4/.mov/.avi 等格式）。",
    )
    required.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="result.npy 的输出路径（含文件名，例如 ./result.npy）。",
    )
    required.add_argument(
        "--output_time_path",
        type=str,
        required=True,
        help="time.txt 的输出路径（含文件名，例如 ./time.txt）。",
    )
    required.add_argument(
        "--mask_output_path",
        type=str,
        required=True,
        help="掩膜图像输出文件夹路径（每段视频输出一张 {video_name}_mask.png）。",
    )

    # ---- 可选参数（调试与调优）----
    optional = parser.add_argument_group("可选参数（调试与调优）")
    optional.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="输出 DEBUG 级别详细日志（默认关闭）。",
    )
    optional.add_argument(
        "--device",
        type=str,
        default="",
        help="推理设备，例如 'cuda:0' 或 'cpu'（默认自动选择）。",
    )
    optional.add_argument(
        "--no_fp16",
        action="store_true",
        default=False,
        help="禁用 FP16 半精度推理（CPU 推理时会自动禁用，此处用于强制禁用）。",
    )
    optional.add_argument(
        "--keyframe_strategy",
        type=str,
        default="motion",
        choices=["motion", "uniform"],
        help="关键帧提取策略：motion（基于运动位移，推荐）或 uniform（均匀采样）。默认：motion。",
    )
    optional.add_argument(
        "--dist_thresh",
        type=float,
        default=40.0,
        help="去重聚类距离阈值（参考坐标系像素数，默认 40.0）。",
    )
    optional.add_argument(
        "--min_observations",
        type=int,
        default=1,
        help="Cluster 最少观测次数，低于此值的 Cluster 将被过滤（默认 1，即不过滤）。",
    )
    optional.add_argument(
        "--detector_weights",
        type=str,
        default=None,
        help="YOLO 检测器权重路径（覆盖默认 models/detector.pt）。",
    )
    optional.add_argument(
        "--classifier_weights",
        type=str,
        default=None,
        help="分类器权重路径（覆盖默认 models/classifier.pt）。",
    )
    optional.add_argument(
        "--dry_run",
        action="store_true",
        default=False,
        help="仅列出找到的视频文件，不执行任何处理（用于验证路径配置）。",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# 路径验证与预检查
# ---------------------------------------------------------------------------

def _validate_args(args: argparse.Namespace, logger: logging.Logger) -> bool:
    """
    验证命令行参数的合法性。

    Parameters
    ----------
    args : argparse.Namespace
        解析后的参数。
    logger : logging.Logger
        日志器。

    Returns
    -------
    bool : 所有参数均合法时返回 True；否则返回 False（并记录错误）。
    """
    ok = True

    # 验证 data_dir
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error("--data_dir 路径不存在: %s", data_dir)
        ok = False
    elif not data_dir.is_dir():
        logger.error("--data_dir 不是文件夹: %s", data_dir)
        ok = False

    # 验证 output_path 的父目录
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not output_path.parent.exists():
        logger.error("--output_path 的父目录无法创建: %s", output_path.parent)
        ok = False

    # 验证 output_time_path 的父目录
    time_path = Path(args.output_time_path)
    time_path.parent.mkdir(parents=True, exist_ok=True)
    if not time_path.parent.exists():
        logger.error("--output_time_path 的父目录无法创建: %s", time_path.parent)
        ok = False

    # 验证（并创建）mask_output_path
    mask_dir = Path(args.mask_output_path)
    mask_dir.mkdir(parents=True, exist_ok=True)
    if not mask_dir.exists():
        logger.error("--mask_output_path 无法创建: %s", mask_dir)
        ok = False

    # 验证自定义权重路径（若提供）
    if args.detector_weights is not None:
        dw = Path(args.detector_weights)
        if not dw.exists():
            logger.warning(
                "--detector_weights 指定的文件不存在: %s\n"
                "  将回退到默认路径 models/detector.pt。",
                dw,
            )
    if args.classifier_weights is not None:
        cw = Path(args.classifier_weights)
        if not cw.exists():
            logger.warning(
                "--classifier_weights 指定的文件不存在: %s\n"
                "  将回退到默认路径 models/classifier.pt。",
                cw,
            )

    # 验证 dist_thresh
    if args.dist_thresh <= 0:
        logger.error("--dist_thresh 必须为正数，当前值: %f", args.dist_thresh)
        ok = False

    # 验证 min_observations
    if args.min_observations < 1:
        logger.error("--min_observations 必须 ≥ 1，当前值: %d", args.min_observations)
        ok = False

    return ok


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    """
    主函数入口。

    Parameters
    ----------
    argv : List[str] | None
        命令行参数列表；None 时使用 sys.argv[1:]。

    Returns
    -------
    int : 退出码（0=成功，1=错误）。
    """
    # ---- 解析参数 ----
    args = _parse_args(argv)

    # ---- 配置日志 ----
    _setup_logging(verbose=args.verbose)
    logger = logging.getLogger("run")

    # ---- 打印启动横幅 ----
    logger.info("=" * 60)
    logger.info("视频螺丝计数系统  Lab4 团队作业")
    logger.info("=" * 60)
    logger.info("参数配置:")
    logger.info("  data_dir          : %s", args.data_dir)
    logger.info("  output_path       : %s", args.output_path)
    logger.info("  output_time_path  : %s", args.output_time_path)
    logger.info("  mask_output_path  : %s", args.mask_output_path)
    logger.info("  device            : %s", args.device or "(自动选择)")
    logger.info("  fp16              : %s", not args.no_fp16)
    logger.info("  keyframe_strategy : %s", args.keyframe_strategy)
    logger.info("  dist_thresh       : %.1f", args.dist_thresh)
    logger.info("  min_observations  : %d", args.min_observations)

    # ---- 验证参数 ----
    if not _validate_args(args, logger):
        logger.error("参数验证失败，请检查上方错误信息后重试。")
        return 1

    # ---- 收集视频文件列表 ----
    from utils.video_io import list_videos, get_video_name

    data_dir = Path(args.data_dir)
    videos = list_videos(data_dir)

    if not videos:
        logger.error("在 '%s' 中未找到任何视频文件。", data_dir)
        logger.error(
            "支持的格式: .mp4 .mov .avi .mkv .m4v .wmv .flv .webm"
        )
        return 1

    logger.info("找到 %d 段视频:", len(videos))
    for i, v in enumerate(videos):
        logger.info("  [%d] %s", i + 1, v.name)

    # ---- dry_run 模式：只列出视频，不处理 ----
    if args.dry_run:
        logger.info("dry_run 模式已启用，不执行处理。退出。")
        return 0

    # ---- 初始化流水线 ----
    from pipeline import VideoPipeline
    from utils.output_formatter import OutputFormatter

    detector_weights = (
        Path(args.detector_weights) if args.detector_weights else None
    )
    classifier_weights = (
        Path(args.classifier_weights) if args.classifier_weights else None
    )

    try:
        pipeline = VideoPipeline(
            detector_weights=detector_weights,
            classifier_weights=classifier_weights,
            use_fp16=not args.no_fp16,
            device=args.device,
            keyframe_strategy=args.keyframe_strategy,
            dist_thresh=args.dist_thresh,
            min_observations=args.min_observations,
        )
    except Exception as e:
        logger.error("流水线初始化失败: %s", e, exc_info=args.verbose)
        return 1

    # ---- 初始化输出封装器 ----
    formatter = OutputFormatter(
        output_path=Path(args.output_path),
        time_path=Path(args.output_time_path),
        mask_dir=Path(args.mask_output_path),
    )

    # ---- 开始计时（包含所有视频的总处理时间）----
    t_global_start = time.perf_counter()

    # ---- 逐视频处理 ----
    results: Dict[str, List[int]] = {}
    masks: Dict[str, object] = {}
    failed_videos: List[str] = []

    for i, video_path in enumerate(videos):
        video_name = get_video_name(video_path)
        logger.info("")
        logger.info("[%d/%d] 正在处理: %s", i + 1, len(videos), video_path.name)

        t_video_start = time.perf_counter()

        try:
            result = pipeline.process_video(video_path)

            t_video_end = time.perf_counter()
            video_elapsed = t_video_end - t_video_start

            results[result.video_name] = result.counts
            mask_img = getattr(result, "_mask_image", None)
            masks[result.video_name] = mask_img

            logger.info(
                "  ✅ 完成: %s  计数=%s  耗时=%.2fs",
                result.video_name, result.counts, video_elapsed,
            )

            # 实时保存 mask（防止后续视频出错导致前面结果丢失）
            if mask_img is not None:
                try:
                    formatter.save_mask(result.video_name, mask_img)
                except Exception as mask_e:
                    logger.warning(
                        "  ⚠️ mask 保存失败 (%s): %s", result.video_name, mask_e
                    )
            else:
                logger.warning(
                    "  ⚠️ 视频 '%s' 的 mask 图像为空，跳过保存。",
                    result.video_name,
                )

        except FileNotFoundError as e:
            logger.error("  ❌ 视频文件不存在: %s", e)
            results[video_name] = [0] * 5
            masks[video_name] = None
            failed_videos.append(video_name)

        except KeyboardInterrupt:
            logger.warning("用户中断！保存已处理结果...")
            # 对未处理的视频补零
            for v in videos[i:]:
                vn = get_video_name(v)
                if vn not in results:
                    results[vn] = [0] * 5
            break

        except Exception as e:
            logger.error(
                "  ❌ 处理视频 '%s' 时发生错误: %s",
                video_name, e,
                exc_info=args.verbose,
            )
            results[video_name] = [0] * 5
            masks[video_name] = None
            failed_videos.append(video_name)

    # ---- 停止计时 ----
    t_global_end = time.perf_counter()
    total_elapsed = t_global_end - t_global_start

    # ---- 保存 result.npy ----
    logger.info("")
    logger.info("=" * 60)
    logger.info("保存输出文件...")

    try:
        formatter.save_result(results)
    except Exception as e:
        logger.error("result.npy 保存失败: %s", e, exc_info=args.verbose)
        return 1

    # ---- 保存 time.txt ----
    try:
        formatter.save_time(total_elapsed)
    except Exception as e:
        logger.error("time.txt 保存失败: %s", e, exc_info=args.verbose)
        return 1

    # ---- 打印结果汇总 ----
    logger.info("")
    logger.info("=" * 60)
    logger.info("处理结果汇总")
    logger.info("=" * 60)
    logger.info(
        "  %-30s  %4s  %4s  %4s  %4s  %4s  %5s",
        "视频名称", "T1", "T2", "T3", "T4", "T5", "总计",
    )
    logger.info("  " + "-" * 60)

    for vname, counts in results.items():
        status = "❌" if vname in failed_videos else "  "
        logger.info(
            "%s %-30s  %4d  %4d  %4d  %4d  %4d  %5d",
            status, vname,
            counts[0], counts[1], counts[2], counts[3], counts[4],
            sum(counts),
        )

    logger.info("=" * 60)
    logger.info("总耗时: %.2f 秒", total_elapsed)
    if len(results) > 0:
        logger.info("平均每视频: %.2f 秒", total_elapsed / len(results))

    if failed_videos:
        logger.warning(
            "⚠️  以下 %d 段视频处理失败（已补零）: %s",
            len(failed_videos), failed_videos,
        )

    # ---- 验证输出 ----
    logger.info("")
    logger.info("验证输出文件...")
    expected_names = list(results.keys())
    ok = formatter.verify_outputs(expected_names)
    if not ok:
        logger.warning("⚠️  部分输出验证失败，请检查上方日志。")

    # ---- 最终状态 ----
    if failed_videos:
        logger.info("")
        logger.info("完成（有失败项）。退出码: 0（已用零值填充失败视频的结果）。")
    else:
        logger.info("")
        logger.info("🎉 所有视频处理完成！退出码: 0")

    return 0


# ---------------------------------------------------------------------------
# 模块兼容性检查（在 import 时执行）
# ---------------------------------------------------------------------------

def _check_dependencies() -> None:
    """
    检查关键依赖是否已安装，并打印友好提示。
    在 main() 之前调用，帮助用户快速定位缺失依赖。
    """
    missing = []
    optional_missing = []

    # 必需依赖
    for pkg, import_name in [
        ("numpy", "numpy"),
        ("opencv-python", "cv2"),
    ]:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg)

    # 可选依赖（缺失时性能降级，但不影响运行）
    for pkg, import_name in [
        ("ultralytics", "ultralytics"),
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("sahi", "sahi"),
        ("scikit-learn", "sklearn"),
    ]:
        try:
            __import__(import_name)
        except ImportError:
            optional_missing.append(pkg)

    if missing:
        print(
            f"\n[ERROR] 缺少必需依赖，请运行:\n"
            f"  pip install {' '.join(missing)}\n",
            file=sys.stderr,
        )
        sys.exit(1)

    if optional_missing:
        print(
            f"\n[WARNING] 以下可选依赖未安装（将使用兜底实现，精度降低）:\n"
            f"  pip install {' '.join(optional_missing)}\n"
        )


# ---------------------------------------------------------------------------
# 脚本入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _check_dependencies()
    sys.exit(main())
