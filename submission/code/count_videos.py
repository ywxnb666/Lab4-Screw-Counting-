#!/usr/bin/env python3
"""
Formal batch screw counting interface.

This script reuses the same pipeline as step4:
1. read video
2. extract keyframes
3. run detector on keyframes
4. register frames into a shared reference coordinate system
5. deduplicate projected detections
6. classify clusters and count screws

Differences from the debug step4 script:
- no intermediate debug images are saved
- supports processing one video or a whole folder of videos
- writes a single txt report with per-video counts and processing time

Examples:
  python count_videos.py --input ../../video_exp --output_txt ./count_results.txt
  python count_videos.py --input ../../video_exp/IMG_2374.MOV --output_txt ./count_results.txt
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import modules.dedup as dedup_module
import modules.registration as registration_module
from modules.classifier import (
    DEFAULT_WEIGHTS as DEFAULT_CLASSIFIER_WEIGHTS,
    ScrewClassifier,
    _classify_cluster_from_detector_votes,
    _cluster_has_detector_multiclass_labels,
)
from modules.dedup import GlobalDedup
from modules.detector import DEFAULT_WEIGHTS, Detector
from modules.registration import FrameRegistration
from pipeline import _extract_keyframes_motion, extract_keyframes_uniform
from utils.video_io import VIDEO_EXTENSIONS, VideoReader


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("count_videos")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Formal batch interface: process one video or a directory of videos and output one txt report."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input video path or directory that contains videos.",
    )
    parser.add_argument(
        "--output_txt",
        type=str,
        default="./count_results.txt",
        help="Output txt path for the final batch report.",
    )
    parser.add_argument(
        "--keyframe_strategy",
        type=str,
        default="uniform",
        choices=["motion", "uniform"],
        help="Keyframe extraction strategy.",
    )
    parser.add_argument(
        "--uniform_count",
        type=int,
        default=30,
        help="Number of keyframes when --keyframe_strategy uniform.",
    )
    parser.add_argument(
        "--detector_weights",
        type=str,
        default="./models/detector.pt",
        help="Detector weights path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Inference device, for example cuda:0 or cpu.",
    )
    parser.add_argument(
        "--no_fp16",
        action="store_true",
        default=False,
        help="Disable FP16 inference.",
    )
    parser.add_argument(
        "--use_sahi",
        action="store_true",
        default=False,
        help="Enable SAHI slicing during detection.",
    )
    parser.add_argument(
        "--feature_type",
        type=str,
        default="ORB",
        choices=["AKAZE", "ORB"],
        help="Registration feature type.",
    )
    parser.add_argument(
        "--anchor_strategy",
        type=str,
        default="first",
        choices=["first", "middle"],
        help="Preferred root-anchor strategy.",
    )
    parser.add_argument(
        "--anchor_count",
        type=int,
        default=10,
        help="Number of anchors used to build the reference coordinate system.",
    )
    parser.add_argument(
        "--inlier_ratio_threshold",
        type=float,
        default=0.25,
        help="Registration validity threshold.",
    )
    parser.add_argument(
        "--min_match_count",
        type=int,
        default=15,
        help="Minimum match count for registration.",
    )
    parser.add_argument(
        "--dist_thresh",
        type=float,
        default=40.0,
        help="Dedup distance threshold in reference coordinates.",
    )
    parser.add_argument(
        "--min_observations",
        type=int,
        default=4,
        help="Minimum observations per cluster.",
    )
    parser.add_argument(
        "--dedup_method",
        type=str,
        default="dbscan",
        choices=["dbscan", "incremental"],
        help="Dedup backend.",
    )
    parser.add_argument(
        "--invalid_reg_fallback",
        type=str,
        default=dedup_module.INVALID_REG_FALLBACK,
        choices=["skip", "identity", "tracker"],
        help="How to handle invalid registrations during dedup.",
    )
    parser.add_argument(
        "--count_mode",
        type=str,
        default="detector_votes",
        choices=["detector_votes", "classifier"],
        help="Count source.",
    )
    parser.add_argument(
        "--classifier_weights",
        type=str,
        default=str(DEFAULT_CLASSIFIER_WEIGHTS),
        help="Classifier weights path, used only when --count_mode classifier.",
    )
    return parser.parse_args()


def _collect_video_paths(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"Unsupported video file: {input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    video_paths = sorted(
        [
            path
            for path in input_path.iterdir()
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        ],
        key=lambda p: p.name.lower(),
    )
    if not video_paths:
        raise FileNotFoundError(f"No supported videos found in directory: {input_path}")
    return video_paths


def _classify_clusters_with_detector_votes(clusters) -> tuple[list, list[int], int]:
    missing_label_clusters = 0
    for cluster in clusters:
        if _cluster_has_detector_multiclass_labels(cluster):
            _classify_cluster_from_detector_votes(cluster)
        else:
            missing_label_clusters += 1
            logger.warning(
                "Cluster #%d has no detector class labels; it will not contribute to counts in detector_votes mode.",
                cluster.cluster_id,
            )
    counts = ScrewClassifier.count_by_type(clusters)
    return clusters, counts, missing_label_clusters


def _process_video(
    video_path: Path,
    args: argparse.Namespace,
    detector: Detector,
) -> dict[str, object]:
    start_time = time.perf_counter()
    logger.info("Processing video: %s", video_path.name)

    registration_module.ANCHOR_STRATEGY = args.anchor_strategy
    registrar = FrameRegistration(
        feature_type=args.feature_type,
        inlier_ratio_threshold=args.inlier_ratio_threshold,
        min_match_count=args.min_match_count,
    )
    deduper = GlobalDedup(
        dist_thresh=args.dist_thresh,
        min_observations=args.min_observations,
        use_dbscan=args.dedup_method == "dbscan",
        invalid_reg_fallback=args.invalid_reg_fallback,
    )

    try:
        with VideoReader(video_path) as reader:
            t_kf0 = time.perf_counter()
            if args.keyframe_strategy == "motion":
                keyframe_ids = _extract_keyframes_motion(reader)
                actual_strategy = "motion"
                if not keyframe_ids:
                    logger.warning(
                        "Motion keyframe extraction returned no frames for %s. Fallback to uniform sampling.",
                        video_path.name,
                    )
                    keyframe_ids = extract_keyframes_uniform(reader, target_count=args.uniform_count)
                    actual_strategy = "uniform_fallback"
            else:
                keyframe_ids = extract_keyframes_uniform(reader, target_count=args.uniform_count)
                actual_strategy = "uniform"

            timing: dict[str, float] = {}
            timing["keyframe_extract_s"] = time.perf_counter() - t_kf0

            if not keyframe_ids:
                raise RuntimeError(f"No keyframes extracted from video: {video_path}")

            mid_frame_id: int = reader.meta.mid_frame_id

            # 一次读出：关键帧 ∪ 中间帧（iter_frames_at 内部按升序 seek，见 video_io）
            read_ids_sorted = sorted(set(keyframe_ids) | {mid_frame_id})

            t_read0 = time.perf_counter()
            store: dict[int, tuple] = {}
            for frame_id, frame_hr, frame_lr in reader.iter_frames_at(
                read_ids_sorted, yield_low_res=True,
            ):
                if frame_hr is None or frame_lr is None:
                    logger.warning(
                        "Failed to read frame %d from %s, skipping.",
                        frame_id, video_path.name,
                    )
                    continue
                store[frame_id] = (frame_hr, frame_lr)

            timing["read_keyframes_s"] = time.perf_counter() - t_read0

            frame_ids = [fid for fid in keyframe_ids if fid in store]
            if not frame_ids:
                raise RuntimeError(f"No valid keyframes were loaded from video: {video_path}")

            frames_lr = [store[fid][1] for fid in frame_ids]
            full_res_scales = [reader.meta.low_res_scale] * len(frame_ids)

            # ── 批量检测：关键帧 ∪ 中间帧（保持 GPU 批次效率）──────────────
            # per_frame_masks: 仅对中间帧生成 seg_mask（用于可视化），
            # 关键帧跳过 seg_mask 的高分辨率 resize（4K 约节省 0.5~1.5s）。
            detect_fids = sorted((set(frame_ids) | {mid_frame_id}) & set(store.keys()))
            frames_det_hr = [store[fid][0] for fid in detect_fids]
            per_frame_masks = [fid == mid_frame_id for fid in detect_fids]

            t_det0 = time.perf_counter()
            batch_out = detector.detect_batch(
                frames_det_hr, detect_fids,
                generate_masks=False,
                per_frame_masks=per_frame_masks,
            )
            timing["detect_s"] = time.perf_counter() - t_det0

            det_map = {fid: dets for fid, dets in zip(detect_fids, batch_out)}
            all_detections = [det_map.get(fid, []) for fid in frame_ids]

            total_detections = sum(len(d) for d in all_detections)
            for index, (fid, dets) in enumerate(zip(frame_ids, all_detections), start=1):
                logger.info(
                    "[%s %d/%d] frame_id=%d detections=%d",
                    video_path.stem,
                    index,
                    len(frame_ids),
                    fid,
                    len(dets),
                )

            mid_frame = store[mid_frame_id][0] if mid_frame_id in store else None
            mid_frame_detections = det_map.get(mid_frame_id, [])
            timing["mid_frame_detect_s"] = 0.0

            if mid_frame is None:
                mid_frame = reader.read_frame(mid_frame_id, low_res=False)
            if mid_frame_id not in det_map and mid_frame is not None:
                t_mid = time.perf_counter()
                mid_frame_detections = detector.detect(
                    mid_frame, frame_id=mid_frame_id, enable_tracking=False,
                )
                timing["mid_frame_detect_s"] = time.perf_counter() - t_mid

            t_reg0 = time.perf_counter()
            registrations = registrar.register_sequence(
                keyframe_images=frames_lr,
                keyframe_ids=frame_ids,
                full_res_scales=full_res_scales,
                anchor_count=max(1, args.anchor_count),
            )
            timing["register_s"] = time.perf_counter() - t_reg0
            reg_sequence_info = registrar.get_last_sequence_info()
            reg_stats = registrar.get_stats()
            valid_reg_count = sum(1 for reg in registrations if reg.valid)

            t_dd0 = time.perf_counter()
            clusters = deduper.run(all_detections, registrations)
            timing["dedup_s"] = time.perf_counter() - t_dd0

            count_mode_details: dict[str, object] = {"count_mode": args.count_mode}
            t_cv0 = time.perf_counter()
            if args.count_mode == "classifier":
                classifier = ScrewClassifier(
                    weights_path=Path(args.classifier_weights),
                    use_fp16=not args.no_fp16,
                    device=args.device,
                )
                clusters, counts = classifier.classify_and_count(clusters)
                count_mode_details["classifier_mode"] = "torch" if classifier.is_torch_mode else "fallback"
                count_mode_details["classifier_weights"] = str(Path(args.classifier_weights))
            else:
                clusters, counts, missing_label_clusters = _classify_clusters_with_detector_votes(clusters)
                count_mode_details["missing_label_clusters"] = missing_label_clusters
            timing["count_vote_s"] = time.perf_counter() - t_cv0

            logger.info(
                "[%s] 耗时拆分(s): kf=%.2f read=%.2f detect=%.2f mid_detect=%.2f register=%.2f "
                "dedup=%.2f count=%.2f",
                video_path.stem,
                timing["keyframe_extract_s"],
                timing["read_keyframes_s"],
                timing["detect_s"],
                timing["mid_frame_detect_s"],
                timing["register_s"],
                timing["dedup_s"],
                timing["count_vote_s"],
            )

            elapsed_sec = time.perf_counter() - start_time
            result = {
                "video_name": video_path.name,
                "status": "ok",
                "counts": counts,
                "total": int(sum(counts)),
                "process_time_sec": elapsed_sec,
                "keyframe_strategy_used": actual_strategy,
                "keyframe_count": len(frame_ids),
                "total_detections": total_detections,
                "n_clusters": len(clusters),
                "registration_valid_frames": valid_reg_count,
                "registration_invalid_frames": len(registrations) - valid_reg_count,
                "reference_frame_id": int(reg_sequence_info.get("reference_frame_id", frame_ids[0])),
                "count_mode_details": count_mode_details,
                "registration_stats": reg_stats,
                "mid_frame": mid_frame,
                "mid_frame_id": int(mid_frame_id),
                "mid_frame_detections": mid_frame_detections,
                "timing_breakdown": timing,
            }
            logger.info(
                "Finished %s: counts=%s total=%d time=%.2fs",
                video_path.name,
                counts,
                int(sum(counts)),
                elapsed_sec,
            )
            return result

    except Exception as exc:
        elapsed_sec = time.perf_counter() - start_time
        logger.error("Failed to process %s: %s", video_path.name, exc, exc_info=True)
        return {
            "video_name": video_path.name,
            "status": "failed",
            "error": str(exc),
            "process_time_sec": elapsed_sec,
        }


def _format_report(
    input_path: Path,
    output_txt: Path,
    args: argparse.Namespace,
    results: list[dict[str, object]],
    total_elapsed_sec: float,
) -> str:
    success_count = sum(1 for item in results if item["status"] == "ok")
    fail_count = len(results) - success_count

    lines = [
        "Screw Counting Batch Report",
        f"generated_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"input: {input_path}",
        f"output_txt: {output_txt}",
        "",
        "Parameters:",
        f"keyframe_strategy: {args.keyframe_strategy}",
        f"uniform_count: {args.uniform_count}",
        f"detector_weights: {Path(args.detector_weights)}",
        f"device: {args.device}",
        f"use_fp16: {not args.no_fp16}",
        f"use_sahi: {args.use_sahi}",
        f"feature_type: {args.feature_type}",
        f"anchor_strategy: {args.anchor_strategy}",
        f"anchor_count: {max(1, args.anchor_count)}",
        f"inlier_ratio_threshold: {args.inlier_ratio_threshold}",
        f"min_match_count: {args.min_match_count}",
        f"dist_thresh: {args.dist_thresh}",
        f"min_observations: {args.min_observations}",
        f"dedup_method: {args.dedup_method}",
        f"invalid_reg_fallback: {args.invalid_reg_fallback}",
        f"count_mode: {args.count_mode}",
        "",
        f"videos_total: {len(results)}",
        f"videos_success: {success_count}",
        f"videos_failed: {fail_count}",
        f"overall_time_sec: {total_elapsed_sec:.3f}",
        "",
    ]

    for item in results:
        lines.append("-" * 48)
        lines.append(f"video: {item['video_name']}")
        lines.append(f"status: {item['status']}")

        if item["status"] == "ok":
            counts = item["counts"]
            lines.extend(
                [
                    f"Type_1: {counts[0]}",
                    f"Type_2: {counts[1]}",
                    f"Type_3: {counts[2]}",
                    f"Type_4: {counts[3]}",
                    f"Type_5: {counts[4]}",
                    f"Total: {item['total']}",
                    f"process_time_sec: {item['process_time_sec']:.3f}",
                ]
            )
        else:
            lines.append(f"error: {item.get('error', 'unknown error')}")
            lines.append(f"process_time_sec: {item['process_time_sec']:.3f}")

        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = _parse_args()

    input_path = Path(args.input).resolve()
    output_txt = Path(args.output_txt).resolve()

    try:
        video_paths = _collect_video_paths(input_path)
    except Exception as exc:
        logger.error("%s", exc)
        return 1

    output_txt.parent.mkdir(parents=True, exist_ok=True)

    detector = Detector(
        weights_path=Path(args.detector_weights),
        use_fp16=not args.no_fp16,
        use_sahi=args.use_sahi,
        device=args.device,
    )
    detector_mode = "YOLO" if detector.is_yolo_mode else "fallback"

    logger.info("Detector mode: %s", detector_mode)
    logger.info("Videos to process: %d", len(video_paths))

    batch_start = time.perf_counter()
    results: list[dict[str, object]] = []
    for video_path in video_paths:
        results.append(_process_video(video_path, args, detector))
    total_elapsed_sec = time.perf_counter() - batch_start

    report_text = _format_report(
        input_path=input_path,
        output_txt=output_txt,
        args=args,
        results=results,
        total_elapsed_sec=total_elapsed_sec,
    )
    output_txt.write_text(report_text, encoding="utf-8")

    success_count = sum(1 for item in results if item["status"] == "ok")
    logger.info("Saved batch report to: %s", output_txt)
    logger.info(
        "Batch finished: success=%d/%d total_time=%.2fs",
        success_count,
        len(results),
        total_elapsed_sec,
    )
    return 0 if success_count == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
