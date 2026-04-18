#!/usr/bin/env python3
"""
Tune step4 parameters against video_exp/true_count.json.

This script caches:
1. keyframe selection
2. low-res frames for registration
3. detector outputs on the selected keyframes

Then it searches registration / dedup / counting parameters without rerunning
detector inference for every parameter combination.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import modules.dedup as dedup_module
import modules.registration as registration_module
from modules.classifier import (
    _classify_cluster_from_detector_votes,
    _cluster_has_detector_multiclass_labels,
)
from modules.dedup import GlobalDedup
from modules.detector import DEFAULT_WEIGHTS as DEFAULT_DETECTOR_WEIGHTS, Detector
from modules.registration import FrameRegistration
from pipeline import _extract_keyframes_motion, extract_keyframes_uniform
from utils.video_io import VIDEO_EXTENSIONS, VideoReader


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("tune_step4_params")


@dataclass
class SequenceCache:
    name: str
    video_path: Path
    frame_ids: list[int]
    frames_lr: list[np.ndarray]
    full_res_scales: list[float]
    detections: list[list]


def _parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _parse_float_list(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def _parse_str_list(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search step4 parameters against true_count.json."
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default=str(PROJECT_ROOT.parent.parent / "video_exp"),
        help="Directory that contains videos and true_count.json.",
    )
    parser.add_argument(
        "--truth_json",
        type=str,
        default=str(PROJECT_ROOT.parent.parent / "video_exp" / "true_count.json"),
        help="Ground-truth count json path.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "tuning_results" / "step4_tuning.json"),
        help="Output json file for the sorted search results.",
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
        default=str(DEFAULT_DETECTOR_WEIGHTS),
        help="Detector weights path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Inference device for detector, for example cuda:0 or cpu.",
    )
    parser.add_argument(
        "--no_fp16",
        action="store_true",
        default=False,
        help="Disable FP16 inference for detector.",
    )
    parser.add_argument(
        "--use_sahi",
        action="store_true",
        default=False,
        help="Enable SAHI slicing during detection.",
    )
    parser.add_argument(
        "--feature_types",
        type=str,
        default="AKAZE",
        help="Comma-separated registration feature types.",
    )
    parser.add_argument(
        "--anchor_strategies",
        type=str,
        default="middle",
        help="Comma-separated anchor strategies.",
    )
    parser.add_argument(
        "--anchor_counts",
        type=str,
        default="10",
        help="Comma-separated anchor counts.",
    )
    parser.add_argument(
        "--inlier_ratio_thresholds",
        type=str,
        default="0.25",
        help="Comma-separated registration inlier thresholds.",
    )
    parser.add_argument(
        "--min_match_counts",
        type=str,
        default="15",
        help="Comma-separated minimum match counts.",
    )
    parser.add_argument(
        "--dist_threshs",
        type=str,
        default="32,36,40,44,48",
        help="Comma-separated dedup distance thresholds.",
    )
    parser.add_argument(
        "--min_observations_list",
        type=str,
        default="1,2,3",
        help="Comma-separated dedup min_observations values.",
    )
    parser.add_argument(
        "--dedup_methods",
        type=str,
        default="dbscan,incremental",
        help="Comma-separated dedup backends.",
    )
    parser.add_argument(
        "--invalid_reg_fallback",
        type=str,
        default=dedup_module.INVALID_REG_FALLBACK,
        choices=["skip", "identity", "tracker"],
        help="How to handle invalid registrations during dedup.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of top results to print.",
    )
    return parser.parse_args()


def _load_truth(path: Path) -> dict[str, dict[str, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    truth: dict[str, dict[str, int]] = {}
    for video_name, counts in payload.items():
        truth[video_name] = {
            "type1": int(counts["type1"]),
            "type2": int(counts["type2"]),
            "type3": int(counts["type3"]),
            "type4": int(counts["type4"]),
            "type5": int(counts["type5"]),
            "total": int(counts["total"]),
        }
    return truth


def _find_video_paths(video_dir: Path, truth: dict[str, dict[str, int]]) -> list[Path]:
    video_paths: list[Path] = []
    for stem in sorted(truth.keys()):
        matched = None
        for suffix in VIDEO_EXTENSIONS:
            candidate = video_dir / f"{stem}{suffix.upper()}"
            if candidate.exists():
                matched = candidate
                break
            candidate = video_dir / f"{stem}{suffix.lower()}"
            if candidate.exists():
                matched = candidate
                break
        if matched is None:
            raise FileNotFoundError(f"Missing video for truth entry: {stem}")
        video_paths.append(matched)
    return video_paths


def _load_sequence_cache(
    video_path: Path,
    detector: Detector,
    keyframe_strategy: str,
    uniform_count: int,
) -> SequenceCache:
    with VideoReader(video_path) as reader:
        if keyframe_strategy == "motion":
            try:
                keyframe_ids = _extract_keyframes_motion(reader)
            except Exception as exc:
                logger.warning("Motion keyframe extraction failed for %s, fallback to uniform: %s", video_path.name, exc)
                keyframe_ids = extract_keyframes_uniform(reader, target_count=uniform_count)
        else:
            keyframe_ids = extract_keyframes_uniform(reader, target_count=uniform_count)

        if not keyframe_ids:
            raise RuntimeError(f"No keyframes extracted from video: {video_path}")

        frame_ids: list[int] = []
        frames_lr: list[np.ndarray] = []
        full_res_scales: list[float] = []
        detections: list[list] = []

        logger.info("Caching sequence %s with %d keyframes...", video_path.name, len(keyframe_ids))

        for index, (frame_id, frame_hr, frame_lr) in enumerate(
            reader.iter_frames_at(keyframe_ids, yield_low_res=True),
            start=1,
        ):
            if frame_hr is None or frame_lr is None:
                logger.warning("Failed to read frame %d from %s, skipping.", frame_id, video_path.name)
                continue

            frame_ids.append(frame_id)
            frames_lr.append(frame_lr.copy())
            full_res_scales.append(reader.meta.low_res_scale)
            detections.append(detector.detect(frame_hr, frame_id=frame_id, enable_tracking=False))

            logger.info("[%s %d/%d] frame_id=%d detections=%d", video_path.stem, index, len(keyframe_ids), frame_id, len(detections[-1]))

    if not frame_ids:
        raise RuntimeError(f"No valid frames cached for video: {video_path}")

    return SequenceCache(
        name=video_path.stem,
        video_path=video_path,
        frame_ids=frame_ids,
        frames_lr=frames_lr,
        full_res_scales=full_res_scales,
        detections=detections,
    )


def _classify_clusters_with_detector_votes(clusters) -> tuple[list, list[int], int]:
    counts = [0, 0, 0, 0, 0]
    missing_label_clusters = 0
    for cluster in clusters:
        if _cluster_has_detector_multiclass_labels(cluster):
            _classify_cluster_from_detector_votes(cluster)
            if 0 <= cluster.pred_class < len(counts):
                counts[cluster.pred_class] += 1
        else:
            missing_label_clusters += 1
    return clusters, counts, missing_label_clusters


def _evaluate_sequence(
    cache: SequenceCache,
    *,
    feature_type: str,
    anchor_strategy: str,
    anchor_count: int,
    inlier_ratio_threshold: float,
    min_match_count: int,
    dist_thresh: float,
    min_observations: int,
    dedup_method: str,
    invalid_reg_fallback: str,
) -> dict[str, object]:
    registration_module.ANCHOR_STRATEGY = anchor_strategy
    registrar = FrameRegistration(
        feature_type=feature_type,
        inlier_ratio_threshold=inlier_ratio_threshold,
        min_match_count=min_match_count,
    )
    registrations = registrar.register_sequence(
        keyframe_images=cache.frames_lr,
        keyframe_ids=cache.frame_ids,
        full_res_scales=cache.full_res_scales,
        anchor_count=max(1, anchor_count),
    )
    reg_stats = registrar.get_stats()
    reg_sequence_info = registrar.get_last_sequence_info()

    deduper = GlobalDedup(
        dist_thresh=dist_thresh,
        min_observations=min_observations,
        use_dbscan=dedup_method == "dbscan",
        invalid_reg_fallback=invalid_reg_fallback,
    )
    clusters = deduper.run(cache.detections, registrations)
    clusters, counts, missing_label_clusters = _classify_clusters_with_detector_votes(clusters)

    return {
        "counts": counts,
        "total": int(sum(counts)),
        "n_clusters": len(clusters),
        "missing_label_clusters": missing_label_clusters,
        "registration_valid_frames": int(sum(1 for reg in registrations if reg.valid)),
        "registration_invalid_frames": int(sum(1 for reg in registrations if not reg.valid)),
        "reference_index": reg_sequence_info.get("reference_index"),
        "reference_frame_id": reg_sequence_info.get("reference_frame_id"),
        "valid_anchor_indices": reg_sequence_info.get("valid_anchor_indices", []),
        "registration_stats": reg_stats,
    }


def _score_prediction(pred_counts: list[int], truth_counts: dict[str, int]) -> dict[str, int]:
    truth_vec = [
        truth_counts["type1"],
        truth_counts["type2"],
        truth_counts["type3"],
        truth_counts["type4"],
        truth_counts["type5"],
    ]
    diffs = [int(pred - truth) for pred, truth in zip(pred_counts, truth_vec)]
    abs_diffs = [abs(value) for value in diffs]
    return {
        "l1_types": int(sum(abs_diffs)),
        "l1_total": int(abs(sum(pred_counts) - truth_counts["total"])),
        "max_abs_type_diff": int(max(abs_diffs)),
        "exact_match": int(all(value == 0 for value in diffs)),
    }


def main() -> int:
    args = _parse_args()

    video_dir = Path(args.video_dir)
    truth_json = Path(args.truth_json)
    output_path = Path(args.output)

    truth = _load_truth(truth_json)
    video_paths = _find_video_paths(video_dir, truth)

    detector = Detector(
        weights_path=Path(args.detector_weights),
        use_fp16=not args.no_fp16,
        use_sahi=args.use_sahi,
        device=args.device,
    )
    logger.info("Detector mode for tuning: %s", "YOLO" if detector.is_yolo_mode else "fallback")

    caches = [
        _load_sequence_cache(
            video_path=video_path,
            detector=detector,
            keyframe_strategy=args.keyframe_strategy,
            uniform_count=args.uniform_count,
        )
        for video_path in video_paths
    ]

    feature_types = _parse_str_list(args.feature_types)
    anchor_strategies = _parse_str_list(args.anchor_strategies)
    anchor_counts = _parse_int_list(args.anchor_counts)
    inlier_ratio_thresholds = _parse_float_list(args.inlier_ratio_thresholds)
    min_match_counts = _parse_int_list(args.min_match_counts)
    dist_threshs = _parse_float_list(args.dist_threshs)
    min_observations_list = _parse_int_list(args.min_observations_list)
    dedup_methods = _parse_str_list(args.dedup_methods)

    search_space = list(itertools.product(
        feature_types,
        anchor_strategies,
        anchor_counts,
        inlier_ratio_thresholds,
        min_match_counts,
        dist_threshs,
        min_observations_list,
        dedup_methods,
    ))
    logger.info("Search combinations: %d", len(search_space))

    results: list[dict[str, object]] = []

    for combo_index, combo in enumerate(search_space, start=1):
        (
            feature_type,
            anchor_strategy,
            anchor_count,
            inlier_ratio_threshold,
            min_match_count,
            dist_thresh,
            min_observations,
            dedup_method,
        ) = combo

        logger.info(
            "[%d/%d] feature=%s anchor_strategy=%s anchor_count=%d inlier=%.3f min_match=%d dist=%.1f min_obs=%d dedup=%s",
            combo_index,
            len(search_space),
            feature_type,
            anchor_strategy,
            anchor_count,
            inlier_ratio_threshold,
            min_match_count,
            dist_thresh,
            min_observations,
            dedup_method,
        )

        per_video: dict[str, object] = {}
        total_l1_types = 0
        total_l1_total = 0
        max_video_l1 = 0
        exact_video_matches = 0

        for cache in caches:
            evaluation = _evaluate_sequence(
                cache,
                feature_type=feature_type,
                anchor_strategy=anchor_strategy,
                anchor_count=anchor_count,
                inlier_ratio_threshold=inlier_ratio_threshold,
                min_match_count=min_match_count,
                dist_thresh=dist_thresh,
                min_observations=min_observations,
                dedup_method=dedup_method,
                invalid_reg_fallback=args.invalid_reg_fallback,
            )
            truth_counts = truth[cache.name]
            score = _score_prediction(evaluation["counts"], truth_counts)
            total_l1_types += score["l1_types"]
            total_l1_total += score["l1_total"]
            max_video_l1 = max(max_video_l1, score["l1_types"])
            exact_video_matches += score["exact_match"]

            per_video[cache.name] = {
                **evaluation,
                "truth": truth_counts,
                "score": score,
            }

        results.append(
            {
                "params": {
                    "feature_type": feature_type,
                    "anchor_strategy": anchor_strategy,
                    "anchor_count": anchor_count,
                    "inlier_ratio_threshold": inlier_ratio_threshold,
                    "min_match_count": min_match_count,
                    "dist_thresh": dist_thresh,
                    "min_observations": min_observations,
                    "dedup_method": dedup_method,
                    "invalid_reg_fallback": args.invalid_reg_fallback,
                    "keyframe_strategy": args.keyframe_strategy,
                    "uniform_count": args.uniform_count,
                    "use_sahi": args.use_sahi,
                },
                "score": {
                    "total_l1_types": total_l1_types,
                    "total_l1_total": total_l1_total,
                    "max_video_l1_types": max_video_l1,
                    "exact_video_matches": exact_video_matches,
                },
                "per_video": per_video,
            }
        )

    results.sort(
        key=lambda item: (
            item["score"]["total_l1_types"],
            item["score"]["total_l1_total"],
            item["score"]["max_video_l1_types"],
            -item["score"]["exact_video_matches"],
        )
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    logger.info("Top %d results:", min(args.top_k, len(results)))
    for rank, item in enumerate(results[: args.top_k], start=1):
        logger.info(
            "#%d total_l1_types=%d total_l1_total=%d max_video_l1=%d exact_video_matches=%d params=%s",
            rank,
            item["score"]["total_l1_types"],
            item["score"]["total_l1_total"],
            item["score"]["max_video_l1_types"],
            item["score"]["exact_video_matches"],
            item["params"],
        )
        for video_name, video_result in item["per_video"].items():
            logger.info(
                "  %s pred=%s truth=%s video_l1=%d",
                video_name,
                video_result["counts"],
                [
                    video_result["truth"]["type1"],
                    video_result["truth"]["type2"],
                    video_result["truth"]["type3"],
                    video_result["truth"]["type4"],
                    video_result["truth"]["type5"],
                ],
                video_result["score"]["l1_types"],
            )

    logger.info("Saved tuning results to: %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
