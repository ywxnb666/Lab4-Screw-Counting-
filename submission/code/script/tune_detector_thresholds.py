#!/usr/bin/env python3
"""
Tune detector class confidence thresholds with 0.01 step.

Goal:
- minimize count error against video_exp/true_count.json
- keep the rest of the counting pipeline aligned with run.py defaults

Strategy:
1) Cache keyframes, detections (with a low global conf), and registrations once.
2) Run coordinate-descent search over 5 class thresholds with step=0.01.

This avoids rerunning YOLO and registration for every threshold candidate.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import modules.registration as registration_module
from interfaces import Detection
from modules.classifier import (
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
logger = logging.getLogger("tune_detector_thresholds")

CLASS_NAMES = ["type1", "type2", "type3", "type4", "type5"]


@dataclass
class VideoCache:
    name: str
    frame_ids: List[int]
    frames_lr: List[np.ndarray]
    full_res_scales: List[float]
    detections: List[List[Detection]]
    registrations: list


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune 5-class detector thresholds with 0.01 step.")
    parser.add_argument(
        "--video_dir",
        type=str,
        default=str(PROJECT_ROOT.parent.parent / "video_exp"),
        help="Directory containing videos.",
    )
    parser.add_argument(
        "--truth_json",
        type=str,
        default=str(PROJECT_ROOT.parent.parent / "video_exp" / "true_count.json"),
        help="Ground truth json path.",
    )
    parser.add_argument(
        "--detector_weights",
        type=str,
        default=str(DEFAULT_WEIGHTS),
        help="Detector weights path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Inference device.",
    )
    parser.add_argument(
        "--no_fp16",
        action="store_true",
        default=False,
        help="Disable FP16 detector inference.",
    )
    parser.add_argument(
        "--keyframe_strategy",
        type=str,
        default="uniform",
        choices=["uniform", "motion"],
        help="Keyframe extraction strategy.",
    )
    parser.add_argument(
        "--uniform_count",
        type=int,
        default=30,
        help="Number of keyframes for uniform strategy.",
    )
    parser.add_argument(
        "--collect_conf",
        type=float,
        default=0.01,
        help="Low detector conf used for caching candidate detections.",
    )
    parser.add_argument(
        "--dist_thresh",
        type=float,
        default=40.0,
        help="Dedup distance threshold, aligned with run.py default.",
    )
    parser.add_argument(
        "--min_observations",
        type=int,
        default=4,
        help="Dedup min_observations, aligned with run.py backend defaults.",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.01,
        help="Threshold search step size.",
    )
    parser.add_argument(
        "--max_rounds",
        type=int,
        default=3,
        help="Max coordinate-descent rounds.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=str(PROJECT_ROOT / "tuning_results" / "detector_threshold_search.json"),
        help="Output json report path.",
    )
    return parser.parse_args()


def _normalize(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _load_truth(path: Path) -> Dict[str, List[int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    truth: Dict[str, List[int]] = {}
    for video_name, counts in payload.items():
        truth[video_name] = [
            int(counts["type1"]),
            int(counts["type2"]),
            int(counts["type3"]),
            int(counts["type4"]),
            int(counts["type5"]),
        ]
    return truth


def _find_video_paths(video_dir: Path, truth: Dict[str, List[int]]) -> List[Path]:
    video_paths: List[Path] = []
    for stem in sorted(truth.keys()):
        found = None
        for ext in VIDEO_EXTENSIONS:
            c1 = video_dir / f"{stem}{ext.lower()}"
            c2 = video_dir / f"{stem}{ext.upper()}"
            if c1.exists():
                found = c1
                break
            if c2.exists():
                found = c2
                break
        if found is None:
            raise FileNotFoundError(f"Missing video for truth entry: {stem}")
        video_paths.append(found)
    return video_paths


def _extract_keyframes(reader: VideoReader, strategy: str, uniform_count: int) -> List[int]:
    if strategy == "motion":
        try:
            ids = _extract_keyframes_motion(reader)
        except Exception as exc:
            logger.warning("Motion keyframe extraction failed, fallback to uniform: %s", exc)
            ids = extract_keyframes_uniform(reader, target_count=uniform_count)
    else:
        ids = extract_keyframes_uniform(reader, target_count=uniform_count)
    if not ids:
        raise RuntimeError("No keyframes extracted")
    return ids


def _classify_detector_votes(clusters) -> List[int]:
    counts = [0, 0, 0, 0, 0]
    for cluster in clusters:
        if _cluster_has_detector_multiclass_labels(cluster):
            _classify_cluster_from_detector_votes(cluster)
            if 0 <= cluster.pred_class < 5:
                counts[cluster.pred_class] += 1
    return counts


def _cache_video(
    video_path: Path,
    detector: Detector,
    keyframe_strategy: str,
    uniform_count: int,
) -> VideoCache:
    def _compact_detection(det: Detection) -> Detection:
        # Seg masks are huge and not used by threshold tuning; drop them to avoid OOM.
        return Detection(
            frame_id=det.frame_id,
            bbox=det.bbox.copy(),
            confidence=float(det.confidence),
            crop=det.crop,
            track_id=int(det.track_id),
            class_id=int(det.class_id),
            class_name=str(det.class_name),
            seg_mask=None,
        )

    with VideoReader(video_path) as reader:
        keyframe_ids = _extract_keyframes(reader, keyframe_strategy, uniform_count)
        frame_ids: List[int] = []
        frames_lr: List[np.ndarray] = []
        full_res_scales: List[float] = []
        detections: List[List[Detection]] = []

        logger.info("Caching %s with %d keyframes...", video_path.name, len(keyframe_ids))
        for i, (fid, frame_hr, frame_lr) in enumerate(
            reader.iter_frames_at(keyframe_ids, yield_low_res=True),
            start=1,
        ):
            if frame_hr is None or frame_lr is None:
                continue
            dets = detector.detect(frame_hr, frame_id=fid, enable_tracking=False)
            dets = [_compact_detection(d) for d in dets]
            frame_ids.append(fid)
            frames_lr.append(frame_lr.copy())
            full_res_scales.append(reader.meta.low_res_scale)
            detections.append(dets)
            logger.info("[%s %d/%d] frame_id=%d det=%d", video_path.stem, i, len(keyframe_ids), fid, len(dets))

    registration_module.ANCHOR_STRATEGY = "first"
    registrar = FrameRegistration(
        feature_type="ORB",
        inlier_ratio_threshold=0.25,
        min_match_count=15,
    )
    registrations = registrar.register_sequence(
        keyframe_images=frames_lr,
        keyframe_ids=frame_ids,
        full_res_scales=full_res_scales,
        anchor_count=10,
    )

    logger.info(
        "Cached %s: frames=%d, det_total=%d, reg_valid=%d/%d",
        video_path.stem,
        len(frame_ids),
        sum(len(x) for x in detections),
        sum(1 for r in registrations if r.valid),
        len(registrations),
    )

    return VideoCache(
        name=video_path.stem,
        frame_ids=frame_ids,
        frames_lr=frames_lr,
        full_res_scales=full_res_scales,
        detections=detections,
        registrations=registrations,
    )


def _filter_detections(
    detections: List[List[Detection]],
    thresholds: Dict[str, float],
) -> List[List[Detection]]:
    filtered: List[List[Detection]] = []
    for frame_dets in detections:
        keep: List[Detection] = []
        for det in frame_dets:
            key = _normalize(det.class_name)
            thr = thresholds.get(key)
            if thr is None:
                continue
            if float(det.confidence) >= float(thr):
                keep.append(det)
        filtered.append(keep)
    return filtered


def _evaluate_thresholds(
    caches: List[VideoCache],
    truth: Dict[str, List[int]],
    thresholds: Dict[str, float],
    dist_thresh: float,
    min_observations: int,
) -> Dict[str, object]:
    per_video: Dict[str, Dict[str, object]] = {}
    total_l1 = 0

    for cache in caches:
        filtered = _filter_detections(cache.detections, thresholds)
        deduper = GlobalDedup(
            dist_thresh=dist_thresh,
            min_observations=min_observations,
            use_dbscan=True,
            invalid_reg_fallback="skip",
        )
        clusters = deduper.run(filtered, cache.registrations)
        pred = _classify_detector_votes(clusters)
        gt = truth[cache.name]
        abs_err = [abs(int(pred[i]) - int(gt[i])) for i in range(5)]
        l1 = int(sum(abs_err))
        total_l1 += l1
        per_video[cache.name] = {
            "pred": pred,
            "gt": gt,
            "abs_err": abs_err,
            "l1": l1,
        }

    return {
        "total_l1": int(total_l1),
        "per_video": per_video,
    }


def _frange(step: float) -> List[float]:
    n = int(round(1.0 / step))
    values = []
    for i in range(1, n):
        values.append(round(i * step, 2))
    if 0.99 not in values:
        values.append(0.99)
    return sorted(set(values))


def _search_coordinate_descent(
    caches: List[VideoCache],
    truth: Dict[str, List[int]],
    init_thresholds: Dict[str, float],
    step: float,
    max_rounds: int,
    dist_thresh: float,
    min_observations: int,
) -> Dict[str, object]:
    grid_values = _frange(step)
    current = {k: round(float(v), 2) for k, v in init_thresholds.items()}
    history: List[Dict[str, object]] = []

    best_eval = _evaluate_thresholds(caches, truth, current, dist_thresh, min_observations)
    logger.info("Initial thresholds=%s total_l1=%d", current, best_eval["total_l1"])

    for round_idx in range(1, max_rounds + 1):
        improved_any = False
        logger.info("=== Round %d/%d ===", round_idx, max_rounds)

        for cls in CLASS_NAMES:
            cls_best_thr = current[cls]
            cls_best_eval = best_eval

            for v in grid_values:
                trial = dict(current)
                trial[cls] = v
                trial_eval = _evaluate_thresholds(caches, truth, trial, dist_thresh, min_observations)

                if trial_eval["total_l1"] < cls_best_eval["total_l1"]:
                    cls_best_eval = trial_eval
                    cls_best_thr = v

            if cls_best_thr != current[cls]:
                improved_any = True
                current[cls] = cls_best_thr
                best_eval = cls_best_eval

            logger.info("Class %s best_thr=%.2f total_l1=%d", cls, current[cls], best_eval["total_l1"])

        history.append(
            {
                "round": round_idx,
                "thresholds": dict(current),
                "score": best_eval,
            }
        )

        if not improved_any:
            logger.info("No improvement in this round. Stop early.")
            break

    return {
        "best_thresholds": current,
        "best_score": best_eval,
        "history": history,
    }


def main() -> int:
    args = _parse_args()

    truth = _load_truth(Path(args.truth_json))
    video_paths = _find_video_paths(Path(args.video_dir), truth)

    detector = Detector(
        weights_path=Path(args.detector_weights),
        class_conf_json_path=None,
        conf_threshold=float(args.collect_conf),
        use_fp16=not args.no_fp16,
        use_sahi=False,
        device=args.device,
    )
    logger.info("Detector mode: %s", "YOLO" if detector.is_yolo_mode else "fallback")

    caches = [
        _cache_video(
            video_path=v,
            detector=detector,
            keyframe_strategy=args.keyframe_strategy,
            uniform_count=args.uniform_count,
        )
        for v in video_paths
    ]

    default_thresholds = {
        "type1": 0.46,
        "type2": 0.80,
        "type3": 0.52,
        "type4": 0.45,
        "type5": 0.67,
    }

    result = _search_coordinate_descent(
        caches=caches,
        truth=truth,
        init_thresholds=default_thresholds,
        step=float(args.step),
        max_rounds=int(args.max_rounds),
        dist_thresh=float(args.dist_thresh),
        min_observations=int(args.min_observations),
    )

    out = {
        "config": {
            "video_dir": str(args.video_dir),
            "truth_json": str(args.truth_json),
            "detector_weights": str(args.detector_weights),
            "device": str(args.device),
            "step": float(args.step),
            "max_rounds": int(args.max_rounds),
            "collect_conf": float(args.collect_conf),
            "dist_thresh": float(args.dist_thresh),
            "min_observations": int(args.min_observations),
            "keyframe_strategy": str(args.keyframe_strategy),
            "uniform_count": int(args.uniform_count),
        },
        "result": result,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    logger.info("Best thresholds: %s", result["best_thresholds"])
    logger.info("Best total_l1: %d", result["best_score"]["total_l1"])
    logger.info("Saved report to: %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
