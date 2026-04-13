#!/usr/bin/env python3
"""
tools/ablation.py - 消融实验记录与对比工具
Owner: D（工程封装）

用途：
  对不同配置（去重策略、分类策略、SAHI 效果等）运行 pipeline 并记录结果，
  生成消融实验对比表格，作为实验报告的核心素材。

对应 plan.md 第 7 节的三组消融实验：
  A. 去重策略对比（Baseline / 锚帧聚类 / 全景拼接）
  B. 分类策略对比（Lab2 直迁 / fine-tune / 单帧 vs 投票）
  C. SAHI 效果对比（整图检测 vs 切片检测）

使用示例：
  # 运行所有消融实验（需要先配置好各模型权重）
  python tools/ablation.py --data_dir vedio_exp/ --output ablation_results/

  # 仅运行去重策略对比
  python tools/ablation.py --data_dir vedio_exp/ --output ablation_results/ --group A

  # 加载已有结果并生成报告（不重新运行）
  python tools/ablation.py --report_only --results_dir ablation_results/

  # 与真实标签对比计算误差（需要提供 gt.npy）
  python tools/ablation.py --data_dir vedio_exp/ --output ablation_results/ \\
      --gt_path gt.npy

依赖：numpy, opencv-python, tabulate（可选，用于格式化表格输出）
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

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
logger = logging.getLogger("ablation")


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class AblationConfig:
    """
    单次消融实验的配置。

    每个 AblationConfig 对应一种参数组合，运行一次完整 pipeline，
    记录计数结果和耗时。
    """

    name: str
    """实验名称（简短标识，用于表格列名）。"""

    description: str
    """实验说明（在报告中显示）。"""

    group: str
    """
    消融实验组别：
    - 'A'：去重策略对比
    - 'B'：分类策略对比
    - 'C'：SAHI 效果对比
    - 'custom'：自定义
    """

    # ---- 关键帧提取参数 ----
    keyframe_strategy: str = "motion"
    """关键帧策略：'motion' | 'uniform'。"""

    # ---- 检测器参数 ----
    use_sahi: bool = True
    """是否使用 SAHI 切片推理（用于消融实验 C）。"""

    detector_weights: Optional[str] = None
    """检测器权重路径（None 表示使用默认路径）。"""

    # ---- 配准与去重参数 ----
    dist_thresh: float = 40.0
    """去重聚类距离阈值（像素）。"""

    min_observations: int = 1
    """Cluster 最少观测次数。"""

    use_dbscan: bool = True
    """是否使用 DBSCAN（否则使用增量聚类）。"""

    invalid_reg_fallback: str = "skip"
    """配准失败时的处理策略：'skip' | 'identity'。"""

    skip_registration: bool = False
    """
    是否跳过配准（用于 Baseline：逐帧检测无去重）。
    True 时等效于"每帧独立计数，不做跨帧去重"。
    """

    # ---- 分类器参数 ----
    classifier_weights: Optional[str] = None
    """分类器权重路径（None 表示使用默认路径）。"""

    use_vote: bool = True
    """
    是否使用 Cluster 级投票分类（否则只用 best_crop 单次预测）。
    用于消融实验 B 的"单帧分类 vs cluster 投票"对比。
    """

    vote_temperature: float = 1.0
    """投票温度参数。"""

    # ---- 推理加速 ----
    use_fp16: bool = True
    """是否使用 FP16。"""

    device: str = ""
    """推理设备。"""

    # ---- 其他 ----
    extra: Dict[str, Any] = field(default_factory=dict)
    """附加参数（供自定义实验使用）。"""

    def to_dict(self) -> Dict[str, Any]:
        """转换为可序列化的字典。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AblationConfig":
        """从字典恢复配置。"""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class AblationResult:
    """
    单次消融实验的结果。

    包含计数结果、各模块耗时、与 GT 对比的误差等信息。
    """

    config_name: str
    """对应的 AblationConfig.name。"""

    counts: Dict[str, List[int]]
    """
    各视频的计数结果。
    {video_name: [n_type1, n_type2, n_type3, n_type4, n_type5]}
    """

    total_time: float
    """处理所有视频的总耗时（秒）。"""

    per_video_time: Dict[str, float] = field(default_factory=dict)
    """各视频的处理耗时（秒）。"""

    errors: Dict[str, List[int]] = field(default_factory=dict)
    """
    与 GT 的绝对误差（若提供了 GT）。
    {video_name: [err_type1, ..., err_type5]}
    """

    mae_per_type: List[float] = field(default_factory=list)
    """各类别的平均绝对误差（MAE），长度为 5。"""

    overall_mae: float = 0.0
    """所有视频所有类别的平均绝对误差。"""

    score: float = 0.0
    """
    按作业评分标准计算的得分（满分 40）：
    - e=0: 4分
    - e<=3: 2分
    - e>=4: 0分
    """

    notes: str = ""
    """附加说明（如模型状态、异常情况等）。"""

    timestamp: str = ""
    """实验执行时间戳（ISO 格式）。"""

    def to_dict(self) -> Dict[str, Any]:
        """转换为可序列化的字典。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AblationResult":
        """从字典恢复结果。"""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def has_gt(self) -> bool:
        """是否已计算与 GT 的误差。"""
        return bool(self.errors)


# ---------------------------------------------------------------------------
# 评分计算（按作业规范）
# ---------------------------------------------------------------------------

def compute_score(
    pred: Dict[str, List[int]],
    gt: Dict[str, List[int]],
    points_perfect: float = 4.0,
    points_partial: float = 2.0,
    partial_thresh: int = 3,
) -> Tuple[float, Dict[str, List[int]], List[float], float]:
    """
    按作业评分标准计算得分。

    评分规则（每个类别 4 分，共 10 个类别值 × 4 分 = 40 分）：
    - e = 0        → points_perfect 分（默认 4 分）
    - 1 ≤ e ≤ 3   → points_partial 分（默认 2 分）
    - e ≥ 4        → 0 分

    Parameters
    ----------
    pred : Dict[str, List[int]]
        预测计数字典 {video_name: [c1,...,c5]}。
    gt : Dict[str, List[int]]
        真实计数字典 {video_name: [c1,...,c5]}。
    points_perfect : float
        误差为 0 时的得分。
    points_partial : float
        误差在 [1, partial_thresh] 时的得分。
    partial_thresh : int
        部分分的误差上限（包含）。

    Returns
    -------
    total_score : float
        总得分。
    errors : Dict[str, List[int]]
        各视频各类别的绝对误差。
    mae_per_type : List[float]
        各类别（Type_1~Type_5）的平均绝对误差。
    overall_mae : float
        所有视频所有类别的平均绝对误差。
    """
    errors: Dict[str, List[int]] = {}
    type_errors: List[List[int]] = [[] for _ in range(5)]
    total_score = 0.0

    for video_name, gt_counts in gt.items():
        pred_counts = pred.get(video_name, [0] * 5)
        video_errors = []
        for i, (p, g) in enumerate(zip(pred_counts, gt_counts)):
            e = abs(int(p) - int(g))
            video_errors.append(e)
            type_errors[i].append(e)

            if e == 0:
                total_score += points_perfect
            elif e <= partial_thresh:
                total_score += points_partial
            # else: 0分

        errors[video_name] = video_errors

    mae_per_type = [
        float(np.mean(errs)) if errs else 0.0
        for errs in type_errors
    ]

    all_errors = [e for errs in errors.values() for e in errs]
    overall_mae = float(np.mean(all_errors)) if all_errors else 0.0

    return total_score, errors, mae_per_type, overall_mae


# ---------------------------------------------------------------------------
# 预定义消融实验配置
# ---------------------------------------------------------------------------

def get_group_a_configs() -> List[AblationConfig]:
    """
    消融实验 A：去重策略对比。

    方案对比：
    1. Baseline    : 逐帧检测，不做跨帧去重（所有帧的检测直接相加）
    2. 增量聚类    : homography + 增量聚类（最近邻合并）
    3. DBSCAN 聚类 : homography + DBSCAN（更稳定）
    """
    return [
        AblationConfig(
            name="A_baseline_no_dedup",
            description="Baseline：逐帧检测，不做去重（展示 over-counting 问题）",
            group="A",
            skip_registration=True,
            use_dbscan=False,
            min_observations=1,
            dist_thresh=0.0,   # 距离阈值=0，每次检测都成为新 Cluster
        ),
        AblationConfig(
            name="A_incremental_cluster",
            description="主线：Homography + 增量聚类去重",
            group="A",
            skip_registration=False,
            use_dbscan=False,
            dist_thresh=40.0,
            min_observations=1,
        ),
        AblationConfig(
            name="A_dbscan_cluster",
            description="主线：Homography + DBSCAN 去重（更稳定）",
            group="A",
            skip_registration=False,
            use_dbscan=True,
            dist_thresh=40.0,
            min_observations=1,
        ),
    ]


def get_group_b_configs() -> List[AblationConfig]:
    """
    消融实验 B：分类策略对比。

    方案对比：
    1. single_crop  : 只用 best_crop 单次分类（无投票）
    2. vote_uniform : Cluster 级等权投票（温度=1.0）
    3. vote_conf    : Cluster 级按置信度加权投票
    """
    return [
        AblationConfig(
            name="B_single_crop",
            description="分类策略：仅用 best_crop 单次分类（无投票）",
            group="B",
            use_vote=False,
        ),
        AblationConfig(
            name="B_vote_uniform",
            description="分类策略：Cluster 级等权投票（温度=1.0）",
            group="B",
            use_vote=True,
            vote_temperature=1.0,
            extra={"conf_weight_exponent": 0.0},  # 等权
        ),
        AblationConfig(
            name="B_vote_conf_weighted",
            description="分类策略：Cluster 级置信度加权投票（温度=1.0，权重指数=2.0）",
            group="B",
            use_vote=True,
            vote_temperature=1.0,
            extra={"conf_weight_exponent": 2.0},
        ),
    ]


def get_group_c_configs() -> List[AblationConfig]:
    """
    消融实验 C：SAHI 效果对比（针对高分辨率 4K 视频）。

    方案对比：
    1. no_sahi  : 直接整图推理
    2. sahi     : SAHI 切片推理（640×640 切片 + NMS 合并）
    """
    return [
        AblationConfig(
            name="C_no_sahi",
            description="检测策略：整图直接推理（不使用 SAHI）",
            group="C",
            use_sahi=False,
        ),
        AblationConfig(
            name="C_with_sahi",
            description="检测策略：SAHI 切片推理（640×640, 重叠 20%）",
            group="C",
            use_sahi=True,
        ),
    ]


def get_all_configs() -> Dict[str, List[AblationConfig]]:
    """返回所有消融实验配置（按组别分类）。"""
    return {
        "A": get_group_a_configs(),
        "B": get_group_b_configs(),
        "C": get_group_c_configs(),
    }


# ---------------------------------------------------------------------------
# 实验运行器
# ---------------------------------------------------------------------------

class AblationRunner:
    """
    消融实验运行器。

    根据 AblationConfig 配置运行 pipeline，记录结果，
    支持与 GT 对比计算误差和得分。

    用法
    ----
    >>> runner = AblationRunner(data_dir="vedio_exp/", gt_path="gt.npy")
    >>> results = runner.run_configs(get_group_a_configs())
    >>> runner.save_results(results, "ablation_results/")
    >>> runner.print_table(results)
    """

    def __init__(
        self,
        data_dir: str | Path,
        gt_path: Optional[str | Path] = None,
        output_dir: Optional[str | Path] = None,
    ) -> None:
        """
        Parameters
        ----------
        data_dir : str | Path
            包含测试视频的文件夹路径。
        gt_path : str | Path | None
            真实标签 .npy 文件路径（格式与 result.npy 相同）。
            若为 None，则不计算误差和得分。
        output_dir : str | Path | None
            结果保存目录；None 时不自动保存。
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else None
        self.gt: Optional[Dict[str, List[int]]] = None

        # 加载 GT（真实标签）
        if gt_path is not None:
            self.gt = self._load_gt(gt_path)

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_gt(self, gt_path: str | Path) -> Dict[str, List[int]]:
        """加载真实标签 .npy 文件。"""
        gt_path = Path(gt_path)
        if not gt_path.exists():
            logger.warning("GT 文件不存在: %s，将跳过误差计算。", gt_path)
            return {}
        try:
            gt = np.load(str(gt_path), allow_pickle=True).item()
            logger.info("已加载 GT: %s（%d 段视频）", gt_path, len(gt))
            return gt
        except Exception as e:
            logger.error("GT 文件加载失败: %s", e)
            return {}

    def run_config(self, config: AblationConfig) -> AblationResult:
        """
        运行单个消融实验配置。

        Parameters
        ----------
        config : AblationConfig
            实验配置。

        Returns
        -------
        AblationResult : 实验结果（含计数、耗时和可选的误差）。
        """
        import datetime

        logger.info("")
        logger.info("=" * 60)
        logger.info("运行消融实验: [%s]", config.name)
        logger.info("说明: %s", config.description)
        logger.info("=" * 60)

        t_start = time.perf_counter()

        try:
            counts, per_video_time, notes = self._run_pipeline(config)
        except Exception as e:
            logger.error("实验 '%s' 运行失败: %s", config.name, e, exc_info=True)
            counts = {}
            per_video_time = {}
            notes = f"运行失败: {e}"

        total_time = time.perf_counter() - t_start

        # 计算误差和得分（若有 GT）
        errors: Dict[str, List[int]] = {}
        mae_per_type: List[float] = []
        overall_mae: float = 0.0
        score: float = 0.0

        if self.gt and counts:
            score, errors, mae_per_type, overall_mae = compute_score(counts, self.gt)
            logger.info(
                "得分: %.1f / 40.0  |  MAE: %.3f  |  耗时: %.2fs",
                score, overall_mae, total_time,
            )
        else:
            logger.info("耗时: %.2fs（无 GT，跳过误差计算）", total_time)

        result = AblationResult(
            config_name=config.name,
            counts=counts,
            total_time=total_time,
            per_video_time=per_video_time,
            errors=errors,
            mae_per_type=mae_per_type,
            overall_mae=overall_mae,
            score=score,
            notes=notes,
            timestamp=datetime.datetime.now().isoformat(),
        )

        # 自动保存单个结果
        if self.output_dir:
            self._save_single_result(result, config)

        return result

    def run_configs(
        self,
        configs: List[AblationConfig],
        stop_on_error: bool = False,
    ) -> List[AblationResult]:
        """
        批量运行多个消融实验配置。

        Parameters
        ----------
        configs : List[AblationConfig]
            实验配置列表。
        stop_on_error : bool
            True: 某个实验失败时立即停止；False: 跳过失败继续运行。

        Returns
        -------
        List[AblationResult] : 与 configs 等长的结果列表。
        """
        results: List[AblationResult] = []
        total = len(configs)

        logger.info("开始运行 %d 个消融实验...", total)

        for i, config in enumerate(configs):
            logger.info("")
            logger.info("[%d/%d] 实验: %s", i + 1, total, config.name)

            result = self.run_config(config)
            results.append(result)

            if stop_on_error and result.notes.startswith("运行失败"):
                logger.error("实验 '%s' 失败，终止批量运行。", config.name)
                break

        logger.info("")
        logger.info("=" * 60)
        logger.info("消融实验批量运行完成: %d / %d", len(results), total)
        logger.info("=" * 60)

        return results

    def _run_pipeline(
        self,
        config: AblationConfig,
    ) -> Tuple[Dict[str, List[int]], Dict[str, float], str]:
        """
        根据 config 配置运行完整 pipeline。

        Returns
        -------
        counts : Dict[str, List[int]]
        per_video_time : Dict[str, float]
        notes : str
        """
        # 将 sys.path 配置好（支持从 tools/ 调用 pipeline）
        project_root = Path(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from pipeline import VideoPipeline
        from utils.video_io import list_videos, get_video_name

        notes_list: List[str] = []

        # ---- 构建 pipeline 初始化参数 ----
        pipeline_kwargs = dict(
            use_fp16=config.use_fp16,
            device=config.device,
            keyframe_strategy=config.keyframe_strategy,
            dist_thresh=config.dist_thresh if not config.skip_registration else 0.0,
            min_observations=config.min_observations,
            use_dbscan=config.use_dbscan,
        )

        if config.detector_weights:
            pipeline_kwargs["detector_weights"] = Path(config.detector_weights)
        if config.classifier_weights:
            pipeline_kwargs["classifier_weights"] = Path(config.classifier_weights)

        pipeline = VideoPipeline(**pipeline_kwargs)

        # 针对特殊配置进行模块级覆盖
        if config.use_sahi is not None:
            pipeline.detector._yolo.use_sahi = config.use_sahi
            if not config.use_sahi:
                notes_list.append("SAHI 已禁用")

        if not config.use_vote:
            # 关闭投票：将 conf_weight_exponent 设为 0，MAX_VOTES 设为 1
            pipeline.classifier.conf_weight_exponent = 0.0
            # 通过设置超大置信度阈值模拟"只用 best_crop"
            notes_list.append("分类投票已禁用（仅用 best_crop）")

        if config.extra.get("conf_weight_exponent") is not None:
            pipeline.classifier.conf_weight_exponent = config.extra["conf_weight_exponent"]

        if config.vote_temperature != 1.0:
            pipeline.classifier.vote_temperature = config.vote_temperature

        # 若跳过配准，需要特殊处理（让 registration 全部返回 valid=False + identity）
        if config.skip_registration:
            notes_list.append("跳过配准（Baseline 模式）")

        # ---- 逐视频运行 ----
        videos = list_videos(self.data_dir)
        counts: Dict[str, List[int]] = {}
        per_video_time: Dict[str, float] = {}

        for video_path in videos:
            video_name = get_video_name(video_path)
            t0 = time.perf_counter()

            try:
                # skip_registration 模式：每帧独立检测，不做去重
                if config.skip_registration:
                    c, elapsed = self._run_no_dedup(
                        video_path, pipeline, config
                    )
                    counts[video_name] = c
                    per_video_time[video_name] = elapsed
                else:
                    result = pipeline.process_video(video_path)
                    counts[result.video_name] = result.counts
                    per_video_time[result.video_name] = time.perf_counter() - t0

            except Exception as e:
                logger.error("视频 '%s' 处理失败: %s", video_name, e)
                counts[video_name] = [0] * 5
                per_video_time[video_name] = time.perf_counter() - t0
                notes_list.append(f"'{video_name}' 处理失败: {e}")

        notes = "; ".join(notes_list) if notes_list else "正常运行"
        return counts, per_video_time, notes

    def _run_no_dedup(
        self,
        video_path: Path,
        pipeline,
        config: AblationConfig,
    ) -> Tuple[List[int], float]:
        """
        Baseline 模式：逐帧检测，不做跨帧去重，直接汇总所有检测数。

        这会导致严重的 over-counting（同一螺丝在多帧中被重复计数），
        用于在消融实验 A 中展示去重的必要性。
        """
        from utils.video_io import VideoReader

        t0 = time.perf_counter()
        total_counts = [0] * 5

        with VideoReader(video_path) as reader:
            meta = reader.meta
            # 均匀采样关键帧
            target = min(30, meta.frame_count)
            frame_ids = list(np.linspace(0, meta.frame_count - 1, target, dtype=int))

            for fid, frame_hr, _ in reader.iter_frames_at(frame_ids, yield_low_res=False):
                if frame_hr is None:
                    continue
                # 检测
                dets = pipeline.detector.detect(frame_hr, frame_id=fid)
                # 分类（每个 detection 独立分类，不做聚类）
                if dets:
                    crops = [d.crop for d in dets]
                    probs = pipeline.classifier._backend.predict_probs(crops)
                    preds = probs.argmax(axis=1)
                    for p in preds:
                        if 0 <= int(p) < 5:
                            total_counts[int(p)] += 1

        elapsed = time.perf_counter() - t0
        return total_counts, elapsed

    def _save_single_result(
        self,
        result: AblationResult,
        config: AblationConfig,
    ) -> None:
        """将单个实验结果保存为 JSON 文件。"""
        if not self.output_dir:
            return

        save_path = self.output_dir / f"{result.config_name}.json"
        data = {
            "config": config.to_dict(),
            "result": result.to_dict(),
        }
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        logger.info("实验结果已保存: %s", save_path)

    def save_results(
        self,
        results: List[AblationResult],
        output_dir: Optional[str | Path] = None,
    ) -> Path:
        """
        将多个实验结果保存为汇总 JSON 文件。

        Parameters
        ----------
        results : List[AblationResult]
            实验结果列表。
        output_dir : str | Path | None
            输出目录；None 时使用 self.output_dir。

        Returns
        -------
        Path : 保存的 JSON 文件路径。
        """
        import datetime

        save_dir = Path(output_dir) if output_dir else self.output_dir
        if save_dir is None:
            save_dir = Path("ablation_results")
        save_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = save_dir / f"ablation_summary_{timestamp}.json"

        data = {
            "meta": {
                "generated_at": datetime.datetime.now().isoformat(),
                "n_experiments": len(results),
                "data_dir": str(self.data_dir),
                "has_gt": self.gt is not None,
            },
            "results": [r.to_dict() for r in results],
        }
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

        logger.info("汇总结果已保存: %s", save_path)
        return save_path

    @staticmethod
    def load_results(json_path: str | Path) -> List[AblationResult]:
        """
        从 JSON 文件加载消融实验结果。

        Parameters
        ----------
        json_path : str | Path
            汇总 JSON 文件路径。

        Returns
        -------
        List[AblationResult]
        """
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        results = []
        for r_dict in data.get("results", []):
            try:
                results.append(AblationResult.from_dict(r_dict))
            except Exception as e:
                logger.warning("加载结果条目失败: %s", e)

        logger.info("已加载 %d 个实验结果: %s", len(results), json_path)
        return results


# ---------------------------------------------------------------------------
# 结果展示与报告生成
# ---------------------------------------------------------------------------

class AblationReporter:
    """
    消融实验报告生成器。

    将 AblationResult 列表格式化为：
    - 控制台表格（tabulate 格式）
    - CSV 文件（用于 Excel 分析）
    - LaTeX 表格（直接粘贴到实验报告）
    - Markdown 表格（GitHub / 文档用）
    """

    def __init__(self, results: List[AblationResult]) -> None:
        self.results = results

    # ------------------------------------------------------------------
    # 控制台输出
    # ------------------------------------------------------------------

    def print_summary(self) -> None:
        """打印消融实验汇总表格（控制台）。"""
        if not self.results:
            print("没有可显示的结果。")
            return

        # 尝试使用 tabulate 美化输出
        rows = self._build_summary_rows()
        headers = self._build_summary_headers()

        try:
            from tabulate import tabulate
            print("\n" + tabulate(rows, headers=headers, tablefmt="fancy_grid", floatfmt=".3f"))
        except ImportError:
            # 不依赖 tabulate：手动格式化
            print("\n消融实验汇总：")
            print("  " + "  ".join(f"{h:>15s}" for h in headers))
            print("  " + "-" * (17 * len(headers)))
            for row in rows:
                print("  " + "  ".join(f"{str(v):>15s}" for v in row))
        print()

    def print_per_video(self) -> None:
        """打印各视频的详细计数结果表格。"""
        if not self.results:
            return

        # 收集所有视频名称
        all_videos = sorted(set(
            vname
            for r in self.results
            for vname in r.counts.keys()
        ))

        for r in self.results:
            print(f"\n[{r.config_name}]  耗时={r.total_time:.2f}s")
            print(f"  {'视频':30s}  T1  T2  T3  T4  T5  总计", end="")
            if r.has_gt():
                print("  MAE", end="")
            print()
            print("  " + "-" * 60)

            for vname in all_videos:
                counts = r.counts.get(vname, [0] * 5)
                row_str = f"  {vname:30s}  " + "  ".join(f"{c:2d}" for c in counts)
                row_str += f"  {sum(counts):3d}"
                if r.has_gt() and vname in r.errors:
                    errs = r.errors[vname]
                    mae = float(np.mean(errs))
                    row_str += f"  {mae:.1f}"
                print(row_str)

    def print_score_breakdown(self) -> None:
        """打印各实验的评分明细。"""
        if not any(r.has_gt() for r in self.results):
            print("无 GT 数据，跳过评分明细。")
            return

        print("\n评分明细（满分 40 分）：")
        print(f"  {'实验名称':35s}  {'得分':>6s}  {'MAE':>6s}  {'Type1':>6s}  {'Type2':>6s}  "
              f"{'Type3':>6s}  {'Type4':>6s}  {'Type5':>6s}")
        print("  " + "-" * 90)

        for r in self.results:
            if not r.has_gt():
                continue
            mae_str = " | ".join(f"{m:.2f}" for m in r.mae_per_type)
            type_maes = r.mae_per_type if r.mae_per_type else [0.0] * 5
            print(
                f"  {r.config_name:35s}  "
                f"{r.score:>6.1f}  "
                f"{r.overall_mae:>6.3f}  "
                + "  ".join(f"{m:>6.2f}" for m in type_maes)
            )
        print()

    # ------------------------------------------------------------------
    # 文件导出
    # ------------------------------------------------------------------

    def export_csv(self, output_path: str | Path) -> Path:
        """
        导出为 CSV 格式（可用 Excel 打开）。

        Parameters
        ----------
        output_path : str | Path
            输出 CSV 文件路径。

        Returns
        -------
        Path : 保存的文件路径。
        """
        import csv

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        rows = self._build_summary_rows()
        headers = self._build_summary_headers()

        with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)

        logger.info("CSV 已导出: %s", output_path)
        return output_path

    def export_markdown(self, output_path: str | Path) -> Path:
        """
        导出为 Markdown 格式表格。

        Parameters
        ----------
        output_path : str | Path
            输出 Markdown 文件路径。

        Returns
        -------
        Path : 保存的文件路径。
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        rows = self._build_summary_rows()
        headers = self._build_summary_headers()

        lines = []
        lines.append("# 消融实验结果汇总\n")
        lines.append("| " + " | ".join(str(h) for h in headers) + " |")
        lines.append("| " + " | ".join("---" for _ in headers) + " |")
        for row in rows:
            lines.append("| " + " | ".join(str(v) for v in row) + " |")
        lines.append("")

        # 添加各组说明
        groups_present = set()
        for r in self.results:
            # 从 config_name 推断 group
            for g in ("A", "B", "C"):
                if r.config_name.startswith(g + "_"):
                    groups_present.add(g)
                    break

        if "A" in groups_present:
            lines.extend([
                "## 消融实验 A：去重策略对比\n",
                "- **Baseline**：逐帧检测，不做去重，展示 over-counting 问题",
                "- **增量聚类**：Homography 配准 + 贪心最近邻聚类",
                "- **DBSCAN 聚类**：Homography 配准 + DBSCAN（更稳定，对噪声鲁棒）",
                "",
            ])
        if "B" in groups_present:
            lines.extend([
                "## 消融实验 B：分类策略对比\n",
                "- **单帧分类**：仅用 best_crop 进行一次分类预测",
                "- **等权投票**：对 Cluster 所有观测等权投票",
                "- **置信度加权投票**：按检测置信度加权投票（本项目主线）",
                "",
            ])
        if "C" in groups_present:
            lines.extend([
                "## 消融实验 C：SAHI 效果对比\n",
                "- **整图推理**：直接将整帧送入 YOLO，适合 ≤1080p 视频",
                "- **SAHI 切片**：640×640 切片推理 + NMS 合并，适合 4K 高分辨率视频",
                "",
            ])

        output_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Markdown 已导出: %s", output_path)
        return output_path

    def export_latex(self, output_path: str | Path) -> Path:
        """
        导出为 LaTeX 表格（可直接粘贴到实验报告）。

        Parameters
        ----------
        output_path : str | Path
            输出 .tex 文件路径。

        Returns
        -------
        Path : 保存的文件路径。
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        rows = self._build_summary_rows()
        headers = self._build_summary_headers()
        n_cols = len(headers)

        lines = [
            "% 消融实验结果汇总表",
            "% 由 tools/ablation.py 自动生成",
            r"\begin{table}[htbp]",
            r"  \centering",
            r"  \caption{消融实验结果汇总}",
            r"  \label{tab:ablation}",
            r"  \begin{tabular}{" + "l" + "r" * (n_cols - 1) + "}",
            r"    \toprule",
            "    " + " & ".join(str(h) for h in headers) + r" \\",
            r"    \midrule",
        ]
        for row in rows:
            lines.append("    " + " & ".join(str(v) for v in row) + r" \\")
        lines.extend([
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
        ])

        output_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("LaTeX 已导出: %s", output_path)
        return output_path

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _build_summary_headers(self) -> List[str]:
        """构建汇总表格的列标题。"""
        headers = ["实验名称", "描述", "耗时(s)"]
        if any(r.has_gt() for r in self.results):
            headers += ["得分(/40)", "MAE", "MAE_T1", "MAE_T2", "MAE_T3", "MAE_T4", "MAE_T5"]
        return headers

    def _build_summary_rows(self) -> List[List]:
        """构建汇总表格的数据行。"""
        rows = []
        has_gt = any(r.has_gt() for r in self.results)

        for r in self.results:
            row = [
                r.config_name,
                r.notes[:30] + "..." if len(r.notes) > 30 else r.notes,
                f"{r.total_time:.2f}",
            ]
            if has_gt:
                if r.has_gt():
                    row += [
                        f"{r.score:.1f}",
                        f"{r.overall_mae:.3f}",
                    ]
                    row += [f"{m:.3f}" for m in (r.mae_per_type or [0.0] * 5)]
                else:
                    row += ["N/A"] * 7
            rows.append(row)

        return rows


# ---------------------------------------------------------------------------
# 命令行接口
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        prog="ablation.py",
        description="消融实验记录与对比工具 (Owner: D)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 运行所有消融实验
  python tools/ablation.py --data_dir vedio_exp/ --output ablation_results/

  # 仅运行去重策略对比（实验组 A）
  python tools/ablation.py --data_dir vedio_exp/ --output ablation_results/ --group A

  # 与真实标签对比计算误差
  python tools/ablation.py --data_dir vedio_exp/ --output ablation_results/ \\
      --gt_path gt.npy

  # 从已有结果文件生成报告（不重新运行）
  python tools/ablation.py --report_only \\
      --results_path ablation_results/ablation_summary_*.json

  # 导出 Markdown + LaTeX 报告
  python tools/ablation.py --data_dir vedio_exp/ --output ablation_results/ \\
      --export_markdown --export_latex
        """,
    )

    # ---- 运行模式 ----
    mode_group = parser.add_argument_group("运行模式（二选一）")
    mode_group.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="包含测试视频的文件夹路径（运行模式时必须提供）。",
    )
    mode_group.add_argument(
        "--report_only",
        action="store_true",
        default=False,
        help="不运行实验，仅从已有 JSON 结果文件生成报告。",
    )
    mode_group.add_argument(
        "--results_path",
        type=str,
        default=None,
        help="--report_only 模式下的汇总 JSON 文件路径。",
    )

    # ---- 实验配置 ----
    exp_group = parser.add_argument_group("实验配置")
    exp_group.add_argument(
        "--group",
        type=str,
        default="all",
        choices=["all", "A", "B", "C"],
        help="要运行的消融实验组别（默认：all，运行所有三组）。",
    )
    exp_group.add_argument(
        "--gt_path",
        type=str,
        default=None,
        help="真实标签 .npy 文件路径（格式与 result.npy 相同）；不提供则跳过误差计算。",
    )

    # ---- 输出 ----
    out_group = parser.add_argument_group("输出")
    out_group.add_argument(
        "--output", "-o",
        type=str,
        default="ablation_results",
        help="结果保存目录（默认：ablation_results/）。",
    )
    out_group.add_argument(
        "--export_markdown",
        action="store_true",
        default=False,
        help="同时导出 Markdown 格式报告。",
    )
    out_group.add_argument(
        "--export_latex",
        action="store_true",
        default=False,
        help="同时导出 LaTeX 格式表格（可粘贴到实验报告）。",
    )
    out_group.add_argument(
        "--export_csv",
        action="store_true",
        default=False,
        help="同时导出 CSV 格式（可用 Excel 分析）。",
    )

    # ---- 其他 ----
    parser.add_argument(
        "--stop_on_error",
        action="store_true",
        default=False,
        help="某个实验失败时立即停止（默认：跳过失败继续运行）。",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="输出详细日志。",
    )

    return parser.parse_args()


def main() -> int:
    """
    消融实验工具主函数。

    Returns
    -------
    int : 退出码（0=成功，非 0=错误）。
    """
    args = _parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ================================================================
    # 模式 1：从已有结果文件生成报告（不重新运行）
    # ================================================================
    if args.report_only:
        if not args.results_path:
            logger.error("--report_only 模式下必须提供 --results_path。")
            return 1

        results_path = Path(args.results_path)
        if not results_path.exists():
            logger.error("结果文件不存在: %s", results_path)
            return 1

        results = AblationRunner.load_results(results_path)
        if not results:
            logger.error("从文件加载结果为空: %s", results_path)
            return 1

        reporter = AblationReporter(results)
        reporter.print_summary()
        reporter.print_per_video()
        reporter.print_score_breakdown()

        if args.export_markdown:
            reporter.export_markdown(output_dir / "ablation_report.md")
        if args.export_latex:
            reporter.export_latex(output_dir / "ablation_table.tex")
        if args.export_csv:
            reporter.export_csv(output_dir / "ablation_results.csv")

        return 0

    # ================================================================
    # 模式 2：运行消融实验
    # ================================================================
    if not args.data_dir:
        logger.error("运行模式下必须提供 --data_dir。")
        return 1

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        logger.error("--data_dir 不是有效目录: %s", data_dir)
        return 1

    # ---- 选取要运行的实验配置 ----
    all_configs = get_all_configs()
    if args.group == "all":
        configs = []
        for g in ("A", "B", "C"):
            configs.extend(all_configs[g])
    else:
        configs = all_configs.get(args.group, [])

    if not configs:
        logger.error("未找到实验配置，group='%s'", args.group)
        return 1

    logger.info("将运行 %d 个消融实验（组别: %s）", len(configs), args.group)
    for cfg in configs:
        logger.info("  [%s] %s", cfg.name, cfg.description)

    # ---- 初始化运行器 ----
    runner = AblationRunner(
        data_dir=data_dir,
        gt_path=args.gt_path,
        output_dir=output_dir,
    )

    # ---- 运行实验 ----
    t_start = time.perf_counter()
    results = runner.run_configs(configs, stop_on_error=args.stop_on_error)
    total_elapsed = time.perf_counter() - t_start

    if not results:
        logger.error("所有实验均未产生结果。")
        return 1

    # ---- 保存汇总结果 ----
    summary_path = runner.save_results(results, output_dir)

    # ---- 生成报告 ----
    reporter = AblationReporter(results)
    reporter.print_summary()
    reporter.print_per_video()
    reporter.print_score_breakdown()

    if args.export_markdown:
        reporter.export_markdown(output_dir / "ablation_report.md")
    if args.export_latex:
        reporter.export_latex(output_dir / "ablation_table.tex")
    if args.export_csv:
        reporter.export_csv(output_dir / "ablation_results.csv")

    logger.info("")
    logger.info("消融实验全部完成，总耗时: %.2fs", total_elapsed)
    logger.info("结果已保存至: %s", output_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
