"""
modules/classifier.py - 5 类螺丝分类器模块
Owner: C（分类与结果融合）

职责：
  - 加载从 Lab2 迁移的 5 类螺丝分类器（或 fine-tune 后的版本）
  - 接收 Cluster 列表，对每个 Cluster 的 best_crop 进行分类
  - 实现 Cluster 级投票（对同一 Cluster 的多个观测分别预测，取加权投票结果）
  - 将分类结果（class_probs、pred_class）写回 Cluster 对象
  - 输出最终带类别标签的 Cluster 列表

TODO (C)：
  [ ] 将 Lab2 分类器权重迁移至 models/classifier.pt（或 models/classifier.pth）
  [ ] 在视频 crop 上测试 baseline 准确率，若 < 80% 立即启动 fine-tune
  [ ] 实现域适应数据增强（随机模糊、亮度抖动、旋转、缩放）
  [ ] 实现 Cluster 级多观测投票（classify_cluster_with_votes）
  [ ] 分析混淆矩阵，针对易混类别增加训练样本
  [ ] 验证：classify_clusters() 调用后所有 Cluster.pred_class != -1

依赖：torch, torchvision, opencv-python, numpy
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from interfaces import Cluster, Detection

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 超参数（C 负责调优）
# ---------------------------------------------------------------------------

NUM_CLASSES: int = 5
"""螺丝类别数（Type_1 ~ Type_5），与作业规范一致，禁止修改。"""

INPUT_SIZE: int = 224
"""分类器输入图像边长（正方形），与 Lab2 保持一致。"""

BATCH_SIZE: int = 32
"""分类推理的批大小。GPU 显存不足时适当降低。"""

USE_FP16: bool = True
"""是否使用 FP16 半精度推理（需要 CUDA GPU）。"""

VOTE_TEMPERATURE: float = 1.0
"""
Cluster 级投票的 Softmax 温度参数。
温度越低，投票越"硬"（高置信度预测权重越大）；
温度越高，投票越"软"（趋向均匀分布）。
TODO (C)：在验证集上调整此参数。
"""

CONF_WEIGHT_EXPONENT: float = 2.0
"""
投票时检测置信度的加权指数。
投票权重 = detection.confidence ** CONF_WEIGHT_EXPONENT
设为 0 时等同于等权投票；设为 2 时高置信度检测权重更大。
TODO (C)：可设为 0 简化，在效果差时再调整。
"""

# 模型权重路径（相对于项目根目录）
DEFAULT_WEIGHTS = Path(__file__).parent.parent / "models" / "classifier.pt"


def _normalize_detector_class_name(name: str) -> str:
    """将 detector 输出的类别名归一化为 'type1' 这种形式。"""
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _detector_name_to_pred_class(name: str) -> int:
    """
    将 detector 的类别名映射到 0-indexed 的 pred_class。

    支持:
    - type1 ~ type5
    - type_1 ~ type_5（归一化后同样可识别）
    """
    norm = _normalize_detector_class_name(name)
    if norm.startswith("type") and norm[4:].isdigit():
        idx = int(norm[4:]) - 1
        if 0 <= idx < NUM_CLASSES:
            return idx
    return -1


def _classify_cluster_from_detector_votes(
    cluster: Cluster,
    conf_weight_exponent: float = CONF_WEIGHT_EXPONENT,
) -> Cluster:
    """
    在分类器不可用时，直接使用 multi-class detector 的类别结果做 Cluster 投票。

    仅当 observations 中存在可识别的 type1~type5 类别时才应调用。
    """
    weights = np.zeros(NUM_CLASSES, dtype=np.float32)

    for det in cluster.observations:
        cls_idx = _detector_name_to_pred_class(getattr(det, "class_name", ""))
        if cls_idx < 0:
            continue
        weights[cls_idx] += float(det.confidence) ** conf_weight_exponent

    if weights.sum() <= 0:
        return cluster

    probs = weights / weights.sum()
    cluster.class_probs = probs.astype(np.float32)
    cluster.pred_class = int(probs.argmax())
    logger.debug(
        "Cluster #%d 使用 detector 多类结果完成兼容分类: %s (probs=%s)",
        cluster.cluster_id,
        cluster.type_label,
        np.round(probs, 3),
    )
    return cluster


def _cluster_has_detector_multiclass_labels(cluster: Cluster) -> bool:
    """检查一个 Cluster 是否包含可直接用于计数的 detector 多类标签。"""
    for det in cluster.observations:
        if _detector_name_to_pred_class(getattr(det, "class_name", "")) >= 0:
            return True
    return False


# ---------------------------------------------------------------------------
# 图像预处理工具
# ---------------------------------------------------------------------------

def _preprocess_crop(
    crop: np.ndarray,
    input_size: int = INPUT_SIZE,
    augment: bool = False,
) -> np.ndarray:
    """
    将 crop 图像预处理为分类器输入张量所需的格式。

    处理流程：
    1. 等比缩放 + Letterbox 填充到 input_size × input_size
    2. BGR → RGB
    3. 归一化到 [0, 1]
    4. 标准化（ImageNet mean/std）

    Parameters
    ----------
    crop : np.ndarray
        输入的螺丝 crop 图像（H×W×3，BGR，uint8）。
    input_size : int
        目标边长（默认 224）。
    augment : bool
        是否应用数据增强（仅用于 fine-tune 数据处理，推理时设为 False）。

    Returns
    -------
    np.ndarray
        形状 (3, input_size, input_size) 的 float32 数组（CHW 格式），
        已归一化，可直接转换为 torch.Tensor。
    """
    if crop is None or crop.size == 0:
        return np.zeros((3, input_size, input_size), dtype=np.float32)

    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((3, input_size, input_size), dtype=np.float32)

    # Step 1: 等比缩放 + Letterbox 填充
    scale = input_size / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    top = (input_size - new_h) // 2
    left = (input_size - new_w) // 2
    canvas[top: top + new_h, left: left + new_w] = resized

    # Step 2: 数据增强（fine-tune 时使用）
    if augment:
        canvas = _apply_augmentation(canvas)

    # Step 3: BGR → RGB
    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    # Step 4: 归一化 + 标准化（ImageNet）
    img = canvas_rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std

    # Step 5: HWC → CHW
    return img.transpose(2, 0, 1)


def _apply_augmentation(image: np.ndarray) -> np.ndarray:
    """
    域适应数据增强（用于 fine-tune 数据生成）。

    增强策略（参考 plan.md 6.2 节）：
    - 随机高斯模糊（模拟运动模糊和对焦不准）
    - 亮度/对比度随机抖动
    - 随机旋转 0-360°
    - 随机缩放 0.8-1.2×

    Parameters
    ----------
    image : np.ndarray
        输入图像（H×W×3，BGR，uint8）。

    Returns
    -------
    np.ndarray : 增强后的图像（H×W×3，BGR，uint8）。

    TODO (C)：根据 baseline 测试结果调整增强强度。
    """
    h, w = image.shape[:2]
    result = image.copy()

    # 随机高斯模糊
    if random.random() < 0.4:
        ksize = random.choice([3, 5, 7])
        result = cv2.GaussianBlur(result, (ksize, ksize), 0)

    # 亮度/对比度抖动
    if random.random() < 0.5:
        alpha = random.uniform(0.7, 1.3)   # 对比度
        beta = random.randint(-30, 30)       # 亮度偏移
        result = np.clip(result.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    # 随机旋转（0-360°，螺丝是旋转对称的）
    if random.random() < 0.8:
        angle = random.uniform(0, 360)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        result = cv2.warpAffine(result, M, (w, h), borderValue=(114, 114, 114))

    # 随机缩放（裁剪中心区域）
    if random.random() < 0.3:
        scale = random.uniform(0.8, 1.2)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        result = cv2.resize(result, (new_w, new_h))
        # 重新裁剪/填充回原尺寸
        canvas = np.full((h, w, 3), 114, dtype=np.uint8)
        cy = (h - new_h) // 2
        cx = (w - new_w) // 2
        # 处理缩小情况
        if scale < 1.0:
            canvas[max(0, cy): max(0, cy) + new_h,
                   max(0, cx): max(0, cx) + new_w] = result[:min(new_h, h), :min(new_w, w)]
        else:
            # 处理放大情况（从中心裁剪）
            src_y = max(0, -cy)
            src_x = max(0, -cx)
            canvas = result[src_y: src_y + h, src_x: src_x + w]
            if canvas.shape[:2] != (h, w):
                canvas = cv2.resize(canvas, (w, h))
        result = canvas

    return result


# ---------------------------------------------------------------------------
# 兜底分类器（C 不需要修改，仅在模型权重缺失时使用）
# ---------------------------------------------------------------------------

class _FallbackClassifier:
    """
    基于图像特征（颜色直方图 + 形状特征）的简单兜底分类器。

    精度极低（随机接近），仅用于：
    1. 分类器权重尚未迁移/训练时验证 pipeline 连通性
    2. 快速 smoke test

    C 无需修改此类，专注于实现 TorchClassifier。
    """

    def __init__(self) -> None:
        logger.warning(
            "⚠️  使用兜底分类器（随机 + 颜色启发式）。"
            "精度极低！请尽快提供 models/classifier.pt。"
        )

    def predict_probs(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        对一批 crop 返回伪分类概率（均匀分布 + 随机扰动）。

        Parameters
        ----------
        crops : List[np.ndarray]
            crop 图像列表。

        Returns
        -------
        np.ndarray : shape (N, 5)，每行为一个 crop 的 5 类概率（已 softmax 归一化）。
        """
        n = len(crops)
        # 均匀分布 + 小随机扰动，使结果稍有差异（避免全部预测同一类）
        logits = np.ones((n, NUM_CLASSES), dtype=np.float32) + \
                 np.random.uniform(-0.3, 0.3, size=(n, NUM_CLASSES)).astype(np.float32)
        # Softmax
        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        return probs

    def predict(self, crops: List[np.ndarray]) -> List[int]:
        """
        对一批 crop 返回预测类别（0-indexed）。

        Parameters
        ----------
        crops : List[np.ndarray]

        Returns
        -------
        List[int] : 每个 crop 的预测类别（0=Type_1, …, 4=Type_5）。
        """
        probs = self.predict_probs(crops)
        return probs.argmax(axis=1).tolist()


# ---------------------------------------------------------------------------
# PyTorch 分类器（C 负责实现）
# ---------------------------------------------------------------------------

class TorchClassifier:
    """
    基于 PyTorch 的 5 类螺丝分类器。

    模型结构建议（C 决定）：
    - 推荐使用 Lab2 中已训练的模型架构（如 ResNet18 / EfficientNet-B0）
    - 最后一层 fc 替换为 Linear(in_features, 5)
    - 加载预训练权重后 fine-tune 末端几层

    TODO (C)：
    1. 确认 Lab2 分类器的模型架构和权重格式
    2. 将权重保存至 models/classifier.pt（torch.save 格式）
    3. 实现 _build_model() 方法，与 Lab2 架构保持一致
    4. 在视频 crop 上测试 baseline 准确率
    5. 若 baseline < 80%，启动 fine-tune（数据增强 + 小 lr）
    6. 记录混淆矩阵，针对易混类别增加样本

    Parameters
    ----------
    weights_path : str | Path
        分类器权重文件路径（.pt / .pth 格式）。
    input_size : int
        输入图像边长（默认 224）。
    use_fp16 : bool
        是否使用 FP16 推理。
    device : str
        推理设备（'cuda:0' / 'cpu' / '' 自动选择）。
    """

    def __init__(
        self,
        weights_path: str | Path = DEFAULT_WEIGHTS,
        input_size: int = INPUT_SIZE,
        use_fp16: bool = USE_FP16,
        device: str = "",
    ) -> None:
        self.weights_path = Path(weights_path)
        self.input_size = input_size
        self.use_fp16 = use_fp16
        self._model = None
        self._device = None
        self._loaded = False

        self._resolve_device(device)
        self._load_model()

    def _resolve_device(self, device: str) -> None:
        """自动选择推理设备。"""
        try:
            import torch
            if device:
                self._device = torch.device(device)
            elif torch.cuda.is_available():
                self._device = torch.device("cuda:0")
                logger.info("分类器使用 GPU: %s", torch.cuda.get_device_name(0))
            else:
                self._device = torch.device("cpu")
                self.use_fp16 = False  # CPU 不支持 FP16 推理
                logger.info("分类器使用 CPU（FP16 已禁用）。")
        except ImportError:
            logger.error("PyTorch 未安装，请运行: pip install torch torchvision")

    def _build_model(self):
        """
        构建分类器模型结构。

        TODO (C)：
        - 将此处替换为 Lab2 中使用的实际模型架构
        - 确保最后一层输出维度为 NUM_CLASSES=5
        - 示例使用 ResNet18；C 可替换为 EfficientNet、ConvNeXt 等

        Returns
        -------
        torch.nn.Module
        """
        try:
            import torch.nn as nn
            import torchvision.models as models

            # ============================================================
            # TODO (C)：替换为 Lab2 实际使用的模型架构
            # 示例：ResNet18
            # ============================================================
            model = models.resnet18(weights=None)
            # 替换最后一层为 5 类输出
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, NUM_CLASSES)
            return model

        except ImportError:
            logger.error("torchvision 未安装，请运行: pip install torchvision")
            return None

    def _load_model(self) -> None:
        """
        加载分类器权重。

        TODO (C)：
        - 确保 _build_model() 的架构与保存权重时的架构完全一致
        - 若权重文件格式为 state_dict，使用 model.load_state_dict()
        - 若权重文件为完整模型（torch.save(model)），直接 torch.load()
        """
        if not self.weights_path.exists():
            logger.warning(
                "分类器权重文件不存在: %s\n"
                "请将 Lab2 分类器权重（或 fine-tune 后的权重）放至此路径。",
                self.weights_path,
            )
            return

        try:
            import torch

            model = self._build_model()
            if model is None:
                return

            # 尝试加载权重（先尝试 state_dict，再尝试完整模型）
            checkpoint = torch.load(
                str(self.weights_path),
                map_location=self._device,
            )

            if isinstance(checkpoint, dict):
                # checkpoint 是 state_dict 或包含 state_dict 的字典
                if "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                elif "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                elif "model" in checkpoint:
                    state_dict = checkpoint["model"]
                else:
                    # 假设整个 checkpoint 就是 state_dict
                    state_dict = checkpoint

                # 处理 DataParallel 包装（移除 'module.' 前缀）
                cleaned_state_dict = {}
                for k, v in state_dict.items():
                    new_k = k.replace("module.", "") if k.startswith("module.") else k
                    cleaned_state_dict[new_k] = v

                model.load_state_dict(cleaned_state_dict, strict=True)
                logger.info("以 state_dict 方式加载权重成功。")
            else:
                # checkpoint 是完整模型对象
                model = checkpoint
                logger.info("以完整模型方式加载权重成功。")

            model.to(self._device)
            model.eval()

            if self.use_fp16:
                try:
                    model = model.half()
                    logger.debug("分类器已启用 FP16。")
                except Exception as e:
                    logger.warning("FP16 转换失败（%s），使用 FP32。", e)
                    self.use_fp16 = False

            self._model = model
            self._loaded = True
            logger.info("✅ 分类器加载完成: %s", self.weights_path)

        except Exception as e:
            logger.error("分类器加载失败: %s", e, exc_info=True)

    # ------------------------------------------------------------------
    # 推理接口
    # ------------------------------------------------------------------

    def predict_probs(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        对一批 crop 进行推理，返回 5 类概率分布。

        Parameters
        ----------
        crops : List[np.ndarray]
            crop 图像列表（H×W×3 BGR，uint8）。
            列表长度可变（内部自动分 batch 处理）。

        Returns
        -------
        np.ndarray : shape (N, 5)，每行为 softmax 后的 5 类概率。

        TODO (C)：
        - 验证预处理流程与 Lab2 训练时的预处理完全一致
        - 若 Lab2 使用了不同的归一化参数，修改 _preprocess_crop() 中的 mean/std
        """
        if not self._loaded or self._model is None:
            logger.warning("模型未加载，返回均匀分布概率。")
            n = len(crops)
            return np.ones((n, NUM_CLASSES), dtype=np.float32) / NUM_CLASSES

        try:
            import torch
            import torch.nn.functional as F

            all_probs = []

            # 分 batch 处理
            for batch_start in range(0, len(crops), BATCH_SIZE):
                batch_crops = crops[batch_start: batch_start + BATCH_SIZE]

                # 预处理 -> (B, 3, H, W) 张量
                preprocessed = [_preprocess_crop(c, self.input_size, augment=False)
                                 for c in batch_crops]
                batch_arr = np.stack(preprocessed, axis=0)  # (B, 3, H, W)

                tensor = torch.from_numpy(batch_arr).to(self._device)
                if self.use_fp16:
                    tensor = tensor.half()

                with torch.no_grad():
                    logits = self._model(tensor)  # (B, 5)
                    probs = F.softmax(logits.float(), dim=1)  # 确保 softmax 在 FP32 下计算

                all_probs.append(probs.cpu().numpy())

            return np.concatenate(all_probs, axis=0)  # (N, 5)

        except Exception as e:
            logger.error("分类推理失败: %s", e, exc_info=True)
            n = len(crops)
            return np.ones((n, NUM_CLASSES), dtype=np.float32) / NUM_CLASSES

    def predict(self, crops: List[np.ndarray]) -> List[int]:
        """
        对一批 crop 返回预测类别（取 argmax）。

        Parameters
        ----------
        crops : List[np.ndarray]

        Returns
        -------
        List[int] : 每个 crop 的预测类别（0-indexed）。
        """
        probs = self.predict_probs(crops)
        return probs.argmax(axis=1).tolist()


# ---------------------------------------------------------------------------
# Cluster 级分类投票逻辑（C 负责实现）
# ---------------------------------------------------------------------------

def classify_cluster_with_votes(
    cluster: Cluster,
    classifier,
    vote_temperature: float = VOTE_TEMPERATURE,
    conf_weight_exponent: float = CONF_WEIGHT_EXPONENT,
) -> Cluster:
    """
    对单个 Cluster 执行多观测投票分类，并写回 class_probs 和 pred_class。

    投票策略（加权平均）：
    1. 对 Cluster 中所有观测（best_crop + 部分 observation crops）分别推理
    2. 各观测的投票权重 = detection.confidence^CONF_WEIGHT_EXPONENT
    3. 将所有观测的概率向量加权平均，得到最终 class_probs
    4. 取 argmax 作为 pred_class

    Parameters
    ----------
    cluster : Cluster
        待分类的螺丝 Cluster（含 best_crop 和 observations）。
    classifier : TorchClassifier | _FallbackClassifier
        已加载的分类器实例。
    vote_temperature : float
        Softmax 温度（影响概率分布的"硬"/"软"程度）。
    conf_weight_exponent : float
        检测置信度的加权指数。

    Returns
    -------
    Cluster : 已填充 class_probs 和 pred_class 的 Cluster（原地修改并返回）。

    TODO (C)：
    - 调整参与投票的观测数量（当前为全部观测，可改为抽样最多 K 个最佳 crop）
    - 验证投票是否比单 crop 分类有提升
    - 考虑对 best_crop 给予更高权重（如 weight = max(conf) * 2）
    """
    if not cluster.observations:
        logger.warning("Cluster #%d 没有观测，跳过分类。", cluster.cluster_id)
        return cluster

    # ---- 收集投票 crop 列表 ----
    # 策略：始终包含 best_crop，再加入部分 observations 的 crop
    # TODO (C)：可调整抽样策略（如只取置信度最高的 K 个）
    vote_crops: List[np.ndarray] = []
    vote_weights: List[float] = []

    # 1. best_crop（最佳观测，权重加倍）
    if cluster.best_crop is not None and cluster.best_crop.size > 0:
        vote_crops.append(cluster.best_crop)
        # best_crop 对应 observations 中置信度最高的那个
        best_conf = max(d.confidence for d in cluster.observations)
        vote_weights.append((best_conf ** conf_weight_exponent) * 2.0)

    # 2. 其余观测的 crop（每个观测按置信度加权）
    # 限制最多使用 10 个观测，防止单 Cluster 观测过多时推理过慢
    MAX_VOTES = 10
    obs_sorted = sorted(cluster.observations, key=lambda d: d.confidence, reverse=True)
    for det in obs_sorted[:MAX_VOTES]:
        if det.crop is not None and det.crop.size > 0:
            vote_crops.append(det.crop)
            vote_weights.append(det.confidence ** conf_weight_exponent)

    if not vote_crops:
        logger.warning("Cluster #%d 没有有效的 crop，使用均匀概率。", cluster.cluster_id)
        cluster.class_probs = np.ones(NUM_CLASSES, dtype=np.float32) / NUM_CLASSES
        cluster.pred_class = 0  # 默认 Type_1
        return cluster

    # ---- 批量推理 ----
    all_probs = classifier.predict_probs(vote_crops)  # (N_votes, 5)

    # ---- 温度缩放（可选）----
    if vote_temperature != 1.0:
        # 在 log 空间做温度缩放再 softmax
        log_probs = np.log(np.clip(all_probs, 1e-10, 1.0)) / vote_temperature
        exp_lp = np.exp(log_probs - log_probs.max(axis=1, keepdims=True))
        all_probs = exp_lp / exp_lp.sum(axis=1, keepdims=True)

    # ---- 加权平均 ----
    weights_arr = np.array(vote_weights[:len(vote_crops)], dtype=np.float32)
    weights_arr = weights_arr / weights_arr.sum()   # 归一化权重
    cluster_probs = (all_probs * weights_arr[:, np.newaxis]).sum(axis=0)  # (5,)

    # ---- 写回 Cluster ----
    cluster.class_probs = cluster_probs.astype(np.float32)
    cluster.pred_class = int(cluster_probs.argmax())

    logger.debug(
        "Cluster #%d 分类结果: %s (probs=%s, n_votes=%d)",
        cluster.cluster_id,
        cluster.type_label,
        np.round(cluster_probs, 3),
        len(vote_crops),
    )
    return cluster


# ---------------------------------------------------------------------------
# 门面类（统一入口，D 在 pipeline.py 中调用此类）
# ---------------------------------------------------------------------------

class ScrewClassifier:
    """
    螺丝分类器门面类（Facade Pattern）。

    自动选择最优的分类后端：
    - 优先使用 TorchClassifier（若权重存在）
    - 权重不存在时自动回退到 _FallbackClassifier

    D 在 pipeline.py 中只调用此类，不直接使用内部分类器。

    用法
    ----
    >>> clf = ScrewClassifier()
    >>> classified_clusters = clf.classify_clusters(clusters)
    >>> counts = clf.count_by_type(classified_clusters)
    >>> print(counts)  # [n_type1, n_type2, n_type3, n_type4, n_type5]
    """

    def __init__(
        self,
        weights_path: str | Path = DEFAULT_WEIGHTS,
        input_size: int = INPUT_SIZE,
        use_fp16: bool = USE_FP16,
        device: str = "",
        vote_temperature: float = VOTE_TEMPERATURE,
        conf_weight_exponent: float = CONF_WEIGHT_EXPONENT,
    ) -> None:
        """
        Parameters
        ----------
        weights_path : str | Path
            分类器权重路径。
        input_size : int
            输入图像边长。
        use_fp16 : bool
            是否使用 FP16。
        device : str
            推理设备。
        vote_temperature : float
            投票温度参数。
        conf_weight_exponent : float
            置信度加权指数。
        """
        self.vote_temperature = vote_temperature
        self.conf_weight_exponent = conf_weight_exponent

        # 尝试加载 Torch 分类器
        self._torch_clf = TorchClassifier(
            weights_path=weights_path,
            input_size=input_size,
            use_fp16=use_fp16,
            device=device,
        )

        if self._torch_clf._loaded:
            self._backend = self._torch_clf
            self._mode = "torch"
            logger.info("ScrewClassifier 初始化完成（PyTorch 模式）。")
        else:
            self._backend = _FallbackClassifier()
            self._mode = "fallback"
            logger.warning(
                "ScrewClassifier 已切换到兜底模式（随机启发式）。"
                "最终提交前请确保分类器权重可用！"
            )

    @property
    def is_torch_mode(self) -> bool:
        """是否处于 PyTorch 推理模式。"""
        return self._mode == "torch"

    def classify_clusters(self, clusters: List[Cluster]) -> List[Cluster]:
        """
        对所有 Cluster 执行分类投票，原地更新 class_probs 和 pred_class。

        这是 D 在 pipeline.py 中调用的主接口。

        Parameters
        ----------
        clusters : List[Cluster]
            待分类的螺丝 Cluster 列表（来自 GlobalDedup 的输出）。
            调用前 pred_class 均为 -1。

        Returns
        -------
        List[Cluster]
            已填充 pred_class 的 Cluster 列表（与输入为同一对象列表）。

        TODO (C)：
        - 验证所有 Cluster.pred_class 在调用后均不为 -1
        - 在 3 段开发视频上统计各类别的分类准确率
        - 分析混淆矩阵，识别易混类别对
        """
        if not clusters:
            return clusters

        logger.info("开始对 %d 个 Cluster 进行分类...", len(clusters))

        for cluster in clusters:
            if (not self.is_torch_mode) and _cluster_has_detector_multiclass_labels(cluster):
                _classify_cluster_from_detector_votes(
                    cluster,
                    conf_weight_exponent=self.conf_weight_exponent,
                )
                continue

            classify_cluster_with_votes(
                cluster,
                self._backend,
                vote_temperature=self.vote_temperature,
                conf_weight_exponent=self.conf_weight_exponent,
            )

        # 验证所有 Cluster 已分类
        unclassified = [c for c in clusters if c.pred_class == -1]
        if unclassified:
            logger.error(
                "%d 个 Cluster 分类后 pred_class 仍为 -1（IDs: %s）。",
                len(unclassified),
                [c.cluster_id for c in unclassified],
            )
        else:
            logger.info("✅ 所有 %d 个 Cluster 分类完成。", len(clusters))

        return clusters

    @staticmethod
    def count_by_type(clusters: List[Cluster]) -> List[int]:
        """
        统计各类螺丝的数量，返回长度为 5 的计数列表。

        Parameters
        ----------
        clusters : List[Cluster]
            已分类（pred_class != -1）的 Cluster 列表。

        Returns
        -------
        List[int]
            长度为 5 的列表，按 [Type_1, Type_2, Type_3, Type_4, Type_5] 顺序。
            对应 pred_class 的 0-indexed 到 Type_X 的映射：
                pred_class=0 → Type_1, …, pred_class=4 → Type_5
        """
        counts = [0] * NUM_CLASSES
        unclassified_count = 0

        for cluster in clusters:
            if cluster.pred_class < 0 or cluster.pred_class >= NUM_CLASSES:
                unclassified_count += 1
                logger.warning(
                    "Cluster #%d pred_class=%d 无效，不计入统计。",
                    cluster.cluster_id, cluster.pred_class,
                )
                continue
            counts[cluster.pred_class] += 1

        if unclassified_count > 0:
            logger.warning(
                "有 %d 个 Cluster 未分类（pred_class 无效），已从计数中排除。",
                unclassified_count,
            )

        logger.info("计数结果: %s（对应 Type_1 ~ Type_5）", counts)
        return counts

    def classify_and_count(
        self,
        clusters: List[Cluster],
    ) -> Tuple[List[Cluster], List[int]]:
        """
        分类 + 计数一步完成（便捷方法）。

        Parameters
        ----------
        clusters : List[Cluster]
            待分类的 Cluster 列表。

        Returns
        -------
        (List[Cluster], List[int])
            已分类的 Cluster 列表，以及长度为 5 的计数列表。
        """
        classified = self.classify_clusters(clusters)
        counts = self.count_by_type(classified)
        return classified, counts

    # ------------------------------------------------------------------
    # 混淆矩阵分析（C 负责用于报告）
    # ------------------------------------------------------------------

    def compute_confusion_matrix(
        self,
        crops: List[np.ndarray],
        labels: List[int],
    ) -> np.ndarray:
        """
        计算分类器在给定数据集上的混淆矩阵。

        用于 C 分析分类器性能，指导 fine-tune 策略。

        Parameters
        ----------
        crops : List[np.ndarray]
            测试 crop 图像列表。
        labels : List[int]
            对应的真实类别标签（0-indexed）。

        Returns
        -------
        np.ndarray : shape (5, 5)，混淆矩阵。
            矩阵[i][j] 表示真实类别 i 被预测为类别 j 的次数。

        TODO (C)：
        - 在视频 crop 标注集上调用此方法，生成混淆矩阵
        - 将混淆矩阵可视化后放入实验报告（消融实验 B）
        """
        preds = self._backend.predict(crops)
        n = NUM_CLASSES
        confusion = np.zeros((n, n), dtype=int)
        for true_label, pred_label in zip(labels, preds):
            if 0 <= true_label < n and 0 <= pred_label < n:
                confusion[true_label][pred_label] += 1
        return confusion

    def print_confusion_matrix(
        self,
        crops: List[np.ndarray],
        labels: List[int],
    ) -> None:
        """
        打印格式化的混淆矩阵报告（C 用于调试和报告）。

        Parameters
        ----------
        crops : List[np.ndarray]
            测试 crop 列表。
        labels : List[int]
            真实标签列表（0-indexed）。
        """
        cm = self.compute_confusion_matrix(crops, labels)
        total = len(labels)
        correct = int(np.diag(cm).sum())
        acc = correct / max(total, 1) * 100

        print("=" * 60)
        print(f"[ScrewClassifier] 混淆矩阵  (总样本={total}, 准确率={acc:.1f}%)")
        print(f"{'':12s}", end="")
        for j in range(NUM_CLASSES):
            print(f"{'Pred_' + str(j + 1):>10s}", end="")
        print()

        for i in range(NUM_CLASSES):
            print(f"{'True_' + str(i + 1):12s}", end="")
            for j in range(NUM_CLASSES):
                mark = "←" if i == j else "  "
                print(f"{cm[i][j]:>8d}{mark}", end="")
            row_total = cm[i].sum()
            row_acc = cm[i][i] / max(row_total, 1) * 100
            print(f"  | recall={row_acc:.0f}%")

        print("=" * 60)
        if acc < 80:
            print(f"⚠️  准确率 {acc:.1f}% < 80%，请立即启动 fine-tune！")
        else:
            print(f"✅ 准确率 {acc:.1f}% ≥ 80%，分类器可用。")
