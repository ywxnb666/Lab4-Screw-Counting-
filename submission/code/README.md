# Lab4 视频螺丝计数 - 团队作业

## 团队成员

| 学号 | 姓名 | 分工 |
|------|------|------|
| （学号A） | （姓名A） | A：几何配准与去重 |
| （学号B） | （姓名B） | B：检测数据与 Detector |
| （学号C） | （姓名C） | C：分类与结果融合 |
| （学号D） | （姓名D） | D：工程封装与评测 |

> **注意**：请各成员在提交前将上方表格中的占位符替换为真实学号和姓名。

---

## 环境配置

### 1. 创建并激活 Conda 环境（推荐）

本项目推荐使用 **Anaconda / Miniconda** 管理环境。

```bash
# 创建名为 screw_count 的 Python 3.10 环境
conda create -n screw_count python=3.10 -y

# 激活环境
conda activate screw_count
```

### 2. 安装 PyTorch（根据 GPU 驱动选择版本）

**有 NVIDIA GPU（CUDA 12.1，适合 RTX 4060 等）：**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**仅 CPU（无 GPU 或驱动版本不匹配时）：**

```bash
pip install torch torchvision
```

> 其他 CUDA 版本请参考 https://pytorch.org/get-started/locally/ 选择对应命令。

### 3. 安装其余依赖

```bash
pip install -r requirements.txt
```

---

**（备选）使用 venv 标准虚拟环境：**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate

# 然后安装依赖
pip install -r requirements.txt
```

### 3. 放置模型权重

将训练好的模型权重文件放至 `models/` 目录：

```
models/
  detector.pt       # B 负责提供：one-class YOLO 螺丝检测器
  classifier.pt     # C 负责提供：5 类螺丝分类器（Lab2 迁移/fine-tune）
```

> **注意**：若 `models/` 目录下缺少权重文件，系统会自动切换到兜底模式
> （OpenCV 检测 + 随机分类），精度极低，**最终提交前必须确保权重文件存在**。

---

## 运行方式

### 标准运行命令（作业规范接口）

```bash
python run.py \
    --data_dir /path/to/test_videos_folder \
    --output_path ./result.npy \
    --output_time_path ./time.txt \
    --mask_output_path ./mask_folder/
```

### 参数说明

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `--data_dir` | str | ✅ | 包含测试视频的文件夹路径 |
| `--output_path` | str | ✅ | result.npy 输出路径（含文件名） |
| `--output_time_path` | str | ✅ | time.txt 输出路径（含文件名） |
| `--mask_output_path` | str | ✅ | 掩膜图像输出文件夹路径 |
| `--device` | str | ❌ | 推理设备，如 `cuda:0` 或 `cpu`（默认自动选择） |
| `--no_fp16` | flag | ❌ | 禁用 FP16 推理（CPU 环境建议添加此参数） |
| `--keyframe_strategy` | str | ❌ | 关键帧策略：`motion`（默认）或 `uniform` |
| `--dist_thresh` | float | ❌ | 去重聚类距离阈值（像素，默认 40.0） |
| `--verbose` | flag | ❌ | 输出详细调试日志 |

### 在开发视频上测试

```bash
# 激活环境后，在 submission/code/ 目录下运行：
conda activate screw_count

# 使用开发视频（GPU）
python run.py \
    --data_dir ../../vedio_exp/ \
    --output_path ./output/result.npy \
    --output_time_path ./output/time.txt \
    --mask_output_path ./output/masks/

# 使用 CPU（无 GPU 时，添加 --no_fp16）
python run.py \
    --data_dir ../../vedio_exp/ \
    --output_path ./output/result.npy \
    --output_time_path ./output/time.txt \
    --mask_output_path ./output/masks/ \
    --device cpu --no_fp16
```

### 验证输出格式

```python
import numpy as np

# 验证 result.npy 格式
result = np.load("./output/result.npy", allow_pickle=True).item()
print(type(result))      # <class 'dict'>
for video_name, counts in result.items():
    print(f"{video_name}: {counts}")  # {'IMG_2374': [14, 7, 6, 22, 3], ...}
    assert len(counts) == 5, "计数列表长度必须为 5"

# 验证 time.txt 格式
with open("./output/time.txt") as f:
    total_time = float(f.read().strip())
print(f"总耗时: {total_time:.4f} 秒")
```

---

## 工具脚本使用

### 关键帧抽取（D 工具，用于标注数据准备）

```bash
# 从开发视频抽取关键帧，保存到 frames/ 目录
python tools/extract_keyframes.py \
    --input ../../vedio_exp/ \
    --output frames/ \
    --max_frames 40 \
    --strategy motion

# 同时导出帧编号清单（供标注工具导入）
python tools/extract_keyframes.py \
    --input ../../vedio_exp/ \
    --output frames/ \
    --export_manifest --manifest_path keyframes.json --manifest_format json
```

### 标注格式转换（D 工具）

```bash
# CVAT XML → YOLO（最常见工作流）
python tools/convert_annotations.py \
    --src annotations/cvat_export.xml \
    --dst annotations/yolo_labels/ \
    --from_fmt cvat --to_fmt yolo \
    --class_names screw

# COCO JSON → YOLO（Roboflow 下载后转换）
python tools/convert_annotations.py \
    --src annotations/roboflow.json \
    --dst annotations/yolo_labels/ \
    --from_fmt coco --to_fmt yolo

# 仅统计标注数量（不转换）
python tools/convert_annotations.py \
    --src annotations/cvat_export.xml \
    --from_fmt cvat --stats_only
```

### 速度 Benchmark（D 工具）

```bash
# 对开发视频跑 3 次 benchmark
python tools/benchmark.py \
    --data_dir ../../vedio_exp/ \
    --runs 3 \
    --output_json reports/benchmark.json \
    --output_md reports/benchmark.md

# 细粒度模块计时（每个步骤单独计时）
python tools/benchmark.py \
    --data_dir ../../vedio_exp/ \
    --detailed
```

### 消融实验（D 工具）

```bash
# 运行所有消融实验（需要 GT 标签）
python tools/ablation.py \
    --data_dir ../../vedio_exp/ \
    --output ablation_results/ \
    --gt_path gt.npy \
    --export_markdown --export_latex

# 仅运行去重策略对比（实验组 A）
python tools/ablation.py \
    --data_dir ../../vedio_exp/ \
    --output ablation_results/ \
    --group A

# 生成 LaTeX 表格（不重新运行，从已有结果生成）
python tools/ablation.py \
    --report_only \
    --results_path ablation_results/ablation_summary_*.json \
    --output ablation_results/ \
    --export_latex
```

---

## 项目结构

```
code/
├── run.py                      # 主入口（作业规范接口）           [D]
├── pipeline.py                 # 视频处理流程编排                 [D]
├── interfaces.py               # 团队协作数据接口定义             [D]
├── requirements.txt            # Python 依赖列表                 [D]
├── README.md                   # 本文档                         [D]
│
├── modules/                    # 核心算法模块
│   ├── detector.py             # 螺丝检测器（YOLO + SAHI）       [B]
│   ├── registration.py         # 锚帧几何配准（AKAZE + Homography）[A]
│   ├── dedup.py                # 全局去重聚类（DBSCAN）           [A]
│   └── classifier.py           # 5 类螺丝分类器                  [C]
│
├── utils/                      # 工程工具包
│   ├── video_io.py             # 视频读取与帧提取                [D]
│   ├── output_formatter.py     # 输出格式封装（npy/time/mask）   [D]
│   └── visualizer.py           # 掩膜叠加与可视化               [D]
│
├── tools/                      # 数据工具（独立可运行脚本）
│   ├── extract_keyframes.py    # 关键帧批量抽取（标注数据准备）   [D]
│   ├── export_crops.py         # 检测 Crop 导出                  [D]
│   ├── convert_annotations.py  # 标注格式转换（CVAT/YOLO/COCO）  [D]
│   ├── benchmark.py            # 速度 Benchmark                  [D]
│   └── ablation.py             # 消融实验记录与对比              [D]
│
└── models/                     # 模型权重目录
    ├── detector.pt             # YOLO 检测器权重（B 提供）
    └── classifier.pt           # 分类器权重（C 提供）
```

---

## 模块接口说明

各模块通过 `interfaces.py` 中定义的数据结构通信：

```python
# 检测器输出（B → A, C, D）
Detection(frame_id, bbox, confidence, crop, track_id)

# 配准输出（A → A, D）
Registration(frame_id, H_to_ref, valid, inlier_ratio)

# 去重聚类输出（A → C, D）
Cluster(cluster_id, observations, best_crop, class_probs, pred_class, ref_center, ref_bbox)

# 最终结果（D 产出）
VideoResult(video_name, counts, clusters, processing_time)
```

---

## 各模块 TODO 状态

| 模块 | Owner | 状态 | 关键 TODO |
|------|-------|------|-----------|
| `modules/detector.py` | B | 框架完成，待训练 | 将 `models/detector.pt` 放到位；调整 CONF/IOU 阈值；SAHI 参数调优 |
| `modules/registration.py` | A | 框架完成，待调优 | 验证 3 段开发视频配准成功率 ≥ 90%；调整 AKAZE 参数 |
| `modules/dedup.py` | A | 框架完成，待调优 | 调整 `CLUSTER_DIST_THRESH`；验证去重后计数误差 |
| `modules/classifier.py` | C | 框架完成，待迁移 | 将 Lab2 权重迁移至 `models/classifier.pt`；若 baseline < 80% 立即 fine-tune |
| `pipeline.py` | D | ✅ 完成 | - |
| `run.py` | D | ✅ 完成 | - |
| `utils/` | D | ✅ 完成 | - |
| `tools/` | D | ✅ 完成 | - |

---

## 常见问题

**Q: 运行时提示 "YOLO 权重文件不存在"，该怎么办？**
A: 系统会自动切换到 OpenCV 兜底检测器（精度很低）。请联系 B 获取 `models/detector.pt`，
或将 B 训练好的权重文件放到 `models/` 目录下。

**Q: 运行时提示 "分类器处于兜底模式（随机启发式）"，该怎么办？**
A: 请联系 C 获取 `models/classifier.pt`（Lab2 分类器权重）。

**Q: 在 CPU 上运行太慢，怎么加速？**
A: 使用 `--keyframe_strategy uniform` 减少关键帧数量，同时添加 `--no_fp16` 参数。

**Q: 运行报错 "ModuleNotFoundError: No module named 'ultralytics'"？**
A: 激活虚拟环境后运行 `pip install -r requirements.txt`。

**Q: 如何只测试输出格式是否正确（不关心精度）？**
A: 使用 `--dry_run` 参数仅列出视频文件，或先运行然后用以下命令验证输出：
```bash
python -c "
import numpy as np
r = np.load('result.npy', allow_pickle=True).item()
print('Keys:', list(r.keys()))
for k, v in r.items():
    print(f'  {k}: {v}  (len={len(v)})')
"
```

---

## 算法概述

```
视频读取 + 方向校正
        ↓
   关键帧提取（基于 ORB 特征点位移）
        ↓
   逐帧螺丝检测（one-class YOLO + SAHI）    [B]
        ↓
   锚帧几何配准（AKAZE + Homography）       [A]
        ↓
   全局坐标系投影 + 去重聚类（DBSCAN）      [A]
        ↓
   Cluster 级 5 类分类投票                  [C]
        ↓
   输出 result.npy / mask / time.txt        [D]
```

**核心思路**：将所有关键帧的检测结果投影到统一的参考坐标系，
通过 DBSCAN 聚类将代表同一颗螺丝的多次检测合并为一个唯一实例（Cluster），
再对每个 Cluster 进行分类投票，最终统计各类别数量。

---

*本项目代码由团队共同开发，各模块分工见上方表格及各文件开头的 Owner 注释。*