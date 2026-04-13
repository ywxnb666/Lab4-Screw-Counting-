# Lab4 视频螺丝计数 - 团队作业（GitHub 协作版）

本仓库用于团队协作开发 **视频螺丝分类与计数** 系统。最终提交需要满足课程作业规范（`run.py` 入口、`result.npy`、`time.txt`、mask 图像）。

> 重要说明：本仓库中**有些大文件不会上传到 GitHub**（例如视频、模型权重、部分标注数据）。README 已写清楚哪些文件需要你本地自行准备，以及应该放到哪里。

---

## 1. 团队成员

| 学号 | 姓名 | 分工 |
|------|------|------|
| （学号A） | （姓名A） | A：几何配准与去重 |
| （学号B） | （姓名B） | B：检测数据与 Detector |
| （学号C） | （姓名C） | C：分类与结果融合 |
| 523030910103 | 魏思齐 | D：工程封装与评测 |

> 提交前请将占位符替换为真实信息。

---

## 2. 仓库目录与“实际运行入口”在哪里？

本仓库根目录下，你需要重点关注 **`submission/code/`** 这一套“可提交/可运行”的代码结构：

- **仓库根目录**：`./`（即你 clone 下来的目录）
- **可运行代码根目录**：`./submission/code/`
  - 作业统一入口：`./submission/code/run.py`
  - 依赖文件：`./submission/code/requirements.txt`
  - 模型权重目录：`./submission/code/models/`
  - 工具脚本：`./submission/code/tools/`

> 你在命令行里运行代码时，**推荐先 `cd submission/code` 再执行**，可以避免相对路径混乱。

---

## 3. GitHub 上缺失 / 不上传的文件（你需要本地准备）

### 3.1 视频数据（不建议上传到 GitHub）
开发视频在我们的本地工程里通常放在：
- `./vedio_exp/`（示例：`IMG_2374.MOV`、`IMG_2375.MOV`、`IMG_2376.MOV`）

### 3.2 模型权重（通常不会上传）
请把权重文件放到 **`./submission/code/models/`**：

```
submission/code/models/
  detector.pt       # B 提供：one-class YOLO 螺丝检测器权重
  classifier.pt     # C 提供：5 类分类器权重（Lab2 迁移 / fine-tune）
```

如果缺少权重：
- 检测会自动退化为 OpenCV 兜底检测（精度低）
- 分类会退化为随机/启发式分类（几乎不可用）
- 但 **run.py 仍能跑通并产出符合格式的输出**（方便工程联调）

### 3.3 标注与中间数据（可选、不一定上传）
你可能会在本地生成这些目录（通常不提交到 GitHub）：
- `./submission/frames/`：抽取出的关键帧图像（供标注）
- `./submission/output/`：运行后输出（`result.npy` / `time.txt` / `masks/`）
- `./annotations/`：标注文件（CVAT/YOLO/COCO 等）

建议在 `.gitignore` 中忽略大文件目录（由队长决定）。

---

## 4. 环境配置（推荐 Conda）

> 以下命令默认你在 **仓库根目录 `./`** 执行。

### 4.1 创建并激活 Conda 环境
```bash
conda create -n screw_count python=3.10 -y
conda activate screw_count
```

### 4.2 安装 PyTorch（按设备选择）
**有 NVIDIA GPU（CUDA 12.1，适合 RTX 4060 等）：**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**仅 CPU：**
```bash
pip install torch torchvision
```

> 其他 CUDA 版本请参考：https://pytorch.org/get-started/locally/

### 4.3 安装其余依赖
```bash
pip install -r submission/code/requirements.txt
```

---

## 5. 一键运行（作业规范接口）

### 5.1 标准运行命令（必须满足）
从仓库根目录运行：
```bash
python submission/code/run.py \
  --data_dir /path/to/test_videos_folder \
  --output_path ./result.npy \
  --output_time_path ./time.txt \
  --mask_output_path ./mask_folder/
```

> 注意：
> - `--data_dir` 是“包含多个视频文件”的文件夹路径
> - 输出的 key 必须是“视频文件名不含后缀”
> - mask 文件名格式必须为 `{video_name}_mask.png`

### 5.2 使用本仓库的开发视频跑通（示例）
假设开发视频在 `./vedio_exp/`，推荐进入可运行代码目录执行：

```bash
cd submission/code

python run.py \
  --data_dir ../../vedio_exp/ \
  --output_path ../output/result.npy \
  --output_time_path ../output/time.txt \
  --mask_output_path ../output/masks/
```

**CPU 环境（或不想用 fp16）**
```bash
cd submission/code

python run.py \
  --data_dir ../../vedio_exp/ \
  --output_path ../output/result.npy \
  --output_time_path ../output/time.txt \
  --mask_output_path ../output/masks/ \
  --device cpu --no_fp16
```

---

## 6. 输出格式自检（非常重要）

运行结束后，你应该得到：

- `result.npy`：`numpy.load(..., allow_pickle=True).item()` 得到 `dict`
- `time.txt`：一个数字（秒）
- `mask_folder/`：每段视频一张 `{video_name}_mask.png`

用以下脚本验证（在任意目录均可）：

```python
import numpy as np

result = np.load("result.npy", allow_pickle=True).item()
assert isinstance(result, dict)
for k, v in result.items():
    assert isinstance(k, str)
    assert len(v) == 5
print("result.npy OK")

with open("time.txt", "r", encoding="utf-8") as f:
    t = float(f.read().strip())
print("time.txt OK:", t)
```

---

## 7. 面向协作者的“按角色工作流”

### 7.1 B（Detector）工作流：训练 one-class YOLO
目标：生成 `submission/code/models/detector.pt`

建议流程：
1. 用 D 的工具抽关键帧：
   ```bash
   conda activate screw_count
   python submission/code/tools/extract_keyframes.py \
     --input vedio_exp/ \
     --output submission/frames/ \
     --max_frames 40 \
     --strategy motion \
     --export_manifest \
     --manifest_path submission/frames/keyframe_manifest.json \
     --manifest_format json
   ```
2. 在 CVAT/Roboflow 标注 bbox（统一 label：`screw`）
3. 用 D 的工具把标注转成 YOLO：
   ```bash
   python submission/code/tools/convert_annotations.py \
     --src annotations/cvat_export.xml \
     --dst annotations/yolo_labels/ \
     --from_fmt cvat --to_fmt yolo \
     --class_names screw
   ```
4. 训练 YOLO（由 B 自行组织训练脚本与配置），训练完成后将权重放到：
   - `submission/code/models/detector.pt`

### 7.2 A（Registration + Dedup）工作流：调参 + 提升去重稳定性
A 主要关心：
- `submission/code/modules/registration.py`
- `submission/code/modules/dedup.py`

建议：
1. 先用现成权重/兜底模式跑通，观察配准 `valid` 比例
2. 调整：
   - `INLIER_RATIO_THRESHOLD`
   - `AKAZE_THRESHOLD` / `RANSAC_REPROJ_THRESH`
   - `CLUSTER_DIST_THRESH`（影响去重半径）
3. 用 D 的 benchmark 工具做速度/稳定性记录（见 §8）

### 7.3 C（Classifier）工作流：迁移 Lab2 分类器 + fine-tune
目标：生成 `submission/code/models/classifier.pt`

建议流程：
1. 从视频检测/标注导出 crop：
   - 若已有 YOLO 标注（推荐）：
     ```bash
     python submission/code/tools/export_crops.py \
       --mode from_labels \
       --frames_dir submission/frames/ \
       --labels_dir annotations/yolo_labels/ \
       --output submission/crops/ \
       --class_names screw \
       --html_preview
     ```
   - 或者直接用 detector 导出（需要 detector.pt）：
     ```bash
     python submission/code/tools/export_crops.py \
       --mode from_detector \
       --video_dir vedio_exp/ \
       --output submission/crops/ \
       --conf 0.35 \
       --html_preview
     ```
2. 人工把 crop 分拣到 `Type_1~Type_5`（具体数据集组织由 C 决定）
3. 训练 / fine-tune 后将权重放到：
   - `submission/code/models/classifier.pt`

---

## 8. Benchmark（速度评估）与 Ablation（消融实验）

### 8.1 速度 Benchmark（D 工具）
```bash
conda activate screw_count

python submission/code/tools/benchmark.py \
  --data_dir vedio_exp/ \
  --runs 3 \
  --output_json submission/reports/benchmark.json \
  --output_md submission/reports/benchmark.md
```

细粒度模块计时（更详细）：
```bash
python submission/code/tools/benchmark.py \
  --data_dir vedio_exp/ \
  --detailed
```

### 8.2 消融实验（D 工具）
```bash
conda activate screw_count

python submission/code/tools/ablation.py \
  --data_dir vedio_exp/ \
  --output submission/ablation_results/ \
  --export_markdown --export_latex
```

> 若你有 GT（真实计数）文件 `gt.npy`（格式与 `result.npy` 相同），可加：
> `--gt_path gt.npy` 生成得分与 MAE。

---

## 9. 项目结构（以仓库根目录为起点的真实路径）

> 下方为“仓库根目录 `./`”视角的路径。  
> 实际可运行代码集中在 `./submission/code/`。

```
./
├── README.md                     # 本文档（协作版说明）
├── vedio_exp/                    # （可能缺失）开发视频目录（建议不上传）
│   ├── IMG_2374.MOV
│   ├── IMG_2375.MOV
│   └── IMG_2376.MOV
└── submission/
    ├── code/
    │   ├── run.py                # 作业规范入口
    │   ├── pipeline.py           # 主流程编排（D）
    │   ├── interfaces.py         # 模块通信接口（D）
    │   ├── requirements.txt
    │   ├── models/               # （可能缺失）模型权重目录
    │   │   ├── detector.pt
    │   │   └── classifier.pt
    │   ├── modules/              # A/B/C 的算法模块
    │   │   ├── detector.py       # B
    │   │   ├── registration.py   # A
    │   │   ├── dedup.py          # A
    │   │   └── classifier.py     # C
    │   ├── utils/                # D 的工程工具
    │   │   ├── video_io.py
    │   │   ├── output_formatter.py
    │   │   └── visualizer.py
    │   └── tools/                # D 的数据/评测工具
    │       ├── extract_keyframes.py
    │       ├── export_crops.py
    │       ├── convert_annotations.py
    │       ├── benchmark.py
    │       └── ablation.py
    ├── output/                   # （本地生成）运行输出目录
    │   ├── result.npy
    │   ├── time.txt
    │   └── masks/
    ├── frames/                   # （本地生成）关键帧导出目录
    └── reports/                  # （本地生成）benchmark 报告
```

---

## 10. 常见问题（协作相关）

### Q1: 我运行 `open /home/.../IMG_2374_mask.png` 在 Windows 上找不到文件？
A: `/home/...` 是 **WSL/Linux 路径**。Windows 应使用：
- `\\wsl.localhost\Ubuntu\home\...` 形式访问
或在 WSL 内用 `explorer.exe` 打开目录。

### Q2: 没有 `models/*.pt` 能跑吗？
A: 能跑通，但精度很差。最终提交必须提供权重。

### Q3: 我应该在哪个目录运行命令？
A: 推荐：
- 从仓库根目录运行：`python submission/code/run.py ...`
- 或进入：`cd submission/code` 后运行 `python run.py ...`

---

## 11. 贡献说明（D）
D 已完成：
- 工程封装：统一入口 `run.py`、主流程 `pipeline.py`
- 输出封装：`result.npy / time.txt / mask` 的格式严格对齐作业要求
- 数据工具：抽帧、crop 导出、标注转换
- 评测工具：benchmark 与消融实验脚本
- Conda 环境配置与端到端跑通验证

---
