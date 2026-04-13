#!/usr/bin/env python3
"""
tools/convert_annotations.py - 标注格式转换工具
Owner: D（工程封装）

用途：
  在团队协同标注过程中，不同工具（Roboflow / CVAT / 手工标注）产出的格式不同。
  本工具提供各主流格式之间的无损转换，确保 B（检测数据）能够直接使用任意来源的标注。

支持的格式：
  - YOLO txt     : 每张图一个 .txt 文件，每行 "class cx cy w h"（归一化）
  - COCO JSON    : COCO 实例检测格式（annotations / images / categories）
  - CVAT XML     : CVAT 1.x 导出的 XML 格式（task / image / box）
  - Pascal VOC   : 每张图一个 XML 文件（< annotation >< object > 结构）

支持的转换方向：
  yolo   → coco
  yolo   → voc
  coco   → yolo
  coco   → voc
  cvat   → yolo
  cvat   → coco
  voc    → yolo
  voc    → coco

使用示例：
  # CVAT XML → YOLO（最常见：从 CVAT 导出后转给 B 训练）
  python tools/convert_annotations.py \\
      --src annotations/cvat_export.xml \\
      --dst annotations/yolo/ \\
      --from_fmt cvat --to_fmt yolo \\
      --class_names screw

  # YOLO → COCO（转换为 COCO 格式供 sahi / mmdetection 使用）
  python tools/convert_annotations.py \\
      --src annotations/yolo/ \\
      --dst annotations/coco.json \\
      --from_fmt yolo --to_fmt coco \\
      --images_dir frames/ \\
      --class_names screw

  # COCO → YOLO（从 Roboflow 下载后转换）
  python tools/convert_annotations.py \\
      --src annotations/roboflow_coco.json \\
      --dst annotations/yolo/ \\
      --from_fmt coco --to_fmt yolo

  # 统计标注数量（不做转换，只计数）
  python tools/convert_annotations.py \\
      --src annotations/cvat_export.xml \\
      --from_fmt cvat --stats_only

依赖：numpy（必需）；Pillow（可选，用于读取图像尺寸）
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

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
logger = logging.getLogger("convert_annotations")

# ---------------------------------------------------------------------------
# 支持的图像后缀
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# ---------------------------------------------------------------------------
# 内部通用数据结构（中间表示）
# ---------------------------------------------------------------------------


class BBox(NamedTuple):
    """
    轴对齐矩形框（绝对像素坐标）。

    Attributes
    ----------
    x1, y1 : float  左上角坐标（像素）
    x2, y2 : float  右下角坐标（像素）
    """
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def to_xywh(self) -> Tuple[float, float, float, float]:
        """返回 (x1, y1, width, height) 格式（COCO 用）。"""
        return self.x1, self.y1, self.width, self.height

    def to_cxcywh_norm(
        self, img_w: float, img_h: float
    ) -> Tuple[float, float, float, float]:
        """返回归一化的 (cx, cy, w, h) 格式（YOLO 用）。"""
        return (
            self.cx / img_w,
            self.cy / img_h,
            self.width / img_w,
            self.height / img_h,
        )

    @classmethod
    def from_cxcywh_norm(
        cls,
        cx_n: float,
        cy_n: float,
        w_n: float,
        h_n: float,
        img_w: float,
        img_h: float,
    ) -> "BBox":
        """从归一化 YOLO 格式构造。"""
        cx = cx_n * img_w
        cy = cy_n * img_h
        w = w_n * img_w
        h = h_n * img_h
        return cls(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    @classmethod
    def from_xywh(
        cls, x: float, y: float, w: float, h: float
    ) -> "BBox":
        """从 COCO (x, y, w, h) 格式构造（x, y 为左上角）。"""
        return cls(x, y, x + w, y + h)

    def clip(self, img_w: float, img_h: float) -> "BBox":
        """将坐标裁剪到图像范围内。"""
        return BBox(
            max(0.0, min(self.x1, img_w)),
            max(0.0, min(self.y1, img_h)),
            max(0.0, min(self.x2, img_w)),
            max(0.0, min(self.y2, img_h)),
        )


class Annotation(NamedTuple):
    """单个对象标注。"""
    class_id: int       # 0-indexed 类别 ID
    class_name: str     # 类别名称（如 "screw"）
    bbox: BBox          # 绝对像素坐标
    segmentation: Optional[List] = None  # 可选轮廓（COCO 格式用）
    ann_id: int = -1    # 标注 ID（COCO 用，-1 表示未分配）
    attributes: Optional[Dict] = None    # 附加属性（CVAT 用）


class ImageRecord(NamedTuple):
    """单张图像及其所有标注。"""
    image_id: int       # 图像 ID（数据集内唯一）
    file_name: str      # 文件名（含相对路径，不含根目录）
    width: int          # 图像宽度（像素）
    height: int         # 图像高度（像素）
    annotations: List[Annotation]  # 该图像上的所有标注


class Dataset(NamedTuple):
    """完整标注数据集（中间表示）。"""
    images: List[ImageRecord]
    class_names: List[str]     # 有序类别名称列表（index = class_id）

    @property
    def n_images(self) -> int:
        return len(self.images)

    @property
    def n_annotations(self) -> int:
        return sum(len(img.annotations) for img in self.images)

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    def class_counts(self) -> Dict[str, int]:
        """返回各类别的标注数量字典。"""
        counts: Dict[str, int] = {name: 0 for name in self.class_names}
        for img in self.images:
            for ann in img.annotations:
                if 0 <= ann.class_id < len(self.class_names):
                    counts[self.class_names[ann.class_id]] += 1
        return counts

    def print_stats(self) -> None:
        """打印数据集统计摘要。"""
        print("\n" + "=" * 50)
        print(f"数据集统计")
        print("=" * 50)
        print(f"  图像数量    : {self.n_images}")
        print(f"  标注数量    : {self.n_annotations}")
        print(f"  类别数量    : {self.n_classes}")
        print(f"  类别列表    : {self.class_names}")
        print(f"\n  各类别标注数:")
        for name, cnt in self.class_counts().items():
            bar = "█" * max(1, cnt // 5)
            print(f"    {name:20s}: {cnt:5d}  {bar}")
        if self.n_images > 0:
            avg = self.n_annotations / self.n_images
            print(f"\n  平均每图标注: {avg:.1f}")
        print("=" * 50 + "\n")


# ---------------------------------------------------------------------------
# 图像尺寸读取工具
# ---------------------------------------------------------------------------


def _read_image_size(image_path: Path) -> Tuple[int, int]:
    """
    读取图像尺寸 (width, height)，优先使用 Pillow，无则用 OpenCV。

    Parameters
    ----------
    image_path : Path

    Returns
    -------
    Tuple[int, int] : (width, height)；读取失败时返回 (0, 0)。
    """
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            return img.width, img.height
    except ImportError:
        pass

    try:
        import cv2
        img = cv2.imread(str(image_path))
        if img is not None:
            h, w = img.shape[:2]
            return w, h
    except ImportError:
        pass

    logger.warning("无法读取图像尺寸（需要 Pillow 或 opencv-python）: %s", image_path)
    return 0, 0


def _collect_image_files(
    images_dir: Path,
    extensions: set = IMAGE_EXTENSIONS,
) -> List[Path]:
    """
    递归收集目录中所有图像文件，按文件名排序。

    Parameters
    ----------
    images_dir : Path
    extensions : set

    Returns
    -------
    List[Path]
    """
    files = sorted(
        p for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    )
    return files


# ---------------------------------------------------------------------------
# 格式解析器（各格式 → 中间表示 Dataset）
# ---------------------------------------------------------------------------


class YOLOParser:
    """
    解析 YOLO txt 格式标注。

    目录结构（两种均支持）：
      方式 1（图像与标注同目录）：
        images/
          img001.jpg
          img001.txt   ← 与图像同名
      方式 2（标准 YOLOv5 结构）：
        images/  img001.jpg ...
        labels/  img001.txt ...

    标注格式（每行）：
        <class_id> <cx_norm> <cy_norm> <w_norm> <h_norm>
    """

    def parse(
        self,
        labels_dir: Path,
        images_dir: Optional[Path],
        class_names: List[str],
    ) -> Dataset:
        """
        解析 YOLO 标注目录。

        Parameters
        ----------
        labels_dir : Path
            标注 .txt 文件所在目录。
        images_dir : Path | None
            图像文件所在目录（用于读取图像尺寸）；
            None 时尝试从 labels_dir 同级的 images/ 推断。
        class_names : List[str]
            类别名称列表（顺序对应 class_id）。

        Returns
        -------
        Dataset
        """
        labels_dir = Path(labels_dir)
        if not labels_dir.is_dir():
            raise NotADirectoryError(f"标注目录不存在: {labels_dir}")

        # 推断图像目录
        if images_dir is None:
            candidate = labels_dir.parent / "images"
            if candidate.is_dir():
                images_dir = candidate
                logger.info("自动推断图像目录: %s", images_dir)
            else:
                images_dir = labels_dir  # 图文同目录

        label_files = sorted(labels_dir.glob("*.txt"))
        logger.info("找到 %d 个 YOLO 标注文件。", len(label_files))

        images: List[ImageRecord] = []
        img_id = 0

        for lf in label_files:
            # 查找对应图像文件
            img_path = self._find_image(lf, images_dir)
            if img_path is None:
                logger.debug("未找到对应图像，跳过: %s", lf.name)
                continue

            w, h = _read_image_size(img_path)
            if w == 0 or h == 0:
                logger.warning("图像尺寸读取失败，跳过: %s", img_path)
                continue

            annotations = self._parse_label_file(lf, w, h, class_names)
            images.append(ImageRecord(
                image_id=img_id,
                file_name=img_path.name,
                width=w,
                height=h,
                annotations=annotations,
            ))
            img_id += 1

        logger.info("YOLO 解析完成: %d 张图像，%d 条标注",
                    len(images), sum(len(i.annotations) for i in images))
        return Dataset(images=images, class_names=class_names)

    @staticmethod
    def _find_image(label_file: Path, images_dir: Path) -> Optional[Path]:
        """查找与标注文件同名的图像文件。"""
        stem = label_file.stem
        for ext in IMAGE_EXTENSIONS:
            candidate = images_dir / (stem + ext)
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def _parse_label_file(
        label_file: Path,
        img_w: int,
        img_h: int,
        class_names: List[str],
    ) -> List[Annotation]:
        """解析单个 YOLO 标注文件，返回 Annotation 列表。"""
        annotations: List[Annotation] = []
        try:
            with open(label_file, encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 5:
                        logger.warning(
                            "%s 行 %d: 字段数不足（%d < 5），跳过。",
                            label_file.name, line_no, len(parts),
                        )
                        continue
                    class_id = int(parts[0])
                    cx_n, cy_n, w_n, h_n = map(float, parts[1:5])

                    if not (0.0 <= cx_n <= 1.0 and 0.0 <= cy_n <= 1.0 and
                            0.0 < w_n <= 1.0 and 0.0 < h_n <= 1.0):
                        logger.warning(
                            "%s 行 %d: 坐标越界（cx=%.4f cy=%.4f w=%.4f h=%.4f），跳过。",
                            label_file.name, line_no, cx_n, cy_n, w_n, h_n,
                        )
                        continue

                    bbox = BBox.from_cxcywh_norm(cx_n, cy_n, w_n, h_n, img_w, img_h)
                    name = (class_names[class_id]
                            if 0 <= class_id < len(class_names)
                            else str(class_id))
                    annotations.append(Annotation(
                        class_id=class_id,
                        class_name=name,
                        bbox=bbox,
                    ))
        except Exception as e:
            logger.error("解析标注文件失败 %s: %s", label_file, e)
        return annotations


class COCOParser:
    """
    解析 COCO JSON 格式标注。

    COCO 格式结构：
    {
        "images": [{"id":1, "file_name":"xxx.jpg", "width":W, "height":H}, ...],
        "annotations": [{"id":1, "image_id":1, "category_id":1,
                          "bbox":[x,y,w,h], "area":..., "segmentation":...}, ...],
        "categories": [{"id":1, "name":"screw"}, ...]
    }
    """

    def parse(self, json_path: Path) -> Dataset:
        """
        解析 COCO JSON 文件。

        Parameters
        ----------
        json_path : Path
            COCO 格式 JSON 文件路径。

        Returns
        -------
        Dataset
        """
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"COCO JSON 文件不存在: {json_path}")

        with open(json_path, encoding="utf-8") as f:
            coco_data = json.load(f)

        # 解析类别（ID 不一定从 1 开始）
        categories = coco_data.get("categories", [])
        if not categories:
            raise ValueError("COCO JSON 中没有 categories 字段。")

        # COCO category_id 到 0-indexed class_id 的映射
        sorted_cats = sorted(categories, key=lambda c: c["id"])
        cat_id_to_class_id = {
            cat["id"]: idx for idx, cat in enumerate(sorted_cats)
        }
        class_names = [cat["name"] for cat in sorted_cats]

        # 解析图像
        images_list = coco_data.get("images", [])
        img_id_to_record: Dict[int, dict] = {img["id"]: img for img in images_list}

        # 按 image_id 聚合标注
        anns_by_image: Dict[int, List[dict]] = {}
        for ann in coco_data.get("annotations", []):
            anns_by_image.setdefault(ann["image_id"], []).append(ann)

        images: List[ImageRecord] = []
        for new_id, img_info in enumerate(images_list):
            raw_id = img_info["id"]
            w = img_info.get("width", 0)
            h = img_info.get("height", 0)
            file_name = img_info.get("file_name", "")

            raw_anns = anns_by_image.get(raw_id, [])
            annotations: List[Annotation] = []
            for raw_ann in raw_anns:
                cat_id = raw_ann.get("category_id", -1)
                class_id = cat_id_to_class_id.get(cat_id, -1)
                if class_id < 0:
                    logger.warning("未知 category_id=%d，跳过。", cat_id)
                    continue

                x, y, bw, bh = raw_ann.get("bbox", [0, 0, 0, 0])
                if bw <= 0 or bh <= 0:
                    logger.debug("标注 %d bbox 无效，跳过。", raw_ann.get("id", -1))
                    continue

                bbox = BBox.from_xywh(x, y, bw, bh)
                annotations.append(Annotation(
                    class_id=class_id,
                    class_name=class_names[class_id],
                    bbox=bbox,
                    segmentation=raw_ann.get("segmentation"),
                    ann_id=raw_ann.get("id", -1),
                ))

            images.append(ImageRecord(
                image_id=new_id,
                file_name=file_name,
                width=w,
                height=h,
                annotations=annotations,
            ))

        logger.info("COCO 解析完成: %d 张图像，%d 条标注，%d 个类别",
                    len(images), sum(len(i.annotations) for i in images), len(class_names))
        return Dataset(images=images, class_names=class_names)


class CVATParser:
    """
    解析 CVAT XML 格式标注（CVAT 1.x 导出格式）。

    CVAT XML 结构（简化）：
    <annotations>
      <meta> ... </meta>
      <image id="0" name="IMG_2374_frame000000.jpg" width="3840" height="2160">
        <box label="screw" xtl="100" ytl="200" xbr="150" ybr="250" occluded="0">
          <attribute name="type">Type_1</attribute>
        </box>
      </image>
    </annotations>
    """

    def parse(
        self,
        xml_path: Path,
        class_names: Optional[List[str]] = None,
    ) -> Dataset:
        """
        解析 CVAT XML 文件。

        Parameters
        ----------
        xml_path : Path
            CVAT 导出的 XML 文件路径。
        class_names : List[str] | None
            类别名称列表；None 时从 XML 中自动提取所有 label 值。

        Returns
        -------
        Dataset
        """
        xml_path = Path(xml_path)
        if not xml_path.exists():
            raise FileNotFoundError(f"CVAT XML 文件不存在: {xml_path}")

        try:
            tree = ET.parse(str(xml_path))
        except ET.ParseError as e:
            raise ValueError(f"XML 解析失败: {e}") from e

        root = tree.getroot()

        # 若 root 不是 <annotations>，尝试找到它
        if root.tag != "annotations":
            ann_elem = root.find("annotations")
            if ann_elem is None:
                raise ValueError("XML 中未找到 <annotations> 根元素。")
            root = ann_elem

        # 第一遍：收集所有 label 名称（用于自动构建 class_names）
        if class_names is None:
            found_labels: List[str] = []
            for image_elem in root.findall("image"):
                for box_elem in image_elem.findall("box"):
                    label = box_elem.get("label", "").strip()
                    if label and label not in found_labels:
                        found_labels.append(label)
            class_names = found_labels if found_labels else ["screw"]
            logger.info("从 XML 自动提取类别: %s", class_names)

        class_name_to_id = {name: idx for idx, name in enumerate(class_names)}

        # 第二遍：解析图像和标注
        images: List[ImageRecord] = []
        new_img_id = 0

        for image_elem in root.findall("image"):
            img_name = image_elem.get("name", f"image_{new_img_id}.jpg")
            img_w = int(image_elem.get("width", 0))
            img_h = int(image_elem.get("height", 0))

            annotations: List[Annotation] = []
            for box_elem in image_elem.findall("box"):
                label = box_elem.get("label", "").strip()
                class_id = class_name_to_id.get(label, -1)
                if class_id < 0:
                    logger.debug("未知标签 '%s'，跳过。", label)
                    continue

                xtl = float(box_elem.get("xtl", 0))
                ytl = float(box_elem.get("ytl", 0))
                xbr = float(box_elem.get("xbr", 0))
                ybr = float(box_elem.get("ybr", 0))

                if xbr <= xtl or ybr <= ytl:
                    logger.debug("box 无效（零面积），跳过。")
                    continue

                bbox = BBox(xtl, ytl, xbr, ybr)

                # 提取属性
                attrs: Dict[str, str] = {}
                for attr_elem in box_elem.findall("attribute"):
                    attr_name = attr_elem.get("name", "")
                    attr_val = (attr_elem.text or "").strip()
                    if attr_name:
                        attrs[attr_name] = attr_val

                annotations.append(Annotation(
                    class_id=class_id,
                    class_name=label,
                    bbox=bbox,
                    attributes=attrs if attrs else None,
                ))

            images.append(ImageRecord(
                image_id=new_img_id,
                file_name=img_name,
                width=img_w,
                height=img_h,
                annotations=annotations,
            ))
            new_img_id += 1

        logger.info("CVAT 解析完成: %d 张图像，%d 条标注",
                    len(images), sum(len(i.annotations) for i in images))
        return Dataset(images=images, class_names=class_names)


class PascalVOCParser:
    """
    解析 Pascal VOC XML 格式标注。

    每张图像对应一个 XML 文件，结构：
    <annotation>
      <filename>img.jpg</filename>
      <size><width>W</width><height>H</height></size>
      <object>
        <name>screw</name>
        <bndbox>
          <xmin>x1</xmin><ymin>y1</ymin><xmax>x2</xmax><ymax>y2</ymax>
        </bndbox>
      </object>
    </annotation>
    """

    def parse(
        self,
        annotations_dir: Path,
        class_names: Optional[List[str]] = None,
    ) -> Dataset:
        """
        解析 Pascal VOC 标注目录（每图一个 XML）。

        Parameters
        ----------
        annotations_dir : Path
            包含所有 .xml 标注文件的目录。
        class_names : List[str] | None
            类别列表；None 时自动从 XML 提取。

        Returns
        -------
        Dataset
        """
        annotations_dir = Path(annotations_dir)
        xml_files = sorted(annotations_dir.glob("*.xml"))
        if not xml_files:
            raise FileNotFoundError(f"目录中未找到 XML 文件: {annotations_dir}")

        logger.info("找到 %d 个 VOC XML 文件。", len(xml_files))

        # 第一遍：收集类别
        if class_names is None:
            found: List[str] = []
            for xf in xml_files:
                try:
                    tree = ET.parse(str(xf))
                    for obj in tree.getroot().findall("object"):
                        name_elem = obj.find("name")
                        if name_elem is not None and name_elem.text:
                            n = name_elem.text.strip()
                            if n and n not in found:
                                found.append(n)
                except ET.ParseError:
                    pass
            class_names = found if found else ["screw"]
            logger.info("自动提取类别: %s", class_names)

        class_name_to_id = {name: idx for idx, name in enumerate(class_names)}

        images: List[ImageRecord] = []
        for img_id, xf in enumerate(xml_files):
            try:
                tree = ET.parse(str(xf))
                root = tree.getroot()
            except ET.ParseError as e:
                logger.warning("XML 解析失败 %s: %s，跳过。", xf.name, e)
                continue

            filename_elem = root.find("filename")
            file_name = (
                filename_elem.text.strip()
                if filename_elem is not None and filename_elem.text
                else xf.stem + ".jpg"
            )

            size_elem = root.find("size")
            if size_elem is not None:
                w = int(size_elem.findtext("width", "0"))
                h = int(size_elem.findtext("height", "0"))
            else:
                w, h = 0, 0

            annotations: List[Annotation] = []
            for obj in root.findall("object"):
                name_elem = obj.find("name")
                if name_elem is None or not name_elem.text:
                    continue
                label = name_elem.text.strip()
                class_id = class_name_to_id.get(label, -1)
                if class_id < 0:
                    continue

                bndbox = obj.find("bndbox")
                if bndbox is None:
                    continue
                x1 = float(bndbox.findtext("xmin", "0"))
                y1 = float(bndbox.findtext("ymin", "0"))
                x2 = float(bndbox.findtext("xmax", "0"))
                y2 = float(bndbox.findtext("ymax", "0"))

                if x2 <= x1 or y2 <= y1:
                    continue
                bbox = BBox(x1, y1, x2, y2)
                annotations.append(Annotation(
                    class_id=class_id,
                    class_name=label,
                    bbox=bbox,
                ))

            images.append(ImageRecord(
                image_id=img_id,
                file_name=file_name,
                width=w,
                height=h,
                annotations=annotations,
            ))

        logger.info("VOC 解析完成: %d 张图像，%d 条标注",
                    len(images), sum(len(i.annotations) for i in images))
        return Dataset(images=images, class_names=class_names)


# ---------------------------------------------------------------------------
# 格式写入器（中间表示 Dataset → 各格式）
# ---------------------------------------------------------------------------


class YOLOWriter:
    """
    将 Dataset 写出为 YOLO txt 格式。

    输出目录结构：
        dst_dir/
            img001.txt
            img002.txt
            ...
            classes.txt     ← 类别名称文件（每行一个类别名）
    """

    def write(
        self,
        dataset: Dataset,
        dst_dir: Path,
        create_classes_file: bool = True,
        skip_zero_ann: bool = False,
    ) -> int:
        """
        写出 YOLO 格式标注。

        Parameters
        ----------
        dataset : Dataset
            待转换的数据集（中间表示）。
        dst_dir : Path
            输出目录。
        create_classes_file : bool
            是否生成 classes.txt。
        skip_zero_ann : bool
            True: 跳过无标注的图像（不生成空 txt）；
            False: 为无标注图像生成空 txt（保持文件完整性）。

        Returns
        -------
        int : 成功写出的文件数。
        """
        dst_dir = Path(dst_dir)
        dst_dir.mkdir(parents=True, exist_ok=True)

        n_written = 0
        for img in dataset.images:
            if skip_zero_ann and not img.annotations:
                continue

            if img.width <= 0 or img.height <= 0:
                logger.warning("图像 '%s' 尺寸无效 (%dx%d)，跳过。",
                               img.file_name, img.width, img.height)
                continue

            stem = Path(img.file_name).stem
            label_path = dst_dir / f"{stem}.txt"

            lines: List[str] = []
            for ann in img.annotations:
                cx_n, cy_n, w_n, h_n = ann.bbox.to_cxcywh_norm(img.width, img.height)
                # 校验归一化坐标
                if not (0.0 < w_n <= 1.0 and 0.0 < h_n <= 1.0):
                    logger.debug(
                        "图像 '%s' 标注坐标归一化后越界，跳过该 bbox。",
                        img.file_name,
                    )
                    continue
                cx_n = max(0.0, min(1.0, cx_n))
                cy_n = max(0.0, min(1.0, cy_n))
                lines.append(
                    f"{ann.class_id} {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}"
                )

            label_path.write_text("\n".join(lines), encoding="utf-8")
            n_written += 1

        if create_classes_file:
            classes_path = dst_dir / "classes.txt"
            classes_path.write_text(
                "\n".join(dataset.class_names), encoding="utf-8"
            )
            logger.info("classes.txt 已写出: %s", classes_path)

        logger.info("YOLO 写出完成: %d 个标注文件 → %s", n_written, dst_dir)
        return n_written


class COCOWriter:
    """
    将 Dataset 写出为 COCO JSON 格式。
    """

    def write(
        self,
        dataset: Dataset,
        dst_path: Path,
        pretty: bool = True,
    ) -> Path:
        """
        写出 COCO JSON 文件。

        Parameters
        ----------
        dataset : Dataset
        dst_path : Path
            输出 JSON 文件路径（含文件名）。
        pretty : bool
            是否美化输出（indent=2）。

        Returns
        -------
        Path : 写出的文件路径。
        """
        dst_path = Path(dst_path)
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        # categories（COCO category_id 从 1 开始）
        categories = [
            {"id": idx + 1, "name": name, "supercategory": "object"}
            for idx, name in enumerate(dataset.class_names)
        ]

        images_list: List[dict] = []
        annotations_list: List[dict] = []
        ann_id = 1

        for img in dataset.images:
            images_list.append({
                "id": img.image_id + 1,   # COCO image_id 从 1 开始
                "file_name": img.file_name,
                "width": img.width,
                "height": img.height,
            })

            for ann in img.annotations:
                x1, y1, w, h = ann.bbox.to_xywh()
                area = max(0.0, ann.bbox.area())

                ann_dict: dict = {
                    "id": ann_id,
                    "image_id": img.image_id + 1,
                    "category_id": ann.class_id + 1,  # COCO category_id 从 1 开始
                    "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
                    "area": round(area, 2),
                    "iscrowd": 0,
                }
                # 保留 segmentation（若有）
                if ann.segmentation is not None:
                    ann_dict["segmentation"] = ann.segmentation
                else:
                    ann_dict["segmentation"] = []

                annotations_list.append(ann_dict)
                ann_id += 1

        coco_output = {
            "info": {
                "description": "Screw Detection Dataset - Lab4",
                "version": "1.0",
            },
            "licenses": [],
            "images": images_list,
            "annotations": annotations_list,
            "categories": categories,
        }

        indent = 2 if pretty else None
        with open(dst_path, "w", encoding="utf-8") as f:
            json.dump(coco_output, f, ensure_ascii=False, indent=indent)

        logger.info(
            "COCO JSON 写出完成: %d 张图像，%d 条标注 → %s",
            len(images_list), len(annotations_list), dst_path,
        )
        return dst_path


class PascalVOCWriter:
    """
    将 Dataset 写出为 Pascal VOC XML 格式（每图一个 XML）。
    """

    def write(
        self,
        dataset: Dataset,
        dst_dir: Path,
        images_dir: Optional[str] = None,
    ) -> int:
        """
        写出 Pascal VOC XML 标注。

        Parameters
        ----------
        dataset : Dataset
        dst_dir : Path
            输出目录。
        images_dir : str | None
            图像所在目录（相对路径，写入 XML 的 <folder> 字段）；
            None 时使用 "images"。

        Returns
        -------
        int : 写出的 XML 文件数。
        """
        dst_dir = Path(dst_dir)
        dst_dir.mkdir(parents=True, exist_ok=True)

        folder = images_dir or "images"
        n_written = 0

        for img in dataset.images:
            stem = Path(img.file_name).stem
            xml_path = dst_dir / f"{stem}.xml"

            root = ET.Element("annotation")
            ET.SubElement(root, "folder").text = folder
            ET.SubElement(root, "filename").text = img.file_name
            ET.SubElement(root, "path").text = str(Path(folder) / img.file_name)

            size_elem = ET.SubElement(root, "size")
            ET.SubElement(size_elem, "width").text = str(img.width)
            ET.SubElement(size_elem, "height").text = str(img.height)
            ET.SubElement(size_elem, "depth").text = "3"

            ET.SubElement(root, "segmented").text = "0"

            for ann in img.annotations:
                obj = ET.SubElement(root, "object")
                ET.SubElement(obj, "name").text = ann.class_name
                ET.SubElement(obj, "pose").text = "Unspecified"
                ET.SubElement(obj, "truncated").text = "0"
                ET.SubElement(obj, "difficult").text = "0"

                bndbox = ET.SubElement(obj, "bndbox")
                ET.SubElement(bndbox, "xmin").text = str(int(round(ann.bbox.x1)))
                ET.SubElement(bndbox, "ymin").text = str(int(round(ann.bbox.y1)))
                ET.SubElement(bndbox, "xmax").text = str(int(round(ann.bbox.x2)))
                ET.SubElement(bndbox, "ymax").text = str(int(round(ann.bbox.y2)))

            tree = ET.ElementTree(root)
            ET.indent(tree, space="  ")
            tree.write(str(xml_path), encoding="utf-8", xml_declaration=True)
            n_written += 1

        logger.info("Pascal VOC 写出完成: %d 个 XML 文件 → %s", n_written, dst_dir)
        return n_written


# ---------------------------------------------------------------------------
# 主转换函数
# ---------------------------------------------------------------------------


def convert(
    src: Union[str, Path],
    dst: Union[str, Path],
    from_fmt: str,
    to_fmt: str,
    class_names: Optional[List[str]] = None,
    images_dir: Optional[Union[str, Path]] = None,
) -> Dataset:
    """
    在任意两种支持的格式之间转换标注文件。

    Parameters
    ----------
    src : str | Path
        源文件或目录路径。
    dst : str | Path
        目标文件或目录路径。
    from_fmt : str
        源格式：'yolo' | 'coco' | 'cvat' | 'voc'。
    to_fmt : str
        目标格式：'yolo' | 'coco' | 'voc'。
    class_names : List[str] | None
        类别名称列表（YOLO / CVAT / VOC 格式需要提供）；
        None 时尝试从源文件自动提取。
    images_dir : str | Path | None
        图像目录（YOLO 解析时用于查找对应图像）。

    Returns
    -------
    Dataset : 转换后的数据集（中间表示），可用于进一步操作或统计。

    Raises
    ------
    ValueError
        不支持的格式组合。
    FileNotFoundError
        源文件不存在。
    """
    src = Path(src)
    dst = Path(dst)

    supported_from = {"yolo", "coco", "cvat", "voc"}
    supported_to = {"yolo", "coco", "voc"}

    if from_fmt not in supported_from:
        raise ValueError(
            f"不支持的源格式: '{from_fmt}'，支持: {sorted(supported_from)}"
        )
    if to_fmt not in supported_to:
        raise ValueError(
            f"不支持的目标格式: '{to_fmt}'，支持: {sorted(supported_to)}"
        )

    # ---- Step 1: 解析 ----
    logger.info("解析源文件 [%s]: %s", from_fmt, src)

    if class_names is None:
        class_names = ["screw"]  # 默认为 one-class 检测
        logger.info("未提供类别名称，使用默认: %s", class_names)

    if from_fmt == "yolo":
        img_dir = Path(images_dir) if images_dir else None
        dataset = YOLOParser().parse(
            labels_dir=src,
            images_dir=img_dir,
            class_names=class_names,
        )
    elif from_fmt == "coco":
        dataset = COCOParser().parse(json_path=src)
    elif from_fmt == "cvat":
        dataset = CVATParser().parse(xml_path=src, class_names=class_names)
    elif from_fmt == "voc":
        dataset = PascalVOCParser().parse(
            annotations_dir=src, class_names=class_names
        )
    else:
        raise ValueError(f"不支持的源格式: '{from_fmt}'")

    dataset.print_stats()

    # ---- Step 2: 写出 ----
    logger.info("写出目标文件 [%s]: %s", to_fmt, dst)

    if to_fmt == "yolo":
        YOLOWriter().write(dataset=dataset, dst_dir=dst)
    elif to_fmt == "coco":
        COCOWriter().write(dataset=dataset, dst_path=dst)
    elif to_fmt == "voc":
        PascalVOCWriter().write(dataset=dataset, dst_dir=dst)
    else:
        raise ValueError(f"不支持的目标格式: '{to_fmt}'")

    logger.info("转换完成: [%s] %s → [%s] %s", from_fmt, src, to_fmt, dst)
    return dataset


# ---------------------------------------------------------------------------
# 数据集验证工具
# ---------------------------------------------------------------------------


def validate_yolo_dir(
    labels_dir: Path,
    images_dir: Optional[Path] = None,
    class_names: Optional[List[str]] = None,
    verbose: bool = False,
) -> Dict[str, int]:
    """
    验证 YOLO 标注目录的完整性和格式正确性。

    检查内容：
    - 每个 .txt 文件格式是否合法（字段数、坐标范围）
    - 是否有对应的图像文件
    - 是否存在空标注文件

    Parameters
    ----------
    labels_dir : Path
    images_dir : Path | None
    class_names : List[str] | None
    verbose : bool

    Returns
    -------
    Dict[str, int] : 统计报告字典，包含错误数、警告数等。
    """
    labels_dir = Path(labels_dir)
    label_files = sorted(labels_dir.glob("*.txt"))

    stats = {
        "total_files": len(label_files),
        "empty_files": 0,
        "missing_images": 0,
        "invalid_lines": 0,
        "bbox_out_of_range": 0,
        "ok_files": 0,
    }

    if images_dir is None:
        candidate = labels_dir.parent / "images"
        images_dir = candidate if candidate.is_dir() else labels_dir

    for lf in label_files:
        img_found = any(
            (images_dir / (lf.stem + ext)).exists()
            for ext in IMAGE_EXTENSIONS
        )
        if not img_found:
            stats["missing_images"] += 1
            if verbose:
                logger.warning("缺少对应图像: %s", lf.name)

        content = lf.read_text(encoding="utf-8").strip()
        if not content:
            stats["empty_files"] += 1
            continue

        file_ok = True
        for line_no, line in enumerate(content.splitlines(), 1):
            parts = line.strip().split()
            if len(parts) < 5:
                stats["invalid_lines"] += 1
                file_ok = False
                if verbose:
                    logger.warning("%s 行 %d: 字段数不足。", lf.name, line_no)
                continue
            try:
                cx_n, cy_n, w_n, h_n = map(float, parts[1:5])
            except ValueError:
                stats["invalid_lines"] += 1
                file_ok = False
                continue

            if not (0.0 <= cx_n <= 1.0 and 0.0 <= cy_n <= 1.0
                    and 0.0 < w_n <= 1.0 and 0.0 < h_n <= 1.0):
                stats["bbox_out_of_range"] += 1
                file_ok = False
                if verbose:
                    logger.warning(
                        "%s 行 %d: 坐标越界 (%.4f, %.4f, %.4f, %.4f)。",
                        lf.name, line_no, cx_n, cy_n, w_n, h_n,
                    )

        if file_ok and img_found:
            stats["ok_files"] += 1

    print("\n[YOLO 验证报告]")
    for k, v in stats.items():
        flag = "⚠️ " if (k != "ok_files" and v > 0) else "✅"
        print(f"  {flag} {k:25s}: {v}")
    print()

    return stats


def validate_coco_json(json_path: Path) -> Dict[str, int]:
    """
    验证 COCO JSON 文件的完整性。

    Parameters
    ----------
    json_path : Path

    Returns
    -------
    Dict[str, int]
    """
    json_path = Path(json_path)
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    images = {img["id"]: img for img in data.get("images", [])}
    anns = data.get("annotations", [])
    cats = data.get("categories", [])
    cat_ids = {c["id"] for c in cats}

    stats = {
        "n_images": len(images),
        "n_annotations": len(anns),
        "n_categories": len(cats),
        "missing_image_id": 0,
        "invalid_category_id": 0,
        "invalid_bbox": 0,
        "zero_area": 0,
        "duplicate_ann_id": 0,
    }

    ann_ids_seen: set = set()
    for ann in anns:
        ann_id = ann.get("id")
        if ann_id in ann_ids_seen:
            stats["duplicate_ann_id"] += 1
        ann_ids_seen.add(ann_id)

        if ann.get("image_id") not in images:
            stats["missing_image_id"] += 1

        if ann.get("category_id") not in cat_ids:
            stats["invalid_category_id"] += 1

        bbox = ann.get("bbox", [])
        if len(bbox) < 4:
            stats["invalid_bbox"] += 1
        elif bbox[2] <= 0 or bbox[3] <= 0:
            stats["zero_area"] += 1

    print("\n[COCO JSON 验证报告]")
    for k, v in stats.items():
        flag = "⚠️ " if (k not in ("n_images", "n_annotations", "n_categories") and v > 0) else "✅"
        print(f"  {flag} {k:25s}: {v}")
    print()

    return stats


# ---------------------------------------------------------------------------
# 命令行参数解析
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        prog="convert_annotations.py",
        description="标注格式转换工具 (Owner: D) — 支持 YOLO / COCO / CVAT / VOC 互转",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
支持的转换方向：
  cvat  → yolo   （最常用：CVAT 导出后转给 B 训练）
  cvat  → coco
  coco  → yolo   （Roboflow 下载后转换）
  coco  → voc
  yolo  → coco
  yolo  → voc
  voc   → yolo
  voc   → coco

示例：
  # CVAT XML → YOLO（one-class 螺丝检测）
  python tools/convert_annotations.py \\
      --src annotations/cvat_export.xml \\
      --dst annotations/yolo_labels/ \\
      --from_fmt cvat --to_fmt yolo \\
      --class_names screw

  # YOLO → COCO
  python tools/convert_annotations.py \\
      --src annotations/yolo_labels/ \\
      --dst annotations/coco.json \\
      --from_fmt yolo --to_fmt coco \\
      --images_dir frames/ \\
      --class_names screw

  # 仅统计（不转换）
  python tools/convert_annotations.py \\
      --src annotations/cvat_export.xml \\
      --from_fmt cvat --stats_only

  # 验证 YOLO 格式
  python tools/convert_annotations.py \\
      --src annotations/yolo_labels/ \\
      --from_fmt yolo --validate_only \\
      --images_dir frames/
        """,
    )

    # ---- 必需参数 ----
    parser.add_argument(
        "--src", "-s",
        required=True,
        help="源文件或目录路径（根据格式决定是文件还是目录）。",
    )
    parser.add_argument(
        "--from_fmt",
        required=True,
        choices=["yolo", "coco", "cvat", "voc"],
        help="源标注格式。",
    )

    # ---- 可选参数 ----
    parser.add_argument(
        "--dst", "-d",
        default=None,
        help="目标文件或目录路径（stats_only / validate_only 时可不提供）。",
    )
    parser.add_argument(
        "--to_fmt",
        choices=["yolo", "coco", "voc"],
        default=None,
        help="目标格式（stats_only / validate_only 时可不提供）。",
    )
    parser.add_argument(
        "--class_names",
        nargs="+",
        default=None,
        help="类别名称列表，空格分隔（例如：--class_names screw）。"
             "YOLO/CVAT/VOC 格式解析时使用；COCO 格式会自动从文件读取。",
    )
    parser.add_argument(
        "--images_dir",
        default=None,
        help="图像文件目录（YOLO 格式解析时用于查找对应图像以读取尺寸）。",
    )

    # ---- 模式选择 ----
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--stats_only",
        action="store_true",
        default=False,
        help="仅统计标注数量，不执行格式转换。",
    )
    mode_group.add_argument(
        "--validate_only",
        action="store_true",
        default=False,
        help="仅验证标注格式正确性，不执行转换。",
    )

    # ---- 其他 ----
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="输出详细验证信息。",
    )
    parser.add_argument(
        "--skip_empty",
        action="store_true",
        default=False,
        help="跳过无标注的图像（转换时不生成对应空文件）。",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------


def main() -> int:
    """
    标注转换工具主函数。

    Returns
    -------
    int : 退出码（0=成功，非 0=失败）。
    """
    args = _parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    src = Path(args.src)
    if not src.exists():
        logger.error("源路径不存在: %s", src)
        return 1

    # ================================================================
    # 模式 1：仅统计
    # ================================================================
    if args.stats_only:
        logger.info("统计模式：解析 [%s] %s", args.from_fmt, src)
        class_names = args.class_names or ["screw"]

        try:
            if args.from_fmt == "yolo":
                img_dir = Path(args.images_dir) if args.images_dir else None
                dataset = YOLOParser().parse(src, img_dir, class_names)
            elif args.from_fmt == "coco":
                dataset = COCOParser().parse(src)
            elif args.from_fmt == "cvat":
                dataset = CVATParser().parse(src, class_names)
            elif args.from_fmt == "voc":
                dataset = PascalVOCParser().parse(src, class_names)
            else:
                logger.error("不支持的格式: %s", args.from_fmt)
                return 1
        except Exception as e:
            logger.error("解析失败: %s", e)
            return 1

        dataset.print_stats()
        return 0

    # ================================================================
    # 模式 2：仅验证
    # ================================================================
    if args.validate_only:
        logger.info("验证模式: [%s] %s", args.from_fmt, src)
        try:
            if args.from_fmt == "yolo":
                img_dir = Path(args.images_dir) if args.images_dir else None
                validate_yolo_dir(
                    src, img_dir,
                    class_names=args.class_names,
                    verbose=args.verbose,
                )
            elif args.from_fmt == "coco":
                validate_coco_json(src)
            else:
                logger.warning(
                    "验证功能目前仅支持 yolo 和 coco 格式，[%s] 暂不支持。",
                    args.from_fmt,
                )
                return 1
        except Exception as e:
            logger.error("验证出错: %s", e)
            return 1
        return 0

    # ================================================================
    # 模式 3：格式转换（主模式）
    # ================================================================
    if not args.dst:
        logger.error("转换模式下必须提供 --dst。")
        return 1
    if not args.to_fmt:
        logger.error("转换模式下必须提供 --to_fmt。")
        return 1

    dst = Path(args.dst)
    logger.info(
        "开始转换: [%s] %s → [%s] %s",
        args.from_fmt, src, args.to_fmt, dst,
    )

    try:
        dataset = convert(
            src=src,
            dst=dst,
            from_fmt=args.from_fmt,
            to_fmt=args.to_fmt,
            class_names=args.class_names,
            images_dir=args.images_dir,
        )
    except (ValueError, FileNotFoundError) as e:
        logger.error("转换失败: %s", e)
        return 1
    except Exception as e:
        logger.error("意外错误: %s", e, exc_info=args.verbose)
        return 1

    logger.info("✅ 转换完成！")
    logger.info("   源格式  : %s", args.from_fmt)
    logger.info("   目标格式: %s", args.to_fmt)
    logger.info("   图像数量: %d", dataset.n_images)
    logger.info("   标注数量: %d", dataset.n_annotations)
    logger.info("   输出路径: %s", dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
