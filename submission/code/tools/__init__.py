"""
tools - 数据工具包
Owner: D（工程封装）

包含以下子模块：
- extract_keyframes   : 关键帧抽取工具（批量导出关键帧图像，供标注使用）
- export_crops        : 检测 crop 导出工具（从视频帧中裁切螺丝区域）
- convert_annotations : 标注格式转换工具（CVAT XML / YOLO / COCO JSON 互转）
- benchmark           : 速度 benchmark 工具（测量各模块耗时）
- ablation            : 消融实验记录工具（对比不同配置的计数误差）
"""
