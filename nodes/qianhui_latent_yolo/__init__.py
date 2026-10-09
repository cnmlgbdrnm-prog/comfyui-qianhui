"""千绘 · Latent 智能修复模块（YOLO 检测 + SAM 精修 + ControlNet）。

界面层已拆成 4 个节点（那个一体化的巨型节点已于 T13 退役，见计划
`docs/superpowers/plans/2026-10-09-latent-node-split.md`）：

    [千绘 · YOLO检测] → [千绘 · SAM精修] → [千绘 · 区域修复] → [千绘 · 全局降噪]

4 个节点之间除了 `Latent` 之外还传一路自定义 socket `QIANHUI_DETECT`
（检测数据：逐帧检测记录 + 阶段标志），它替代了单体里那个跨阶段共享的
`YoloRunContext`。T13 同时删掉了 `node.py` / `mixins/pipeline/report.py` /
`prepare_run` / `YoloRunContext`，并把单体的旧注册名从根 `__init__.py` 的注册表摘掉
（那个名字现在是 `QianhuiYoloDetect` 等 4 个）。
"""
from .node_detect import QianhuiYoloDetect
from .node_global import QianhuiGlobalPass
from .node_repair import QianhuiRegionRepair
from .node_sam import QianhuiSamRefine

NODE_CLASS_MAPPINGS = {
    "QianhuiYoloDetect": QianhuiYoloDetect,
    "QianhuiSamRefine": QianhuiSamRefine,
    "QianhuiRegionRepair": QianhuiRegionRepair,
    "QianhuiGlobalPass": QianhuiGlobalPass,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "QianhuiYoloDetect": "千绘 · YOLO检测",
    "QianhuiSamRefine": "千绘 · SAM精修",
    "QianhuiRegionRepair": "千绘 · 区域修复",
    "QianhuiGlobalPass": "千绘 · 全局降噪",
}

__all__ = [
    "QianhuiYoloDetect",
    "QianhuiSamRefine",
    "QianhuiRegionRepair",
    "QianhuiGlobalPass",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
