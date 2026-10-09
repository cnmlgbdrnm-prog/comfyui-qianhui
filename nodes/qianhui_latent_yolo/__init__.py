"""千绘 · Latent 智能修复模块（YOLO 检测 + SAM 精修 + ControlNet）。

界面层已拆成 4 个节点（巨型节点 `QianhuiLatentYOLO` 退役中，见计划
`docs/superpowers/plans/2026-10-09-latent-node-split.md`）：

    [千绘 · YOLO检测] → [千绘 · SAM精修] → [千绘 · 区域修复] → [千绘 · 全局降噪]

`QianhuiLatentYOLO` 与 4 个新节点**并存**（T8~T11 阶段：新旧都能跑，便于对拍），
摘掉单体的那一步（T13）由根 `__init__.py` 的注册表决定 —— 那个文件是唯一的注册处。
"""
from .node import QianhuiLatentYOLO
from .node_detect import QianhuiYoloDetect
from .node_global import QianhuiGlobalPass
from .node_repair import QianhuiRegionRepair
from .node_sam import QianhuiSamRefine

NODE_CLASS_MAPPINGS = {
    "QianhuiLatentYOLO": QianhuiLatentYOLO,
    "QianhuiYoloDetect": QianhuiYoloDetect,
    "QianhuiSamRefine": QianhuiSamRefine,
    "QianhuiRegionRepair": QianhuiRegionRepair,
    "QianhuiGlobalPass": QianhuiGlobalPass,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "QianhuiLatentYOLO": "千绘 · Latent智能修复 (YOLO+SAM+CN V6.11.3)",
    "QianhuiYoloDetect": "千绘 · YOLO检测",
    "QianhuiSamRefine": "千绘 · SAM精修",
    "QianhuiRegionRepair": "千绘 · 区域修复",
    "QianhuiGlobalPass": "千绘 · 全局降噪",
}

__all__ = [
    "QianhuiLatentYOLO",
    "QianhuiYoloDetect",
    "QianhuiSamRefine",
    "QianhuiRegionRepair",
    "QianhuiGlobalPass",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
