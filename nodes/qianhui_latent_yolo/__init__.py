"""千绘 · Latent 智能修复模块（YOLO 检测 + SAM 精修 + ControlNet）。"""
from .node import QianhuiLatentYOLO

NODE_CLASS_MAPPINGS = {
    "QianhuiLatentYOLO": QianhuiLatentYOLO,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "QianhuiLatentYOLO": "千绘 · Latent智能修复 (YOLO+SAM+CN V6.11.3)",
}

__all__ = ["QianhuiLatentYOLO", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
