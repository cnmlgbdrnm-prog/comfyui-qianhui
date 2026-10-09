"""千绘 · 水印节点 — QianhuiWatermarkAdd / QianhuiWatermarkExtract"""
import numpy as np
import torch

from .invisible import _embed_invisible, _extract_invisible
from .visible import _apply_visible_watermark

# ── ComfyUI 节点类 ─────────────────────────────────────────────────────────────

class QianhuiWatermarkAdd:
    """ComfyUI 节点：在图像上添加可见/隐形水印"""

    CATEGORY = "千绘/水印"
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("原始图像", "水印图像")
    FUNCTION = "apply_watermark"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "watermark_type": (["文字", "图片", "隐形文字", "文字+隐形"],),
                "text": ("STRING", {"default": "© 版权所有", "multiline": False}),
                "opacity": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "spacing": ("INT", {"default": 100, "min": 10, "max": 500}),
                "rotation": ("INT", {"default": -30, "min": -180, "max": 180}),
                "font_size": ("INT", {"default": 36, "min": 8, "max": 200}),
                "color": ("STRING", {"default": "#FFFFFF"}),
            },
            "optional": {
                "watermark_image": ("IMAGE",),
            },
        }

    def apply_watermark(
        self,
        image: "torch.Tensor",
        watermark_type: str,
        text: str,
        opacity: float,
        spacing: int,
        rotation: int,
        font_size: int,
        color: str,
        watermark_image: "torch.Tensor | None" = None,
    ):
        """
        处理批量图像，返回 (原始图像, 水印图像)。
        两者均为 (B, H, W, C) float32 Tensor，值域 [0, 1]。
        """
        # image: (B, H, W, C) float32 [0,1]
        B = image.shape[0]
        original = image.clone()
        results = []

        wm_np = None
        if watermark_image is not None:
            # 取第 0 帧作为水印图片
            wm_np = (watermark_image[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

        for i in range(B):
            frame_np = (image[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

            if watermark_type == "隐形文字":
                out_np = _embed_invisible(frame_np, text)
            elif watermark_type == "文字+隐形":
                out_np = _apply_visible_watermark(
                    frame_np, wm_type="文字", text=text,
                    opacity=opacity, spacing=spacing, rotation=rotation,
                    font_size=font_size, color=color,
                    watermark_image=None,
                )
                out_np = _embed_invisible(out_np, text)
            else:
                out_np = _apply_visible_watermark(
                    frame_np, wm_type=watermark_type, text=text,
                    opacity=opacity, spacing=spacing, rotation=rotation,
                    font_size=font_size, color=color,
                    watermark_image=wm_np,
                )

            results.append(torch.from_numpy(out_np.astype(np.float32) / 255.0))

        watermarked = torch.stack(results, dim=0)
        return (original, watermarked)


class QianhuiWatermarkExtract:
    """ComfyUI 节点：从图像中提取隐形 DCT-QIM 水印"""

    CATEGORY = "千绘/水印"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("提取文本",)
    FUNCTION = "extract_watermark"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    def extract_watermark(self, image: "torch.Tensor"):
        """
        从图像第 0 帧提取隐形水印。

        Args:
            image: (B, H, W, C) float32 Tensor

        Returns:
            tuple: (提取文本 str,)  — 无水印时返回 ("",)
        """
        if image.shape[0] > 1:
            print(f"[WatermarkExtract] 警告：批量输入 B={image.shape[0]}，仅处理第 0 帧")

        frame_np = (image[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        result = _extract_invisible(frame_np)
        if not result:
            print("[WatermarkExtract] 警告：未检测到有效隐形水印，返回空字符串")
        return (result,)


NODE_CLASS_MAPPINGS = {
    "QianhuiWatermarkAdd": QianhuiWatermarkAdd,
    "QianhuiWatermarkExtract": QianhuiWatermarkExtract,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "QianhuiWatermarkAdd": "千绘 · 水印添加",
    "QianhuiWatermarkExtract": "千绘 · 水印提取",
}
