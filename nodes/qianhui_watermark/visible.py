"""可见水印 —— 平铺文字/图片水印。"""
import numpy as np
from PIL import Image, ImageDraw

from .fonts import _load_font

# ── 可见水印 ───────────────────────────────────────────────────────────────────

def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    """解析 '#RRGGBB' 颜色字符串；格式无效时回退白色并打印警告"""
    color = color.strip()
    if len(color) == 7 and color.startswith("#"):
        try:
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
            return (r, g, b)
        except ValueError:
            pass
    print(f"[WatermarkAdd] 警告：颜色值 '{color}' 无效，回退为白色 #FFFFFF")
    return (255, 255, 255)


def _make_text_tile(text: str, font_size: int, color: tuple[int, int, int],
                    rotation: int) -> Image.Image:
    """生成旋转后的文字水印 tile（RGBA 透明背景）"""
    # 先在足够大的画布上渲染文字
    dummy = Image.new("RGBA", (1, 1))
    draw = ImageDraw.Draw(dummy)
    font = _load_font(font_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0] + 4
    th = bbox[3] - bbox[1] + 4

    tile = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
    ImageDraw.Draw(tile).text((2, 2), text, font=font,
                               fill=(*color, 255))
    return tile.rotate(rotation, expand=True)


def _apply_visible_watermark(
    img_rgb: np.ndarray,
    wm_type: str,
    text: str,
    opacity: float,
    spacing: int,
    rotation: int,
    font_size: int,
    color: str,
    watermark_image: "np.ndarray | None",
) -> np.ndarray:
    """
    在 RGB 图像上平铺可见水印（文字或图片）。

    Returns:
        watermarked uint8 ndarray, shape (H, W, 3)
    """
    H, W = img_rgb.shape[:2]
    rgb = _hex_to_rgb(color)

    # 处理图片模式缺失 watermark_image 的情况
    if wm_type == "图片" and watermark_image is None:
        print("[WatermarkAdd] 警告：图片模式未连接 watermark_image，回退为文字模式")
        wm_type = "文字"

    # 构建 tile
    if wm_type == "图片" and watermark_image is not None:
        tile_pil = Image.fromarray(
            (watermark_image * 255).clip(0, 255).astype(np.uint8)
            if watermark_image.dtype != np.uint8 else watermark_image,
            mode="RGB"
        ).convert("RGBA")
        tile_pil = tile_pil.rotate(rotation, expand=True)
    else:
        tile_pil = _make_text_tile(text, font_size, rgb, rotation)

    tw, th = tile_pil.size

    # 创建透明水印层
    wm_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    for y in range(-th, H + th, th + spacing):
        for x in range(-tw, W + tw, tw + spacing):
            wm_layer.paste(tile_pil, (x, y), tile_pil)

    # 调整 tile 透明度
    r2, g2, b2, a2 = wm_layer.split()
    a2 = a2.point(lambda p: int(p * opacity))
    wm_layer = Image.merge("RGBA", (r2, g2, b2, a2))

    # 合成到原图
    base = Image.fromarray(img_rgb, mode="RGB").convert("RGBA")
    composited = Image.alpha_composite(base, wm_layer)
    return np.array(composited.convert("RGB"), dtype=np.uint8)
