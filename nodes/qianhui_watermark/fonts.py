"""字体加载 —— 按候选列表挑选支持中文的字体。"""
import os

from PIL import ImageFont

# 按优先级依次尝试的字体列表（优先中文字体，兼容 Windows/Linux/macOS）
_FONT_CANDIDATES = [
    # Windows 中文字体
    r"C:\Windows\Fonts\msyh.ttc",      # 微软雅黑
    r"C:\Windows\Fonts\simhei.ttf",    # 黑体
    r"C:\Windows\Fonts\simsun.ttc",    # 宋体
    r"C:\Windows\Fonts\msyhbd.ttc",    # 微软雅黑 Bold
    # macOS 中文字体
    "/System/Library/Fonts/PingFang.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    # Linux 常见中文字体
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    # 通用西文备用
    "arial.ttf",
]


def _load_font(font_size: int) -> ImageFont.FreeTypeFont:
    """加载支持中文的字体，按候选列表依次尝试，最终回退内置字体"""
    for path in _FONT_CANDIDATES:
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, font_size)
            except (IOError, OSError):
                continue
    return ImageFont.load_default()
