"""隐形水印 —— DCT-QIM 嵌入与提取。"""
import numpy as np
from PIL import Image

from .dct import _dct8, _get_block_q, _idct8
from .payload import _decode_payload, _encode_payload, _require_reedsolo

# ── 隐形水印（DCT QIM） ────────────────────────────────────────────────────────

_QIM_REPEAT = 3    # 每 bit 重复嵌入次数（3× repetition coding）；量化步长由 _get_block_q() 自适应决定


def _embed_invisible(img_rgb: np.ndarray, text: str) -> np.ndarray:
    """
    将版权文字以 QIM-DCT 方式嵌入 RGB 图像。

    增强策略：
      - 自适应量化步长（按块方差：平坦 q=25，普通 q=50，高纹理 q=80）
      - DCT 系数位置 [2,1]（低频，更抗压缩/缩放）
      - 3× 重复编码（每 bit 写入 3 个连续块）
      - Y + Cb 双通道冗余
      - Reed-Solomon 纠错（RSCodec(4)，2 字节容错）

    Args:
        img_rgb: uint8 ndarray, shape (H, W, 3)
        text:    要嵌入的 UTF-8 文字

    Returns:
        watermarked uint8 ndarray, shape (H, W, 3)

    Raises:
        ValueError: 图像容量不足或文字超出容量
    """
    _require_reedsolo("隐形水印")
    H, W = img_rgb.shape[:2]
    bh, bw = H // 8, W // 8
    capacity_bits  = bh * bw // _QIM_REPEAT
    capacity_bytes = capacity_bits // 8 - 12  # 8B header + 4B RS 校验

    if capacity_bytes <= 0:
        raise ValueError(
            f"图像太小无法嵌入水印（{H}×{W}）。"
            f"至少需要 bh*bw >= {12 * _QIM_REPEAT * 8} 个完整 8×8 块（约 128×128）。"
        )
    text_bytes = text.encode("utf-8")
    if len(text_bytes) > capacity_bytes:
        raise ValueError(
            f"文字 UTF-8 字节数 ({len(text_bytes)}) 超出图像容量 ({capacity_bytes} 字节)。"
        )

    bits = _encode_payload(text)

    # RGB → YCbCr
    pil_img = Image.fromarray(img_rgb, mode="RGB").convert("YCbCr")
    ycbcr = np.array(pil_img, dtype=np.float64)
    Y  = ycbcr[:, :, 0].copy()
    Cb = ycbcr[:, :, 1].copy()

    def _embed_channel(ch: np.ndarray) -> np.ndarray:
        """在单个通道上做自适应 q 的 3× 重复 QIM 嵌入。
        for/else/break 惯用法：内层 break 跳过 else，触发外层 break。
        """
        ch = ch.copy()
        block_idx = 0
        for bit in bits:
            for _ in range(_QIM_REPEAT):
                if block_idx >= bh * bw:
                    break
                r = block_idx // bw
                c = block_idx % bw
                block = ch[r*8:(r+1)*8, c*8:(c+1)*8]
                q = _get_block_q(float(np.var(block)), embed=True)
                coef = _dct8(block)
                val = coef[2, 1]
                if bit == 1:
                    step = round((val / q - 1) / 2) * 2 + 1
                else:
                    step = round(val / q / 2) * 2
                coef[2, 1] = step * q
                ch[r*8:(r+1)*8, c*8:(c+1)*8] = _idct8(coef)
                block_idx += 1
            else:
                continue
            break
        return ch

    Y  = _embed_channel(Y)
    Cb = _embed_channel(Cb)

    Y  = np.clip(Y,  0, 255)
    Cb = np.clip(Cb, 0, 255)
    ycbcr[:, :, 0] = np.round(Y)
    ycbcr[:, :, 1] = np.round(Cb)
    result_pil = Image.fromarray(ycbcr.astype(np.uint8), mode="YCbCr").convert("RGB")
    return np.array(result_pil, dtype=np.uint8)


def _extract_invisible(img_rgb: np.ndarray) -> str:
    """
    从 RGB 图像中提取 QIM-DCT 隐形水印文字。

    自适应 q（含死区）+ Y/Cb 双通道 6 票多数决 + RS 纠错。

    Returns:
        提取的文字；若无有效水印或 RS 纠错失败返回空字符串 ""
    """
    _require_reedsolo("水印提取")
    H, W = img_rgb.shape[:2]
    bh, bw = H // 8, W // 8
    total_blocks = bh * bw

    pil_img = Image.fromarray(img_rgb, mode="RGB").convert("YCbCr")
    ycbcr = np.array(pil_img, dtype=np.float64)
    Y  = ycbcr[:, :, 0]
    Cb = ycbcr[:, :, 1]

    def _read_raw_bits(ch: np.ndarray) -> list[int]:
        """从通道读取每个块在 [2,1] 系数上的自适应 QIM 判决 bit"""
        raw = []
        for block_idx in range(total_blocks):
            r = block_idx // bw
            c = block_idx % bw
            block = ch[r*8:(r+1)*8, c*8:(c+1)*8]
            q = _get_block_q(float(np.var(block)), embed=False)
            coef = _dct8(block)
            raw.append(int(round(coef[2, 1] / q)) % 2)
        return raw

    y_raw  = _read_raw_bits(Y)
    cb_raw = _read_raw_bits(Cb)

    _TOTAL_VOTES    = _QIM_REPEAT * 2           # Y 通道 + Cb 通道各 REPEAT 票
    _VOTE_THRESHOLD = _TOTAL_VOTES // 2 + 1     # 严格多数（> 半数）
    n_bits = total_blocks // _QIM_REPEAT
    decoded: list[int] = []
    for i in range(n_bits):
        votes = (
            sum(y_raw[i * _QIM_REPEAT:(i + 1) * _QIM_REPEAT])
            + sum(cb_raw[i * _QIM_REPEAT:(i + 1) * _QIM_REPEAT])
        )
        decoded.append(1 if votes >= _VOTE_THRESHOLD else 0)

    return _decode_payload(decoded)
