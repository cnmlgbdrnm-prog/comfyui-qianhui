"""Payload 编解码 —— 同步头 + Reed-Solomon 纠错。"""
try:
    from reedsolo import RSCodec
    _REEDSOLO_IMPORT_ERROR = None
except ImportError as exc:
    RSCodec = None
    _REEDSOLO_IMPORT_ERROR = exc

# ── Payload 编解码 ─────────────────────────────────────────────────────────────

_SYNC = 0xDEADBEEF  # 4 字节同步头
_RS = RSCodec(4) if RSCodec is not None else None


def _require_reedsolo(feature: str) -> None:
    if _RS is not None:
        return
    raise RuntimeError(
        f"{feature} 依赖 reedsolo，请在 ComfyUI 使用的 Python 环境中安装：pip install reedsolo"
    ) from _REEDSOLO_IMPORT_ERROR


def _int_to_bits(value: int, n_bits: int) -> list[int]:
    """将整数转为 n_bits 位的 bit 列表，MSB-first"""
    return [(value >> (n_bits - 1 - i)) & 1 for i in range(n_bits)]


def _bits_to_int(bits: list[int]) -> int:
    """将 MSB-first bit 列表转为整数"""
    result = 0
    for b in bits:
        result = (result << 1) | b
    return result


def _encode_payload(text: str) -> list[int]:
    """将文字编码为 bit 列表：[4B sync][4B rs_len][RS-encoded bytes]，MSB-first。

    rs_len = len(original_utf8) + 4（含 4 字节 RS 校验符）。
    """
    _require_reedsolo("隐形水印编码")
    utf8 = text.encode("utf-8")
    rs_bytes = bytes(_RS.encode(utf8))  # len = len(utf8) + 4
    rs_len = len(rs_bytes)
    bits: list[int] = []
    bits.extend(_int_to_bits(_SYNC, 32))
    bits.extend(_int_to_bits(rs_len, 32))
    for byte in rs_bytes:
        bits.extend(_int_to_bits(byte, 8))
    return bits


def _decode_payload(bits: list[int]) -> str:
    """从 bit 列表解析 RS-protected payload；同步头不匹配或 RS 解码失败返回空字符串。"""
    _require_reedsolo("隐形水印解码")
    if len(bits) < 64:
        return ""
    if _bits_to_int(bits[:32]) != _SYNC:
        return ""
    rs_len = _bits_to_int(bits[32:64])
    if rs_len < 4 or len(bits) < 64 + rs_len * 8:
        return ""
    rs_bytes = bytes(
        _bits_to_int(bits[64 + i * 8: 64 + (i + 1) * 8])
        for i in range(rs_len)
    )
    try:
        decoded_bytes = bytes(_RS.decode(rs_bytes)[0])
        return decoded_bytes.decode("utf-8")
    except Exception:
        return ""
