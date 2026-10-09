"""DCT 工具 —— 8×8 正交 DCT-II 与自适应 QIM 量化步长。"""
import numpy as np

# ── DCT 工具 ──────────────────────────────────────────────────────────────────

def _make_dct_matrix() -> np.ndarray:
    """构造 8×8 正交 DCT-II 变换矩阵 D，满足 D @ D.T == I"""
    D = np.zeros((8, 8), dtype=np.float64)
    for k in range(8):
        for n in range(8):
            D[k, n] = np.cos(np.pi * k * (2 * n + 1) / 16.0)
    D[0] *= 1.0 / np.sqrt(8)
    D[1:] *= np.sqrt(2.0 / 8)
    return D

_DCT_D = _make_dct_matrix()   # 模块级缓存，避免重复计算


def _dct8(block: np.ndarray) -> np.ndarray:
    """对 8×8 float64 块做二维 DCT-II，返回系数矩阵"""
    return _DCT_D @ block @ _DCT_D.T


def _idct8(coef: np.ndarray) -> np.ndarray:
    """对 8×8 DCT 系数矩阵做逆变换，返回空间域 float64 块"""
    return _DCT_D.T @ coef @ _DCT_D


def _get_block_q(variance: float, embed: bool) -> float:
    """根据 8×8 块 Y 通道像素方差返回自适应 QIM 量化步长。

    embed=True  使用原始嵌入阈值（较严格）。
    embed=False 使用含死区的提取阈值（较宽松），吸收嵌入导致的方差微增。

    嵌入阈值：var <  100 → 25.0；var <  800 → 50.0；var ≥  800 → 80.0
    提取阈值：var <  150 → 25.0；var < 1000 → 50.0；var ≥ 1000 → 80.0
    """
    if embed:
        if variance < 100:
            return 25.0
        elif variance < 800:
            return 50.0
        else:
            return 80.0
    else:
        if variance < 150:
            return 25.0
        elif variance < 1000:
            return 50.0
        else:
            return 80.0
