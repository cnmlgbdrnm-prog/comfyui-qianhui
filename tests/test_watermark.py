# -*- coding: utf-8 -*-
"""水印模块单测（算法层 + 节点层）。

**搬运规则（定死）**：17 个用例逐一对应源文件 `水印/test_watermark.py` 的同名用例，
输入、断言边界、边界值全部照搬，只做三处机械改写：

1. `def test_x():` → 类内 `def test_x(self):`
2. `from watermark_nodes import X` → `from qh_watermark.<模块> import X`
3. `assert <expr>, "<msg>"` → `self.assertTrue(<expr>, "<msg>")`，
   比较式改 `self.assertEqual(<got>, <want>, "<msg>")`

去掉源文件里的 `print("PASS ...")`。**不要重写输入、不要收紧或放宽边界**——
源测试的边界值是实测出来的（例如 `test_adaptive_q` 的 11 个点专门卡在
阈值两侧，`test_rs_roundtrip` 的 64/72/80 三个 bit 偏移对应 1/2/3 字节损坏）。

源文件 `test_invisible_capacity_error` 用的是 try/except（没抛异常只打印 FAIL、
不会真的失败），此处按它的 docstring 意图写成 `assertRaises`——这是强化，
不是改写，并已在此注明。
"""
import importlib.util
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _loader import pkgs_dir  # noqa: E402

# 只把包**注册**进 sys.modules，不执行它的 __init__.py。
#
# 为什么：`__init__.py` 会 `from .nodes import ...` → `import torch`。
# 若在这里 exec 它，torch 就成了**整个模块**的导入期依赖——13 个纯 numpy 的
# 算法用例会因为 import torch 失败而整体报错，而不是让 4 个节点用例单独跳过。
# 源测试特意在每个节点用例**内部**才 `import torch`，就是为了让算法层独立；
# 这里用不执行 __init__ 的注册方式保住同样的分层。
# 子模块导入照常可用：Python 在 sys.modules 里找到 qh_watermark 后，
# 用它的 __path__ 去定位 qh_watermark.dct / .payload / ... ，不会回头跑 __init__。
_PKG_DIR = pkgs_dir("qianhui_watermark")
_spec = importlib.util.spec_from_file_location(
    "qh_watermark", os.path.join(_PKG_DIR, "__init__.py"),
    submodule_search_locations=[_PKG_DIR],
)
sys.modules["qh_watermark"] = importlib.util.module_from_spec(_spec)


class TestDct(unittest.TestCase):
    def test_dct_roundtrip(self):
        """DCT 后 IDCT 应还原原始块（误差 < 1e-9）"""
        from qh_watermark.dct import _dct8, _idct8
        rng = np.random.default_rng(42)
        block = rng.random((8, 8)) * 255
        reconstructed = _idct8(_dct8(block))
        self.assertTrue(
            np.allclose(block, reconstructed, atol=1e-9),
            f"DCT round-trip error: max={np.abs(block - reconstructed).max()}",
        )

    def test_adaptive_q(self):
        """_get_block_q 应按方差和 embed 模式返回正确的 q 值"""
        from qh_watermark.dct import _get_block_q

        # embed=True 阈值：<100=25, <800=50, >=800=80
        self.assertEqual(_get_block_q(0.0,    embed=True), 25.0, "flat embed")
        self.assertEqual(_get_block_q(99.9,   embed=True), 25.0, "flat-edge embed")
        self.assertEqual(_get_block_q(100.0,  embed=True), 50.0, "mid-low embed")
        self.assertEqual(_get_block_q(799.9,  embed=True), 50.0, "mid-high embed")
        self.assertEqual(_get_block_q(800.0,  embed=True), 80.0, "texture embed")
        self.assertEqual(_get_block_q(9999.0, embed=True), 80.0, "high-texture embed")

        # embed=False 阈值（含死区）：<150=25, <1000=50, >=1000=80
        self.assertEqual(_get_block_q(0.0,    embed=False), 25.0, "flat extract")
        self.assertEqual(_get_block_q(149.9,  embed=False), 25.0, "flat-edge extract")
        self.assertEqual(_get_block_q(150.0,  embed=False), 50.0, "mid-low extract")
        self.assertEqual(_get_block_q(999.9,  embed=False), 50.0, "mid-high extract")
        self.assertEqual(_get_block_q(1000.0, embed=False), 80.0, "texture extract")


class TestPayload(unittest.TestCase):
    def test_payload_roundtrip_ascii(self):
        """ASCII 文字 payload 编码后解码应还原原文"""
        from qh_watermark.payload import _encode_payload, _decode_payload
        text = "© Alice 2026"
        bits = _encode_payload(text)
        result = _decode_payload(bits)
        self.assertEqual(result, text, f"Expected {text!r}, got {result!r}")

    def test_payload_roundtrip_cjk(self):
        """中文 payload 编码后解码应还原原文"""
        from qh_watermark.payload import _encode_payload, _decode_payload
        text = "版权所有 张三"
        bits = _encode_payload(text)
        result = _decode_payload(bits)
        self.assertEqual(result, text, f"Expected {text!r}, got {result!r}")

    def test_decode_no_sync(self):
        """无同步头的 bit 流应返回空字符串"""
        from qh_watermark.payload import _decode_payload
        random_bits = [0, 1] * 64   # 128 bits，无有效同步头
        result = _decode_payload(random_bits)
        self.assertEqual(result, "", f"Expected '', got {result!r}")

    def test_rs_roundtrip(self):
        """RS 编码后人工损坏 1-2 字节，_decode_payload 仍能恢复原文"""
        from qh_watermark.payload import _encode_payload, _decode_payload

        text = "© RS 测试"
        bits = _encode_payload(text)

        # 正常解码
        self.assertEqual(_decode_payload(bits), text, "normal decode failed")

        # 损坏第 1 个 RS 数据字节（bit 64-71 对应 rs_bytes[0]）
        corrupted = list(bits)
        for i in range(64, 72):
            corrupted[i] ^= 1
        self.assertEqual(_decode_payload(corrupted), text, "1-byte corruption not recovered")

        # 损坏第 2 个 RS 数据字节（bits 72-79 也翻转）
        for i in range(72, 80):
            corrupted[i] ^= 1
        self.assertEqual(_decode_payload(corrupted), text, "2-byte corruption not recovered")

        # 损坏 3 个字节：超出纠错能力，结果不再等于原文
        for i in range(80, 88):
            corrupted[i] ^= 1
        result = _decode_payload(corrupted)
        self.assertNotEqual(result, text,
                            f"3-byte corruption should not recover original, got {result!r}")


class TestInvisible(unittest.TestCase):
    def test_invisible_roundtrip(self):
        """隐形水印嵌入后提取应还原原文，PNG 无损路径"""
        from qh_watermark.invisible import _embed_invisible, _extract_invisible
        rng = np.random.default_rng(0)
        img = (rng.random((512, 512, 3)) * 255).astype(np.uint8)
        text = "© 版权 Alice 2026"
        watermarked = _embed_invisible(img, text)
        result = _extract_invisible(watermarked)
        self.assertEqual(result, text, f"Expected {text!r}, got {result!r}")

    def test_invisible_no_watermark(self):
        """无水印图像应返回空字符串"""
        from qh_watermark.invisible import _extract_invisible
        rng = np.random.default_rng(1)
        img = (rng.random((256, 256, 3)) * 255).astype(np.uint8)
        result = _extract_invisible(img)
        self.assertEqual(result, "", f"Expected '', got {result!r}")

    def test_invisible_psnr(self):
        """隐形水印 PSNR 应 ≥ 30dB（q=50 预计约 38 dB）"""
        from qh_watermark.invisible import _embed_invisible
        rng = np.random.default_rng(2)
        img = (rng.random((512, 512, 3)) * 255).astype(np.uint8)
        text = "© Alice2026 Corp"   # 16 bytes UTF-8
        watermarked = _embed_invisible(img, text)
        mse = np.mean((img.astype(np.float64) - watermarked.astype(np.float64)) ** 2)
        psnr = 10 * np.log10(255.0 ** 2 / mse) if mse > 0 else float("inf")
        self.assertTrue(psnr >= 30.0, f"PSNR too low: {psnr:.2f} dB")

    def test_invisible_capacity_error(self):
        """payload 超容量应抛 ValueError"""
        from qh_watermark.invisible import _embed_invisible
        img = np.zeros((64, 64, 3), dtype=np.uint8)  # 容量 = (8*8)//3//8 - 12 < 0 → ValueError
        # 源测试此处用 try/except 且不 assert（没抛异常也只打印 FAIL），
        # 按 docstring 意图改为 assertRaises——强化，非改写。
        with self.assertRaises(ValueError):
            _embed_invisible(img, "x")


class TestVisible(unittest.TestCase):
    def test_visible_text_shape(self):
        """文字平铺水印输出形状应与输入相同"""
        from qh_watermark.visible import _apply_visible_watermark
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        result = _apply_visible_watermark(
            img, wm_type="文字", text="TEST",
            opacity=0.3, spacing=80, rotation=-30,
            font_size=24, color="#FF0000",
            watermark_image=None,
        )
        self.assertEqual(result.shape, (256, 256, 3), f"Shape mismatch: {result.shape}")
        # 水印应改变至少一个像素
        self.assertFalse(np.array_equal(img, result), "Watermark had no effect")

    def test_visible_invalid_color_fallback(self):
        """无效颜色值应回退白色，不抛异常"""
        from qh_watermark.visible import _apply_visible_watermark
        img = np.ones((64, 64, 3), dtype=np.uint8) * 50
        result = _apply_visible_watermark(
            img, wm_type="文字", text="X",
            opacity=1.0, spacing=40, rotation=0,
            font_size=16, color="not-a-color",
            watermark_image=None,
        )
        self.assertEqual(result.shape, (64, 64, 3))

    def test_visible_image_wm_missing_fallback(self):
        """图片模式但 watermark_image=None 应回退文字模式，不抛异常"""
        from qh_watermark.visible import _apply_visible_watermark
        img = np.zeros((128, 128, 3), dtype=np.uint8)
        result = _apply_visible_watermark(
            img, wm_type="图片", text="fallback",
            opacity=0.5, spacing=60, rotation=0,
            font_size=20, color="#FFFFFF",
            watermark_image=None,
        )
        self.assertEqual(result.shape, (128, 128, 3))


class TestNodes(unittest.TestCase):
    def test_node_watermarkadd_text(self):
        """WatermarkAdd 文字模式：输出形状与输入相同，双输出"""
        import torch
        from qh_watermark.nodes import QianhuiWatermarkAdd
        node = QianhuiWatermarkAdd()
        # ComfyUI IMAGE tensor: (B, H, W, C) float32 0-1
        img = torch.zeros(1, 64, 64, 3)
        original, watermarked = node.apply_watermark(
            image=img,
            watermark_type="文字",
            text="TEST © 2026",
            opacity=0.5,
            spacing=30,
            rotation=-30,
            font_size=16,
            color="#FFFFFF",
        )
        self.assertEqual(original.shape, (1, 64, 64, 3), f"original shape: {original.shape}")
        self.assertEqual(watermarked.shape, (1, 64, 64, 3), f"watermarked shape: {watermarked.shape}")
        # 原图应未被修改
        self.assertTrue(torch.equal(original, img), "original image was modified")

    def test_node_watermarkadd_invisible(self):
        """WatermarkAdd 隐形模式：原图输出不变，水印图肉眼无差异"""
        import torch
        from qh_watermark.nodes import QianhuiWatermarkAdd
        node = QianhuiWatermarkAdd()
        rng = np.random.default_rng(7)
        arr = rng.random((1, 512, 512, 3)).astype(np.float32)
        img = torch.from_numpy(arr)
        original, watermarked = node.apply_watermark(
            image=img,
            watermark_type="隐形文字",
            text="copyright",
            opacity=0.3,
            spacing=100,
            rotation=-30,
            font_size=36,
            color="#FFFFFF",
        )
        self.assertFalse(torch.equal(original, watermarked), "invisible watermark had no effect")
        diff = (original - watermarked).abs().max().item()
        # 源测试行尾注释原文："conservative bound; actual max ~0.07"。
        # 该实测值已过时——本模块实际测量约 0.2076，仍远低于 0.40 的界。
        # 断言本身照搬源文件未动；这行注释只是不再复述那个假数字。
        self.assertTrue(diff < 0.40, f"Invisible watermark caused large pixel diff: {diff}")

    def test_node_watermarkextract_roundtrip(self):
        """WatermarkExtract 应能从 WatermarkAdd 隐形输出中还原文字"""
        import torch
        from qh_watermark.nodes import QianhuiWatermarkAdd, QianhuiWatermarkExtract
        add = QianhuiWatermarkAdd()
        extract = QianhuiWatermarkExtract()

        rng = np.random.default_rng(99)
        arr = rng.random((1, 256, 256, 3)).astype(np.float32)
        img = torch.from_numpy(arr)
        text = "hello 水印"

        _, watermarked = add.apply_watermark(
            image=img, watermark_type="隐形文字", text=text,
            opacity=0.3, spacing=100, rotation=-30,
            font_size=36, color="#FFFFFF",
        )
        (extracted_text,) = extract.extract_watermark(image=watermarked)
        self.assertEqual(extracted_text, text, f"Expected {text!r}, got {extracted_text!r}")

    def test_node_watermarkextract_no_wm(self):
        """无水印图像应返回空字符串"""
        import torch
        from qh_watermark.nodes import QianhuiWatermarkExtract
        extract = QianhuiWatermarkExtract()
        img = torch.zeros(1, 128, 128, 3)
        (result,) = extract.extract_watermark(image=img)
        self.assertEqual(result, "", f"Expected '', got {result!r}")


if __name__ == "__main__":
    unittest.main()
