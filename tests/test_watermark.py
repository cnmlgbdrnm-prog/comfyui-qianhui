# -*- coding: utf-8 -*-
"""水印模块单测（算法层 + 节点层）。搬运自 水印/test_watermark.py 的 17 个用例。"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _loader import load_package, pkgs_dir  # noqa: E402

WM = load_package("qh_watermark", pkgs_dir("qianhui_watermark"))


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
        """块方差分档应返回 25 / 50 / 80，且提取阈值比嵌入阈值宽松"""
        from qh_watermark.dct import _get_block_q
        self.assertEqual(_get_block_q(50.0, embed=True), 25.0)
        self.assertEqual(_get_block_q(400.0, embed=True), 50.0)
        self.assertEqual(_get_block_q(2000.0, embed=True), 80.0)
        # 提取侧死区更宽：120 在嵌入侧是 50，在提取侧仍是 25
        self.assertEqual(_get_block_q(120.0, embed=True), 50.0)
        self.assertEqual(_get_block_q(120.0, embed=False), 25.0)


class TestPayload(unittest.TestCase):
    def test_payload_roundtrip_ascii(self):
        from qh_watermark.payload import _encode_payload, _decode_payload
        text = "© Alice 2026"
        self.assertEqual(_decode_payload(_encode_payload(text)), text)

    def test_payload_roundtrip_cjk(self):
        from qh_watermark.payload import _encode_payload, _decode_payload
        text = "版权所有 张三"
        self.assertEqual(_decode_payload(_encode_payload(text)), text)

    def test_decode_no_sync(self):
        from qh_watermark.payload import _decode_payload
        self.assertEqual(_decode_payload([0, 1] * 64), "")

    def test_rs_roundtrip(self):
        """RS 编码能纠正 2 字节错误"""
        from qh_watermark.payload import _encode_payload, _decode_payload
        bits = _encode_payload("hello")
        flipped = bits[:]
        for i in (70, 71, 72):          # 翻一个字节内的 3 个 bit
            flipped[i] ^= 1
        self.assertEqual(_decode_payload(flipped), "hello")


class TestInvisible(unittest.TestCase):
    def _img(self, h=256, w=256, seed=0):
        rng = np.random.default_rng(seed)
        return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)

    def test_invisible_roundtrip(self):
        from qh_watermark.invisible import _embed_invisible, _extract_invisible
        img = self._img()
        out = _embed_invisible(img, "版权所有 2026")
        self.assertEqual(out.shape, img.shape)
        self.assertEqual(_extract_invisible(out), "版权所有 2026")

    def test_invisible_no_watermark(self):
        from qh_watermark.invisible import _extract_invisible
        # 纯噪声图不应解出有效水印
        self.assertEqual(_extract_invisible(self._img(seed=7)), "")

    def test_invisible_psnr(self):
        """嵌入后 PSNR 应高于 30dB（肉眼不可见）"""
        from qh_watermark.invisible import _embed_invisible
        img = self._img(h=512, w=512)
        out = _embed_invisible(img, "版权")
        mse = np.mean((img.astype(np.float64) - out.astype(np.float64)) ** 2)
        psnr = 10 * np.log10(255.0 ** 2 / max(mse, 1e-12))
        self.assertGreater(psnr, 30.0, f"PSNR 过低: {psnr:.2f}dB")

    def test_invisible_capacity_error(self):
        """图像太小应抛 ValueError"""
        from qh_watermark.invisible import _embed_invisible
        with self.assertRaises(ValueError):
            _embed_invisible(self._img(h=64, w=64), "x" * 500)


class TestVisible(unittest.TestCase):
    def _img(self, h=128, w=128):
        return np.full((h, w, 3), 128, dtype=np.uint8)

    def test_visible_text_shape(self):
        from qh_watermark.visible import _apply_visible_watermark
        out = _apply_visible_watermark(
            self._img(), "文字", "测试", 0.5, 40, -30, 24, "#FF0000", None
        )
        self.assertEqual(out.shape, (128, 128, 3))
        self.assertEqual(out.dtype, np.uint8)

    def test_visible_invalid_color_fallback(self):
        """非法颜色串应回退白色且不抛异常"""
        from qh_watermark.visible import _hex_to_rgb
        self.assertEqual(_hex_to_rgb("#GGGGGG"), (255, 255, 255))
        self.assertEqual(_hex_to_rgb("#FF0000"), (255, 0, 0))

    def test_visible_image_wm_missing_fallback(self):
        """图片模式但未接图 → 回退文字模式"""
        from qh_watermark.visible import _apply_visible_watermark
        out = _apply_visible_watermark(
            self._img(), "图片", "回退", 0.5, 40, 0, 24, "#FFFFFF", None
        )
        self.assertEqual(out.shape, (128, 128, 3))


class TestNodes(unittest.TestCase):
    """节点层测试。

    输入、断言、边界值**照搬源测试** `水印/test_watermark.py` 中同名用例，
    只把导入方式从 `from watermark_nodes import X` 改为包内模块导入。
    不要重写输入或收紧边界——源测试的边界是实测出来的。
    """

    def test_node_watermarkadd_text(self):
        import torch
        from qh_watermark.nodes import QianhuiWatermarkAdd
        node = QianhuiWatermarkAdd()
        # ComfyUI IMAGE tensor: (B, H, W, C) float32 0-1
        img = torch.zeros(1, 64, 64, 3)
        original, watermarked = node.apply_watermark(
            image=img, watermark_type="文字", text="TEST © 2026",
            opacity=0.5, spacing=30, rotation=-30, font_size=16, color="#FFFFFF",
        )
        self.assertEqual(tuple(original.shape), (1, 64, 64, 3))
        self.assertEqual(tuple(watermarked.shape), (1, 64, 64, 3))
        # 原图应未被修改
        self.assertTrue(torch.equal(original, img), "original image was modified")

    def test_node_watermarkadd_invisible(self):
        import torch
        from qh_watermark.nodes import QianhuiWatermarkAdd
        node = QianhuiWatermarkAdd()
        rng = np.random.default_rng(7)
        arr = rng.random((1, 512, 512, 3)).astype(np.float32)
        img = torch.from_numpy(arr)
        original, watermarked = node.apply_watermark(
            image=img, watermark_type="隐形文字", text="copyright",
            opacity=0.3, spacing=100, rotation=-30, font_size=36, color="#FFFFFF",
        )
        self.assertFalse(torch.equal(original, watermarked),
                         "invisible watermark had no effect")
        diff = (original - watermarked).abs().max().item()
        # 边界 0.40 照搬源测试；实测约 0.2076
        # （源测试行尾注释里的 "actual max ~0.07" 是失效的实测记录，未照搬）
        self.assertLess(diff, 0.40, f"Invisible watermark caused large pixel diff: {diff}")

    def test_node_watermarkextract_roundtrip(self):
        import torch
        from qh_watermark.nodes import QianhuiWatermarkAdd, QianhuiWatermarkExtract
        add = QianhuiWatermarkAdd()
        extract = QianhuiWatermarkExtract()
        rng = np.random.default_rng(99)
        arr = rng.random((1, 256, 256, 3)).astype(np.float32)
        img = torch.from_numpy(arr)
        text = "hello 水印"
        _orig, watermarked = add.apply_watermark(
            image=img, watermark_type="隐形文字", text=text,
            opacity=0.3, spacing=100, rotation=-30, font_size=36, color="#FFFFFF",
        )
        (extracted_text,) = extract.extract_watermark(image=watermarked)
        self.assertEqual(extracted_text, text)

    def test_node_watermarkextract_no_wm(self):
        import torch
        from qh_watermark.nodes import QianhuiWatermarkExtract
        extract = QianhuiWatermarkExtract()
        img = torch.zeros(1, 128, 128, 3)
        (result,) = extract.extract_watermark(image=img)
        self.assertEqual(result, "")


if __name__ == "__main__":
    unittest.main()
