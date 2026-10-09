# -*- coding: utf-8 -*-
"""LatentAlgorithms 纯函数单测（只依赖 torch/numpy，无需 ComfyUI）。"""
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _loader import load_module, pkgs_dir  # noqa: E402

ALG = load_module(
    "qh_algorithms", os.path.join(pkgs_dir("qianhui_latent_yolo"), "algorithms.py")
)
LA = ALG.LatentAlgorithms


class TestLatentAwareResize(unittest.TestCase):
    def test_shape(self):
        x = torch.randn(1, 4, 16, 16)
        out = LA.latent_aware_resize(x, 32, 32)
        self.assertEqual(tuple(out.shape), (1, 4, 32, 32))

    def test_same_size_is_identity(self):
        x = torch.randn(1, 4, 16, 16)
        out = LA.latent_aware_resize(x, 16, 16)
        self.assertTrue(torch.allclose(out, x, atol=1e-6))


class TestGaussianFeather(unittest.TestCase):
    def test_range_and_shape(self):
        mask = torch.zeros(1, 1, 32, 32)
        mask[:, :, 12:20, 12:20] = 1.0
        out = LA.gaussian_feather(mask, sigma=2.0)
        self.assertEqual(tuple(out.shape), tuple(mask.shape))
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0 + 1e-6)

    def test_center_higher_than_edge(self):
        """中心权重应不低于边缘（羽化是向外衰减）"""
        mask = torch.zeros(1, 1, 32, 32)
        mask[:, :, 14:18, 14:18] = 1.0
        out = LA.gaussian_feather(mask, sigma=2.0)
        self.assertGreaterEqual(float(out[0, 0, 16, 16]), float(out[0, 0, 0, 0]))


class TestBlends(unittest.TestCase):
    def _pair(self, h=8, w=8):
        src = torch.zeros(1, 4, h, w)
        dst = torch.ones(1, 4, h, w)
        return src, dst

    def test_simple_blend_mask_zero_keeps_dst(self):
        src, dst = self._pair()
        mask = torch.zeros(1, 1, 8, 8)
        out = LA.simple_blend(src, dst, mask)
        self.assertTrue(torch.allclose(out, dst, atol=1e-5))

    def test_simple_blend_mask_one_takes_src(self):
        src, dst = self._pair()
        mask = torch.ones(1, 1, 8, 8)
        out = LA.simple_blend(src, dst, mask)
        self.assertTrue(torch.allclose(out, src, atol=1e-5))

    def test_poisson_blend_shape(self):
        src, dst = self._pair()
        mask = torch.zeros(1, 1, 8, 8)
        mask[:, :, 2:6, 2:6] = 1.0
        out = LA.poisson_blend_latent(src, dst, mask, iterations=3)
        self.assertEqual(tuple(out.shape), tuple(dst.shape))

    def test_mpb_blend_shape(self):
        src, dst = self._pair()
        mask = torch.zeros(1, 1, 8, 8)
        mask[:, :, 2:6, 2:6] = 1.0
        out = LA.mpb_blend_latent(src, dst, mask, color_preserve=0.5)
        self.assertEqual(tuple(out.shape), tuple(dst.shape))

    def test_smart_soft_blend_shape(self):
        src, dst = self._pair()
        mask = torch.zeros(1, 1, 8, 8)
        mask[:, :, 2:6, 2:6] = 1.0
        out = LA.smart_soft_blend(src, dst, mask)
        self.assertEqual(tuple(out.shape), tuple(dst.shape))


class TestColorMatch(unittest.TestCase):
    def test_simple_color_match_shape(self):
        src = torch.rand(1, 4, 8, 8)
        ref = torch.rand(1, 4, 8, 8)
        mask = torch.ones(1, 1, 8, 8)
        out = LA.simple_color_match(src, ref, mask)
        self.assertEqual(tuple(out.shape), tuple(src.shape))

    def test_frequency_aware_color_match_shape(self):
        src = torch.rand(1, 4, 8, 8)
        ref = torch.rand(1, 4, 8, 8)
        mask = torch.ones(1, 1, 8, 8)
        out = LA.frequency_aware_color_match(src, ref, mask, strength=0.5)
        self.assertEqual(tuple(out.shape), tuple(src.shape))


class TestPackageRelativeImports(unittest.TestCase):
    """守卫：包内相对导入的目标必须真实存在。

    背景：`mixins/pipeline/` 比 `mixins/` 深一层，写成 `from ..algorithms` 会解析到
    不存在的 `qianhui_latent_yolo.mixins.algorithms`。这类错误
    ast 语法检查 / 自由变量扫描 / 字段闭合检查**全都看不见**（它们只看语法），
    而现有测试没有一个 import node.py —— 于是表现为「零告警的全量失效」：
    整包导入抛 ModuleNotFoundError，节点从 ComfyUI 菜单里消失。
    """

    def test_relative_import_targets_exist(self):
        import ast
        import importlib.util

        root = pkgs_dir("qianhui_latent_yolo")
        pkg_parent = os.path.dirname(root)          # .../nodes
        broken = []
        for dirpath, _dirs, files in os.walk(root):
            pkg = os.path.relpath(dirpath, pkg_parent).replace(os.sep, ".")
            for fn in sorted(files):
                if not fn.endswith(".py"):
                    continue
                path = os.path.join(dirpath, fn)
                with open(path, encoding="utf-8") as fh:
                    tree = ast.parse(fh.read())
                for n in ast.walk(tree):
                    if not isinstance(n, ast.ImportFrom) or not n.level:
                        continue
                    rel = "." * n.level + (n.module or "")
                    where = "%s:%d" % (
                        os.path.relpath(path, pkg_parent).replace(os.sep, "/"), n.lineno)
                    try:
                        target = importlib.util.resolve_name(rel, pkg)
                    except ImportError as exc:      # 点太多，超出顶层包
                        broken.append("%s  from %s import ...  ->  %s" % (where, rel, exc))
                        continue
                    fs = os.path.join(pkg_parent, *target.split("."))
                    if not (os.path.isfile(fs + ".py") or os.path.isdir(fs)):
                        broken.append("%s  from %s import ...  ->  %s 不存在"
                                      % (where, rel, target))
        self.assertEqual(
            broken, [], "相对导入目标不存在（导入即失败，节点不会注册）:\n" + "\n".join(broken)
        )


if __name__ == "__main__":
    unittest.main()
