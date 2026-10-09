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
    """守卫：包内相对导入的**目标模块**与**被导入的名字**都必须真实存在。

    背景：`mixins/pipeline/` 比 `mixins/` 深一层，写成 `from ..algorithms` 会解析到
    不存在的 `qianhui_latent_yolo.mixins.algorithms`。这类错误
    ast 语法检查 / 自由变量扫描 / 字段闭合检查**全都看不见**（它们只看语法），
    而现有测试没有一个 import node.py —— 于是表现为「零告警的全量失效」：
    整包导入抛 ModuleNotFoundError，节点从 ComfyUI 菜单里消失。

    三条判据，全部纯静态（不 import 目标，故不需要 torch / ComfyUI）：
      1. 目标模块 / 包真实存在；
      2. `from M import N`：N 是 M 的顶层绑定（M 是包时 N 也可为其子模块）；
      3. `from .. import N`（module-less）：N 是上一级包的子模块或 `__init__` 顶层绑定。

    层级算术以**插件根**为顶层：运行期插件包是 `<plugin>.nodes.qianhui_latent_yolo`，
    `nodes/` 是 PEP 420 命名空间包，故 `....foo`（= `nodes.foo`）合法、不该报红；
    只有越过插件根才判红。
    """

    @staticmethod
    def _module_names(tree):
        """收集模块的顶层绑定名。返回 (names, opaque)。

        opaque=True 表示静态集合不完备（顶层有 `from X import *`，或定义了 PEP 562 的
        `__getattr__`）—— 这类目标放弃名字校验，宁可漏报也不误报。
        不进入函数/类内部作用域：那些不是模块级绑定。
        """
        import ast

        names = set()
        opaque = False

        def bind_target(t):
            if isinstance(t, ast.Name):
                names.add(t.id)
            elif isinstance(t, (ast.Tuple, ast.List)):
                for e in t.elts:
                    bind_target(e)
            elif isinstance(t, ast.Starred):
                bind_target(t.value)

        def walk_body(body):
            nonlocal opaque
            for st in body:
                if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    names.add(st.name)
                    if st.name == "__getattr__":
                        opaque = True
                elif isinstance(st, ast.Assign):
                    for t in st.targets:
                        bind_target(t)
                elif isinstance(st, ast.AnnAssign):
                    bind_target(st.target)
                elif isinstance(st, ast.Import):
                    for a in st.names:
                        names.add(a.asname or a.name.split(".")[0])
                elif isinstance(st, ast.ImportFrom):
                    for a in st.names:
                        if a.name == "*":
                            opaque = True
                        else:
                            names.add(a.asname or a.name)
                elif isinstance(st, (ast.For, ast.AsyncFor)):
                    bind_target(st.target)
                    walk_body(st.body)
                    walk_body(st.orelse)
                elif isinstance(st, (ast.If, ast.While)):
                    walk_body(st.body)
                    walk_body(st.orelse)
                elif isinstance(st, (ast.With, ast.AsyncWith)):
                    for it in st.items:
                        if it.optional_vars is not None:
                            bind_target(it.optional_vars)
                    walk_body(st.body)
                elif isinstance(st, ast.Try):
                    walk_body(st.body)
                    walk_body(st.orelse)
                    walk_body(st.finalbody)
                    for h in st.handlers:
                        if h.name:
                            names.add(h.name)
                        walk_body(h.body)

        walk_body(tree.body)
        return names, opaque

    @classmethod
    def _name_ok(cls, target_fs, is_pkg, name):
        """名字能否在目标处解析：True / False / None(=无法静态判定，跳过)。"""
        import ast

        if is_pkg:
            if (os.path.isfile(os.path.join(target_fs, name + ".py"))
                    or os.path.isdir(os.path.join(target_fs, name))):
                return True                                  # 子模块
            init_py = os.path.join(target_fs, "__init__.py")
            if not os.path.isfile(init_py):
                return False
            with open(init_py, encoding="utf-8") as fh:
                names, opaque = cls._module_names(ast.parse(fh.read()))
        else:
            with open(target_fs + ".py", encoding="utf-8") as fh:
                names, opaque = cls._module_names(ast.parse(fh.read()))
        return None if opaque else (name in names)

    @classmethod
    def _scan(cls, root, plugin_root):
        """扫描 root 下所有 .py 的相对导入，返回问题清单（空 = 全部合法）。"""
        import ast
        import importlib.util

        broken = []
        for dirpath, _dirs, files in os.walk(root):
            pkg = os.path.relpath(dirpath, plugin_root).replace(os.sep, ".")
            for fn in sorted(files):
                if not fn.endswith(".py"):
                    continue
                path = os.path.join(dirpath, fn)
                with open(path, encoding="utf-8") as fh:
                    tree = ast.parse(fh.read())
                where_file = os.path.relpath(path, plugin_root).replace(os.sep, "/")
                for n in ast.walk(tree):
                    if not isinstance(n, ast.ImportFrom) or not n.level:
                        continue
                    where = "%s:%d" % (where_file, n.lineno)
                    rel = "." * n.level + (n.module or "")
                    try:
                        target = importlib.util.resolve_name(rel, pkg)
                    except ImportError as exc:               # 越过插件根
                        broken.append("%s  from %s import ...  ->  %s" % (where, rel, exc))
                        continue
                    fs = os.path.join(plugin_root, *target.split("."))
                    is_pkg = os.path.isdir(fs)
                    if not (os.path.isfile(fs + ".py") or is_pkg):
                        broken.append("%s  from %s import ...  ->  %s 不存在"
                                      % (where, rel, target))
                        continue
                    for a in n.names:
                        if a.name == "*":                    # `import *` 无名字可查
                            continue
                        if cls._name_ok(fs, is_pkg, a.name) is False:
                            broken.append("%s  from %s import %s  ->  %s 里没有这个名字"
                                          % (where, rel, a.name, target))
        return broken

    def test_relative_import_targets_exist(self):
        nodes_dir = pkgs_dir()                       # <plugin>/nodes
        plugin_root = os.path.dirname(nodes_dir)     # <plugin>
        root = os.path.join(nodes_dir, "qianhui_latent_yolo")
        broken = self._scan(root, plugin_root)
        self.assertEqual(
            broken, [],
            "相对导入有 %d 处非法（导入期即失败，节点不会注册）:\n%s"
            % (len(broken), "\n".join(broken)),
        )


if __name__ == "__main__":
    unittest.main()
