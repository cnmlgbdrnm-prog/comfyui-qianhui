# -*- coding: utf-8 -*-
"""LatentAlgorithms 纯函数单测（只依赖 torch/numpy，无需 ComfyUI）。"""
import ast
import io
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


class TestPresets(unittest.TestCase):
    def _pres(self):
        return load_module(
            "qh_presets",
            os.path.join(pkgs_dir("qianhui_latent_yolo"), "presets.py"),
        )

    def _cfg(self):
        """取**真实**的 YoloConfig 实例——不要手搓假对象，理由见上方说明。"""
        ctx = load_module(
            "qh_ctx",
            os.path.join(pkgs_dir("qianhui_latent_yolo"), "mixins", "context.py"),
        )
        return ctx.YoloConfig()

    def test_five_presets_defined(self):
        pres = self._pres()
        self.assertEqual(
            sorted(pres.PRESETS.keys()),
            sorted(["快速模式(低质量)", "平衡模式(推荐)", "质量模式(慢速)",
                    "极致质量(超慢)", "省显存模式"]),
        )

    def test_no_preset_is_noop(self):
        pres = self._pres()
        cfg = self._cfg()
        cfg.region_steps = 999
        pres.apply_preset(cfg, pres.NO_PRESET)
        self.assertEqual(cfg.region_steps, 999)
        pres.apply_preset(cfg, pres.CUSTOM)
        self.assertEqual(cfg.region_steps, 999)

    def test_fast_preset_overrides(self):
        pres = self._pres()
        cfg = self._cfg()
        pres.apply_preset(cfg, "快速模式(低质量)")
        self.assertEqual(cfg.region_steps, 12)
        self.assertEqual(cfg.fix_rounds, 1)
        self.assertFalse(cfg.multi_scale)

    def test_extreme_is_heaviest(self):
        pres = self._pres()
        fast = pres.PRESETS["快速模式(低质量)"]
        extreme = pres.PRESETS["极致质量(超慢)"]
        self.assertLess(fast["区域步数"], extreme["区域步数"])
        self.assertLessEqual(fast["修复轮次"], extreme["修复轮次"])

    def test_missing_field_does_not_raise(self):
        """cfg 缺字段时只覆盖存在的键，不抛异常——且**不能凭空造出新属性**。"""
        pres = self._pres()
        cfg = self._cfg()
        before = set(vars(cfg))
        pres.apply_preset(cfg, "快速模式(低质量)")
        # 断言 1：不抛异常（走到这里即成立）
        # 断言 2：真属性被改了（用真 cfg 才有意义）
        self.assertEqual(cfg.region_steps, 12)
        # 断言 3：没有多出预设表之外的属性
        self.assertTrue(
            set(vars(cfg)) <= before | set(pres.PRESETS["快速模式(低质量)"]),
            "apply_preset 造出了配置类里本不存在的属性",
        )

    def test_every_preset_key_maps_to_a_real_config_field(self):
        """★ 本组最要紧的一条。

        预设表用的是**控件名**（前端 `x.name === key` 靠它匹配 widgets），
        而 `apply_preset` 要写的是 **YoloConfig 的字段名**（英文局部名）。
        两者不是一个键空间。没有这条测试，`FIELD_OF` 里漏一个键
        就会让那个参数**静默失效**：用户选了预设，控件看着变了，但配置没变。
        """
        from dataclasses import fields
        pres = self._pres()
        real = {f.name for f in fields(self._cfg())}
        for name, table in pres.PRESETS.items():
            for k in table:
                self.assertIn(k, pres.FIELD_OF, f"预设「{name}」的键 {k} 没有字段映射")
                self.assertIn(
                    pres.FIELD_OF[k], real,
                    f"预设「{name}」的键 {k} 映射到不存在的字段 {pres.FIELD_OF[k]}",
                )
        # 反向：FIELD_OF 里的每个字段也必须是真字段（防拼错）
        for widget_key, field_name in pres.FIELD_OF.items():
            self.assertIn(field_name, real, f"FIELD_OF['{widget_key}'] 指向不存在的字段")

    def test_parse_config_mounts_preset_from_kwargs_not_cfg(self):
        """★ 选择路径：预设名必须从 `kwargs` 取，**不能**从 `cfg` 取。

        为什么需要这条：`配置预设` 是控件（`node.py:97`），但 `process()` **从不读它**，
        所以它**不在** `YoloConfig` 的 99 个字段里 —— 实测 `配置预设` 在
        `mixins/context.py` 里出现 **0** 次。于是
        `getattr(cfg, "配置预设", None)` **恒为 None**，`or "[不使用预设]"` 恒取到
        `NO_PRESET`，`apply_preset` 第一句就 `return`：面板能用、切了没反应、**不报错**。

        上面那几条用例全都**直接**调 `apply_preset(cfg, name)`，跳过了这次查找，
        所以一条都抓不到它。这条必须静态钉住调用点的实参形态。
        """
        src = io.open(
            os.path.join(pkgs_dir("qianhui_latent_yolo"),
                         "mixins", "pipeline", "prepare.py"),
            encoding="utf-8",
        ).read()
        calls = [
            n for n in ast.walk(ast.parse(src))
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name) and n.func.id == "apply_preset"
        ]
        self.assertEqual(
            len(calls), 1,
            f"prepare.py 里 apply_preset 的调用点应为 1 个，实为 {len(calls)}",
        )
        arg = ast.unparse(calls[0].args[1])
        self.assertIn("kwargs", arg, f"预设名没从 kwargs 取：{arg}")
        self.assertNotIn("getattr", arg,
                         f"预设名绕道 cfg 取了，而 cfg 里没有这个字段、恒为 None：{arg}")


if __name__ == "__main__":
    unittest.main()
