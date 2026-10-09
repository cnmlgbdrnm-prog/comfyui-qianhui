# -*- coding: utf-8 -*-
"""LatentAlgorithms 纯函数单测（只依赖 torch/numpy，无需 ComfyUI）。"""
import ast
import io
import os
import re
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
        # 扫描**整个 nodes/**，不是只扫 qianhui_latent_yolo：
        # qianhui_watermark 是第二个包，只扫前者会让它完全没有导入目标守卫。
        # _scan 已按文件位置自算包路径，故换根即可一并覆盖所有既有节点文件。
        nodes_dir = pkgs_dir()                       # <plugin>/nodes
        plugin_root = os.path.dirname(nodes_dir)     # <plugin>
        root = nodes_dir
        broken = self._scan(root, plugin_root)
        self.assertEqual(
            broken, [],
            "相对导入有 %d 处非法（导入期即失败，节点不会注册）:\n%s"
            % (len(broken), "\n".join(broken)),
        )


# ============================ node.py 控件声明（ast 解析，不 import） ============================
#
# 为什么不 import node.py：它 `import comfy.samplers`，还要 ComfyUI 运行时；
# 更要命的是把 presets/node.py 变成**模块级硬依赖**——那正是 commit da27cc3
# 刻意去掉的失败模式（presets.py 出语法错就整包崩、节点从菜单消失）。
# 静态解析则天然免疫：node.py 语法错 → 这里直接报语法错，不会连累别的用例。

def _dict_node_value(d, key):
    """取 ast.Dict 里某个字面量键对应的值节点。"""
    for k, v in zip(d.keys, d.values):
        if isinstance(k, ast.Constant) and k.value == key:
            return v
    raise KeyError(key)


_MISSING = object()          # 「不是字面量」的哨兵，与「键根本不存在」区分开


def _const_value(node):
    """字面量常量（含一元负号）→ 值；否则抛。"""
    if isinstance(node, ast.Constant):
        return node.value
    if (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub)
            and isinstance(node.operand, ast.Constant)):
        return -node.operand.value
    raise ValueError("不是字面量常量: " + ast.dump(node))


def _try_const(node):
    """字面量 → 值；非字面量 → _MISSING（不抛）。

    node.py 里有合法的非字面量声明（`YOLO模型` 的 default 是条件表达式、
    `SAM模型选择` 的选项来自 SAMModelManager.get_model_list()）。
    一律抛会让这两项把整条用例带崩；一律当默认 0 又会造假绿灯。
    故显式返回 _MISSING，由调用方决定「不可静态校验」怎么记。
    """
    try:
        return _const_value(node)
    except ValueError:
        return _MISSING


def _widget_spec(decl):
    """把 INPUT_TYPES 里一项的声明解析成 (kind, spec)。

    spec 含 default / min / max / step（字面量才有）；
    `_dynamic` 列出**声明了但非字面量**的元数据名（供监管用例拒绝静默跳过）；
    COMBO 另带 options（选项非全字面量时降级为 DYNAMIC）。
    """
    if not (isinstance(decl, ast.Tuple) and decl.elts):
        raise ValueError("控件声明不是元组: " + ast.dump(decl))
    type_node = decl.elts[0]
    spec = {}
    dynamic = set()
    if len(decl.elts) > 1 and isinstance(decl.elts[1], ast.Dict):
        for k, v in zip(decl.elts[1].keys, decl.elts[1].values):
            key = _try_const(k)
            if key is _MISSING:
                continue
            if key in ("default", "min", "max", "step"):
                val = _try_const(v)
                if val is _MISSING:
                    dynamic.add(key)
                else:
                    spec[key] = val
    if dynamic:
        spec["_dynamic"] = sorted(dynamic)
    if isinstance(type_node, ast.List):
        opts = [_try_const(e) for e in type_node.elts]
        if any(o is _MISSING for o in opts):
            return "DYNAMIC", spec
        spec["options"] = opts
        return "COMBO", spec
    if isinstance(type_node, ast.Constant) and isinstance(type_node.value, str):
        return type_node.value, spec
    return "DYNAMIC", spec


def _latent_node_widget_decls():
    """{控件名: (kind, spec)}，合并 node.py INPUT_TYPES 的 required 与 optional。

    定位方式：找那个**同时有 required 与 optional 两个字面量键**的 dict
    （即 INPUT_TYPES 的返回体；配置预设 是它 optional 子 dict 的键，不是它自己的键），
    再合并两个子 dict。不依赖行号，抗重排。
    """
    path = os.path.join(pkgs_dir("qianhui_latent_yolo"), "node.py")
    with io.open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())

    def keys_of(d):
        return {k.value for k in d.keys if isinstance(k, ast.Constant)}

    root = None
    for n in ast.walk(tree):
        if isinstance(n, ast.Dict) and {"required", "optional"} <= keys_of(n):
            root = n
            break
    assert root is not None, "node.py 里找不到同时含 required/optional 的 INPUT_TYPES 字面量"

    decls = {}
    for section in ("required", "optional"):
        sub = _dict_node_value(root, section)
        for k, v in zip(sub.keys, sub.values):
            decls[_const_value(k)] = _widget_spec(v)
    assert "配置预设" in decls, "解析出的控件表里没有 配置预设 —— 锚点可能找错了 dict"
    return decls


# context.py 里每个字段旁的机读注释，形如：  cache_device_mode: str = '自动'   # 键: 结果缓存设备
_KEY_COMMENT_RE = re.compile(r"#\s*键:\s*(\S+)\s*$")


def _context_key_map():
    """从 context.py 的 `# 键: <控件名>` 注释建 {字段名: 控件名}。

    这是**双向**映射的廉价来源：它同时知道「这是哪个控件」和「这个控件对应哪个字段」，
    因此能一次钉住 FIELD_OF 的两侧，而不必解析 node.py。
    """
    path = os.path.join(pkgs_dir("qianhui_latent_yolo"), "mixins", "context.py")
    with io.open(path, encoding="utf-8") as fh:
        src = fh.read()
    lines = src.splitlines()
    tree = ast.parse(src)

    cls = None
    for n in ast.walk(tree):
        if isinstance(n, ast.ClassDef) and n.name == "YoloConfig":
            cls = n
            break
    assert cls is not None, "context.py 里找不到 YoloConfig"

    out = {}
    for st in cls.body:
        if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name):
            m = _KEY_COMMENT_RE.search(lines[st.lineno - 1])
            if m:
                out[st.target.id] = m.group(1)
    return out


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
        """cfg 缺字段时只覆盖存在的键，不抛异常——且**不能凭空造出新属性**。

        ★ 断言 3 在 `YoloConfig` 加上 `slots=True` 之后换了写法，理由说清楚：
        原来它用 `set(vars(cfg))` 前后取快照比集合，只能证明「**这一次**调用没造
        出新属性」。`slots=True` 把这件事从「事后抽查」变成「解释器强制」——
        类里没有 `__dict__`，写任何未声明字段直接 `AttributeError`。
        所以现在直接验证**那道闸门本身**：既证明能力没退化，
        又覆盖到别的调用点（`vars()` 也正好在 slots 类上会 `TypeError`，
        不换写法这条用例会直接红，而不是静默失去意义）。
        """
        pres = self._pres()
        cfg = self._cfg()
        pres.apply_preset(cfg, "快速模式(低质量)")
        # 断言 1：不抛异常（走到这里即成立）
        # 断言 2：真属性被改了（用真 cfg 才有意义）
        self.assertEqual(cfg.region_steps, 12)
        # 断言 3：容器确实带 slots —— 写未声明属性会当场 AttributeError，
        #         「凭空造出新属性」在解释器层已不可能。
        self.assertFalse(
            hasattr(cfg, "__dict__"),
            "YoloConfig 应带 @dataclass(slots=True)；一旦有了 __dict__，"
            "写错字段名（region_step vs region_steps）就会静默新建属性、真字段不动",
        )
        with self.assertRaises(AttributeError):
            cfg.definitely_not_a_field = 1

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

    def test_preset_combo_matches_preset_labels(self):
        """★ 下拉框的选项表必须逐字等于 `presets.PRESET_LABELS`，**含顺序**。

        来源有三处而只有一处是权威：
          - `presets.py:90` 定义 PRESET_LABELS；
          - `presets.py:117` 由 as_payload() 以 "labels" 发出；
          - `node.py:97-105` **手抄**了同一份 7 项列表（含 `"[不使用预设]"` 字面量）。
        今天三处一致（已核），所以这条用例现在是绿的；它的价值在于：
        以后加第 6 档时只改 presets.py 而忘了 node.py，**那一档在界面上根本不存在**，
        但整批测试仍然是绿的——除非有人把它钉住。
        """
        pres = self._pres()
        kind, spec = _latent_node_widget_decls()["配置预设"]
        self.assertEqual(kind, "COMBO", f"配置预设 应是下拉框，实为 {kind}")
        self.assertEqual(
            spec["options"], pres.PRESET_LABELS,
            "node.py 的档位列表与 presets.PRESET_LABELS 不一致（含顺序）：\n"
            f"  node.py : {spec['options']}\n  presets : {pres.PRESET_LABELS}",
        )
        self.assertEqual(
            spec.get("default"), pres.NO_PRESET,
            f"默认档位应为 NO_PRESET({pres.NO_PRESET!r})，实为 {spec.get('default')!r}",
        )


class TestFieldOfBridge(unittest.TestCase):
    """`FIELD_OF`（控件名 → YoloConfig 字段名）两侧都必须钉死。

    今天只有「字段名那一侧」被 `test_every_preset_key_maps_to_a_real_config_field` 守着。
    控件名那一侧完全裸奔：把 `"多尺度检测"` 写成 `"多尺度检測"`（异体字/错字），
    每个测试照样绿，而那个参数**永远不再联动**——用户选了预设、控件看着没变、
    后端也没改，全程零告警。
    """

    def _pres(self):
        return load_module(
            "qh_presets",
            os.path.join(pkgs_dir("qianhui_latent_yolo"), "presets.py"),
        )

    @staticmethod
    def _covered_widgets(pres):
        """预设实际覆盖的控件名 = 各档表键的并集（不用 PRESET_LABELS，那是含哨兵的档位名）。"""
        covered = set()
        for table in pres.PRESETS.values():
            covered |= set(table)
        return covered

    def test_field_of_keys_are_real_node_widgets(self):
        """方向一（直接）：FIELD_OF 的每个键都必须是 node.py 里真实存在的控件名。

        这是那句「写一个 node.py 里不存在的控件名，所有测试仍绿」的正面闭合。
        """
        pres = self._pres()
        decls = _latent_node_widget_decls()
        missing = sorted(set(pres.FIELD_OF) - set(decls))
        self.assertEqual(
            missing, [],
            f"FIELD_OF 里有 {len(missing)} 个控件名在 node.py 的 INPUT_TYPES 里不存在：{missing}\n"
            "这些参数会静默失去联动（预设改了、控件不动）。",
        )

    def test_field_of_matches_context_key_comments(self):
        """方向二（配对）：用 context.py 的 `# 键: <控件名>` 注释反查 FIELD_OF。

        注释是逐字段写在**声明旁边**的，所以它同时编码了配对关系；
        取「注释里提到的、确实属于预设覆盖集」的那些，反转后应与 FIELD_OF 完全相等。
        这能抓住「控件名对、但指向了错误的字段」——方向一那种集合比较看不见。
        """
        pres = self._pres()
        covered = self._covered_widgets(pres)
        comment_map = _context_key_map()            # {字段名: 控件名}
        derived = {widget: field for field, widget in comment_map.items()
                   if widget in covered}
        self.assertEqual(
            derived, dict(pres.FIELD_OF),
            "context.py 的 `# 键:` 注释与 presets.FIELD_OF 对不上：\n"
            f"  只在 FIELD_OF  : {sorted(set(pres.FIELD_OF) - set(derived))}\n"
            f"  只在注释       : {sorted(set(derived) - set(pres.FIELD_OF))}\n"
            f"  指向不同字段   : "
            f"{sorted(w for w in set(derived) & set(pres.FIELD_OF) if derived[w] != pres.FIELD_OF[w])}",
        )


class TestPresetValuesAreLegal(unittest.TestCase):
    """每个预设**值**都必须是它那个控件合法取值：类型、min/max、步进格点、下拉成员。

    控制器机检过 19 控件 × 5 档 = 95 个值，当时零违例，所以本组建立时应全绿；
    它的作用是把「以后往表里填一个 0.63 或一个越界值」变成红灯，而不是
    在用户机器上变成 ComfyUI 静默把值夹到边界（用户看到的效果和预设表写的不一样）。
    """

    def _pres(self):
        return load_module(
            "qh_presets",
            os.path.join(pkgs_dir("qianhui_latent_yolo"), "presets.py"),
        )

    @staticmethod
    def _violation(widget, value, kind, spec):
        """返回违规说明；合法则返回 None。"""
        if kind == "DYNAMIC":
            return (f"控件 {widget!r} 的声明含运行期表达式，无法静态校验其取值 "
                    f"{value!r}——请人工确认，或把该控件移出预设表")
        dynamic = set(spec.get("_dynamic", ()))
        if kind in ("INT", "FLOAT") and dynamic & {"min", "max", "step"}:
            return (f"控件 {widget!r} 的 {sorted(dynamic & {'min', 'max', 'step'})} 不是字面量，"
                    f"范围/步进无法静态校验（值 {value!r}）——别让它静默跳过检查")
        if kind == "BOOLEAN":
            return None if isinstance(value, bool) else f"应为 BOOLEAN，实为 {value!r}"
        if kind == "COMBO":
            return None if value in spec.get("options", []) else \
                f"不在下拉选项里：{value!r} ∉ {spec.get('options')}"
        if kind == "STRING":
            return None if isinstance(value, str) else f"应为 STRING，实为 {value!r}"

        # INT / FLOAT
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return f"应为 {kind}，实为 {value!r}"
        if kind == "INT" and not isinstance(value, int):
            return f"应为 INT，实为浮点 {value!r}"
        lo, hi = spec.get("min"), spec.get("max")
        if lo is not None and value < lo:
            return f"{value!r} < min {lo!r}"
        if hi is not None and value > hi:
            return f"{value!r} > max {hi!r}"
        step = spec.get("step")
        if step:
            base = lo if lo is not None else 0
            n = (value - base) / step
            if abs(n - round(n)) > 1e-6:
                return f"{value!r} 不在步进格点上（min={lo!r}, step={step!r}）"
        return None

    def test_all_preset_values_legal(self):
        pres = self._pres()
        decls = _latent_node_widget_decls()
        bad = []
        checked = 0
        for tier, table in pres.PRESETS.items():
            for widget, value in table.items():
                self.assertIn(widget, decls, f"预设「{tier}」覆盖了 node.py 里没有的控件 {widget!r}")
                kind, spec = decls[widget]
                checked += 1
                why = self._violation(widget, value, kind, spec)
                if why:
                    bad.append(f"预设「{tier}」· {widget}: {why}")
        self.assertGreater(checked, 0, "一个值都没检查到，用例是假的")
        self.assertEqual(bad, [], "预设值非法 %d 处:\n%s" % (len(bad), "\n".join(bad)))

    # 注：本可以再加一条「覆盖集 ⊆ FIELD_OF」，但
    # test_every_preset_key_maps_to_a_real_config_field 已经逐键 assertIn 了，
    # 再加一条只是重复，徒增噪音。


if __name__ == "__main__":
    unittest.main()
