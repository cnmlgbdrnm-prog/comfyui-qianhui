# -*- coding: utf-8 -*-
"""执行级导入冒烟：在桩掉 ComfyUI 运行时的前提下，**真正 exec** 各模块的模块体。

## 为什么必须有这一条

`test_latent_algorithms.py::test_relative_import_targets_exist` 是**静态**守卫：
它只检查「相对导入的目标模块存在、被导入的名字在目标里存在」。
它对下面三类破损**结构性失明**（评审用变异测试实证过，控制器独立复现过）：

1. **导入目标在导入期就抛异常**的模块级代码（例如某个 stage 模块顶层 `raise`）；
2. **循环导入**——两个模块、两个名字都真实存在，静态判据一条都不违反；
3. `if TYPE_CHECKING:` 里的绑定——静态看名字在，运行期根本不存在。

三类的共同症状都一样：整包 import 失败 → ComfyUI 里节点**静默消失**。
这正是本计划的一个 Critical（四个相对导入少了一个点）能穿过全部既有测试的原因：
**没有任何测试 import 过 node.py**。静态守卫与执行级冒烟是互补的，两条都留着。

## 覆盖范围与**已知盲区**（不要高估这条测试）

`mkpkg("nodes", <plugin>/nodes)` 把 `nodes` 这个名字指向**插件自己的** nodes 目录。
真实 ComfyUI 里 `import nodes` 解析到的是 ComfyUI 根目录的 `nodes.py`。
因此这条测试**看不见**「裸 `import nodes` 在真实环境会解析到哪里」这类问题——
它只保证插件内部的模块体在桩环境下能执行完。别把它当成全量导入验证。

## 与 test_watermark.py 的分工

`test_watermark.py:39-44` **故意**只把 qianhui_watermark 注册进 sys.modules 而不执行它的
`__init__.py`，好让 13 个纯 numpy 算法用例不被 `import torch` 拖累（分层是刻意的，别动它）。
代价是 `__init__.py` 从没被任何东西执行过。这条测试补上那一格，**不改**前者的设计。
"""
import ast
import importlib.util
import os
import sys
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _loader import ROOT  # noqa: E402

PLUGIN_NODES = os.path.join(ROOT, "nodes")
STAGES = ("prepare", "detect", "repair", "global_pass", "report")

# 记录装载前的 sys.modules 快照，测试结束后还原，避免污染同批次的其它测试。
_SNAPSHOT = None
_INJECTED = set()


class _Stub(types.ModuleType):
    """任意属性都返回「调用即 None」的假件：够模块体执行完，不触发真逻辑。"""

    def __getattr__(self, k):
        if k.startswith("__"):
            raise AttributeError(k)
        return lambda *a, **kw: None


def _mkpkg(name, path):
    """注册成包但**不执行** __init__.py，避免触发整条插件加载链。"""
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(path, "__init__.py"), submodule_search_locations=[path]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    _INJECTED.add(name)
    return mod


def _exec(name, path):
    """真正执行模块体 —— 这正本条测试的全部意义。"""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    _INJECTED.add(name)
    spec.loader.exec_module(mod)
    return mod


def setUpModule():
    global _SNAPSHOT
    _SNAPSHOT = set(sys.modules)
    for n in ("folder_paths", "comfy", "comfy.model_management", "comfy.utils",
              "comfy.samplers", "comfy.sample", "comfy.controlnet"):
        if n not in sys.modules:
            sys.modules[n] = _Stub(n)
            _INJECTED.add(n)

    _mkpkg("nodes", PLUGIN_NODES)
    _mkpkg("nodes.qianhui_latent_yolo", os.path.join(PLUGIN_NODES, "qianhui_latent_yolo"))
    _mkpkg("nodes.qianhui_latent_yolo.mixins",
           os.path.join(PLUGIN_NODES, "qianhui_latent_yolo", "mixins"))
    _mkpkg("nodes.qianhui_latent_yolo.mixins.pipeline",
           os.path.join(PLUGIN_NODES, "qianhui_latent_yolo", "mixins", "pipeline"))
    _mkpkg("nodes.qianhui_watermark", os.path.join(PLUGIN_NODES, "qianhui_watermark"))


def tearDownModule():
    for n in _INJECTED:
        if _SNAPSHOT is not None and n in _SNAPSHOT:
            continue                      # 装载前就在的，别动
        sys.modules.pop(n, None)


class TestPipelineStageImports(unittest.TestCase):
    """5 个阶段模块的模块体必须能执行完。"""

    def test_stage_modules_execute(self):
        pkg = os.path.join(PLUGIN_NODES, "qianhui_latent_yolo", "mixins", "pipeline")
        failed = []
        for mod in STAGES:
            full = "nodes.qianhui_latent_yolo.mixins.pipeline." + mod
            try:
                _exec(full, os.path.join(pkg, mod + ".py"))
            except Exception as e:                    # noqa: BLE001 - 报出真实异常类型
                failed.append("%s.py: %s: %s" % (mod, type(e).__name__, e))
        self.assertEqual(
            failed, [],
            "阶段模块导入期失败 %d 个（节点会在 ComfyUI 里静默消失）:\n%s"
            % (len(failed), "\n".join(failed)),
        )

    def test_stage_modules_expose_their_entrypoint(self):
        """光「没抛异常」不算数：每个阶段模块得真的导出 prepare.py 调用的那个函数。"""
        pkg = os.path.join(PLUGIN_NODES, "qianhui_latent_yolo", "mixins", "pipeline")
        expected = {
            "prepare": "prepare_run",
            "detect": "run_detection",
            "repair": "run_repair",
            "global_pass": "run_global_pass",
            "report": "build_report",
        }
        for mod, fn in expected.items():
            m = _exec("nodes.qianhui_latent_yolo.mixins.pipeline." + mod,
                      os.path.join(pkg, mod + ".py"))
            self.assertTrue(
                callable(getattr(m, fn, None)),
                "%s.py 里没有可调用的 %s（node.py 依赖它）" % (mod, fn),
            )


class TestNodeModuleImport(unittest.TestCase):
    """node.py 的模块体必须能执行完，且聚合出的类与文档一致。"""

    DOCUMENTED_MRO = ["QianhuiLatentYOLO", "DetectionMixin", "SamMixin", "GeometryMixin",
                      "MasksMixin", "PolicyMixin", "InpaintMixin", "LatentKSamplerUpscaler",
                      "object"]

    def _load_node(self):
        return _exec("nodes.qianhui_latent_yolo.node",
                     os.path.join(PLUGIN_NODES, "qianhui_latent_yolo", "node.py"))

    def test_node_module_executes_and_resolves_class(self):
        m = self._load_node()
        cls = getattr(m, "QianhuiLatentYOLO", None)
        self.assertIsNotNone(cls, "node.py 没导出 QianhuiLatentYOLO")
        self.assertTrue(isinstance(cls, type))

    def test_mro_is_the_documented_one(self):
        """MRO 就是 node.py docstring 里写死的那条：基类必须末位，否则 upscale_latent 解析不到。"""
        cls = self._load_node().QianhuiLatentYOLO
        got = [c.__name__ for c in cls.__mro__]
        self.assertEqual(got, self.DOCUMENTED_MRO, "MRO 与 node.py docstring 不符")

    def test_node_contract_surfaces(self):
        """执行完还得真的可用：INPUT_TYPES / FUNCTION / RETURN_TYPES 齐备。"""
        cls = self._load_node().QianhuiLatentYOLO
        self.assertTrue(callable(getattr(cls, "INPUT_TYPES", None)))
        self.assertEqual(cls.FUNCTION, "process")
        self.assertTrue(hasattr(cls, "process"), "FUNCTION 指向的 process 不存在")

    def test_node_module_survives_without_optional_comfy_surface(self):
        """node.py 只该硬依赖 comfy.samplers。

        静态守卫看不见「多删一个 import 就炸」这类**新增**依赖；这条把它们钉住：
        把 comfy.sample / comfy.utils / comfy.model_management / comfy.controlnet
        从桩里摘掉后，node.py 仍须能执行完（各 mixin 自己会 import 它们，
        但那些模块此刻已在 sys.modules 里，不受影响）。
        """
        saved = {n: sys.modules.pop(n) for n in
                 ("comfy.sample", "comfy.utils", "comfy.model_management", "comfy.controlnet")
                 if n in sys.modules}
        try:
            m = self._load_node()
            self.assertTrue(hasattr(m, "QianhuiLatentYOLO"))
        finally:
            sys.modules.update(saved)


class TestWatermarkPackageImport(unittest.TestCase):
    """qianhui_watermark 的 __init__.py 此前**没有任何地方执行过**，这里补上。"""

    def _load(self):
        return _exec("nodes.qianhui_watermark",
                     os.path.join(PLUGIN_NODES, "qianhui_watermark", "__init__.py"))

    def test_init_executes_and_exports_nodes(self):
        m = self._load()
        for name in ("QianhuiWatermarkAdd", "QianhuiWatermarkExtract"):
            self.assertTrue(isinstance(getattr(m, name, None), type),
                            "__init__.py 没导出节点类 %s" % name)

    def test_node_class_mappings_contents(self):
        m = self._load()
        self.assertEqual(
            set(m.NODE_CLASS_MAPPINGS),
            {"QianhuiWatermarkAdd", "QianhuiWatermarkExtract"},
        )
        # 映射的值必须与 __init__ 导出的类同一个对象，否则注册的与实际用的会漂移
        self.assertIs(m.NODE_CLASS_MAPPINGS["QianhuiWatermarkAdd"], m.QianhuiWatermarkAdd)
        self.assertIs(m.NODE_CLASS_MAPPINGS["QianhuiWatermarkExtract"], m.QianhuiWatermarkExtract)

    def test_root_init_registers_watermark_names(self):
        """根 __init__.py 真的按这两个名字注册（ast 读，不 import 根包）。

        不 import 根包：它会连带 import 全部 13 个节点模块（含 torch 与 ComfyUI 运行时），
        且它有单模块失败隔离，静默少注册一个节点时 import 照样成功。
        """
        with open(os.path.join(ROOT, "__init__.py"), encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        registered = set()
        for n in ast.walk(tree):
            # NODE_CLASS_MAPPINGS["X"] = ...   /  NODE_CLASS_MAPPINGS.update({...})
            if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call):
                f = n.value.func
                if isinstance(f, ast.Attribute) and f.attr == "update":
                    for arg in n.value.args:
                        if isinstance(arg, ast.Dict):
                            for k in arg.keys:
                                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                                    registered.add(k.value)
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    if (isinstance(t, ast.Subscript)
                            and isinstance(t.value, ast.Name)
                            and t.value.id == "NODE_CLASS_MAPPINGS"
                            and isinstance(t.slice, ast.Constant)):
                        registered.add(t.slice.value)
        for name in ("QianhuiLatentYOLO", "QianhuiWatermarkAdd", "QianhuiWatermarkExtract"):
            self.assertIn(name, registered, "根 __init__.py 没有注册 %s" % name)

    def test_root_init_registers_the_four_split_nodes(self):
        """★ 拆出来的 4 个节点必须都在根 __init__.py 的注册表里。

        为什么单独一条、而且要**正向断言**：T13 的验收标准之一是「旧注册名
        `QianhuiLatentYOLO` 从根 __init__.py 消失」，而删那 3 行的同时很容易
        把新名的注册一起碰掉 —— 那样「换掉了」就成了「换没了」：4 个节点在菜单里
        全部消失，而 `test_root_init_registers_watermark_names` 只管水印那两个，
        一个测试都不会红。所以这里把「4 个新名都在」钉死，与「旧名在不在」
        分开断言（旧名那条的取舍见 test_root_init_registers_watermark_names）。

        与另一条同样的读法：ast 读根 `__init__.py`，**不 import** 根包。
        """
        with open(os.path.join(ROOT, "__init__.py"), encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        registered = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call):
                f = n.value.func
                if isinstance(f, ast.Attribute) and f.attr == "update":
                    for arg in n.value.args:
                        if isinstance(arg, ast.Dict):
                            for k in arg.keys:
                                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                                    registered.add(k.value)
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    if (isinstance(t, ast.Subscript)
                            and isinstance(t.value, ast.Name)
                            and t.value.id == "NODE_CLASS_MAPPINGS"
                            and isinstance(t.slice, ast.Constant)):
                        registered.add(t.slice.value)
        expected = {"QianhuiYoloDetect", "QianhuiSamRefine",
                    "QianhuiRegionRepair", "QianhuiGlobalPass"}
        self.assertEqual(
            sorted(expected - registered), [],
            "根 __init__.py 少了这些新注册名：%s\n"
            "  ⇒ 那 4 个节点会从菜单里消失（而且不会有别的测试变红）。"
            % sorted(expected - registered),
        )


if __name__ == "__main__":
    unittest.main()
