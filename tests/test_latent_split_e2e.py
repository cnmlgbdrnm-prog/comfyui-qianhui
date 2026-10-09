# -*- coding: utf-8 -*-
"""T7：把单体 `QianhuiLatentYOLO` 的 4 路输出冻成 e2e 基准（golden）。

## 这个文件为什么必须在删单体之前存在

T13 会删掉单体；T7 之后如果没有基准落盘，**永久失去对拍基准**。
所以这里的判据只有一条：**逐位**。张量存 `sha256(contiguous 字节)` + 诊断摘要
（hash 相等 ⟺ `torch.equal` 为真，因为两者都是「同一 dtype/shape 下逐字节」），
而**不是** `allclose`。

## 四个入口（T11 会 import 它们，名字照抄）

    run_monolith(scenario) -> dict   # 跑单体，返回 4 路输出（+ detections 诊断）
    run_split(scenario)    -> dict   # T7 只留空壳 + NotImplementedError，T11 填
    load_golden()          -> dict
    SCENARIOS              -> list[dict]

## 基准怎么生成（**绝不由测试自己写**）

    python -X utf8 tests/test_latent_split_e2e.py --write-golden

`discover` 只会 import 本模块，`__main__` 分支不会被执行 ⇒ 基准不会随被测代码漂移。
生成基准那一次必须跑在**未改动的单体**上，`--write-golden` 会把当时的
`git rev-parse HEAD` 记进 `meta`。

## 桩：**自写一份，不 import `test_import_smoke`**

`test_import_smoke.py` 的 `_mkpkg`/`_exec` 往**它自己的**模块级 `_INJECTED` 集合记账，
而 `setUpModule` 只在该模块的用例被跑时才执行 —— 跨模块复用会让「注入的桩归谁清理」
变成共享状态耦合，与本项目踩过的 `unittest.TestSuite.run()` 重跑 `setUpModule`
那个**假绿坑**同源。故这里**照抄它的形状**（`_Stub`/`_mkpkg`/`_exec`/快照还原），
但自持一套记账。

桩分两类，报告里逐条列了：
  * **环境桩**：ComfyUI 运行时（`comfy.*` / `folder_paths` / `nodes`）与用户磁盘上的
    模型目录 —— 本机根本没有 ComfyUI 运行时，也没有模型权重；
  * **重活桩**：YOLO 推理 / SAM 推理 / VAE 解码 / 权重加载 —— 需要 GB 级权重或 GPU，
    且**本就是非确定性来源**。它们的替身全部是纯函数。
其余（4 个算法、mask、policy、geometry、blend、report 全部）都是**真跑**。

## 行尾

本文件与 golden JSON 一律 LF（`io.open(..., newline="")`）。
"""
import contextlib
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import time
import types
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _loader import ROOT, pkgs_dir  # noqa: E402

THIS_FILE = os.path.abspath(__file__)
PLUGIN_NODES = os.path.join(ROOT, "nodes")
LATENT_PKG = pkgs_dir("qianhui_latent_yolo")
GOLDEN_PATH = os.path.join(os.path.dirname(THIS_FILE), "golden", "latent_split_e2e.json")

# 非控件名：场景表里的测试侧键（`batch_size` 在 ComfyUI 里不是控件，
# 批量来自 Latent 张量的第 0 维；这里保留这个名字是为了让场景表可读）。
NON_WIDGET_KEYS = ("name", "batch_size")


# ============================================================
# 桩设施 —— 形状照抄 tests/test_import_smoke.py:49-101（自持记账）
# ============================================================

_SNAPSHOT = None          # 装载前的 sys.modules 快照
_CUDA_AVAILABLE_ORIG = None
_NUM_THREADS_ORIG = None
_MANAGER_CACHE_ORIG = None


class _Stub(types.ModuleType):
    """任意属性都返回「调用即 None」的假件：够模块体执行完，不触发真逻辑。"""

    def __getattr__(self, k):
        if k.startswith("__"):
            raise AttributeError(k)
        return lambda *a, **kw: None


class _FolderPathsStub(_Stub):
    """`folder_paths` 桩，但 `models_dir` 必须是**字符串**。

    `_Stub` 的 `models_dir` 会返回一个 lambda，`os.path.join(lambda, "yolo")`
    当场 TypeError —— 而 `INPUT_TYPES()` 要经 `YOLOModelManager.get_model_list()`
    读它。故这里补一个必然存在的**非目录**路径：父级是文件，
    被测代码里那句兜底的 `os.makedirs` 会失败（被它自己的 bare except 吞掉），
    因此**既不会真建目录、也不改变行为**。
    """

    models_dir = os.path.join(THIS_FILE, "models_never_created")


def _mkpkg(name, path):
    """注册成包但**不执行** __init__.py，避免触发整条插件加载链。"""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(path, "__init__.py"), submodule_search_locations=[path]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    return mod


def _exec(name, path):
    """真正执行模块体。"""
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ============================================================
# 确定性替身（重活桩）
# ============================================================

class _FakeKSampler:
    """`comfy.samplers.KSampler` 的确定性替身。

    `_Stub` 会让 `KSampler()` 返回 None → `ctx.ksampler.sample(...)` 直接炸，
    采样根本没法测。这里的返回值由 (latent, seed, steps, cfg, denoise, noise_mask,
    conditioning 的结构戳) 唯一决定，且**mask 参与运算** —— 于是 mask/降噪/种子/
    ControlNet 强度与范围 任一处的行为漂移都会被基准抓到。

    语义按本仓自己的注释（`_prepare_inpaint_latent` docstring）：
    `noise_mask=1` 的区域加噪、`=0` 的区域完全保持原样。

    ★ 为什么值得让替身对 conditioning 敏感：真采样器**确实**吃 conditioning
    （ControlNet 就挂在 `conditioning[1]["control"]` 上、靠 strength/percent 影响去噪）。
    替身若完全无视它，`cn_on` 那条场景就会与 base 逐位相同 —— 「覆盖了 CN」变成一句空话。
    """

    SAMPLERS = ("euler", "ddim")
    SCHEDULERS = ("normal", "karras")

    def __init__(self, *a, **kw):
        pass

    @staticmethod
    def sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative,
               latent, denoise=1.0, **kw):
        x = latent["samples"] if isinstance(latent, dict) else latent
        mask = latent.get("noise_mask") if isinstance(latent, dict) else None

        gen = torch.Generator().manual_seed(int(seed) % (2 ** 31))
        noise = torch.randn(tuple(x.shape), generator=gen, dtype=torch.float32)
        amp = float(denoise) * (1.0 + float(steps) / 1000.0)
        delta = amp * noise
        if mask is not None:
            delta = delta * mask.float().unsqueeze(1)
        bias = (1e-3 * float(cfg) + 1e-4 * (len(str(sampler_name)) + len(str(scheduler)))
                + 1e-5 * (_cond_stamp(positive) + 2 * _cond_stamp(negative)))
        out = (x.float() + delta + bias).to(x.dtype)
        return ({"samples": out},)


def _cond_stamp(cond) -> int:
    """conditioning 的**稳定结构戳**：把「挂没挂 ControlNet、强度/范围/控制图是什么」压成整数。

    只用确定性量（不做 `hash()` —— 字符串 hash 每进程都不同），且**不读** conditioning
    里的张量内容之外的任何运行期状态。
    """
    if not cond:
        return 0
    total = 0
    for entry in cond:
        try:
            keys = entry[1]
        except Exception:
            total += 7
            continue
        ctrl = keys.get("control") if hasattr(keys, "get") else None
        total += 1 if ctrl is None else 100 + int(ctrl.stamp())
    return total


class _FakeNet:
    """`CONTROL_NET` 输入的替身。

    只需真 `ControlNetHelper.apply_controlnet` 用到的那三个方法，外加一个**确定性戳**
    （`stamp()`）让「强度 / 起止比例 / 控制图」的变化能穿过替身采样器，
    最终落到 4 路输出上。

    ★ `copy()` 必须返回**新对象**：`apply_controlnet` 每次调用都会
    `copy().set_cond_hint(...)`，共享同一个实例会让先前挂在 conditioning 上的控制对象
    被后一次调用改写，于是「逐目标的 CN 强度」在输出上就分不出来了。
    """

    def __init__(self):
        self._stamp = 0

    def copy(self):
        other = _FakeNet()
        other._stamp = self._stamp
        return other

    def set_cond_hint(self, image, strength, percent_range):
        try:
            start, end = percent_range
        except Exception:
            start, end = 0.0, 1.0
        nonzero = int(torch.count_nonzero(image)) if isinstance(image, torch.Tensor) else 0
        self._stamp = (int(round(float(strength) * 1000)) * 100003
                       + int(round(float(start) * 1000)) * 1009
                       + int(round(float(end) * 1000)) * 101
                       + (nonzero % 997))
        return self

    def set_previous_controlnet(self, prev):
        return self

    def stamp(self) -> int:
        return self._stamp


class _FakeModel:
    """`MODEL` 输入的替身（采样被桩掉了，没人真读它）。"""


class _FakeVAE:
    """`VAE` 输入的替身（VAE 解码被桩掉了）。"""


#: 桩返回的「YOLO 模型」哨兵。
_YOLO_SENTINEL = object()
#: 桩返回的模型路径哨兵：用本测试文件自己的路径 —— 它必然存在，
#: 从而通过 `prepare_run` 里 `os.path.exists(resolved_path)` 那道闸门。
#: 它**从不**被当作权重读取（`load_model` 也是桩）。
_MODEL_PATH_SENTINEL = THIS_FILE
#: 检测桩每条记录的类别。三种分别压 `_auto_params_for_det` 的三条分支
#: （脸 / 手脚 / 其余），从而真的走到手脚增强的参数路径。
_DET_CLASSES = ("face", "hand", "foot")
#: 检测桩返回的 bbox，单位是**检测图**（`_upscale_image_for_yolo` 放大后）的像素。
#: 三者互不重叠（避免 `_merge_overlapping_detections` 引入合并歧义）。
_DET_BBOXES = ([104.0, 120.0, 416.0, 480.0],
               [512.0, 300.0, 700.0, 620.0],
               [260.0, 700.0, 470.0, 940.0])

#: 运行期状态：桩靠它区分「本次是不是无检测场景」。
_RUN_STATE = {"no_detect": False, "stdout": ""}

#: 每个场景只跑一次的缓存：全场景扫描是**分钟级**开销，而多数用例并不需要新鲜结果。
#: ★ 需要「新鲜两次」的用例（确定性、两次运行的性能行对比）**绕开**这个缓存直接调
#: `run_monolith` —— 缓存不能成为掩盖非确定性的地方。
_SWEEP_CACHE = {}


def _box_mask(shape, bbox, shrink=0.1):
    """在 bbox 内填 1 的 2D 掩码（确定性的合成 seg_mask）。"""
    h, w = int(shape[0]), int(shape[1])
    x1, y1, x2, y2 = bbox
    dx, dy = (x2 - x1) * shrink, (y2 - y1) * shrink
    m = np.zeros((h, w), dtype=np.float32)
    ax1, ay1 = max(0, int(x1 + dx)), max(0, int(y1 + dy))
    ax2, ay2 = min(w, int(x2 - dx)), min(h, int(y2 - dy))
    if ax2 > ax1 and ay2 > ay1:
        m[ay1:ay2, ax1:ax2] = 1.0
    return m


def _stub_detect_with_yolo(self, model, image, confidence, imgsz=640,
                           tile_enabled=True, tile_size=640, tile_overlap=0.25,
                           use_seg=True, target_classes=None, multi_scale=True):
    """固定 3 条记录：两条带合成 seg_mask、一条 None（照计划 §验证①）。

    ★ 每次调用**构造全新对象**：detect.py 会把 bbox 原地缩放到像素空间、
    把 seg_mask 原地 resize —— 复用同一个 dict 会让第二帧被二次缩放。
    真实的 YOLO 也是每帧返回新对象。
    """
    if _RUN_STATE["no_detect"]:
        return []
    h, w = int(image.shape[0]), int(image.shape[1])
    dets = []
    for idx, (cls, bbox) in enumerate(zip(_DET_CLASSES, _DET_BBOXES)):
        det = {
            "class_name": cls,
            "class_id": idx,
            "confidence": 0.9 - 0.1 * idx,
            "bbox": list(bbox),
            "seg_mask": None,
        }
        if idx < 2:                      # 第 3 条（foot）故意不带 seg_mask
            det["seg_mask"] = _box_mask((h, w), bbox)
            det["seg_mask_shape"] = (h, w)
        dets.append(det)
    return dets


def _stub_detect_single_enhanced(self, model, image_np, confidence, imgsz, use_seg,
                                 target_classes):
    """二次检测精修桩：裁一小块区域内返回 1 条记录（带 seg_mask）。

    必须处理 tensor 入参：`run_detection` 传进来的是 `decoded_det[...]` 的**张量切片**。
    """
    h, w = int(image_np.shape[0]), int(image_np.shape[1])
    if h < 16 or w < 16:
        return []
    bbox = [float(w) // 4, float(h) // 4, float(w) * 3 // 4, float(h) * 3 // 4]
    return [{
        "class_name": _DET_CLASSES[0],
        "class_id": 0,
        "confidence": 0.95,
        "bbox": bbox,
        "seg_mask": _box_mask((h, w), bbox),
    }]


def _stub_refine_with_sam(self, image_np, detections, model_path="", conf_threshold=0.7):
    """SAM 精修桩：填确定的 `seg_mask` + `sam_score`。

    掩码尺寸取 `image_np` 的 H/W —— 与真实现同形：真 `_refine_with_sam`
    把 SAM 在**送进来的那张图**上跑出的 mask 直接塞进记录（`sam.py:462`），
    这里送进来的是 `decoded_det`（检测图空间），照抄，不改口径。
    """
    h, w = int(image_np.shape[0]), int(image_np.shape[1])
    out = []
    for idx, det in enumerate(detections):
        d = dict(det)
        d["seg_mask"] = _box_mask((h, w), d.get("bbox", [0, 0, w, h]))
        d["sam_score"] = round(0.5 + 0.1 * idx, 6)
        out.append(d)
    return out


def _stub_extract_hand_skeleton(self, image_np, bbox, target_h, target_w):
    """手骨架提取桩 → None（MediaPipe 不在本机，且要真图）。"""
    return None


def _stub_unload_sam_model(self):
    """SAM 卸载桩：真实现动 `self.sam_predictor` 等 ComfyUI 运行时状态。"""
    return None


def _stub_decode_latent_to_image(self, vae, latent_samples):
    """VAE 解码桩：跳过真 VAE，输出由 latent 的通道均值决定。

    ★ 刻意让输出**依赖输入 latent**：否则「latent 变了但预览没变」这类漂移
    会被基准放过。
    """
    x = latent_samples[0].float()                       # [C, lh, lw]
    h, w = int(x.shape[-2]) * 8, int(x.shape[-1]) * 8
    base = x.mean(dim=0, keepdim=True).unsqueeze(0)      # [1,1,lh,lw]
    up = torch.nn.functional.interpolate(base, size=(h, w), mode="nearest").squeeze(0)[0]
    img = torch.empty((h, w, 3), dtype=torch.float32)
    for c in range(3):
        img[:, :, c] = torch.clamp(up * (1.0 + 0.1 * c), 0.0, 1.0)
    return (img,)


def _stub_resolve_model_path(self, yolo_model_selection, manual_path):
    """模型路径解析桩。

    真实现读 `folder_paths.models_dir` 下的**用户磁盘**——本机既无 ComfyUI 运行时
    也无模型权重，且结果会随机器变化。返回一个必然存在的文件（本测试文件自身），
    好让 `prepare_run` 的 `os.path.exists` 闸门通过、`enable_yolo` 不被关掉。
    """
    return _MODEL_PATH_SENTINEL


def _stub_resolve_sam_model_path(self, sam_model_selection, manual_path):
    """同上；`sam_model_path` 只决定 `refine_with_sam` 走不走那条分支。"""
    return _MODEL_PATH_SENTINEL


def _stub_load_model(self, model_path, precision="FP32(默认)"):
    """YOLO 权重加载桩 → 哨兵（不真加载权重）。"""
    return _YOLO_SENTINEL


# ============================================================
# 环境装配
# ============================================================

_NODE_MODULE = None
_NODE_CLS = None
_WIDGET_DEFAULTS = None
_WIDGET_DECLS = None


def setUpModule():
    global _SNAPSHOT, _CUDA_AVAILABLE_ORIG, _NUM_THREADS_ORIG, _MANAGER_CACHE_ORIG
    global _NODE_MODULE, _NODE_CLS, _WIDGET_DEFAULTS, _WIDGET_DECLS

    _SNAPSHOT = dict(sys.modules)

    # ⓪ ★ 先**清掉**上一批测试留下的插件模块与 comfy/folder_paths 桩。
    #   不这么做会踩一个极隐蔽的坑（我实测踩到过，全量跑红、单跑绿）：
    #   `test_import_smoke` 的 `_exec` 会被它的 tearDown 摘掉，但它**传递导入**的
    #   子模块（例如 `...pipeline.sam_refine`，由 detect.py 的
    #   `from .sam_refine import ...` 带进来）不在它的 `_INJECTED` 里 ⇒ **留在**
    #   sys.modules 里。那些模块的 globals 里 `comfy` 绑定的是**上一批**的桩对象；
    #   我这里把 sys.modules["comfy"] 换成新桩，**换不掉它们手里的旧引用** ⇒
    #   `comfy.model_management` 在那个旧 `_Stub` 上退化成 lambda
    #   （`AttributeError: 'function' object has no attribute ...`）。
    #   删掉 `nodes*` 让它们按我的桩重新导入；快照里已有原件，tearDown 会还原。
    for name in [n for n in sys.modules
                 if n == "nodes" or n.startswith("nodes.")]:
        del sys.modules[name]

    # ① CPU 路径固定：基准要能在无 GPU 的机器上复现，GPU 分支（搬张量到 cuda）
    #    不仅不可移植，还会把内存/显存状态写进基准。
    _CUDA_AVAILABLE_ORIG = torch.cuda.is_available
    torch.cuda.is_available = lambda: False
    # ② 线程数固定：CPU 归约（sum/mean）的累加顺序随线程数变化，float32 下会改低位。
    _NUM_THREADS_ORIG = torch.get_num_threads()
    torch.set_num_threads(1)

    for n in ("folder_paths", "comfy", "comfy.model_management", "comfy.utils",
              "comfy.sample", "comfy.controlnet"):
        sys.modules[n] = _FolderPathsStub(n) if n == "folder_paths" else _Stub(n)
    # `comfy.samplers` 不能是裸 `_Stub`：`INPUT_TYPES` 读
    # `comfy.samplers.KSampler.SAMPLERS`，而 `_Stub` 返回的是 lambda（取属性即 AttributeError）。
    samplers = _Stub("comfy.samplers")
    samplers.KSampler = _FakeKSampler
    sys.modules["comfy.samplers"] = samplers
    # ★ 必须把子模块**挂到 `comfy` 的属性上**：被测代码写的是
    #   `import comfy.model_management` + `comfy.model_management.xxx()` ——
    #   名字 `comfy` 绑定的是这个 stub 模块，属性查找走 `_Stub.__getattr__`
    #   （返回 lambda），**根本不查 sys.modules** ⇒ 不挂就是 AttributeError。
    for n in ("model_management", "utils", "sample", "controlnet", "samplers"):
        setattr(sys.modules["comfy"], n, sys.modules["comfy." + n])
    # 中断异常要是真类（node.py 的 `except InterruptProcessingException` 需要它可被 catch）。
    sys.modules["comfy.model_management"].InterruptProcessingException = type(
        "InterruptProcessingException", (Exception,), {})

    _mkpkg("nodes", PLUGIN_NODES)
    # 真 ComfyUI 里 `nodes.py` 就定义着 KSampler；`prepare_run` 会 `nodes.KSampler()`。
    sys.modules["nodes"].KSampler = _FakeKSampler
    _mkpkg("nodes.qianhui_latent_yolo", LATENT_PKG)
    _mkpkg("nodes.qianhui_latent_yolo.mixins", os.path.join(LATENT_PKG, "mixins"))
    _mkpkg("nodes.qianhui_latent_yolo.mixins.pipeline",
           os.path.join(LATENT_PKG, "mixins", "pipeline"))
    _mkpkg("nodes.qianhui_watermark", os.path.join(PLUGIN_NODES, "qianhui_watermark"))

    # ③ 模型列表固定：真实现扫用户磁盘 + 有 30 秒 TTL 缓存（TTL 一过就重扫 ⇒ 结果随机器变）。
    #    连 `_last_scan_time` 一起钉住，否则缓存第一个用例就过期。
    from nodes.qianhui_latent_yolo.model_managers import YOLOModelManager, SAMModelManager
    _MANAGER_CACHE_ORIG = (YOLOModelManager, YOLOModelManager._cached_models,
                           YOLOModelManager._last_scan_time,
                           SAMModelManager, SAMModelManager._cached_models,
                           SAMModelManager._last_scan_time)
    now = time.time()
    for mgr, empty in ((YOLOModelManager, "[无可用模型]"), (SAMModelManager, "[无可用SAM模型]")):
        mgr._cached_models = {empty: "", "[手动输入路径]": "MANUAL_INPUT"}
        mgr._last_scan_time = now

    # ④ 共享缓存单例：它的 `low_vram_mode` 由 `torch.cuda.get_device_properties` 决定
    #    （随机器变），且会一路写进 `node._low_vram_mode`。丢掉旧实例重建。
    from nodes.qianhui_latent_yolo import model_cache
    model_cache._SHARED_CACHE = None

    _NODE_MODULE = _exec("nodes.qianhui_latent_yolo.node",
                         os.path.join(LATENT_PKG, "node.py"))
    _NODE_CLS = _NODE_MODULE.QianhuiLatentYOLO

    # ⑤ 重活桩：贴在**类**上（不是实例），4 路输出里就不含任何需要权重/GPU 的东西。
    _NODE_CLS._detect_with_yolo = _stub_detect_with_yolo
    _NODE_CLS._detect_single_enhanced = _stub_detect_single_enhanced
    _NODE_CLS._refine_with_sam = _stub_refine_with_sam
    _NODE_CLS._extract_hand_skeleton = _stub_extract_hand_skeleton
    _NODE_CLS._unload_sam_model = _stub_unload_sam_model
    _NODE_CLS._decode_latent_to_image = _stub_decode_latent_to_image
    _NODE_CLS._resolve_model_path = _stub_resolve_model_path
    _NODE_CLS._resolve_sam_model_path = _stub_resolve_sam_model_path
    model_cache.YOLOModelCache.load_model = _stub_load_model

    from test_latent_algorithms import _latent_node_widget_decls
    _WIDGET_DECLS = _latent_node_widget_decls()
    _WIDGET_DEFAULTS = _defaults_from_input_types(_NODE_CLS)


def tearDownModule():
    global _SNAPSHOT
    _SWEEP_CACHE.clear()
    from nodes.qianhui_latent_yolo import model_cache
    yolo, y_cached, y_time, sam, s_cached, s_time = _MANAGER_CACHE_ORIG
    yolo._cached_models, yolo._last_scan_time = y_cached, y_time
    sam._cached_models, sam._last_scan_time = s_cached, s_time
    if hasattr(model_cache.YOLOModelCache, "load_model"):
        del model_cache.YOLOModelCache.load_model
    model_cache._SHARED_CACHE = None

    torch.cuda.is_available = _CUDA_AVAILABLE_ORIG
    torch.set_num_threads(_NUM_THREADS_ORIG)

    # 精确还原 sys.modules：删掉本次新增的每个名字，再把快照覆盖回去。
    for name in [n for n in sys.modules if n not in _SNAPSHOT]:
        del sys.modules[name]
    sys.modules.update(_SNAPSHOT)
    _SNAPSHOT = None


# ============================================================
# 场景表
# ============================================================

SCENARIOS = [
    # 基线：只钉 seed 与批量，其余全部走 INPUT_TYPES 的声明默认值 ——
    # 于是「改一个默认值」这类漂移也会被抓到。
    {"name": "base",             "seed": 1234, "batch_size": 3},
    {"name": "scale1.5",         "seed": 1234, "batch_size": 3, "放大倍数": 1.5},
    {"name": "work_mode_保持",    "seed": 1234, "batch_size": 3,
     "工作模式": "仅修复(保持输入Latent)"},
    {"name": "work_mode_全图降噪", "seed": 1234, "batch_size": 3,
     "工作模式": "修复后全图降噪", "全图降噪强度": 0.25},
    # ★ 唯一压 `ctx.frame_history` 的场景：跨帧数值状态只在 run_global_pass 里读写
    {"name": "video",            "seed": 1234, "batch_size": 3,
     "视频模式": True, "参考帧窗口": 2, "时序一致性强度": 0.3},
    {"name": "crop_on",          "seed": 1234, "batch_size": 3, "开启裁剪精修": True},
    {"name": "crop_off",         "seed": 1234, "batch_size": 3, "开启裁剪精修": False},
    {"name": "blend_智能柔和",    "seed": 1234, "batch_size": 3, "融合模式": "智能柔和融合"},
    {"name": "blend_Poisson",    "seed": 1234, "batch_size": 3, "融合模式": "Poisson融合"},
    {"name": "blend_MPB增强",     "seed": 1234, "batch_size": 3, "融合模式": "MPB增强融合"},
    {"name": "blend_简单混合",    "seed": 1234, "batch_size": 3, "融合模式": "简单混合"},
    {"name": "sam_on",           "seed": 1234, "batch_size": 3, "启用SAM精细化": True},
    {"name": "sam_off",          "seed": 1234, "batch_size": 3, "启用SAM精细化": False},
    # 无检测场景（桩让检测返回空）：两条才真走到「未检测时处理」的两个分支
    {"name": "no_detect_保持原样", "seed": 1234, "batch_size": 3,
     "未检测时处理": "保持原样", "_stub_no_detect": True},
    {"name": "no_detect_全图重绘", "seed": 1234, "batch_size": 3,
     "未检测时处理": "全图重绘", "_stub_no_detect": True},
    # 性能行（残差 R2）唯一被走到的场景：不设它，归一化那条分支就是死代码
    {"name": "profiling",        "seed": 1234, "batch_size": 3, "启用性能分析": True},
    {"name": "refine_det",       "seed": 1234, "batch_size": 3, "二次检测精修": True},
    {"name": "ema_on",           "seed": 1234, "batch_size": 3, "启用EMA注意力": True},
    {"name": "hand_skeleton_cn", "seed": 1234, "batch_size": 3, "手骨架CN模型": _FakeNet()},
    {"name": "cn_on",            "seed": 1234, "batch_size": 3,
     "CN模型": _FakeNet(), "CN图像": torch.zeros(1, 64, 64, 3), "CN强度": 0.7},
    # `presets.py` 那条路：预设会在 parse_config 之后**改写 19 个 cfg 字段**，
    # T12 正要按节点拆它 —— 没有场景钉住它，T12 就无从对拍。
    {"name": "preset_快速模式",   "seed": 1234, "batch_size": 3, "配置预设": "快速模式(低质量)"},
    {"name": "preset_质量模式",   "seed": 1234, "batch_size": 3, "配置预设": "质量模式(慢速)"},
    # 类别规则：`_get_preset_class_rules` 与 `_parse_class_rules` 两条入口各一
    {"name": "rules_预设",        "seed": 1234, "batch_size": 3,
     "类别规则预设": "人像全面(face+hand+foot)"},
    {"name": "rules_自定义",      "seed": 1234, "batch_size": 3,
     "类别规则": "face: steps=20,denoise=0.4,limit=4\nhand: denoise=0.75,limit=2"},
]


# ============================================================
# 输入构造
# ============================================================

def _default_of(spec):
    """从 `INPUT_TYPES` 的一条声明里取默认值（与 ComfyUI 的口径一致）。"""
    if not isinstance(spec, (tuple, list)) or not spec:
        return None
    kind = spec[0]
    opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
    if isinstance(kind, (list, tuple)):          # COMBO：默认 = options[0] 或显式 default
        return opts.get("default", kind[0] if kind else None)
    return opts.get("default", None)


def _defaults_from_input_types(cls):
    """真调 `INPUT_TYPES()`（不是照抄常量）取每个控件的默认值。"""
    out = {}
    for section in ("required", "optional"):
        for name, spec in cls.INPUT_TYPES()[section].items():
            out[name] = _default_of(spec)
    assert "配置预设" in out, "INPUT_TYPES 里没有 配置预设 —— 解析口径可能变了"
    return out


def _make_latent(batch=3, channels=4, lh=16, lw=16, seed=1234):
    gen = torch.Generator().manual_seed(seed)
    return {"samples": torch.randn((batch, channels, lh, lw), generator=gen,
                                   dtype=torch.float32)}


def _base_inputs(batch=3):
    """ComfyUI 里由连线喂进来的 5 个必填输入（无默认值）。"""
    cond = [[torch.zeros(1, 77, 8), {"pooled_output": None}]]
    return {
        "Latent": _make_latent(batch=batch),
        "模型": _FakeModel(),
        "VAE": _FakeVAE(),
        "正面提示": cond,
        "负面提示": [[t.clone(), dict(k)] for t, k in cond],
    }


def _inputs_for(scenario):
    """场景 → 一次 `process(**kwargs)` 的完整入参。

    先铺满 `INPUT_TYPES` 的默认值，再覆盖 5 个必填输入，最后叠场景覆盖 ——
    于是**没有被场景点名的每一个控件都真的按它的声明默认值参与运算**。
    `**kwargs` 会静默吞掉拼错的控件名，故 `_require_widget` 先把名字验一遍。
    """
    inputs = dict(_WIDGET_DEFAULTS)
    inputs.update(_base_inputs(batch=int(scenario.get("batch_size", 3))))
    for key, value in scenario.items():
        if key in NON_WIDGET_KEYS or key.startswith("_"):
            continue
        _require_widget(key)
        inputs[key] = value
    return inputs


def _require_widget(name):
    if name not in _WIDGET_DECLS:
        raise AssertionError(
            "场景表里的 %r 不是 node.py 的真实控件名 —— 它会被 **kwargs 静默吞掉，"
            "测试照样绿而根本没测到。真实控件名见 INPUT_TYPES。" % (name,))


def _require_env():
    if _NODE_CLS is None or _WIDGET_DEFAULTS is None:
        raise RuntimeError("环境未装配：setUpModule 没跑（本模块的用例必须经 discover/unittest 跑）")


# ============================================================
# 跑单体 / 跑拆分（T11 填）
# ============================================================

def run_monolith(scenario) -> dict:
    """跑单体 `QianhuiLatentYOLO.process()`，返回 4 路输出。

    Returns:
        {"latent": Tensor,      # 第 1 路 {"samples": ...} 里的 samples
         "preview": Tensor,     # 第 2 路 检测预览 IMAGE
         "mask": Tensor,        # 第 3 路 检测遮罩 MASK
         "info": str,           # 第 4 路 检测信息 STRING
         "detections": list}    # 诊断用：本次的 all_dets（不在 4 路输出里）
    """
    _require_env()
    inputs = _inputs_for(scenario)
    _RUN_STATE["no_detect"] = bool(scenario.get("_stub_no_detect", False))

    captured = []
    orig_build_report = _NODE_MODULE.build_report

    def _capture(node, ctx):
        res = orig_build_report(node, ctx)
        captured.append(list(ctx.all_dets))       # 4 路输出之外，只有这里能观察到检测记录
        return res

    _NODE_MODULE.build_report = _capture
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):     # 被测代码 print 极多；不改行为，只收噪音
            out = _NODE_CLS().process(**inputs)
    finally:
        _NODE_MODULE.build_report = orig_build_report
        _RUN_STATE["stdout"] = buf.getvalue()

    latent, preview, mask, info = out
    return {"latent": latent["samples"], "preview": preview, "mask": mask,
            "info": info, "detections": captured[0] if captured else []}


def run_split(scenario) -> dict:
    """跑拆分后的 4 节点链，返回与 `run_monolith` **同形状**的 dict。

    T7 只留空壳；**T11 填这个**（`nodes/qianhui_latent_yolo/node_detect.py` /
    `node_sam.py` / `node_repair.py` / `node_global.py` 串起来跑同一场景）。
    填完把 `TestSplitSide.test_split_matches_golden` 上的 `@unittest.skip` 摘掉即可。

    ★ **跑法必须逐条照抄下面四条口径**，否则不是「苹果对苹果」。
      T7 之后单体即删 —— 这是最后一次能把它们钉进**代码**（而不只是报告）的机会：

    1. **入参用 `_inputs_for(scenario)`**：它先铺满 `INPUT_TYPES()` 的**声明默认值**、
       再叠场景覆盖。**别自己写死常量**，否则「改一个默认值」这类漂移就测不出来。
    2. **`detections` 必须从跨帧累加器取**（单体里是 `ctx.all_dets`）：
       `global_pass.py:96` 每帧结尾都会 `del ... ctx.detections ...`，而
       `report_bits.py:44,54` 的 `检测总数(使用后)` 读的正是那个累加器。
       改成逐帧 `ctx.detections` ⇒ 运行期 AttributeError。
    3. **`seg_mask` 的空间口径两条路本来就不一致，照抄别修**：
       走 detect 阶段的那类（`base`）是**像素空间 128×128**（detect.py 会把检测图空间的
       掩码 resize 回来）；走 SAM 之后的那类（`sam_on`）**不再归一化**，是**检测图空间
       1024×1024**（`sam.py:462` 塞进去的就是 SAM 在送入那张图上跑出的掩码）。
       4 个节点的 SAM 节点必须复现这个口径，不要「顺手修」。
    4. **检测阶段的排序共三处，拆节点时三处都要搬**：`detect.py:269` 与 `detect.py:287`
       各一次 `node._sort_detections(...)`（后者在合并重叠框之后），`detect.py:363`
       一次 `_priority_sort_detections(...)`。只搬一处会改变最终顺序。

    比对用 `TestSplitSide.test_split_matches_golden`；判据是 `sha256` / `shape` / `dtype`
    （即逐位），`min/max/sum` **只作失败信息里的附注** —— `x.sum()` 是并行归约，
    累加序随 CPU 线程数变，放进断言会「张量逐位相同却整片红」。
    """
    raise NotImplementedError("T11 填这个：串起 4 个节点，返回与 run_monolith 同形状的 dict")


def _cached_run(scenario) -> dict:
    """按场景名缓存一次运行结果（**只为省时间**，见 `_SWEEP_CACHE` 的说明）。"""
    name = scenario["name"]
    if name not in _SWEEP_CACHE:
        _SWEEP_CACHE[name] = run_monolith(scenario)
    return _SWEEP_CACHE[name]


def _sweep():
    """把每个场景都真跑一遍（缺谁补谁），返回 {name: result}。"""
    for scenario in SCENARIOS:
        _cached_run(scenario)
    return _SWEEP_CACHE


def scenario_coverage(result) -> dict:
    """场景 → 每一路输出的实测覆盖（给报告用的覆盖表，也是反空转判据）。

    ★ `mask` 要分两件事看：**形状非空**（numel > 0）与**内容非零**。
    无检测场景下 `ctx.detection_mask` 保持全零 —— 那是**设计如此**的输出，
    不是「没跑到」；用一个 `nonzero` 判据会把设计行为误判成空转。
    """
    return {
        "latent": {"numel": int(result["latent"].numel()), "nonzero": _nonzero(result["latent"])},
        "preview": {"numel": int(result["preview"].numel()), "nonzero": _nonzero(result["preview"])},
        "mask": {"numel": int(result["mask"].numel()), "nonzero": _nonzero(result["mask"])},
        "detections": len(result["detections"]),
        "info_len": len(result["info"]),
    }


def load_golden() -> dict:
    """读基准。文件不存在就抛 ---- **绝不自动生成**（自动生成等于没有基准）。"""
    with io.open(GOLDEN_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def golden_entries(golden):
    """{"meta": ..., "scenarios": {name: entry}}。"""
    return {"meta": golden.get("meta", {}),
            "scenarios": {e["name"]: e for e in golden["scenarios"]}}


# ============================================================
# 归一化 / 摘要
# ============================================================

def digest(tensor) -> dict:
    """张量的摘要：`sha256(contiguous 字节)` + `shape`/`dtype` + 诊断量。

    ★ **判据只有 `sha256` / `shape` / `dtype`**（见 `_ANCHORED_DIGEST_FIELDS`）：
    sha256 相等 ⟺ 同一 dtype/shape 下逐位相等（⟺ `torch.equal`），所以这就是「逐位」，
    不是 `allclose`。`numel`/`min`/`max`/`sum` 是**只出现在失败信息里的附注** ——
    它们里面 `sum` 是**并行归约**，累加序随 CPU 线程数变（实测同一份字节在 1/4/8 线程下
    sha256 全同、而 `sum` 的 float32 低位会变）⇒ 放进断言会「逐位相同却整片红」。
    """
    if tensor is None:
        return None
    x = tensor.detach().to("cpu").contiguous()
    return {
        "sha256": hashlib.sha256(x.numpy().tobytes()).hexdigest(),
        "shape": [int(v) for v in x.shape],
        "dtype": str(x.dtype).replace("torch.", ""),
        "numel": int(x.numel()),
        "min": round(float(x.min()), 6),
        "max": round(float(x.max()), 6),
        "sum": round(float(x.sum().to(torch.float64)), 6),
    }


#: 判据只用这三个字段。`sha256` 是逐位量；`shape`/`dtype` 是它的前提（也可读性强得多）。
#: 其余字段（`numel`/`min`/`max`/`sum`）**刻意不进断言**，只作失败附注 —— 理由见 `digest`。
_ANCHORED_DIGEST_FIELDS = ("sha256", "shape", "dtype")


def route_identity(d):
    """判据元组：`(sha256, shape, dtype)`。"""
    return tuple(d[f] for f in _ANCHORED_DIGEST_FIELDS)


def _digest_note(live, entry) -> str:
    """失败信息里的**附注**：把逐位之外的诊断量差异列出来（不参与判定）。"""
    lines = []
    for field in ("numel", "min", "max", "sum"):
        a, b = live.get(field), entry.get(field)
        if a != b:
            lines.append("      %-6s 实测 %r ｜ 基准 %r" % (field, a, b))
    if not lines:
        lines.append("      （min/max/sum 也都一致）")
    return "\n".join(lines)


def assert_route_matches(case, name, route, live, entry):
    """比一条路由：**只锚 `sha256`/`shape`/`dtype`**，诊断量只写进失败信息。

    抽成模块级函数是为了让 `TestSplitSide.test_split_matches_golden`（T11 打开的那条）
    用**完全同一套判据** —— 否则两侧判据漂移，对拍就成了自说自话。
    """
    if route_identity(live) == route_identity(entry):
        return
    case.fail(
        "%s 的第 %s 路与基准不一致（判据 = sha256/shape/dtype，即逐位）\n"
        "    实测 sha256/shape/dtype: %r\n"
        "    基准 sha256/shape/dtype: %r\n"
        "    诊断附注（**不参与判定**，sum 是并行归约、随线程数变）：\n%s\n"
        "    本次 stdout 片段:\n%s"
        % (name, route, route_identity(live), route_identity(entry),
           _digest_note(live, entry), _stdout_tail()))


def normalize_detections(detections) -> list:
    """检测记录规范化：键排序、浮点定点舍入、数组转摘要。存 JSON 用。"""
    out = []
    for det in detections:
        item = {}
        for key in sorted(det):
            value = det[key]
            if isinstance(value, np.ndarray):
                item[key] = digest(torch.from_numpy(np.ascontiguousarray(value)))
            elif isinstance(value, torch.Tensor):
                item[key] = digest(value)
            elif isinstance(value, float):
                item[key] = round(value, 6)
            elif isinstance(value, (list, tuple)):
                item[key] = [round(float(v), 6) if isinstance(v, float) else v for v in value]
            else:
                item[key] = value
        out.append(item)
    return out


#: 性能行里三个「必然不同」的数值位（残差 R2）：总耗时 / 平均每帧 / 显存行。
_PERF_TOTAL_RE = re.compile(r"(\[性能统计\] 总耗时: )[0-9.]+(秒 \| 平均每帧: )[0-9.]+(秒)")
_PERF_VRAM_RE = re.compile(r"(\[性能统计\] 显存: 已分配 )[0-9.]+(MB \| 已保留 )[0-9.]+(MB)")
#: 归一化用的占位符：必须是**非数字**，否则再次归一化会不稳。
PERF_PLACEHOLDER = "@"


def normalize_info(info: str) -> str:
    """把性能行里的三个数值换成占位符（R2：单体量整条流水线，节点 D 只能求和 4 段）。

    只动 `[性能统计]` 那两行（锚定前缀），其余行一个字节都不碰 ——
    否则「归一化」会变成「什么都放过」。
    """
    if not info:
        return info
    info = _PERF_TOTAL_RE.sub(r"\1" + PERF_PLACEHOLDER + r"\2" + PERF_PLACEHOLDER + r"\3", info)
    info = _PERF_VRAM_RE.sub(r"\1" + PERF_PLACEHOLDER + r"\2" + PERF_PLACEHOLDER + r"\3", info)
    return info


def strip_perf_lines(info: str) -> str:
    """去掉性能行，剩下的部分必须逐字相同。"""
    return "\n".join(line for line in (info or "").split("\n") if "[性能统计]" not in line)


#: 非空判定：任一元素非零即算「有内容」（用于场景覆盖表）。
def _nonzero(tensor) -> bool:
    return tensor is not None and bool(torch.count_nonzero(tensor) > 0)


def _scenario_entry(scenario):
    """跑一次，产出 golden 里的一条。"""
    result = run_monolith(scenario)
    return {
        "name": scenario["name"],
        "seed": scenario.get("seed"),
        "batch_size": int(scenario.get("batch_size", 3)),
        "latent": digest(result["latent"]),
        "preview": digest(result["preview"]),
        "mask": digest(result["mask"]),
        "detections": normalize_detections(result["detections"]),
        "info": normalize_info(result["info"]),
    }


def write_golden(path=GOLDEN_PATH, stream=sys.stdout):
    """生成基准。**只能从命令行显式调用**（`--write-golden`）。"""
    _require_env()
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL).decode("ascii").strip()
    except Exception:
        head = "unknown"
    entries = [_scenario_entry(s) for s in SCENARIOS]
    doc = {
        "meta": {
            "generated_from": "QianhuiLatentYOLO.process（未改动的单体）",
            "git_head": head,
            "torch": torch.__version__,
            "device": "cpu",
            "num_threads": torch.get_num_threads(),
            "scenario_count": len(entries),
        },
        "scenarios": entries,
    }
    directory = os.path.dirname(path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    with io.open(path, "w", encoding="utf-8", newline="") as fh:   # newline="" ⇒ LF
        fh.write(json.dumps(doc, ensure_ascii=False, indent=2, sort_keys=True))
        fh.write(u"\n")
    stream.write("已写入基准: %s\n" % path)
    stream.write("生成时的 git HEAD: %s\n" % head)
    stream.write("场景数: %d\n" % len(entries))


# ============================================================
# 用例
# ============================================================

class TestScenarioTable(unittest.TestCase):
    """场景表自身的守卫：控件名错了会被 `**kwargs` 静默吞掉，测试照样绿。"""

    def test_every_scenario_key_is_a_real_widget(self):
        offenders = []
        for scenario in SCENARIOS:
            for key in scenario:
                if key in NON_WIDGET_KEYS or key.startswith("_"):
                    continue
                if key not in _WIDGET_DECLS:
                    offenders.append("%s: %r" % (scenario["name"], key))
        self.assertEqual(offenders, [], "场景表里出现了 node.py 没有的控件名（会被静默吞掉）")

    def test_scenario_names_are_unique(self):
        names = [s["name"] for s in SCENARIOS]
        self.assertEqual(len(names), len(set(names)), "场景名重复：%r" % (names,))

    def test_override_values_are_legal_widget_options(self):
        """下拉框类控件（COMBO）的取值必须真的在选项里，否则运行期会被静默忽略。"""
        offenders = []
        for scenario in SCENARIOS:
            for key, value in scenario.items():
                if key in NON_WIDGET_KEYS or key.startswith("_"):
                    continue
                kind = _WIDGET_DECLS[key][0]
                if kind != "COMBO" or not isinstance(value, str):
                    continue
                if value not in _WIDGET_DECLS[key][1]["options"]:
                    offenders.append("%s: %s=%r 不在 %r"
                                     % (scenario["name"], key, value,
                                        _WIDGET_DECLS[key][1]["options"]))
        self.assertEqual(offenders, [], "场景表里有非法的下拉框取值")

    def test_the_briefed_scenario_families_are_all_present(self):
        """简报的每一项都在（「别减项」）—— 名字改了也要保持这些家族齐全。

        `preset` / `rules` 是我加的两族（简报没点名，但 T12 要拆预设表，
        没有场景钉住那条路就无从对拍）。
        """
        required = ("base", "scale1.5", "work_mode", "video", "crop", "blend",
                    "sam", "no_detect", "preset", "rules")
        names = " ".join(s["name"] for s in SCENARIOS)
        missing = [r for r in required if r not in names]
        self.assertEqual(missing, [], "场景表缺了这些家族：%r" % (missing,))
        video = [s for s in SCENARIOS if s["name"] == "video"][0]
        self.assertTrue(video.get("视频模式"))
        self.assertEqual(video.get("参考帧窗口"), 2,
                         "压 ctx.frame_history 的那条必须带 frame_window=2")


class TestNormalization(unittest.TestCase):
    """归一化本身必须**既必要又充分**：放过该放过的，拦住不该放过的。"""

    RAW = ("处理完成: 3 张\n"
           "工作模式: 仅修复(保持输入Latent)\n"
           "检测总数(使用后): 9\n"
           "\n[性能统计] 总耗时: 12.34秒 | 平均每帧: 4.11秒\n"
           "[性能统计] 显存: 已分配 123MB | 已保留 456MB")

    def test_perf_numbers_are_normalized(self):
        got = normalize_info(self.RAW)
        self.assertIn(PERF_PLACEHOLDER, got)
        self.assertNotIn("12.34", got)
        self.assertNotIn("4.11", got)
        self.assertNotIn("456", got)

    def test_non_perf_numbers_are_not_normalized(self):
        """反空转：非性能行的数字必须原样保留，否则这条归一化就是个橡皮图章。"""
        got = normalize_info(self.RAW)
        self.assertIn("处理完成: 3 张", got)
        self.assertIn("检测总数(使用后): 9", got)
        self.assertIn("工作模式: 仅修复(保持输入Latent)", got)

    def test_same_info_with_different_timings_normalizes_equal(self):
        other = self.RAW.replace("12.34", "99.99").replace("4.11", "33.33").replace(
            "123MB", "7MB").replace("456MB", "8MB")
        self.assertNotEqual(self.RAW, other)
        self.assertEqual(normalize_info(self.RAW), normalize_info(other))

    def test_a_changed_non_perf_value_is_not_masked(self):
        """对照组：把**非**性能行的数字改掉，归一化不该把差异藏起来。"""
        tampered = self.RAW.replace("检测总数(使用后): 9", "检测总数(使用后): 8")
        self.assertNotEqual(normalize_info(self.RAW), normalize_info(tampered))

    def test_strip_perf_lines_keeps_everything_else(self):
        self.assertNotIn("[性能统计]", strip_perf_lines(self.RAW))
        self.assertIn("检测总数(使用后): 9", strip_perf_lines(self.RAW))


class TestDigest(unittest.TestCase):
    """摘要的判据必须红得起来（「一个不会红的检查等于没有」）。"""

    def test_single_bit_change_changes_the_digest(self):
        base = torch.zeros(4, dtype=torch.float32)
        other = base.clone()
        other[2] = 1e-7                     # float32 下确实改了一位
        self.assertFalse(torch.equal(base, other))
        self.assertNotEqual(digest(base)["sha256"], digest(other)["sha256"])

    def test_digest_is_stable_for_the_same_value(self):
        a = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        self.assertEqual(digest(a), digest(a.clone()))

    def test_digest_ignores_layout_but_not_value(self):
        """同一批数值的非连续视图必须与 contiguous 一致（hashes the bytes, not the strides）。"""
        a = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        self.assertEqual(digest(a)["sha256"], digest(a.t().contiguous().t().contiguous())["sha256"])


class TestComparisonCriterion(unittest.TestCase):
    """★ 判据的边界：只锚 `sha256`/`shape`/`dtype`，诊断量放不进断言。

    这两条是**修复轮 1** 加的：原先 `_compare` 比的是整个 `digest` dict，于是
    `x.sum()`（并行归约）随 CPU 线程数变低位 ⇒ 只把线程钉子换成 `pass`（本机默认 8 线程）
    就有 **20 条场景红在第 2 路 preview**，而 `sha256` 一模一样、报错却是 `sum` 不符。
    """

    def test_diagnostic_fields_are_not_part_of_the_criterion(self):
        live = digest(torch.arange(8, dtype=torch.float32))
        entry = dict(live)
        entry["sum"] = live["sum"] + 1.0            # 只动并行归约出来的诊断量
        entry["min"] = live["min"] - 1.0
        entry["max"] = live["max"] + 1.0
        entry["numel"] = live["numel"] + 7
        assert_route_matches(self, "synthetic", "preview", live, entry)   # 不该红

    def test_sha256_shape_dtype_are_still_the_criterion(self):
        a = digest(torch.zeros(4, dtype=torch.float32))
        for field, value in (("sha256", "0" * 64),
                             ("shape", [5]),
                             ("dtype", "float64")):
            entry = dict(a)
            entry[field] = value
            with self.assertRaises(AssertionError, msg="改 %s 必须红" % field):
                assert_route_matches(self, "synthetic", "preview", a, entry)


class TestGoldenFile(unittest.TestCase):
    """基准文件必须**可读**（不是一坨 base64），且覆盖每一个场景。"""

    @classmethod
    def setUpClass(cls):
        if not os.path.isfile(GOLDEN_PATH):
            raise AssertionError(
                "基准不存在：%s。用 `python -X utf8 tests/test_latent_split_e2e.py "
                "--write-golden` 在**未改动的单体**上生成。" % GOLDEN_PATH)

    def test_golden_is_plain_text_json(self):
        with io.open(GOLDEN_PATH, encoding="utf-8") as fh:
            raw = fh.read()
        self.assertNotIn("\r", raw, "基准必须是 LF")
        doc = json.loads(raw)
        self.assertIn("scenarios", doc)
        # 「可读」的判据：不许出现**大块不透明数据**。阈值 1000 而不是 200 ——
        # `检测信息` 本来就是整段文本（profiling 场景已 199 字符），
        # 卡在 200 会让以后往 info 里多写两行就误报。
        blobs = [v for v in _walk_strings(doc) if len(v) > 1000]
        self.assertEqual(blobs, [], "基准里有超长字符串（疑似 base64 大块）：%r" % (blobs[:1],))

    def test_golden_covers_every_scenario(self):
        entries = golden_entries(load_golden())["scenarios"]
        self.assertEqual(sorted(entries), sorted(s["name"] for s in SCENARIOS))

    def test_golden_records_the_monolith_head(self):
        meta = load_golden()["meta"]
        self.assertEqual(meta.get("device"), "cpu")
        self.assertRegex(meta.get("git_head", ""), r"^[0-9a-f]{7,40}$|^unknown$")

    def test_every_entry_has_all_four_routes(self):
        for name, entry in golden_entries(load_golden())["scenarios"].items():
            for route in ("latent", "preview", "mask"):
                self.assertIn(route, entry, "%s 缺 %s" % (name, route))
                self.assertIn("sha256", entry[route])
            self.assertIn("info", entry, "%s 缺 info" % name)
            self.assertIn("detections", entry, "%s 缺 detections" % name)
        # 非空转：至少有一个场景的每条路都真有内容
        base = golden_entries(load_golden())["scenarios"]["base"]
        for route in ("latent", "preview", "mask"):
            self.assertGreater(base[route]["numel"], 0, "base 的 %s 是空的" % route)


class TestMonolithAgainstGolden(unittest.TestCase):
    """★ 本任务的核心：单体今天必须与基准**逐位**一致。"""

    def _compare(self, name, result, entry):
        for route in ("latent", "preview", "mask"):
            assert_route_matches(self, name, route, digest(result[route]), entry[route])
        self.assertEqual(
            normalize_detections(result["detections"]), entry["detections"],
            "%s 的 detections 与基准不一致" % name)
        self.assertEqual(normalize_info(result["info"]), entry["info"],
                         "%s 的 检测信息 与基准不一致\n实测:\n%s\n基准:\n%s"
                         % (name, result["info"], entry["info"]))

    def test_monolith_matches_golden(self):
        entries = golden_entries(load_golden())["scenarios"]
        sweep = _sweep()
        for scenario in SCENARIOS:
            with self.subTest(scenario=scenario["name"]):
                self._compare(scenario["name"], sweep[scenario["name"]],
                              entries[scenario["name"]])

    def test_scenarios_actually_produce_outputs(self):
        """★ 场景覆盖表 + 反空转网：每个场景的 4 路输出都得**真的**有内容。

        与 `test_monolith_matches_golden` 共用同一次扫描（`_sweep`），不额外跑。
        「哪几路非空」的实测表见 task-7-report.md。断言的是：
          * 三路张量**形状**非空（numel > 0），`info` 非空串；
          * latent / preview **内容非零**（全零说明桩空转了）；
          * mask：有检测时必须非零；无检测时**必须全零**（那是设计行为，钉住它）。
        """
        sweep = _sweep()
        bad = []
        for scenario in SCENARIOS:
            name = scenario["name"]
            cov = scenario_coverage(sweep[name])
            for route in ("latent", "preview", "mask"):
                if cov[route]["numel"] <= 0:
                    bad.append("%s.%s 形状为空" % (name, route))
            for route in ("latent", "preview"):
                if not cov[route]["nonzero"]:
                    bad.append("%s.%s 内容全零（桩可能空转）" % (name, route))
            if not cov["info_len"]:
                bad.append("%s.info 为空" % name)
            if scenario.get("_stub_no_detect"):
                if cov["mask"]["nonzero"]:
                    bad.append("%s.mask 应为全零（无检测是设计行为）" % name)
            elif not cov["mask"]["nonzero"]:
                bad.append("%s.mask 内容全零（有检测却没遮罩）" % name)
        self.assertEqual(bad, [], "场景覆盖不达标：%r" % (bad,))
        # 反空转：检测桩在非无检测场景下必须真的产出记录（否则「检测路径」全没测到）
        for scenario in SCENARIOS:
            if scenario.get("_stub_no_detect"):
                self.assertEqual(sweep[scenario["name"]]["detections"], [],
                                 "%s 应无检测" % scenario["name"])
            else:
                self.assertGreater(
                    len(sweep[scenario["name"]]["detections"]), 0,
                    "%s 一条检测都没有 —— 检测桩没被走到" % scenario["name"])
        # 且每条检测的 seg_mask 覆盖「两条有、一条 None」这个形状（计划 §验证①）
        dets = sweep["base"]["detections"]
        self.assertEqual(len(dets), 9, "base 应是 3 帧 × 3 目标")
        with_seg = sum(1 for d in dets if d.get("seg_mask") is not None)
        self.assertEqual(with_seg, 6, "base 里应有 6 条带 seg_mask（3 帧 × 2 条）")

    def test_monolith_run_is_deterministic(self):
        """同一场景跑两遍必须逐位一致；否则「基准」本身就没有意义。

        ★ 刻意**绕开** `_SWEEP_CACHE` 真跑两次。
        """
        scenario = [s for s in SCENARIOS if s["name"] == "base"][0]
        a, b = run_monolith(scenario), run_monolith(scenario)
        for route in ("latent", "preview", "mask"):
            self.assertEqual(digest(a[route])["sha256"], digest(b[route])["sha256"], route)
        self.assertEqual(normalize_info(a["info"]), normalize_info(b["info"]))
        self.assertEqual(strip_perf_lines(a["info"]), strip_perf_lines(b["info"]))


class TestScenarioWiring(unittest.TestCase):
    """★ 反空转：每个场景的旋钮必须**真的接了线**。

    一条「与 base 逐位相同」的场景，如果它本来就是重述默认值，那是对的；
    如果它设的是一个**非默认**旋钮却仍然与 base 相同，那说明那个旋钮没被走到 ——
    测试照样绿，而它宣称覆盖的东西一个都没覆盖。这份名单是刻意写死的。
    """

    #: 只重述默认值的场景：必须与 base 逐位相同（钉住「显式给默认值 ≡ 不给」）。
    REDUNDANT_VS_BASE = ("work_mode_保持", "crop_on", "blend_智能柔和", "sam_off")
    #: 只压**代码路径**、数值上必然等于 base 的场景（替身在那一步返回空，
    #: 真实现里 MediaPipe 没检测到手部时也是这条路径）。
    PATH_ONLY_VS_BASE = ("hand_skeleton_cn",)
    #: 只改 `检测信息`、三路张量必然与 base 相同的场景。
    INFO_ONLY_VS_BASE = ("profiling",)

    def _routes(self, result):
        return {r: digest(result[r])["sha256"] for r in ("latent", "preview", "mask")}

    def test_scenarios_are_wired_and_controls_match_base(self):
        sweep = _sweep()
        base = sweep["base"]
        base_routes, base_dets = self._routes(base), normalize_detections(base["detections"])

        for name in self.REDUNDANT_VS_BASE + self.PATH_ONLY_VS_BASE:
            with self.subTest(scenario=name):
                self.assertEqual(self._routes(sweep[name]), base_routes,
                                 "%s 重述了默认值，却与 base 不同 —— 默认值口径变了？" % name)
                self.assertEqual(normalize_detections(sweep[name]["detections"]), base_dets, name)

        for name in self.INFO_ONLY_VS_BASE:
            with self.subTest(scenario=name):
                self.assertEqual(self._routes(sweep[name]), base_routes,
                                 "%s 只该改 检测信息" % name)
                self.assertEqual(normalize_detections(sweep[name]["detections"]), base_dets, name)
                self.assertNotEqual(sweep[name]["info"], base["info"],
                                    "%s 的 检测信息 没变 ⇒ 性能统计那段没被走到" % name)

        known = set(self.REDUNDANT_VS_BASE + self.PATH_ONLY_VS_BASE + self.INFO_ONLY_VS_BASE)
        dead = []
        for scenario in SCENARIOS:
            name = scenario["name"]
            if name == "base" or name in known:
                continue
            changed = (self._routes(sweep[name]) != base_routes
                       or normalize_detections(sweep[name]["detections"]) != base_dets
                       or sweep[name]["info"] != base["info"])
            if not changed:
                dead.append(name)
        self.assertEqual(dead, [], "这些场景的旋钮没接线（输出与 base 完全相同）：%r" % (dead,))


class TestInfoPerfLines(unittest.TestCase):
    """★ 残差 R2：性能行**只差**总耗时 / 平均每帧 / 显存这三个数值。

    让 R2 被「证明存在且被限定」，而不是被「跳过」。
    """

    def _profiling_scenario(self):
        return [s for s in SCENARIOS if s["name"] == "profiling"][0]

    def _profiling_result(self):
        return _cached_run(self._profiling_scenario())

    def test_profiling_scenario_really_emits_the_perf_lines(self):
        info = self._profiling_result()["info"]
        self.assertIn("[性能统计] 总耗时:", info, "profiling 场景没走到性能行 ⇒ 下面全是空转")
        perf = [l for l in info.split("\n") if "[性能统计]" in l]
        # CPU 下 `torch.cuda.is_available()` 为假 ⇒ 不打印「显存」行。
        # 显存行的归一化由 TestNormalization 用合成输入覆盖（本机造不出真 CUDA 行）。
        self.assertEqual(len(perf), 1, "CPU 基准应只有一条性能行，实为 %r" % (perf,))
        self.assertRegex(perf[0], r"^\[性能统计\] 总耗时: [0-9.]+秒 \| 平均每帧: [0-9.]+秒$")

    def test_two_runs_differ_only_in_the_perf_numbers(self):
        """★ 真跑两次（绕开缓存）：除了性能行，两条 `检测信息` 必须逐字相同。"""
        scenario = self._profiling_scenario()
        a, b = run_monolith(scenario), run_monolith(scenario)
        self.assertEqual(strip_perf_lines(a["info"]), strip_perf_lines(b["info"]),
                         "除了性能行，两条 检测信息 必须逐字相同")
        self.assertEqual(normalize_info(a["info"]), normalize_info(b["info"]))
        self.assertEqual(normalize_detections(a["detections"]),
                         normalize_detections(b["detections"]))
        # 时间数值本身是这次运行的真实观测量（不参与断言，只写进失败信息里）
        self.assertEqual(len([l for l in a["info"].split("\n") if "[性能统计]" in l]), 1,
                         "两次运行之间性能行的条数变了 —— 那不是 R2，那是行为漂移")

    def test_the_normalization_boundary_is_exact(self):
        """两个方向都钉住：性能数值可换，别处一个字都不许换。"""
        info = self._profiling_result()["info"]
        m = re.search(r"总耗时: ([0-9.]+)秒", info)
        self.assertIsNotNone(m)
        # ① 只改性能数值 → 归一化后相同
        self.assertEqual(normalize_info(info), normalize_info(info.replace(m.group(1), "77.77")))
        # ② 改非性能行 → 归一化后必须不同（否则归一化是橡皮图章）
        head = info.split("\n")[0]
        tampered = info.replace(head, head + "x")
        self.assertNotEqual(normalize_info(info), normalize_info(tampered))

    def test_non_profiling_info_is_fully_literal(self):
        """不开性能分析时不该有性能行 —— 于是 检测信息 完全逐字可比。"""
        scenario = [s for s in SCENARIOS if s["name"] == "base"][0]
        info = _cached_run(scenario)["info"]
        self.assertNotIn("[性能统计]", info)
        self.assertEqual(normalize_info(info), info)


class TestSplitSide(unittest.TestCase):
    """T11 打开这一格：拆分后的 4 节点链必须与同一份基准逐位一致。"""

    @unittest.skip("T11 填这个：实现 run_split 后摘掉本 skip")
    def test_split_matches_golden(self):
        """T11 填这个：`run_split(scenario)` → 与 golden 逐张量比（同 `assert_route_matches`）。"""
        entries = golden_entries(load_golden())["scenarios"]
        for scenario in SCENARIOS:
            with self.subTest(scenario=scenario["name"]):
                result = run_split(scenario)
                entry = entries[scenario["name"]]
                for route in ("latent", "preview", "mask"):
                    assert_route_matches(self, scenario["name"], route,
                                         digest(result[route]), entry[route])
                self.assertEqual(normalize_detections(result["detections"]),
                                 entry["detections"], scenario["name"])
                self.assertEqual(normalize_info(result["info"]), entry["info"],
                                 scenario["name"])


# ============================================================
# 小工具
# ============================================================

def _walk_strings(node):
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for v in node.values():
            for s in _walk_strings(v):
                yield s
    elif isinstance(node, list):
        for v in node:
            for s in _walk_strings(v):
                yield s


def _stdout_tail(limit=2000):
    text = _RUN_STATE.get("stdout", "")
    return text[-limit:]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--write-golden" in argv:
        setUpModule()
        try:
            write_golden()
        finally:
            tearDownModule()
        return 0
    return 0 if unittest.main(module=sys.modules[__name__], argv=[sys.argv[0]],
                              exit=False).result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
