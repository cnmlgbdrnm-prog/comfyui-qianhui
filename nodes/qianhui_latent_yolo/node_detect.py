# -*- coding: utf-8 -*-
"""节点 A 千绘 · YOLO检测（QianhuiYoloDetect）—— 4 节点链的第一环。

链条：A(YOLO检测) → B(SAM精修) → C(区域修复) → D(全局降噪)

## 本节点负责原单体 `process()` 的哪一段

原单体一帧一帧地跑 `run_detection → run_repair → run_global_pass`；拆开后本节点
**把全部帧的检测一次跑完**（阶段式批处理），逐帧产物装进 `DetectionPayload` 交给下游。
「阶段式与逐帧交错行为等价」由计划 §已确证的事实给出（唯一跨帧数值状态
`ctx.frame_history` 只在 `run_global_pass` 内读写，种子都是 `ctx.seed + ctx.i` 的纯函数）。

## 三路输出

    (检测预览 IMAGE, 检测数据 QIANHUI_DETECT, 检测用Latent LATENT)

* `检测预览` = `report_bits.build_preview` over 逐帧预览（原单体第 2 路输出）。
* `检测数据` = `DetectionPayload`，检测记录逐帧带下去。
* `检测用Latent` = `ctx.latent_for_detection` 的整批 —— **不是**用户输入的那个 Latent。
  它是「先做一次全图预降噪得到清晰图再检测」的产物，正是原单体
  `run_repair` 第一句 `ctx.processed_latent = ctx.latent_for_detection` 的输入。

## 与 SAM 的交接（★ 不要「顺手修」）

原单体 `run_detection` 结尾会调 `refine_with_sam`。拆分后那一段归**节点 B**。
本节点让 `ctx.sam_model_path` **保持为 None**（不是默认值凑巧为 None，是本节点
派生 Context 里显式声明、从不赋值的槽）——`refine_with_sam` 的第一道闸门就是
`if ctx.cfg.enable_sam and dets and ctx.sam_model_path:`，于是它在本节点里
**天然整段跳过**，一行都不用改 `detect.py`。检测记录原样带进 payload。

## prepare 的来路

本节点的 prepare 块 = `prepare.py:prepare_run` 里产出「本节点所读 ctx 字段」的那些行，
逐行归属见 `.superpowers/sdd/2026-10-09-latent-node-split/prepare-split-design.md` §3.2。
两处与那段代码**逐字不同**，各有理由：

1. `ctx.out_device`：原式是三分支（`结果缓存设备 == CPU / GPU / 自动`）。
   本节点**不暴露** `结果缓存设备` 与 `低显存模式` 两个控件（计划名册就是 61 个），
   所以那两个 cfg 值恒为声明默认值 —— 三个分支里只有「自动 + 非低显存」
   （= `ctx.samples.device`）可达。等值改写，不是简化行为。
2. 启动横幅：原 19 行一次打印四个节点的设置，4 个节点里没有任何一个能整段复现，
   故各节点只打印属于自己的那几行；「横幅被拆成 4 段、顺序与单体不同」已登记进
   `docs/superpowers/verification/verify_split_behaviour_preservation.py`。
"""
import os
import time
from dataclasses import dataclass
from typing import Any, Optional

import comfy.samplers
import nodes
import torch

from .algorithms import LatentAlgorithms
from .base_node import LatentKSamplerUpscaler
from .mixins.context import DetectCtx
from .mixins.detection import DetectionMixin
from .mixins.geometry import GeometryMixin
from .mixins.inpaint import InpaintMixin
from .mixins.masks import MasksMixin
from .mixins.policy import PolicyMixin
from .mixins.sam import SamMixin
from .mixins.pipeline.detect import run_detection
from .mixins.pipeline.prepare import parse_config
from .mixins.pipeline.report_bits import build_preview
from .model_cache import get_shared_cache
from .model_managers import YOLOModelManager
from .payload import DetectionFrame, DetectionPayload

try:
    from comfy.model_management import InterruptProcessingException
except ImportError:
    class InterruptProcessingException(Exception):
        pass


@dataclass(slots=True)
class DetectRunCtx(DetectCtx):
    """节点 A 的运行上下文。

    `DetectCtx` 的 41 个槽是**按 Load 集**机检出来的（哪些字段被读到），而
    `run_detection` 还会**写**四个 `DetectCtx` 没声明的逐帧槽。少一个槽的症状
    正是本计划反复猎的那种：`slots=True` 让写未声明属性当场 `AttributeError`，
    但**只在该分支被走到时**才炸。故这里显式补齐，保住那层保护。

    * `preview_image` / `detection_mask` / `_decoded_np` —— `detect.py` 写的逐帧槽
      （`_decoded_np` 是 repair 侧读、detect 侧写，所以它不在 DetectCtx 里）。
    * `interrupted` —— 中断回填那条分支要写它（读它的是节点 D 的 `build_info`）。
    * `sam_model_path` —— ★ 本节点**故意不赋值**：`refine_with_sam` 见它为 None
      就整段跳过，SAM 归节点 B。不是漏搬。
    """

    preview_image: Any = None
    detection_mask: Any = None
    _decoded_np: Any = None
    interrupted: bool = False
    sam_model_path: Optional[str] = None


class _DecodeCapture:
    """临时接管 `node._decode_latent_to_image`，把 `decoded_original` 抄下来。

    `decoded_original` 是 `detect.py` 的**局部变量**（`detect.py:128`），ctx 里没有它，
    也不该有 —— 它是纯中间量。但拆分要把它交给下游：

      * 节点 B 要 `_upscale_image_for_yolo(frame.decoded, min_size=1024)` 喂 SAM；
      * 节点 C 要据它算 MediaPipe 骨架（原 `ctx._decoded_np`）。

    计划 §内存 明写**无条件携带**它、**不许下游重算**：VAE decode 的位级可复现性
    受 tiling / 设备影响，「重算一次」等于拿「精度可能不一致」换内存。

    于是这里用一层透传包装把它抄出来 —— 与 T7 给 `build_report` 套的那层同形：
    先调原件、再抄结果，**不改行为**。`__exit__` 里把实例属性删掉还原到类上的实现，
    实例上不留任何被改过的状态。
    """

    __slots__ = ("_node", "_orig", "decoded")

    def __init__(self, node):
        self._node = node
        self._orig = None
        self.decoded = None

    def __enter__(self):
        self._orig = self._node._decode_latent_to_image

        def _capture(vae, latent_samples):
            out = self._orig(vae, latent_samples)
            self.decoded = out[0]
            return out

        self._node._decode_latent_to_image = _capture
        return self

    def __exit__(self, exc_type, exc, tb):
        del self._node._decode_latent_to_image   # 还原到类上的实现
        return False


def _finalize_interrupt(node, ctx, frames) -> None:
    """中断收尾：未处理帧回填原图（源 `report.py:finalize_interrupt` 的逐字搬迁）。

    单体里这段在 `node.process` 的 `except InterruptProcessingException` 分支里，
    拆分后归**节点 A**（计划 §静默失效 d：「`report.py:22-33` 的中断回填搬到
    节点 A 的异常分支」）。它按 `ctx.out_latents` 的形状回填，而节点 A 的
    `ctx.out_latents` 正是「检测用Latent」的承载张量，语义完全对齐。

    ★ 本路径**不在对拍基准里**（T7 已登记为已知盲区：单体自身在这条路上会崩）。
    """
    ctx.interrupted = True
    print(f"\n{'='*60}")
    print(f"[中断] 用户停止处理")
    print(f"[中断] 已完成: {ctx.processed_count}/{ctx.batch_size} 张")
    print(f"[中断] 返回已处理的结果（未处理的部分保持原图）")
    print(f"{'='*60}")

    if ctx.out_latents is not None:
        for j in range(ctx.processed_count, ctx.batch_size):
            orig_sample = ctx.original_samples[j:j+1]
            if ctx.放大倍数 > 1.0:
                if ctx.放大算法 == "latent_aware" or ctx.cfg.use_latent_aware:
                    target_h = ctx.out_latents.shape[2]
                    target_w = ctx.out_latents.shape[3]
                    orig_sample = LatentAlgorithms.latent_aware_resize(orig_sample, target_h, target_w)
                else:
                    orig_sample = node.upscale_latent(orig_sample, ctx.放大算法, ctx.放大倍数)
            ctx.out_latents[j:j+1] = orig_sample.detach().to(ctx.out_device)
            empty_preview = torch.zeros((1, 64, 64, 3))
            ctx.out_previews.append(empty_preview)
            ctx.out_masks.append(torch.zeros((64, 64)))

    # 补帧必须显式标成 `processed=False`：下游节点看到它就跳过全部分支直接透传
    # （否则下游会对「原图 latent」再做一遍修复）。
    for j in range(ctx.processed_count, ctx.batch_size):
        if j < len(frames):
            frames[j].processed = False


def _cpu_image(decoded):
    """`detect.py:131-132` 对 `decoded_original` 的同一处置（CPU 张量上是恒等）。"""
    if hasattr(decoded, 'is_cuda') and decoded.is_cuda:
        return decoded.cpu()
    return decoded


class QianhuiYoloDetect(DetectionMixin, SamMixin, GeometryMixin, MasksMixin,
                        PolicyMixin, InpaintMixin, LatentKSamplerUpscaler):
    """千绘 · YOLO检测：预降噪 → VAE 解码 → YOLO 检测 → 记录后处理。

    聚合的 Mixin 与单体**同一套**（MRO 也照抄）。这是刻意的：4 个节点共用
    这 6 个 leaf mixin 与 `base_node`，谁也没被裁剪 —— 行为保持的最大保障是
    「数值逻辑零改动」，而裁剪 Mixin 只会制造「某节点少一个方法」的静默风险。
    界面层的隔离靠 `INPUT_TYPES`（61 个控件）与 `DetectRunCtx`（41+5 个槽）。
    """

    @classmethod
    def INPUT_TYPES(cls):
        yolo_models = YOLOModelManager.get_model_list()

        return {
            "required": {
                "Latent": ("LATENT",),
                "模型": ("MODEL",),
                "VAE": ("VAE",),
                "正面提示": ("CONDITIONING",),
                "负面提示": ("CONDITIONING",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "步数": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "CFG": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.5}),
                "采样器": (comfy.samplers.KSampler.SAMPLERS,),
                "调度器": (comfy.samplers.KSampler.SCHEDULERS,),
                "降噪": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01}),
                "放大算法": (["nearest", "bilinear", "bicubic", "area", "latent_aware"],),
                "放大倍数": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 4.0, "step": 0.25}),
            },
            "optional": {
                # ========== ControlNet ==========
                "CN模型": ("CONTROL_NET",),
                "CN图像": ("IMAGE",),
                "CN强度": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "ControlNet强度，建议0.6-0.8（过高会丢失纹理细节）"}),
                "CN开始": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "CN结束": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),

                # ========== V6.6 配置预设 ==========
                "配置预设": ([
                    "[不使用预设]",
                    "快速模式(低质量)",
                    "平衡模式(推荐)",
                    "质量模式(慢速)",
                    "极致质量(超慢)",
                    "省显存模式",
                    "自定义"
                ], {"default": "[不使用预设]", "tooltip": "一键应用预设配置，快速适配不同场景"}),

                # ========== YOLO 核心参数 ==========
                "启用YOLO": ("BOOLEAN", {"default": True}),
                "YOLO模型": (yolo_models if yolo_models else ["[手动输入路径]"], {"default": yolo_models[0] if yolo_models else "[手动输入路径]"}),
                "YOLO路径": ("STRING", {"default": "", "multiline": False, "placeholder": "手动输入模型完整路径"}),
                "置信度": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "YOLO模型1的检测置信度阈值"}),
                "启用多模型": ("BOOLEAN", {"default": False, "tooltip": "启用多模型分批检测，降低显存压力，提高检测准确率"}),
                "YOLO模型2": (["[不使用]"] + (yolo_models if yolo_models else []), {"default": "[不使用]", "tooltip": "多模型模式：第2个检测模型"}),
                "YOLO模型2置信度": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "YOLO模型2的独立置信度（0=使用主置信度）"}),
                "YOLO模型3": (["[不使用]"] + (yolo_models if yolo_models else []), {"default": "[不使用]", "tooltip": "多模型模式：第3个检测模型"}),
                "YOLO模型3置信度": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "YOLO模型3的独立置信度（0=使用主置信度）"}),
                "YOLO模型4": (["[不使用]"] + (yolo_models if yolo_models else []), {"default": "[不使用]", "tooltip": "多模型模式：第4个检测模型"}),
                "YOLO模型4置信度": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "YOLO模型4的独立置信度（0=使用主置信度）"}),
                "目标类别": ("STRING", {"default": "", "multiline": False, "placeholder": "留空=全部 | 支持部分匹配: foot,hand,face"}),
                "YOLO推理尺寸": ("INT", {"default": 768, "min": 320, "max": 1280, "step": 32, "tooltip": "YOLO imgsz参数，提高尺寸可提升小目标检测精度"}),"多尺度检测": ("BOOLEAN", {"default": True, "tooltip": "使用多个尺度检测同一图像，提高准确度（略慢）"}),
                "推理精度": (["FP32(默认)", "FP16(省显存)", "BF16(平衡)"], {"default": "FP32(默认)", "tooltip": "FP16可节省40%显存并提速，BF16在支持的GPU上更稳定"}),

                # ========== V6.7 高级检测优化 ==========
                "使用WIoU优化": ("BOOLEAN", {"default": True, "tooltip": "使用Weighted IoU提高边界框定位精度10-15%"}),
                "WIoU权重参数": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10.0, "step": 0.5, "tooltip": "WIoU聚焦参数，越大越关注困难样本"}),

                # ========== V6.0 核心优化 ==========
                "启用Tile检测": ("BOOLEAN", {"default": True, "tooltip": "大图分块检测，提高召回率"}),
                "Tile大小": ("INT", {"default": 640, "min": 320, "max": 1280, "step": 64}),
                "Tile重叠": ("FLOAT", {"default": 0.3, "min": 0.1, "max": 0.5, "step": 0.05}),
                "使用Seg分割": ("BOOLEAN", {"default": True, "tooltip": "使用YOLO-seg模型的分割mask（需要seg模型）"}),

                # ========== YOLO 多框合并 / 聚类 ==========
                "合并重叠框": ("BOOLEAN", {"default": True}),
                "合并IOU阈值": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 0.95, "step": 0.05}),
                "合并仅同类": ("BOOLEAN", {"default": True}),
                "最多目标数": ("INT", {"default": 0, "min": 0, "max": 200, "step": 1}),

                # ========== YOLO 框后处理 ==========
                "过滤最小面积比": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.5, "step": 0.01}),
                "过滤最大面积比": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 1.0, "step": 0.01}),
                "过滤最小长宽比": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 5.0, "step": 0.05}),
                "过滤最大长宽比": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 50.0, "step": 0.5}),
                "排序方式": (["置信度", "面积"], {"default": "置信度"}),
                "每类最多目标数": ("INT", {"default": 0, "min": 0, "max": 50, "step": 1}),
                "左右分组保护": ("BOOLEAN", {"default": True}),
                "合并后二次扩张像素": ("INT", {"default": 0, "min": 0, "max": 256, "step": 8}),
                "合并后二次扩张比例": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),

                # ========== 优先级 / 规则 / 自适应 ==========
                "修复优先级": (["按置信度", "按面积", "脸优先", "手优先", "脚优先", "自定义优先级"], {"default": "脸优先"}),
                "自定义优先级关键词": ("STRING", {"default": "face,hand,foot", "multiline": False}),
                "类别规则预设": ([
                    "[不使用预设]",
                    "人像优化(face高质量)",
                    "手部优化(hand高降噪)",
                    "脚部优化(foot高降噪)",
                    "人像全面(face+hand+foot)",
                    "自定义规则"
                ], {"default": "[不使用预设]", "tooltip": "快速应用预设规则模板"}),
                "类别规则": ("STRING", {"default": "", "multiline": True, "placeholder": "自定义规则格式:\nface: steps=20,denoise=0.4,limit=4\nhand: denoise=0.75,limit=2"}),

                # ========== 二次检测精修 ==========
                "二次检测精修": ("BOOLEAN", {"default": False}),
                "二次检测最多目标": ("INT", {"default": 6, "min": 1, "max": 50, "step": 1}),
                "二次检测扩展": ("INT", {"default": 32, "min": 0, "max": 256, "step": 8}),

                # ========== V5.1 新增：高级融合选项 ==========
                "使用Latent感知插值": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "QIANHUI_DETECT", "LATENT")
    RETURN_NAMES = ("检测预览", "检测数据", "检测用Latent")
    FUNCTION = "process"
    CATEGORY = "千绘/修复"

    def __init__(self):
        super().__init__()
        # V6.10.5: 使用智能多模型缓存管理器（自动适配显存大小）
        # ★ 4 个节点共用**同一个**实例（计划「静默失效 a」）：
        #   修复节点会伸手进 node.yolo_cache 卸模型整理显存，而加载发生在检测节点。
        #   各持一份的话修复节点卸的是自己那个**空**缓存 —— 不报错，只是显存一直涨。
        self.yolo_cache = get_shared_cache()
        # 兼容旧代码
        self.yolo_model = None
        self.yolo_model_path = None
        self._printed_classes = False

        # V6.10.5: 同步低显存模式状态
        self._low_vram_mode = getattr(self.yolo_cache, 'low_vram_mode', False)

    def process(self, Latent, 模型, VAE, 正面提示, 负面提示, seed, 步数,
                CFG, 采样器, 调度器, 降噪, 放大算法, 放大倍数, **kwargs):
        perf_start = time.time()
        ctx = self._prepare(kwargs, Latent=Latent, VAE=VAE,
                            正面提示=正面提示, 负面提示=负面提示, seed=seed,
                            模型=模型, 步数=步数, CFG=CFG, 采样器=采样器,
                            调度器=调度器, 降噪=降噪, 放大算法=放大算法,
                            放大倍数=放大倍数)

        frames = []
        try:
            for i in range(ctx.batch_size):
                ctx.i = i
                with _DecodeCapture(self) as capture:
                    run_detection(self, ctx)
                ctx.out_previews.append(ctx.preview_image.unsqueeze(0))
                ctx.out_masks.append(ctx.detection_mask)
                # `latent_for_detection` 就是「检测用Latent」：先全图预降噪得到清晰图
                # 再检测，下游修复要在**这张** latent 上做（原单体
                # `run_repair` 第一句 `ctx.processed_latent = ctx.latent_for_detection`）。
                ctx.out_latents[i:i + 1] = ctx.latent_for_detection.detach().to(ctx.out_device)
                frames.append(DetectionFrame(
                    i=i, processed=True, detections=ctx.detections,
                    pixel_h=ctx.pixel_h, pixel_w=ctx.pixel_w,
                    latent_h=ctx.latent_h, latent_w=ctx.latent_w,
                    decoded=_cpu_image(capture.decoded),
                ))
                ctx.processed_count = i + 1
        except InterruptProcessingException:
            for j in range(ctx.processed_count, ctx.batch_size):
                frames.append(DetectionFrame(
                    i=j, processed=False, detections=[],
                    pixel_h=0, pixel_w=0, latent_h=0, latent_w=0,
                ))
            _finalize_interrupt(self, ctx, frames)

        # 承载张量由 `detect.py:41-42` 在第 0 帧分配；第 0 帧就中断时它还是 None，
        # 此时退回原始输入（与 `report.py:build_report` 的兜底同一个形状）。
        detect_latent = (ctx.out_latents if ctx.out_latents is not None
                         else ctx.original_samples.clone())

        payload = DetectionPayload(
            frames=frames, batch_size=ctx.batch_size,
            stage_flags={"enable_yolo": ctx.cfg.enable_yolo,
                         "放大倍数": ctx.放大倍数},
            perf={"detect": time.time() - perf_start},
            interrupted=ctx.interrupted,
        )
        return (build_preview(ctx), payload, {"samples": detect_latent})

    # ------------------------------------------------------------------
    # prepare（prepare_run 里产出本节点所读 ctx 字段的那些行）
    # ------------------------------------------------------------------

    def _prepare(self, kwargs, Latent, VAE, 正面提示, 负面提示, seed, 模型,
                 步数, CFG, 采样器, 调度器, 降噪, 放大算法, 放大倍数):
        cfg = parse_config(kwargs, node_key="detect")
        ctx = DetectRunCtx(cfg=cfg, 模型=模型, VAE=VAE, 正面提示=正面提示,
                           负面提示=负面提示, seed=seed, 步数=步数, CFG=CFG,
                           采样器=采样器, 调度器=调度器, 降噪=降噪,
                           放大算法=放大算法, 放大倍数=放大倍数)

        ctx.has_cn = ctx.cfg.CN模型 is not None and ctx.cfg.CN图像 is not None
        ctx.has_hand_skeleton_cn = ctx.cfg.hand_skeleton_cn_model is not None

        # 如果选择了预设，应用预设规则
        if ctx.cfg.class_rules_preset and ctx.cfg.class_rules_preset != "[不使用预设]" and ctx.cfg.class_rules_preset != "自定义规则":
            ctx.cfg.class_rules_text = self._get_preset_class_rules(ctx.cfg.class_rules_preset)
        ctx.class_rules = self._parse_class_rules(ctx.cfg.class_rules_text)
        ctx.target_classes = self._parse_target_classes(ctx.cfg.target_classes_str)
        ctx.samples = Latent["samples"]
        ctx.batch_size = ctx.samples.shape[0]
        print(f"\n{'='*60}")
        print(f"[千绘·YOLO检测 V6.11.3] 批量: {ctx.batch_size}")
        print(f"  放大: {ctx.放大倍数}x | 基础降噪: {ctx.降噪} | 色差修复: {ctx.cfg.color_strength}")
        print(f"  V6核心: Tile={'ON' if ctx.cfg.tile_enabled else 'OFF'} | Seg={'ON' if ctx.cfg.use_seg else 'OFF'} | 多尺度={'ON' if ctx.cfg.multi_scale else 'OFF'}")
        if ctx.cfg.use_wiou:
            print(f"  ★WIoU优化: 启用 (α={ctx.cfg.wiou_alpha}) | 提升检测框精度10-15%")
        if ctx.has_cn:
            print(f"  ControlNet: 强度={ctx.cfg.CN强度}, 范围=[{ctx.cfg.CN开始}, {ctx.cfg.CN结束}]")

        # 初始化YOLO（V6.9优化：支持每个模型独立置信度）
        ctx.yolo_models_to_use = []
        ctx.yolo_confidences_to_use = []  # V6.9新增：独立置信度列表
        if ctx.cfg.enable_yolo:
            # 首先添加主模型
            resolved_path = self._resolve_model_path(ctx.cfg.yolo_model_selection, ctx.cfg.manual_path)
            if resolved_path and os.path.exists(resolved_path):
                ctx.yolo_models_to_use.append(resolved_path)
                ctx.yolo_confidences_to_use.append(ctx.cfg.confidence)  # 主模型使用主置信度

            # 如果启用多模型，添加额外模型
            if ctx.cfg.enable_multi_model:
                additional_models = [
                    (ctx.cfg.yolo_model2, ctx.cfg.confidence2 if ctx.cfg.confidence2 > 0 else ctx.cfg.confidence),
                    (ctx.cfg.yolo_model3, ctx.cfg.confidence3 if ctx.cfg.confidence3 > 0 else ctx.cfg.confidence),
                    (ctx.cfg.yolo_model4, ctx.cfg.confidence4 if ctx.cfg.confidence4 > 0 else ctx.cfg.confidence),
                ]
                for model_sel, model_conf in additional_models:
                    if model_sel and model_sel != "[不使用]":
                        model_path = YOLOModelManager.get_model_path(model_sel)
                        if model_path and model_path != "MANUAL_INPUT" and os.path.exists(model_path):
                            if model_path not in ctx.yolo_models_to_use:  # 避免重复
                                ctx.yolo_models_to_use.append(model_path)
                                ctx.yolo_confidences_to_use.append(model_conf)  # 使用对应的置信度

            if not ctx.yolo_models_to_use:
                print(f"  [警告] YOLO模型路径无效，禁用YOLO")
                ctx.cfg.enable_yolo = False
            elif len(ctx.yolo_models_to_use) > 1:
                print(f"  [多模型模式] 准备加载 {len(ctx.yolo_models_to_use)} 个YOLO模型:")
                for idx, (mpath, conf) in enumerate(zip(ctx.yolo_models_to_use, ctx.yolo_confidences_to_use), 1):
                    print(f"    [{idx}] {os.path.basename(mpath)} | 置信度={conf:.2f}")
                print(f"  分批检测模式 | 每个模型独立置信度阈值")
            else:
                model_conf = ctx.yolo_confidences_to_use[0] if ctx.yolo_confidences_to_use else ctx.cfg.confidence
                print(f"  YOLO模型: {os.path.basename(ctx.yolo_models_to_use[0])} | 置信度={model_conf:.2f}")

        # 输出缓存设备策略：本节点不暴露 `结果缓存设备` / `低显存模式`，
        # 故原三分支只剩「自动 + 非低显存」这一支（= samples.device）可达。
        ctx.out_device = ctx.samples.device
        ctx.out_latents = None
        ctx.out_previews = []
        ctx.out_masks = []

        # V6.10.8: 重置CUDA损坏标志（每次新执行时重置）
        self._cuda_corrupted = False
        ctx.ksampler = nodes.KSampler()

        # V6.6: 进度条支持
        try:
            from comfy.utils import ProgressBar
            ctx.pbar = ProgressBar(ctx.batch_size)
        except:
            ctx.pbar = None

        # V6.10.7: 中断保护 - 保存中间结果防止黑图
        ctx.interrupted = False
        ctx.processed_count = 0
        # 保存原始输入latent作为fallback
        ctx.original_samples = ctx.samples.clone()

        return ctx


__all__ = ["QianhuiYoloDetect"]
