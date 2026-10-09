# -*- coding: utf-8 -*-
"""节点 C 千绘 · 区域修复（QianhuiRegionRepair）—— 4 节点链的第三环。

    [千绘 · YOLO检测] → [千绘 · SAM精修] → **[千绘 · 区域修复]** → [千绘 · 全局降噪]

## 两路输出

    (修复后Latent LATENT, 检测遮罩 MASK)

* `修复后Latent` = 逐帧 `ctx.processed_latent` 的整批。原单体里它由
  `run_repair` 产出、随后被 `run_global_pass` 写进 `ctx.out_latents`
  （那一句在节点 D）。本节点整批 `torch.cat` 出来交给 D。
* `检测遮罩` = `report_bits.build_masks` over 逐帧 `ctx.detection_mask`。
  遮罩**必须**由本节点产出：它的并集是 `repair.py:33` 建的，而 expand/feather
  又来自 policy —— 逐目标算完再合成，合成之后信息就丢了（羽化不满足交换律）。

## 逐帧重绑（「静默失效 b」的处置）

单体里 `ctx.detections` 是单帧槽且每帧被 `del`；`slots=True` 下「读到上一帧残值」
不会报错。所以本节点**每帧重新绑定**它要用的全部逐帧槽：

    i / latent_for_detection / detections / pixel_h / pixel_w /
    latent_h / latent_w / base_pos / base_neg / pos_full / neg_full /
    detection_mask / _decoded_np

其中 `detection_mask` 的重绑是**必须**的：单体由 `detect.py:105`
每帧铺一张全零 mask，`repair.py` 只在「有检测」时才覆盖它；没有这一句，
无检测帧会读到上一帧的旧遮罩（第 3 路输出就变了）。

## 为什么 `检测数据` 与 `Latent` 是两个独立输入口

* `Latent` 接的是**节点 A 的「检测用Latent」**（不是用户输入的那个）：
  单体 `run_repair` 第一句就是 `ctx.processed_latent = ctx.latent_for_detection`。
* `检测数据` 接的是**节点 B** 的透传输出（B 原地更新了检测记录）。
  本节点的 `RETURN_TYPES` 没有 `QIANHUI_DETECT`（计划名册如此），
  所以节点 D 的 `检测数据` 也从 **B** 那条线分叉过去。
"""

import time

import comfy.samplers
import nodes
import torch

from .base_node import LatentKSamplerUpscaler
from .mixins.context import RepairCtx
from .mixins.detection import DetectionMixin
from .mixins.geometry import GeometryMixin
from .mixins.inpaint import InpaintMixin
from .mixins.masks import MasksMixin
from .mixins.pipeline.prepare import parse_config
from .mixins.pipeline.repair import run_repair
from .mixins.pipeline.report_bits import build_masks
from .mixins.policy import PolicyMixin
from .mixins.sam import SamMixin
from .model_cache import get_shared_cache
from .payload import iter_frames_in_order

try:
    from comfy.model_management import InterruptProcessingException
except ImportError:
    class InterruptProcessingException(Exception):
        pass


class QianhuiRegionRepair(DetectionMixin, SamMixin, GeometryMixin, MasksMixin,
                          PolicyMixin, InpaintMixin, LatentKSamplerUpscaler):
    """千绘 · 区域修复：逐目标裁剪 → 条件编码 → 局部重绘 → 融合贴回。

    直接用 `RepairCtx`（28 个槽），**不派生** —— 机检确认 `repair.py` 的
    Load ∪ Store 集恰好等于那 28 个，一个不缺、一个不多。

    聚合的 Mixin 与单体**同一套**（理由见 `node_detect.py` 的类 docstring）。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Latent": ("LATENT",),
                "检测数据": ("QIANHUI_DETECT",),
                "模型": ("MODEL",),
                "正面提示": ("CONDITIONING",),
                "负面提示": ("CONDITIONING",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "步数": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "CFG": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.5}),
                "采样器": (comfy.samplers.KSampler.SAMPLERS,),
                "调度器": (comfy.samplers.KSampler.SCHEDULERS,),
                "降噪": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                # ========== ControlNet ==========
                "CN模型": ("CONTROL_NET",),
                "CN图像": ("IMAGE",),
                "CN强度": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "ControlNet强度，建议0.6-0.8（过高会丢失纹理细节）"}),
                "CN开始": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "CN结束": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),

                # ========== 性能/低显存 ==========
                "低显存模式": ("BOOLEAN", {"default": False}),
                "每帧清理缓存": ("BOOLEAN", {"default": False}),

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

                # ========== V6.7 EMA 注意力增强 ==========
                "启用EMA注意力": ("BOOLEAN", {"default": False, "tooltip": "启用多尺度注意力机制，聚焦关键区域"}),
                "注意力聚焦强度": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 3.0, "step": 0.1, "tooltip": "注意力聚焦系数，越大越集中于高置信度区域"}),

                # ========== V6.0 核心优化 ==========
                "使用Seg分割": ("BOOLEAN", {"default": True, "tooltip": "使用YOLO-seg模型的分割mask（需要seg模型）"}),
                "Seg边缘羽化": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.5}),

                # ========== 智能策略 ==========
                "智能策略": (["关闭", "人像(脸/手/脚)", "通用(按面积)"], {"default": "人像(脸/手/脚)"}),

                # ========== V6.10手脚增强（高降噪+扩展参数+多轮修复） ==========
                "手脚增强模式": ("BOOLEAN", {"default": True, "tooltip": "手脚区域自动提高降噪、扩展mask、增加padding/steps（建议在prompt中添加'perfect hands, detailed fingers'以获得更好效果）"}),
                "手脚降噪下限": ("FLOAT", {"default": 0.80, "min": 0.5, "max": 1.0, "step": 0.05, "tooltip": "手脚区域的最低降噪值（V6.10提升至0.80，参考HandRefiner最佳实践）"}),
                "手脚额外轮次": ("INT", {"default": 1, "min": 0, "max": 3, "step": 1, "tooltip": "手脚区域额外修复轮次"}),
                # V6.11 新增：手脚优化进阶参数（基于HandRefiner最佳实践）
                "手脚CFG加成": ("FLOAT", {"default": 1.2, "min": 1.0, "max": 2.0, "step": 0.1, "tooltip": "手脚区域CFG增强倍数（1.2-1.5效果最佳，过高会导致过度饱和）"}),
                "手脚CN强度": ("FLOAT", {"default": 0.7, "min": 0.3, "max": 1.0, "step": 0.05, "tooltip": "手脚区域ControlNet强度（0.5-0.8效果最佳，1.0可能丢失自然纹理）"}),
                "手脚降噪衰减": ("FLOAT", {"default": 0.85, "min": 0.5, "max": 1.0, "step": 0.05, "tooltip": "多轮修复时降噪的衰减系数（每轮 denoise *= 衰减系数）"}),
                "手脚衰减下限": ("FLOAT", {"default": 0.40, "min": 0.2, "max": 0.8, "step": 0.05, "tooltip": "多轮修复时降噪的最低值（防止后续轮次降噪过低）"}),

                # ========== V6.11.3 手骨架ControlNet（内置骨架引导，避免多指）==========
                "手骨架CN模型": ("CONTROL_NET", {"tooltip": "手部骨架专用ControlNet模型（推荐: control_openpose / control_dw_pose）。提供此模型后，修手时自动用MediaPipe提取骨架注入，避免生成多余手指"}),
                "骨架CN强度": ("FLOAT", {"default": 0.65, "min": 0.1, "max": 1.5, "step": 0.05, "tooltip": "手骨架ControlNet引导强度（0.5-0.75通常效果最好）"}),
                "骨架CN结束": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "骨架CN在扩散过程的结束步比例（0.7=前70%步有骨架约束，后30%自由细化）"}),

                # ========== 优先级 / 规则 / 自适应 ==========
                "按置信度自适应降噪": ("BOOLEAN", {"default": True}),
                "自适应降噪_低置信缩放": ("FLOAT", {"default": 0.85, "min": 0.1, "max": 1.5, "step": 0.05}),
                "自适应降噪_高置信缩放": ("FLOAT", {"default": 1.00, "min": 0.1, "max": 1.5, "step": 0.05}),
                "类别规则预设": ([
                    "[不使用预设]",
                    "人像优化(face高质量)",
                    "手部优化(hand高降噪)",
                    "脚部优化(foot高降噪)",
                    "人像全面(face+hand+foot)",
                    "自定义规则"
                ], {"default": "[不使用预设]", "tooltip": "快速应用预设规则模板"}),
                "类别规则": ("STRING", {"default": "", "multiline": True, "placeholder": "自定义规则格式:\nface: steps=20,denoise=0.4,limit=4\nhand: denoise=0.75,limit=2"}),

                # ========== 遮罩参数 ==========
                "遮罩扩展": ("INT", {"default": 32, "min": 0, "max": 256, "step": 8}),
                "遮罩羽化": ("INT", {"default": 24, "min": 0, "max": 128, "step": 4, "tooltip": "增加羽化可让边缘更柔和自然，建议20-32"}),

                # ========== 修复采样参数 ==========
                "区域步数": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "区域CFG": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.5}),
                "区域降噪": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "提高降噪强度可获得更彻底的修复效果"}),
                "修复轮次": ("INT", {"default": 1, "min": 1, "max": 6, "step": 1}),
                "逐次降噪递减": ("BOOLEAN", {"default": True}),
                "未检测时处理": (["保持原样", "全图重绘"], {"default": "保持原样"}),

                # ========== 裁剪精修 ==========
                "开启裁剪精修": ("BOOLEAN", {"default": True}),
                "裁剪扩展": ("INT", {"default": 64, "min": 0, "max": 512, "step": 8}),
                "裁剪最大边": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 64}),
                "裁剪放大倍率": ("FLOAT", {"default": 1.6, "min": 1.0, "max": 4.0, "step": 0.1}),
                "裁剪步数": ("INT", {"default": 20, "min": 1, "max": 200, "step": 1, "tooltip": "增加步数可提升修复质量，推荐18-25步"}),
                "裁剪CFG": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.5}),
                "裁剪降噪": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "提高降噪可获得更高质量的修复，推荐0.55-0.70"}),
                "裁剪轮次": ("INT", {"default": 2, "min": 1, "max": 4, "step": 1}),
                "裁剪使用ControlNet": ("BOOLEAN", {"default": True}),

                # ========== V5.1 新增：高级融合选项 ==========
                "融合模式": (["Poisson融合", "MPB增强融合", "频率感知", "简单混合", "智能柔和融合"], {"default": "智能柔和融合", "tooltip": "智能柔和融合：多层次渐进羽化+色彩对齐+细节恢复，柔化色差和朦胧感"}),
                "Poisson迭代次数": ("INT", {"default": 40, "min": 0, "max": 100, "step": 5}),
                "MPB颜色保持": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "MPB模式下的颜色保持强度，0=完全融合，1=完全保持"}),
                "色差修复强度": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "提高色差修复强度可更好地平衡修复区域和原始区域的色彩"}),
                "清晰度恢复强度": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "修复后的细节和清晰度恢复强度，可减少朦胧感"}),
                "使用Latent感知插值": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("LATENT", "MASK")
    RETURN_NAMES = ("修复后Latent", "检测遮罩")
    FUNCTION = "process"
    CATEGORY = "千绘/修复"

    def __init__(self):
        super().__init__()
        # ★ 与其余三个节点共用**同一个**缓存实例（计划「静默失效 a」）。
        #   本节点的 `run_repair` 会伸手清空它整理显存；各持一份的话
        #   检测节点加载的模型再没人卸 —— 不报错，只是显存一直涨。
        self.yolo_cache = get_shared_cache()
        # 兼容旧代码
        self.yolo_model = None
        self.yolo_model_path = None
        self._printed_classes = False

        # V6.10.5: 同步低显存模式状态
        self._low_vram_mode = getattr(self.yolo_cache, 'low_vram_mode', False)

    def process(self, Latent, 检测数据, 模型, 正面提示, 负面提示, seed, 步数,
                CFG, 采样器, 调度器, 降噪, **kwargs):
        perf_start = time.time()
        ctx = self._prepare(kwargs, 检测数据=检测数据, 模型=模型, seed=seed,
                            步数=步数, CFG=CFG, 采样器=采样器, 调度器=调度器,
                            降噪=降噪)

        repaired = []
        try:
            for frame in iter_frames_in_order(检测数据):
                ctx.i = frame.i
                frame_latent = Latent["samples"][frame.i:frame.i + 1]
                if not frame.processed:
                    # 中断补帧：**跳过全部分支直接透传**（计划 §R1）。
                    repaired.append(frame_latent)
                    ctx.out_masks.append(torch.zeros((64, 64)))
                    continue

                ctx.latent_for_detection = frame_latent
                ctx.detections = frame.detections
                ctx.pixel_h, ctx.pixel_w = frame.pixel_h, frame.pixel_w
                ctx.latent_h, ctx.latent_w = frame.latent_h, frame.latent_w
                ctx.base_pos, ctx.base_neg = 正面提示, 负面提示
                ctx.pos_full, ctx.neg_full = ctx.base_pos, ctx.base_neg
                if ctx.has_cn:
                    ctx.pos_full, ctx.neg_full = self._maybe_apply_controlnet(
                        ctx.base_pos, ctx.base_neg, True, ctx.cfg.CN模型, ctx.cfg.CN图像,
                        ctx.cfg.CN强度, ctx.cfg.CN开始, ctx.cfg.CN结束,
                        ctx.pixel_h, ctx.pixel_w, ctx.i
                    )
                # 单体里由 `detect.py:105` 每帧铺的全零 mask；`repair.py` 只在
                # 「有检测」时才覆盖它。不重绑的话无检测帧会读到上一帧的旧遮罩。
                ctx.detection_mask = torch.zeros((ctx.pixel_h, ctx.pixel_w), dtype=torch.float32)
                if ctx.has_hand_skeleton_cn:
                    ctx._decoded_np = (frame.decoded.numpy()
                                       if isinstance(frame.decoded, torch.Tensor)
                                       else frame.decoded)
                else:
                    ctx._decoded_np = None

                run_repair(self, ctx)
                ctx.out_masks.append(ctx.detection_mask)
                repaired.append(ctx.processed_latent)
        except InterruptProcessingException:
            # 与节点 A 同一处置：剩下的帧原样透传（`processed=False` 已由 A 标好）。
            print(f"\n{'='*60}\n[中断] 区域修复停止\n{'='*60}")
            while len(repaired) < ctx.batch_size:
                j = len(repaired)
                repaired.append(Latent["samples"][j:j + 1])
                ctx.out_masks.append(torch.zeros((64, 64)))

        payload = 检测数据
        # 节点 D 的「检测信息」要复述这几个值，而它在自己那份 cfg 里读不到
        # （D 不暴露这几个控件）—— 故由**真正用了它们的本节点**记进快照。
        payload.stage_flags["blend_mode"] = ctx.cfg.blend_mode
        payload.stage_flags["use_latent_aware"] = ctx.cfg.use_latent_aware
        payload.stage_flags["enable_crop"] = ctx.cfg.enable_crop
        payload.perf["repair"] = time.time() - perf_start
        return ({"samples": torch.cat(repaired, dim=0)}, build_masks(ctx))

    # ------------------------------------------------------------------
    # prepare（prepare_run 里产出本节点所读 ctx 字段的那些行）
    # ------------------------------------------------------------------

    def _prepare(self, kwargs, 检测数据, 模型, seed, 步数, CFG, 采样器, 调度器, 降噪):
        cfg = parse_config(kwargs, node_key="repair")
        ctx = RepairCtx(cfg=cfg, 模型=模型, seed=seed, 步数=步数, CFG=CFG,
                        采样器=采样器, 调度器=调度器, 降噪=降噪)

        ctx.has_cn = ctx.cfg.CN模型 is not None and ctx.cfg.CN图像 is not None
        ctx.has_hand_skeleton_cn = ctx.cfg.hand_skeleton_cn_model is not None

        # 如果选择了预设，应用预设规则
        if ctx.cfg.class_rules_preset and ctx.cfg.class_rules_preset != "[不使用预设]" and ctx.cfg.class_rules_preset != "自定义规则":
            ctx.cfg.class_rules_text = self._get_preset_class_rules(ctx.cfg.class_rules_preset)
        ctx.class_rules = self._parse_class_rules(ctx.cfg.class_rules_text)

        ctx.batch_size = 检测数据.batch_size
        print(f"\n{'='*60}")
        print(f"[千绘·区域修复 V6.11.3] 批量: {ctx.batch_size}")
        print(f"  融合模式: {ctx.cfg.blend_mode} | Latent感知插值: {'ON' if ctx.cfg.use_latent_aware else 'OFF'} | 裁剪精修: {'ON' if ctx.cfg.enable_crop else 'OFF'}")
        if ctx.cfg.hand_foot_enhance:
            print(f"  ★手脚增强: 降噪≥{ctx.cfg.hand_foot_denoise}, +{ctx.cfg.hand_foot_extra_rounds}轮, CFG×{ctx.cfg.hand_foot_cfg_boost}, CN={ctx.cfg.hand_foot_cn_strength}")
        if ctx.has_hand_skeleton_cn:
            print(f"  ★手骨架CN: 强度={ctx.cfg.skeleton_cn_strength}, 结束={ctx.cfg.skeleton_cn_end} (MediaPipe自动提取骨架)")

        ctx.out_masks = []

        # V6.10.8: 重置CUDA损坏标志（每次新执行时重置）
        # 计划明文：它留在区域修复节点（读它的只有 `_inpaint_sample`，
        # 而调用点只有 repair.py 的两处）。不要为它新增跨节点抑制机制。
        self._cuda_corrupted = False
        ctx.ksampler = nodes.KSampler()

        return ctx


__all__ = ["QianhuiRegionRepair"]
