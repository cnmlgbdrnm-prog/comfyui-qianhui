# -*- coding: utf-8 -*-
"""节点 D 千绘 · 全局降噪（QianhuiGlobalPass）—— 4 节点链的终点。

    [千绘 · YOLO检测] → [千绘 · SAM精修] → [千绘 · 区域修复] → **[千绘 · 全局降噪]**

## 两路输出

    (Latent LATENT, 检测信息 STRING)

* `Latent` = 链条终点的整批结果（`ctx.out_latents`，逐帧写进去）。
* `检测信息` = `report_bits.build_info`。它需要的 4 个**外来**值
  （`融合模式` / `使用Latent感知插值` / `启用YOLO` / `开启裁剪精修`）以及
  `放大倍数` 一律**从 `payload.stage_flags` 快照取**，不复制控件 ——
  这样字符串里的值恒等于各阶段**实际用的**值（计划 §四路输出归属 明文要求）。

## ★ 必须按序遍历

`run_global_pass` 里的视频时序平滑读 `ctx.frame_history` —— 那是唯一跨帧数值状态。
所以本节点**只有** `payload.iter_frames_in_order()` 一个遍历入口，
`payload.frames` 一律不碰（守卫 `test_payload_frames_are_never_iterated_directly` 盯着）。

## 逐帧重绑

`run_global_pass` 结尾会 `del` 掉五个槽
（`preview_image` / `detection_mask` / `detections` / `processed_latent` / `base_latent`），
下一个槽位若不再赋值就是 `AttributeError`。故本节点每帧重绑它们；
其中 `detection_mask` 的值只喂给 `out_masks.append`，而 `out_masks`
在节点 D 里**没有任何读者**（遮罩是节点 C 的第 2 路输出），故用同形状的全零占位。
"""

import time
from dataclasses import dataclass
from typing import Any, List

import comfy.samplers
import nodes
import torch

from .base_node import LatentKSamplerUpscaler
from .mixins.context import GlobalCtx
from .mixins.detection import DetectionMixin
from .mixins.geometry import GeometryMixin
from .mixins.inpaint import InpaintMixin
from .mixins.masks import MasksMixin
from .mixins.pipeline.global_pass import run_global_pass
from .mixins.pipeline.prepare import parse_config
from .mixins.pipeline.report_bits import build_info
from .mixins.policy import PolicyMixin
from .mixins.sam import SamMixin
from .model_cache import get_shared_cache
from .payload import iter_frames_in_order

try:
    from comfy.model_management import InterruptProcessingException
except ImportError:
    class InterruptProcessingException(Exception):
        pass


@dataclass(slots=True)
class GlobalRunCtx(GlobalCtx):
    """节点 D 的运行上下文。

    在 `GlobalCtx` 的 27 个槽之外补 5 个：

    * `samples` —— `prepare_run` 里 `ctx.samples = Latent["samples"]`；本节点需要它
      做两件事：`out_device` 决策的分支、以及最终 `.to(ctx.samples.device)`。
    * `base_latent` —— `run_global_pass` 结尾 `del` 它。删一个**未声明**的槽是
      `AttributeError`；本节点没有它的语义（原单体那是检测阶段的中间量），
      逐帧绑 `None` 即可。
    * `has_cn` / `pixel_h` / `pixel_w` —— 重算 `pos_full` / `neg_full` 要用
      （原单体由 `detect.py:48-51` 算出来）。★ 这不是装饰：`工作模式=修复后全图降噪`
      那条路**就是**用 `ctx.pos_full` 采样的，不重算 CN 会在 CN 工作流里改变数值。
    """

    samples: Any = None
    base_latent: Any = None
    has_cn: bool = False
    pixel_h: int = 0
    pixel_w: int = 0


class QianhuiGlobalPass(DetectionMixin, SamMixin, GeometryMixin, MasksMixin,
                        PolicyMixin, InpaintMixin, LatentKSamplerUpscaler):
    """千绘 · 全局降噪：修复后全图降噪 + 视频时序平滑 + 检测信息汇总。

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
                "结果缓存设备": (["自动", "GPU", "CPU"], {"default": "自动"}),
                "启用性能分析": ("BOOLEAN", {"default": False, "tooltip": "显示各步骤耗时统计，帮助定位性能瓶颈"}),

                # ========== 工作模式 ==========
                "工作模式": (["仅修复(保持输入Latent)", "修复后全图降噪"], {"default": "仅修复(保持输入Latent)"}),
                "全图降噪强度": ("FLOAT", {"default": 0.25, "min": 0.05, "max": 0.5, "step": 0.05, "tooltip": "修复后全图降噪的强度，建议0.2-0.35"}),

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

                # ========== V6.7 视频模式 ==========
                "视频模式": ("BOOLEAN", {"default": False, "tooltip": "启用视频模式，优化帧间一致性"}),
                "时序一致性强度": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "帧间平滑强度，越大越平滑但可能损失细节"}),
                "参考帧窗口": ("INT", {"default": 2, "min": 1, "max": 5, "step": 1, "tooltip": "用于时序一致性的参考帧数量"}),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("Latent", "检测信息")
    FUNCTION = "process"
    CATEGORY = "千绘/修复"

    def __init__(self):
        super().__init__()
        # ★ 与其余三个节点共用**同一个**缓存实例（计划「静默失效 a」）。
        self.yolo_cache = get_shared_cache()
        # 兼容旧代码
        self.yolo_model = None
        self.yolo_model_path = None
        self._printed_classes = False

        # V6.10.5: 同步低显存模式状态
        self._low_vram_mode = getattr(self.yolo_cache, 'low_vram_mode', False)

    def process(self, Latent, 检测数据, 模型, 正面提示, 负面提示, seed, 步数,
                CFG, 采样器, 调度器, **kwargs):
        perf_start = time.time()
        ctx = self._prepare(kwargs, 检测数据=检测数据, Latent=Latent, 模型=模型,
                            seed=seed, 步数=步数, CFG=CFG, 采样器=采样器,
                            调度器=调度器)

        try:
            for frame in iter_frames_in_order(检测数据):
                ctx.i = frame.i
                ctx.processed_latent = Latent["samples"][frame.i:frame.i + 1]
                if not frame.processed:
                    # 中断补帧：**跳过全部分支直接透传**（计划 §R1）。
                    # ★ 这里**不推进** `ctx.processed_count`：它的语义是「**已完成**的
                    #   帧数」（单体由 `run_global_pass` 结尾 `ctx.processed_count =
                    #   ctx.i + 1` 推进，`finalize_interrupt` 不碰它）。透传帧没被处理，
                    #   推进它会让 `build_info` 报出「[已中断] 处理: 3/3 张」——
                    #   与同一段文字里的「未处理的图像已使用原图填充」自相矛盾
                    #   （复审 I-1；判据见 e2e 的 `TestInterruptAtFirstFrame`）。
                    ctx.out_latents[frame.i:frame.i + 1] = ctx.processed_latent.detach().to(ctx.out_device)
                    continue

                ctx.detections = frame.detections
                ctx.pixel_h, ctx.pixel_w = frame.pixel_h, frame.pixel_w
                ctx.pos_full, ctx.neg_full = 正面提示, 负面提示
                if ctx.has_cn:
                    ctx.pos_full, ctx.neg_full = self._maybe_apply_controlnet(
                        ctx.pos_full, ctx.neg_full, True, ctx.cfg.CN模型, ctx.cfg.CN图像,
                        ctx.cfg.CN强度, ctx.cfg.CN开始, ctx.cfg.CN结束,
                        ctx.pixel_h, ctx.pixel_w, ctx.i
                    )
                # `run_global_pass` 结尾会把这两个槽 del 掉，必须逐帧重绑。
                # `preview_image` 只喂给那句 `out_previews.append`（D 的 out_previews
                # 无读者）；`detection_mask` 同理（遮罩是节点 C 的输出）。
                ctx.preview_image = frame.decoded
                ctx.detection_mask = torch.zeros((frame.pixel_h, frame.pixel_w), dtype=torch.float32)
                ctx.base_latent = None

                run_global_pass(self, ctx)
        except InterruptProcessingException:
            ctx.interrupted = True
            print(f"\n{'='*60}\n[中断] 全局降噪停止\n{'='*60}")
            for j in range(ctx.processed_count, ctx.batch_size):
                ctx.out_latents[j:j + 1] = Latent["samples"][j:j + 1].detach().to(ctx.out_device)

        final_samples = ctx.out_latents.to(ctx.samples.device) if ctx.out_device != ctx.samples.device else ctx.out_latents
        info = build_info(ctx, final_samples)

        # ★ 终审 M-4：brief §七 要求「4 个节点**无条件**把自身耗时写进
        #   `payload.perf[stage]`」，而 A/B/C 都写了、D 一条没写（规格未履行一半）。
        #   D 不产出 payload（它返回 `(Latent, 检测信息)`），但手里就有上游那份
        #   —— A/B/C 也正是就地往同一份里写，所以这里同样往**收到的那份**里补一笔。
        #   当前全仓没有读者读 `payload.perf`（终审 grep 过）；补它是为了让规格成真、
        #   让将来加读者时 4 个阶段都有数据。
        检测数据.perf["global"] = time.time() - perf_start

        print(f"\n{'='*60}\n{info}\n{'='*60}\n")
        return ({"samples": final_samples}, info)

    # ------------------------------------------------------------------
    # prepare（prepare_run 里产出本节点所读 ctx 字段的那些行）
    # ------------------------------------------------------------------

    def _prepare(self, kwargs, 检测数据, Latent, 模型, seed, 步数, CFG, 采样器, 调度器):
        cfg = parse_config(kwargs, node_key="global")
        ctx = GlobalRunCtx(cfg=cfg, 模型=模型, seed=seed, 步数=步数, CFG=CFG,
                           采样器=采样器, 调度器=调度器)

        # V6.6: 性能分析器
        ctx.perf_start_time = time.time() if ctx.cfg.enable_profiling else None
        ctx.has_cn = ctx.cfg.CN模型 is not None and ctx.cfg.CN图像 is not None

        ctx.samples = Latent["samples"]
        ctx.batch_size = ctx.samples.shape[0]

        # ★ 4 个外来值 + 放大倍数一律从 payload 快照取，不复制控件：
        #   `build_info` 复述的必须是各阶段**实际用的**值（计划 §四路输出归属）。
        flags = 检测数据.stage_flags or {}
        ctx.cfg.blend_mode = flags.get("blend_mode", ctx.cfg.blend_mode)
        ctx.cfg.use_latent_aware = flags.get("use_latent_aware", ctx.cfg.use_latent_aware)
        ctx.cfg.enable_yolo = flags.get("enable_yolo", ctx.cfg.enable_yolo)
        ctx.cfg.enable_crop = flags.get("enable_crop", ctx.cfg.enable_crop)
        ctx.放大倍数 = flags.get("放大倍数", ctx.放大倍数)

        print(f"\n{'='*60}")
        print(f"[千绘·全局降噪 V6.11.3] 批量: {ctx.batch_size}")
        print(f"  工作模式: {ctx.cfg.work_mode}" + (f" (降噪={ctx.cfg.final_denoise_strength})" if ctx.cfg.work_mode == "修复后全图降噪" else ""))
        if ctx.cfg.video_mode:
            print(f"  ★视频模式: 时序平滑={ctx.cfg.temporal_strength} | 参考帧={ctx.cfg.frame_window} | 帧间一致性优化")
        if ctx.has_cn:
            print(f"  ControlNet: 强度={ctx.cfg.CN强度}, 范围=[{ctx.cfg.CN开始}, {ctx.cfg.CN结束}]")

        # 输出缓存设备策略
        if ctx.cfg.cache_device_mode == "CPU":
            ctx.out_device = torch.device("cpu")
        elif ctx.cfg.cache_device_mode == "GPU":
            ctx.out_device = ctx.samples.device
        else:
            ctx.out_device = torch.device("cpu") if ctx.cfg.low_vram else ctx.samples.device
        # 承载张量：形状/类型照抄 `detect.py:41-42` 的分配式，只是尺寸来自本节点收到的
        # 那一批（节点 C 的修复结果与检测用 Latent 同形状）。
        ctx.out_latents = torch.empty(
            (ctx.batch_size, ctx.samples.shape[1], ctx.samples.shape[2], ctx.samples.shape[3]),
            device=ctx.out_device, dtype=ctx.samples.dtype)
        ctx.out_previews = []
        ctx.out_masks = []
        ctx.all_dets = []

        ctx.ksampler = nodes.KSampler()

        # ★ 这里**没有** `ctx.pbar`：进度条只有检测阶段在推（`detect.py:17`），
        #   而 `GlobalCtx` 的 27 个槽里没有 `pbar`（它是按 Load 集机检出来的）。
        #   照搬 `prepare_run:279-283` 会当场 `AttributeError` —— 这正是
        #   `slots=True` 应当有的行为，不是缺字段。

        # V6.7: 视频模式帧缓存 —— ★ 唯一跨帧数值状态，必须由本节点本地维护
        ctx.frame_history = [] if ctx.cfg.video_mode else None

        # V6.10.7: 中断保护
        ctx.interrupted = bool(检测数据.interrupted)
        ctx.processed_count = 0

        return ctx


__all__ = ["QianhuiGlobalPass"]
