# -*- coding: utf-8 -*-
"""节点 B 千绘 · SAM精修（QianhuiSamRefine）—— 4 节点链的第二环。

    [千绘 · YOLO检测] → **[千绘 · SAM精修]** → [千绘 · 区域修复] → [千绘 · 全局降噪]

## 本节点做什么

原单体里「SAM 精修」那一整块（`detect.py:366-469`，T4 抽成
`mixins/pipeline/sam_refine.py:refine_with_sam`）**整段**归本节点。
本节点把它从「帧内」挪到「帧外」：逐帧取出检测记录、重算送进 SAM 的那张图、
原地更新记录，然后把同一个 payload 交下去。

## ★ 坐标口径照抄，不要「顺手修」

`detect.py:398` 送进 SAM 的是 **det 空间**的图（不是像素空间）：那是
`_upscale_image_for_yolo(decoded_original, min_size=1024)` 的输出。
本节点据此**重算**同一张图（`frame.decoded` 是节点 A 抄下来的 `decoded_original`，
`_upscale_image_for_yolo` 是确定性函数），再喂 `_refine_with_sam`。

为什么是「重算」而不是「把图也搬过来」：那张图是 1024² 三通道浮点，比
`decoded` 大十几倍，而重算是**纯函数**（同一输入同一输出）。计划 §内存 只要求
无条件携带 `decoded`，没要求携带放大后的副本。

## 签名口径（对 4 个节点都成立）

**只有必填 socket 做具名形参，其余控件一律走 `**kwargs`** —— 这不是风格：
`parse_config(kwargs)` 是从 `kwargs` 里 `.get` 那 99 个旋钮的。把可选控件写成
具名形参会让它们**进不了 kwargs**，于是 `cfg` 里恒为声明默认值 —— 界面上旋钮还在、
改了没用、不报错。
"""

import os
import time
from dataclasses import dataclass

from .base_node import LatentKSamplerUpscaler
from .mixins.context import SamCtx
from .mixins.detection import DetectionMixin
from .mixins.geometry import GeometryMixin
from .mixins.inpaint import InpaintMixin
from .mixins.masks import MasksMixin
from .mixins.policy import PolicyMixin
from .mixins.pipeline.prepare import parse_config
from .mixins.pipeline.sam_refine import refine_with_sam
from .mixins.sam import SamMixin
from .model_cache import get_shared_cache
from .model_managers import SAMModelManager
from .payload import iter_frames_in_order


@dataclass(slots=True)
class SamRunCtx(SamCtx):
    """节点 B 的运行上下文 —— 与 `SamCtx` 的 3 个槽**完全一致**，不增不减。

    `refine_with_sam` 只读 `ctx.cfg.enable_sam` / `ctx.cfg.sam_conf_threshold` /
    `ctx.sam_model_path` / `ctx.模型`，四个都在。所以这是本仓唯一一个
    「精简 Context 直接够用」的节点 —— 不需要派生。
    """


class QianhuiSamRefine(DetectionMixin, SamMixin, GeometryMixin, MasksMixin,
                       PolicyMixin, InpaintMixin, LatentKSamplerUpscaler):
    """千绘 · SAM精修：用 SAM 把 YOLO 的检测框细化为精确分割 mask。

    聚合的 Mixin 与单体**同一套**（理由见 `node_detect.py` 的类 docstring）。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "检测数据": ("QIANHUI_DETECT",),
                "模型": ("MODEL",),
            },
            "optional": {
                "配置预设": ([
                    "[不使用预设]",
                    "快速模式(低质量)",
                    "平衡模式(推荐)",
                    "质量模式(慢速)",
                    "极致质量(超慢)",
                    "省显存模式",
                    "自定义"
                ], {"default": "[不使用预设]", "tooltip": "一键应用预设配置，快速适配不同场景"}),

                # ========== V6.7 高级检测优化 ==========
                "启用SAM精细化": ("BOOLEAN", {"default": False, "tooltip": "使用SAM模型精细化分割（支持SAM/SAM2/SAM2.1）"}),
                "SAM模型选择": (SAMModelManager.get_model_list(), {"default": SAMModelManager.get_model_list()[0], "tooltip": "选择SAM模型（支持v1/v2/v2.1，自动检测版本）"}),
                "SAM模型手动路径": ("STRING", {"default": "", "tooltip": "手动输入SAM模型路径（选择[手动输入路径]时使用）"}),
                "SAM置信度阈值": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "SAM分割置信度阈值，提高可获得更精确的分割"}),
            }
        }

    RETURN_TYPES = ("QIANHUI_DETECT",)
    RETURN_NAMES = ("检测数据",)
    FUNCTION = "process"
    CATEGORY = "千绘/修复"

    def __init__(self):
        super().__init__()
        # ★ 与其余三个节点共用**同一个**缓存实例（计划「静默失效 a」）。
        #   `refine_with_sam` 会伸手清空它来给 SAM 腾显存；各持一份的话
        #   检测节点加载的模型再没人卸 —— 不报错，只是显存一直涨。
        self.yolo_cache = get_shared_cache()
        # 兼容旧代码
        self.yolo_model = None
        self.yolo_model_path = None
        self._printed_classes = False

        # V6.10.5: 同步低显存模式状态
        self._low_vram_mode = getattr(self.yolo_cache, 'low_vram_mode', False)

    def process(self, 检测数据, 模型, **kwargs):
        perf_start = time.time()
        cfg = parse_config(kwargs, node_key="sam")
        ctx = SamRunCtx(cfg=cfg, 模型=模型)
        ctx.sam_model_path = self._resolve_sam_model_path(
            cfg.sam_model_selection, cfg.sam_model_manual)

        if ctx.cfg.enable_sam:
            sam_model_name = os.path.basename(ctx.sam_model_path) if ctx.sam_model_path else "[未设置]"
            print(f"  ★SAM精细化: {sam_model_name} | 阈值={ctx.cfg.sam_conf_threshold} | 超高精度分割")

        # 与 `refine_with_sam` 的第一道闸门同一个条件：不成立时它原样返回 dets，
        # 故这里直接短路 —— 省掉每帧一次没用的 cv2 上采样，**结果逐位相同**。
        if ctx.cfg.enable_sam and ctx.sam_model_path:
            for frame in iter_frames_in_order(检测数据):
                if not frame.processed:
                    # 中断补帧：**跳过全部分支直接透传**（计划 §R1）。
                    continue
                # 与 `detect.py:398` 送进 SAM 的图同一张（det 空间，min_size=1024）。
                decoded_det = self._upscale_image_for_yolo(frame.decoded, min_size=1024)
                frame.detections = refine_with_sam(self, ctx, frame.detections, decoded_det)

        检测数据.perf["sam"] = time.time() - perf_start
        return (检测数据,)


__all__ = ["QianhuiSamRefine"]
