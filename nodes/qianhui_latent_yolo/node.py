"""千绘 · Latent 智能修复节点（QianhuiLatentYOLO）。

由 6 个 Mixin 聚合而成，MRO 顺序：
Detection → Sam → Geometry → Masks → Policy → Inpaint → LatentKSamplerUpscaler
基类置于末位以保证 upscale_latent 可解析。
"""
import os
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
import folder_paths
import comfy.samplers
import comfy.sample
import comfy.utils
import comfy.model_management
import comfy.controlnet
import nodes  # ComfyUI 根目录的 nodes.py，取 nodes.KSampler()

from .algorithms import LatentAlgorithms
from .model_managers import YOLOModelManager, SAMModelManager
from .model_cache import YOLOModelCache
from .controlnet import ControlNetHelper
from .base_node import LatentKSamplerUpscaler
from .mixins.detection import DetectionMixin
from .mixins.sam import SamMixin
from .mixins.geometry import GeometryMixin
from .mixins.masks import MasksMixin
from .mixins.policy import PolicyMixin
from .mixins.inpaint import InpaintMixin
from .mixins.pipeline.prepare import prepare_run
from .mixins.pipeline.detect import run_detection
from .mixins.pipeline.repair import run_repair
from .mixins.pipeline.global_pass import run_global_pass
from .mixins.pipeline.report import build_report, finalize_interrupt

try:
    from comfy.model_management import InterruptProcessingException
except ImportError:
    class InterruptProcessingException(Exception):
        pass


class QianhuiLatentYOLO(DetectionMixin, SamMixin, GeometryMixin, MasksMixin,
                       PolicyMixin, InpaintMixin, LatentKSamplerUpscaler):
    """
    Latent空间K采样器与放大 - 支持YOLO智能检测+修复 + ControlNet（极致优化版）

    V5.1核心优化：
    - Latent感知插值（替代bilinear，保留语义结构）
    - 真高斯羽化（替代avg_pool，边缘更自然）
    - 频率感知色差修复（只调低频，保留细节）
    - Poisson梯度域融合（无缝贴回，消除接缝）
    - 多轮精修循环BUG修复
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
                "启用SAM精细化": ("BOOLEAN", {"default": False, "tooltip": "使用SAM模型精细化分割（支持SAM/SAM2/SAM2.1）"}),
                "SAM模型选择": (SAMModelManager.get_model_list(), {"default": SAMModelManager.get_model_list()[0], "tooltip": "选择SAM模型（支持v1/v2/v2.1，自动检测版本）"}),
                "SAM模型手动路径": ("STRING", {"default": "", "tooltip": "手动输入SAM模型路径（选择[手动输入路径]时使用）"}),
                "SAM置信度阈值": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "SAM分割置信度阈值，提高可获得更精确的分割"}),

                # ========== V6.7 视频模式 ==========
                "视频模式": ("BOOLEAN", {"default": False, "tooltip": "启用视频模式，优化帧间一致性"}),
                "时序一致性强度": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "帧间平滑强度，越大越平滑但可能损失细节"}),
                "参考帧窗口": ("INT", {"default": 2, "min": 1, "max": 5, "step": 1, "tooltip": "用于时序一致性的参考帧数量"}),

                # ========== V6.7 EMA 注意力增强 ==========
                "启用EMA注意力": ("BOOLEAN", {"default": False, "tooltip": "启用多尺度注意力机制，聚焦关键区域"}),
                "注意力聚焦强度": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 3.0, "step": 0.1, "tooltip": "注意力聚焦系数，越大越集中于高置信度区域"}),

                # ========== V6.0 核心优化 ==========
                "启用Tile检测": ("BOOLEAN", {"default": True, "tooltip": "大图分块检测，提高召回率"}),
                "Tile大小": ("INT", {"default": 640, "min": 320, "max": 1280, "step": 64}),
                "Tile重叠": ("FLOAT", {"default": 0.3, "min": 0.1, "max": 0.5, "step": 0.05}),
                "使用Seg分割": ("BOOLEAN", {"default": True, "tooltip": "使用YOLO-seg模型的分割mask（需要seg模型）"}),
                "Seg边缘羽化": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.5}),

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
                "修复优先级": (["按置信度", "按面积", "脸优先", "手优先", "脚优先", "自定义优先级"], {"default": "脸优先"}),
                "自定义优先级关键词": ("STRING", {"default": "face,hand,foot", "multiline": False}),
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

                # ========== 二次检测精修 ==========
                "二次检测精修": ("BOOLEAN", {"default": False}),
                "二次检测最多目标": ("INT", {"default": 6, "min": 1, "max": 50, "step": 1}),
                "二次检测扩展": ("INT", {"default": 32, "min": 0, "max": 256, "step": 8}),

                # ========== V5.1 新增：高级融合选项 ==========
                "融合模式": (["Poisson融合", "MPB增强融合", "频率感知", "简单混合", "智能柔和融合"], {"default": "智能柔和融合", "tooltip": "智能柔和融合：多层次渐进羽化+色彩对齐+细节恢复，柔化色差和朦胧感"}),
                "Poisson迭代次数": ("INT", {"default": 40, "min": 0, "max": 100, "step": 5}),
                "MPB颜色保持": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "MPB模式下的颜色保持强度，0=完全融合，1=完全保持"}),
                "色差修复强度": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "提高色差修复强度可更好地平衡修复区域和原始区域的色彩"}),
                "清晰度恢复强度": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "修复后的细节和清晰度恢复强度，可减少朦胧感"}),
                "使用Latent感知插值": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("LATENT", "IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("Latent", "检测预览", "检测遮罩", "检测信息")
    FUNCTION = "process"
    CATEGORY = "千绘/修复"

    def __init__(self):
        super().__init__()
        # V6.10.5: 使用智能多模型缓存管理器（自动适配显存大小）
        self.yolo_cache = YOLOModelCache(max_cached_models=3, vram_threshold_mb=500)
        # 兼容旧代码
        self.yolo_model = None
        self.yolo_model_path = None
        self._printed_classes = False

        # V6.10.5: 同步低显存模式状态
        self._low_vram_mode = getattr(self.yolo_cache, 'low_vram_mode', False)

    def process(self, Latent, 模型, VAE, 正面提示, 负面提示, seed, 步数,
                CFG, 采样器, 调度器, 降噪, 放大算法, 放大倍数, **kwargs):
        ctx = prepare_run(self, kwargs, Latent=Latent, VAE=VAE,
                          正面提示=正面提示, 负面提示=负面提示, seed=seed,
                          模型=模型, 步数=步数, CFG=CFG, 采样器=采样器, 调度器=调度器,
                          降噪=降噪, 放大算法=放大算法, 放大倍数=放大倍数)
        try:
            for i in range(ctx.batch_size):
                ctx.i = i
                run_detection(self, ctx)
                run_repair(self, ctx)
                run_global_pass(self, ctx)
        except InterruptProcessingException:
            finalize_interrupt(self, ctx)
        return build_report(self, ctx)
