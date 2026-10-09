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

        # ControlNet
        CN模型 = kwargs.get('CN模型', None)
        CN图像 = kwargs.get('CN图像', None)
        CN强度 = float(kwargs.get('CN强度', 1.0))
        CN开始 = float(kwargs.get('CN开始', 0.0))
        CN结束 = float(kwargs.get('CN结束', 1.0))
        has_cn = CN模型 is not None and CN图像 is not None

        # 低显存/性能
        low_vram = bool(kwargs.get("低显存模式", False))
        clear_each = bool(kwargs.get("每帧清理缓存", False))
        cache_device_mode = (kwargs.get("结果缓存设备", "自动") or "自动").strip()
        enable_profiling = bool(kwargs.get("启用性能分析", False))

        # V6.6: 性能分析器
        import time
        perf_start_time = time.time() if enable_profiling else None

        # 工作模式
        work_mode = (kwargs.get("工作模式", "仅修复(保持输入Latent)") or "仅修复(保持输入Latent)").strip()
        final_denoise_strength = float(kwargs.get("全图降噪强度", 0.25) or 0.25)

        # YOLO参数（V6.9优化：每个模型独立置信度）
        enable_yolo = bool(kwargs.get('启用YOLO', True))
        yolo_model_selection = kwargs.get('YOLO模型', '')
        manual_path = kwargs.get('YOLO路径', '')
        enable_multi_model = bool(kwargs.get('启用多模型', False))
        yolo_model2 = kwargs.get('YOLO模型2', '[不使用]')
        yolo_model3 = kwargs.get('YOLO模型3', '[不使用]')
        yolo_model4 = kwargs.get('YOLO模型4', '[不使用]')
        confidence = float(kwargs.get('置信度', 0.25))  # 主置信度（模型1）
        confidence2 = float(kwargs.get('YOLO模型2置信度', 0.0))  # 模型2独立置信度
        confidence3 = float(kwargs.get('YOLO模型3置信度', 0.0))  # 模型3独立置信度
        confidence4 = float(kwargs.get('YOLO模型4置信度', 0.0))  # 模型4独立置信度
        target_classes_str = kwargs.get('目标类别', '')
        yolo_imgsz = int(kwargs.get("YOLO推理尺寸", 768) or 768)
        multi_scale = bool(kwargs.get("多尺度检测", True))
        inference_precision = kwargs.get("推理精度", "FP32(默认)")

        # V6.7 WIoU 优化
        use_wiou = bool(kwargs.get("使用WIoU优化", False))
        wiou_alpha = float(kwargs.get("WIoU权重参数", 3.0) or 3.0)

        # V6.7 SAM 集成
        enable_sam = bool(kwargs.get("启用SAM精细化", False))
        # V6.8: SAM模型路径选择
        sam_model_selection = kwargs.get("SAM模型选择", "")
        sam_model_manual = kwargs.get("SAM模型手动路径", "")
        sam_model_path = self._resolve_sam_model_path(sam_model_selection, sam_model_manual)
        sam_conf_threshold = float(kwargs.get("SAM置信度阈值", 0.7) or 0.7)

        # V6.7 视频模式
        video_mode = bool(kwargs.get("视频模式", False))
        temporal_strength = float(kwargs.get("时序一致性强度", 0.3) or 0.3)
        frame_window = int(kwargs.get("参考帧窗口", 2) or 2)

        # V6.7 EMA 注意力
        enable_ema_attention = bool(kwargs.get("启用EMA注意力", False))
        attention_focus = float(kwargs.get("注意力聚焦强度", 1.5) or 1.5)

        # V6.0 核心参数
        tile_enabled = bool(kwargs.get("启用Tile检测", True))
        tile_size = int(kwargs.get("Tile大小", 640) or 640)
        tile_overlap = float(kwargs.get("Tile重叠", 0.3) or 0.3)
        use_seg = bool(kwargs.get("使用Seg分割", True))
        seg_feather = float(kwargs.get("Seg边缘羽化", 3.0) or 3.0)
        # use_inpaint功能已移除（与某些插件冲突导致CUDA崩溃）

        # 合并/过滤等
        merge_boxes = bool(kwargs.get('合并重叠框', True))
        merge_iou = float(kwargs.get('合并IOU阈值', 0.35))
        merge_same_class = bool(kwargs.get('合并仅同类', True))
        max_targets = int(kwargs.get('最多目标数', 0) or 0)

        min_area_ratio = float(kwargs.get('过滤最小面积比', 0.0))
        max_area_ratio = float(kwargs.get('过滤最大面积比', 1.0))
        min_aspect = float(kwargs.get('过滤最小长宽比', 0.0))
        max_aspect = float(kwargs.get('过滤最大长宽比', 10.0))
        sort_mode = kwargs.get('排序方式', '置信度')
        per_class_limit = int(kwargs.get('每类最多目标数', 0) or 0)
        lr_protect = bool(kwargs.get('左右分组保护', True))
        bbox_expand2_px = int(kwargs.get('合并后二次扩张像素', 0) or 0)
        bbox_expand2_ratio = float(kwargs.get('合并后二次扩张比例', 0.0))

        # 智能策略
        smart_strategy = (kwargs.get("智能策略", "人像(脸/手/脚)") or "关闭").strip()
        
        # V6.4 手脚增强（含提示词引导）
        hand_foot_enhance = bool(kwargs.get("手脚增强模式", True))
        hand_foot_denoise = float(kwargs.get("手脚降噪下限", 0.80) or 0.80)
        hand_foot_extra_rounds = int(kwargs.get("手脚额外轮次", 1) or 1)
        # V6.11 新增手脚优化参数
        hand_foot_cfg_boost = float(kwargs.get("手脚CFG加成", 1.2) or 1.2)
        hand_foot_cn_strength = float(kwargs.get("手脚CN强度", 0.7) or 0.7)
        hand_foot_denoise_decay = float(kwargs.get("手脚降噪衰减", 0.85) or 0.85)
        hand_foot_denoise_min = float(kwargs.get("手脚衰减下限", 0.40) or 0.40)
        # V6.11.3 手骨架ControlNet参数
        hand_skeleton_cn_model = kwargs.get("手骨架CN模型", None)
        skeleton_cn_strength = float(kwargs.get("骨架CN强度", 0.65) or 0.65)
        skeleton_cn_end = float(kwargs.get("骨架CN结束", 0.7) or 0.7)
        has_hand_skeleton_cn = hand_skeleton_cn_model is not None
        # 提示词增强已移除（导致CUDA崩溃）

        # 优先级/规则/自适应
        priority_mode = (kwargs.get('修复优先级', '脸优先') or '脸优先').strip()
        custom_priority = kwargs.get('自定义优先级关键词', 'face,hand,foot') or ''
        adaptive_dn = bool(kwargs.get('按置信度自适应降噪', True))
        conf_low_scale = float(kwargs.get('自适应降噪_低置信缩放', 0.85) or 0.85)
        conf_high_scale = float(kwargs.get('自适应降噪_高置信缩放', 1.0) or 1.0)

        # V6.5: 类别规则预设处理
        class_rules_preset = kwargs.get('类别规则预设', '[不使用预设]')
        class_rules_text = kwargs.get('类别规则', '')

        # 如果选择了预设，应用预设规则
        if class_rules_preset and class_rules_preset != "[不使用预设]" and class_rules_preset != "自定义规则":
            class_rules_text = self._get_preset_class_rules(class_rules_preset)

        class_rules = self._parse_class_rules(class_rules_text)

        # 遮罩
        expand = int(kwargs.get('遮罩扩展', 32))
        feather = int(kwargs.get('遮罩羽化', 24))

        # 修复采样参数
        region_steps = int(kwargs.get("区域步数", 0) or 0)
        region_cfg = float(kwargs.get("区域CFG", 0.0) or 0.0)
        region_denoise = float(kwargs.get('区域降噪', 0.6))
        fix_rounds = int(kwargs.get("修复轮次", 1) or 1)
        decay = bool(kwargs.get("逐次降噪递减", True))
        no_detect_mode = kwargs.get('未检测时处理', '保持原样')

        # 裁剪精修
        enable_crop = bool(kwargs.get("开启裁剪精修", True))
        crop_pad = int(kwargs.get("裁剪扩展", 64))
        crop_max_edge = int(kwargs.get("裁剪最大边", 512))
        crop_upscale = float(kwargs.get("裁剪放大倍率", 1.6))
        crop_steps = int(kwargs.get("裁剪步数", 20))
        crop_cfg = float(kwargs.get("裁剪CFG", 0.0) or 0.0)
        crop_denoise = float(kwargs.get("裁剪降噪", 0.5))
        crop_rounds = int(kwargs.get("裁剪轮次", 2))
        crop_use_cn = bool(kwargs.get("裁剪使用ControlNet", True))

        # 二次检测
        refine_det = bool(kwargs.get("二次检测精修", False))
        refine_max = int(kwargs.get("二次检测最多目标", 6))
        refine_pad = int(kwargs.get("二次检测扩展", 32))

        # V5.1 新增融合选项（V6.9增强：智能柔和融合）
        blend_mode = (kwargs.get("融合模式", "智能柔和融合") or "智能柔和融合").strip()
        poisson_iters = int(kwargs.get("Poisson迭代次数", 40) or 40)
        mpb_color_preserve = float(kwargs.get("MPB颜色保持", 0.5) or 0.5)
        color_strength = float(kwargs.get("色差修复强度", 0.85) or 0.85)
        clarity_strength = float(kwargs.get("清晰度恢复强度", 0.3) or 0.3)
        use_latent_aware = bool(kwargs.get("使用Latent感知插值", True))

        target_classes = self._parse_target_classes(target_classes_str)

        samples = Latent["samples"]
        batch_size = samples.shape[0]

        print(f"\n{'='*60}")
        print(f"[Latent YOLO智能处理+ControlNet V6.11.3] 批量: {batch_size}")
        print(f"  工作模式: {work_mode}" + (f" (降噪={final_denoise_strength})" if work_mode == "修复后全图降噪" else ""))
        print(f"  放大: {放大倍数}x | 基础降噪: {降噪} | 色差修复: {color_strength}")
        print(f"  融合模式: {blend_mode} | Latent感知插值: {'ON' if use_latent_aware else 'OFF'} | 裁剪精修: {'ON' if enable_crop else 'OFF'}")
        print(f"  V6核心: Tile={'ON' if tile_enabled else 'OFF'} | Seg={'ON' if use_seg else 'OFF'} | 多尺度={'ON' if multi_scale else 'OFF'}")
        if use_wiou:
            print(f"  ★WIoU优化: 启用 (α={wiou_alpha}) | 提升检测框精度10-15%")
        if enable_sam:
            sam_model_name = os.path.basename(sam_model_path) if sam_model_path else "[未设置]"
            print(f"  ★SAM精细化: {sam_model_name} | 阈值={sam_conf_threshold} | 超高精度分割")
        if video_mode:
            print(f"  ★视频模式: 时序平滑={temporal_strength} | 参考帧={frame_window} | 帧间一致性优化")
        if hand_foot_enhance:
            print(f"  ★手脚增强: 降噪≥{hand_foot_denoise}, +{hand_foot_extra_rounds}轮, CFG×{hand_foot_cfg_boost}, CN={hand_foot_cn_strength}")
        if has_hand_skeleton_cn:
            print(f"  ★手骨架CN: 强度={skeleton_cn_strength}, 结束={skeleton_cn_end} (MediaPipe自动提取骨架)")
        if has_cn:
            print(f"  ControlNet: 强度={CN强度}, 范围=[{CN开始}, {CN结束}]")

        # 初始化YOLO（V6.9优化：支持每个模型独立置信度）
        yolo_models_to_use = []
        yolo_confidences_to_use = []  # V6.9新增：独立置信度列表
        if enable_yolo:
            # 首先添加主模型
            resolved_path = self._resolve_model_path(yolo_model_selection, manual_path)
            if resolved_path and os.path.exists(resolved_path):
                yolo_models_to_use.append(resolved_path)
                yolo_confidences_to_use.append(confidence)  # 主模型使用主置信度

            # 如果启用多模型，添加额外模型
            if enable_multi_model:
                additional_models = [
                    (yolo_model2, confidence2 if confidence2 > 0 else confidence),
                    (yolo_model3, confidence3 if confidence3 > 0 else confidence),
                    (yolo_model4, confidence4 if confidence4 > 0 else confidence),
                ]
                for model_sel, model_conf in additional_models:
                    if model_sel and model_sel != "[不使用]":
                        model_path = YOLOModelManager.get_model_path(model_sel)
                        if model_path and model_path != "MANUAL_INPUT" and os.path.exists(model_path):
                            if model_path not in yolo_models_to_use:  # 避免重复
                                yolo_models_to_use.append(model_path)
                                yolo_confidences_to_use.append(model_conf)  # 使用对应的置信度

            if not yolo_models_to_use:
                print(f"  [警告] YOLO模型路径无效，禁用YOLO")
                enable_yolo = False
            elif len(yolo_models_to_use) > 1:
                print(f"  [多模型模式] 准备加载 {len(yolo_models_to_use)} 个YOLO模型:")
                for idx, (mpath, conf) in enumerate(zip(yolo_models_to_use, yolo_confidences_to_use), 1):
                    print(f"    [{idx}] {os.path.basename(mpath)} | 置信度={conf:.2f}")
                print(f"  分批检测模式 | 每个模型独立置信度阈值")
            else:
                model_conf = yolo_confidences_to_use[0] if yolo_confidences_to_use else confidence
                print(f"  YOLO模型: {os.path.basename(yolo_models_to_use[0])} | 置信度={model_conf:.2f}")

        # 输出缓存设备策略
        if cache_device_mode == "CPU":
            out_device = torch.device("cpu")
        elif cache_device_mode == "GPU":
            out_device = samples.device
        else:
            out_device = torch.device("cpu") if low_vram else samples.device

        out_latents = None
        out_previews = []
        out_masks = []
        all_dets = []

        # V6.10.8: 重置CUDA损坏标志（每次新执行时重置）
        self._cuda_corrupted = False

        ksampler = nodes.KSampler()

        # V6.6: 进度条支持
        try:
            from comfy.utils import ProgressBar
            pbar = ProgressBar(batch_size)
        except:
            pbar = None

        # V6.7: 视频模式帧缓存
        frame_history = [] if video_mode else None

        # V6.10.7: 中断保护 - 保存中间结果防止黑图
        interrupted = False
        processed_count = 0
        # 保存原始输入latent作为fallback
        original_samples = samples.clone()

        try:
            for i in range(batch_size):
                # V6.10: 检查中断信号，允许用户随时停止
                comfy.model_management.throw_exception_if_processing_interrupted()

                if pbar:
                    pbar.update(1)
                print(f"\n  [{i+1}/{batch_size}] 处理中...")
                single_sample = samples[i:i+1]

                # V6.11.2: 确保samples在GPU上
                if torch.cuda.is_available() and single_sample.device.type != 'cuda':
                    print(f"    [设备警告] samples在{single_sample.device}，移动到GPU")
                    single_sample = single_sample.to('cuda')
                    # 同步更新原samples
                    samples = samples.to('cuda')

                # 放大（V5.1：支持latent_aware模式）
                if 放大倍数 > 1.0:
                    if 放大算法 == "latent_aware" or use_latent_aware:
                        target_h = int(single_sample.shape[2] * 放大倍数)
                        target_w = int(single_sample.shape[3] * 放大倍数)
                        single_sample = LatentAlgorithms.latent_aware_resize(single_sample, target_h, target_w)
                    else:
                        single_sample = self.upscale_latent(single_sample, 放大算法, 放大倍数)

                latent_h, latent_w = single_sample.shape[2], single_sample.shape[3]
                pixel_h, pixel_w = latent_h * 8, latent_w * 8

                if out_latents is None:
                    out_latents = torch.empty((batch_size, single_sample.shape[1], latent_h, latent_w), device=out_device, dtype=single_sample.dtype)

                base_pos = 正面提示
                base_neg = 负面提示

                pos_full, neg_full = base_pos, base_neg
                if has_cn:
                    pos_full, neg_full = self._maybe_apply_controlnet(
                        base_pos, base_neg, True, CN模型, CN图像, CN强度, CN开始, CN结束, pixel_h, pixel_w, i
                    )

                # 1) base_latent初始化
                base_latent = single_sample

                # V6.11.2: 确保base_latent在GPU上
                if torch.cuda.is_available() and base_latent.device.type != 'cuda':
                    print(f"    [设备警告] base_latent在{base_latent.device}，移动到GPU")
                    base_latent = base_latent.to('cuda')

                # 1.5) YOLO预处理：先进行全图降噪得到清晰图片
                # 原理：原始latent包含大量噪声，直接解码会得到模糊图片
                # 解决：先做一次全图降噪，得到清晰的基础图像供YOLO检测
                # 优势：YOLO能准确识别目标，然后对这些目标做高质量区域精修
                if enable_yolo and yolo_models_to_use and 降噪 > 0.0:
                    # 检测专用降噪上限 0.45：保留手部/脚部结构，防止KSampler"创意重绘"导致目标消失
                    # 用户设 0.75 时不影响最终修复质量，只影响检测预处理图像
                    detect_denoise = min(float(降噪), 0.45)
                    print(f"    [1/3] 全图降噪 (检测用={detect_denoise}, 修复用={降噪}) 生成清晰基础图像...")

                    # V6.10.1: 添加异常处理和内存管理
                    try:
                        # 清理CUDA缓存
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        # 确保base_latent连续存储
                        if hasattr(base_latent, 'contiguous'):
                            base_latent = base_latent.contiguous()

                        pre_latent = {"samples": base_latent}
                        pre_result = ksampler.sample(
                            模型, seed + i, 步数, CFG, 采样器, 调度器,
                            pos_full, neg_full, pre_latent, denoise=detect_denoise
                        )
                        # 用降噪后的清晰latent进行YOLO检测
                        latent_for_detection = pre_result[0]["samples"]
                        print(f"    [2/3] YOLO检测中...")

                    except Exception as e:
                        print(f"    [1/3] 全图降噪失败，使用原始latent: {e}")
                        latent_for_detection = base_latent
                        print(f"    [2/3] YOLO检测中...")
                        # 清理CUDA缓存
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                else:
                    # 如果不启用YOLO或降噪为0，直接使用原始latent
                    latent_for_detection = base_latent

                # V6.11.3: 不再强制latent移到GPU——ComfyUI的VAE.decode()内部会自行处理设备放置
                # 强制.to('cuda')反而会在SDXL占满显存时造成额外压力，导致模型换出卡顿

                # 2) YOLO检测
                detection_mask = torch.zeros((pixel_h, pixel_w), dtype=torch.float32)
                detections = []
                preview_image = None
                _decoded_np = None  # V6.11.3: 原图numpy，供手骨架提取使用

                if enable_yolo and yolo_models_to_use:
                    # V6.11.3: VAE解码前显式释放显存，避免SDXL(~5GB)换出时卡顿
                    # SDXL采样完成后不再需要占用VRAM，释放后VAE可以直接加载而无需走PCIe换出
                    print(f"    [显存] 采样完成，释放显存准备VAE解码...")
                    try:
                        comfy.model_management.soft_empty_cache()
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()
                        import gc as _gc_vae; _gc_vae.collect()
                        if torch.cuda.is_available():
                            _free_mb = (torch.cuda.get_device_properties(0).total_memory
                                        - torch.cuda.memory_allocated(0)) / 1024**2
                            print(f"    [显存] 清理后可用: {_free_mb:.0f}MB")
                    except Exception as _vae_mem_err:
                        print(f"    [显存] 释放警告（继续）: {_vae_mem_err}")

                    # 解码降噪后的latent（而不是原始latent）
                    decoded_original = self._decode_latent_to_image(VAE, latent_for_detection)[0]

                    # V6.11.3: decode完成后立刻移到CPU，释放VRAM给后续SDXL修复阶段
                    if hasattr(decoded_original, 'is_cuda') and decoded_original.is_cuda:
                        decoded_original = decoded_original.cpu()

                    # V6.11.3: 保存原图numpy供手骨架提取（需要时才转换，zero-copy）
                    if has_hand_skeleton_cn:
                        _decoded_np = decoded_original.numpy() if isinstance(decoded_original, torch.Tensor) else decoded_original
                    else:
                        _decoded_np = None

                    # 上采样图片以提高 YOLO 检测质量
                    decoded_det = self._upscale_image_for_yolo(decoded_original, min_size=1024)

                    det_h, det_w = decoded_det.shape[0], decoded_det.shape[1]

                    # V6.9：支持多模型分批检测（每个模型独立置信度）
                    if len(yolo_models_to_use) > 1:
                        # 多模型模式
                        raw = self._detect_with_multi_models(
                            yolo_models_to_use, decoded_det, yolo_confidences_to_use,  # V6.9: 传入置信度列表
                            imgsz=yolo_imgsz,
                            tile_enabled=tile_enabled,
                            tile_size=tile_size,
                            tile_overlap=tile_overlap,
                            use_seg=use_seg,
                            target_classes=target_classes,
                            multi_scale=multi_scale,
                            precision=inference_precision
                        )
                    else:
                        # 单模型模式（兼容旧版）
                        yolo_model = self.yolo_cache.load_model(yolo_models_to_use[0], precision=inference_precision) if hasattr(self, 'yolo_cache') else self._load_yolo_model(yolo_models_to_use[0])
                        model_confidence = yolo_confidences_to_use[0] if yolo_confidences_to_use else confidence
                        raw = self._detect_with_yolo(
                            yolo_model, decoded_det, model_confidence,  # V6.9: 使用对应置信度
                            imgsz=yolo_imgsz,
                            tile_enabled=tile_enabled,
                            tile_size=tile_size,
                            tile_overlap=tile_overlap,
                            use_seg=use_seg,
                            target_classes=target_classes,
                            multi_scale=multi_scale
                        )
                    # 无结果时降低置信度自动重试一次（解决边界置信度漏检问题）
                    if not raw:
                        retry_conf = max(0.10, confidence * 0.6)
                        print(f"    [YOLO] 未检测到目标，降低置信度至 {retry_conf:.2f} 自动重试...")
                        try:
                            if len(yolo_models_to_use) > 1:
                                raw = self._detect_with_multi_models(
                                    yolo_models_to_use, decoded_det,
                                    [max(0.10, c * 0.6) for c in yolo_confidences_to_use],
                                    imgsz=yolo_imgsz, tile_enabled=tile_enabled,
                                    tile_size=tile_size, tile_overlap=tile_overlap,
                                    use_seg=use_seg, target_classes=target_classes,
                                    multi_scale=multi_scale, precision=inference_precision
                                )
                            else:
                                raw = self._detect_with_yolo(
                                    yolo_model, decoded_det, retry_conf,
                                    imgsz=yolo_imgsz, tile_enabled=tile_enabled,
                                    tile_size=tile_size, tile_overlap=tile_overlap,
                                    use_seg=use_seg, target_classes=target_classes,
                                    multi_scale=multi_scale
                                )
                            if raw:
                                print(f"    [YOLO] 重试成功，检测到 {len(raw)} 个目标（置信度={retry_conf:.2f}）")
                            else:
                                print(f"    [YOLO] 重试仍无结果")
                        except Exception as retry_err:
                            print(f"    [YOLO] 重试失败: {retry_err}")

                    if raw:
                        dets = raw  # 已经在检测时过滤了target_classes

                        # 如果检测图片被上采样，需要缩放坐标回原始尺寸
                        if det_h != pixel_h or det_w != pixel_w:
                            scale_x = pixel_w / det_w
                            scale_y = pixel_h / det_h
                            print(f"    [YOLO] 缩放检测坐标: {det_w}x{det_h} -> {pixel_w}x{pixel_h}")

                            for det in dets:
                                # 缩放边界框
                                if "bbox" in det:
                                    x1, y1, x2, y2 = det["bbox"]
                                    det["bbox"] = [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]

                                # 缩放 seg_mask（V6.9优化：更鲁棒的形状处理）
                                if det.get("seg_mask") is not None:
                                    try:
                                        with torch.no_grad():
                                            seg = det["seg_mask"]

                                            # 确保seg是numpy数组
                                            if not isinstance(seg, np.ndarray):
                                                seg = np.array(seg)

                                            # 确保是2D数组 (H, W) - 使用循环确保完全降维
                                            while seg.ndim > 2:
                                                if seg.shape[0] == 1:
                                                    seg = seg.squeeze(0)  # 移除大小为1的第一维
                                                else:
                                                    seg = seg[0]  # 取第一个通道

                                            # 转换为tensor并resize
                                            seg_t = torch.from_numpy(seg.astype(np.float32))

                                            # 确保是4D张量 (1, 1, H, W)
                                            if seg_t.ndim == 2:
                                                seg_t = seg_t.unsqueeze(0).unsqueeze(0)
                                            elif seg_t.ndim == 3:
                                                seg_t = seg_t.unsqueeze(0)

                                            # Resize到目标尺寸
                                            seg_t = F.interpolate(seg_t, size=(pixel_h, pixel_w), mode="bilinear", align_corners=False)

                                            # 降维到2D并转回numpy
                                            seg_resized = seg_t.squeeze().cpu().numpy()

                                            # 最终确保是2D
                                            while seg_resized.ndim > 2:
                                                if seg_resized.shape[0] == 1:
                                                    seg_resized = seg_resized.squeeze(0)
                                                else:
                                                    seg_resized = seg_resized[0]

                                            det["seg_mask"] = seg_resized
                                            del seg_t

                                    except Exception as e:
                                        # seg mask缩放失败，移除mask但保留检测框
                                        det["seg_mask"] = None
                                        print(f"    [警告] Seg mask缩放失败（已移除mask）: {e}")

                        print(f"    [过滤] 原始检测数: {len(dets)} | 几何过滤参数: 最小面积比={min_area_ratio}, 最大面积比={max_area_ratio}, 最大长宽比={max_aspect}")
                        dets = self._filter_detections_geometry(dets, pixel_w, pixel_h,
                                                               min_area_ratio=min_area_ratio, max_area_ratio=max_area_ratio,
                                                               min_aspect=min_aspect, max_aspect=max_aspect)
                        print(f"    [过滤] 几何过滤后: {len(dets)}")
                        dets = self._sort_detections(dets, sort_mode=sort_mode)
                        dets = self._limit_detections_per_class(dets, per_class_limit=per_class_limit)
                        print(f"    [过滤] 类别限制后(limit={per_class_limit}): {len(dets)}")
                        if max_targets and max_targets > 0 and len(dets) > max_targets:
                            dets = dets[:max_targets]
                        print(f"    [过滤] 最多目标限制后(max={max_targets}): {len(dets)}")

                        if merge_boxes and len(dets) > 1:
                            dets = self._merge_overlapping_detections(dets, iou_threshold=merge_iou,
                                                                      same_class=merge_same_class, lr_protect=lr_protect, image_width=pixel_w,
                                                                      use_wiou=use_wiou, wiou_alpha=wiou_alpha)
                        print(f"    [过滤] 合并重叠后: {len(dets)}")

                        if (bbox_expand2_px and bbox_expand2_px > 0) or (bbox_expand2_ratio and bbox_expand2_ratio > 0.0):
                            for d in dets:
                                d["bbox"] = self._expand_bbox_xyxy(d.get("bbox",[0,0,0,0]), pixel_w, pixel_h,
                                                                  extra_px=bbox_expand2_px, extra_ratio=bbox_expand2_ratio)

                        dets = self._sort_detections(dets, sort_mode=sort_mode)

                        # 二次检测精修（V6.0更新 / V6.5支持多模型）
                        if refine_det and dets and (len(dets) <= refine_max):
                            # V6.5: 获取用于二次检测的模型（使用第一个模型或根据检测来源选择）
                            refine_model = None
                            if len(yolo_models_to_use) == 1:
                                # 单模型模式：使用已加载的模型
                                refine_model = yolo_model if 'yolo_model' in locals() else None
                            else:
                                # 多模型模式：使用第一个模型进行二次检测
                                if hasattr(self, 'yolo_cache'):
                                    refine_model = self.yolo_cache.load_model(yolo_models_to_use[0], precision=inference_precision)

                            if refine_model is None:
                                print(f"    [警告] 二次检测：无法获取模型，跳过")
                            else:
                                refined = []
                                ref_conf = max(0.05, float(confidence) * 0.5)
                                # 计算坐标缩放比例（从 pixel 坐标到 det 坐标）
                                scale_to_det_x = det_w / pixel_w if pixel_w != det_w else 1.0
                                scale_to_det_y = det_h / pixel_h if pixel_h != det_h else 1.0

                                for dd in dets:
                                    try:
                                        # bbox 坐标是 pixel 尺寸，需要缩放到 det 尺寸用于裁剪
                                        bx1, by1, bx2, by2 = [float(v) for v in dd.get("bbox", [0, 0, 0, 0])]
                                        bx1_det = bx1 * scale_to_det_x
                                        by1_det = by1 * scale_to_det_y
                                        bx2_det = bx2 * scale_to_det_x
                                        by2_det = by2 * scale_to_det_y

                                        cx1 = max(0, int(bx1_det - refine_pad)); cy1 = max(0, int(by1_det - refine_pad))
                                        cx2 = min(det_w, int(bx2_det + refine_pad)); cy2 = min(det_h, int(by2_det + refine_pad))
                                        if (cx2 - cx1) < 8 or (cy2 - cy1) < 8:
                                            refined.append(dd)
                                            continue

                                        crop_img = decoded_det[cy1:cy2, cx1:cx2, :]
                                        ref_dets = self._detect_single_enhanced(refine_model, crop_img, ref_conf, yolo_imgsz, use_seg, [dd.get("class_name","")])
                                        if ref_dets:
                                            best = max(ref_dets, key=lambda x: float(x.get("confidence", 0.0)))
                                            rx1, ry1, rx2, ry2 = [float(v) for v in best.get("bbox", [0, 0, 0, 0])]
                                            dd = dd.copy()
                                            # 新坐标：从 det 坐标转回 pixel 坐标
                                            new_x1 = (cx1 + rx1) / scale_to_det_x
                                            new_y1 = (cy1 + ry1) / scale_to_det_y
                                            new_x2 = (cx1 + rx2) / scale_to_det_x
                                            new_y2 = (cy1 + ry2) / scale_to_det_y
                                            dd["bbox"] = [new_x1, new_y1, new_x2, new_y2]
                                            # 合并seg mask（已经是 pixel 尺寸）
                                            if best.get("seg_mask") is not None:
                                                try:
                                                    seg = best["seg_mask"]
                                                    # 确保seg是2D数组
                                                    if seg.ndim > 2:
                                                        seg = seg.squeeze()
                                                    # 缩放 seg_mask 到 pixel 尺寸
                                                    with torch.no_grad():
                                                        seg_t = torch.from_numpy(seg).unsqueeze(0).unsqueeze(0)
                                                        seg_t = F.interpolate(seg_t, size=(pixel_h, pixel_w), mode="bilinear", align_corners=False)
                                                        dd["seg_mask"] = seg_t.squeeze().cpu().numpy()
                                                        del seg_t
                                                except Exception as e_seg:
                                                    # seg mask处理失败，但保留检测框
                                                    dd["seg_mask"] = None
                                        else:
                                            # V6.11.3修复：二次检测无结果时保留原始检测框，而非静默丢弃
                                            # 原bug：此处缺少else分支，导致refine无结果时dd不被append
                                            print(f"    [二次检测] {dd.get('class_name','')} 裁剪区域无结果，保留原始框")
                                        refined.append(dd)
                                    except Exception:
                                        refined.append(dd)
                                dets = refined
                                print(f"    [过滤] 二次检测精修后: {len(dets)}")

                        dets = self._priority_sort_detections(dets, priority_mode, custom_priority)
                        dets = self._apply_rule_limits(dets, class_rules, default_per_class_limit=per_class_limit, max_targets=max_targets)
                        print(f"    [过滤] 规则限制后(最终): {len(dets)}")

                        # V6.8: SAM 精细化（可选，支持SAM/SAM2/SAM2.1）
                        if enable_sam and dets and sam_model_path:
                            # V6.10: 检查中断信号（SAM处理可能耗时较长）
                            comfy.model_management.throw_exception_if_processing_interrupted()

                            # V6.11.1: SAM加载前清理显存（保守方式，不完全卸载主模型）
                            # 使用soft_empty_cache而非unload_all_models，避免主模型无法自动重载
                            print(f"    [SAM] 清理显存为SAM腾出空间...")
                            try:
                                # 请求ComfyUI清理缓存（不会强制卸载模型）
                                comfy.model_management.soft_empty_cache()
                                # 清理YOLO缓存
                                if hasattr(self, 'yolo_cache'):
                                    while len(self.yolo_cache.models) > 0:
                                        self.yolo_cache._unload_lru_model()
                                # 多轮GC确保清理彻底
                                import gc
                                for _ in range(2):
                                    gc.collect()
                                if torch.cuda.is_available():
                                    torch.cuda.synchronize()
                                    torch.cuda.empty_cache()
                                    # 打印显存状态
                                    free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                                    print(f"    [SAM] 显存已清理，可用: {free_mem/1024**2:.0f}MB")
                            except Exception as cleanup_err:
                                print(f"    [SAM] 清理显存时出错: {cleanup_err}")

                            # V6.10.8: 完整的SAM错误保护，防止CUDA错误传播到inpaint阶段
                            sam_success = False
                            try:
                                # 转换图像格式供 SAM 使用
                                decoded_for_sam = decoded_det.cpu().numpy()
                                if decoded_for_sam.max() <= 1.0:
                                    decoded_for_sam = (decoded_for_sam * 255).astype(np.uint8)
                                else:
                                    decoded_for_sam = decoded_for_sam.astype(np.uint8)

                                # 显式同步CUDA确保之前操作完成
                                if torch.cuda.is_available():
                                    torch.cuda.synchronize()

                                dets = self._refine_with_sam(decoded_for_sam, dets, model_path=sam_model_path, conf_threshold=sam_conf_threshold)
                                sam_success = True

                            except RuntimeError as sam_err:
                                error_msg = str(sam_err).lower()
                                if 'cuda' in error_msg or 'illegal memory' in error_msg:
                                    print(f"    [SAM错误] CUDA错误，跳过SAM精细化: {sam_err}")
                                    print(f"    [SAM错误] 将使用YOLO检测结果（无精细分割mask）")
                                    # 尝试恢复CUDA状态
                                    try:
                                        import gc
                                        gc.collect()
                                        if torch.cuda.is_available():
                                            torch.cuda.empty_cache()
                                            # 重置CUDA设备上下文
                                            torch.cuda.synchronize()
                                    except:
                                        pass
                                else:
                                    print(f"    [SAM错误] 运行时错误: {sam_err}")

                            except Exception as sam_err:
                                print(f"    [SAM错误] 异常: {sam_err}")

                            finally:
                                # V6.10性能优化：SAM使用完后立即释放
                                try:
                                    self._unload_sam_model()
                                except:
                                    pass

                                # V6.11.1: SAM完成后彻底清理，为主模型重新加载腾出空间
                                if torch.cuda.is_available():
                                    try:
                                        torch.cuda.synchronize()
                                        import gc
                                        for _ in range(3):  # 多轮GC确保清理循环引用
                                            gc.collect()
                                        torch.cuda.empty_cache()
                                        # 请求ComfyUI清理
                                        comfy.model_management.soft_empty_cache()
                                        # 打印显存状态
                                        free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                                        print(f"    [SAM] 释放完成，显存可用: {free_mem/1024**2:.0f}MB")
                                    except Exception as cleanup_err:
                                        print(f"    [CUDA警告] 清理时出错: {cleanup_err}")

                                # V6.11.2: 显式重新加载主模型到GPU，确保后续采样正常
                                try:
                                    print(f"    [模型] 重新加载主模型到GPU...")
                                    comfy.model_management.load_model_gpu(模型)
                                    if torch.cuda.is_available():
                                        allocated = torch.cuda.memory_allocated(0) / 1024**2
                                        print(f"    [模型] 加载完成，显存占用: {allocated:.0f}MB")
                                        # V6.11.2: 检查模型实际设备
                                        try:
                                            model_device = next(模型.model.parameters()).device
                                            print(f"    [模型] 模型实际设备: {model_device}")
                                        except:
                                            pass
                                except Exception as load_err:
                                    print(f"    [模型警告] 重新加载模型时出错: {load_err}")

                        detections = dets

                    if detections:
                        preview_image = self._draw_detection_preview(decoded_original, detections)
                        print(f"    [3/3] 检测到 {len(detections)} 个目标，开始区域精修...")
                    else:
                        preview_image = decoded_original
                        print(f"    [2/3] 未检测到目标")

                    # 检测完成后立即清理中间变量，释放显存
                    del decoded_det, decoded_original
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                else:
                    preview_image = self._decode_latent_to_image(VAE, base_latent)[0]
                    latent_for_detection = base_latent

                # 3) 区域精修
                # 使用预处理降噪后的latent作为基础（如果进行了预处理）
                # 这样后续的区域修复是在清晰的基础上进行的
                processed_latent = latent_for_detection

                # V6.11.2: 确保latent在GPU上（SAM后可能导致tensor留在CPU）
                if torch.cuda.is_available() and processed_latent.device.type != 'cuda':
                    print(f"    [设备] latent在{processed_latent.device}，移动到GPU...")
                    processed_latent = processed_latent.to('cuda')

                if detections:
                    masks_for_union = []
                    for d in detections:
                        pol = self._build_det_policy(d, pixel_w, pixel_h, smart_strategy, expand, feather, region_denoise, crop_pad, crop_upscale, crop_max_edge, crop_steps, float(CFG if crop_cfg <= 0.0 else crop_cfg), adaptive_dn, conf_low_scale, conf_high_scale, class_rules, hand_foot_enhance, hand_foot_denoise, hand_foot_extra_rounds, hand_foot_cfg_boost, hand_foot_cn_strength, hand_foot_denoise_decay, hand_foot_denoise_min)
                        m_expand, m_feather = pol['mask_expand'], pol['mask_feather']
                        masks_for_union.append((d, m_expand, m_feather))
                
                    max_expand = max(x[1] for x in masks_for_union) if masks_for_union else expand
                    max_feather = max(x[2] for x in masks_for_union) if masks_for_union else feather
                    detection_mask = self._create_detection_mask(detections, pixel_h, pixel_w, max_expand, max_feather, seg_feather=seg_feather)

                    # V6.7: 应用 EMA 注意力机制
                    if enable_ema_attention:
                        attention_map = self._calculate_ema_attention(detections, pixel_h, pixel_w, attention_focus)
                        # 将注意力图应用到检测遮罩（增强高置信度区域）
                        detection_mask = detection_mask * attention_map

                    # ========== 裁剪精修 V5.1 ==========
                    if enable_crop:
                        # V6.10.5: 修复前彻底清理显存+碎片整理
                        if getattr(self, '_low_vram_mode', False):
                            print(f"    [低显存模式] 修复前清理显存...")
                            # 确保YOLO缓存已清空
                            if hasattr(self, 'yolo_cache') and len(self.yolo_cache.models) > 0:
                                while len(self.yolo_cache.models) > 0:
                                    self.yolo_cache._unload_lru_model()
                            # 确保SAM已卸载
                            self._unload_sam_model()
                            # 请求ComfyUI释放非必要模型
                            comfy.model_management.soft_empty_cache()
                            import gc
                            gc.collect()
                            if torch.cuda.is_available():
                                # V6.10.5: 同步并进行碎片整理
                                torch.cuda.synchronize()
                                gc.collect()
                                torch.cuda.empty_cache()

                                # 检查碎片化程度
                                reserved = torch.cuda.memory_reserved(0) / 1024**2
                                allocated = torch.cuda.memory_allocated(0) / 1024**2
                                free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)
                                frag = (reserved - allocated) / reserved * 100 if reserved > 0 else 0

                                print(f"    [低显存模式] 清理后可用显存: {free_mem/1024**2:.0f}MB (碎片化: {frag:.1f}%)")

                                # 如果碎片化严重，再次整理
                                if frag > 30:
                                    print(f"    [低显存模式] 碎片化较高，执行深度整理...")
                                    gc.collect()
                                    torch.cuda.empty_cache()
                                    if hasattr(self, 'yolo_cache'):
                                        self.yolo_cache.defragment_vram()

                        # V6.11.3修复：低显存清理后显式重载主模型到GPU
                        # 原因：soft_empty_cache/gc/empty_cache可能使ComfyUI模型管理认为模型需要重载
                        # 若不提前reload，第一个crop的_inpaint_sample会触发5GB PCIe传输，显示为0%GPU占用
                        # （与SAM路径的load_model_gpu保持一致，见line 4676）
                        try:
                            print(f"    [模型] 显存清理后重新确认主模型在GPU...")
                            _load_fn = getattr(comfy.model_management, 'load_model_gpu', None) or \
                                       (lambda m: comfy.model_management.load_models_gpu([m]))
                            _load_fn(模型)
                            if torch.cuda.is_available():
                                allocated = torch.cuda.memory_allocated(0) / 1024**2
                                print(f"    [模型] 确认完成，已分配显存: {allocated:.0f}MB")
                                try:
                                    model_device = next(模型.model.parameters()).device
                                    print(f"    [模型] 主模型设备: {model_device}")
                                except Exception:
                                    pass
                        except Exception as _reload_err:
                            print(f"    [警告] 主模型确认失败: {_reload_err}（将在采样时重试）")

                        cur = processed_latent.clone()

                        # V6.11.2: 确保latent在GPU上（SAM后模型重载可能导致latent留在CPU）
                        if torch.cuda.is_available():
                            if cur.device.type != 'cuda':
                                print(f"      [设备] latent在{cur.device}，移动到GPU...")
                                cur = cur.to('cuda')
                                print(f"      [设备] latent已移动到{cur.device}")

                        # V6.11.3: 精修开始前做一次轻量 CUDA 上下文检查（移除 synchronize）
                        # 原代码有2次 synchronize，相当于在整批修复开始前就额外阻塞2次
                        if torch.cuda.is_available():
                            try:
                                torch.cuda.set_device(0)
                                # 只分配小张量检测上下文状态，不做 synchronize
                                test_tensor = torch.zeros(1, device='cuda')
                                del test_tensor

                            except RuntimeError as cuda_err:
                                error_msg = str(cuda_err).lower()
                                if 'illegal memory' in error_msg or 'cuda error' in error_msg:
                                    print(f"    [CUDA严重错误] 检测到CUDA上下文损坏: {cuda_err}")
                                    print(f"    [CUDA严重错误] 尝试恢复CUDA上下文...")
                                    try:
                                        import gc
                                        gc.collect()
                                        torch.cuda.empty_cache()
                                        torch.cuda.set_device(0)
                                        print(f"    [CUDA] 上下文恢复成功")
                                    except Exception as recovery_err:
                                        print(f"    [CUDA严重错误] 无法恢复，建议重启ComfyUI: {recovery_err}")
                                else:
                                    print(f"    [CUDA警告] 检测到错误: {cuda_err}")
                                    import gc
                                    gc.collect()
                                    torch.cuda.empty_cache()

                        for pass_i in range(max(1, crop_rounds)):
                            # V6.10: 检查中断信号（裁剪轮次循环）
                            comfy.model_management.throw_exception_if_processing_interrupted()

                            dn = float(crop_denoise)
                            if pass_i > 0 and decay:
                                dn = float(max(0.0, dn * (0.75 ** pass_i)))

                            for di, d in enumerate(detections):
                                # V6.10: 检查中断信号（遍历每个检测目标）
                                comfy.model_management.throw_exception_if_processing_interrupted()

                                # V6.10.5: 低显存模式下，每个目标处理前检查显存和碎片化
                                if getattr(self, '_low_vram_mode', False) and torch.cuda.is_available():
                                    reserved = torch.cuda.memory_reserved(0)
                                    allocated = torch.cuda.memory_allocated(0)
                                    total = torch.cuda.get_device_properties(0).total_memory
                                    free_mem = total - reserved
                                    free_mb = free_mem / 1024**2
                                    frag = (reserved - allocated) / reserved * 100 if reserved > 0 else 0

                                    # 如果可用显存低于1GB 或 碎片化超过50%，主动清理
                                    if free_mb < 1024 or frag > 50:
                                        reason = f"显存不足({free_mb:.0f}MB)" if free_mb < 1024 else f"碎片化过高({frag:.0f}%)"
                                        print(f"      [显存警告] 目标{di+1}处理前{reason}，执行整理...")
                                        import gc
                                        torch.cuda.synchronize()
                                        gc.collect()
                                        torch.cuda.empty_cache()
                                        comfy.model_management.soft_empty_cache()
                                        # 再次检查
                                        free_mem = total - torch.cuda.memory_reserved(0)
                                        reserved_new = torch.cuda.memory_reserved(0)
                                        allocated_new = torch.cuda.memory_allocated(0)
                                        frag_new = (reserved_new - allocated_new) / reserved_new * 100 if reserved_new > 0 else 0
                                        print(f"      [显存警告] 清理后: {free_mem/1024**2:.0f}MB可用, 碎片化{frag_new:.1f}%")

                                bx1, by1, bx2, by2 = d.get("bbox",[0,0,0,0])
                                pol = self._build_det_policy(d, pixel_w, pixel_h, smart_strategy, expand, feather, dn, crop_pad, crop_upscale, crop_max_edge, crop_steps, float(CFG if crop_cfg <= 0.0 else crop_cfg), adaptive_dn, conf_low_scale, conf_high_scale, class_rules, hand_foot_enhance, hand_foot_denoise, hand_foot_extra_rounds, hand_foot_cfg_boost, hand_foot_cn_strength, hand_foot_denoise_decay, hand_foot_denoise_min)
                                m_expand, m_feather = pol['mask_expand'], pol['mask_feather']
                                dn_use = pol['denoise']
                                pad_use = int(pol['crop_pad'])
                                upscale_use = float(pol['crop_upscale'])
                                max_edge_use = int(pol['crop_max_edge'])
                                step_use = int(pol['steps'])
                                cfg_use = float(pol['cfg'])
                                local_rounds = int(pol.get('local_rounds', 1))
                                is_hand_foot = pol.get('is_hand_foot', False)
                                # V6.11 新增：提取手脚优化参数
                                cn_strength_override = pol.get('cn_strength_override', None)
                                denoise_decay = float(pol.get('denoise_decay', 0.75))
                                denoise_min = float(pol.get('denoise_min', 0.0))

                                # 打印手脚增强信息
                                if is_hand_foot and hand_foot_enhance:
                                    print(f"      [{d.get('class_name','')}] ★手脚增强: denoise={dn_use:.2f}, rounds={local_rounds}, cfg={cfg_use:.1f}, CN强度={cn_strength_override if cn_strength_override else 'default'}")

                                crop_lat, (lx1, ly1, lx2, ly2) = self._crop_latent(cur, bx1, by1, bx2, by2, pad_use, pixel_w, pixel_h)
                                if crop_lat is None:
                                    continue
                                crop_lat = crop_lat.clone()

                                # ★ 先定义ox_px/oy_px，后续ControlNet和mask都要用
                                ox_px = float(lx1) * 8.0
                                oy_px = float(ly1) * 8.0

                                crop_lh, crop_lw = crop_lat.shape[2], crop_lat.shape[3]
                                crop_ph, crop_pw = crop_lh*8, crop_lw*8

                                target_pw = int(round(crop_pw * float(upscale_use)))
                                target_ph = int(round(crop_ph * float(upscale_use)))
                                max_edge = int(max_edge_use)
                                if max_edge > 0:
                                    s = min(1.0, float(max_edge) / float(max(target_pw, target_ph)))
                                    target_pw = max(64, int(round(target_pw * s)))
                                    target_ph = max(64, int(round(target_ph * s)))
                                target_lw = max(8, int(round(target_pw / 8)))
                                target_lh = max(8, int(round(target_ph / 8)))

                                # V6.0：使用Latent感知插值放大裁剪
                                if use_latent_aware and ((target_lh != crop_lh) or (target_lw != crop_lw)):
                                    crop_lat_big = LatentAlgorithms.latent_aware_resize(crop_lat, target_lh, target_lw)
                                elif (target_lh != crop_lh) or (target_lw != crop_lw):
                                    crop_lat_big = F.interpolate(crop_lat, size=(target_lh, target_lw), mode="bilinear", align_corners=False)
                                else:
                                    crop_lat_big = crop_lat

                                # 裁剪conditioning
                                pos_crop, neg_crop = base_pos, base_neg

                                if crop_use_cn and has_cn:
                                    if CN图像.shape[0] > 1 and i < CN图像.shape[0]:
                                        ctrl_full = CN图像[i:i+1]
                                    else:
                                        ctrl_full = CN图像[0:1]
                                    cx1p = int(max(0, ox_px)); cy1p = int(max(0, oy_px))
                                    cx2p = int(min(pixel_w, float(lx2)*8.0)); cy2p = int(min(pixel_h, float(ly2)*8.0))
                                    ctrl_crop = ctrl_full[:, cy1p:cy2p, cx1p:cx2p, :]
                                    ctrl_crop = ControlNetHelper.prepare_control_image(ctrl_crop, target_lh*8, target_lw*8)
                                    # V6.11: 手脚区域使用专用CN强度（0.5-0.8效果更好，避免过度约束）
                                    cn_strength_use = cn_strength_override if cn_strength_override is not None else CN强度
                                    pos_crop, neg_crop = ControlNetHelper.apply_controlnet(
                                        base_pos, base_neg, CN模型, ctrl_crop, cn_strength_use, CN开始, CN结束
                                    )

                                # V6.11.3: 手骨架ControlNet（仅手部区域，自动提取MediaPipe骨架避免多指）
                                if is_hand_foot and has_hand_skeleton_cn and _decoded_np is not None:
                                    try:
                                        skeleton_np = self._extract_hand_skeleton(
                                            _decoded_np, d.get("bbox", [bx1, by1, bx2, by2]),
                                            target_lh * 8, target_lw * 8
                                        )
                                        if skeleton_np is not None:
                                            skeleton_t = torch.from_numpy(skeleton_np).unsqueeze(0)  # [1,H,W,3]
                                            pos_crop, neg_crop = ControlNetHelper.apply_controlnet(
                                                pos_crop, neg_crop,
                                                hand_skeleton_cn_model, skeleton_t,
                                                skeleton_cn_strength, 0.0, skeleton_cn_end
                                            )
                                            print(f"      [手骨架] ✓ 骨架CN注入成功，强度={skeleton_cn_strength:.2f} 结束={skeleton_cn_end:.1f}")
                                        else:
                                            print(f"      [手骨架] MediaPipe未检测到手部关键点，跳过骨架CN")
                                    except Exception as _sk_err:
                                        print(f"      [手骨架] 骨架提取失败（跳过）: {_sk_err}")

                                # ========== V6.0: Inpaint采样（mask外不漂移） ==========
                                step_use = int(step_use)
                                cfg_use = float(cfg_use)
                                current_crop = crop_lat_big
                                original_crop = crop_lat_big.clone()  # 保存原始用于边缘混合
                            
                                # 获取inpaint mask（latent级）- 增加羽化
                                l_expand = max(2, m_expand // 8)
                                l_feather = max(2.0, float(m_feather) / 8.0)  # 增大羽化值
                            
                                inpaint_mask = self._get_crop_inpaint_mask(
                                    d, target_lh, target_lw, (ox_px, oy_px),
                                    expand_latent=l_expand, feather_latent=l_feather, use_seg=use_seg
                                )

                                # V6.7: 对裁剪区域应用EMA注意力
                                if enable_ema_attention:
                                    # 为单个检测创建注意力权重
                                    crop_attention = self._calculate_ema_attention([d], target_lh * 8, target_lw * 8, attention_focus)
                                    # 下采样到latent尺寸
                                    crop_attention_t = crop_attention.unsqueeze(0).unsqueeze(0)
                                    crop_attention_lat = F.interpolate(crop_attention_t, size=(target_lh, target_lw), mode="bilinear", align_corners=False)
                                    crop_attention_lat = crop_attention_lat.squeeze()
                                    # V6.10.2修复：确保crop_attention_lat与inpaint_mask在同一设备上，避免CUDA非法内存访问
                                    crop_attention_lat = crop_attention_lat.to(inpaint_mask.device)
                                    # 应用注意力权重
                                    inpaint_mask = inpaint_mask * crop_attention_lat

                                # 为边缘混合准备更大羽化的mask
                                blend_mask = self._get_crop_inpaint_mask(
                                    d, target_lh, target_lw, (ox_px, oy_px),
                                    expand_latent=l_expand + 1, feather_latent=l_feather * 1.5, use_seg=use_seg
                                )

                                # V6.7: 对blend mask也应用EMA注意力
                                if enable_ema_attention:
                                    # V6.10.2修复：确保crop_attention_lat与blend_mask在同一设备上
                                    crop_attention_lat = crop_attention_lat.to(blend_mask.device)
                                    blend_mask = blend_mask * crop_attention_lat

                                blend_mask_4d = blend_mask.unsqueeze(0).unsqueeze(0).expand_as(current_crop).to(current_crop.device)
                                # V6.10性能优化：crop_mask_4d移到循环外，避免重复创建
                                crop_mask_4d = inpaint_mask.unsqueeze(0).unsqueeze(0).expand_as(current_crop).to(current_crop.device)

                                for local_i in range(max(1, local_rounds)):
                                    # V6.10: 检查中断信号（本地精修循环）
                                    comfy.model_management.throw_exception_if_processing_interrupted()

                                    # V6.11: 渐进式降噪（手脚区域使用优化的衰减策略）
                                    dn_local = float(dn_use)
                                    if local_i > 0 and decay:
                                        # 使用可配置的衰减系数（手脚区域用denoise_decay，其他用0.75）
                                        decay_factor = denoise_decay if is_hand_foot else 0.75
                                        dn_local = float(dn_local * (decay_factor ** local_i))
                                        # 应用下限（防止降噪过低导致无效果）
                                        dn_local = float(max(denoise_min, dn_local))

                                    # V6.11.2: 采样前确保所有tensor在GPU上
                                    if torch.cuda.is_available():
                                        if current_crop.device.type != 'cuda':
                                            current_crop = current_crop.to('cuda')
                                        if inpaint_mask.device.type != 'cuda':
                                            inpaint_mask = inpaint_mask.to('cuda')

                                    # V6.10：启用noise_mask精确噪点注入（只在mask区域加噪，mask外完全保持原样）
                                    print(f"      [V6.11.3] 采样 pass={pass_i+1} target={di+1}/{len(detections)} local={local_i+1} steps={step_use} denoise={dn_local:.2f}")
                                    sampled_crop = self._inpaint_sample(
                                        ksampler, 模型, seed + i + (pass_i*1000) + di + (local_i*100),
                                        step_use, cfg_use, 采样器, 调度器,
                                        pos_crop, neg_crop, current_crop, inpaint_mask, denoise=float(dn_local)
                                    )
                                    print(f"      [V6.11.3] 采样完成 pass={pass_i+1} target={di+1}")

                                    # V6.10优化：noise_mask已确保mask外不变，仅需轻量融合或直接使用
                                    # 后混合（V6.9：新增智能柔和融合）
                                    if blend_mode == "智能柔和融合":
                                        # 智能柔和融合：集成色彩对齐+细节恢复+清晰度增强
                                        try:
                                            current_crop = LatentAlgorithms.smart_soft_blend(
                                                sampled_crop, current_crop, crop_mask_4d[:, :1, :, :],
                                                color_strength=color_strength,
                                                clarity_strength=clarity_strength
                                            )
                                        except Exception as e:
                                            print(f"    [警告] 智能柔和融合失败: {e}，回退到简单混合")
                                            current_crop = current_crop * (1.0 - crop_mask_4d) + sampled_crop * crop_mask_4d
                                    elif blend_mode == "Poisson融合" and poisson_iters > 0:
                                        # Poisson融合：先色差修复，再梯度域融合
                                        if color_strength > 0:
                                            try:
                                                sampled_crop = LatentAlgorithms.frequency_aware_color_match(
                                                    sampled_crop, current_crop, crop_mask_4d[:, :1, :, :],
                                                    strength=color_strength
                                                )
                                            except Exception as e:
                                                print(f"    [警告] 色差修复失败: {e}")
                                                pass
                                        try:
                                            current_crop = LatentAlgorithms.poisson_blend_latent(
                                                sampled_crop, current_crop, crop_mask_4d, iterations=poisson_iters
                                            )
                                        except Exception as e:
                                            print(f"    [警告] Poisson融合失败: {e}，回退到简单混合")
                                            current_crop = current_crop * (1.0 - crop_mask_4d) + sampled_crop * crop_mask_4d
                                    elif blend_mode == "MPB增强融合":
                                        # MPB融合：先色差修复，再MPB混合
                                        if color_strength > 0:
                                            try:
                                                sampled_crop = LatentAlgorithms.frequency_aware_color_match(
                                                    sampled_crop, current_crop, crop_mask_4d[:, :1, :, :],
                                                    strength=color_strength
                                                )
                                            except Exception as e:
                                                print(f"    [警告] 色差修复失败: {e}")
                                                pass
                                        try:
                                            current_crop = LatentAlgorithms.mpb_blend_latent(
                                                sampled_crop, current_crop, crop_mask_4d, color_preserve=mpb_color_preserve
                                            )
                                        except Exception as e:
                                            print(f"    [警告] MPB融合失败: {e}，回退到简单混合")
                                            current_crop = current_crop * (1.0 - crop_mask_4d) + sampled_crop * crop_mask_4d
                                    else:
                                        # 简单混合/频率感知：先色差修复，再简单混合
                                        if color_strength > 0:
                                            try:
                                                sampled_crop = LatentAlgorithms.frequency_aware_color_match(
                                                    sampled_crop, current_crop, crop_mask_4d[:, :1, :, :],
                                                    strength=color_strength
                                                )
                                            except Exception as e:
                                                print(f"    [警告] 色差修复失败: {e}")
                                                pass
                                        try:
                                            current_crop = current_crop * (1.0 - crop_mask_4d) + sampled_crop * crop_mask_4d
                                        except Exception as e:
                                            print(f"    [警告] 简单混合失败: {e}，保持原crop")
                                            pass

                                    # 清理临时变量
                                    del sampled_crop

                                # V5.1：使用Latent感知插值缩回
                                blended = current_crop
                                if blended.shape[2] != crop_lh or blended.shape[3] != crop_lw:
                                    if use_latent_aware:
                                        blended = LatentAlgorithms.latent_aware_resize(blended, crop_lh, crop_lw)
                                    else:
                                        blended = F.interpolate(blended, size=(crop_lh, crop_lw), mode="bilinear", align_corners=False)

                                cur[:, :, ly1:ly2, lx1:lx2] = blended

                                del crop_lat, crop_lat_big, current_crop, original_crop, blended

                                # V6.11.3: 合并双重 empty_cache 为一次（原来每个目标后调用2次）
                                # empty_cache 本身是重操作（重新整理显存分配信息）
                                if getattr(self, '_low_vram_mode', False) or low_vram:
                                    import gc
                                    gc.collect()
                                    self._gc_if_needed(low_vram, clear_each)  # 内部已含 empty_cache
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                                    if getattr(self, '_low_vram_mode', False):
                                        free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)
                                        print(f"      [目标{di+1}/{len(detections)}完成] 可用显存: {free_mem/1024**2:.0f}MB")

                            # V6.10.5: 每轮结束后清理显存（pass_i循环结束）
                            if getattr(self, '_low_vram_mode', False):
                                self._gc_if_needed(True, True)
                                if torch.cuda.is_available():
                                    free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)
                                    print(f"    [第{pass_i+1}轮完成] 可用显存: {free_mem/1024**2:.0f}MB")

                        processed_latent = cur

                    else:
                        # 全图mask修复（无裁剪精修时）
                        try:
                            latent_mask = F.interpolate(detection_mask.unsqueeze(0).unsqueeze(0),
                                                        size=(latent_h, latent_w),
                                                        mode="bilinear", align_corners=False)
                            # 确保是2D
                            while latent_mask.dim() > 2:
                                latent_mask = latent_mask.squeeze(0)
                            latent_mask = latent_mask.clamp(0.0, 1.0)
                        except Exception as e:
                            print(f"    [警告] latent_mask处理失败: {e}")
                            latent_mask = torch.ones((latent_h, latent_w), dtype=torch.float32)

                        step_use = int(步数 if region_steps <= 0 else region_steps)
                        cfg_use = float(CFG if region_cfg <= 0.0 else region_cfg)

                        cur = processed_latent

                        # V6.10性能优化：安全创建4D mask（避免重复计算）
                        try:
                            latent_mask_4d = latent_mask.unsqueeze(0).unsqueeze(0)
                            if latent_mask_4d.shape[2:] != cur.shape[2:]:
                                latent_mask_4d = F.interpolate(latent_mask_4d, size=cur.shape[2:], mode="bilinear", align_corners=False)
                            latent_mask_4d = latent_mask_4d.expand_as(cur).to(cur.device)
                        except Exception as e:
                            print(f"    [警告] latent_mask_4d创建失败: {e}")
                            latent_mask_4d = torch.ones_like(cur)
                    
                        for r in range(max(1, fix_rounds)):
                            # V6.10: 检查中断信号（全图mask修复循环）
                            comfy.model_management.throw_exception_if_processing_interrupted()

                            dn = float(region_denoise)
                            if r > 0 and decay:
                                dn = float(max(0.0, dn * (0.75 ** r)))
                        
                            # V6.10：启用noise_mask精确噪点注入（全图mask修复）
                            original = cur.clone()
                            sampled = self._inpaint_sample(
                                ksampler, 模型, seed + i + r*1000,
                                step_use, cfg_use, 采样器, 调度器,
                                pos_full, neg_full, cur, latent_mask, denoise=float(dn)
                            )

                            # 后混合（V6.9：新增智能柔和融合）
                            if blend_mode == "智能柔和融合":
                                # 智能柔和融合：集成色彩对齐+细节恢复+清晰度增强
                                try:
                                    cur = LatentAlgorithms.smart_soft_blend(
                                        sampled, original, latent_mask_4d[:, :1, :, :],
                                        color_strength=color_strength,
                                        clarity_strength=clarity_strength
                                    )
                                except Exception as e:
                                    print(f"    [警告] 智能柔和融合失败: {e}，回退到简单混合")
                                    cur = original * (1.0 - latent_mask_4d) + sampled * latent_mask_4d
                            elif blend_mode == "Poisson融合" and poisson_iters > 0:
                                # Poisson融合：先色差修复，再梯度域融合
                                if color_strength > 0:
                                    try:
                                        sampled = LatentAlgorithms.frequency_aware_color_match(
                                            sampled, original, latent_mask_4d[:, :1, :, :], strength=color_strength
                                        )
                                    except Exception:
                                        pass
                                cur = LatentAlgorithms.poisson_blend_latent(sampled, original, latent_mask_4d, iterations=poisson_iters)
                            elif blend_mode == "MPB增强融合":
                                # MPB融合：先色差修复，再MPB混合
                                if color_strength > 0:
                                    try:
                                        sampled = LatentAlgorithms.frequency_aware_color_match(
                                            sampled, original, latent_mask_4d[:, :1, :, :], strength=color_strength
                                        )
                                    except Exception:
                                        pass
                                cur = LatentAlgorithms.mpb_blend_latent(sampled, original, latent_mask_4d, color_preserve=mpb_color_preserve)
                            else:
                                # 简单混合/频率感知：先色差修复，再简单混合
                                if color_strength > 0:
                                    try:
                                        sampled = LatentAlgorithms.frequency_aware_color_match(
                                            sampled, original, latent_mask_4d[:, :1, :, :], strength=color_strength
                                        )
                                    except Exception:
                                        pass
                                cur = original * (1.0 - latent_mask_4d) + sampled * latent_mask_4d
                        
                            # V6.10.5: 每轮全图修复后清理
                            if getattr(self, '_low_vram_mode', False) or low_vram:
                                del original, sampled
                                self._gc_if_needed(True, True)

                            if low_vram:
                                self._gc_if_needed(low_vram, clear_each)
                                # V6.10: 显式清理CUDA缓存
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                        processed_latent = cur

                else:
                    # 没检测到目标
                    if no_detect_mode == "全图重绘":
                        # V6.10.1: 添加异常处理和内存管理
                        try:
                            # 清理CUDA缓存
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                            single_latent = {"samples": processed_latent}
                            res = ksampler.sample(
                                模型, seed + i, 步数, CFG, 采样器, 调度器,
                                pos_full, neg_full, single_latent, denoise=float(降噪)
                            )
                            processed_latent = res[0]["samples"]

                        except Exception as e:
                            print(f"    [全图重绘] 失败，保持原latent: {e}")
                            # 清理CUDA缓存
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                # ★ V6.1: 修复后全图降噪（可选）
                if work_mode == "修复后全图降噪":
                    print(f"    [全图降噪] 执行全图降噪，强度={final_denoise_strength}")

                    # V6.10.1: 优化报错修复 - 添加异常处理和内存管理
                    try:
                        # 清理CUDA缓存，防止OOM错误
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        # 确保processed_latent在正确的设备上
                        if hasattr(processed_latent, 'device'):
                            processed_latent = processed_latent.contiguous()

                        single_latent = {"samples": processed_latent}
                        res = ksampler.sample(
                            模型, seed + i + 99999, 步数, CFG, 采样器, 调度器,
                            pos_full, neg_full, single_latent, denoise=float(final_denoise_strength)
                        )
                        processed_latent = res[0]["samples"]
                        print(f"    [全图降噪] 完成")

                    except RuntimeError as e:
                        error_msg = str(e)
                        if 'out of memory' in error_msg.lower() or 'cuda' in error_msg.lower():
                            print(f"    [全图降噪] CUDA内存错误，跳过全图降噪: {e}")
                            # 强制清理缓存
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            import gc
                            gc.collect()
                            print(f"    [全图降噪] 已跳过，保持修复后的结果")
                        else:
                            print(f"    [全图降噪] 运行时错误，跳过全图降噪: {e}")

                    except Exception as e:
                        print(f"    [全图降噪] 发生异常，跳过全图降噪: {e}")
                        print(f"    [全图降噪] 已跳过，保持修复后的结果")
                        import traceback
                        traceback.print_exc()

                # V6.7: 视频模式时序平滑
                if video_mode and frame_history and temporal_strength > 0:
                    # 使用历史帧加权平均进行平滑
                    weights = []
                    historical_latents = []

                    # 收集历史帧
                    for hist_idx in range(min(frame_window, len(frame_history))):
                        hist_latent = frame_history[-(hist_idx + 1)]
                        # 距离越近权重越大
                        weight = (1.0 / (hist_idx + 2)) ** 1.5
                        weights.append(weight)
                        historical_latents.append(hist_latent)

                    if historical_latents:
                        # 归一化权重
                        total_weight = sum(weights)
                        weights = [w / total_weight for w in weights]

                        # 计算历史帧的加权平均
                        hist_avg = torch.zeros_like(processed_latent)
                        for hist_lat, w in zip(historical_latents, weights):
                            hist_avg += hist_lat * w

                        # 将当前帧与历史平均混合
                        processed_latent = processed_latent * (1.0 - temporal_strength) + hist_avg * temporal_strength

                # 保存到历史（仅保留最近的帧）
                if video_mode:
                    frame_history.append(processed_latent.detach().clone())
                    if len(frame_history) > frame_window:
                        frame_history.pop(0)

                out_latents[i:i+1] = processed_latent.detach().to(out_device)

                if preview_image is None:
                    preview_image = self._decode_latent_to_image(VAE, processed_latent)[0]
                out_previews.append(preview_image.unsqueeze(0))

                out_masks.append(detection_mask)
                all_dets.extend(detections)

                # V6.10.7: 记录已完成数量
                processed_count = i + 1

                # 清理批次中的临时变量
                del preview_image, detection_mask, detections, processed_latent, base_latent

                if (i + 1) % 2 == 0:
                    self._gc_if_needed(low_vram, clear_each)
                    # 每处理2个批次清理一次 CUDA 缓存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        except InterruptProcessingException:
            # V6.10.7: 捕获中断异常，返回已处理的结果而不是黑图
            interrupted = True
            print(f"\n{'='*60}")
            print(f"[中断] 用户停止处理")
            print(f"[中断] 已完成: {processed_count}/{batch_size} 张")
            print(f"[中断] 返回已处理的结果（未处理的部分保持原图）")
            print(f"{'='*60}")

            # 对于未处理的批次，使用原始输入latent
            if out_latents is not None:
                for j in range(processed_count, batch_size):
                    # 如果latent尺寸不同（因为有放大），需要调整原始latent
                    orig_sample = original_samples[j:j+1]
                    if 放大倍数 > 1.0:
                        if 放大算法 == "latent_aware" or use_latent_aware:
                            target_h = out_latents.shape[2]
                            target_w = out_latents.shape[3]
                            orig_sample = LatentAlgorithms.latent_aware_resize(orig_sample, target_h, target_w)
                        else:
                            orig_sample = self.upscale_latent(orig_sample, 放大算法, 放大倍数)
                    out_latents[j:j+1] = orig_sample.detach().to(out_device)

                    # 为未处理的批次创建空preview和mask
                    empty_preview = torch.zeros((1, 64, 64, 3))
                    out_previews.append(empty_preview)
                    out_masks.append(torch.zeros((64, 64)))

        # 合并输出
        # V6.10.7: 处理中断导致out_latents为None的情况
        if out_latents is None:
            # 如果在第一张图处理前就中断了，返回原始输入（可能需要放大）
            print(f"[中断] 未完成任何处理，返回原始输入")
            if 放大倍数 > 1.0:
                if 放大算法 == "latent_aware" or use_latent_aware:
                    target_h = int(samples.shape[2] * 放大倍数)
                    target_w = int(samples.shape[3] * 放大倍数)
                    out_latents = LatentAlgorithms.latent_aware_resize(original_samples, target_h, target_w)
                else:
                    out_latents = self.upscale_latent(original_samples, 放大算法, 放大倍数)
            else:
                out_latents = original_samples.clone()

            # 创建空的preview和mask
            for _ in range(batch_size):
                out_previews.append(torch.zeros((1, 64, 64, 3)))
                out_masks.append(torch.zeros((64, 64)))

        final_samples = out_latents.to(samples.device) if out_device != samples.device else out_latents
        final_preview = torch.cat(out_previews, dim=0) if out_previews else torch.zeros((batch_size, 64, 64, 3))
        
        max_h = max(m.shape[0] for m in out_masks) if out_masks else 64
        max_w = max(m.shape[1] for m in out_masks) if out_masks else 64
        masks_resized = []
        for m in out_masks:
            if m.shape[0] != max_h or m.shape[1] != max_w:
                mm = F.interpolate(m.unsqueeze(0).unsqueeze(0), size=(max_h, max_w), mode="bilinear", align_corners=False).squeeze()
            else:
                mm = m
            masks_resized.append(mm)
        final_masks = torch.stack(masks_resized, dim=0) if masks_resized else torch.zeros((batch_size, max_h, max_w))

        # V6.10.7: 根据是否中断显示不同信息
        if interrupted:
            info_lines = [
                f"[已中断] 处理: {processed_count}/{batch_size} 张",
                f"工作模式: {work_mode}",
                f"融合模式: {blend_mode} | Latent感知插值: {'ON' if use_latent_aware else 'OFF'}",
                f"放大: {放大倍数}x | 输出latent: {tuple(final_samples.shape)}",
                f"YOLO: {'ON' if enable_yolo else 'OFF'} | 裁剪精修: {'ON' if enable_crop else 'OFF'}",
                f"检测总数(使用后): {len(all_dets)}",
                f"[提示] 未处理的图像已使用原图填充",
            ]
        else:
            info_lines = [
                f"处理完成: {batch_size} 张",
                f"工作模式: {work_mode}",
                f"融合模式: {blend_mode} | Latent感知插值: {'ON' if use_latent_aware else 'OFF'}",
                f"放大: {放大倍数}x | 输出latent: {tuple(final_samples.shape)}",
                f"YOLO: {'ON' if enable_yolo else 'OFF'} | 裁剪精修: {'ON' if enable_crop else 'OFF'}",
                f"检测总数(使用后): {len(all_dets)}",
            ]
        if all_dets:
            class_counts = {}
            for d in all_dets:
                name = d.get("class_name","unknown")
                class_counts[name] = class_counts.get(name, 0) + 1
            for k,v in class_counts.items():
                info_lines.append(f"  - {k}: {v}")

        info = "\n".join(info_lines)

        # V6.6: 性能统计
        if enable_profiling and perf_start_time is not None:
            total_time = time.time() - perf_start_time
            info_lines.append(f"\n[性能统计] 总耗时: {total_time:.2f}秒 | 平均每帧: {total_time/batch_size:.2f}秒")
            # 显示显存信息
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**2
                reserved = torch.cuda.memory_reserved() / 1024**2
                info_lines.append(f"[性能统计] 显存: 已分配 {allocated:.0f}MB | 已保留 {reserved:.0f}MB")
            info = "\n".join(info_lines)

        print(f"\n{'='*60}\n{info}\n{'='*60}\n")
        return ({"samples": final_samples}, final_preview, final_masks, info)
