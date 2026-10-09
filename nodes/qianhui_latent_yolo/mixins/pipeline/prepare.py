"""process() 阶段 1：配置解析与运行前准备（源 4131-4385）。"""
import os

import nodes
import torch

from ...model_managers import YOLOModelManager
from ..context import YoloConfig, YoloRunContext


def parse_config(kwargs) -> YoloConfig:
    """99 个 kwargs 解析结果（源 4135-4288）。字段名 = 原局部变量名。"""
    cfg = YoloConfig()

    # ControlNet
    cfg.CN模型 = kwargs.get('CN模型', None)
    cfg.CN图像 = kwargs.get('CN图像', None)
    cfg.CN强度 = float(kwargs.get('CN强度', 1.0))
    cfg.CN开始 = float(kwargs.get('CN开始', 0.0))
    cfg.CN结束 = float(kwargs.get('CN结束', 1.0))

    # 低显存/性能
    cfg.low_vram = bool(kwargs.get("低显存模式", False))
    cfg.clear_each = bool(kwargs.get("每帧清理缓存", False))
    cfg.cache_device_mode = (kwargs.get("结果缓存设备", "自动") or "自动").strip()
    cfg.enable_profiling = bool(kwargs.get("启用性能分析", False))

    # 工作模式
    cfg.work_mode = (kwargs.get("工作模式", "仅修复(保持输入Latent)") or "仅修复(保持输入Latent)").strip()
    cfg.final_denoise_strength = float(kwargs.get("全图降噪强度", 0.25) or 0.25)

    # YOLO参数（V6.9优化：每个模型独立置信度）
    cfg.enable_yolo = bool(kwargs.get('启用YOLO', True))
    cfg.yolo_model_selection = kwargs.get('YOLO模型', '')
    cfg.manual_path = kwargs.get('YOLO路径', '')
    cfg.enable_multi_model = bool(kwargs.get('启用多模型', False))
    cfg.yolo_model2 = kwargs.get('YOLO模型2', '[不使用]')
    cfg.yolo_model3 = kwargs.get('YOLO模型3', '[不使用]')
    cfg.yolo_model4 = kwargs.get('YOLO模型4', '[不使用]')
    cfg.confidence = float(kwargs.get('置信度', 0.25))  # 主置信度（模型1）
    cfg.confidence2 = float(kwargs.get('YOLO模型2置信度', 0.0))  # 模型2独立置信度
    cfg.confidence3 = float(kwargs.get('YOLO模型3置信度', 0.0))  # 模型3独立置信度
    cfg.confidence4 = float(kwargs.get('YOLO模型4置信度', 0.0))  # 模型4独立置信度
    cfg.target_classes_str = kwargs.get('目标类别', '')
    cfg.yolo_imgsz = int(kwargs.get("YOLO推理尺寸", 768) or 768)
    cfg.multi_scale = bool(kwargs.get("多尺度检测", True))
    cfg.inference_precision = kwargs.get("推理精度", "FP32(默认)")

    # V6.7 WIoU 优化
    cfg.use_wiou = bool(kwargs.get("使用WIoU优化", False))
    cfg.wiou_alpha = float(kwargs.get("WIoU权重参数", 3.0) or 3.0)

    # V6.7 SAM 集成
    cfg.enable_sam = bool(kwargs.get("启用SAM精细化", False))
    # V6.8: SAM模型路径选择
    cfg.sam_model_selection = kwargs.get("SAM模型选择", "")
    cfg.sam_model_manual = kwargs.get("SAM模型手动路径", "")
    cfg.sam_conf_threshold = float(kwargs.get("SAM置信度阈值", 0.7) or 0.7)

    # V6.7 视频模式
    cfg.video_mode = bool(kwargs.get("视频模式", False))
    cfg.temporal_strength = float(kwargs.get("时序一致性强度", 0.3) or 0.3)
    cfg.frame_window = int(kwargs.get("参考帧窗口", 2) or 2)

    # V6.7 EMA 注意力
    cfg.enable_ema_attention = bool(kwargs.get("启用EMA注意力", False))
    cfg.attention_focus = float(kwargs.get("注意力聚焦强度", 1.5) or 1.5)

    # V6.0 核心参数
    cfg.tile_enabled = bool(kwargs.get("启用Tile检测", True))
    cfg.tile_size = int(kwargs.get("Tile大小", 640) or 640)
    cfg.tile_overlap = float(kwargs.get("Tile重叠", 0.3) or 0.3)
    cfg.use_seg = bool(kwargs.get("使用Seg分割", True))
    cfg.seg_feather = float(kwargs.get("Seg边缘羽化", 3.0) or 3.0)
    # use_inpaint功能已移除（与某些插件冲突导致CUDA崩溃）

    # 合并/过滤等
    cfg.merge_boxes = bool(kwargs.get('合并重叠框', True))
    cfg.merge_iou = float(kwargs.get('合并IOU阈值', 0.35))
    cfg.merge_same_class = bool(kwargs.get('合并仅同类', True))
    cfg.max_targets = int(kwargs.get('最多目标数', 0) or 0)
    cfg.min_area_ratio = float(kwargs.get('过滤最小面积比', 0.0))
    cfg.max_area_ratio = float(kwargs.get('过滤最大面积比', 1.0))
    cfg.min_aspect = float(kwargs.get('过滤最小长宽比', 0.0))
    cfg.max_aspect = float(kwargs.get('过滤最大长宽比', 10.0))
    cfg.sort_mode = kwargs.get('排序方式', '置信度')
    cfg.per_class_limit = int(kwargs.get('每类最多目标数', 0) or 0)
    cfg.lr_protect = bool(kwargs.get('左右分组保护', True))
    cfg.bbox_expand2_px = int(kwargs.get('合并后二次扩张像素', 0) or 0)
    cfg.bbox_expand2_ratio = float(kwargs.get('合并后二次扩张比例', 0.0))

    # 智能策略
    cfg.smart_strategy = (kwargs.get("智能策略", "人像(脸/手/脚)") or "关闭").strip()

    # V6.4 手脚增强（含提示词引导）
    cfg.hand_foot_enhance = bool(kwargs.get("手脚增强模式", True))
    cfg.hand_foot_denoise = float(kwargs.get("手脚降噪下限", 0.80) or 0.80)
    cfg.hand_foot_extra_rounds = int(kwargs.get("手脚额外轮次", 1) or 1)
    # V6.11 新增手脚优化参数
    cfg.hand_foot_cfg_boost = float(kwargs.get("手脚CFG加成", 1.2) or 1.2)
    cfg.hand_foot_cn_strength = float(kwargs.get("手脚CN强度", 0.7) or 0.7)
    cfg.hand_foot_denoise_decay = float(kwargs.get("手脚降噪衰减", 0.85) or 0.85)
    cfg.hand_foot_denoise_min = float(kwargs.get("手脚衰减下限", 0.40) or 0.40)
    # V6.11.3 手骨架ControlNet参数
    cfg.hand_skeleton_cn_model = kwargs.get("手骨架CN模型", None)
    cfg.skeleton_cn_strength = float(kwargs.get("骨架CN强度", 0.65) or 0.65)
    cfg.skeleton_cn_end = float(kwargs.get("骨架CN结束", 0.7) or 0.7)
    # 提示词增强已移除（导致CUDA崩溃）

    # 优先级/规则/自适应
    cfg.priority_mode = (kwargs.get('修复优先级', '脸优先') or '脸优先').strip()
    cfg.custom_priority = kwargs.get('自定义优先级关键词', 'face,hand,foot') or ''
    cfg.adaptive_dn = bool(kwargs.get('按置信度自适应降噪', True))
    cfg.conf_low_scale = float(kwargs.get('自适应降噪_低置信缩放', 0.85) or 0.85)
    cfg.conf_high_scale = float(kwargs.get('自适应降噪_高置信缩放', 1.0) or 1.0)

    # V6.5: 类别规则预设处理
    cfg.class_rules_preset = kwargs.get('类别规则预设', '[不使用预设]')
    cfg.class_rules_text = kwargs.get('类别规则', '')

    # 遮罩
    cfg.expand = int(kwargs.get('遮罩扩展', 32))
    cfg.feather = int(kwargs.get('遮罩羽化', 24))

    # 修复采样参数
    cfg.region_steps = int(kwargs.get("区域步数", 0) or 0)
    cfg.region_cfg = float(kwargs.get("区域CFG", 0.0) or 0.0)
    cfg.region_denoise = float(kwargs.get('区域降噪', 0.6))
    cfg.fix_rounds = int(kwargs.get("修复轮次", 1) or 1)
    cfg.decay = bool(kwargs.get("逐次降噪递减", True))
    cfg.no_detect_mode = kwargs.get('未检测时处理', '保持原样')

    # 裁剪精修
    cfg.enable_crop = bool(kwargs.get("开启裁剪精修", True))
    cfg.crop_pad = int(kwargs.get("裁剪扩展", 64))
    cfg.crop_max_edge = int(kwargs.get("裁剪最大边", 512))
    cfg.crop_upscale = float(kwargs.get("裁剪放大倍率", 1.6))
    cfg.crop_steps = int(kwargs.get("裁剪步数", 20))
    cfg.crop_cfg = float(kwargs.get("裁剪CFG", 0.0) or 0.0)
    cfg.crop_denoise = float(kwargs.get("裁剪降噪", 0.5))
    cfg.crop_rounds = int(kwargs.get("裁剪轮次", 2))
    cfg.crop_use_cn = bool(kwargs.get("裁剪使用ControlNet", True))

    # 二次检测
    cfg.refine_det = bool(kwargs.get("二次检测精修", False))
    cfg.refine_max = int(kwargs.get("二次检测最多目标", 6))
    cfg.refine_pad = int(kwargs.get("二次检测扩展", 32))

    # V5.1 新增融合选项（V6.9增强：智能柔和融合）
    cfg.blend_mode = (kwargs.get("融合模式", "智能柔和融合") or "智能柔和融合").strip()
    cfg.poisson_iters = int(kwargs.get("Poisson迭代次数", 40) or 40)
    cfg.mpb_color_preserve = float(kwargs.get("MPB颜色保持", 0.5) or 0.5)
    cfg.color_strength = float(kwargs.get("色差修复强度", 0.85) or 0.85)
    cfg.clarity_strength = float(kwargs.get("清晰度恢复强度", 0.3) or 0.3)
    cfg.use_latent_aware = bool(kwargs.get("使用Latent感知插值", True))

    from ...presets import apply_preset, NO_PRESET
    # 预设名从 kwargs 取，**不是**从 cfg 取 —— 见下方警告。
    apply_preset(cfg, kwargs.get("配置预设") or NO_PRESET)

    return cfg


def prepare_run(node, kwargs, Latent, 模型, VAE, 正面提示, 负面提示, seed, 步数, CFG, 采样器, 调度器, 降噪, 放大算法, 放大倍数) -> YoloRunContext:
    """规则预设 / 模型列表 / 缓存设备 / ksampler 等运行前准备（源 4131-4385 的非配置部分）。"""
    cfg = parse_config(kwargs)
    ctx = YoloRunContext(cfg=cfg, 模型=模型, VAE=VAE, 正面提示=正面提示, 负面提示=负面提示,
                         seed=seed, 步数=步数, CFG=CFG, 采样器=采样器, 调度器=调度器,
                         降噪=降噪, 放大算法=放大算法, 放大倍数=放大倍数)

    ctx.has_cn = ctx.cfg.CN模型 is not None and ctx.cfg.CN图像 is not None

    # V6.6: 性能分析器
    import time
    ctx.perf_start_time = time.time() if ctx.cfg.enable_profiling else None
    ctx.sam_model_path = node._resolve_sam_model_path(ctx.cfg.sam_model_selection, ctx.cfg.sam_model_manual)
    ctx.has_hand_skeleton_cn = ctx.cfg.hand_skeleton_cn_model is not None

    # 如果选择了预设，应用预设规则
    if ctx.cfg.class_rules_preset and ctx.cfg.class_rules_preset != "[不使用预设]" and ctx.cfg.class_rules_preset != "自定义规则":
        ctx.cfg.class_rules_text = node._get_preset_class_rules(ctx.cfg.class_rules_preset)
    ctx.class_rules = node._parse_class_rules(ctx.cfg.class_rules_text)
    ctx.target_classes = node._parse_target_classes(ctx.cfg.target_classes_str)
    ctx.samples = Latent["samples"]
    ctx.batch_size = ctx.samples.shape[0]
    print(f"\n{'='*60}")
    print(f"[Latent YOLO智能处理+ControlNet V6.11.3] 批量: {ctx.batch_size}")
    print(f"  工作模式: {ctx.cfg.work_mode}" + (f" (降噪={ctx.cfg.final_denoise_strength})" if ctx.cfg.work_mode == "修复后全图降噪" else ""))
    print(f"  放大: {ctx.放大倍数}x | 基础降噪: {ctx.降噪} | 色差修复: {ctx.cfg.color_strength}")
    print(f"  融合模式: {ctx.cfg.blend_mode} | Latent感知插值: {'ON' if ctx.cfg.use_latent_aware else 'OFF'} | 裁剪精修: {'ON' if ctx.cfg.enable_crop else 'OFF'}")
    print(f"  V6核心: Tile={'ON' if ctx.cfg.tile_enabled else 'OFF'} | Seg={'ON' if ctx.cfg.use_seg else 'OFF'} | 多尺度={'ON' if ctx.cfg.multi_scale else 'OFF'}")
    if ctx.cfg.use_wiou:
        print(f"  ★WIoU优化: 启用 (α={ctx.cfg.wiou_alpha}) | 提升检测框精度10-15%")
    if ctx.cfg.enable_sam:
        sam_model_name = os.path.basename(ctx.sam_model_path) if ctx.sam_model_path else "[未设置]"
        print(f"  ★SAM精细化: {sam_model_name} | 阈值={ctx.cfg.sam_conf_threshold} | 超高精度分割")
    if ctx.cfg.video_mode:
        print(f"  ★视频模式: 时序平滑={ctx.cfg.temporal_strength} | 参考帧={ctx.cfg.frame_window} | 帧间一致性优化")
    if ctx.cfg.hand_foot_enhance:
        print(f"  ★手脚增强: 降噪≥{ctx.cfg.hand_foot_denoise}, +{ctx.cfg.hand_foot_extra_rounds}轮, CFG×{ctx.cfg.hand_foot_cfg_boost}, CN={ctx.cfg.hand_foot_cn_strength}")
    if ctx.has_hand_skeleton_cn:
        print(f"  ★手骨架CN: 强度={ctx.cfg.skeleton_cn_strength}, 结束={ctx.cfg.skeleton_cn_end} (MediaPipe自动提取骨架)")
    if ctx.has_cn:
        print(f"  ControlNet: 强度={ctx.cfg.CN强度}, 范围=[{ctx.cfg.CN开始}, {ctx.cfg.CN结束}]")

    # 初始化YOLO（V6.9优化：支持每个模型独立置信度）
    ctx.yolo_models_to_use = []
    ctx.yolo_confidences_to_use = []  # V6.9新增：独立置信度列表
    if ctx.cfg.enable_yolo:
        # 首先添加主模型
        resolved_path = node._resolve_model_path(ctx.cfg.yolo_model_selection, ctx.cfg.manual_path)
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

    # 输出缓存设备策略
    if ctx.cfg.cache_device_mode == "CPU":
        ctx.out_device = torch.device("cpu")
    elif ctx.cfg.cache_device_mode == "GPU":
        ctx.out_device = ctx.samples.device
    else:
        ctx.out_device = torch.device("cpu") if ctx.cfg.low_vram else ctx.samples.device
    ctx.out_latents = None
    ctx.out_previews = []
    ctx.out_masks = []
    ctx.all_dets = []

    # V6.10.8: 重置CUDA损坏标志（每次新执行时重置）
    node._cuda_corrupted = False
    ctx.ksampler = nodes.KSampler()

    # V6.6: 进度条支持
    try:
        from comfy.utils import ProgressBar
        ctx.pbar = ProgressBar(ctx.batch_size)
    except:
        ctx.pbar = None

    # V6.7: 视频模式帧缓存
    ctx.frame_history = [] if ctx.cfg.video_mode else None

    # V6.10.7: 中断保护 - 保存中间结果防止黑图
    ctx.interrupted = False
    ctx.processed_count = 0
    # 保存原始输入latent作为fallback
    ctx.original_samples = ctx.samples.clone()

    return ctx
