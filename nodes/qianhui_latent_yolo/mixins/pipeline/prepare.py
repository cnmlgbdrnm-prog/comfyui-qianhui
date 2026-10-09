"""process() 阶段 1：配置解析与运行前准备（源 4131-4385）。"""
import os

import nodes
import torch

from ...model_managers import YOLOModelManager
from ..context import YoloConfig


def parse_config(kwargs, node_key=None) -> YoloConfig:
    """99 个 kwargs 解析结果（源 4135-4288）。字段名 = 原局部变量名。

    `node_key`（T12）：本调用方的节点短名（`presets.NODE_OF` 的值）。
    它只影响**预设**那一步 —— 4 个节点各自只有一部分控件，套预设时只能写
    自己的那些键（理由见 presets.NODE_KEYS 的注释）。99 个 `.get` 一个不动：
    本节点没声明的控件在这里会拿到与控件默认值相同的默认值，cfg 仍然是满的。
    """
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

    # 延迟导入（import 留在函数内）：`presets` 不能成为 node.py 的**模块级**硬依赖，
    # 否则 presets.py 一坏，整个包 import 失败，节点直接在菜单里消失（da27cc3 的初衷）。
    # 但只把 import 挪进来还不够：那样坏掉时症状会从「节点消失」翻成
    # 「节点在，但每次执行都抛 ImportError」—— 降级方向被弄反了。
    # 所以这里补上兜底：**预设后端不可用时降级为「不套用预设」**，节点照常跑。
    # 兜底不是静默：根部 __init__.py 的 `# 8c.` 块 import 同一个模块，
    # 失败时会在启动日志里打 ❌（该 print 是本计划早先加的）。
    try:
        from ...presets import apply_preset, NO_PRESET
    except Exception:
        # 故意不写 `except ImportError`：SyntaxError 不是 ImportError 的子类，
        # 而「presets.py 写坏了」最常见的形态恰恰是语法错误 —— 只接 ImportError
        # 会让这种坏法重新变成「每次执行都炸」。
        apply_preset = None
        # NO_PRESET 置 None 而不是再抄一份 "[不使用预设]" 字面量：
        # 它唯一的消费者就是上面这个 apply_preset，而这里它恒为 None，
        # 所以这个值永远不会被读到 —— 抄字面量只会多一处可能漂移的重复。
        NO_PRESET = None

    if apply_preset is not None:
        # 预设名从 kwargs 取，**不是**从 cfg 取（cfg 里没有 `配置预设` 这个字段）。
        # T12：node_key 只限定**写哪几个键**，不改这个取值路径。
        apply_preset(cfg, kwargs.get("配置预设") or NO_PRESET, node_key=node_key)

    return cfg
