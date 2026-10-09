"""process() 阶段间的状态容器。

YoloConfig: 99 个 kwargs 解析结果，字段名与原局部变量名逐字一致，
           使阶段搬运成为机械替换（局部名 → cfg.局部名）。
DetectCtx / SamCtx / RepairCtx / GlobalCtx: T6 新增的 4 个**精简 Context** ——
           拆出的 4 个节点各拿自己那部分字段，见文件末尾。

★ 所有类都带 `slots=True`，这不是性能优化，是**防笔误**：
   字段名是「原局部变量名逐字照搬」来的（`region_steps` / `region_step`、
   `handoff` / `hand_off` 这种一字之差肉眼极难发现）。普通 dataclass 上
   `ctx.region_step = 5` 会**静默新建一个属性**，真字段一动不动 ——
   与本计划已经踩过一次的 Critical 是同一种失败形状。
   `slots=True` 让「写未声明属性」当场 `AttributeError`，把静默改判成崩溃。
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


@dataclass(slots=True)
class YoloConfig:
    """99 个 kwargs 解析结果。字段名 = 原局部变量名，逐字不改写。

    分组（与原文件注释一致）：ControlNet / 低显存性能 / 工作模式 /
    YOLO核心 / 高级检测 / 视频 / EMA / Tile / 合并过滤 / 智能策略 /
    手脚增强 / 骨架CN / 优先级规则 / 遮罩 / 修复采样 / 裁剪精修 /
    二次检测 / 融合。
    """

    # ---- ControlNet ----
    CN模型: Any = None   # 键: CN模型
    CN图像: Any = None   # 键: CN图像
    CN强度: float = 1.0   # 键: CN强度
    CN开始: float = 0.0   # 键: CN开始
    CN结束: float = 1.0   # 键: CN结束

    # ---- 低显存 / 性能 ----
    low_vram: bool = False   # 键: 低显存模式
    clear_each: bool = False   # 键: 每帧清理缓存
    cache_device_mode: str = '自动'   # 键: 结果缓存设备
    enable_profiling: bool = False   # 键: 启用性能分析

    # ---- 工作模式 ----
    work_mode: str = '仅修复(保持输入Latent)'   # 键: 工作模式
    final_denoise_strength: float = 0.25   # 键: 全图降噪强度

    # ---- YOLO 核心 ----
    enable_yolo: bool = True   # 键: 启用YOLO
    yolo_model_selection: str = ''   # 键: YOLO模型
    manual_path: str = ''   # 键: YOLO路径
    enable_multi_model: bool = False   # 键: 启用多模型
    yolo_model2: str = '[不使用]'   # 键: YOLO模型2
    yolo_model3: str = '[不使用]'   # 键: YOLO模型3
    yolo_model4: str = '[不使用]'   # 键: YOLO模型4
    confidence: float = 0.25   # 键: 置信度
    confidence2: float = 0.0   # 键: YOLO模型2置信度
    confidence3: float = 0.0   # 键: YOLO模型3置信度
    confidence4: float = 0.0   # 键: YOLO模型4置信度
    target_classes_str: str = ''   # 键: 目标类别

    # ---- 高级检测 ----
    yolo_imgsz: int = 768   # 键: YOLO推理尺寸
    multi_scale: bool = True   # 键: 多尺度检测
    inference_precision: str = 'FP32(默认)'   # 键: 推理精度
    use_wiou: bool = False   # 键: 使用WIoU优化
    wiou_alpha: float = 3.0   # 键: WIoU权重参数
    enable_sam: bool = False   # 键: 启用SAM精细化
    sam_model_selection: str = ''   # 键: SAM模型选择
    sam_model_manual: str = ''   # 键: SAM模型手动路径
    sam_conf_threshold: float = 0.7   # 键: SAM置信度阈值

    # ---- 视频 ----
    video_mode: bool = False   # 键: 视频模式
    temporal_strength: float = 0.3   # 键: 时序一致性强度
    frame_window: int = 2   # 键: 参考帧窗口

    # ---- EMA 注意力 ----
    enable_ema_attention: bool = False   # 键: 启用EMA注意力
    attention_focus: float = 1.5   # 键: 注意力聚焦强度

    # ---- Tile / Seg ----
    tile_enabled: bool = True   # 键: 启用Tile检测
    tile_size: int = 640   # 键: Tile大小
    tile_overlap: float = 0.3   # 键: Tile重叠
    use_seg: bool = True   # 键: 使用Seg分割
    seg_feather: float = 3.0   # 键: Seg边缘羽化

    # ---- 合并 / 过滤 ----
    merge_boxes: bool = True   # 键: 合并重叠框
    merge_iou: float = 0.35   # 键: 合并IOU阈值
    merge_same_class: bool = True   # 键: 合并仅同类
    max_targets: int = 0   # 键: 最多目标数
    min_area_ratio: float = 0.0   # 键: 过滤最小面积比
    max_area_ratio: float = 1.0   # 键: 过滤最大面积比
    min_aspect: float = 0.0   # 键: 过滤最小长宽比
    max_aspect: float = 10.0   # 键: 过滤最大长宽比
    sort_mode: str = '置信度'   # 键: 排序方式
    per_class_limit: int = 0   # 键: 每类最多目标数
    lr_protect: bool = True   # 键: 左右分组保护
    bbox_expand2_px: int = 0   # 键: 合并后二次扩张像素
    bbox_expand2_ratio: float = 0.0   # 键: 合并后二次扩张比例

    # ---- 智能策略 ----
    smart_strategy: str = '关闭'   # 键: 智能策略

    # ---- 手脚增强 ----
    hand_foot_enhance: bool = True   # 键: 手脚增强模式
    hand_foot_denoise: float = 0.8   # 键: 手脚降噪下限
    hand_foot_extra_rounds: int = 1   # 键: 手脚额外轮次
    hand_foot_cfg_boost: float = 1.2   # 键: 手脚CFG加成
    hand_foot_cn_strength: float = 0.7   # 键: 手脚CN强度
    hand_foot_denoise_decay: float = 0.85   # 键: 手脚降噪衰减
    hand_foot_denoise_min: float = 0.4   # 键: 手脚衰减下限

    # ---- 骨架 CN ----
    hand_skeleton_cn_model: Any = None   # 键: 手骨架CN模型
    skeleton_cn_strength: float = 0.65   # 键: 骨架CN强度
    skeleton_cn_end: float = 0.7   # 键: 骨架CN结束

    # ---- 优先级 / 规则 / 自适应 ----
    priority_mode: str = '脸优先'   # 键: 修复优先级
    custom_priority: str = ''   # 键: 自定义优先级关键词
    adaptive_dn: bool = True   # 键: 按置信度自适应降噪
    conf_low_scale: float = 0.85   # 键: 自适应降噪_低置信缩放
    conf_high_scale: float = 1.0   # 键: 自适应降噪_高置信缩放
    class_rules_preset: str = '[不使用预设]'   # 键: 类别规则预设
    class_rules_text: str = ''   # 键: 类别规则

    # ---- 遮罩 ----
    expand: int = 32   # 键: 遮罩扩展
    feather: int = 24   # 键: 遮罩羽化

    # ---- 修复采样 ----
    region_steps: int = 0   # 键: 区域步数
    region_cfg: float = 0.0   # 键: 区域CFG
    region_denoise: float = 0.6   # 键: 区域降噪
    fix_rounds: int = 1   # 键: 修复轮次
    decay: bool = True   # 键: 逐次降噪递减
    no_detect_mode: str = '保持原样'   # 键: 未检测时处理

    # ---- 裁剪精修 ----
    enable_crop: bool = True   # 键: 开启裁剪精修
    crop_pad: int = 64   # 键: 裁剪扩展
    crop_max_edge: int = 512   # 键: 裁剪最大边
    crop_upscale: float = 1.6   # 键: 裁剪放大倍率
    crop_steps: int = 20   # 键: 裁剪步数
    crop_cfg: float = 0.0   # 键: 裁剪CFG
    crop_denoise: float = 0.5   # 键: 裁剪降噪
    crop_rounds: int = 2   # 键: 裁剪轮次
    crop_use_cn: bool = True   # 键: 裁剪使用ControlNet

    # ---- 二次检测 ----
    refine_det: bool = False   # 键: 二次检测精修
    refine_max: int = 6   # 键: 二次检测最多目标
    refine_pad: int = 32   # 键: 二次检测扩展

    # ---- 融合 ----
    blend_mode: str = '智能柔和融合'   # 键: 融合模式
    poisson_iters: int = 40   # 键: Poisson迭代次数
    mpb_color_preserve: float = 0.5   # 键: MPB颜色保持
    color_strength: float = 0.85   # 键: 色差修复强度
    clarity_strength: float = 0.3   # 键: 清晰度恢复强度
    use_latent_aware: bool = True   # 键: 使用Latent感知插值



# ==================== 4 个精简 Context（T6，对应拆分后的 4 个节点） ====================
#
# 拆成 4 个节点后，每个节点只该看见**自己那部分**字段。这既是隔离，也是一张网：
# 字段的归属由「哪个阶段真的读它」机检得来（AST 取 `ctx.<字段>` 的 Load 集），
# 不是手抄的清单 —— 手抄会漏，而漏掉的字段在运行期**只在该分支被走到时**才炸。
#
# 归属（并集恰好 = 50 个字段名，一个不漏、一个不多）：
#     DetectCtx ← detect.py ∪ node_detect.py:_finalize_interrupt ∪ report_bits.py:build_preview = 41
#     SamCtx    ← sam_refine.py                                                                 =  3
#     RepairCtx ← repair.py ∪ report_bits.py:build_masks                                        = 28
#     GlobalCtx ← global_pass.py ∪ report_bits.py:build_info                                    = 26
#
# 为什么这么归：`_finalize_interrupt`（中断回填，原 `report.py:finalize_interrupt`，
# T13 随单体一并搬进节点 A）与 `build_preview` 归节点 A，`build_masks` 归 C、`build_info` 归 D。
#
# ★ T13 已退役单体：`YoloRunContext` / `report.py` / `prepare_run` **都已删除**，
#   本文件里不再有「原件」可供对账。这四个类现在的对账基准是**冻结快照** ——
#   `tests/test_latent_split_guard.py::YOLO_RUN_CONTEXT_FIELD_SNAPSHOT`
#   （T13 之前从**活的** `YoloRunContext` 逐字段 dump 的 50 条 `{名: (注解, 默认值)}`）。
#   别再写「照抄 YoloRunContext」——那个类已经不存在了，改指快照。
#
# ★ `cfg` 四个类都要：每个阶段都读 `ctx.cfg.<旋钮>`，它在快照里也是一个槽。
#   它不是 YoloConfig 字段（那些走 `ctx.cfg.<名>`），所以这里保留为**必填**槽（无默认值）。
#
# ★ 字段名与默认值**逐字对齐**冻结快照（含中文名如 `模型`/`步数`）：
#   默认值不同会改运行期行为。声明顺序也按快照，便于逐行对账。
#
# ★ `GlobalCtx` 没有 `VAE` 槽：它唯一的读者是 `global_pass.py` 里 T13 删掉的那段死代码
#   （`if ctx.preview_image is None: ... ctx.VAE ...`）。全图降噪节点**不声明 VAE 输入**，
#   所以这个槽本就不该在 —— 是那条守卫（每个槽都要有来源读它）在 T13 后把它揪出来的。
#   `VAE` 只由检测节点读（`detect.py` 解码预览图），并集仍是 50 靠的就是 `DetectCtx` 那一份。


@dataclass(slots=True)
class DetectCtx:
    """节点 A 千绘·YOLO检测：detect.py ∪ report.py:finalize_interrupt ∪ report_bits.py:build_preview。"""

    cfg: YoloConfig
    模型: Any = None
    VAE: Any = None
    正面提示: Any = None
    负面提示: Any = None
    seed: int = 0
    步数: int = 20
    CFG: float = 8.0
    采样器: str = 'euler'
    调度器: str = 'normal'
    降噪: float = 1.0
    放大算法: str = 'latent_aware'
    放大倍数: float = 1.0
    yolo_models_to_use: List[str] = field(default_factory=list)
    yolo_confidences_to_use: List[float] = field(default_factory=list)
    has_cn: bool = False
    has_hand_skeleton_cn: bool = False
    class_rules: Any = None
    target_classes: Any = None
    samples: Any = None
    batch_size: int = 0
    out_device: Any = None
    out_latents: Any = None
    out_previews: List[torch.Tensor] = field(default_factory=list)
    out_masks: List[torch.Tensor] = field(default_factory=list)
    ksampler: Any = None
    pbar: Any = None
    processed_count: int = 0
    original_samples: Any = None
    i: int = 0
    base_pos: Any = None
    base_neg: Any = None
    pos_full: Any = None
    neg_full: Any = None
    base_latent: Any = None
    latent_for_detection: Any = None
    latent_h: int = 0
    latent_w: int = 0
    pixel_h: int = 0
    pixel_w: int = 0
    detections: List[Any] = field(default_factory=list)


@dataclass(slots=True)
class SamCtx:
    """节点 B 千绘·SAM精修：sam_refine.py。"""

    cfg: YoloConfig
    模型: Any = None
    sam_model_path: Optional[str] = None


@dataclass(slots=True)
class RepairCtx:
    """节点 C 千绘·区域修复：repair.py ∪ report_bits.py:build_masks。"""

    cfg: YoloConfig
    模型: Any = None
    seed: int = 0
    步数: int = 20
    CFG: float = 8.0
    采样器: str = 'euler'
    调度器: str = 'normal'
    降噪: float = 1.0
    has_cn: bool = False
    has_hand_skeleton_cn: bool = False
    class_rules: Any = None
    batch_size: int = 0
    out_masks: List[torch.Tensor] = field(default_factory=list)
    ksampler: Any = None
    i: int = 0
    base_pos: Any = None
    base_neg: Any = None
    pos_full: Any = None
    neg_full: Any = None
    latent_for_detection: Any = None
    latent_h: int = 0
    latent_w: int = 0
    pixel_h: int = 0
    pixel_w: int = 0
    detection_mask: Any = None
    detections: List[Any] = field(default_factory=list)
    _decoded_np: Any = None
    processed_latent: Any = None


@dataclass(slots=True)
class GlobalCtx:
    """节点 D 千绘·全局降噪：global_pass.py ∪ report_bits.py:build_info。"""

    cfg: YoloConfig
    模型: Any = None
    seed: int = 0
    步数: int = 20
    CFG: float = 8.0
    采样器: str = 'euler'
    调度器: str = 'normal'
    放大倍数: float = 1.0
    perf_start_time: Any = None
    batch_size: int = 0
    out_device: Any = None
    out_latents: Any = None
    out_previews: List[torch.Tensor] = field(default_factory=list)
    out_masks: List[torch.Tensor] = field(default_factory=list)
    all_dets: List[Any] = field(default_factory=list)
    ksampler: Any = None
    frame_history: Any = None
    interrupted: bool = False
    processed_count: int = 0
    i: int = 0
    pos_full: Any = None
    neg_full: Any = None
    detection_mask: Any = None
    detections: List[Any] = field(default_factory=list)
    preview_image: Any = None
    processed_latent: Any = None
