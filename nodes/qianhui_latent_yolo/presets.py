"""配置预设表（后端唯一来源，前端 JS 通过 /qianhui_preset_data 拉取）。

覆盖 19 个参数 × 5 档。不纳入预设的参数表达用户意图，见设计文档 2.6。

★ 键空间说明：`PRESETS` 的键是**控件名**（前端按控件名找 widget），
  `apply_preset` 通过 `FIELD_OF` 翻成 `YoloConfig` 的**字段名**再写入。
  新增预设项时**两处都要动**，`test_every_preset_key_maps_to_a_real_config_field` 会拦住漏改。
"""

NO_PRESET = "[不使用预设]"
CUSTOM = "自定义"

# 控件名 -> YoloConfig 字段名。
# 由 Task 4 步骤 3 的生成器输出逐个核对得来（`# 键: xxx` 那列），不是手写的。
FIELD_OF = {
    "多尺度检测": "multi_scale",
    "启用Tile检测": "tile_enabled",
    "YOLO推理尺寸": "yolo_imgsz",
    "使用Seg分割": "use_seg",
    "启用SAM精细化": "enable_sam",
    "启用EMA注意力": "enable_ema_attention",
    "二次检测精修": "refine_det",
    "开启裁剪精修": "enable_crop",
    "裁剪最大边": "crop_max_edge",
    "裁剪步数": "crop_steps",
    "裁剪降噪": "crop_denoise",
    "裁剪轮次": "crop_rounds",
    "区域步数": "region_steps",
    "区域降噪": "region_denoise",
    "修复轮次": "fix_rounds",
    "融合模式": "blend_mode",
    "低显存模式": "low_vram",
    "每帧清理缓存": "clear_each",
    "结果缓存设备": "cache_device_mode",
}

PRESETS = {
    "快速模式(低质量)": {
        "多尺度检测": False, "启用Tile检测": False, "YOLO推理尺寸": 640,
        "使用Seg分割": False,
        "启用SAM精细化": False, "启用EMA注意力": False, "二次检测精修": False,
        "开启裁剪精修": False, "裁剪最大边": 384, "裁剪步数": 12,
        "裁剪降噪": 0.50, "裁剪轮次": 1,
        "区域步数": 12, "区域降噪": 0.55, "修复轮次": 1,
        "融合模式": "简单混合",
        "低显存模式": False, "每帧清理缓存": False, "结果缓存设备": "自动",
    },
    "平衡模式(推荐)": {
        "多尺度检测": True, "启用Tile检测": True, "YOLO推理尺寸": 768,
        "使用Seg分割": True,
        "启用SAM精细化": False, "启用EMA注意力": False, "二次检测精修": False,
        "开启裁剪精修": True, "裁剪最大边": 512, "裁剪步数": 20,
        "裁剪降噪": 0.60, "裁剪轮次": 2,
        "区域步数": 20, "区域降噪": 0.65, "修复轮次": 1,
        "融合模式": "智能柔和融合",
        "低显存模式": False, "每帧清理缓存": False, "结果缓存设备": "自动",
    },
    "质量模式(慢速)": {
        "多尺度检测": True, "启用Tile检测": True, "YOLO推理尺寸": 768,
        "使用Seg分割": True,
        "启用SAM精细化": True, "启用EMA注意力": True, "二次检测精修": True,
        "开启裁剪精修": True, "裁剪最大边": 640, "裁剪步数": 26,
        "裁剪降噪": 0.65, "裁剪轮次": 2,
        "区域步数": 28, "区域降噪": 0.70, "修复轮次": 2,
        "融合模式": "智能柔和融合",
        "低显存模式": False, "每帧清理缓存": False, "结果缓存设备": "自动",
    },
    "极致质量(超慢)": {
        "多尺度检测": True, "启用Tile检测": True, "YOLO推理尺寸": 960,
        "使用Seg分割": True,
        "启用SAM精细化": True, "启用EMA注意力": True, "二次检测精修": True,
        "开启裁剪精修": True, "裁剪最大边": 768, "裁剪步数": 32,
        "裁剪降噪": 0.70, "裁剪轮次": 3,
        "区域步数": 36, "区域降噪": 0.75, "修复轮次": 3,
        "融合模式": "Poisson融合",
        "低显存模式": False, "每帧清理缓存": False, "结果缓存设备": "自动",
    },
    "省显存模式": {
        "多尺度检测": False, "启用Tile检测": True, "YOLO推理尺寸": 640,
        "使用Seg分割": False,
        "启用SAM精细化": False, "启用EMA注意力": False, "二次检测精修": False,
        "开启裁剪精修": True, "裁剪最大边": 384, "裁剪步数": 16,
        "裁剪降噪": 0.55, "裁剪轮次": 1,
        "区域步数": 16, "区域降噪": 0.60, "修复轮次": 1,
        "融合模式": "智能柔和融合",
        "低显存模式": True, "每帧清理缓存": True, "结果缓存设备": "CPU",
    },
}

PRESET_LABELS = [NO_PRESET] + list(PRESETS.keys()) + [CUSTOM]


def apply_preset(cfg, name: str) -> None:
    """按预设定名就地覆盖 cfg 字段。

    NO_PRESET 与 CUSTOM 均为「不覆盖」。
    表里出现 `FIELD_OF` 没有的键时**抛 KeyError**，不静默跳过——
    静默跳过会让「预设没生效」变成只有用户能发现的症状。
    """
    if not name or name in (NO_PRESET, CUSTOM):
        return
    table = PRESETS.get(name)
    if not table:
        return
    for key, value in table.items():
        field_name = FIELD_OF.get(key)
        if field_name is None:
            raise KeyError(
                f"预设「{name}」的键 {key!r} 不在 FIELD_OF 里；"
                f"新增预设项时必须同时补 FIELD_OF（控件名 -> YoloConfig 字段名）"
            )
        setattr(cfg, field_name, value)


def as_payload() -> dict:
    """供 /qianhui_preset_data 路由返回。**只给前端控件名**，不带 FIELD_OF。"""
    return {"presets": PRESETS, "labels": PRESET_LABELS}
