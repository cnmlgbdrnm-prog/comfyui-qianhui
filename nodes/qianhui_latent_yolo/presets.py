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


# ==================== T12：按节点分表 ====================
#
# 界面层拆成 4 个节点之后，「一键套预设」不能再是无差别的整张表：
# 每个节点只有一部分控件，把 19 个键全都写进它的 cfg，会让**本节点没有的旋钮**
# 也被改 —— 用户看不见、也不报错，只是 cfg 里悄悄多了个新值。
#
# ★ 分表的判据与控件归属**同一把尺子**（brief §「控件归属的唯一判据」）：
#   这个键属于哪个节点 = 它的字段被哪个节点的阶段模块读到。
#   所以 NODE_KEYS 的并集恰好 == FIELD_OF 的 19 个键，一个不多一个不少
#   （少了谁，那个旋钮在**任何**节点上都对预设免疫 —— 这正是要防的）。
#   守卫 `test_preset_keys_are_partitioned_across_nodes` 把这个并集钉死。

NODE_OF = {
    "QianhuiYoloDetect": "detect",
    "QianhuiSamRefine": "sam",
    "QianhuiRegionRepair": "repair",
    "QianhuiGlobalPass": "global",
}

NODE_KEYS = {
    # 检测节点：detect.py 读到的 5 个（多尺度 / Tile / 推理尺寸 / Seg / 二次精修）
    "detect": ("多尺度检测", "启用Tile检测", "YOLO推理尺寸", "使用Seg分割", "二次检测精修"),
    # SAM 节点：sam_refine.py 只读 enable_sam 这一个开关（阈值与模型路径不属预设表）
    "sam": ("启用SAM精细化",),
    # 修复节点：repair.py 读到的那些（含 使用Seg分割 —— ★ 见 brief §七，
    # 少了它，「检测=False、修复=True」时修复会**忽略** payload 里真实的 seg mask
    # （inpaint.py:453），而单体不会 —— 那是会改数值的）
    # ★ `结果缓存设备` **有意不在**这张表里（brief 的原表里有，是规格笔误）：
    #   修复节点根本不声明这个控件、repair.py 也不读 cache_device_mode ——
    #   放进来就是「前端 find 不到 widget 静默跳过 + 后端写了没人读」的双重空转。
    #   守卫 test_every_node_preset_key_is_a_widget_of_that_node 当场逮住了它。
    "repair": ("启用EMA注意力", "开启裁剪精修", "裁剪最大边", "裁剪步数", "裁剪降噪", "裁剪轮次",
               "区域步数", "区域降噪", "修复轮次", "融合模式",
               "使用Seg分割",
               "低显存模式", "每帧清理缓存"),
    # 全局降噪节点：只剩显存三件套。
    # ★ 融合模式 / 开启裁剪精修 / 使用Latent感知插值 **有意不在**这张表里：
    #   它们由节点 C 记进 `payload.stage_flags`、D 在 `_prepare` 里**覆盖**掉，
    #   预设写进去会被立刻盖掉 —— 那是「界面看着能改、实际改不动」。
    #   同理 启用YOLO（D 不暴露该控件）。
    "global": ("低显存模式", "每帧清理缓存", "结果缓存设备"),
}


def apply_preset(cfg, name, node_key=None) -> None:
    """按预设定名就地覆盖 cfg 字段。

    NO_PRESET 与 CUSTOM 均为「不覆盖」。

    `node_key=None`（默认）= 覆盖整张表 —— 保留给「没有节点身份」的调用方
    （既有测试、以及将来可能的后端批处理）。给了 `node_key` 就只覆盖
    `NODE_KEYS[node_key]` 里属于那个节点的键，理由见上面 NODE_KEYS 的注释。

    ★ `node_key` 是**关键字形参**：冻结用例
      `test_parse_config_mounts_preset_from_kwargs_not_cfg` 钉住调用点的
      `args[1]` 里必须出现 `kwargs`，所以预设名仍占第 2 位。

    表里出现 `FIELD_OF` 没有的键时**抛 KeyError**，不静默跳过——
    静默跳过会让「预设没生效」变成只有用户能发现的症状。
    """
    if not name or name in (NO_PRESET, CUSTOM):
        return
    table = PRESETS.get(name)
    if not table:
        return
    # ★ 不认识的 node_key 当场 KeyError（拼错节点名 = 一个键都写不进去，
    #   而症状是「预设在这台机器上没用」，必须炸在调用点而不是留给用户）。
    allowed = None if node_key is None else frozenset(NODE_KEYS[node_key])
    for key, value in table.items():
        if allowed is not None and key not in allowed:
            continue
        field_name = FIELD_OF.get(key)
        if field_name is None:
            raise KeyError(
                f"预设「{name}」的键 {key!r} 不在 FIELD_OF 里；"
                f"新增预设项时必须同时补 FIELD_OF（控件名 -> YoloConfig 字段名）"
            )
        setattr(cfg, field_name, value)


def as_payload() -> dict:
    """供 /qianhui_preset_data 路由返回。**只给前端控件名**，不带 FIELD_OF。

    T12 起按节点分表：

        {"nodes":   {注册名: 节点短名},          # 前端拿 nodeData.name 查这个
         "presets": {节点短名: {档位名: {控件名: 值}}},   # 每档只含**本节点**的键
         "labels":  [...7 项...]}

    ★ 键的形状（`presets[nodeKey][name]`）是前端 `applyToNode` 的契约，
      改它必须与 js/latent_yolo_presets.js 同一次提交（brief §七）。
    """
    return {
        "nodes": NODE_OF,
        "presets": {
            node_key: {
                name: {k: v for k, v in table.items() if k in frozenset(keys)}
                for name, table in PRESETS.items()
            }
            for node_key, keys in NODE_KEYS.items()
        },
        "labels": PRESET_LABELS,
    }

