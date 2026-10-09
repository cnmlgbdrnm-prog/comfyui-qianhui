# 千绘 · 水印插件整合 + 超文件拆分 + 4 节点控件精简 设计文档

日期: 2026-10-09
目标: (A) 把 `水印/` 插件的 3 个节点整合进 `comfyui-qianhui`，并把 5633 行的 `水印/nodes.py` 按功能拆成小组件；(B) 精简千绘现有 4 个节点的控件（去掉标签数字前缀、删除指定控件）
状态: 已确认设计，待实施

## 背景与问题

`E:\comfuyui 插件\水印\` 是一个独立的 ComfyUI 插件，含 3 个节点：

| 文件 | 行数 | 内容 |
|---|---:|---|
| `nodes.py` | 5633 | Latent YOLO 智能修复 + SAM 精修 + ControlNet（注册 1 个节点） |
| `watermark_nodes.py` | 497 | 可见水印 + DCT-QIM 隐形水印（注册 2 个节点） |
| `__init__.py` | 36 | 注册 3 个节点 + 一批中文别名 |

两个问题：

1. **游离在外**——它是独立插件，与千绘的 10 个节点分开维护，安装/升级要管两处。
2. **`nodes.py` 是个 5633 行的巨石**——其中 `LatentKSamplerUpscalerYOLO` 一个类占 4286 行，`process()` 单个方法占 1477 行。任何改动都要在近 6000 行里定位，无法单独读懂、单独测试、单独审查。

实测行数分布（本次拆分依据）：

| 功能块 | 行数 | 行范围 |
|---|---:|---|
| `process()` 主流程（单方法） | 1477 | 4131-5607 |
| YOLO 检测（加载/多模型/TTA） | 806 | 1541-2346 |
| SAM 精修 | 600 | 2347-2946 |
| `YOLOModelCache`（LRU + 显存） | 432 | 728-1159 |
| `LatentAlgorithms`（插值/羽化/融合） | 426 | 108-533 |
| 规则/预设/策略 | 420 | 3398-3817 |
| 采样/局部重绘/ControlNet | 313 | 3818-4130 |
| 几何（NMS/WIoU/过滤/合并） | 232 | 2947-3178 |
| 遮罩/EMA/预览 | 219 | 3179-3397 |
| `INPUT_TYPES` | 219 | 1322-1540 |
| `YOLOModelManager` + `SAMModelManager` | 182 | 538-723 |
| `ControlNetHelper` | 71 | 1164-1234 |
| `LatentKSamplerUpscaler`（父类） | 79 | 1239-1317 |

## 决策摘要

| # | 决策 | 选择 |
|---|---|---|
| 1 | 注册名 | 只保留千绘前缀，旧名与中文别名一律不注册 |
| 2 | 拆分粒度 | 细粒度：支撑类外移 + 巨型类切 Mixin + `process()` 切阶段函数 + 上下文对象 |
| 3 | 源目录 `水印/` | 保留不动 |
| 4 | 测试 | 搬入现有 17 个测试 + 补充算法层新测试 |
| 5 | 死控件 `配置预设` | 接上它（新增功能） |
| 6 | 预设生效方式 | 前端 JS 联动 + 后端兜底；预设表以后端为唯一来源，JS 走路由拉取 |
| 7 | 基准验证 | 用户先跑基准存图，拆分完成后同 seed 重跑比对 |
| 8 | 4 节点控件精简 | 删除标签数字前缀 + 删除 8 个指定控件（详见第三节） |
| 9 | `模型类型` 删除后 | 不写死模型，改为输出纯标签（不加权重），与场景节点既有做法对齐 |
| 10 | 旧工作流兼容 | 不考虑——用户后续统一更新工作流，故改键名/删控件导致的错位不作为约束 |

## 目标

1. `水印/` 的 3 个节点成为千绘的一部分，注册名与分类符合千绘约定，单个模块加载失败不影响其余节点。
2. `nodes.py` 拆成职责单一的模块，最大文件 ≤ 800 行，每个文件可单独读懂。
3. 水印/Latent 修复侧的拆分为**纯结构变更**——除「接上 `配置预设`」这一项外，节点行为逐位不变。第三节的 4 节点改动是有意的行为变更，不在此约束内。
4. 落地 `配置预设` 功能：下拉选中即生效，浏览器与 API 两种调用方式下都生效。
5. 水印算法层获得可自动运行的单元测试覆盖。
6. 精简 4 个既有节点（角色定制/肢体动作师/场景光效大师/战斗指挥官）的控件：去掉 67 个键名的数字前缀、删除 8 个控件。
7. 删除 `模型类型` 后不写死模型，改为输出纯标签——SDXL 与 anima 都能直接用，与场景节点既有做法对齐。
8. 这 4 个节点获得自动化单测覆盖（纯 STRING 输出，无需 GPU）。

## 非目标

- 不改千绘现有 10 个节点中除第三节列出的 4 个（角色定制/肢体动作师/场景光效大师/战斗指挥官）以外的节点。
- 不重命名 YOLO/水印节点的输入参数名（YOLO 节点维持中文参数名，水印节点维持英文参数名）。
- 不重命名 `YOLO模型2/3/4` 这类多模型槽位，不改变其语义。
- 不给 YOLO 修复侧补单元测试（需 mock `torch` / `comfy` / `ultralytics` / GPU，成本极高、价值低）。
- 不拆分 `LatentKSamplerUpscaler` 与 `LatentKSamplerUpscalerYOLO` 的继承关系。
- 不调整水印算法参数与像素级行为。
- 不为旧工作流做兼容垫片（见决策 10）。

## 一、整合方案

### 1.1 注册与命名

class 名 = 注册 key（沿用千绘约定，README「注册约定」一节）：

| 注册 key | class | 显示名 | 分类 |
|---|---|---|---|
| `QianhuiLatentYOLO` | `QianhuiLatentYOLO` | `千绘 · Latent智能修复 (YOLO+SAM+CN V6.11.3)` | `千绘/修复` |
| `QianhuiWatermarkAdd` | `QianhuiWatermarkAdd` | `千绘 · 水印添加` | `千绘/水印` |
| `QianhuiWatermarkExtract` | `QianhuiWatermarkExtract` | `千绘 · 水印提取` | `千绘/水印` |

- 旧注册名 `LatentKSamplerUpscalerYOLO` / `WatermarkAdd` / `WatermarkExtract` 与全部中文别名（`添加水印`、`加水印`、`水印提取` 等）**不注册**。既有 workflow 引用旧类名的需重新选择节点——与千绘 README 的既有约定一致。
- 新增两个分类分组：`千绘/修复`、`千绘/水印`。
- 版本号统一为 `V6.11.3`。源文件 docstring 已是 V6.11.3，但注册显示名与启动打印仍是 V6.11.2，本次对齐。

### 1.2 `__init__.py` 挂载

沿用现有 try/except 模式追加两个加载块（编号 11、12），单个失败只打印、不中断其余节点：

```python
# 11. Latent 智能修复模块
try:
    from .nodes.qianhui_latent_yolo import QianhuiLatentYOLO
    NODE_CLASS_MAPPINGS["QianhuiLatentYOLO"] = QianhuiLatentYOLO
    NODE_DISPLAY_NAME_MAPPINGS["QianhuiLatentYOLO"] = "千绘 · Latent智能修复 (YOLO+SAM+CN V6.11.3)"
except Exception as e:
    print(f"❌ 千绘 Latent修复模块加载失败: {e}")

# 12. 水印模块
try:
    from .nodes.qianhui_watermark import QianhuiWatermarkAdd, QianhuiWatermarkExtract
    ...
except Exception as e:
    print(f"❌ 千绘 水印模块加载失败: {e}")
```

### 1.3 依赖

千绘目前没有 `requirements.txt`，新增一个，内容：`reedsolo`（隐形水印的 Reed-Solomon 纠错依赖）。

`reedsolo` 缺失时的行为保持现状：导入期软失败（`RSCodec = None`），调用隐形水印时才抛带安装指引的 `RuntimeError`。可见水印不依赖 `reedsolo`，缺依赖时仍可用。

### 1.4 移植安全性核实结论

- **模型目录**：`YOLOModelManager.get_model_dir()` 走 `folder_paths.models_dir`（绝对路径），`scan_models()` 走 `os.path.dirname(folder_paths.models_dir)`。挪文件不影响模型发现。
- **模块级可变态**：`nodes.py` 除 `NODE_CLASS_MAPPINGS` / `NODE_DISPLAY_NAME_MAPPINGS` 外无模块级赋值。`YOLOModelCache` 在节点 `__init__` 里实例化，是实例级状态，无共享状态迁移问题。
- **`import nodes`**：源文件用 `import nodes` 取 ComfyUI 根目录的 `nodes.KSampler()`（2 处）。Python 3 绝对导入不会解析到本地 `comfyui-qianhui.nodes` 子包，无冲突；迁移后保持原写法，仅将导入点下移到使用它的模块内。
- **`super().__init__()`**：源文件 1518 行 `super().__init__() if hasattr(super(), '__init__') else None`：`hasattr` 恒为真，实际等价于 `super().__init__()`。迁入 `node.py` 时简化为 `super().__init__()`，语义不变。
- **父类死代码**：`LatentKSamplerUpscaler` 中 `INPUT_TYPES`（1242-1265）与 `process`（1284-1317）从未被调用，该基类本身也未注册。只有 `upscale_latent`（1266-1283）被 4 处调用（1301、4410、5522、5541）——是活代码。`base_node.py` **只移植 `upscale_latent`**，两个不可达方法不再移植。

## 二、拆分方案

### 2.1 水印模块 `nodes/qianhui_watermark/`

源文件 `watermark_nodes.py`（497 行）本就不算巨石，但依赖层次清晰，按层切开：

| 新文件 | 内容 | 源行范围 | 预估行数 |
|---|---|---:|---:|
| `__init__.py` | 导出 2 个节点类 + `NODE_*_MAPPINGS` | — | ~20 |
| `fonts.py` | `_FONT_CANDIDATES` + `_load_font` | 13-39 | ~30 |
| `dct.py` | `_make_dct_matrix`、`_DCT_D`、`_dct8`、`_idct8`、`_get_block_q` | 41-103 | ~70 |
| `payload.py` | `_SYNC`、`_RS`、`_require_reedsolo`、`_int_to_bits`、`_bits_to_int`、`_encode_payload`、`_decode_payload` | 66-153 | ~90 |
| `invisible.py` | `_QIM_REPEAT`、`_embed_invisible`、`_extract_invisible` | 156-291 | ~140 |
| `visible.py` | `_hex_to_rgb`、`_make_text_tile`、`_apply_visible_watermark` | 294-380 | ~90 |
| `nodes.py` | `QianhuiWatermarkAdd`、`QianhuiWatermarkExtract` | 383-497 | ~120 |

依赖方向单向：`nodes.py` → `invisible`/`visible` → `payload`/`dct`/`fonts`。无循环依赖。

### 2.2 Latent 修复模块 `nodes/qianhui_latent_yolo/`

顶层文件：

| 新文件 | 内容 | 源行范围 | 预估 |
|---|---|---:|---:|
| `__init__.py` | 导出 `QianhuiLatentYOLO` + `NODE_*_MAPPINGS` + 预设表查询接口 | — | ~25 |
| `algorithms.py` | `LatentAlgorithms`（10 个 `@staticmethod`） | 104-533 | ~430 |
| `model_managers.py` | `YOLOModelManager` + `SAMModelManager` | 534-723 | ~190 |
| `model_cache.py` | `YOLOModelCache` | 724-1159 | ~440 |
| `controlnet.py` | `ControlNetHelper` + `_extract_hand_skeleton`（手骨架提取，服务于骨架 CN） | 1160-1234 + 1789-1888 | ~180 |
| `base_node.py` | `LatentKSamplerUpscaler`（仅 `upscale_latent`） | 1266-1283 | ~30 |
| `node.py` | `QianhuiLatentYOLO` = 6 个 Mixin 聚合 + `INPUT_TYPES` + `__init__` + `RETURN_*` | 1322-1540, 1512-1515 | ~250 |
| `presets.py` | `配置预设` 预设表 + `apply_preset()`（新增功能） | — | ~80 |

`mixins/` 子包——巨型类的方法按职责归位（`node.py` 里 `class QianhuiLatentYOLO(DetectionMixin, SamMixin, GeometryMixin, MasksMixin, PolicyMixin, InpaintMixin, LatentKSamplerUpscaler)`）：

| Mixin 文件 | 方法（源行号） | 预估 |
|---|---|---:|
| `detection.py` | `_get_ultralytics_version`(1541)、`_check_yolo_capabilities`(1552)、`_load_yolo_model`(1591)、`_decode_latent_to_image`(1745)、`_upscale_image_for_yolo`(1750)、`_detect_with_multi_models`(1889)、`_detect_with_yolo`(1969)、`_detect_single_enhanced`(2181) | ~730 |
| `sam.py` | `_refine_with_sam`(2347)、`_unload_sam_model`(2839) | ~610 |
| `geometry.py` | `_nms_detections_enhanced`(2947)、`_iou_xyxy`(2978)、`_wiou_xyxy`(2993)、`_match_class_name`(3058)、`_filter_detections_by_target`(3069)、`_bbox_area`(3073)、`_bbox_aspect`(3078)、`_assign_lr_group`(3084)、`_filter_detections_geometry`(3090)、`_sort_detections`(3107)、`_limit_detections_per_class`(3114)、`_expand_bbox_xyxy`(3124)、`_merge_overlapping_detections`(3134) | ~250 |
| `masks.py` | `_create_detection_mask`(3179)、`_calculate_ema_attention`(3260)、`_draw_detection_preview`(3329) | ~220 |
| `policy.py` | `_resolve_model_path`(3371)、`_resolve_sam_model_path`(3381)、`_parse_target_classes`(3392)、`_auto_params_for_det`(3398)、`_get_preset_class_rules`(3451)、`_parse_class_rules`(3476)、`_match_rule`(3513)、`_get_rule_for_det`(3524)、`_priority_sort_detections`(3539)、`_apply_rule_limits`(3569)、`_build_det_policy`(3592) | ~350 |
| `inpaint.py` | `_decode_for_detection`(3702)、`_crop_latent`(3722)、`_build_crop_mask`(3740)、`_maybe_apply_controlnet`(3766)、`_gc_if_needed`(3780)、`_encode_hand_foot_prompts`(3818)、`_merge_conditioning`(3834)、`_get_enhanced_conditioning_simple`(3854)、`_prepare_inpaint_latent`(3872)、`_inpaint_sample`(3932)、`_get_crop_inpaint_mask`(4030) | ~440 |
| `context.py` | `YoloConfig` + `YoloRunContext`（见 2.4） | 新增 | ~200 |

`mixins/pipeline/` 子包——`process()`（1477 行）按原有语句顺序切成 5 个阶段方法：

| 阶段文件 | 内容 | 源行范围 | 预估 |
|---|---|---:|---:|
| `prepare.py` | `_parse_config`（99 个 `kwargs.get` → `YoloConfig`）、规则预设处理、模型列表解析、缓存设备决策、`sampler` 准备 | 4131-4386 | ~290 |
| `detect.py` | 逐帧主循环前半：预处理 → 全图预降噪 → VAE 解码 → YOLO 检测（多模型/单模型）→ 检测后处理（seg mask 转换/过滤/合并/扩张）→ 二次检测精修 → SAM 精修 → 中间变量清理 | 4387-4862 | ~500 |
| `repair.py` | `3) 区域精修`：逐目标裁剪 → 条件编码 → 局部重绘 → ControlNet → 融合贴回 | 4864-5406 | ~570 |
| `global_pass.py` | `4) 修复后全图降噪` + 视频模式时序平滑 + 帧历史维护 | 5407-5481 | ~100 |
| `report.py` | 预览图解码、遮罩汇总、`检测信息` 文本组装、性能统计、最终 return | 5483-5607 | ~150 |

拆分后最大文件 ~730 行（`detection.py`），远低于原来的 5633 行。

### 2.3 Mixin 之间的调用关系

Mixin 只定义方法，不定义 `__init__`、不持有类属性。跨 Mixin 调用一律通过 `self.`（Python MRO 在实例上解析，与原先同类的直接调用等价）。

方法体内对 `self.yolo_cache`、`self.yolo_model`、`self._printed_classes` 的访问保持不变——这些仍由 `node.py` 的 `__init__` 建立。

### 2.4 上下文对象 `mixins/context.py`

`process()` 拆分需要显式化跨阶段状态。设计两个 dataclass：

**`YoloConfig`**——99 个解析后参数的容器。字段名**逐字沿用原局部变量名**（含中文名如 `放大倍数`、`降噪`、`步数`），使迁移是机械替换 `局部名` → `cfg.局部名`，不做语义改写。按原注释分组（ControlNet / 低显存 / 工作模式 / YOLO 核心 / 检测优化 / 视频 / EMA / Tile / 合并过滤 / 策略 / 手脚增强 / 骨架 CN / 优先级规则 / 遮罩 / 修复采样 / 裁剪精修 / 二次检测 / 融合）。

**`YoloRunContext`**——跨阶段运行时状态：

| 字段 | 阶段 | 说明 |
|---|---|---|
| `cfg: YoloConfig` | 全部 | 配置 |
| `yolo_models_to_use` / `yolo_confidences_to_use` | prepare → detect | 解析后的模型路径与置信度 |
| `batch_size` / `pbar` | prepare → detect | 批次信息与进度条 |
| `frames`（`out_latents` / `original_samples`） | detect → repair → global_pass | 逐帧 Latent 累积 |
| `frame_previews` / `frame_masks` / `info_lines` | detect → report | 输出累积 |
| `frame_history` | global_pass | 视频模式帧历史 |
| `perf_start_time` / `perf_marks` | prepare → report | 性能统计 |
| `yolo_cache` | 全部 | 从 `self.yolo_cache` 取，保持单实例 |

**准确字段集由实施步骤用扫描确定**：对每个阶段切片，求「定义在前段、读取在后段」的变量集合，全部提升为 `ctx` 字段。不做凭记忆的猜测。

### 2.5 等价性约束（实施时强制）

- 拆 Mixin 阶段**不改方法体文本**，只改所在文件与 `self` 前缀（原本就是 `self.`，故 Mixin 阶段几乎是剪切粘贴）。
- 搬 `process()` 到阶段函数时**保持原语句顺序与缩进层级语义**，不合并、不重排、不"顺手优化"。
- 原注释（含 `# V6.11.3修复：` 等 bug 修复说明）一律随代码搬走，不删不改写。
- 除 2.6 的 `配置预设` 接线外，不新增/删除任何 `if` 分支。

### 2.6 落地 `配置预设`（新增功能）

源文件中 `配置预设` 在 `INPUT_TYPES`（行 1372）声明了 7 个选项，但 `process()` 从未读取——是个对用户无效的假控件。本次接线，并补齐前后端。

**预设表**（覆盖 19 个参数，`presets.py` 中唯一定义）：

| 参数 | 快速模式(低质量) | 平衡模式(推荐) | 质量模式(慢速) | 极致质量(超慢) | 省显存模式 |
|---|---|---|---|---|---|
| 多尺度检测 | False | True | True | True | False |
| 启用Tile检测 | False | True | True | True | True |
| YOLO推理尺寸 | 640 | 768 | 768 | 960 | 640 |
| 使用Seg分割 | True | True | True | True | True |
| 启用SAM精细化 | False | False | True | True | False |
| 启用EMA注意力 | False | False | True | True | False |
| 二次检测精修 | False | False | True | True | False |
| 开启裁剪精修 | False | True | True | True | True |
| 裁剪最大边 | 384 | 512 | 640 | 768 | 384 |
| 裁剪步数 | 12 | 20 | 26 | 32 | 16 |
| 裁剪降噪 | 0.50 | 0.60 | 0.65 | 0.70 | 0.55 |
| 裁剪轮次 | 1 | 2 | 2 | 3 | 1 |
| 区域步数 | 12 | 20 | 28 | 36 | 16 |
| 区域降噪 | 0.55 | 0.65 | 0.70 | 0.75 | 0.60 |
| 修复轮次 | 1 | 1 | 2 | 3 | 1 |
| 融合模式 | 简单混合 | 智能柔和融合 | 智能柔和融合 | Poisson融合 | 智能柔和融合 |
| 低显存模式 | False | False | False | False | True |
| 每帧清理缓存 | False | False | False | False | True |
| 结果缓存设备 | 自动 | 自动 | 自动 | 自动 | CPU |

**不纳入预设的参数**（表达用户意图，不该被预设覆盖）：`seed`、`步数`/`CFG`/`采样器`/`调度器`/`降噪`（顶层主采样）、`放大算法`/`放大倍数`、`工作模式`/`全图降噪强度`、`启用YOLO`/`YOLO模型`/`置信度`/`目标类别`、`类别规则预设`/`类别规则`、`手骨架CN模型`、提示词输入。

**`[不使用预设]` 与 `自定义`**：两者行为相同——不覆盖任何参数。保留两个选项是因为前者语义为"不用预设"，后者语义为"我已手动调过参数"，且既有 workflow 可能已保存这两个值之一。

**生效路径**：

1. **后端兜底**（权威）：`prepare.py` 解析完 `YoloConfig` 后调用 `presets.apply_preset(cfg, 配置预设)`，非 `[不使用预设]`/`自定义` 时按表覆盖字段。保证 `/prompt` API 无头调用、批量脚本、队列提交等 JS 不执行的场景下预设依然生效。
2. **前端联动**（体验）：新增 `js/latent_yolo_presets.js`，沿用 `role_random.js` 的写法（`beforeRegisterNodeDef` + widget `callback` 包装 + `getApp()` 的 ES module/老前端兼容兜底）。节点 `type === "QianhuiLatentYOLO"` 时给 `配置预设` 挂 callback，选中后把表中值写进对应 widget 并触发其 callback，使用户**看得见**控件值变化。

**预设表唯一来源**：只在后端 `presets.py` 定义一份。新增路由 `GET /qianhui_preset_data`（沿用 `/qianhui_role_data` 的模式，注册失败不影响节点）返回该表，JS 启动时拉取并缓存。避免前后端两份表漂移。

**冲突语义**：预设覆盖的是"控件当前值"，用户之后手动改动某个被覆盖的控件不受限制（预设是一次性写入，不是持续锁定）。若用户改完又重选同一个预设，值会被重新覆盖。

## 三、4 节点控件精简

范围：千绘现有 4 个节点——角色定制（`character_node.py`）、肢体动作师（`pose_node.py`）、场景光效大师（`scene_node.py`）、战斗指挥官（`combat_node.py`）。两类改动。

### 3.1 改动一：删除标签的数字前缀

数字是**参数键名的一部分**（如 `"00_工作安全策略"`），ComfyUI 直接把键名当控件标签显示。所以「删掉界面上的数字」= 改键名。

共 67 个键：角色定制 34、肢体动作师 14、场景 11、战斗 8。规则：只去掉开头的「数字 + 可选小写字母 + 下划线」，其余原样保留。

| 原键名 | 新键名 |
|---|---|
| `00_工作安全策略` | `工作安全策略` |
| `02a_季节` | `季节` |
| `02b_天气` | `天气` |
| `03_光效_整体氛围` | `光效_整体氛围` |
| `00_🎲随机动作模式` | `🎲随机动作模式` |

- 键名的**先后顺序不变**（顺序由 INPUT_TYPES 的书写顺序决定，与名字无关），因此控件在界面上的排布不动。
- 内层下划线保留原样（`光效_整体氛围`）。若希望显示为 `光效·整体氛围`，改用中点即可，但不作为本次默认。
- 改键名会改变 Widget 的名字，故函数体内全部 `kwargs[...]` / `kwargs.get(...)` 的字符串键必须同步替换——这是本改动的主要工作量，也最容易漏。

### 3.2 改动二：删除 8 个控件

| 节点 | 删除的控件 | 连带改动 |
|---|---|---|
| 角色定制 | `31_构图范围`、`32_Lora触发词`、`33_手动补充` | 输出里去掉 `shot_tag`、`lora_trigger`、`manual` 三段 |
| 肢体动作师 | `06_手动补充`、`09_模型类型` | 去掉 `manual` 段；权重语法整个移除，输出纯标签 |
| 场景光效大师 | `07_手动补充`、`08_环境总权重` | 去掉 `manual` 段；权重滑杆移除，恒为纯标签输出 |
| 战斗指挥官 | `07_模型类型`、`06_补充指令` | 去掉 `manual_add` 段；5 处 `is_anima` 分支收敛 |

### 3.3 「不写死」的落地方式：输出纯标签

删除 `模型类型` 后，权重语法不再由用户切换，但**也不写死成某一种模型**——而是把权重整个拿掉，输出纯标签。这样 SDXL 与 anima 都能直接使用，与场景节点自身的既有设计哲学一致（`scene_node.py` 开头注释：「权重滑杆设 1.0 即纯标签输出，符合官方'硬标签层+质量词+氛围句'规范，无需模型类型区分」）。

改动前后输出对比：

| 节点 | 改前 | 改后 |
|---|---|---|
| 肢体动作师 | SDXL：`(standing:1.2), looking at viewer` / anima：`(standing:2), looking at viewer` | `standing, looking at viewer` |
| 战斗指挥官 | anima：`(combat mode:2), dynamic pose, (katana:2)` / SDXL：`(combat mode:1.2), dynamic pose, (katana:1.2)` | `combat mode, dynamic pose, katana` |
| 场景光效大师 | 权重 1.1：`(sunset, orange sky, silhouette:1.1)` | `sunset, orange sky, silhouette` |

用户需要加权重的场合，在后续提示词节点里自行书写 `(tag:1.2)`。

### 3.4 纯标签化的代码位置

| 文件 | 位置 | 改动 |
|---|---|---|
| `pose_node.py` | 167 | 删除 `w_global = ":2)" if is_anima else ":1.2)"` |
| `pose_node.py` | 310 | `f"({parts['global']}{w_global}"` → `parts["global"]` |
| `combat_node.py` | 138 | 删除 `is_anima = ...` |
| `combat_node.py` | 172 | 相机分支统一取纯标签 `"motion blur, depth of field, dynamic angle"` |
| `combat_node.py` | 187-190 | 统一为 `"combat mode, dynamic pose, intense atmosphere"` |
| `combat_node.py` | 197 | 统一为 `slot_weapon`（去 `:2` / `:1.2` 包装） |
| `combat_node.py` | 204 | 统一为 `slot_vfx`（去 `:1.1` 包装） |
| `scene_node.py` | 372-384 | 删除 `weight` 取值与 `== 1.0` 分支，恒返回纯标签 `content` |

注意 `combat_node.py:172` 不只是权重差异——SDXL 分支是 `(dynamic angle:1.3)`、anima 分支是 `dynamic angle`，纯标签化后统一取后者。

### 3.5 文档同步

- 4 个节点的模块 docstring 中描述「模型类型: SDXL 用 `:1.2` 轻强调 / anima 按官方权重规范」的段落改写为「输出纯标签，不含权重」。
- README 节点表：删除肢体动作师与战斗指挥官「模型类型切换 SDXL / anima」的描述；改写场景光效大师「权重滑杆全局加权（1.0 纯标签）」的描述。

### 3.6 验证：这 4 个节点可以自动测

这 4 个节点输出纯 `STRING`，**不依赖 torch/GPU/ComfyUI 运行时**，因此可以用千绘现有的 `spec_from_file_location` 方式加载并断言：

- 输出串中不含 `:数字)` 形式的权重语法
- `INPUT_TYPES` 的键名中不含数字前缀（正则 `^\d+[a-z]?_` 无匹配）
- 被删的 8 个控件名不再出现在 `INPUT_TYPES` 中
- 函数体内无对已删键名的 `kwargs` 残留引用（否则运行期 KeyError）

这是本次唯一可以自动化验证的**行为**改动，价值高，必须做。

## 四、测试与验证

### 4.1 关键约束（必须先说清楚）

- **水印侧可自动验证**：算法层只依赖 `numpy` / `PIL` / `reedsolo`，无 ComfyUI 依赖，可跑单元测试。
- **YOLO 侧无法自动验证**：需要真实 ComfyUI + YOLO/SAM 模型 + GPU 才能执行。`process()` 拆成阶段函数**必然改动代码文本**（局部变量 → `ctx.` 引用），无法用"逐字节一致"证明等价。
- 因此 `process()` 拆分**必须靠真实跑图验证**，这是本方案唯一可靠的手段。

### 4.2 基准图对比（用户执行）

拆分**前**，用户用固定 `seed` 跑一组覆盖主要路径的工作流，保存输出到 `docs/superpowers/baseline/`：

| 场景 | 覆盖路径 |
|---|---|
| A | 纯 YOLO 修复（`[不使用预设]`、关 SAM、关 CN、仅修复模式） |
| B | 开 SAM 精细化 |
| C | 开 ControlNet + 手骨架 CN |
| D | 修复后全图降噪模式 |
| E | 裁剪精修关闭（走全图 mask 修复分支） |

每个场景保存：输出 Latent 解码图、`检测预览`、`检测遮罩`、`检测信息` 文本。场景必须都用 `配置预设 = [不使用预设]`——该默认值下行为不受本次新增功能影响，因此基准在拆分后依然有效。

拆分**后**，同 seed 重跑同一组，逐像素比对图、逐字符比对文本。同机、同库版本、固定 seed 下应**完全一致**；不一致即定位到对应阶段文件排查。

### 4.3 单元测试

- `tests/test_watermark.py`：搬入现有 17 个测试（`test_dct_roundtrip`、`test_payload_roundtrip_ascii`/`_cjk`、`test_decode_no_sync`、`test_invisible_roundtrip`、`test_invisible_no_watermark`、`test_invisible_psnr`、`test_invisible_capacity_error`、`test_visible_text_shape`、`test_visible_invalid_color_fallback`、`test_visible_image_wm_missing_fallback`、`test_node_watermarkadd_text`、`test_node_watermarkadd_invisible`、`test_node_watermarkextract_roundtrip`、`test_node_watermarkextract_no_wm`、`test_adaptive_q`、`test_rs_roundtrip`）。
  - 源文件用扁平 `from watermark_nodes import ...`；搬入后改为千绘现有的 `importlib.util.spec_from_file_location` 按路径直载风格（见 `tests/test_role_logic.py`），沿包目录注入 `sys.modules` 以支持模块间相对导入。
- `tests/test_latent_algorithms.py`（新增）：覆盖 `algorithms.py` 的纯函数——`latent_aware_resize`、`gaussian_feather`、`frequency_aware_color_match`、`simple_color_match`、`smart_soft_blend`、`poisson_blend_latent`、`mpb_blend_latent`、`simple_blend`、`_mild_sharpen`。只依赖 `torch` + `numpy`，无需 ComfyUI。
  - 断言性质（而非固定数值）：形状保持、mask 全 0 时输出等于 dst、mask 全 1 时输出趋近 dst、羽化权重随距离单调递减、数值范围不越界。
- 测试素材：`test_cn_wm.png` 搬入 `tests/assets/`。
- `python -m unittest discover tests` 应全绿。

### 4.4 导入健全性检查

拆分后逐模块导入（`python -c "import ..."` 配合 ComfyUI 桩模块），验证：
- `qianhui_watermark` 各子模块无循环导入
- `qianhui_latent_yolo` 各 Mixin 与 pipeline 阶段模块可独立导入
- 缺失 `reedsolo` 时 `qianhui_watermark` 仍可导入（软失败）
- 主 `__init__.py` 在任一模块加载失败时仍注册其余节点

## 五、错误处理

| 场景 | 行为 |
|---|---|
| `reedsolo` 未安装 | 导入期软失败；调用隐形水印时报 `RuntimeError`，提示 `pip install reedsolo`；可见水印不受影响 |
| 单个模块导入失败 | `__init__.py` 的 try/except 打印错误，其余节点照常注册 |
| `/qianhui_preset_data` 路由注册失败 | 静默跳过（沿用现有 `pass` 模式），JS 拉取失败时保持 ComfyUI 侧默认控件值，预设仍由后端兜底生效 |
| 前端 JS 弹错 | `getApp()` 兜底返回 null 时直接 return，节点可用，仅失去可见联动 |
| 预设表缺项 | `apply_preset` 只覆盖表中存在的键，缺失键保持 `INPUT_TYPES` 默认值 |
| YOLO/SAM 模型文件缺失 | 保持源码既有行为，不改 |

## 六、实施步骤概览

分 7 步。第 0 步（用户跑基准）是第 4、5 步的前置。第 3、4 步分开是为了给 `process()` 拆分一个**独立的验证检查点**——两者都改结构，但阶段化改控制流、风险高得多，混在一起出问题难定位。最终交付仍是完整细粒度拆分，不缩水。

| 步 | 内容 | 验证 |
|---|---|---|
| 0 | 用户跑基准工作流（场景 A-E）并存输出 | 基准文件就位 |
| 1 | 整合骨架：建两个包目录、`__init__.py` 挂载、`requirements.txt`、分类改名 | 模块可导入；旧节点名消失；其余节点正常 |
| 2 | 水印拆分（7 文件）+ 搬入 17 个测试 + 补算法测试 | `unittest discover` 全绿 |
| 3 | Latent 修复：支撑类外移 + 巨型类切 6 个 Mixin + `node.py` 聚合。`process()` **暂不动** | 基准图比对（首次） |
| 4 | `process()` 阶段化：`YoloConfig` + `YoloRunContext` + 5 个阶段函数 | 基准图比对（关键检查点） |
| 5 | `配置预设` 接线：`presets.py` + 后端兜底 + `js/latent_yolo_presets.js` + `/qianhui_preset_data` 路由 | 浏览器选中预设看控件联动；API 提交验证后端兜底 |
| 6 | 4 节点控件精简：67 个键改名 + 删 8 个控件 + 纯标签化 + docstring/README 同步 | 新增节点单测全绿（数字前缀、权重语法、残留键引用） |
| 7 | 文档：README 节点表/目录结构/依赖说明 | 人工核对 |

第 6 步与第 1-5 步操作不同文件、互不依赖，顺序可调；放在最后是因为它改的是既有节点，独立验证不干扰前面的基准图比对。

## 七、风险与对策

| 风险 | 等级 | 对策 |
|---|---|---|
| `process()` 阶段化改变行为且无法自动检测 | **高** | 只做机械搬运不改逻辑；保持语句顺序；步骤 3/4 分开以便二分定位；基准图逐像素比对 |
| 跨阶段变量集合凭记忆列全会漏 | 中 | 实施步骤 4 用扫描法求「前段定义、后段读取」集合，不靠推断；漏掉的变量会导致 `NameError`，在基准跑图时立即暴露 |
| Mixin 方法间通过 `self` 调用，MRO 顺序错导致方法遮蔽 | 中 | 6 个 Mixin 方法名无交集（已核对方法归属表）；`LatentKSamplerUpscaler` 置于末位保证 `upscale_latent` 可解析 |
| 千绘其余 10 节点被误伤 | 低 | 只新增包与 `__init__.py` 追加块，不改既有节点文件；步骤 1 即验证其余节点 |
| 旧 workflow 因注册名变更失效 | 中（已知并接受） | 决策 1 已确认；README 需注明旧类名清单，供用户重选节点 |
| 预设表前后端漂移 | 低 | 表只在后端定义，JS 走路由拉取 |
| `import nodes` 在包内解析异常 | 低 | Python 3 绝对导入不会解析到本地 `nodes` 子包；步骤 1 导入检查覆盖 |
| 67 个键改名后 `kwargs` 引用漏改 → 运行期 KeyError | 中 | 单测断言「无对已删/已改名键的残留引用」；改名与引用替换同一提交内完成，不留半成品 |
| 纯标签化改变 4 节点输出，用户既有提示词习惯受影响 | 中（已知并接受） | 决策 9 已确认；README 与 docstring 同步说明；需要加权重时由用户在后续提示词节点自行书写 |
| 第 6 步改既有节点，若与第 1-5 步同批验证会互相干扰 | 低 | 操作文件无交集；第 6 步排最后，且用独立单测验证 |

## 八、待办与后续（不在本次范围）

- `配置预设` 的 5 档具体数值是本次拟定值，首次实跑后可能需按实际效果微调——表集中在 `presets.py`，调整成本低。
- 水印算法层可考虑补 DCT 抗压缩/抗缩放鲁棒性测试。
- `qianhui_latent_yolo` 的 YOLO 修复侧若要补测试，需引入 `torch`/`comfy` 桩模块与合成检测结果，属独立课题。
