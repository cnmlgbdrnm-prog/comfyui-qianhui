# Latent智能修复：单节点拆成 4 个节点

## Context（为什么做这个）

`千绘 · Latent智能修复`（`QianhuiLatentYOLO`）现在是**一个节点、一列 113 个控件**，
在画布上是一条长到没法用的竖条。用户要的是把它**在界面上拆成几个独立节点**，
每个只暴露自己那部分参数 —— 不是把 `.py` 拆小。

⚠️ 上一轮计划已把它的 **5633 行源文件拆成了模块**（`mixins/` + 5 个阶段），
但**界面上仍是一个节点、仍是那一长条**。那一轮是代码层，这一轮是界面层。

**终态**（用户已逐条选定）：

```
[千绘 · YOLO检测] → [千绘 · SAM精修] → [千绘 · 区域修复] → [千绘 · 全局降噪]
```

巨型节点**从注册表删除**（老 workflow 失效，用户已接受）。**行为必须逐位保持**。

**诚实的收益边界**：拆完四个节点是 **≈61 / 8 / 61 / 15** 个控件 —— 不是预览里那个
20/6/3/40/8。YOLO 的 40 多个旋钮天然全归检测节点，裁剪/融合/手脚/策略全归修复。
它们是真实存在的独立参数。用户已明确接受（不加折叠分组）。

---

## 锁定的决策

| # | 决策 |
|---|---|
| 1 | **4 个节点**；巨型节点移除。老 workflow 失效（已接受） |
| 2 | **「遮罩处理」不单独成节点**，并回区域修复 |
| 3 | **跨节点控件两边各放一份**，用户手工保持一致；本节点那份说了算 |
| 4 | **配置预设改成每节点各自一个下拉**，`presets.py` 按归属拆表 |
| 5 | **不再加折叠分组**，接受检测/修复各 ~61 控件 |
| 6 | 标签保留中文；注册名保留 `Qianhui` 前缀（class 名 == 注册 key） |

---

## 已确证的事实（机读，非推测）

**为什么能沿阶段线切**：所有 `cfg.<字段>` 读取**只出现在 5 个 pipeline 阶段模块**里。
6 个 leaf mixin（`detection`/`sam`/`geometry`/`masks`/`policy`/`inpaint`）**一个 cfg 都不读**，
只接参数。→ 控件归属有客观依据。

**计数对账（全部吻合）**：113 控件 = 13 必填（`process()` 具名参数）+ 100 可选（`**kwargs`）；
`prepare.py:16-155` 共 100 处 `kwargs.get`，99 处赋给 99 个 `YoloConfig` 字段，
第 100 处是 `配置预设`（**无 cfg 字段**，只被 `apply_preset` 消费）；
`FIELD_OF` 恰好 19 条，`PRESETS` = 5 档 × 同 19 键；分组注释是 20 个（不是 19）。

**字段读取数（订正）**：`detect` 37 / `repair` 48 / `global_pass` 7 / `report` 6 是
**各模块的 cfg 读取条数**，不是「独占字段数」。减去 `prepare` 的解析读、并去掉被 ≥2 个
非 prepare 阶段读的 13 个共享字段后，真正归属是 **15 / 29 / 37 / 4 / 1 + 13 共享**。
按读者表直接生成节点控件表，**不要**拿 37/48/7/6 当控件清单。

**「遮罩处理」单独拆是人为的**：遮罩算子本身是纯的，但每个目标用多少膨胀/羽化是按
`class`/`confidence`/`bbox` 现算的（`mixins/policy.py:230-337`），合成一张 `MASK` 后全丢；
**羽化不满足交换律**（逐目标羽化再合并 ≠ 合并后统一羽化）；真正用于重绘的 inpaint 遮罩
建在逐目标循环内部（`repair.py:143-330`）。

**阶段式批处理与逐帧交错行为等价**（已证，不要再推翻）：唯一跨帧数值状态
`ctx.frame_history` 只在 `run_global_pass` 内读写；所有种子都是 `ctx.seed + ctx.i` 的纯函数。
只有出错路径不同（见残差 R1）。

---

## 四个节点的名册

命名：**类名 == 注册名**，中文只进 `NODE_DISPLAY_NAME_MAPPINGS`。四个都用 `CATEGORY = "千绘/修复"`。

| 节点 | 类名 / 注册名 | 显示名 | 控件数 | RETURN_TYPES → NAMES |
|---|---|---|---|---|
| 检测 | `QianhuiYoloDetect` | 千绘 · YOLO检测 | ~61 | `(IMAGE, QIANHUI_DETECT, LATENT)` → `(检测预览, 检测数据, 检测用Latent)` |
| SAM | `QianhuiSamRefine` | 千绘 · SAM精修 | ~8 | `(QIANHUI_DETECT,)` → `(检测数据,)` |
| 修复 | `QianhuiRegionRepair` | 千绘 · 区域修复 | ~61 | `(LATENT, MASK)` → `(修复后Latent, 检测遮罩)` |
| 全局 | `QianhuiGlobalPass` | 千绘 · 全局降噪 | ~15 | `(LATENT, STRING)` → `(Latent, 检测信息)` |

**控件归属要点**
- 检测：13 必填 + CN 5 + YOLO 模型解析 11 + detect 专属 27 + 跨切面 4（`使用Seg分割`/
  `使用Latent感知插值`/`类别规则预设`/`类别规则`）+ `配置预设`
- SAM：`检测数据` + `模型`（SAM 后重载用）+ 4 个 SAM 控件 + 3 个共享
- 修复：11 必填（`步数`/`CFG`/`降噪` 不是装饰 —— `repair.py:449-450` 用它们做回退，
  `:547` 用于「未检测时处理=全图重绘」）+ repair 专属 37 + CN 5 + `使用Seg分割` +
  `使用Latent感知插值` + `类别规则预设`/`类别规则` + 低显存/每帧清理/结果缓存设备 + `配置预设`
- 全局：`Latent`/`检测数据`/`模型`/正负提示/seed/步数/CFG/采样器/调度器 + CN 5 +
  `工作模式`/`全图降噪强度`/`视频模式`/`时序一致性强度`/`参考帧窗口` +
  低显存/每帧清理/结果缓存设备 + `启用性能分析` + `配置预设`

**四路输出归属**（各有代码级理由）
- `检测预览 IMAGE` → **检测**（`detect.py:474` 画的；`global_pass.py:85` 的兜底分支永不触发）
- `检测遮罩 MASK` → **修复**（并集由 `repair.py:33` 建，expand/feather 来自 policy —— 
  正是「遮罩留在修复内」的前提）
- `Latent` → **全局**（链条终点）
- `检测信息 STRING` → **全局**（只有它在所有帧后仍持有汇总）；它需要的 4 个外来值
  （`blend_mode`/`use_latent_aware`/`enable_yolo`/`enable_crop`）**从 payload 的
  `stage_flags` 快照取**，而不是复制控件 —— 这样字符串里的值恒等于各阶段实际用的值

---

## 跨节点载荷（本计划的核心）

**类型名 `"QIANHUI_DETECT"`**，Python 侧 `DetectionPayload`，定义在**新文件**
`nodes/qianhui_latent_yolo/payload.py`（~120 行）。

★ `nodes/` 是 PEP 420 命名空间包，**只有相对导入能工作**（已实测记录在台账里）。
4 个节点模块写 `from .payload import ...`，`mixins/pipeline/` 里的写 `from ..payload import ...`。
**不要**写 `from nodes.qianhui_latent_yolo.payload import ...`。

```python
TYPE_NAME = "QIANHUI_DETECT"

@dataclass(slots=True)
class DetectionFrame:
    i: int                      # 帧号 —— 所有 seed 偏移的唯一来源
    processed: bool             # False = 中断补帧，下游跳过全部分支
    detections: list            # 逐字沿用单体的记录，不新增不带键
    pixel_h: int; pixel_w: int
    latent_h: int; latent_w: int
    decoded: Any = None         # decoded_original：CPU tensor [H,W,3] float32∈[0,1]
    det_image_size: Any = None

@dataclass(slots=True)
class DetectionPayload:
    frames: list                # 索引对齐：frames[i].i == i
    batch_size: int
    stage_flags: dict           # 各阶段记录自己真正用的值（供「检测信息」复述）
    perf: dict                  # {"detect": s, "sam": s, "repair": s, "global": s}
    interrupted: bool = False
```

**检测记录的键**（沿用单体，不新增）：`class_name` / `class_id` / `confidence` /
`bbox`(像素空间) / `seg_mask`(np float32, 像素空间) / `seg_mask_shape` / `sam_score` / `source_model`。
**不进 payload** 的：`processed_latent`（走 LATENT 口）、`detection_mask`（修复自产）、
`preview_image`（检测自产）。

**批次承载（唯一必须动结构的地方）**：单体里 `ctx.detections` 是单帧槽且每帧被 `del`。
stage-wise 下每个下游节点在**循环头部逐帧重绑**本节点 ctx 的逐帧槽
（`ctx.i = fr.i`、`ctx.detections = fr.detections`、`ctx._decoded_np = fr.decoded.numpy()` …），
`del` 的语义被自然替换成「下一轮重新赋值」，`slots=True` 不再有 `AttributeError` 风险。

**内存**：`decoded` 是每帧一张 CPU `[H,W,3] float32`（1024² ≈ 12.6 MB/帧）。
**无条件携带**它，因为下游重算需要 VAE decode，而 VAE decode 的位级可复现性受
tiling/设备影响 —— 等于拿「精度可能不一致」换内存，不可接受。

★ **不要「顺手修」SAM 的坐标口径**：`detect.py:398` 送进 SAM 的是 **det 空间**的图
（不是 pixel 空间）。节点 B 必须照抄：`_upscale_image_for_yolo(frame.decoded, min_size=1024)`
再喂 `_refine_with_sam`。

---

## 静默失效的处置（**5 处，不是 6 处**）

| # | 位置 | 处置 |
|---|---|---|
| a | `node.yolo_cache`（`repair.py:47-49`/`:75-76` 伸手卸载、`detect.py:378-380` 在 SAM 前清空） | `model_cache.py` 加 **`get_shared_cache()` 模块级单例**；4 个节点 `__init__` 都用它。否则修复节点倒的是自己那个**空缓存**，检测节点的模型再没人卸，**不报错**。已知日志残差：模型类别表可能多打一次（数值无影响） |
| b | 被 `del` 的槽位 × `slots=True`（`global_pass.py:96`） | 每节点自己的精简 Context（`DetectCtx`/`SamCtx`/`RepairCtx`/`GlobalCtx`，都带 `slots=True`）+ 循环头部逐帧重绑。**不是必崩，而是会读到上一帧残值** |
| c | `ctx.frame_history` | 全局降噪节点**本地 list**，按 `payload.frames` 顺序维护；`global_pass.py:51` 的守卫逐字保留。★ 节点 D 必须按序遍历（用 `payload.iter_frames_in_order()` 唯一入口，并加测试禁止直接 `for fr in payload.frames`） |
| d | `ctx.out_latents` 形状由第 0 帧定死（`detect.py:40-41`） | 分配点**搬到 repair**；节点 D 收到的就是一个 batch 张量。`report.py:22-33` 的中断回填**搬到节点 A 的异常分支** |
| e | `_decoded_np`（`repair.py:241-245` 做 MediaPipe 骨架） | 放 payload 的 `frame.decoded`，节点 C 按需 `.numpy()`（CPU 张量零拷贝，值恒等） |

### ★ 明确**不需要**处置的一项：`_cuda_corrupted`

前一份探查把它列为「跨节点静默失效」是**误判**，已实测推翻：
- 全部 4 处出现：`inpaint.py:353`(读) / `:368`,`:413`(写) / `prepare.py:275`(复位)
- 读它的只有 `_inpaint_sample`，而 `_inpaint_sample` 的调用点**只有** `repair.py:326`、`:474`
- 全局降噪走的是 `ctx.ksampler.sample(...)`（`global_pass.py:24`），**从不查这个标志**

→ 「修复检测到损坏、全局降噪失去抑制」**今天就不成立**。把它留在区域修复节点、
在该节点 `process` 开头复位，行为逐字相同。**不要**为它新增跨节点抑制（那是加行为）。

### 顺带删掉的死代码

`global_pass.py:85-86` 的 `if ctx.preview_image is None: _decode_latent_to_image(...)`
在单体里**永不执行**（检测阶段每帧都赋值）。节点 D 不带 VAE socket、不留这段。
这是删死代码不是改行为，但要登记进对拍脚本的「源侧丢失」清单。

---

## 预设重设计

`presets.py` 里 `FIELD_OF` 与 `PRESETS` **一字不动**（仍是唯一真值源），新增按节点的分表：

```python
NODE_OF = {"QianhuiYoloDetect":"detect", "QianhuiSamRefine":"sam",
           "QianhuiRegionRepair":"repair", "QianhuiGlobalPass":"global"}
NODE_KEYS = {
  "detect": ("多尺度检测","启用Tile检测","YOLO推理尺寸","使用Seg分割","二次检测精修"),
  "sam":    ("启用SAM精细化",),
  "repair": ("启用EMA注意力","开启裁剪精修","裁剪最大边","裁剪步数","裁剪降噪","裁剪轮次",
             "区域步数","区域降噪","修复轮次","融合模式",
             "使用Seg分割",                    # ← 必须补：use_seg 在 repair.py:272/291 被读
             "低显存模式","每帧清理缓存","结果缓存设备"),
  "global": ("低显存模式","每帧清理缓存","结果缓存设备"),
}
def apply_preset(cfg, node_key, name): ...   # 唯一改动：多一个形参
```

★ `使用Seg分割` 进修复的表是**必须**的：否则「检测=False、修复=True」时修复会**忽略**
payload 里真实的 seg mask（`inpaint.py:453`），而单体不会 —— 这是会改数值的。

- 路由 `/qianhui_preset_data` **路径与函数名不变**，只是 payload 变成按节点分表
- ★ **表结构与 JS 必须同一次提交**（唯一一处必须原子）
- `js/latent_yolo_presets.js`：`nodeData.name` 判断改成查 `NODE_OF`；`applyToNode` 取
  `presetData.presets[nodeKey][name]`；`coveredKeys(nodeKey)` 只看本节点的键
- ★ `applying` 页面级抑制标志与 `_qhPendingPreset` 补应用时机**逐字不动**（踩过坑的）
- `启用性能分析` 只放全局降噪节点；4 个节点**无条件**把自身耗时写进 `payload.perf[stage]`

---

## 文件布局

**新增**：`payload.py`(~120) · `node_detect.py`(~330) · `node_sam.py`(~150) ·
`node_repair.py`(~330) · `node_global.py`(~220) ·
`mixins/pipeline/sam_refine.py`(~120，从 `detect.py:366-469` 原样搬出) ·
`mixins/pipeline/report_bits.py`(~180，预览/遮罩/信息段抽成函数) ·
`tests/test_latent_split_guard.py` · `tests/test_latent_split_e2e.py`

**修改**：根 `__init__.py`（一个 try 块注册 4 个，照 `qianhui_watermark` 写法，`:120-127`）·
包 `__init__.py`（导出 4 类）· `mixins/context.py`（`YoloConfig` **99 字段不动**；
新增 4 个精简 Context）· `mixins/pipeline/prepare.py`（`parse_config` 保留全部 99 个 `.get`；
删 `prepare_run`；`apply_preset` 调用点带 `node_key`）· `mixins/pipeline/detect.py`（删 SAM 段）·
`model_cache.py`（加 `get_shared_cache()`）· `js/latent_yolo_presets.js` · `README.md` · 相关测试

**退役**：删 `node.py`（单体）· 删 `mixins/pipeline/report.py`（内容进 `report_bits.py`）·
删 `YoloRunContext`（避免死字段）

★ **6 个 leaf mixin、`algorithms.py`、`model_cache.py`、`model_managers.py`、`controlnet.py`、
`base_node.py` 全部保留不动** —— 这是行为保持的最大保障：数值逻辑零改动。
搬运时注意相对导入深度：`pipeline/` 里引用同包用 `.`、引用包根用 `...`（见 `detect.py:7`）。

---

## 任务序列（每个 commit 结束时全绿）

统一命令：`python -X utf8 -m unittest discover -s tests -p "test_*.py"`（插件根目录）。
每个 commit 结尾加 `Co-Authored-By: Claude Code <noreply@anthropic.com>`。

| # | 任务 | 为什么这个顺序 |
|---|---|---|
| T1 | 建 `payload.py` + 守卫测试里的 payload 用例（索引对齐、顺序保持、`TYPE_NAME`） | 纯新增，零风险 |
| T2 | 静态守卫：ast 提取各阶段 `cfg.<字段>` 读取 → 双向闭合断言 | **把网建在拆之前** |
| T3 | `report_bits.py`：抽出预览/遮罩/信息段，`report.py` 调它们 | 纯结构搬移；同步对拍脚本 target |
| T4 | `sam_refine.py`：`detect.py:366-469` 原样搬出，`run_detection` 调它 | 纯结构搬移，为节点 B 铺路 |
| T5 | `model_cache.get_shared_cache()`；单体改用它 | 纯结构，单体行为不变 |
| T6 | `context.py` 加 4 个精简 Context（`YoloRunContext` 先留着） | 新增：每个 Context 的 slots == 对应阶段的 `ctx.` 读取集 |
| **T7** | **`tests/test_latent_split_e2e.py`：跑单体，把 4 路输出存成 golden** | ★ **必须在删单体之前**，否则永久失去对拍基准 |
| T8 | `node_detect.py` + 导出 + 注册（单体仍在） | 新旧并存，都不破 |
| T9 | `node_sam.py` + 注册 | 同上 |
| T10 | `node_repair.py` + 注册 | 同上 |
| T11 | `node_global.py` + 注册；**打开**拆分侧对拍，与 golden 逐张量 `torch.equal` | 到此 4 节点链可端到端对拍 |
| T12 | 预设按节点拆（`NODE_KEYS`/`apply_preset` 新签名/`as_payload` 新形状/JS） | ★ **必须原子** |
| T13 | 摘掉单体：删 `node.py`/`report.py`/`YoloRunContext`/`prepare_run`；根注册去掉旧名 | 单体最后删 |
| T14 | README + 重新生成对拍报告并提交 | 证据落盘 |

---

## 验证

**① 固定种子端到端对拍**（`tests/test_latent_split_e2e.py`，T7 建、T11 打开）
- 桩：`nodes.KSampler`→确定性返回；`_detect_with_yolo`→固定 3 条记录（两条带合成 seg_mask、
  一条 None）；`_refine_with_sam`→填确定 `seg_mask`+`sam_score`；`_decode_latent_to_image`→
  确定性函数（跳过真 VAE）；`_extract_hand_skeleton`→None；`YOLOModelCache.load_model`→哨兵
- 输入覆盖：`seed=1234`、`batch_size=3`、`放大倍数` 1.0/1.5 各一遍、`work_mode` 两种、
  ★`video_mode=True`+`frame_window=2`（专压 `frame_history`）、`enable_crop` 两种、
  `blend_mode` 取两种、`enable_sam` 两态、`no_detect_mode` 两态
- 比：`latents`/`preview`/`mask` 用 `torch.equal` 逐位；detections 用规范化 JSON；
  `检测信息` **归一化后**逐字（见残差 R2）

**② 既有对拍脚本**（`docs/superpowers/verification/verify_split_behaviour_preservation.py`）
仍然适用且必须继续跑（leaf mixin 应逐字节不变 —— 那正是数值保持的主体）。需改 4 处：
`PROCESS_SPLIT_TARGETS` 换成 4 个 `process` + `sam_refine` + `report_bits`；
`EXPECTED_DIFF_FUNCTIONS` 给 4 个新 `process` 登记理由；
`EXPECTED_STMT_LOSSES` 新增登记（每帧 `del` 段、全局降噪里那段永不执行的 VAE decode）；
§4 的对照数字重算而非手改。跑法不变，**退出码 0 才算过**。

**③ 静态守卫**（`tests/test_latent_split_guard.py`）：每节点字段 ⊆ 其控件；
每节点 Context slots ⊇ 其阶段 `ctx.` 读取集；payload 顺序契约（禁止直接遍历 `frames`）；
共享缓存单例身份；**同名共享控件在 4 个节点 INPUT_TYPES 里的定义逐个相等**
（把「手工保持一致」从「值」缩到只在「运行时选择」上）。

---

## 必须提前知道的残差（无法逐字保持的部分）

| # | 残差 | 说明 |
|---|---|---|
| **R1** | **中断语义** | 单体一个 `try` 跑完 4 阶段，异常被接住后**未处理帧回填原图并照常返回 4 路输出**。4 节点链上节点 A 抛异常会让 ComfyUI 中止整个 prompt，下游根本不执行 → 部分结果丢失。**部分可消除**：payload 带 `processed` 标志 + 节点 A 在异常里补全帧、节点 C 对 `processed=False` 的帧跳过全部分支直接透传。**不可逆的部分**：下游节点是否被调度不由我们决定 |
| **R2** | **`检测信息` 的性能行** | 单体量的是整条流水线；节点 D 只能求和 4 段，少了节点间调度/搬运开销。数值部分（帧数/shape/计数）可逐字；**性能行不可**。测试里要**显式断言**这些行只差 `总耗时`/`显存` 两个数值 —— 让 R2 被「证明存在且被限定」，而不是被「跳过」 |
| **R3** | **缓存设备粒度** | 单体一次执行只跑一遍 `out_device` 决策；4 节点各自决策。数值不变，但若修复与全局的 `结果缓存设备` 不一致，最终 LATENT 的 device 会与单体不同。这是决策 3（各放一份）的必然代价，登记为已知残差 |

**另有两条已知的日志级差异**：模型类别表可能多打一次（共享缓存但 `_printed_classes` 仍是每节点）；
`检测信息` 之外的分阶段打印顺序会变（每个节点的 print 各自成段）。均无数值影响。
