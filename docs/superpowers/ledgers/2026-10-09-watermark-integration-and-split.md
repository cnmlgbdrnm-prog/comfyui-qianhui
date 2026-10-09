# SDD ledger — plan: docs/superpowers/plans/2026-10-09-watermark-integration-and-split.md

Task 0: 待用户采集基准图（不阻塞 T1/T2/T5，阻塞 T3/T4 的验证）
Task 1: complete (commits 59ccf99..fefa345, review clean — Spec ✅ / Task quality **Approved** / 0 Critical / 0 Important)
Task 1: minor (deferred): 子包 mapping dict 与根加载器重复（display name 两处来源）。裁决：千绘 9 个既有节点模块同样自带未被聚合的 mapping dict（negative_node.py 亦然），属仓库既有约定，根加载器为唯一注册权威。已写入设计文档 1.2 定死约定，防止 T2/T3 审查重复提。最终审查复核。
Task 1: minor (deferred): 根 __init__.py 里 # 11./# 12. 跳过 # 10.（夹在 # 9. 与 # 8b. 之间）。plan-mandated，编号由简报逐字指定，纯美观。最终审查复核。
Task 1: controller note: 预检断言注释 10+2 → 9+1+2 已修正；补入「版本号处理规则」（源末尾注册块+启动横幅不移植；process 内 4296 行 print 的 V6.11.2 → V6.11.3；代码注释里的历史版本号不许动）
Task 1: controller note: 审查提的 2 个 ⚠️ 已裁定 —— (a) commit trailer 已核实存在；(b) 千绘/修复 与 千绘/水印 的 CATEGORY 属 Task 2/3 职责，已在这两个任务的派发指令里点名
Task 2: complete (commit a0d5ecd) — impl-a-t2 死于 402，控制者接管收尾
Task 2: controller note: impl-a-t2 在 402 前已产出 7 个模块且端到端可用（隐形水印 max_diff=0.2076 在 0.40 界内、回读 'copyright'、可见模式形状正确、原图未被改、注册表正确）。缺的 4 个节点层用例 + tests/assets/test_cn_wm.png 由控制者补齐，17/17 通过。全量 41/41 通过（50f8a56 + a0d5ecd 之后实测）。
Task 2: 审查 clean-with-fix (rev-a-t2) — Spec ✅ 但「逐字搬运测试」一条 ❌ / Task quality Approved
Task 2: 审查器结论：算法代码与节点类逐字一致（去注释/空行的机器比对，全部 IDENTICAL）；
  源插件目录未被改动；类名/CATEGORY/注册/requirements 全对；7 文件拆分依赖图无环且可解析；
  测试素材 md5 两侧一致。问题**只在测试**。
Task 2: review-important —— 17 个用例全在，但断言边界被改松。根因是**计划自身**：
  步骤 2 贴的「示例测试代码」是源测试的缩水重写（test_adaptive_q 只剩 5/11 个点，
  丢掉的正是 100/800/150/1000 四个阈值两侧的边界；test_rs_roundtrip 换了输入；
  test_invisible_psnr 换了 PSNR 算式）。66ed939 补了「照搬」规则却没改这份示例，
  简报自相矛盾 → 实现者照抄。
Task 2: 控制者修复（commit 3d9611f）：tests/test_watermark.py 全量重写为机械翻译；
  计划步骤 2 整段重写为「不手写、只机械翻译」+ 映射表 + 断言计数自检脚本。
  实测 用例 17/17、断言 源=33 移植=34（+1 为 test_invisible_capacity_error 的预期强化）、
  逐用例无变弱、无缺失/多出。17/17 通过。
Task 2: 内联修复而非重派修复轮的理由：原实现者 impl-a-t2 已死于 402 无法续跑；
  且这是「把测试搬对」而非「改设计」，机械翻译可机器自检。
Task 2: 审查器后续订正：受影响用例数是 **9 个**（先前自称 6 个，系误计）。
  finding 1 adaptive_q（5/11 点，四个阈值边界全丢）、finding 2 rs_roundtrip
  （1/4 断言，丢了「3 字节损坏必须解不出」这个唯一的负向断言）、
  finding 3 visible_text_shape（"有改动"断言被换成 dtype 检查）——三条均已确认并已修。
Task 2: scoped re-review 回来了（rev-a-t2）。Q1 三条 finding 确认已解（逐点核到值：
  adaptive_q 的 100/800/150/1000 四点、rs_roundtrip 的 64/72/80 偏移与 "© RS 测试"、
  visible_text_shape 的「有改动」断言）；Q2 它按 AST 逐语句对拍全部 17 个用例，
  16 个在允许的三种机械变换+改名后字节相同，剩 1 个正是登记在案的 assertRaises；
  Q3 无空断言、无过约束、辅助函数不再藏输入。
Task 2: ★ scoped re-review 的 finding 1（本轮最锋利的一条，已修 commit a65a752）：
  我新写的自检门按「断言条数 移植≥源」判定，**看不到**改过的常数/种子/形状/运算符/
  消息，也看不到「换成另一个同样条数的弱断言」。它给了杀手级反例：
  **我刚替换掉的那份缩水测试正好能过这个门**（visible_text_shape 改前 2 条=形状+dtype，
  改后 2 条=形状+生效，条数不变）。已换成逐值门：两侧规范化后逐用例比对 AST。
  两向实测：对现文件全绿；对 a0d5ecd 那份缩水测试**点出 16 个里的 15 个**
  （含审查者手工找到的那 3 条，比手工多 6 条）。门已能拒绝它本该拒绝的东西。
Task 2 已派 rev-a-t2 做**scoped re-review**（已完成，见上）（审 3d9611f 的测试文件新状态），
  重点托付一件我机器验不了的事：断言**计数**对得上不代表**取值**对得上 ——
  源与移植都是 "1" 的行里仍可能藏着改过的常数/种子/atol/shape，请逐值对拍。
  同时请它检查重写是否引入新问题（过约束、空断言 assertEqual(x, x)、辅助函数藏输入）。
Task 3: complete (commit 44c34bf) — impl-a-t3-2 (sonnet)，DONE_WITH_CONCERNS
Task 3: 产出 15 个模块：algorithms(436)/model_managers(195)/model_cache(442)/controlnet(83)/
  base_node(32)/node(1723，process 整体原样搬入)/mixins 6 个(49 方法)/__init__(11)/
  tests/test_latent_algorithms.py(133, 11 用例)
Task 3: 【控制者独立复核，不采信自述】逐方法字节级对拍：
  源 81 方法 / 新 81 方法，79 逐字相同，0 缺失；
  INPUT_TYPES 156 vs 156 行，0 差异；process 1289 vs 1289 行，恰好 1 行差异
  （即版本规则指定的 V6.11.2→V6.11.3）；3 个 @staticmethod 运行期仍是 staticmethod
Task 3: 简报自检全过：geometry @staticmethod=3、mixins def=49、mixins 横幅注释=25
Task 3: 全量 60/60 通过（控制者复跑）
Task 3: 实现者 concern 2（从 controlnet.py 导入表删掉 `hp`）—— 核实属实：
  `hp` 在整份源文件中出现 0 次，也无同名模块，import hp 会在加载时 ModuleNotFoundError
Task 3: 实现者二次回执：base_node.py 陷阱它早先已正确处理（@staticmethod 计数=0，
  1265 行是 4 空格不是装饰器）；测试文件与重生成后的简报逐行相同（105 vs 105，零差异）；
  自检全过。其报告更新至 302 行。
Task 3: 顺手清掉计划里它两次上报的两处：controlnet.py 导入表的 `hp`（已删并加注，
  核实源文件 0 次出现）；步骤 10 用例总数（改为实测 60）。
  重新生成 task-2-brief.md(434 行含新逐值门) 与 task-3-brief.md(413 行)。
Task 3: 审查 clean-with-fix (rev-a-t3, opus) — Spec ✅ / Task quality Approved
Task 3: 审查者逐项证实：5 个支撑类外移、6 个 Mixin 49 方法互不重名且顺序与表一致、
  25 行横幅注释**顺序+内容**与源逐行相同、MRO 运行期实测、44 个 self.x() 调用点全可解析、
  node.py 装配正确且**未漏搬**源 5609-5633 的注册块与启动横幅、
  process 独立字节比对 1476 行仅 1 行差异、symtable 导入完备性（含注解与默认参数）、
  无循环导入、运行期可实例化。测试文件与简报 105 行逐行相同。
Task 3: ★★ Finding 1（Important）—— **字节对拍结构上看不见的真回归**，已修 cb2883f：
  SAMModelManager.get_sam_dir 的第三个候选路径用 os.path.dirname(__file__)。
  方法体与源 639 行逐字相同，但源 nodes.py 在插件根、__file__ 就是插件根；
  搬深两层后会静默指向 nodes/qianhui_latent_yolo/models/sam（不存在）。
  已补回两层 (_PLUGIN_ROOT)，解析验证：得到 comfyui-qianhui/models/sam。
  穷举全包+源的 __file__ 只有两处，另一处是 sam2 模块自己的，不受影响；
  兄弟方法 get_model_dir 锚 folder_paths.models_dir（绝对），安全。确为孤例。
Task 3: 审查 Finding 1 修复已由审查者独立复验（normcase 相等、包仍可导入）。
Task 3: review-minor #2：两个横幅注释成了**文件末尾的孤儿**
  （mixins/policy.py:339「低显存检测」、mixins/inpaint.py:540-542「核心处理函数 V5.1」）。
  这是 brief 表格自己规定的「随前一个方法」，且与源中位置一致，非实现者偏离；
  但它们标注的那一节现在住在别的文件里（_decode_for_detection 在 inpaint.py、process 在 node.py），
  于是谁也不标注。纯装饰性问题，已 park。
Task 3: review-minor #3：死导入（algorithms.py 的 math/np；controlnet.py 的 os/np/F/comfy/folder_paths；
  model_cache.py 的 comfy；base_node.py 的 F；node.py 的 random/folder_paths）。
  控制者核实：**源文件里同样不用**（源 ControlNetHelper 对 folder_paths/comfy./F./np./os. 各 0 次；
  源 process 内 random. 也是 0 次）。即简报导入表过于宽泛、实现者忠实照做，无害。已 park。
Task 3: review-minor #4：报告 §1 把 tests/test_latent_algorithms.py 写成 133 行，实为 105。仅报告不准。
Task 3: review-minor #5：_mild_sharpen 无直接测试（经 latent_aware_resize 间接覆盖）；
  11 个用例中 8 个只断言形状（返回形状正确的垃圾也能过），3 个断言真实性质。
  但简报点名的头号风险（丢 @staticmethod 导致 TypeError）11 个用例都能抓到。
Task 3: ★ 行数口径订正：审查者报 process 1476 行 / INPUT_TYPES 176 行，我先前报 1289 / 156。
  **两者都对，是口径不同**：1476/176 是原始跨度，1289/156 是去掉空行后的行数
  （我引用的正则是 strip 掉空行后计数的）。「1 行差异 / 0 处差异」的结论两种口径下都成立。
  我先前引用时没说明是哪种口径，这里订正。
Task 3: 审查者 ⚠️ 无法验证项（重要，需用户侧闭环）：
  (a) 功能等价性 —— 无基准图；(b) 真实 ComfyUI 加载 —— 其导入跑在 stub 上；
  (c) `import nodes` 在插件真实 sys.path 下解析到哪一个 —— 需要用户的启动日志；
  (d) 已存工作流的兼容性 —— 类名 LatentKSamplerUpscalerYOLO -> QianhuiYOLO，旧 JSON 会解析失败。
      ★ (d) 用户早已裁定：「这个不用管，后面都会更新！所以这点没啥问题！」——不要再提。
Task 3: 已回问 rev-a-t3 要 findings 2+ 全文，并托付一个同族审计：
  除 __file__ 外是否还有「字节相同但环境变了导致行为不同」的地方
  （sys.path 相对导入、方法内 cwd 相对路径、__name__ 比较、inspect/traceback）
Task 3: 审查已派 rev-a-t3（审查包 review-A-T3-44c34bf.diff，291KB / 15 文件，按路径限定）。
  已告知审查者我机器验过的部分（字节对拍数字），请它专注机器验不了的：
  逐文件导入完备性、MRO 遮蔽、25 行横幅的**落位**、node.py 装配、
  以及是否漏搬了源 5609-5633 的注册块与启动横幅（不许移植）
Task 3: controller note: 早期 impl-a-t3 产出为零（402 在开工前就断了），本次是重派而非续跑
Task 3: controller note: Interfaces 段与排查提示里还留着已废弃的「_extract_hand_skeleton 挪去 controlnet.py 并改成模块级函数」，
  与方法体段（归 inpaint.py、保持实例方法不动）直接矛盾。已统一，commit 2935649。
Task 4: 进行中 — impl-a-t4 (sonnet)。brief=task-4-brief.md (666 行)。BASE=a65a752
Task 4: controller note (commit 已并入 Task 4 前的整理)：计划 step 1 的扫描脚本原本只打印
  逐对清单与它们的和 220。但同一变量会出现在多个阶段对里（ksampler 在 3 对里都有），
  逐对相加会重复计数。**真正要放进 YoloRunContext 的是并集：147**。
  已补并集计算 + 「按首次定义阶段分组」表，并在 Expected 里讲清三者关系：
    YoloConfig 99 个 = kwargs 解析出的配置项
    YoloRunContext 147 = 阶段间传递的运行时状态（上界）
    220 = 逐对相加（有重复计数，不是字段数）
Task 4: 派发时点名告知实现者：step 8 基准图是用户的活（跳过）、step 9 提交不要做、
  不改 tests/test_latent_algorithms.py（那是前一个任务的产物）
Task 5: 未开始
Task 6: 未开始 — 与 B-T3 同改 README.md，必须串行

【阻塞记录】impl-a-t2 / impl-a-t3 均以 "API Error: 402 Insufficient Balance" 终止
  - request_id: a8bb8611-1921-4f24-a9ad-24d5f8c622dc (t2)
  - request_id: f68b9013-6dbd-4133-837b-3dc1ea403f63 (t3)
  - 后果：SDD 的「实现 -> 审查 -> 修复」回路无法继续派发；B-T1 与 A-T2 的审查均欠账
  - impl-a-t3 在死前报「Confirmed a real brief defect」，控制者已独立复核并修正（见下）

【控制者修正：简报缺陷，commit 8f95af4 + 2c2ccd8】
  1. Mixin 表按 def 行给区间 -> 丢 @staticmethod（_iou_xyxy/_wiou_xyxy/_match_class_name，装饰器行 2977/2992/3057）。
     后果不是崩溃而是运行时 TypeError（self 被多绑一个实参）。已改为装饰器行起 + 3 条自检。
  2. 成员间注释横幅少算 detection.py 的 11 行（源 1529-1539）。实为 9 处 25 行，已逐行核对。
  3. node.py 搬入区间 `INPUT_TYPES(1332-1516)` 的 1332 是类文档字符串收尾，会切出孤立 `"""`。
     已改为 1334-1515，并写明文档字符串 1323-1332、__init__ 1517-1527。
  4. Task 4 阶段扫描脚本不可运行（把函数体缩进片段各自 ast.parse）。
     已改为整文件解析 + ast.walk(proc) 按 lineno 归类；顺带修掉 prepare_a/b 重叠与 repair/report 缺口。
     实测输出：跨阶段变量 220 个（该数字即 YoloRunContext 字段集的下界参考）。
  5. LatentAlgorithms 的 @staticmethod 是 9 个不是 10 个。

共享前置：tests/_loader.py 由控制者提前建立并提交 (1b03cfa)，避免 A-T2/A-T3 被它强制串行
子代理通道：402 后已恢复。2026-10-09 重派 rev-b-t1 / rev-a-t2（补 B-T1 与 A-T2 欠的审查）+ impl-a-t3-2（A-T3 实现）
A-T3 完成前的 BASE：2935649
控制者内联修复记录（不属任何子任务）：
  - B-T1 审查 Important#1/#3：tests/test_node_widgets.py 残留检查放宽为裸 `in src`，
    据此清掉 nodes/scene_node.py 的 4 处旧名文档残留 → commit f5b5e3f。全量 41/41 通过。
    内联而非重派修复轮的理由：原实现者 impl-b-t1 已死于 402 无法续跑，改动仅 1 行谓词 + 4 行文档。
  - commit 2935649：计划 Interfaces 段残留已废弃的 extract_hand_skeleton 改造，与方法体段矛盾
  - commit e4199f6：B 路 Task 2 控件数 8→9、补 test_key_count_unchanged 更新步（31/12/9/6）
审查欠账：B-T1（review-59ccf99..50f8a56.diff）、A-T2（review-1b03cfa..a0d5ecd.diff）派发中
B-T2 暂缓派发：它改的是 B-T1 动过的同 4 个文件，等 B-T1 审查结论落地再派，避免在可能有问题的基底上叠
BASE for wave 2: 1b03cfa

## Controller: Task 5 计划修复（已提交 4515894）

**发现（会话中最严重的一处）**：`PRESETS` 按**控件名**（中文）键，`apply_preset`
的 `setattr(cfg, key, value)` 要的是 **YoloConfig 英文字段名**。原 `hasattr` 守卫
对 19/19 个键全部不匹配 → 预设切换静默无效、不报错。实测结论：`19 / 19 个预设参数不会生效`。

**放大器**：原用例 `cfg = type("C", (), {})()` 的假 config 恰好由预设表的键填满，
`hasattr` 永真 —— 缺陷存在与否用例都绿。两层缺陷互相掩护。

**修法**：加 `FIELD_OF`（控件名→字段名，由 gen_yoloconfig.py 输出机械生成，19 键逐个核对），
`apply_preset` 改走 FIELD_OF 且未知键**抛 KeyError**；用例改载**真实** YoloConfig（qh_ctx），
新增 `test_every_preset_key_maps_to_a_real_config_field` 钉住映射完整性。
计数修正：17 个用例 / 区域步数 0→12 / 67 个用例。

**附带修复**：计划文件行 1188 多出一个裸 ``` 围栏（先前拼接残留），围栏总数 91（奇数）
→ awk 的 infence 翻错 → `task-brief` 报 `task 5 not found`。
删除后围栏 90、配对全对、7 个 Task 标题可见。brief 已重新成功导出（411 行）。

**验证**：brief 内容抽检 7 项全 OK；假 config 残留 1 处，在行 23 的引用块里，
是解释"为什么假 config 是错的"的说明文字，非代码。diff 逐块核对：仅围栏删除 +
两段替换 + 三处计数，无误伤。

## Controller: 关闭「nodes 包名遮蔽」疑点（无需改动）

**疑点**：插件根有 `nodes/` 目录且**无 `__init__.py`**（PEP 420 命名空间包），
而插件根本身会被加进 `sys.path`；同时我们代码里有 3 处 `import nodes` 是取
**ComfyUI 核心**的 `nodes.py`（`nodes.KSampler()`、`from nodes import CLIPTextEncode`、
`pipeline/prepare.py`）。若命名空间包抢先，会解析成没有 `NODE_CLASS_MAPPINGS` 的空包。

**实测（PEP 420 语义）**：命名空间部分**只有在整个 sys.path 上找不到同名常规模块时**才生效。
构造最小复现：`plugin/nodes/`（无 __init__）+ `comfy/nodes.py`，
PYTHONPATH 把**插件根排在前面**（最坏情况）→ `import nodes` 仍解析到 `comfy/nodes.py`，MARKER 正确。
仅放插件根时才会得到无 MARKER 的命名空间包 —— 而该情形在 ComfyUI 下不可能出现
（core 根目录是 `main.py` 的所在目录，始终在 sys.path 且 `import nodes` 早已进 sys.modules）。

**结论**：无遮蔽风险，不改动。`import nodes` 三处保持原样。

**顺带核对**：`requirements.txt` 已存在（9 字节 = `reedsolo`），Task 1 步骤 2 已落地。

## Controller: Task 4 结构等价性独立核验（测于 node.py 已缩到 267 行时）

不采信自述，用 AST 多重集对拍源 `process()`（4131-5606）与 `mixins/pipeline/*.py` 五个模块。
归一化：`ctx.X`/`cfg.X` → 裸 `X`，`self.m`/`node.m` → `@m`，只计 Load。

**结果：真正只在源里 = 1 条，即 `InterruptProcessingException` ×1。**
—— 它是 `node.py:265` 的 `except` 子句，落在通配范围之外，**不是漏搬**。
（`node.py` 里 `except InterruptProcessingException: finalize_interrupt(self, ctx)`，
 处理器体搬进了 `report.py:11 finalize_interrupt`。）

**真正只在管道里 = 35 条，全部是归一化的不对称**：源里 `samples.device` 的根名 `samples`
是局部名、本就不计数；管道写成 `ctx.samples.device`，根名成了 `ctx`，于是被计入
（`out_previews.append`/`模型.model`/`base_latent.device` 等同理）。`cfg` ×283 是 `ctx.cfg` 本身。
`YoloConfig`/`YoloRunContext`/`parse_config` 是本次新增的接线，属预期。

**次数不等 = 16 条，全部 ±1 且逐条有据**：
- `batch_size` −1、`range` −1 —— 批次主循环 `for i in range(ctx.batch_size)`
  **上移到 `node.py`**，脱离管道通配范围。两条同时 −1 正是同一个循环。
- `kwargs` +1 —— `parse_config` 多读一次。
- 其余 13 条**全是 +1**（`放大倍数`/`降噪`/`模型`/`步数`/`CFG`/`采样器`/`调度器`/`seed`/`VAE`/
  `放大算法`/`正面提示`/`负面提示`/`tuple`）—— 每个都是 `cfg.X = X` / `ctx.X = X`
  这次**升格赋值**在右值处多出的一次读取。`正面提示` 源 1 → 管道 2 正是
  「源里读一次就完事」变成「存进 ctx、后面再读一次」。

**另一项：cfg/ctx 字段与引用完全闭合。**
`YoloConfig` 字段 = 99（与计划「99 个 kwargs 解析结果」一致），`YoloRunContext` = 50。
管道里 `cfg.X` 引用去重 99 个、`ctx.X` 去重 50 个，
未定义引用 0、定义未使用 0（两侧集合相等）。
自由变量扫描（symtable）：除 Python 3.14 注解作用域的 `__classdict__`/`__annotate__`
两个合成名外，**无任何未绑定的自由变量** —— 即没有「漏升格导致裸名残留」的隐患。

**结论：Task 4 的搬运在结构上忠实，未发现漏搬或改写。** 语义等价仍需 Task 0 基准图（用户侧）。

## Controller: base_node.py 只搬 upscale_latent —— 经查为**计划明确要求**，非擅自删减

`base_node.py` 文档字符串称「源类 LatentKSamplerUpscaler 的 INPUT_TYPES 与 process
从未被调用（该基类未注册），不再移植」。逐条核实：
- 源 `NODE_CLASS_MAPPINGS` 只注册 `LatentKSamplerUpscalerYOLO`，基类确实未注册；
- 源里基类唯一被引用的方式是 `class LatentKSamplerUpscalerYOLO(LatentKSamplerUpscaler)`
  与 `__init__` 里的 `super().__init__()`，**没有 `super().process()`**；
- 计划第 813 行**逐字写着**「源 1242-1265 的 INPUT_TYPES 与 1284-1317 的 process
  从未被调用（该基类未注册），不再移植」，第 646 行也写明只搬 `upscale_latent`。

→ 实现与简报一致，**不是 finding**。已排除「擅自删代码」这一嫌疑。

**顺带发现（供最终审查）**：源文件有**两个** `process`——基类 `LatentKSamplerUpscaler.process`
（1284-1315）与子类覆写版（4131-5606）。任何按名字取 `process` 的脚本都要按 lineno 消歧，
否则会静默取到小的那个（我第一版脚本正是踩了这个坑，第二次才发现）。

## Task 4: 实现完成并提交 74e1d86，审查中（rev-a-t4, opus）

实现者报 **DONE_WITH_CONCERNS**，未提交（按指示）。控制者已提交其 8 个文件：
`mixins/context.py`(225) + `mixins/pipeline/`(prepare 271 / detect 487 / repair 555 /
global_pass 102 / report 119 / __init__ 1) + `node.py` 1723→267。
行数门全过：`process()` **15 行** ≤15、包内最大文件 **555** ≤800。生成器脚本已用完即删。

**★ 简报缺陷 3 与 4（本轮最严重）**：简报步骤 4 的行表**结构上不可能成立**——
`report.py=5483-5607` 里既有循环体内代码（5483-5500）又有 `except` 体（5502-5528）；
`prepare.py=4131-4386` 的 4386 是 `try:`，不可能是函数体末行。
又：步骤 5 的片段没有 `for`/`try`/`except`，且 `prepare_run(...)` 漏传
`模型`/`步数`/`CFG`/`采样器`/`调度器`/`降噪`（这 6 个是 `process` 的具名形参、不在
`**kwargs` 里，被 detect/repair/global 使用）—— 照字面做必是运行期 AttributeError。
**实现者没照抄，按语句结构落地并逐条上报 4 处被迫偏离**（保留 try/for/except、
逐帧累积落 global_pass、多 6 个形参、新增 finalize_interrupt）。控制者已核实并订正计划 → 75c7e49。

**控制者 AST 复核（不采信自述）**：源里 `try:` 4386、`for i in range(batch_size)` 4387
**循环体直到 5500**、`except` 5502-5528。故 detect/repair/global 与逐帧累积**确实同在一个
循环体内**，`build_report` 只调用一次 —— 实现者的 4 条偏离**条条被迫**，无一可选。

**控制者已做的独立等价性核验**（见上文「Task 4 结构等价性独立核验」节）：
读变量多重集对拍后真正只在源里剩 1 条（`InterruptProcessingException`，在 node.py:265，
落在通配范围外），16 条次数差全部 ±1 且逐条有据；99/50 字段零死零缺；无未绑定自由名。
实现者自建的「逆变换回源文本逐行比对」（6 函数 216/415/482/77/24/70 行全一致）
与我的独立口径**两条互不相干的方法结论一致**。

**待审查裁决**：§7 的三条保留意见（`del` 实例属性回落默认值 / `class_rules_text` 存展开文本 /
node.py 死导入）、以及测试基线 61 vs 简报写的 60（实现者称 61 是既有基线，60 是陈旧数字）。

## Controller: Task 5 计划对「已提交的真实代码」的闭合核验（A-T4 落地后补做）

`FIELD_OF` 是我在 `context.py` **提交之前**从实现者的生成器输出机械生成的。
现在拿**已提交的 99 字段 `YoloConfig`** 反向核验：

- 19 条 `FIELD_OF` 条目 → **全部命中真实字段，指向不存在字段的 = 0**。
- 下拉一致性：`node.py` 的 `配置预设` 实为 **7 个**选项
  （`[不使用预设]` / 5 档预设 / `自定义`），`PRESETS` 恰好 5 个顶级键 ——
  差集正是两个**语义哨兵**，不是漏配。
- 哨兵有守卫：简报第 234 行 `if not name or name in (NO_PRESET, CUSTOM): return`，
  第 242 行才对**其它**未知键 `raise KeyError` —— 我先前把「静默跳过」改成「抛错」，
  不会误伤这两个哨兵。
- 第 224 行 `PRESET_LABELS = [NO_PRESET] + list(PRESETS.keys()) + [CUSTOM]` ——
  标签表**由 PRESETS 派生**，结构上不可能与表漂移。这与「下拉 7 个 vs 表 5 个」
  看似矛盾，实为设计：两个哨兵不参与覆盖。

**结论：Task 5 的计划、已提交的 `node.py`、已提交的 `context.py` 三者闭合，可以派发。**

## Task 6 计划修复（已提交 6833971）—— 未派发前拦截的 3 处缺陷

Task 6 尚未派发，控制者拿**已提交的 HEAD** 预核验它的简报，发现 3 处会产出错文档：

1. **漏改** README:15 —— 「分组统一为 千绘/（提示词、剧情、角色、场景、动作、姿势、战斗、网格）」
   恰好是**既有** 8 个分类；新增的 `千绘/修复`、`千绘/水印` 简报只字未提。
   漏改则整合完成当日文档即错。已补该步 + 实测依据（grep CATEGORY）。
2. **目录树双尾** —— 现树里 `nodes/` 末位 `└── lora_grid_node.py`、顶层末位 `└── combat_data/`；
   简报片段要在这两处之后追加 3 条，照抄会出现两个 `└──`。已加警告要求先改为 `├──` 并重排注释列。
3. **措辞** —— 「注意事项」写「补一条」，其下代码块却是两条。已改为「补两条」。

同类缺陷计数：本会话至此已拦截 **9 处**「照字面执行会产出错结果」的计划缺陷
（前 6 处见上，第 7 处为 Task 5 预设表，第 8 处为 Task 4 行表，第 9 处为本条）。
所有修复均以**已提交代码机读实测**为依据，未凭记忆。

**Task 6 仍未派发** —— 需等 A-T4 审查结论与 A-T5 落地（完成标准含
「`nodes/qianhui_latent_yolo/` 下最大文件 ≤ 800 行」，现为 repair.py 555 ✅，
但 A-T5 会新增 presets.py，届时需复测）。

## Controller: 核验 A-T4 报告 §7 的 `del` 保留意见（结论：无行为差异，但报告有处不准）

源 5494（循环体末尾、**无守卫**）：
`del preview_image, detection_mask, detections, processed_latent, base_latent`
生成的 `global_pass.py:96`：`del ctx.preview_image, ctx.detection_mask, ctx.detections, ctx.processed_latent, ctx.base_latent`

**为何无行为差异**：这 5 个名字在每一帧都被重新赋值，故 `del` 每次都能在实例 `__dict__` 里找到：
`base_latent`/`preview_image`/`processed_latent`/`detection_mask` 由 detect/repair 无条件赋值；
`detections` 在 4846/4848-4852 的两个分支里都会赋值。

**但报告 §7 第 2 条有一处不准**（值得记，别让它进最终文档）：
它说「再读会回落到 dataclass 类默认值（None/[]）而不是 NameError」。这对
`base_latent: Any = None` 这类**成立**（类属性确实是 `None`）；
但 `detections: List[Any] = field(default_factory=list)` 是 `default_factory`
—— `@dataclass` 会把该字段的**类属性删掉**，所以未赋值时**读也会 AttributeError**，
不存在「回落到 `[]`」。即 `detections` 与源语义一致（都报错），
另外 4 个才是「静默返回 None」。因每帧必赋值，三种情形都不触发，**代码无需改**；
仅报告措辞需更正。

**顺带**：源 5483 `if preview_image is None:` 会在 `del` 之前**读** `preview_image`，
这正说明「每帧必赋值」是源码自身的既有前提，不是本次拆分引入的假设。

## ★★ Task 4 审查结论 (rev-a-t4, opus): Spec ❌ / Task quality needs work —— 1 Critical

**Critical：`mixins/pipeline/` 下 4 处相对导入少一层点 → 整包导入即失败。**
```
detect.py:7    from ..algorithms import LatentAlgorithms    ← 应为 ...
repair.py:6    from ..algorithms import LatentAlgorithms    ← 应为 ...
repair.py:8    from ..controlnet import ControlNetHelper    ← 应为 ...
report.py:7    from ..algorithms import LatentAlgorithms    ← 应为 ...
```
`detect.py` 的包是 `...mixins.pipeline`，故 `.`=pipeline、`..`=mixins、`...`=qianhui_latent_yolo。
`algorithms.py`/`controlnet.py` 在 `qianhui_latent_yolo/` 下，`..algorithms` 落到
`mixins.algorithms` —— **不存在**。控制者已独立复核（`find` 列实际文件 + 逐条解析）。
铁证：同目录 `prepare.py:7` 写的是 `from ...model_managers` —— **三层、正确**，属复制粘贴漏改。
`from ..context import ...`（4 个文件都有）**是对的**，`context.py` 确在 `mixins/` 下，不能动。

**后果**：`node.py:32-36` 在**模块顶层**导入全部 5 个 pipeline 模块，
`qianhui_latent_yolo/__init__.py` 又导入 `node.py` → 整包 `ModuleNotFoundError`，
被 `__init__.py` 的 try/except 吞掉并打印 ❌ → `QianhuiLatentYOLO` 不注册、节点从菜单消失。

**为什么我先前三项独立核验全部漏掉（重要）**：读变量多重集、cfg/ctx 字段闭合、
symtable 自由变量扫描 —— **全都只做语法层面，看不见「导入目标不存在」**；
而 61 个测试**没有一个 import node.py**。即审查者所说「零告警的全量失效」。
**这是控制者的盲区**：我把「结构等价」验到了极致，却从没验过「这东西能不能被导入」。

**修复轮 1 已派回 impl-a-t4**（resume，非新派）：
- 改那 4 处 `..` → `...`（不动 `..context`）
- **补一条守卫用例**（控制者批准的、对简报「本任务不新增测试」的定点修订）：
  遍历包内所有 `.py`，用 ast 取 `ImportFrom(level=N, module=M)`，按所在包把 N 个点
  解析成绝对路径，断言目标 `.py` 真实存在；**要求先在当前代码上跑红**再修绿。
- 重跑全量（预期 62 = 61 + 1），不提交。

**审查者另两条 Minor 与四条偏离判定**：
- `config.py` 无 `配置预设` 字段 → 审查者已独立发现 Task 5 挂载点会恒等于 NO_PRESET。
  控制者据此修了计划（见下条 commit b851787）。**审查者与我在同一处会合，互相印证。**
- `node.py` 10 个死 import（实现者已披露，维持不动）。
- `repair.py` 源 5 处「只含空格的空行」变真空行（无影响）。
- **四条偏离审查者逐条独立判定为「被迫」**，并自建了逆变换复核：
  run_detection 415/415、run_repair 482/482、run_global_pass 77/77、
  finalize_interrupt 24/24、build_report 70/70 全部逐行一致（含空行计数）；
  全 1476 行除三档前缀与 1 行版本号（该行 diff base 已是 V6.11.3，属前序任务）外零字符改动。

## Task 5 计划修复 #2（已提交 b851787）—— 计划缺陷 #10，同一失败模式的第二处

计划步骤 5 的挂载点 `apply_preset(cfg, getattr(cfg, "配置预设", None) or "[不使用预设]")`
**恒等于 NO_PRESET**：`配置预设` 是控件但 `process()` 从不读它 → 不在 99 字段内
（实测在 `context.py` 里出现 0 次）→ `getattr` 恒 None。
**这正是我先前修掉的「表键空间错位」的同一个失败模式** ——
那次我修了表**内部**的映射，却漏查**喂给表的那次查找**。
已改为 `kwargs.get("配置预设") or NO_PRESET`，并补第 7 条守卫用例
（解析 AST 断言 `apply_preset` 第二实参含 `kwargs` 不含 `getattr`）。
计数更新：18 个用例 / 全量 68。简报已重新生成（466 行）并复核。

**计划缺陷累计：10 处。**

## Controller: 修复轮 1 独立复核（impl-a-t4 的产出，未采信自述）

**4 处导入已改对**，且 `..context` 四处**未被动过**（正确，`context.py` 确在 `mixins/` 下）：
```
detect.py:7   ...algorithms     ✅   detect.py:8   ..context   ✅（不变）
repair.py:6   ...algorithms     ✅   repair.py:7   ..context   ✅
repair.py:8   ...controlnet     ✅   report.py:8   ..context   ✅
report.py:7   ...algorithms     ✅   prepare.py:7  ...model_managers ✅（本来就对）
```

**守卫用例已补**：`tests/test_latent_algorithms.py` 新增
`TestPackageRelativeImports.test_relative_import_targets_exist`。
它没有硬编码期望，而是用 `importlib.util.resolve_name(rel, pkg)` 把每个
`ImportFrom(level=N)` 解析成绝对模块名，再断言对应 `.py`/目录真实存在
（还顺带捕获「点太多、超出顶层包」的 `ImportError`）。这个做法比硬列表更稳。

**控制者两向实测（这是守卫唯一的价值所在）** —— 用守卫自身的解析逻辑逐条判定：
```
  ..algorithms     -> qianhui_latent_yolo.mixins.algorithms    **不存在 → 报红**  ← 旧 bug 会被抓住
  ...algorithms    -> qianhui_latent_yolo.algorithms           OK
  ..controlnet     -> qianhui_latent_yolo.mixins.controlnet    **不存在 → 报红**  ← 旧 bug
  ...controlnet    -> qianhui_latent_yolo.controlnet           OK
  ..context        -> qianhui_latent_yolo.mixins.context       OK
  ...context       -> qianhui_latent_yolo.context              **不存在 → 报红**  ← 反向也守
  ...model_managers-> qianhui_latent_yolo.model_managers       OK
```
即**两向都守**：既抓得住原来的少一层点，也抓得住「把本来正确的 `..context`
顺手改成 `...context`」这种反向过度修正。

**全量实测**：`Ran 62 tests OK`（61 + 新增 1），与预期一致。

## 修复轮 1 · 实现者报告要点（DONE）+ 控制者复核

**根因（实现者自查，控制者复核属实）**：生成器的 `IMPORT_LINE` 表把
`algorithms`/`controlnet` 两行按 **`mixins/` 的写法**抄来 ——
`mixins/inpaint.py:10-11` 实测正是 `from ..algorithms` / `from ..controlnet`，
在那个深度**正确**；放进 `pipeline/` 就少一级。同表 `model_managers` 写的是三层点（正确）
→ 抄表漏改层级，非设计分歧。

**控制者独立审计**：遍历整包 **34 处**相对导入，`importlib.util.resolve_name` + 文件系统判定
→ **0 处失效**。与实现者自报的「34 处、恰好 4 处 BAD（修复前）」完全一致。

**为什么 6 项既有检查全漏（实现者自述，我认同并记档）**：
读变量对拍只比**代码行文本**；`ast.parse` 只做**语法**；自由名扫描只问
**「名字是否绑定」**（`from ... import X` 无论几个点都把 `X` 记成已绑定）；字段闭合只看**属性名**。
**没有一项问过「导入目标是否存在」**。这与控制者三项检查（读变量多重集、cfg/ctx 字段闭合、
symtable 自由变量）的盲区**完全同源**。

**实现者另建执行级探针** `.superpowers/.../probe_imports.py`（gitignore，不入库）：
桩掉 ComfyUI 后 `exec_module` 真正执行 5 个阶段模块与 `node.py` 的模块体。
它故意把 `detect.py` 改回错误层级验证过，复现出与推断一致的两级失败：
```
FAIL detect.py  ModuleNotFoundError: No module named 'nodes.qianhui_latent_yolo.mixins.algorithms'
FAIL node.py    ImportError: cannot import name 'run_detection' from '...pipeline.detect'
```
→ 即 `run_detection` 拿不到 → 节点不注册、菜单消失。恢复后全部 OK、MRO 正确。

**两条 Minor：实现者同意不修，控制者采纳**：
- `node.py` 10 个死 import：本轮是**纯结构搬移**，删 import 会把「搬移」与「清理」混进同一 diff；
  且 `import nodes` / `comfy.*` 这类顶层导入误删的失效模式**与本轮 Critical 同类**（导入期炸）。
  → 建议单独一个小提交。
- `repair.py` 5 行空白：根因确认 —— 搬运函数里 `if text.strip() == "": out.append("")`
  把纯空白行归一成真空行；源行实测为纯空格（长度 28/28/20/24/24）。
  若日后重新生成，改成 `out.append(text[base:])` 即逐字节还原。不为纯空白去动已审查的 555 行文件。

**遗留（同上一轮，未闭环）**：能导入 ≠ 行为等价。步骤 8（用户重启 ComfyUI、
同 seed 重跑场景 A-E、逐像素 + `检测信息` 逐字符比对）仍是唯一的等价性结论。

## 控制者提请最终审查考虑的一项（不自行加码）

那条 Critical 暴露的不只是「点了几个点」，还有**「61 个测试没有一个 import node.py」**。
这意味着**任何**导入期失效（名字拼错、基类缺失、MRO 冲突、符号不存在）都同样零告警。
实现者的执行级探针证明这类缺陷可以低成本捕获（桩 ComfyUI + 真 `exec_module`）。
**控制者只批准了一条静态守卫**（已验两向有鉴别力）；是否再把执行级冒烟测试**固化为长期用例**，
留给计划 A 的最终整分支审查裁决 —— 避免我在修复轮里自行扩大范围。

## Task 4 · scoped 复审回执 (rev-a-t4)：**findings 全部解决**

**Q1 Critical 已解** —— 审查者**自己实测**（非采信控制者）：
- 4 处确已 `...`；5 处 `..context` **未被过度修正**（它指出 `...context` 会指向
  不存在的 `qianhui_latent_yolo.context`，与我的两向实测结论一致）
- 它自己的解析器扫**整个 `nodes/`**（不只 `qianhui_latent_yolo`，共 **40** 条相对导入）
  → **解析失败 0 条**（修复前 4 条）
- 桩掉第三方后 `exec` 五个 pipeline 模块**真实源码** → 全部 OK（修复前 detect/repair/report 三文件报错）

**Q2 守卫有鉴别力** —— 审查者在临时目录（不碰仓库）用**该用例自己的代码**做两向实测：
```
A  未改动              -> GREEN  Ran 12 tests OK
B  4 处 .. 全部回退     -> RED    FAILED，点名 文件:行 + 解析出的不存在目标
B2 只回退一部分         -> RED    （鉴别力细到单条）
   恢复                 -> GREEN
```
即报红是真红、报绿是真绿。

**Q4 无新问题**：遍历全包不误报；`__pycache__` 的 `.pyc` 被后缀过滤；
PEP 420 命名空间包**不会造成假红**（守卫纯用文件系统 + `resolve_name` 的级别算术，
不经过 import 系统，级别与包名前缀无关）→ 换机器/换路径结果一致。

**Q5 两条 Minor 维持原判**（均未被本轮修复触及）。

**结论：findings 全部解决，未引入新 Critical/Important。**

## ★ 但 Q3 找出守卫自身的两个**真实盲区** + 一个假红 → 控制者开修复轮 2

审查者用投毒实测逐条验证（每条都在运行期必然炸）：

| 形态 | 守卫 | 是否现实 |
|---|---|---|
| `from .. import algorithms`（module-less） | **GREEN（漏）** | **现实且典型**。`n.module` 空 → `rel = ".."` → `resolve_name` 返回**包自身** → `isdir` 通过 → 不存在却判绿 |
| `from ...algorithms import NoSuchClass`（名字拼错） | **GREEN（漏）** | **现实**。只校验模块**路径**存在，不校验被导入的**名字** |
| `import 不存在的模块` / 漏装第三方 | GREEN（漏） | 超出「相对导入」命题范围 |
| `from ....xxx import y`（level ≥ 4） | **RED（假红）** | 低。守卫把顶层包硬编码为 `qianhui_latent_yolo`，而运行期插件包是 `<plugin>.nodes.qianhui_latent_yolo`、`nodes/` 是命名空间包 → level 4 合法 |

**控制者的决定**：审查者判「非阻塞」，但我开**修复轮 2** 修掉 ①②③。理由：
那条 Critical 的失败类别是「**整包导入即失败**」，而 `from .. import algorithms`
**恰恰属于同一类别**。留一个**已知有洞**的守卫、却记成「已加回归守卫」= 把保护说满了。
假红同样有害 —— 会诱使后来者把守卫改弱。

轮 2 要求（已派回 impl-a-t4）：①module-less 时按「包 + 子模块名」判定；
②静态校验被导入**名字**存在（AST 扫目标模块顶层绑定，不真 import）；
③把「顶层」重设为**插件根**（`nodes/` 的父目录）使层级算术与运行期一致。
验收：①②各投毒**先红后绿**且报红点得出 `文件:行`；③反向验证合法 `....foo` **不报红**；
真实代码仍全绿；用例数仍 62。并授权它：若某条静态做不到，**说出来并给取舍**，不要硬凑会误报的实现。

**注意**：在轮 2 落地前，对本守卫的准确表述是
「**已覆盖 level 1–3 的模块路径存在性**，module-less 与名字级错误尚不覆盖」——
不得简写为「已加回归守卫」。

## Task 5 计划修复 #3（已提交 da27cc3）—— 计划缺陷 #11，与 T4 Critical 同类

原步骤 7 让实现在 `nodes/qianhui_latent_yolo/__init__.py` 加**模块级**
`from .presets import as_payload as preset_payload` 并进 `__all__`。
该文件现仅 11 行，加模块级导入 = 把 `presets.py` 变成**整个节点包的硬依赖**：
presets 里一个笔误（正是本任务最易错处）→ 包导入失败 → 根 `__init__.py` 的 try/except 吞掉
→ **`QianhuiLatentYOLO` 不注册、节点从菜单消失**。失败模式与刚修的 T4 Critical 同类。

节点本身不需要 presets（步骤 5 的导入在 `parse_config` **函数体内**，调用时才执行）；
唯一消费者是步骤 8 的路由，而路由**已有** try/except。
→ 步骤 7 改为「不要做模块级导入」，步骤 8 直接
`from .nodes.qianhui_latent_yolo.presets import as_payload as preset_payload`，
失败面限定为「路由不可用」而非「节点消失」。

**顺带**：我改这段时引入过围栏奇偶失衡（`~~```python` 不以 ``` 开头，不被计为开启围栏
→ 总数变奇数 → `task-brief` 会重新认不出 Task 标题）。已修，现围栏 90、7 个 Task 标题全可见。

**其余 Task 5 锚点预核验（拿已提交代码查，全部属实）**：
- `# 8b. 前端联动数据路由` 块在**根** `__init__.py:128`（含 `from aiohttp import web`、
  `from server import PromptServer`、`@PromptServer.instance.routes.get("/qianhui_role_data")`）
  —— **不在** `qianhui_latent_yolo/__init__.py`。简报里「`__init__.py`」措辞含糊，
  但步骤 7/8 的代码块分别用 `.presets` 与 `.nodes.` 前缀，足以消歧，未改。
- 步骤 3 的插入锚点「第 9 个加载块之后、`# 8b.` 之前」与根文件结构一致。
- 19 条 `FIELD_OF` 全部命中已提交的 99 字段（先前已验）。
- 哨兵守卫 `if not name or name in (NO_PRESET, CUSTOM): return` 在位。

**计划缺陷累计：11 处。**

## Task 5 派发前预核验（拿已提交代码机读，未改任何文件）

**预设值的类型与范围合法性** —— 从 `node.py` 实测解析 19 个控件的声明
（`(类型, 选项字典)`），与简报里 5 档 `PRESETS` 的取值逐项比对：

| 类别 | 控件数 | 结论 |
|---|---|---|
| BOOLEAN | 9 | 预设值全为布尔 ✅ |
| INT | 5 | 全落在 `[min,max]` 内 ✅ |
| FLOAT | 2 | 全落在 `[0.0,1.0]` 内 ✅ |
| 下拉（list） | 2 | 预设值全部取自其选项表 ✅ |

**非法预设值 = 0。** 顺带印证计划那句「`区域步数` 默认是 **0**（`INPUT_TYPES` 实测），
不是 20」—— 实测 `INT [0,10000] 默认 0`，故预设 12…36 合法且从 0 变 12 是最直观的验证点。

**其余锚点**（先前已验，此处汇总）：`# 8b.` 路由块在根 `__init__.py:128`；
19 条 `FIELD_OF` 全命中 99 字段；哨兵守卫在位；步骤 6 的层级自检期望 `level = 3`
与修复后的三层点一致。

**T5 派发时机**：`tests/test_latent_algorithms.py` 是同一条串行链上的共享文件
（T5 要追加 `TestPresets`，而 T4 轮 2 的复审若需修改会在 `TestPackageRelativeImports` 内），
故等复审回执后再派，不冒并发改同一文件的风险。

## ★ Task 4: complete（74e1d86 + 修复轮 1 = 5d24a2f + 修复轮 2 = 27ee555）

**审查与两轮复审结论**：
- 任务审查 (rev-a-t4)：Spec ❌ → 1 Critical（4 处相对导入少一层点 → 整包导入即失败）
- 修复轮 1 → scoped 复审：Critical 彻底关闭，审查者独立复现（扫全 `nodes/` 40 条相对导入 0 失败；
  桩掉第三方真 `exec` 五个阶段模块全 OK）
- 修复轮 2（补守卫盲区）→ scoped 复审：**轮 2 可以关闭**。它逐条投毒复现
  （`from .. import <不存在>` → RED；`from ...algorithms import NoSuchClass` → RED；
  原 Critical 形态 → **仍 RED**，覆盖未削弱），并确认假红已消。
- **它为我（实现者）那处偏离背书**：`from ... import QianhuiLatentYOLO` 是合法导入，
  而 `qianhui_latent_yolo/QianhuiLatentYOLO.py` **不存在**（该名字仅是 `__init__.py` 的顶层绑定）
  → 只认「子模块」必误报。并对照 CPython `_handle_fromlist`
  （先 `getattr(module, name)`，失败再 import 子模块）证明「或」判据与运行期同构。

**parked Minor（交最终审查三角分级，不进修复轮）**：
rev-a-t4 在轮 2 复审里用 `exec` 在 Python 3.14.2 上先确认「运行期确实做模块级绑定」，
再判定守卫 —— 找出 **4 处误报**，均为「静态收集器不完整」而非判据错，加 4 个 `isinstance`
分支即可收口：
`match ... case int() as X`（`ast.Match`）/ `except*`（`ast.TryStar`）/
`(X := 1)`（`ast.NamedExpr`）/ `type X = int`（`ast.TypeAlias`）。
另两处**漏报**（`if TYPE_CHECKING:` 内绑定、`except E as X` 的 handler 名）属安全方向。
**现实性：4 处误报都要求「某模块用这类形态做模块级绑定」且「被别人按名 `from … import`」——
本仓库零处**（无 match、无顶层海象、无类型别名、无 except*）。

控制者裁决：**不自己扩大修复轮**，记入 parked Minor 交最终审查。
理由：对本仓库零触发，而 T5 才是用户可见价值所在；且误报方向的风险由
「本仓库零处」这一机读事实兜住。若最终审查要求收口，改动是 4 个 `isinstance` 分支。

**遗留（未闭环，需用户）**：步骤 8 基准图比对仍是唯一的等价性终审。
**另一条已实现的保障**：实现者的执行级探针（`.superpowers/.../probe_imports.py`，gitignore）
能真 `exec_module` 五个阶段模块与 `node.py`；控制者已提请最终审查裁决是否将其固化为长期用例
（因为这条 Critical 暴露的还有「61 个测试没有一个 import node.py」）。

## Task 5 已派发 —— impl-a-t5 (sonnet)，BASE = da27cc3

brief = `task-5-brief.md`（484 行，已含我在派发前做的 3 处计划修正）。
report = `task-5-report.md`。

派发词里额外交代了简报不可能知道的东西：
- `YoloConfig` 已存在且恰 99 字段（`mixins/context.py`）；`parse_config(kwargs) -> YoloConfig` 已存在
- **相对导入必须三层点**（步骤 6 会自检 `level = 3`）；并说明为何绝对导入
  `from nodes.qianhui_latent_yolo.presets import ...` 必失败（`nodes/` 被核心 `nodes.py` 抢占）
- **`配置预设` 不是 `YoloConfig` 字段**（`process()` 从不读它），其值只在 `kwargs` 里
- 我派发前已用已提交代码机读校正的三处，明确要求**照简报现在的写法、不许改回去**：
  `FIELD_OF` 19 条（字段全部真实存在）、挂载点 `kwargs.get(...)`、
  步骤 7「不要做模块级导入」（`qianhui_latent_yolo/__init__.py` 一行都不要动）
- 已核实 5 档预设的**全部取值**对 19 个控件都合法（类型 + 范围），**不许改数值**
- **`TestPackageRelativeImports` 一个字都不许动**，`TestPresets` 追加在文件末尾
- 单文件跑测试必须用 `discover -s tests -p "..."`（`tests/` 不是包）
- 不 `git add`、不 commit、不 `git add -A`、不顺手清理无关代码

## Task 5: 实现完成并提交 140b90a，审查中（rev-a-t5, opus）

控制者复核（未采信自述）：
- `git status --short` 暂存前恰好 6 项：T5 的 5 个路径 + 计划文档 1 个（我自己的订正）。
- 暂存后 `git diff --cached --name-only` = 恰好那 5 个路径；计划文档留在未暂存区。
- 根 `__init__.py` 的 diff **只有** 8c 块追加（已逐行看过），未夹带 Plan B 的改动。
  `nodes/qianhui_latent_yolo/__init__.py` 全程未被触碰（不在 git status 中）。
- 两个提交：`140b90a`（T5 实现，5 文件 / +314 行）、`5112ce0`（我的计划订正）。
  分离提交是为了让审查包 `da27cc3..140b90a` 恰好等于 T5 的产出。

审查包 = `review-da27cc3..140b90a.diff`（18208 B，1 commit，5 文件）。

派发词原文携带了实现者的 3 条自述（FIELD_OF 反向无人守 / 用例数 19+69 / 插入点在 main 块前），
**逐条要求审查者自行核验并自行判断**，明确写了「它们是 claim 不是事实」「不要假设控制者已裁决」，
未做任何预判，也未写「不要报」。

## Task 6 计划修复（已提交 5112ce0 之外的更早 6833971）—— 仍待派发
Task 6 依赖 T5 完成（它要写 README 里的 `配置预设` 一行、`requirements.txt`、目录树）。
T5 审查闭环后即可派发；与 B-T3 同改 README.md，需串行。

## Task 6 派发前预核验（已提交 8301af4）—— 又拦下 2 处，均为「编辑自身破坏计数」

方法同 T5：拿**已提交的 HEAD** 机读，未改任何文件。两处都属「计划让改 A，
但改完 A 会让同一文档里的 B 变错」，与已拦下的 #9 同族。

**#12** 目录树首行 `├── __init__.py  # 注册入口 (10 节点统一 Qianhui 前缀)`。
机读 `__init__.py`：11 处 `NODE_CLASS_MAPPINGS["…"] = …` 直接赋值
＋ `NODE_CLASS_MAPPINGS.update({…})` 中的 story 2 个 = **13**。
简报只让补三行树条目，从未提这个数字。已写入计划：改成 `13 节点`。

**#13** README:3 `ComfyUI 综合提示词插件，由四个历史插件整合而成：`，
其下表格现恰好 **4 行**（机读确认）。简报让加「水印插件」一行 → 5 行，
但没让改「四个」。已写入计划：改成「五个」。

两处都给了机读依据与确切目标值，且注明「全 README 各只有这一处人工计数」（已机读确认）。

**同时机读确认、判为「不是缺陷」的两项**（避免误报）：
- 节点列表表内「节点」列**故意不写版本后缀**：现存 10 行全部如此
  （表内 `千绘 · 角色定制` vs 实测显示名 `千绘 · 角色定制 (V36)`；
  10 行无一例外）。故简报新行 `千绘 · Latent智能修复`（不带
  `(YOLO+SAM+CN V6.11.3)`）**符合既有约定**，不是缺陷。
- 完成标准「`nodes/qianhui_latent_yolo/` 下最大文件 ≤ 800 行」已成立：
  实测最大 = `mixins/detection.py` 727 行。criterion 可达，不是空头支票。

## Task 6 已派发 —— impl-a-t6 (sonnet)，BASE = 8301af4
与 rev-a-t5 并行：**文件集不相交**（T6 只动 README.md，rev-a-t5 只读）。
Plan B 已全部完成（其台账末行为「全部完成」），README.md 当前工作区干净，
不存在与 B-T3 的争用。

## Task 5 审查回执 (rev-a-t5, opus)：Spec ✅ / 0 Critical / 1 Important(plan-mandated) / 6+ Minor

**控制者独立复核了它的三条 ⚠️ 与两条 Minor，未采信自述。**

**★ ⚠️ #2「预设究竟有没有影响采样？」—— 已机读闭环，结论是「有，不是缺口」。**
这是 T5 若不成立则整件事作废的那个问题。审查者说它「住在 diff 之外」，那正是控制者该接的。
方法：取 `presets.py` 的 19 条 `FIELD_OF` 控件名，用 `ast` 在全包（22 个 py）
枚举**所有** `kwargs.get("控件名")` / `kwargs["控件名"]` 取值点。结果：

    取值点总数 = 19，其中在 pipeline/prepare.py 之外 = **0**

19 条全部且仅仅落在 `prepare.py` 的 `parse_config` 里（:23,24,25,45,46,54,66,70,73,
126,128,129,134,136,138,140,141,145,150 —— 正是 FIELD_OF 的 19 条）。
反向抽查下游确实读 cfg 字段而非 kwargs：
`region_steps` → repair.py:449；`fix_rounds` → repair.py:464；`blend_mode` → repair.py:335。
**故 `apply_preset(cfg, …)` 写进 cfg 的值会被全链路读到，预设真实生效。**
这也顺带把 FIELD_OF 的**两个方向**都坐实了（键是控件名、值是 cfg 字段）。

**⚠️ #1「浏览器路径」** —— 属实无法静态验证，是用户动作（需在真 ComfyUI 里点一次）。
**⚠️ #3「全量基数 62」** —— 审查者自行核了本文件基线 12（2+2+5+2+1），与 19/69 自洽。

**Minor #5 我复核为真**：`prepare.py:158` 注释写「见下方警告」，其下一行即
`apply_preset(...)`、再下一行 `return cfg` —— **下方没有任何警告**。逐字抄自简报，
而简报里那段警告在代码块**之外**。真实、可一行修掉。
**Minor #6 我复核为真**：`as_payload()` 返回 `{"presets":…, "labels": PRESET_LABELS}`，
而 `js/latent_yolo_presets.js` 全文**零次**读 `.labels`（只在 :22 读 `.presets`）。
即 `labels` 是死载荷。真实。

**Important #1 是 plan-mandated**：简报步骤 8 原文（brief:385）
「沿用同样的静默失败模式」+ `except Exception:`。已核 8c 与既有 8b 逐字同构，
且文件里 11 个**注册**块全部是 `print(f"❌ …")`——两种模式按用途分。
按流程 plan-mandated 须交用户裁决（静默 vs 记一行日志 vs 至少加注释），
**控制者不自行改**。已计入待问用户清单。

**Minor #2 是本轮最有价值的一条**（FIELD_OF 反向无人守）：审查者给了个便宜闭环 ——
`mixins/context.py:31-151` 每个字段旁就有 `# 键: <控件名>` 注释，是这套桥接的
**机读陈述**；解析它做一次字典比对即可同时钉住两个方向，无需解析 node.py。
转交最终审查裁决。

## Task 5: complete（140b90a + 计划订正 5112ce0）—— 判决：Spec ✅ / Task quality Approved

审查者三项裁决我逐条核过，均**独立复现**：
- 用例数：本文件既有 12（TestLatentAwareResize 2 + TestGaussianFeather 2 + TestBlends 5
  + TestColorMatch 2 + TestPackageRelativeImports 1）+ 新增 7 = 19；62+7=69，与 `Ran 69` 自洽。
- 插入点：diff 末尾即 `if __name__ == "__main__": unittest.main()`，
  其后定义的东西在直接执行时永不收集 —— 简报字面照做会「假绿」。
- FIELD_OF 反向：真实但**潜伏**（19/19 键、95/95 取值经两方各自手核全合法），
  失败模式仅「面板显示失真、后端仍按 FIELD_OF 生效」，故 Minor、不必在本任务内关闭。

## ★ 控制者查设计文档，把两条 Minor 判到位（这是审查者查不到的一层）

**Minor #7（预设是否持续锁定）—— 设计文档 §2.6 末句有张力，须用户定夺。**
- 设计文档 line 258 **明确要求**：后端为权威，`prepare.py` 解析完即在**每次运行**
  按表覆盖那 19 个字段（为覆盖 API/无头/队列等 JS 不跑的路径）。
- 设计文档 line 263 **同时**说：ⓐ 预设是「一次性写入，**不是**持续锁定」；
  ⓑ 「若用户改完**又重选**同一个预设，值会被重新覆盖」——ⓑ 的措辞暗示
  「不重选就不覆盖」。
- 实测实现（prepare.py:159 在全部 kwargs 读取**之后**执行）：只要 `配置预设`
  仍停在那一档，后端**每次运行**都会覆盖 → 事实上就是持续锁定。
  故 ⓑ 这句与实现不符（ⓐ 只是「没拦着你别改」，不是「改了算数」）。
- 代码本身**忠实执行了 line 258 的明令**，故我**不判为代码缺陷**；这是
  「文档两句自相矛盾 + 想要的 UX 语义未定」→ 交用户。

**Important #1 的 plan-mandated 属性被设计文档坐实到更强一级**：
设计文档 line 418 异常处理表原文「`/qianhui_preset_data` 路由注册失败 →
**静默跳过（沿用现有 `pass` 模式）**，JS 拉取失败时保持 ComfyUI 侧默认控件值，
预设仍由后端兜底生效」。即：这不是计划作者的疏忽，而是**用户已批准的、
带记录理由的设计决定**。仍按流程交用户裁决，但呈报时必须带上这条依据。

**关键澄清：Minor #6 的 `labels` 不是死代码，是 spec 要求的接口。**
设计文档 line 261/258 要求预设表「以后端为唯一来源」，`labels` 属该接口的
组成部分（供前端展示档位名）。审查者自己也说「spec-mandated，保留它」。
我不改判，但记账时区分「spec 要求的载荷」与「实现遗漏的消费点」。

## Task 5 待用户裁决清单（与 T6、终审项合并成一次提问）
1. 8c 的静默 `except`（设计文档 418 明令 vs 审查 rubric）——保留 / 记一行日志 / 至少加注释
2. 预设语义：每次运行覆盖（当前实现）还是仅重选时覆盖（文档 263 的口径）
3. 浏览器路径与基准图比对（用户动作）

## Task 6 第 1 轮 —— impl-a-t6 (sonnet) DONE_WITH_CONCERNS，README 改动控制者机读复核通过

**控制者机读复核（未采信自述）**：
- 注册名集合：README 13 个 vs 代码 13 个 —— **set 相等**，两侧差集均为空。
- 分类集合：代码 10 个 vs README 括号里 10 个 —— **set 相等**，差集为空。
- `四个→五个` ✅；`10 节点→13 节点` ✅；两处 `└──`→`├──` 均在 ✅；
  `nodes/` 子树的 `#` 列已统一（含新增两行）✅；`requirements.txt` 行作为顶层末位 `└──` ✅。
- `git status --short` = ` M README.md`，无夹带。

**实现者的 4 条顾虑，处置如下**：
1. ★ **采纳并扩大任务范围** —— 见下「第 14 号缺陷」。实现者按 README-only 授权
   **如实上报而未越界**，这正是我要的行为。
2. 简报自己的两行树条目差 1 列（`qianhui_latent_yolo/` 的 `#` 在第 29 列、
   `qianhui_watermark/` 在第 28 列）—— **是我的简报错误**，实现者归一化到 29 列，
   保留不动。
3. `js/`、`tests/`、`tools/`、`docs/` 不在树里 —— **HEAD 既有缺口**（该树历来只列
   节点/数据目录），非本次引入。**不扩范围**，转最终审查。
4. 「README 第 15 行」是 HEAD 行号，加完表行后落在 16 —— 实现者按**内容**定位，
   正确。已记入报告事实。

**实现者自报的返工**（值得记，因为它自己说第一遍的自检没抓住）：
写坏两次（先压平层级、再错拼子节点），两次都作废，最终从 `git show HEAD:README.md`
一次性重生成。故审阅的 diff 是单次干净应用，无修补残渣 —— 控制者的集合等价与
对齐检查也从侧面印证了这点。

## ★ 第 14 号缺陷（已提交 00e0ec2）—— 同一类人工计数的第 3 处，实现者上报，控制者确认

`__init__.py` 的模块 docstring 也是文档，里面同样硬点了这些数：
- `:4` `由四个历史插件整合而成：`，其下 4 条来源插件（`:5-8`）
- `:10` `统一注册全部 10 个节点（10 类，剧情模块含编排/调度两节点）`

本次整合后为 **5 个来源插件 / 13 个节点**。只改 README 不改它，
两份文档当场互斥，而本任务的完成标准恰是「与代码一致」。
**已写入计划并重新生成简报**：改 `:4`、`:8` 之后补第 5 条来源、`:10` 的两个 10 都改 13。

⚠️ 我在简报里点明了一个陷阱：`:10` 的**两个 10 是节点数/类数**，不是分类数；
分类数**恰好也是 10**（8 旧 + 修复 + 水印），是巧合。若不点明，很容易
「看括号里是 10 就以为在说分类、于是留着不动」。机读确认 13 个注册名对应
13 个互异的类。

## Task 6 第 2 轮已派发（同一实现者 impl-a-t6，与 T5 同法续跑）
只要三处 docstring 改动 + 追加同一报告文件的第二节；不 commit、不 `git add`。
审阅 BASE 定为 `00e0ec2`（计划文档提交），使审查包只含实现者的产出。

## 控制者对 T6 报告的补充机读（读完全文后，未采信自述）

**目录树对齐独立复核**：顶层 `#` 全部落在第 26 列、子层全部第 30 列（各自统一）；
每层**恰好一个** `└──`，且都在该层末位。实现者自报的「归一化到 29 列」与我的
读数差 1，是我用 `find()+1`（1-based）与它口径不同所致，**不是分歧** —— 关键结论
「同层同列」一致。脚本一度报「顶层参差」，经查是我把根行 `comfyui-qianhui/`
（整行无 `#`，`find` 返回 -1 → 0）也算进集合所致的**我自己的假红**，非真问题。

**报告 §3、§4 里值得存档的两条**：
- §3.3 简报「`qianhui_latent_yolo/` 比 `lora_grid_node.py` 长 3 个字符」成立
  **唯因结尾斜杠**（不带 19 字符=长 2，带 20 字符=长 3）。实现者据此把名称域宽
  取 21 而非 20 —— 这正是「差一格会画出 `qianhui_latent_yolo/#` 无空格」的边界，
  它抓对了。
- §4 实现者自述的两次返工（压平层级、错拼子节点）与「以 `git show HEAD:README.md`
  一次性重建」的决定 —— 与我的集合等价/对齐检查结论一致（diff 是单次干净应用）。

**另两条转最终审查（均为 Minor、非本次引入）**：
- §5.2 目录树未列 `js/`、`tests/`、`tools/`、`docs/`（HEAD 既有缺口）
- §5.3 「注册约定」段的旧名例子与新增的注意事项第 2 条**例子集合分处两节**，
  语义一致但可合并 —— 实现者未改，我认可不越界；转终审。

## Task 6: 实现完成并提交 62605fc（README + __init__.py 模块头），审查中（rev-a-t6, sonnet）

**控制者复核（未采信自述）**：
- `git status --short` 暂存前 = ` M README.md` + ` M __init__.py`，**仅此两个**。
- `__init__.py` 的 diff = 恰好 3 个 hunk（`@@ -4 +4 @@`、`@@ -8,0 +9 @@`、`@@ -10 +11 @@`），
  全在 docstring（第 4–11 行）之内；**docstring 之外零改动**（脚本按 hunk 头过滤确认）。
- 三处计数一致：docstring 五个 ✅ / `统一注册全部 13 个节点（13 类…` ✅ / 残留旧数「无」✅。
- 来源插件条目：docstring 5 条 ↔ README 表 5 行 ✅。

**第 2 轮前的一次假象（记录以备复盘）**：我一度收到 impl-a-t6 的**重复投递**——
内容与第 1 轮逐字相同的 idle 通知。我没有据此认为第 2 轮已完成，而是直接查
工作区（`__init__.py` 仍是旧数、报告无第 2 节），确认是陈旧重投后**重发了指令**。
教训：**投递内容不等于当前状态**；状态只能从工作区读。

## 终审准备（已就绪，待 T6 审查闭环后派发）

**审查包** = `final-review-9d4f276..HEAD.diff`（365KB / 7818 行 / 37 个文件）。
- BASE 取 `9d4f276`（两条计划线之前的最后一个提交）。
- 用**路径限定**把范围收成计划 A 拥有的文件：`__init__.py`、`requirements.txt`、
  `js/latent_yolo_presets.js`、`nodes/qianhui_latent_yolo/**`、`nodes/qianhui_watermark/**`、
  `tests/_loader.py`、`tests/test_latent_algorithms.py`、`tests/test_watermark.py`、
  `tests/assets/`、`README.md`。
- **排除计划 B 的文件**（`nodes/{character,combat,pose,scene}_node.py`、
  `tests/test_node_widgets.py`）—— 已单独终审并判「可以合并」，重审只添噪音。
- 两条线**互相不碰**的机读依据：计划 B 改的 4 个节点里没有 `QianhuiLatentYOLO`
  也没有水印节点，故 A 的预设「按控件名匹配」与 B 的键名去前缀无交集。
  （唯一交叠：`cb2883f` 是 B 的审查修复，碰到了 A 的 `model_managers.py`
  与 `test_watermark.py` —— 已在包内，会被终审看到。）

**存量问题清单** = `parked-findings.md`（40 行），按 skill 要求交给终审分诊：
A1–A8（T5 审查者）+ B1–B4（T4 遗留，含「是否把执行级探针固化为长期用例」）
+C1–C3（T6）+ 4 项用户侧动作。
每条都带**证据与出处**；A1/A3 明标 plan-mandated，A7 明标「设计文档自相矛盾、须用户定夺」。

## 订正上一条「第 2 轮前的假象」的归因（实现者给出 mtime 证据，我复核并采信）

我先前把那次困惑归因为「投递内容 ≠ 当前状态」。**结论对，但归因不完整。**
实现者补上另一半，我核了 mtime：

    README.md     19:21:17   ← 第 1 轮写完
    __init__.py   19:23:40   ← 第 2 轮写完
    task-6-report 19:24:36   ← 第 2 轮报告写完

我那次查工作区（看到旧数、报告无第 2 节）**发生在 19:23:40 之前** —— 即两次写
之间的真实中间态，不是幻觉。我随后重发指令是**多余但无害**的一步；实现者回
「ALREADY DONE」并给出可复现证据，是对的。

**根因是两条叠加**：
1. 路由把第 1 轮的报告**重复投递**了一次（我收到的「新消息」内容与上一轮逐字相同）；
2. 我的检查早于第 2 轮的写盘。
两者都指向「必须读工作区」，但**只读工作区不够 —— 还要在时间上确认读到的是哪一版**。

**另一条实现者指出的、我当时判断错的事实**：`.superpowers/` 整棵树在 `.gitignore:4`
里被忽略，故报告**永远不会**出现在 `git status` / `git log` / `git show` 里。
我当时用 grep 直接读磁盘文件（读到了 section 1–6），所以那一步没错；
但如果换成任何 git 视角去看，会误判为「报告不存在」。**台账与报告是 gitignore 的
scratch，只能从磁盘读。**

**终态复核（未采信自述）**：`git status --short` 为空；`git show HEAD:__init__.py`
前 14 行三处改动齐全（`由五个` ✅、水印插件条目 ✅、`- ` 条目数 = 5 ✅、
`13 个节点（13 类…）` ✅、残留 `由四个`/`10 个节点` = 无 ✅）；
报告第 2 节在磁盘第 249 行。**无丢失、无缺陷。**

## Task 6 审查回执 (rev-a-t6, sonnet)：Spec ✅ / 0 Critical / 1 Important / 4 Minor

审查者做了我要求的那件核心事 —— **逐条把新写的散文拿代码对照**（文档任务最容易说假话），
而且全部为真（我抽查复核）：
「6 Mixin」= mixins/ 下 6 个类 ✅、「5 阶段函数」= pipeline/ 下 5 个模块 ✅、
「五档」= presets.py、「DCT-QIM」= invisible.py、「依赖 reedsolo」= payload.py、
「不保留旧注册名」= 全仓无 `LatentKSamplerUpscalerYOLO` 且无中文映射键 ✅、
「未安装时可见水印仍可用」= visible.py 只 import numpy/PIL/.fonts，不在 reedsolo 路径上 ✅。

它还独立确认：`10` 这个字符串在 `__init__.py` 里**已完全不出现**（我复核 `grep -c "10"`
= **0** ✅）；树的两层 `#` 列**实测**统一（非目测）；9 行被重排的既有条目
**仅空白差异、无注释改写**。

### ★ 它抓到的唯一 Important —— 我在简报里写的**假话**（第 15 号计划缺陷）

`README.md:104` 原文：「`Latent 智能修复` 与 `水印` 模块需要额外依赖：`pip install reedsolo`」。
**这是错的。** 我机读复核：

    grep -rn "reedsolo" --include=*.py nodes/ __init__.py
      → 命中全部在 nodes/qianhui_watermark/**（payload.py:3,19；invisible.py:34,107）
    grep ... | grep -v qianhui_watermark
      → **空**

即 Latent 修复节点**不需要** reedsolo。照计划写会误导用户以为不装它修复节点就降级。
`requirements.txt` 内容恰为 `reedsolo` 一行，与「只有水印需要」一致。

**处置**：审查者按流程标了 plan-mandated 并提请我签核。我判定**这是计划缺陷（#15），
不是计划的有意抉择** —— 计划自己的完成标准就写着「文档与代码一致」，
而这一句让文档与代码不一致。故与 #12/#13/#14 同法：**订正计划**（已提交 `d0b3d0c`）、
重新生成简报、派实现者改。**同时在给用户的汇报里明写这一处，不静默处理。**

顺带采纳审查者的 Minor #4（同一行内）：运行期报错 `payload.py:19` 本就写着
「请在 ComfyUI 使用的 Python 环境中安装」，README 漏了这句 —— 既然这一行要动，
就一起补上（审查者自己也建议「若该行本就要改则一并对齐」）。

### 转终审的 Minor（已并入 parked-findings.md）
- C1 目录树未列 `js/`、`tests/`、`tools/`、`docs/`（HEAD 既有缺口）
- C2 `__init__.py:111/:119` 段落编号跳 `# 10.`（既有编号问题，非节点计数）
- C3 报告 §1.5/§2 的列号标签差 1（0-based 25/29 vs 1-based 26/30）——纯记账，不影响产物

## Task 6 第 3 轮已派发（同一实现者）
只改 `README.md:104` 一行；不 commit、不 `git add`、行数须不变。

## Task 6 第 3 轮（修复轮 1）—— 已提交 5727895，scoped 复审已派（rev-a-t6 续跑）

**控制者复核（未采信自述）**：
- `git diff --numstat README.md` = **1 1**（单行替换，未增删行）。
- diff 逐字即目标改动：删「`Latent 智能修复` 与」、补「需装到 ComfyUI 使用的 Python 环境」。
- 全文 `reedsolo` 只剩 3 行：L33（水印节点表行）、L58（树里 requirements.txt 行）、
  L104（本注意事项）—— 三处**都在讲水印**，合理。
- **同句兼有 `reedsolo` 与 `Latent` 的行 = 无** ✅（这是本条 finding 的判据）。
- README 行数 105 不变；`__init__.py`、`requirements.txt` 无 diff；工作区已净。

提交 `5727895`。复审包 `review-62605fc..5727895.diff`（2 个提交 ——
含我的计划订正 `d0b3d0c`，已在派发词里**明确标注那是控制者的文档改动、
不算实现者的 fix-diff 破坏**，避免复审误判）。

**实现者自报的一条过程问题，值得记**：它三轮里**自己的校验脚本共 4 个 bug**
（第 1 轮 2 个、第 3 轮 2 个：一个「reedsolo 只出现 1 行」的伪断言——实际合法地
出现 3 行；一个 Windows GBK 下读 grep 输出崩溃）。它明确说「四个都没影响产物，
但这是我自己该收紧的模式，不是一次性事故」。**主动交代自身工具缺陷且区分
「影响产物/不影响」**，这正是我希望实现者做的事。

## Task 6 · scoped 复审回执 (rev-a-t6)：**All findings addressed, no new breakage**

- Finding「README:104 假依赖归属」→ **ADDRESSED**。复审者独立核了
  `grep -n "reedsolo" README.md` 只命中 33/58/104，
  且 `grep -n "reedsolo" README.md | grep -c "Latent"` = **0**。
  它另认定新增的「需装到 ComfyUI 使用的 Python 环境」属实（与 payload.py:19 逐字一致），
  README 与运行期报错从此口径统一。
- New breakage: **None**。1 行进 1 行出；上下文（注意事项标题、前 4 条 bullet、
  其后「整合自水印插件的 3 个节点…」）与它在 62605fc 审过的版本逐字节相同；
  105 行不变；4 行围栏完好。
- 它还主动核了范围里那个「不属于实现者」的提交 `d0b3d0c`（我的计划订正），
  结论是内容无误、不构成 fix-diff 破坏 —— 与我派发时的标注一致。

## ★★ 计划 A 六个任务全部闭环

| 任务 | 交付 | 审查 |
|---|---|---|
| T1 水印模块整合 | `a0d5ecd` + `3d9611f` | ✅ |
| T2 骨架/依赖/注册 | `fefa345` | ✅ |
| T3 Latent 拆 15 模块 | `44c34bf` | ✅ |
| T4 process() 阶段化 | `74e1d86` + 修复 `5d24a2f` `27ee555` | ✅（审查 Critical 已闭环） |
| T5 配置预设五档 | `140b90a` | ✅ Spec ✅ / 0 Critical / quality Approved |
| T6 文档同步 | `62605fc` + 修复 `5727895` | ✅（复审 All addressed） |

## 整分支终审已派发 —— final-rev-a (opus)

**审查包** = `final-review-9d4f276..HEAD.diff`（**已重建**，366KB / 7824 行 / 37 文件）。
重建原因：先前那份是在 `62605fc` 时生成的，此后多了 `5727895` 的修复行 —— 若不重建，
终审会漏看那一处。

派发词里交代的、终审需要而它自己查不到的东西：
- 计划 A 的两条核心承诺（**拆分是行为保持的**；唯一有意行为变更是新预设功能）
- 计划 B 已单独终审通过、其 5 个文件**刻意排除**在本包外，及「两线不共享任何节点」的依据
- 两条**我刻意不自行裁决、提请终审裁定**的事项：
  ① 执行级导入冒烟探针是否固化为长期用例（背景：T4 那条 Critical 暴露了
     「69 个测试没有一个 import node.py」）
  ② 静态守卫对 `ast.Match`/`TryStar`/`NamedExpr`/`TypeAlias` 四种顶层绑定的假红面
     是否现在就收口（本仓零处触发）
- 指向 `parked-findings.md`，要求它对 A1–A8 / B1–B4 / C1–C3 逐条分诊
  「合并前必修 / 可以留」；并特别提示 A1 要权衡**设计文档 line 418 的明令**
  与审查 rubric 对「吞异常」的立场。

## 用户两项原始要求的终态机读核验（控制者做，等终审期间）

**要求①「把水印插件整合进千绘 + 把超大文件按功能拆成小组件」**
- 拆分后**最大单文件 727 行**（`mixins/detection.py`），拆分前是 5633 行一行文件。
  全包 6544 行 / 22 个 py 文件，`node.py` 聚合器 267 行、`process()` 15 行调 5 阶段。
- 注册名 **13 个，无一例外全带 `Qianhui` 前缀**（非千绘前缀的 = 无）。
- 三个旧注册名 `LatentKSamplerUpscalerYOLO` / `WatermarkAdd` / `WatermarkExtract`
  作为注册键的出现次数均为 **0**。
- 水印包落地为 6 模块 + `requirements.txt`(reedsolo)；`千绘/修复`、`千绘/水印` 两个分类存在。

**要求②「4 个插件左侧数字全部删掉 + 圈选的几个控件删掉」**
- 4 个节点的 `INPUT_TYPES` 顶层控件名里，带数字前缀的 = **0 个**（四个节点全 ✅）。
- 圈选的控件名（构图范围 / Lora触发词 / 手动补充 / 模型类型 / 环境总权重 / 补充指令）
  作为控件名出现 **均 0 处** ✅。
- 控件数实测 **31 / 12 / 9 / 6**，与计划 B 台账数字**逐一对上**。

**一次我自己的假红，记下来**：我先把「任何以 `NN_` 开头的字符串」当数字前缀，
于是在肢体动作师/战斗指挥官里报出 `01_global.json`、`01_weapons.json` 等 ——
那是**数据文件名不是控件标签**。同样，我按「INPUT_TYPES 函数体内所有 dict 的键」
数控件，得到 32/19/10/10（把嵌套选项 dict 也算了进去），与台账的 31/12/9/6 不符；
改按**顶层 `required`/`optional` 键**数，立刻逐一对上。**两次都是我的量法错，
不是产物错** —— 记下来是因为「数字对不上」时第一反应该是查口径，不是查代码。

**全量测试**：`Ran 69 tests ... OK`；单文件 `Ran 19 tests ... OK`。

## ★★ 整分支终审回执 (final-rev-a, opus) —— 前半部分，控制者逐条复核

**它做了一件本会话最有价值的独立验证：拿源目录对拍，证明「行为保持」不是空话。**
源目录按约束保留不动，故 `E:\comfuyui 插件\水印\nodes.py` 是真基准。它按名配对函数：
源 105 个函数 → 新包缺失 **0**；按名配对 112 个 → 逐字节相同 **109**，
不同的 3 个（`__init__` 聚合类、`get_sam_dir`、`process`）都有正当理由。
`process()` 抽出 902 条叶子语句 → 新代码 **901 条逐字命中**，
唯一对不上的是版本横幅 `V6.11.2`→`V6.11.3`（计划强制）。
**这是整套改动最难挣的东西，它挣到了 —— 而且是别人重算出来的，不是自述。**

它另外独立确认（我抽查复核）：
- `YoloConfig` 99 字段：写 99 / 读 99 / 未声明读 0 / 未声明写 0 / **死字段 0**；
  `YoloRunContext` 50 字段未声明访问 0。正是「局部变量→属性」最容易出错的地方。
- 100 个 `kwargs.get()` 键全是真控件名；113 个控件全部有归属 → 无「永远静默取默认值」。
- 95 个预设取值全部类型/范围合法；`PRESET_LABELS` 与控件表**连顺序都相等**。
- 水印 17 个用例与源 `水印/test_watermark.py` 同名 1:1，**逐用例比断言条数与每个
  数值边界字面量 → 0 个被削弱**；`test_invisible_capacity_error` 反而从 try/except
  **强化**为 `assertRaises`。
- 降级路径端到端验证：屏蔽 `reedsolo` 后包正常导入、节点正常注册、可见水印正常、
  隐形路径抛出的消息与 README 那句**字面一致**。
- 注册约束：13 处注册、10 分类全 `千绘/*`；8 个旧 key 全消失；
  `RETURN_TYPES`/`FUNCTION` 原样，只有 `CATEGORY` 按计划改。

**架构判断（回答「是不是化妆式拆分」）**：它用探针实跑验证依赖图**单向无环**：
`algorithms`/`model_managers`/`model_cache`/`controlnet`/`base_node`/`context` 是叶子，
5 个阶段模块**彼此不互相导入**、只经 `ctx` 交换状态，`node.py` 是唯一聚合点。
它的原话结论：不是化妆式拆分；共享状态走显式容器是分阶段管线的正常做法。

**Critical：无。** 它说明自己专门找了能撑起 Critical 的几类失败
（语句丢失 / 字段错映射 / kwargs 控件漂移 / 预设越界 / 断掉的导入图），逐类查过都没找到。

### Important I1 —— 与我的 A7 同源（预设每次运行覆盖用户控件，界面不承认）
它给了**具体修法**：只动前端约 6 行 —— 给那 19 个控件各挂 callback，
手改时把 `配置预设` 置为 `自定义`。界面立刻诚实，文档 258/263 的矛盾不再影响行为。
它判定这是**计划/设计缺陷而非实现缺陷**，与我先前的判断一致。

### Important I2 —— 我逐条复核，**成立**（这是测试缺口，非产品缺陷）
它指「让 T4 那条 Critical 溜过去的两个缺口都还开着」。我的机读复核：
1. **守卫的 root 写死**：`tests/test_latent_algorithms.py:254`
   `root = os.path.join(nodes_dir, "qianhui_latent_yolo")` —— 只扫这一个包。
   水印包**完全没有导入图守卫**。
2. **水印包的 `__init__.py` 从未被执行**：`tests/test_watermark.py:39-44` 用
   `spec_from_file_location` + `submodule_search_locations` 把 `qh_watermark`
   **只注册进 `sys.modules`、不 exec `__init__.py`**。文件里写明了理由（若 exec，
   `from .nodes import ...` → `import torch` 会成为整个模块的导入期依赖，
   13 个纯 numpy 的算法用例会整体报错，而不是让 4 个节点用例单独跳过）。
   故 `__init__.py:2` 那条导入**无任何测试跑过**。
3. **但当前不是活缺陷**：我用 `importlib.util.resolve_name` 把水印包**全部 6 条**
   相对导入逐条解析，**全部 ✅ 正确**（`__init__.py:2`、`invisible.py:5,6`、
   `nodes.py:5,6`、`visible.py:5`）。即：与 T4 那条 Critical 同一失败形状，
   在这一侧**潜伏而非现患**。
4. 修法很便宜：`_scan(root, plugin_root)` 本来就是个带 root 参数的通用函数，
   把水印包也传进去即可（改 2 行，或加一个用例方法）。

## 终审 Important I3 / I4 —— 我逐条复核，并发现 I3 与我自己的一条计划修复**互相拉扯**

### I3 成立：`PRESET_LABELS` 是个**没人消费的「唯一来源」**
机读证据：
- `presets.py:90` `PRESET_LABELS = [NO_PRESET] + list(PRESETS.keys()) + [CUSTOM]`（7 项）
- `presets.py:117` `as_payload()` 把它当 `"labels"` 发出去
- **JS 从不读它**（`grep -c labels js/latent_yolo_presets.js` = 0）
- **`node.py:96-106` 手抄了同一份 7 项列表**（顺序也相同），连
  `"default": "[不使用预设]"` 都是字面量

于是 `presets.py` docstring 与设计文档 line 261「预设表以后端为唯一来源」这句话，
**对下拉框本身是假的**：今天两份一致（我核过，连顺序都对），但加第 6 档要改两处，
漏改 `node.py` 那一处 → 该档**选了也不可达**，而**测试全绿**（`TestPresets` 只测
`presets.py`，没有任何用例读 `node.py` 的 `INPUT_TYPES`）。

### ★ 但我发现 I3 的修法与**我自己在 da27cc3 做的计划修复互相拉扯**
- 终审给 I3 的修法是「用 `PRESET_LABELS` 生成控件列表」→ 那要在 `node.py` 里
  **模块级** `import presets`。
- 而 I4 引用的 da27cc3（**我的计划修复**）当初正是为了**避免让 `presets` 成为
  `node.py` 的模块级硬依赖**（否则 `presets.py` 一有语法错，整包导入失败、
  节点从菜单消失 —— 与 T4 那条 Critical 同类）。
- 即：照 I3 的字面修法会**重新引入** da27cc3 要消除的失败模式。

**可行的调和（留给修复轮/用户）**：不动模块级导入图，改为**加一个测试**断言
`node.py` 的控件列表与 `PRESET_LABELS` 逐字同序 —— 用 `ast` 解析 `node.py` 取字面量，
**不需要 import `node.py`**。既堵住「漏改一处且全绿」，又不改依赖关系。
（若确实想消灭重复，次选是在 `INPUT_TYPES()` **函数体内**导入并兜底，
但那是运行时路径，收益不如测试直接。）

### I4 成立但确实最弱：惰性导入把「降级」掉了个头
`prepare.py:158` 的 `from ...presets import apply_preset, NO_PRESET` 在**函数体内**。
方向对（不把 presets 变成 node 的硬依赖），但效果与初衷相反：
`presets.py` 坏掉时不再是「节点消失 + 启动日志 ❌」，而是
**「节点在、每次运行抛 ImportError」**。而 `presets.py` 本身不 import 任何包内部东西，
它要防的风险极小；且根 `__init__.py` 8c 块已经 import 它了（在 try/except 里）。
终审自己标注「这是我最弱的 Important，两行可修，别让它挡住合并」。
**注意**：改这一条时要与 I3 的调和方案一起看，别把依赖图改回去。

## ★ 终审 Ruling 1 / Ruling 2 —— 控制者**独立复现了变异测试**（未采信其表格）

方法：把 `nodes/` 与守卫用例复制到临时目录（**从不触碰仓库**），用 `ast` 单独抽出
`TestPackageRelativeImports` 类并 exec，然后逐项注入故障、直接调 `_scan`：

| 注入的故障 | 终审表格 | **我的复现** |
|---|---|---|
| 干净基线 | GREEN | **GREEN** ✅ |
| 模块级 `raise RuntimeError`（**导入期实际执行**） | GREEN（盲） | **★ GREEN（盲）** ✅ 复现 |
| 循环导入（`global_pass` ↔ `report`，两边模块名与名字都真实） | GREEN（盲） | **★ GREEN（盲）** ✅ 复现 |
| 相对导入少一层点（复现 T4 那条 Critical） | RED | **RED** ✅ 复现 |
| `if TYPE_CHECKING:` 内绑定的名字被别处 import | —（终审另指） | **★ GREEN（假绿！）** ✅ 复现 |

**结论：终审的变异表格属实，且它额外指出的 `TYPE_CHECKING` 假绿也属实。**
`test_latent_algorithms.py:172-174` 无条件下潜 `ast.If` 体，
故 `if TYPE_CHECKING: Y = ...` 会把 `Y` 当成真顶层绑定 → 别处 `from .m import Y` 判绿，
而运行时必然 ImportError。**这是假绿，方向危险**（前 4 种构造全是假红、方向安全）。

**故 Ruling 1（把执行级探针固化为长期用例）我判定为成立且应采纳**：
两类结构性失明（导入期执行、循环导入）静态守卫原理上就看不见，
而探针是**唯一碰 `node.py` 的东西**（顺带验证 6 Mixin 的 MRO 聚合，此外无任何用例覆盖该路径）。
终审的三条附带条件我也认同：保留静态守卫（互补而非替代）、
别过度吹探针覆盖面（它自己 `mkpkg("nodes", PLUG)` 把 `nodes` 指向插件自己的目录，
看不见真 ComfyUI 里 `import nodes` 的遮蔽）、
顺带把守卫扫描根从 `qianhui_latent_yolo` 放宽到 `nodes/`（一改关掉 I2）。

**Ruling 2（4 个假红构造现在不收口）我也认同**，理由比我预想的更充分：
四者包内各 0 处；且**漏收集产生假红而非假绿**（会响亮打在守卫用例上、当场被加语法的人看到）；
现在补 4 个分支 = 往自研静态分析器里加 4 段**没有测试的**新代码，
而这个分析器最大的风险恰恰是腐烂（已有 4 个已知盲区）。**记为已知限制。**

## 终审对存量清单的分诊结果
- **合并前必修**：**A7**（= I1，唯一用户可见缺陷，前端约 6 行）、
  **B4**（= Ruling 1，清单一里价值最高的一条）
- 其余 A1–A6、A8、B1–B3、C1–C3 全部**可合并**
- 值得注意：终审对 **A1** 的判词 —— 设计文档 418 行明令静默 `pass` 属计划强制，
  但它指出「文件里 11 个注册块全 print，唯 8c 不 print，『与邻居一致』反而支持补上
  `print(f"❌ …{e}")`」。这与审查者 rev-a-t5 的立场一致，且给了新的论据。

## ★ 控制者纠正终审的一处论证错误（不向下传播）

终审在确认 `TYPE_CHECKING` 假绿时加了一句：「这一类**已经存在于隔壁包** ——
`qianhui_watermark/payload.py:2-5` 是顶层 `try: from reedsolo import RSCodec /
except ImportError: RSCodec = None`，守卫的 `ast.Try` 分支会 walk handler 体，
故 `RSCodec` 被记成顶层绑定 —— 与 `if TYPE_CHECKING` 同一个假绿类，
只因扫描根是 `qianhui_latent_yolo` 才没咬到。**又一条支持放宽范围+固化探针的理由**。」

**它的前半（守卫行为）对，后半（结论）错。** 我机读复核：
- `payload.py:2-7` 在 `except ImportError as exc:` 分支里**显式绑定** `RSCodec = None`。
- 屏蔽 `reedsolo` 后实跑该模块：**执行成功，`RSCodec` 确实被绑定**（值 `None`）。
- 故 `from .payload import RSCodec` 在运行时会**成功** → 守卫报 GREEN 是**真绿**。

区分两个机制（不可混为一谈）：
| 形态 | 运行时该名字是否存在 | 守卫 GREEN 的性质 |
|---|---|---|
| `if TYPE_CHECKING: Y = 1` | **不存在**（TYPE_CHECKING 为 False） | **假绿** ← 真缺陷 |
| `try: from X import Y`<br>`except ImportError: Y = None` | **一定存在**（except 分支绑定了它） | **真绿** ← 不是缺陷 |

即：守卫的 `ast.Try` 分支把 handler 体的绑定收进来**是对的**，因为
「try/except 兜底赋值」这一形态的**语义就是**该名字在两种路径下都存在。
（若换成 `except ImportError: pass` 而不绑定，才会变成假绿 —— 但 payload.py 不是这样。）

**处置**：我**不把这条传给修复轮**，也不会据此去改守卫的 `ast.Try` 分支。
`TYPE_CHECKING` 那条假绿仍成立（我自己复现过），Ruling 1 / Ruling 2 结论**不受影响**。

## 整分支终审总评：**With fixes**
- **合并前必修**：I1（预设与界面不一致）、B4（= Ruling 1 探针固化）
- 其余 A1–A6、A8、B1–B3、C1–C3 全部「可合并」
- 对拆分的签字依据：105 源函数 0 缺失 / 109 逐字节相同 / process() 902 条只差版本号一行
- 终审明说 A1、B1 属**计划强制**、不算缺陷

## 已停下等用户裁决（skill：所有任务完成后即可停）
三项待用户定：I1 的 UX 方向、合并前收口范围、A1 静默异常是否补 print。
另有 4 项**只能由用户闭环**的动作（真 ComfyUI 重启 / 浏览器肉眼确认预设联动 /
固定 seed 比对基准图 A–E / `data/composition.json` 留删）。

## 用户裁决（2026-10-09，三项）

1. **I1 预设与界面不一致** → **改前端：手改即翻回「自定义」**
2. **合并前收口范围** → **按终审建议全做**（① 探针入 tests/；② 守卫扫描根放宽到 nodes/；
   ③ 断言 node.py 预设列表与 PRESET_LABELS 逐字同序；④ 删 node.py 死导入）
3. **A1 8c 静默异常** → **补一行 print，与 11 个邻居一致**

## 收口派发前控制者的现场核验（三处，其中一次我把自己的结论推翻）

**① 探针文件存在**：`probe_imports.py`（2614 B）+ `probe_guard.py`（5033 B）。
探针用 `mkpkg` 注册包但不 exec `__init__.py`，然后真 `exec_module` 5 个阶段模块与
`node.py`。**注意**：它算 `ROOT` 用的是 `__file__/../../..`（它住三层深），
搬到 `tests/` 后层级变了 —— 已写进派发词要求改用 `_loader.ROOT`。

**② ★ 我连续两次把自己的结论推翻，最终确认终审的 14 是对的**
- 第一遍我用「字符串只在导入行出现」当判据 → 报 23 个死导入，
  连 **6 个 Mixin 与 5 个阶段函数** 都被判死。**我的阈值写错了**（`>1` 应为 `>=1`），
  而它们在类定义行里明明被用到。
- 第二遍改用 AST（`Name(Load)` + `Attribute` 根名）→ 报 **10 个**，
  于是我在给用户的汇报里写了「正确是 10，不是 14」—— **这句话本身也错了**。
  原因：`import comfy.sample` / `utils` / `model_management` / `controlnet` 四行
  **绑定的都是同一个名字 `comfy`**，我的 AST 见 `comfy` 被用到就把 5 行全算活。
  但它们之中只有 `comfy.samplers` 被真的以属性路径使用。
- 第三遍按「属性路径」查 → `node.py` 里用到的 comfy 路径**只有** `comfy.samplers.*`。
  **终审的 14 是对的。**
- 再逐条查了「删了会不会让某模块变成孤儿」：`algorithms.py` 另被 base_node / inpaint /
  masks / detect / repair / report 六个模块导入；`controlnet.py` 另被 inpaint.py:11 与
  repair.py:8 导入 → **都不成孤儿**。且 `comfy.model_management` 的每个使用方
  （detection / inpaint / detect / repair / sam / model_cache）**都自己 import 了它**。
- 结论：14 行全可安全删除。**教训再记一次：同一份代码三种量法给出 23 / 10 / 14，
  只有第三种对 —— 「数字对不上」时先怀疑自己的口径，这次三遍都是我的量法出问题。**

**③ 派发词里写死的两个陷阱（实现者若不知道几乎必错）**
- **JS 的 `applyToNode` 会调用被改控件的 callback**（js:28）→ 天真实现会在用户
  刚选中预设的瞬间就把下拉翻回「自定义」，**静默废掉这个功能**。要求用标志位
  在程序化写入期间抑制翻转。
- **Item 4 不能用「把控件列表改成从 `PRESET_LABELS` 生成」来修** —— 那会让 `presets`
  变成 `node.py` 的模块级硬依赖，正是 `da27cc3`（我做的计划修复）要消除的失败模式。
  要求改用 `ast` 解析、**不 import `node.py`**。

## 收口任务已派发 —— fix-a-final (opus)，一次性覆盖 8 项
1. JS 诚实化（用户裁决）2. 探针入 tests/（Ruling 1）3. 守卫放宽到 nodes/（I2）
4. 预设列表对钉（I3）5. FIELD_OF 双向守卫 + 95 取值合法性（A2 / 终审 Rec 3、4）
6. 删 14 行死导入（M1/B2）7. 8c + 8b 补 print（用户裁决）8. 设计文档 258/263 自相矛盾订正
报告写入 `final-fix-report.md`；**要求它用变异证明新冒烟测试真的会红**
（弄坏一处相对导入 → 必须红；模块级 raise → 必须红而静态守卫仍绿），
这是本轮最重要的证据 —— 否则那条测试不配进 tests/。

## 终审收口（fix-a-final, opus）DONE_WITH_CONCERNS —— 8 项全做，控制者独立核验

**独立复跑（未采信自述）**：
- 全量 `Ran 82 tests ... OK`（原 69，+13）
- `tests/test_latent_algorithms.py` `Ran 23 ... OK`（原 19，+4）
- 新 `tests/test_import_smoke.py` `Ran 9 ... OK`
- `node --check js/latent_yolo_presets.js` 退出码 0
- 工作区恰好 6 个路径（5 改 + 1 新），与 8 项一一对应

**CRLF 卫生（它主动提醒的）**：`git ls-files --eol` 全仓检查 →
所有文件 `w/lf`，**工作区无一处 CRLF 混入**；也没有「status 显示改动、diff 为空」的陷阱。
它自己发现并修好了「用 Python 文本模式写回导致 6 个文件 LF→CRLF」的问题，值得记。

**新增诊断的一处澄清**：新测试里 `sub.keys` / `-node.operand.value` 被 Pyright 标红 ——
属**类型收窄不足**（`_dict_node_value(d, key)` 的 `d` 无标注，Pyright 推成 `ast.expr`），
**不是运行期缺陷**：测试实跑通过即为证。其余 `comfy.samplers` / `server` / 相对导入
无法解析，全是「插件只在真 ComfyUI 里能解析」的环境产物，非缺陷。
（可留一条给用户的 Minor：`sub` 少一句 `isinstance(sub, ast.Dict)` 断言，
将来 node.py 结构一变会抛难懂的 AttributeError 而非清晰断言。）

## ★★ 循环导入那一格：我做了三次才做对，前两次的注入是**假的**

**第一次（/tmp 隔离）**：把 `from .report import build_report` **追加在文件末尾**。
→ 文件体已跑完、名字已定义，循环**良性**，两边都绿。我据此「确认」了守卫盲 —— 结论对但证据虚假。

**第二次（真仓，同上追加）**：同样良性。守卫绿、冒烟也绿 ——
**这解释了实现者所说「静态守卫结果是 load-error 而非绿」为何与我不同：它的注入大概也是末尾追加。**

**第三次（真仓，把 import 插在模块定义之前，构成真循环）**：
```
静态守卫用例  → Ran 23 tests ... OK          ← BLIND（绿）
新冒烟测试    → 3 个 ERROR：
    ImportError: cannot import name 'run_global_pass' from partially
    initialized module ... (most likely due to a circular import)   ← RED，且信息精确
```
**真循环下：守卫盲、冒烟红。** 终审的变异表格因此**成立**，
而实现者的「load-error」观察是**假注入的产物**。已还原，工作区干净。

**这条证据正是新冒烟测试存在的理由**：静态守卫对良性与恶性循环一视同仁地绿，
而只有真执行模块体才能撞出「部分初始化模块」。Ruling 1 站得住。

**教训（本会话第 N 次）**：「注入故障」的**位置**决定实验是否有效；
追加在文件末尾的导入几乎总是良性的。要在写入前先问「这个注入真的会激起我要的那类失败吗」。

## 控制者通读修好后的 JS（js/latent_yolo_presets.js，133 行）—— 逐点判定

**抑制是真承重的**：`applying` 只在 js:39 一处被读（`flipToCustom` 首行早退），
而 js:65 的 `w.callback(w.value)` 正是会再次触发挂上去的那个监听器 ——
**删掉 js:39 这一行，选中预设的瞬间下拉就会把自己翻成「自定义」**。
js:59/67 的 `try/finally` 保证某个控件 callback 抛异常也不会把标志卡死。

**几处正确性我逐条核过**：
- js:89 `if (name === WIDGET) continue;` —— 不给下拉框自己挂监听，避免自我否决
  （其实 `配置预设` 本就不在预设表的 19 个键里，这是防御性写法，无害）。
- js:45 翻转后同步触发下拉的 callback → 走 `applyToNode(nd, "自定义")` →
  js:57 `if (!table) return` 早退，**不会递归、不会再翻**。
  即 `自定义` 在前后端两条路径上都是干净的终止态。
- js:86-100 `_qhPresetCovered` 保证只包装一次，重复调用不会把 callback 套娃。
- js:106 `prev.call(this, v)` 保留 `this`。
- **不会覆盖已存 workflow 的控件值**：`_qhPresetSync` 只在
  `this._qhPendingPreset` 有值时才 `applyToNode`（js:101）—— 即只在
  「用户确实选了预设、但表还没到」时补应用；**不在节点创建时无条件重放当前档**。
  实现者主动说明它刻意避开了后者，判断正确。

**我发现一处窄边界（Minor，不阻塞）**：
用户先选预设（表未到 → js:53 记 pending），**随后**在表到达前手改某个被覆盖控件，
数据到达后 js:101 重放 pending → 用户那次手改被覆盖，且因 `applying=true` 不会翻下拉。
窗口是首屏亚秒级，且语义上「选了预设」本就意味着「按这档来」，故可辩护；
但严格说这是 `_qhPendingPreset` 机制的一个小副作用。记为已知边界。

**我无法验证的部分（照实说）**：浏览器里真实前端的行为没跑过 ——
Node 桩测的是「逻辑正确」，不是「在真 ComfyUI 里点得动」。这是用户侧动作。

## ★★ N1（复审新发现，Potential 回归）：我查上游源码把它**证伪**了

**复审的主张**：I1 的 `flipToCustom` 监听器只由 `applying` 保护，而 `applying` 只在
本扩展自己的写入期间为 true。**任何**其它在 `applying=false` 时给被覆盖控件写值并
触发其 callback 的调用方，都会把 `配置预设` 翻成「自定义」→ 后端于是**跳过预设**。
加重情节：`配置预设` 在 113 个控件里排 **index 24**，而被覆盖的 19 个中有 **16 个排在它之后**；
litegraph 的 `widgets_values` 还原是按下标顺序，于是那 16 个会在档位之后逐个写值、逐个翻走 →
用户看到「自定义」（像是自己选的），而实际运行与已存 workflow 静默不同。

**我先独立核了它的索引论断 —— 完全属实**：
机读 `node.py::INPUT_TYPES` 声明顺序：控件总数 **113**、`配置预设` index **24**、
19 个被覆盖控件中**在它之前 3 个、之后 16 个**
（之后的是 YOLO推理尺寸 / 多尺度检测 / 启用SAM精细化 / 启用EMA注意力 / 启用Tile检测 /
使用Seg分割 / 区域步数 / 区域降噪 / 修复轮次 / 开启裁剪精修 / 裁剪最大边 / 裁剪步数 /
裁剪降噪 / 裁剪轮次 / 二次检测精修 / 融合模式）。

**但我把上游源码读了，它的加重情节不成立。**
`comfy-org/litegraph.js` 的 `LGraphNode.configure(info)`（`src/LGraphNode.ts:721`）
对 `info.widgets_values` 的处理是**纯直接赋值**：
```js
widget.value = info.widgets_values[i]
```
**没有** `widget.callback(...)`、**没有** `onWidgetChanged(...)`、也没有任何变更通知。
`configure` 里唯一被调用的是节点级钩子 `this.onPropertyChanged?.(k, ...)`（复制 properties 时）
与末尾的 `this.onConfigure?.(info)` —— 都不是 per-widget callback。

**结论**：**还原已存 workflow 不会触发我们挂的 `widget.callback`**，
故「16 个控件按下标顺序翻走档位」这一情节**不成立**。
N1 的机制（第三方程序化写值 + 主动调 callback 会翻）仍然真实，
但本插件内唯一的此类调用方就是 `applyToNode`，它被 `applying` 正确地挡住了；
ComfyUI 核心在 restore 路径上不做这件事。

**我保留的诚实限定**：
① 我读的是 litegraph **master**，用户实际运行的是 ComfyUI pin 的版本，二者可能不同；
② ComfyUI 自己的前端 `app.js` 理论上可能在 `configure` 之后再补一次带 callback 的同步
  —— 这一步我没有源码可查（本机无 ComfyUI 前端）。
故 N1 由「Important / 可能是我引入的回归」**降级为「Minor / 潜在」，并附上上述两个未验证点**，
交用户与最终判定参考，不作为阻塞合并的理由。

**处置**：`js/latent_yolo_presets.js` 不改（改了反而是在没有证据的情况下增加复杂度）。
把两个未验证点写进给用户的收尾说明，建议用户在真 ComfyUI 里做一次「存 → 重开」的肉眼确认
（这本来就是用户侧清单里的一项，现在多了一层理由）。

## N1 最终定级：**Minor / 潜在**，且复审查得比我更远

复审用 `gh` 直接读了 **ComfyUI 实际内置的那份 fork**
（`Comfy-Org/ComfyUI_frontend` @ main，`src/lib/litegraph/src/LGraphNode.ts`，4621 行）：
- `:1279` `if (restored) widget.value = restored.value` —— **直接赋值**
- 整个文件里字符串 `callback` **只出现 1 次**（`:2315`），且是**真值判断，从不调用**
- `configure` 唯一的钩子是节点级 `onConfigure`，本插件未定义

它还**逐个走查了全部 59 个提到 `widget.callback` 的文件直到调用点**，
逐条列出：`setNodeWidgetValue`（调用方是遮罩编辑器/light info/合成器/画笔/图像控件适配器，
**无一条是还原路径**）、`widgetInputs.ts:82`（刷新 combo 选项并夹取）、
`widgetInputs.ts:572`（用户断开连接后重建控件）、下载模板应用（用户动作）、
重命名控件（右键菜单）、`useWatchWidget.ts:66`（**包装** callback，从不调用它）。
**用户主动触发的那些，「翻成自定义」本来就是想要的语义。**

**它主动交代的未定部分**：0.3.10 之前的单体 `web/scripts/app.js` 读不到
（v0.3.10 已是 3 行 shim；v0.2.0 树里没有 litegraph），且**用户实际跑的 ComfyUI 版本
从这里无法得知**。它据此把「遗留还原路径会触发 callback」判为**低概率**并计入。

**两级证据一致**：我查 `comfy-org/litegraph.js` @ master 得到同样的「直接赋值」，
复审查 `ComfyUI_frontend` 内置 fork 得到同样结论 —— **两个独立来源互相印证**。

**我的处置决定：不改 JS。** 理由（已告知复审）：
1. 按修订后的定级，**没有已知调用方能触发它**，改它就是凭猜测加复杂度；
2. 复审提出的「按取值比较」的改法**并不是当前写法的严格超集** ——
   它对「程序化写入了一个**不同**的值」仍会翻，而那恰恰是假想的还原类调用方会呈现的形态，
   于是只是把一种「未枚举调用方风险」换成另一种，并引入新的语义细节
   （用户手改成与档位相同的值不会翻 —— 无害，因为生效配置完全相同）。
→ 记为**可选的未来加固**，不行动。

**用户侧建议因此加了一条**：在真 ComfyUI 里做一次「存 workflow → 重开」的肉眼确认
（原本清单里的「浏览器里确认预设联动」现在多了一层理由）。

## N1 收尾：两个「门」都被复审关掉，且它**自我纠正了加重情节的依据**

**关掉我的限定 #2（ComfyUI 自己的前端）**：它 `gh api` 读了 ComfyUI 实际内置的 fork ——
`configure()` 直接赋值（`:1279`），字符串 `callback` 在 4621 行的文件里**只出现 1 次**（`:2315`），
且是**真值判断而非调用**（`if (!callback && !w.options.callback && !w.options.property)`）；
`configure` 的钩子只有属性循环里的 `onPropertyChanged` 与末尾 `onConfigure`，本插件都没定义。

**连 undo/redo 这扇门也关了**（这点我原本没想到）：
`src/scripts/changeTracker.ts:490-496` —— `undo()`/`redo()` 都走 `updateState()`，
而 `:478` 是 `await app.loadGraphData(prevState, …)`，即**重新进入同一套 graph-configure 路径**，
于是继承的是直接赋值，而不是重放带 callback 的操作。

**★ 它主动纠正了自己提出的加重情节的依据**（这条我要单独记）：
现代 `configure` 是通过**还原槽 + `widgets_values_named`**（`getWidgetRestorationSlot`，`:1270-1279`）
还原的，**不是**按原始位置数组。故它自己那句「16 个被覆盖控件在档位之后按下标还原、
逐个把档位翻走」应标注为**旧的位置模型的推理**。
它的原话态度：*「现在已无实际影响，但它不该比 N1 活得更久」*。

**修正我自己的台账**：我机读的 113 / index 24 / 之后 16 个，作为
**「声明顺序」的事实依然成立**；但由此推出「还原时按下标逐个翻走」这一步，
建立在旧位置模型上，**必须连同 N1 一起作废，不得单独存活**。
（该行号引用来自复审，我未逐行独立复核 —— 但它与我在 `comfy-org/litegraph.js`
@ master 上读到的「直接赋值」结论一致，两个独立来源不冲突。）

**仍开的唯一限定**：ComfyUI pin 的 litegraph 版本 vs `master` 的差异 ——
复审也关不掉（本机无 ComfyUI 安装；`v0.3.10` 的 `web/scripts/app.js` 已是 3 行 shim、
`v0.2.0` 树里根本没有 litegraph）。记为已知残余不确定性。

**它给出的「若真有活路径则怎么修」**（不采纳，存档）：
把翻转改成**取值感知** —— 若写入的值恰等于当前档位对该控件的取值，则不翻。
它自评这是「唯一能robust 应对我枚举不出的调用方」的选项；
同时它也承认另外两条路（挂 `onConfigure` + `_qhConfiguring` 标志、DOM/指针监听）
要么无收益要么对 canvas 绘制的控件无效。

## ★★ 整分支终审 scoped 复审判决：**Should this merge? Yes**

「All findings addressed, no new Critical/Important breakage（N1 已降为 Minor/latent）」。
它给出的理由：本轮关掉了它提的每一条；两条承重主张经它**自己的变异测试**成立
（删掉抑制行恰好失败那三个具名用例；`finally` 复位与 pending 重放各自独立承重）；
新测试代码是**变异验证过的**而非装饰性的；diff 里没有任何阻塞项。

**New Breakage**：N2/N3/N4 三条 Minor（都不阻塞）
- **N2** `js:89` 的 `if (name === WIDGET) continue;` 是**不可达的防御代码**
  （`coveredKeys()` 取自预设表键，而 `配置预设` 从不出现其中；删掉它探针仍 15/15）。
- **N3** `test_latent_algorithms.py`：`_widget_spec` 守自己的输入，
  但 `_latent_node_widget_decls` 对 `sub` 没有 `isinstance(sub, ast.Dict)` —— **不对称，像是疏忽**
  （正是我先前从 Pyright 诊断里发现的那处；复审确认并补上了「不对称」这个更好的论据）。
- **N4** `test_import_smoke.py` 的 `tearDownModule` 只 pop `_INJECTED`，
  而 `_exec(node.py)` 会传递性导入未被追踪的子模块、残留整个会话（无观察到的影响，82/82 绿）。

## ★ 复审纠正了我**对用户说过**的一句话（我复核后确认它是对的）

我说过「`git ls-files --eol` 全仓 w/lf、工作区无一处 CRLF 混入」。**错。**
- `__init__.py` 是 `i/lf  w/crlf`；全仓**共 51 个**已跟踪文件工作区是 CRLF
  （含 `README.md`、`combat_data/*.json`、`configs/*.json`）。
- **我的两个检查都坏了**：① 我把输出 `grep -v "w/crlf\|w/mixed"` **过滤掉**了 CRLF 行，
  然后误把「列出的都是 lf」读成「没有 CRLF」；② `awk '$1 ~ /w\/crlf/'` 匹配的是
  **index 列**（`i/lf`），该看的是 **worktree 列 `$2`** → 对 i/lf 文件恒不匹配 → 恒空。
- **但结论侥幸无害**：`git show HEAD:__init__.py | grep -c $'\r'` = **0**，
  `git status` 干净 —— CRLF 只是 Windows 上 autocrlf 的**正常签出形态**。
- **教训**：「我用两个独立检查确认了 X」这句话，在检查本身有 bug 时是**双重虚假的安心**。
  过滤式与半列式的检查都必须先证明「它红得起来」。

## ★ 复审推翻了我的一个**理由**（我的决定不变，但它是对的）

我说「取值感知改法只是把一种未枚举调用方风险换成另一种」——**不准确**。
它逐格枚举：对「写入**不同**值的调用方」两种守卫**都会翻** → 那是**平手**；
只有「写入与该档**相同**的值」那两格，取值感知才更优或中性。
→ 正确表述是「**它防住严格更大的调用方集合，只是边缘那部分现在是空的**」。
我的决定（不改 JS）仍成立，但**理由要改**成它给的那句：
> 严格更强，但已无动机 —— 只在将来真的发现带 callback 的还原路径时才回头做。

它同时**自我纠正**了更早的过度断言（称该改法「彻底移除对前端内部的依赖」—— 多算了一格）。

## 用户裁决（第二批）
1. **把 Rec 5/6 + I4 也做掉**
2. 工作区：**台账归档进 `docs/`，其余删掉**

## 已派发 fix-a-recs (opus)
- **Item 1 = Rec 5**：给 `YoloConfig` 与 `YoloRunContext` 加 `@dataclass(slots=True)`。
  派发词里点名了三个必须先查的连带风险：`vars()`/`__dict__` 依赖
  （已知 `test_missing_field_does_not_raise` 里有 `set(vars(cfg))`，会当场 TypeError）、
  `fields()/replace()/asdict()`、pickle/deepcopy。并要求做**对照演示**：
  错写属性名现在立刻报错、而普通 dataclass 上静默成功。
- **Item 2 = Rec 6**：把「行为保持」的对拍固化成**可重跑的脚本 + 提交的报告**，
  放在 `docs/superpowers/` 下。要求照复审的数字复现（105 函数 / 缺 0 / 109 逐字节相同 /
  3 个合理不同；902 条叶子语句 / 901 命中 / 唯一差异是版本横幅），
  并明确**不许为了凑数字而放宽归一化**，数字对不上就如实报差异。
- **Item 3 = I4**：**用兜底而非上提** —— 保持惰性导入，但加
  `except ImportError` → `apply_preset = None`，使「presets 坏掉」降级为
  「预设不生效」而不是「每次运行抛 ImportError」。理由写进派发词：
  这同时满足 `da27cc3` 的「节点不能消失」与消除 per-run 崩溃，
  且失败在启动期仍然响亮（根 `__init__.py` 的 8c 块现在会 print ❌）。

---

## fix-a-recs 完成报告（DONE_WITH_CONCERNS，未提交）

改动三处：`context.py`（两个 dataclass 加 `slots=True`）、
`prepare.py`（惰性 import 加兜底）、`test_latent_algorithms.py`（断言 3 换写法）。
另新增 `docs/superpowers/verification/`（可重跑脚本 + 生成的对拍报告）。

### 它提出三条 concern —— 控制器逐条独立复核

**Concern 1：它没听我的 `except ImportError`，用了 `except Exception`。**
控制器实测：`issubclass(SyntaxError, ImportError)` = **False**；
import 一个语法错误的模块抛的是 `SyntaxError`，`except ImportError` **接不住**。
→ **它的偏离是对的，我的派发词是错的。** 它给的理由正是我点名的那一类坏法。

**Concern 2：`slots=True` 要 Python ≥ 3.10，而全仓没有 `requires-python`。**
控制器实测本机 **Python 3.14.2**，`slots=True` 行为正确
（写未声明属性 → `AttributeError`，`hasattr(c,'__dict__')` = False）。
`requirements.txt` 只有 `reedsolo` 一行。这是本仓**第一个 ≥3.10 语法构造**
（38 个 nodes/*.py 都能在 3.9 下 parse）。→ 交 scoped 复审裁决是否值得引入版本下限。

**Concern 3：复审所报 902 条叶子语句 / 901 命中，它复现不出来，且拒绝凑数字。**
控制器独立写了第 10 个口径探针（`qh_probe_stmt_counts.py`），源 process() @4131-5606
（1476 行，同名 2 个取最大者，与双方一致），九种自然口径全算：

| 口径 | 条数 |
|---|---|
| 全部 ast.stmt | 886 |
| **叶子 ast.stmt** | **674** |
| 叶子 + ExceptHandler | 711 |
| 全部 + ExceptHandler | 923 |
| 唯一行号的全部 / 叶子 | 881 / 669 |
| 总行 / 非空行 / 非空非注释行 | 1476 / 1289 / 1074 |

**没有任何自然口径给出 902 或 901。**
且我的 886/923 与它的 887/924 差 1 —— 差的就是 `def process` 自身那一行
（它的 walk 计入函数定义语句），**两套数字完全对得上**。
→ 裁决：**674 是唯一站得住的口径；902/901 不可复现，作废。**
数字的权威归到**已提交、可重跑**的脚本，不再引用复审那次的口算。

### 它另外自查出的两处（复审未计入）

1. `hasattr(node,…)` vs `hasattr(self,…)`：三种获批改写只覆盖 `node.x` 形态，
   **不覆盖以值出现的裸 `node`** → 这是第 4 处合理差异，属口径边界非行为差异。
   控制器确认报告 §3 已如实列出。
2. `配置预设` **控件在源文件里本就有**（源 `nodes.py:1372-1373`「V6.6 配置预设」），
   新版只补了后端 → 所以 `INPUT_TYPES` 逐字节相同**不是漏改**。
   控制器复核报告 §3 末段已写明。

### 控制器另外独立做的回归面检查

- 全仓 `grep vars( / __dict__ / asdict / dataclasses.replace / deepcopy`（排除测试文件）
  → **零命中**。`slots=True` 无 `__dict__` 依赖面，无回归。
- 两个类的消费者：只有 pipeline 的 5 个阶段模块 + 测试，全部按属性访问。
- `context.py:15` 的 `Dict` 已不再使用（Pyright 提示）—— **diff 未动 import 段，
  是既有问题**，非本批引入 → 记为 Minor 交终审分诊。

### 控制器独立复现兜底行为（不采信带 bug 的探针）

派发报告里那份 `probe_presets_degrade.py` 被 Pyright 标出
`"saved" is not defined`（未定义变量），其证据可信度存疑，
故控制器重写独立探针 `qh_probe_presets_import.py`，用 `sys.meta_path`
只对 `nodes.qianhui_latent_yolo.presets` 抛 **SyntaxError**（不是 ImportError）：

```
CONTROL（预设应生效）：不抛异常 ✓  region_steps == 12 ✓
BROKEN （presets 抛 SyntaxError）：不抛异常 ✓  region_steps == 0 ✓（降级不崩）
```

**两个方向都跑，是因为只有 CONTROL 变绿才能证明 BROKEN 的绿不是空的**
—— 一个恒为空的 `apply_preset` 也能让 BROKEN 变绿。

### 测试证据（控制器实跑）

`Ran 82 tests OK` / `test_latent_algorithms.py: Ran 23 OK` / `test_import_smoke.py: Ran 9 OK`

### ★ 控制器推翻 concern 2 的风险定性（新增，在复审进行中查到的）

fix-a-recs 说「`slots=True` 要 ≥3.10，而全仓没有 `requires-python`，请确认运行时」。
但它**漏看了计划头部那一行**。控制器实测：

- `docs/superpowers/plans/2026-10-09-watermark-integration-and-split.md:9`
  → `**Tech Stack:** Python 3.14、ComfyUI 节点 API、…`
- `docs/superpowers/plans/2026-10-09-node-widget-cleanup.md:9`
  → `**Tech Stack:** Python 3.14、unittest、…`（Plan B 同样声明）

→ **≥3.10 是本项目既有的、已书面的约束，不是本批新引入的。**
「没有 requires-python」是真的，但那是**打包元数据**的缺口；
语言版本由计划 Tech Stack 行声明，两处都写着 3.14。
本机实测解释器 3.14.2，与声明一致。
→ 风险定性从「未知的部署下限」降为「已声明约束 + 一处可补的元数据缺口」。

它的 `ast.parse(src, feature_version=(3,9))` 复检也只证明**语法**可解析
（`@dataclass(slots=True)` 在 3.9 下语法合法），不证明运行时可支持 ——
它自己的表述是「3.9-parseable」，用词准确，不算过度断言。

**动作**：把这条事实补发给正在跑的复审（我的派发词只抄了 Global Constraints 段，
没抄计划头部的 Tech Stack 行，属我派发词的遗漏，非诱导结论）。
**不改代码**：复审正在看 139d386，此刻动 `context.py` 会让它的复审作废。
若它把版本下限判成问题，再在修复轮里把「需要 ≥3.10（计划声明 3.14）」写进注释。

### ★ 复审抓到一处**我漏掉**的动态写点 —— 控制器复核后给出精确措辞

复审 caveat：报告 §1.3 称「不存在靠动态写新属性的用法」**按字面为假**，
因为 `presets.py:112` 有 `setattr(cfg, field_name, value)`，
而那份 AST 扫描只匹配 `cfg.<x> =` **文本形态**，结构上看不见 setattr。

**控制器复核（我自己漏了这个模式）**：

1. **我错在哪**：我给用户报的 grep 是
   `vars( / __dict__ / asdict / dataclasses.replace / deepcopy` —— **没搜 `setattr`**。
   所以「零命中 → 无回归面」这个推理**不完整**。补搜后确有动态写点。
2. **全部动态写点（`grep -rn "setattr(\|delattr("`）**：
   - `presets.py:112` `setattr(cfg, field_name, value)` ← **唯一**作用在 slots 类上的
   - `detection.py:593/615/619` → 作用在 `model`，不是我们的 dataclass
   - `sex_pose_node.py:324-338` → 作用在节点 `self`，另一个类
3. **它安全，且失败模式变响不变哑**：`apply_preset` 先 `FIELD_OF.get(key)`，
   为 None 就 `raise KeyError`（带「新增预设项时必须同时补 FIELD_OF」的行动指引），
   **根本走不到 setattr**；且 `test_every_preset_key_maps_to_a_real_config_field`
    **双向**钉死（正：每个预设键都在 FIELD_OF 且指向真字段；反：FIELD_OF 每个值都是真字段）。
4. **措辞裁决（我不采纳复审「按字面为假」的强度）**：
   - **结论为真**：确实不存在「写**新**属性」——字段名被 FIELD_OF + KeyError + 测试三重夹住。
   - **证据不足**：所引的那份 AST 扫描**只能**证明 `cfg.<x> =` 形态，
     结构上无法证明「不存在动态写」。即**真结论 + 不够格的证据**。
   - 正确表述：§1.3 的**覆盖声明**夸大了（不是结论错）。
5. **该文本不在已提交产物里**：`grep -rn "动态\|新属性\|setattr\|__dict__" ` 扫
   `verify_split_behaviour_preservation.py` 与 `behaviour-preservation-report.txt`
   → **零命中**。它是 `recs-report.md:88`（**未提交**、按用户裁决将删）。
   → 唯一已提交的夸大是我的 commit 里那 4 个字「无回归面」。
6. **动作**：派发词里我给复审的 question C 前提**是错的**（我假定这句由脚本 emit），
   已发更正，避免它去脚本里白找。severity 属 **Minor（文档措辞，非行为）**，
   按 skill「Minor 不进修复环」→ 记入台账交终审分诊。

## ★★ scoped 复审判决 —— 三项全 ADDRESSED，三条 concern 全 UPHELD

| 项 | 判决 | 证据 |
|---|---|---|
| Item 1（Rec 5 slots） | **ADDRESSED** | `context.py:20-21` / `:166-167`；理由注释 `:7-12` |
| Item 2（Rec 6 对拍） | **ADDRESSED** | 脚本 580 行 + 报告 127 行均已提交 |
| Item 3（I4 兜底） | **ADDRESSED** | `prepare.py:164-174` try **只包 import**，调用在 `:178` 之外 |

### 三条 concern：全部 UPHELD，且**我的裁决三条全 CORRECT**

1. **SyntaxError 不是 ImportError 子类** → UPHELD，我 CORRECT。
   复审补：探针 case B 实证 import 期 SyntaxError 会穿透修复前的裸惰性导入。
2. **≥3.10 下限** → UPHELD，我 CORRECT，**且比它第一轮承认的更充分**。
   ★ **复审主动撤回自己第一轮的框架**：它曾写「repo 里没记录这条约束」——
   错。它说「我因为派发词只给了 constraints 段，就把计划文档当参考性的，
   应该去读头部而不是下结论说约束未被记录。行号我引对了，
   但把它们定性为不充分是错的。**撤回**。」
   → 下限从「未记录」改为「治理本批的计划已声明」，降级为 **Minor：打包元数据缺口**
   （`requirements.txt` 只有 reedsolo、README 未提 Python 版本、无 `requires-python`）
   —— 只影响第三方往旧 ComfyUI 装，不影响本机 3.14.2 的本地扩展。
   → 它另测：全仓 **49/49** 个 `.py` 在 `feature_version=(3,9)` 下可解析；
   `slots=True` 作为**代码**只出现在 `context.py:20` 与 `:166` → ≥3.10 需求**恰两行**。
3. **902 不可复现** → UPHELD，我 CORRECT。

### 902 终局：双方独立收敛到同一个数

复审独立算：**叶 674** / 配对 672（定义：非 def 自身、非嵌套 def、非 docstring Expr、
且 `body`/`orelse`/`finalbody`/`handlers` 里**没有子 ast.stmt** 的 stmt）。
它给的其它口径与我**逐格相同**：全部 887含def / 886不含 · 叶 674 · 叶+handler 711 ·
全部+handler 924 / 923 · 唯一行全 881 · 唯一行叶 669 · 总行 1476 · 非空 1289 ·
非空非注释 1074 · tokenize NEWLINE 952。
**「跨约 25 种自然定义，我造不出 901 或 902。」**
→ 与我先前的九口径结论一致：**902/901 作废，674 是唯一站得住的口径。**
（它另记一处小漂移：实现者 §2.4 表的非空 1286 / 非空非注释 1072 vs 它的 1289/1074
 —— 计数方法微差，但**任一方向都不靠近 902**。）

### ★ 复审**采纳了我的措辞裁决**并撤回自己的过强表述

我先前指出它「按字面为假」太强。它回：
> 「你的读法对，我的在结论上过头了。精确切分：
> **结论为真**（`presets.py:112` 是动态写，但从不写**新**属性）；
> 有缺陷的是**证据的覆盖声明** —— 只匹配 `cfg.<x> =` 文本形态的 AST 扫描
> 无法证明「不存在基于 setattr 的写」。
> 『按字面为假』是错的；『真结论、夸大覆盖』是对的。」
它另补一个更准的论据：**歧义本身就是危害** ——
读作「没有用法动态造属性」为真；读作「扫描证明了不存在动态写路径」为假。
**一个扫描范围内的结果被当作全仓结论呈现** → 这才是它读成 overclaim 的原因。
→ 它定位到确切的 `recs-report.md:94`。**无需动作，结论成立。**

### 它也复核了我的动态写点枚举（并承认自己先前被 `head` 截断）

`grep -rn --include=*.py -E "setattr\(|delattr\(|__setattr__"` → 恰三处：
`presets.py:112`（作用 cfg）、`detection.py:593/615/619`（作用 model）、
`sex_pose_node.py:324-326` setattr + `:332/335/338` delattr（作用节点 self）。
**零个 `__setattr__` 覆写。** → `presets.py:112` 是唯一触及 cfg/ctx 的，
无其它动态写路径能到达 slots 类。

### Item 3 复审细节（补全）

- `NO_PRESET = None`（`:174`）在兜底路径上**确实不被读**：此时 `apply_preset is None`，
  `:176` 的守卫为假，`:178` 永不执行；`grep -n NO_PRESET` → `:165, :171(注释), :174, :178`，
  模块内唯一读者就是 `:178`。
- 它自己跑了探针：control `region_steps == 12`；A（`sys.modules[...] = None`）→ 0；
  B（语法错误）→ 0；**C（还原修复前代码）→ `SyntaxError: invalid syntax (presets.py, line 1)`**
  —— C 是关键对照：证明修复前的代码在语法错误下**真的会抛**。
  跑完 `git status --porcelain` 为空。
- `tests/test_latent_algorithms.py:503-532` 在新形态下仍成立。

### ★ 控制器对 New Breakage 那一条的更正（复审的假设不可达）

复审报 Minor：`verify_split_behaviour_preservation.py:567-575` 的 `_flush`
把「写报告」和 `os.path.relpath(REPORT_FILE, REPO_ROOT)` 包在同一个 try 里，
跨盘 `relpath` 会…（被截断）。**控制器读码更正**：

- `REPORT_FILE`（`:74` = `_HERE/behaviour-preservation-report.txt`）与
  `REPO_ROOT`（`:72` = `_HERE/../../..`）**同源** → **恒同盘**，
  跨盘 `ValueError` **结构上不可达**。→ 该假设应标注为不可达，而非活失败模式。
- 顺序：`with io.open(...)` 的写在 `:568-569` **先完成**，
  `relpath` 只在 `:571` 的 `print` 实参里求值。**即使 relpath 真炸，报告已落盘**，
  但 `:573` 的 handler 会打「[报告写入失败]」→ **把成功误报成失败**。
- **真正的缺陷**：写**真的**失败时（目录只读 / 磁盘满），`except` 打完字
  仍然 `sys.exit(exit_code)`，而 `exit_code` 只由「有没有未登记差异」决定
  → **写失败照样退出码 0**。本脚本的全部意义就是产出那份报告；
  产出不了却返回成功 = **静默失败**，且正是本项目一路在猎的失败形状。
- 已要求复审明确：该条 Minor 是记在 relpath 不可达上、还是记在退出码上，并给出改点行号。
  **控制器倾向修这一条**（本批新增脚本、2 行、非节点代码），
  修完须复验仍能逐字节复现（md5 `57022f1e3d3183ae2a9f72cf452cd2ae`）。

### New Breakage 全表（复审，已收全）

| # | severity | file:line | 内容 |
|---|---|---|---|
| 1 | Minor | `verify_...py:569-576` | `_flush` 的 try 同时包写入与 `relpath` 显示行 |
| 2 | Minor | 同文件 `:116` | `_OUT` 模块级全局从不复位 → 第二次 in-process `main()` 重复追加 |
| 3 | Minor（产物准确性） | `behaviour-preservation-report.txt:77` | 报告写 `\bnode\.` 新 **57**，而 `recs-report.md:173` 叙述 **56** —— **产物是对的** |
| 4 | Minor（产物准确性） | `behaviour-preservation-report.txt:74` | 标签写「整个新包 / 整个源文件」，实际只统计**函数体** |

**复审明确否定了我的一半判断**（我接受）：我问「跨盘 relpath 是不是会让报告静默写不出来」，
它答：**「报告仍然会被写出 —— 只有显示路径降级。它没有『静默什么都不产出』的形状。」**
它实测：把 `REPORT_FILE` 重指到另一盘，重生成的文件与已提交报告**逐字节相同**
（md5 均 `57022f1e3d3183ae2a9f72cf452cd2ae`），且退出码 0。
→ 我先前对用户说的「写不出来」只适用于**真·写失败**那一支，不适用于 relpath 那支。
它同时**同意**真·写失败那一支「是真实的自动化静默失败形状」——那条才是要修的。

**Minor 3 无需动作**：产物对（57 正确），错的 56 在 `recs-report.md:173`（未提交、将删）。

### 控制器应用修复（提交 6c53ecb，仅动本批新增脚本，未碰节点代码）

| 复审 # | 改法 | 我自己的实测 |
|---|---|---|
| 1 真·写失败 | 写入失败强制 `sys.exit(1)`，不再用 exit_code | `REPORT_FILE` 指向不存在目录 → **退出码 1**（修复前 0） |
| 1 状态行 | 把显示行拆出独立 try，relpath 失败降级为绝对路径 | 指向另一盘 → 报告**确实写出**、**未被误报**、退出码 0 |
| 4 标签 | emit + 注释改为「两侧**提取出的函数体内**，非整个文件」 | 重生成后与旧版**只差这一行**，所有数字不变 |
| 2 `_OUT` | `main()` 入口 `del _OUT[:]` | 连跑两次 → 标题出现 **1** 次（修复前 2 次） |

**关键回归证据**：重生成的报告与上一版 **diff 恰为 1 行**（那行标签），
`382/661/57`、`674/672` 等**全部数字逐格不变** —— 证明我的改动没有触动任何测量。
脚本重跑两次逐字节相同；`Ran 82 tests ... OK`（退出码 0）。

已请复审对 6c53ecb 做 scoped 复核，并再次索要 Out-of-Scope 余下部分 + Verdict 行。

## ★ 复审 Verdict（已收全）+ 三处**它纠正了我**

**Verdict 原文**：
> **Batch: All items addressed, no new Critical/Important breakage.**
> Non-blocking remainders: the two artifact-wording defects
> (`behaviour-preservation-report.txt:74` and `:77`), the two scratch nitpicks in
> `verify_split_behaviour_preservation.py` (`:116`, `:574-576`),
> the `recs-report.md:94` coverage overclaim whose conclusion is nonetheless sound,
> and the packaging-metadata gap for Python ≥3.10.

### ★★★ 纠正 1：我对用户说过一句**假话**，必须更正

我说：「复审报告里那份 `probe_presets_degrade.py` 被 Pyright 标出
`saved` is not defined（未定义变量），**其证据可信度存疑**」。

**复审驳回，控制器实测确认它是对的**：
- `grep -n "saved" probe_presets_degrade.py` → **只有** `saved_nodes`（`:110` 定义、`:152` 使用），
  **没有任何裸 `saved`**。
- Pyright 曾指的 `:122` 实际是 `try:` / `exec_module("nodes.qianhui_latent_yolo.node", ...)` 一行。
- 我**自己跑了那个探针**：退出码 0，两个方向都跑通，
  且 case C（还原修复前代码）复现 `SyntaxError: invalid syntax (presets.py, line 1)`。
- 文件 mtime = `2026-10-09 20:33:05` —— 我的印象来自**更早的快照**。

→ **教训（与前一次「投递内容 ≠ 当前状态」同型，我又犯了一次）**：
我拿一个**过期**的编辑器诊断去质疑一份被实现者仍在改写的文件。
**它的探针一直是好的**；我不该污蔑它。
我的**独立**探针结论仍成立（降级确实发生），但那是**另取的证据**，
不能用来推出「它的探针坏了」。

### ★ 纠正 2：改写的断言 3 **不是**静默弱化（复审给了变异证据）

复审把 `tests/test_latent_algorithms.py:471-479` 的断言在**去 slots 的临时副本**上重放：
`assertFalse(hasattr(cfg, "__dict__"))` **失败**、`assertRaises(AttributeError)` **失败**。
断言 1/2（`:467-470` 不抛异常 / `region_steps == 12`）未变，
仍通过「不抛异常」那条路径带着原有的动态覆盖。
→ 即「去掉 slots 测试就会红」——**这正是我想要的变异检查**，复审代我做了。

### 纠正 3（无害）：`recs-report.md:5` 自称「未 git add、未 commit」

属实不符（改动已成 `139d386`）。原因是我在报告写完之后才提交的，无害。

### 复审对我 _flush 更正的**确认**

它写明同意我的两点，并明确：
- `relpath` 那条标为 **latent / unreachable by construction**，不作为活失败模式；
- **carrying the severity 的是「写失败仍退 0」**，判 **Minor**（不是 Important），
  因为「错误文本确实到达 stdout，所以只对自动化静默；且触发条件是 I/O 故障
  而非可达逻辑路径」。
→ 与我的修复取向一致（两条我都已修，`6c53ecb`）。

### 其余 Out-of-Scope（非阻塞，不进修复环）

- `≥3.10` 下限：计划已声明，但 `requirements.txt` / README / 无 `requires-python`
  → 第三方装到旧 ComfyUI 不会得到警告；本地部署不受影响。
- `probe_presets_degrade.py` 是仓内未跟踪的 scratch（`.gitignore:4` 忽略 `.superpowers/`），在 diff 之外。
