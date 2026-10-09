# SDD ledger — plan: docs/superpowers/plans/2026-10-09-node-widget-cleanup.md

Task 1: complete (commit 50f8a56) — 67 个键名重命名，137 处引用（character 68 / pose 30 / scene 22 / combat 17）
Task 1: controller note: 3/3 用例通过（test_all_keys_have_no_prefix / test_source_has_no_old_key_leftovers / test_key_count_unchanged 34/14/11/8）。全量 41/41 通过。
Task 1: 审查 clean (rev-b-t1) — Spec ✅ / Task quality Approved / 0 Critical / 2 Important / 3 Minor
Task 1: 审查器核对了 70 个 kwargs 读点与新键名逐一相等、无拼错；旧名在 4 个源文件中零残留；
  无旁枝误伤（JSON 文件名/返回值/分类/选项文本未动）；键序未变。
Task 1: review-important #1 —— 残留检查只匹配双引号，窄于 docstring 承诺。
  控制者内联修复（原实现者 impl-b-t1 已死于 402，无法续跑；改动仅 1 行谓词）→ commit f5b5e3f
Task 1: review-important #3 —— 加固后暴露 scene_node.py 4 处旧名文档残留（:6/:9/:11/:376），已清 → 同上
Task 1: review-important #2 —— 默认 gbk 控制台跑测试会 UnicodeEncodeError。
  判定为「计划层已解决」：两计划所有测试命令均带 -X utf8（66ed939），审查者读的是旧简报。
  character_node.py:61 的 import 期 ⏳ 打印先于本 diff 存在，非本任务引入。已重生成简报。
Task 1: minor (deferred): 测试从未真正调用生成函数（default_kwargs/gen_func 是死代码），
  「读点被改成另一个新键名」这类拼错三连测试都发现不了。Task 2 会复用这两个辅助函数。
Task 1: minor (deferred): tests/test_node_widgets.py 自建 load_node，与 tests/_loader.py 的 load_module 重复，
  仓库里两套装载器。属简报逐字复制所致。
Task 2: 进行中 — impl-b-t2 (sonnet)。brief=task-2-brief.md (316 行)。BASE=e4199f6
Task 2: controller note (commit e4199f6)：计划两处错已修
  (a) 控件数是 9 不是 8 —— 机读 4 个节点的 INPUT_TYPES 逐个确认 9 个圈选控件都存在；
      DELETED 字典本身对，只有散文数字错（Goal/Architecture/标题/期望/核对/commit 共 6 处）
  (b) 漏步 —— Task 1 的 test_key_count_unchanged 钉的是删前的 34/14/11/8，
      Task 2 删完必失败，而计划没写要更新它。已补：改为 31/12/9/6
Task 2: controller note：核对清单「其余控件顺序不变」不准 —— scene 的氛围句会上移 2 位，
  属预期（用户已确认工作流重新生成）
Task 2: 派发时告知实现者：不要跑全量测试（A 路正在写 nodes/qianhui_latent_yolo/，
  其 test_latent_algorithms.py 可能半成品），只跑 tests/test_node_widgets.py
Task 2: 未开始 — brief 已预生成（task-2-brief.md），待派发
Task 3: 进行中 — impl-b-t3 (sonnet)。brief=task-3-brief.md (53 行)。BASE=a65a752
Task 3: 与 A-T6 同改 README.md，两者必须串行；本任务与 A-T4 并行（A-T4 只动
  nodes/qianhui_latent_yolo/，零交集）。派发时明确：只许改 README.md，
  且不要把 README 里其他已过时的说法顺手改掉——A-T6 之后还要动同一个文件。
Task 3: complete (commit 1b5259d)
Task 3: controller note：派发词里我写坏了一段 character 的控件说明（自相矛盾），
  已立刻发更正规的清单纠正：存活 31/12/9/6，末键分别为 特殊饰品 / 表情细节 /
  氛围句 / 📸战斗运镜；并列出 9 个已删控件，提醒 画面画质风格 与 氛围句 是**保留**的。
Task 3: ★ 实现者顶回了我的指令，且它是对的（已撤回并记档）
  我在更正末尾写「9 个已删控件不要在 README 里出现」，但简报步骤 2 逐字要求的
  **迁移说明**内容正是这 9 个名字 —— 照我字面执行就要删掉简报要求的东西。
  实现者未照做，改为查证 9 个名字在 README 的**全部**出现位置并请我裁决。
  控制者逐条核实全部属实：92-93 行明确列为已删；22 行「手动补充」属剧情编排机
  （story_node.py 仍有）；27 行「模型类型」属姿势大师（sex_pose_node.py 仍有）；
  「补充指令」README 中不出现。→ 撤回那句话，保留简报原文。
Task 3: 另授权修 README:25 —— 同一子句既说「原权重滑杆已移除」又让用户设
  「anima 用户权重 1.0」，自相矛盾。删除 `anima 用户权重 1.0 + `。
  实现者先前只上报未擅改（超简报授权片段），判断正确。
Task 3: 角色定制行未改：该行未提及被删的三控件，简报替换条件不成立（正确判断）。

并行约束：本路与 A 路共用同一 checkout。本路只改 nodes/{character,pose,scene,combat}_node.py 与 tests/test_node_widgets.py，与 A 路零交集。

## Controller: 用户两项要求的终态独立核验（按真实 INPUT_TYPES 机读）

不采信测试、不采信实现者自述，直接装载 4 个节点模块读 `INPUT_TYPES()`：

| 节点 | 控件数 | 带数字前缀 | 9 个已圈选控件残留 | 末 3 键 |
|---|---|---|---|---|
| 角色定制 QianhuiCharacterCreator | 31 | 无 | 无 | 颈饰 / 手部饰品 / 特殊饰品 |
| 肢体动作师 QianhuiActionMaster | 12 | 无 | 无 | 🦵左脚动作 / 🦵右脚动作 / 表情细节 |
| 场景光效大师 QianhuiSceneMaster | 9 | 无 | 无 | 画面画质风格 / 全局衣物质感 / 氛围句 |
| 战斗指挥官 QianhuiCombatMaster | 6 | 无 | 无 | 🩸战损程度 / ✨技能特效 / 📸战斗运镜 |

**用户两项要求（删掉全部数字前缀 + 删掉圈选的 9 个控件）= 全部满足。**
31/12/9/6 与 `test_key_count_unchanged` 钉的值一致；末键与我先前发给 B-T3 的清单一致。

## Task 2: 审查完成 (rev-b-t2) — Spec ✅ / Task quality **Approved**

0 Critical / 0 Important / **6 Minor** / 2 ⚠️。控制者已把 2 个 ⚠️ 各自裁定：

⚠️-a「步骤 8 全量测试从未有人执行（预期 32）」→ **已由控制者闭环**：
  在 HEAD 实测 `python -X utf8 -m unittest discover -s tests -p "test_*.py"` = **Ran 61 tests OK**。
  61 含计划 A 的用例；计划 B 自身的 tests/test_node_widgets.py = **12/12 OK**（单独实测）。
  「预期 32」是简报里更早期的陈旧数字。
⚠️-b「步骤 9 真实 ComfyUI 界面核对无人在运行中的 UI 里目视确认」→ **真实且未闭环**，
  属用户动作：需重启 ComfyUI 目视 9 个控件消失、scene「氛围句」上移 2 位。
  另：rev-b-t2 已用静态方式证明位置变化（按 INPUT_TYPES 顺序推断），且控制者机读
  确认 31/12/9/6 与无数字前缀 —— 只剩「肉眼确认」这一层。

**6 条 Minor（全部转入最终审查三角分级，不进修复轮）**：
1. `nodes/scene_node.py:15` docstring 仍写「加权输出 STRING」，scene 已恒输出纯标签
2. `nodes/character_node.py:5` docstring 仍列「构图」维度，`构图范围` 本任务已删
3. `nodes/pose_node.py:297` 报告 §三.3 称保留 `# 核心动作加权`，实际提交为 `# 核心动作` —— **报告过期，代码无误**
4. `tests/test_node_widgets.py:149-155`、`:114-122` 测试抓不到残留读点
5. `nodes/combat_node.py:165-166` 该 `is_anima` 三元式在 `is_random` 分支内，默认 kwargs 走不到；审查者用 60 次随机跑验证 0 权重
6. `data/composition.json` 已无代码引用但文件仍在；简报只要求删定义与加载，属范围内正确

**控制者对 Minor #4 的初步裁定（留待最终审查复核）**：疑为**误报**。
`test_no_bare_deleted_widget_name_left`（该文件 149-168 行）正是为这条风险写的，
其谓词是裸 `n in src`，`DELETED["character"]` 含 `"手动补充"` ——
把 `kwargs.get("手动补充", "")` 留在源码里该用例**必失败**。
审查者可能指的是另一层限制（读点被改成**另一个新键名**时三连测试都发现不了），
那一条 B-T1 早已 parked。两条不冲突，故并存记录、不驳回。

## ★ 控制者勘误：我先前对 Minor #4 的「疑为误报」裁定是**错的**，现撤回

我在上文写「控制者对 Minor #4 的初步裁定：疑为误报」，并给了一串看似成立的推理
（`test_no_bare_deleted_widget_name_left` 的谓词是裸 `n in src` …）。
**那个推理用了错的基线**：我读的是**当前 HEAD** 的测试文件，却把该用例
当成了 B-T2 那个 commit（01e79ea）的产物。

实测（`git show 01e79ea:tests/test_node_widgets.py`）：
- 01e79ea 版 **11** 个 `def test_`，**没有** `no_bare_deleted` 这个守卫（`grep -c` = 0）。
- HEAD 版 **12** 个，多出的正是 `test_no_bare_deleted_widget_name_left`。
- 该守卫由 **cb2883f**（我自己的审查修复提交，ledger 里写作「一处守卫盲区」）引入。

所以：**rev-b-t2 的 Minor #4 在它审查的那个 commit 上完全成立**。
独立复核者 rev-b-t2b 还给出了**可复现的失败场景**：往 01e79ea 的 `scene_node.py` 塞回
`manual = kwargs.get("手动补充","").strip()` + `if manual: parts.append(manual)`，
提交的那份测试仍报 `Ran 11 tests OK`；换成 HEAD 的测试文件则
`AssertionError: 残留已删控件的裸名: ['手动补充']`。

**教训（写给最终审查，也写给我自己）**：
审查**某个 commit** 的取证，必须用**那个 commit 的树**（`git show <sha>:<path>`），
不能用工作区/HEAD 的文件 —— 否则会拿「后来者补的守卫」去否定「当时确实存在的缺口」，
把一条真 finding 判成误报。这与本会话反复出现的同一类错误同源：
**在错误的基线上做正确的推理**。

## Plan B 的 Minor 现状（据 rev-b-t2b 独立复核后重排）

| # | 内容 | 现状 |
|---|---|---|
| 1 | `scene_node.py:15` docstring「加权输出 STRING」 | **cb2883f 已修** |
| 2 | `character_node.py:5` docstring 仍列「构图」维度 | **cb2883f 已修** |
| 3 | pose_node.py:297 `# 核心动作加权` vs `# 核心动作` | **报告过期，代码无误**；另发现 .superpowers 下独立简报文件仍留该旧文本（scratch，不入库） |
| 4 | 测试抓不到漏删的裸名读点 | **cb2883f 已修**（守卫已补，rev-b-t2b 两向实测） |
| 5 | `combat_node.py:165-166` 三元式在 `is_random` 分支内，默认 kwargs 走不到 | 待终审裁决 |
| 6 | `data/composition.json` 成孤儿文件 | 待终审裁决 |
| 7（新） | 01e79ea 多改了计划文档 8 行，越出「只许改 6 个文件」 | **落笔人是我**（控制者提交时顺手订正了一处会踩残留守卫的措辞）。如实记账，非实现者越权。 |

## ★ 计划 B 最终整分支审查 (final-rev-b, opus): **结论 = 可以合并**

审查者先核对审查对象本身：我生成的 `final-review-B.diff` 与我另行重生成的
`git diff 3e4aa03 1b5259d -- <6 文件>` **逐字节相同**，378/202 属实，路径限定未混入计划 A。
并声明它审查的 6 个文件相对 1b5259d **零改动**（当时工作树正被计划 A 改脏，它作了区分）。

**方法**：在隔离目录里用**真实数据**重建 `old`(3e4aa03) 与 `new`(HEAD) 两棵 4 节点树实跑，不靠肉眼。

**键集合**
- 键数 34→31 / 14→12 / 11→9 / 8→6（67→58）
- 改名保真：`去前缀(old 键) == new 键`，四文件全 True；反向差集恰好 9 项且与 `DELETED` 逐字相同
- `OLD_KEYS` 完整性：测试里 67 项与 3e4aa03 实际 `INPUT_TYPES` 逐字相等，`MISSING=[]`、`NOT_REAL=[]`

**引用自洽**（AST 扫每个 `kwargs[...]` / `kwargs.get(...)`）
- `declared == read` 且 1:1：character 31/31、pose 12/12、scene 9/9、combat 6/6
- `declared-but-never-read = []`（没有「控件还在但没人读」）
- `READ-BUT-NOT-DECLARED = []`（没有指向已删/已改名键的读点）
- 带前缀字面量命中的是 `"01_global.json"` 这类**数据文件名**，非控件键（已人工确认为假阳性）

**输出侧**（同语义输入，old 的 9 个待删控件置为不指定/默认）
- character：old == new **逐字节相同**（795B）→ 删的 3 个控件确实零贡献
- pose：old 去权重 + 下划线转空格 == new（390→378B）
- scene：old 去掉最外层 `(…:1.1)` 包裹 == new（527→521B）
- combat：old 去权重 == new（282→264B）

**结论：三次任务之间没有互相推翻**，行为差异恰好只有计划声明的那几项。
跨文件面亦干净（`js/role_random.js` 等均不引用旧键；全仓无 `COMPOSITION_DATA` 残留）。

## ★ 关于 Minor #4「谁对」的最终调和 —— 三个复核者其实都对，差别只在**基线**

- rev-b-t2 审 `01e79ea`（11 个用例，**无**裸名守卫）→ 说「抓不到」→ **在它的基线上成立**
- rev-b-t2b 在 `01e79ea` 上注入读点复现 `Ran 11 tests OK` 的逃逸 → **证实上一条**
- final-rev-b 测 HEAD，六种变体注入，四种 CAUGHT、两种 ESCAPED → 说「不成立」→ **在 HEAD 上成立**

调和：**该 finding 在被提出的那个 commit 上完全成立**，守卫由 `cb2883f` 补上。
我先前的勘误（撤回「疑为误报」）**保持有效**。
final-rev-b 另给了原审查者可能的误判路径：它引用的 `:149-155` 恰是
`test_no_bare_deleted_widget_name_left` 的 **docstring 正文**（那段为了论证新用例必要性，
明写了「这两条测试都抓不到裸名」）—— 把**对盲区的描述**读成了「没有守卫」。

**逃逸面（记录，非现实回归形态）**：`kwargs.get("手动" + "补充")` 这类**运行时拼装**的名字。
裸 `n in src` 拦不住，但这不是真实的回归写法。

**Minor ⑤ 实质仍成立**：`default_kwargs` 把「🔥激战模式」取为 `default=False`，
故**战斗指挥官整条随机分支零覆盖**（含武器-招式联动、`random` 抽取）。非缺陷，是覆盖缺口。

**Minor ⑥ + 新发现 1 是同一件事的两面**：`data/composition.json` 已成孤儿文件，
且与 `README.md:64` 不自洽。审查者把它归为**用户的素材库决定** ——
即不该由我们擅自删。**需向用户提出。**

## 待用户裁决的一项（控制者已核实事实，未擅自处置）

**`data/composition.json` 与 `README.md:64` 是同一件事的两面。**

事实（机读）：
- `data/composition.json`：435 字节，顶层键只有一个 `shot_type`，mtime 2025-12-12。
- 全仓 `grep -rn composition --include=*.py --include=*.js`（排除 .superpowers）→ **零命中**。
  即：无 loader、无 `glob`/`listdir` 枚举 `data/`，`COMPOSITION_DATA` 已随 `构图范围` 控件一并删除。
- `README.md:64`（「数据目录」表 `data/` 行）仍写着
  「角色素材（身份/年龄/种族/体型/发型/服装/饰品/**构图**）……」—— `构图` 已不存在。

**控制者的处置**：
- README:64 的 `构图` 属**文档与代码不一致**，是本次改动自己造成的，应删（等终审对它的定级与建议后再动，
  避免在刚通过「可以合并」的 6 文件范围外追加提交而不重新取证）。
- `data/composition.json` **不擅自删**。删用户的素材文件不可逆、且属素材库决策，
  审查者也独立把它归为「用户的素材库决定」。留文件、只报事实。

## ★ 计划 B 收尾：终审通过 + 三处文档修正已提交（9135cf8）

**终审「合并前必须做的事」= 无阻塞项。** 未发现 Critical / Important、
未发现功能缺陷、未发现三次任务互相推翻、未发现任务审查看不见的功能问题。

**终审者主动修正了自己的裁决**：它原把 Minor #4 判为「不成立／误报」，
在收到我关于基线的澄清后改为「**不修（真 finding，已由 cb2883f 修复）**」，
并自行指出原审查者可能的误判路径（引用的 `:149-155` 恰是守卫用例的 docstring 正文，
那段为论证新用例必要性而明写「这两条测试都抓不到裸名」，被读成了「没有守卫」）。
→ **三个复核者现已完全一致**：该 finding 在 `01e79ea` 上成立，由 `cb2883f` 修复。
我先前的勘误保持有效。

**已按终审的「非阻塞建议」修正三处并提交 9135cf8**（各一行，均以机读事实为据）：
1. `README.md:64` —— `data/` 行去掉「/构图」（控件与 `COMPOSITION_DATA` 已删，
   `composition.json` 全仓零引用；不改则用户编辑该文件**静默无效**）
2. `README.md:91-95` —— 迁移说明逐一列全 **9 个**被删控件（原漏战斗指挥官的「补充指令」，
   且用「各节点的手动补充」笼统带过，实际只有 3 个节点有该控件）
3. `README.md:24` + `nodes/character_node.py:5` —— 「30+ 维度」→「30 个维度」
   （实测 31 控件 −「工作安全策略」= 30 个维度）
测试：`Ran 62 tests OK`。

**未处置、留待用户决定的一项**：`data/composition.json`（435 B，仅 `shot_type` 键）留删。
审查者独立归为「用户的素材库决定」，控制者同意：删用户的素材文件不可逆，不擅自动手。
（`README.md:64` 已改，故「留删」都不影响文档自洽。）

## 计划 B 状态：全部完成

| 任务 | 状态 |
|---|---|
| T1 键名去前缀（67 键 / 137 处） | ✅ 50f8a56 已审（0 Critical；2 Important 已由 f5b5e3f 内联修复） |
| T2 删 9 控件 + 纯标签输出 | ✅ 01e79ea 已审（两轮独立复核） |
| T3 README 同步 | ✅ 1b5259d |
| 终审（整分支） | ✅ **可以合并**，无阻塞项 |
| 终审的文档修正 | ✅ 9135cf8 |

**用户两项原始要求（终态机读核验）**：
无数字前缀 ✅；9 个圈选控件零残留 ✅；控件数 **31 / 12 / 9 / 6**；
`declared == read` 四文件均 1:1，无「控件还在但没人读」、无「读点指向已删键」。
**剩余未闭环的只有「肉眼在运行中的 ComfyUI 里确认」**（用户动作）。

工作区保留（未按 skill 默认删除）：本 ledger 是这次 10 处计划缺陷与三轮审查的
唯一完整记录，且计划 A 仍在进行中；待两条线都收尾后再一并清理。
