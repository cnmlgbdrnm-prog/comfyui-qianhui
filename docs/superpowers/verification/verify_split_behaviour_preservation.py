# -*- coding: utf-8 -*-
"""拆分「行为保持」证据复现脚本（整分支复审 Recommendation 6）。

## 这个脚本证明什么

`nodes/qianhui_latent_yolo/` 是从**一个 5633 行的单文件**拆出来的。
拆分唯一的**有意**改动是新增「配置预设」功能；除此之外，拆分必须**逐字保持行为**。
复审独立对拍过这件事，但结论只活在一句 commit message 和一份临时台账里。
本脚本把那套对拍变成可复跑的产物，并把输出落盘成
`behaviour-preservation-report.txt`（与脚本同目录）。

## 它读哪个源

    SOURCE_FILE = E:\\comfuyui 插件\\水印\\nodes.py

**该源文件在插件仓库之外，且是只读的**（拆分计划明确的约束：源目录保留不动，
好让「拆出来的东西 = 原文件」这件事永远可对拍）。所以：

  * 在**本机**（源仍在原位）这个脚本能跑，输出就是台账；
  * 在**别的机器**上源文件不存在，脚本会立刻以清晰的消息退出 —— 这不是 bug。
    届时请看 `behaviour-preservation-report.txt`：**报告已随脚本一起提交**，
    正是为了源消失后证据仍然在。

如果源目录真的丢了，用同一插件历史版本里那份 5633 行文件放回同一路径即可复跑，
但**必须核对行数与 md5**（报告首节记录），否则对拍的不是同一份基准。

## 两套对拍

1. **函数级**：源里每个函数与新版按名配对，施加计划批准的**三种机械改写**后逐字节比对。
2. **语句级**：源 `process()` 的叶子语句，与新版承接它的那 12 个函数（`PROCESS_SPLIT_TARGETS` 所列），作为**多重集**比对。

## 术语与判据（写死在此，避免口径漂移）

* **函数** = `ast.FunctionDef` + `ast.AsyncFunctionDef` + `ast.Lambda`。
  这一口径把复审报的 105 / 112 两个数字**逐一对上**（见报告 §1）。
* **叶子语句** = 自身不再含任何子语句的语句（`body`/`orelse`/`finalbody`/`handlers`
  四个语句表里都没有 `ast.stmt`）。复合语句只贡献它的叶子，不贡献自己。
  docstring 是叶子 `Expr`，但它是**新增注释**而非行为，两侧都排除。
* **三种获批改写**（**仅此三种**；多一种就算把口径放宽、把命中数灌水）：
    1. `cfg.x` / `ctx.x` → `x`
    2. `node.x` → `self.x`
    3. 剥掉**新增**的 docstring（两侧都剥，并单独统计「新增了 docstring」的条数）
* 函数级比的是**函数体**（不含签名/装饰器）：拆分把方法体搬成了模块级函数，
  签名必然不同，签名不是「行为」的一部分。
* 语句级比的是 `ast.unparse` 的**规范化文本**（而不是原文字面量）：
  拆分顺手重排了长行折行，拿字面量比会把「折行」误报成「行为变了」。
  代价是语句**内部**的注释不参与比对（整行注释本来就不属于任何语句）。

## 跑法

    python -X utf8 docs/superpowers/verification/verify_split_behaviour_preservation.py

退出码 0 = 对拍结果与「已登记的差异集」一致；1 = 出现未登记的差异，或源文件缺失。
"""
import ast
import hashlib
import io
import os
import re
import sys
import textwrap
from collections import Counter, OrderedDict

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------

#: 拆分基准（**插件仓库之外、只读**；见模块 docstring）。
SOURCE_FILE = r"E:\comfuyui 插件\水印\nodes.py"

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, os.pardir))
NEW_PACKAGE = os.path.join(REPO_ROOT, "nodes", "qianhui_latent_yolo")
REPORT_FILE = os.path.join(_HERE, "behaviour-preservation-report.txt")

#: 源 `process()` 被拆进这 12 个函数。
#: 前 8 个 = `node.py` 的 `process()` 实际调用的那些函数（读 node.py:239-253 得到），
#: 不是猜的：prepare_run → run_detection → run_repair → run_global_pass
#: → (finalize_interrupt | build_report)；parse_config 由 prepare_run 调用。
#: 第 9~11 个（T3 新增）= `report_bits.py` 的 build_preview / build_masks / build_info ——
#: 它们是从 build_report 里再拆出来的**子段**，`process()` 并不直接调用；但源
#: `process()` 里对应的语句现在由它们承接，不列进来就会把搬走的语句误判成源侧丢失。
#: 第 12 个（T4 新增）= `sam_refine.py` 的 refine_with_sam —— 同理，是从
#: run_detection 里再拆出来的**子段**（SAM 精修块，源 `process()` 里那 104 行）。
#:
#: ★ T13（单体退役）之后，承接那 1476 行 `process()` 的**主力换了人**：
#:   巨型 `process()` 与 `prepare_run` 本身都不在了，改由 4 个 node_*.py 各自的
#:   `process()` 承接（它们按阶段切开了那批语句）。所以这里把 4 个 process 补进来、
#:   并把 `node_detect.py` 的 `_finalize_interrupt`（= 原 `report.py:finalize_interrupt`
#:   逐字搬过去的那段）也列上 —— 不列就会把「搬走了」误判成「丢了」。
#:   映射的 `PROCESS_SPLIT_TARGETS` 必须与真身一一对应：**每一段源语句都要有人接**。
#: ★ T13 的两步（必须按顺序落地，否则中间那一步会留下未登记的语句级丢失）：
#:   ① 单体退役前：把**新**的承接者补进来（下面 4 个 process + `_finalize_interrupt`），
#:      旧条目原样留着 —— 它们此时还在磁盘上，去掉就会把源语句判成「丢了」。
#:   ② 单体退役后：删掉这 4 条（`("node.py","process")` / `("prepare.py","prepare_run")` /
#:      `("report.py","finalize_interrupt")` / `("report.py","build_report")` 里的**前两条**
#:      与 report.py 那两条 —— 它们指向的文件/函数已经不存在），
#:      并在 EXPECTED_STMT_LOSSES 里登记「原来由它们承接、现在没人承接」的那几段。
#:      ★ 那几段的原文用 `only_src` 的输出抄，**不要凭记忆写**。
PROCESS_SPLIT_TARGETS = (
    # 单体退役前仍在的旧承接者（退役后删这 4 条）
    ("node.py", "process"),
    ("mixins/pipeline/prepare.py", "prepare_run"),
    ("mixins/pipeline/report.py", "finalize_interrupt"),
    ("mixins/pipeline/report.py", "build_report"),
    # ★ T13 新增：真正的新承接者
    ("node_detect.py", "process"),
    ("node_sam.py", "process"),
    ("node_repair.py", "process"),
    ("node_global.py", "process"),
    # prepare_run 的正文（初始化 / 横幅 / 模型解析 / 承载张量）拆进了 3 个节点的
    # `_prepare` —— 不列它们，那 50 来条语句会被判成「源侧丢失」。
    # （node_sam 没有 `_prepare`：它在 `process` 里就地做完，上面那条已经覆盖它。）
    ("node_detect.py", "_prepare"),
    ("node_repair.py", "_prepare"),
    ("node_global.py", "_prepare"),
    ("node_detect.py", "_finalize_interrupt"),
    ("mixins/pipeline/prepare.py", "parse_config"),
    ("mixins/pipeline/detect.py", "run_detection"),
    ("mixins/pipeline/repair.py", "run_repair"),
    ("mixins/pipeline/global_pass.py", "run_global_pass"),
    ("mixins/pipeline/report_bits.py", "build_preview"),
    ("mixins/pipeline/report_bits.py", "build_masks"),
    ("mixins/pipeline/report_bits.py", "build_info"),
    ("mixins/pipeline/sam_refine.py", "refine_with_sam"),
)

#: 三种获批改写。
_REWRITES = (
    (re.compile(r"\bcfg\."), ""),
    (re.compile(r"\bctx\."), ""),
    (re.compile(r"\bnode\."), "self."),
)

#: 已登记的、有正当理由的函数级差异。多出任何一个都算回归。
EXPECTED_DIFF_FUNCTIONS = {
    "__init__": "聚合类 QianhuiLatentYOLO.__init__：新基类链用 super().__init__()，"
                "源用 `super().__init__() if hasattr(super(),'__init__') else None` 的兼容写法；"
                "T5 起 yolo_cache 改由模块级单例 get_shared_cache() 提供"
                "（源是每个节点各 new 一个 YOLOModelCache；单例内部的构造参数与源逐字相同："
                "max_cached_models=3, vram_threshold_mb=500）",
    "get_sam_dir": "新代码改用 folder_paths 的路径解析，去掉了源里的 os.path 兜底分支",
    "process": "源 1476 行的巨型 process() 被拆成 node.process + 7 个阶段函数（本脚本 §3 逐语句对拍）"
               "；T8 起另有 4 个 node_*.py 各自实现本节点的 process，同名条目由 2 条变 5 条，"
               "按名配对会配到不同实现上 —— 差异的性质不变（都是「巨型 process 被拆开」）",
    "INPUT_TYPES": "T8 起 4 个 node_*.py 各自声明**本节点那一部分**控件（节点名册见 "
                   "task-8-14-brief.md §一），同名条目由源的 2 条变成 6 条。按名配对于是把源那条"
                   "**注册节点**的 INPUT_TYPES 配到了某个**子集**上 —— 子集当然不等于全集，"
                   "差异是拆分本身造成的，不是单体被改。★ 但«单体控件表一个字节都没动»"
                   "不能只靠这条登记：它由下面 §2b 按 (类名, 函数名) 定点比对单独护住，"
                   "那一处**不可登记豁免**（必须逐字节相等）。",
    "detect_sam_version": "SAM 3 追加分支（用户后续明确要求，优先于「leaf mixin 与 "
                          "model_managers.py 冻结」的原计划）：新增 sam3 关键词判定，"
                          "且**必须排在 sam2 的 `hiera` 之前** —— SAM3 权重名 "
                          "`sam3_hiera_l.pt` 里含 `hiera`，顺序反了会被判成 sam2，"
                          "于是拿 SAM2 的加载器去装 SAM3 的权重。",
    "_refine_with_sam": "同上（SAM 3）：新增 `sam_version == \"sam3\"` 的加载分支，"
                        "并把它上面那条 `if` 改成 `elif`。SAM1/SAM2 两条路的语句一条没动；"
                        "两个版本接口形状的差异全部关在新加的 `_SAM3BoxPredictor` 适配层里"
                        "（该类在源里不存在，属新包侧未配对项，不参与本表）。",
}

#: 已登记的、有正当理由的语句级差异（源侧多出的条目，按子串识别）。
EXPECTED_STMT_LOSSES = (
    ("V6.11.2", "版本横幅 V6.11.2 → V6.11.3，拆分计划强制"),
    ("hasattr(self, 'yolo_cache')",
     "新侧写作 hasattr(node, 'yolo_cache')：三种获批改写只覆盖 `node.x`，"
     "不覆盖以值的形式出现的裸 `node`，故此处对不上（属口径边界，非行为差异）"),

    # ==================== T13：单体退役带来的 8 条 ====================
    #
    # ★ 全部是**用工作区实测**出来的，不是凭记忆写的：把 T13 在一份仓外副本上做完、
    #   跑一遍本脚本、把 `only_src` 的原文逐条抄过来（当时的未登记条数是 12，
    #   其中 2 条已由上表两条覆盖）。**每条都要有理由，理由必须成立。**
    ("import time",
     "prepare_run 里那处**函数内** `import time` 没有逐字搬过去：4 个节点在模块级 "
     "`import time`（模块级语句不在 §3 的扫描范围内），函数体内不再出现这一句。"
     "同一件事换了位置，不是丢掉。"),
    ("out_latents = original_samples.clone()",
     "单体在 prepare_run 里**预分配** `out_latents` 的「原图回填」内容（三路分支之一）；"
     "拆分后各节点只分配空张量，未处理帧改由各自的异常分支**显式回填**"
     "（node_detect/_finalize_interrupt、node_global 的 except 段）。"
     "值一样、位置不同 —— 三条分支因此各少一条。"),
    ("out_latents = self.upscale_latent(original_samples, 放大算法, 放大倍数)",
     "同上：预分配的第三条分支。★ 真正的放大回填逻辑**没有丢** —— "
     "node_detect.py 的 `_finalize_interrupt` 里 `放大倍数 > 1.0` 那条分支逐字保留"
     "（latent_aware_resize / upscale_latent 都在），只是赋值目标从 `out_latents` "
     "变成了局部量 `orig_sample`，所以文本对不上。"),
    ("out_latents = LatentAlgorithms.latent_aware_resize(original_samples, target_h, target_w)",
     "同上：预分配的第二条分支（理由见上一条）。"),
    ("out_previews.append(torch.zeros((1, 64, 64, 3)))",
     "同一件事在源 process() 里是**一行**（append 里直接构造），"
     "抽进 finalize_interrupt 后写成**两行**（先 `empty_preview = torch.zeros(...)` "
     "再 append）。多重集按语句文本比对，一行 vs 两行必然对不上；语义相同。"),
    ("preview_image = self._decode_latent_to_image(VAE, processed_latent)[0]",
     "★ T13 删掉的正是这一段：`global_pass.py:85-86` 的 "
     "`if ctx.preview_image is None: ... _decode_latent_to_image(...)` 是**永不执行**"
     "的兜底（每个调用点都在循环头绑好了 `ctx.preview_image`，恒不为 None）。"
     "删死代码不是改行为 —— 但它确实少一条语句，故在此登记。"),
    ("print(f'[中断] 未完成任何处理，返回原始输入')",
     "中断文案在 T3 抽 `finalize_interrupt` 时改写成了 5 行"
     "（「[中断] 用户停止处理」/「已完成 N/M 张」/「返回已处理的结果（未处理的部分保持原图）」…）；"
     "源那一条独立成句的文案在文字上不复存在。属获批的文案改写，非行为差异。"),
    ("return ({'samples': final_samples}, final_preview, final_masks, info)",
     "单体一次返回四路；拆分后由 4 个节点各自的 return + **链条接线**取代"
     "（A→B→C→D，见 README「4 节点链」）。接线就是这条语句的实现，故它本身不再存在。"),
    ("target_h = int(samples.shape[2] * 放大倍数)",
     "单体在 process() 里按**整批**的张量尺寸算放大目标；拆分后每个节点在自己的 "
     "`_prepare` 里按自己那批算（写法不同：`ctx.pixel_h` / `target_h`）。"
     "同一件事换了算的位置与写法。"),
    ("target_w = int(samples.shape[3] * 放大倍数)",
     "同上（宽那一半）。"),
)

_STMT_LIST_FIELDS = ("body", "orelse", "finalbody", "handlers")

_OUT = []


def emit(line=""):
    _OUT.append(line)
    print(line)


# ---------------------------------------------------------------------------
# 抽取
# ---------------------------------------------------------------------------


def _is_docstring_stmt(stmt):
    return (isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str))


def _body_source(lines, fn_node):
    """函数体原文（不含签名/装饰器）。返回 (body_without_docstring, has_docstring)。

    行区间 = 「首个语句起 → 函数末行」，**不**跳过 docstring 行（跳了会把紧跟在
    docstring 后面的注释切掉，而那些注释正是最容易在搬运时漂移的东西）。
    docstring 本身另用 AST 判定、单独统计。
    """
    stmts = fn_node.body
    has_doc = bool(stmts) and _is_docstring_stmt(stmts[0])
    if not stmts:
        return "", has_doc
    raw = "\n".join(lines[stmts[0].lineno - 1:fn_node.end_lineno])
    return textwrap.dedent(raw), has_doc


def _strip_docstring(text, fn_node):
    if fn_node.body and _is_docstring_stmt(fn_node.body[0]):
        first = fn_node.body[0]
        lines = text.splitlines()
        # docstring 占的行 = 相对函数体起始的偏移
        base = fn_node.body[0].lineno
        rel = fn_node.body[1].lineno - base if len(fn_node.body) > 1 else None
        if rel is None:
            return ""
        del lines[:rel]
        return "\n".join(lines)
    return text


class Fn(object):
    """一个「函数」：def / async def / lambda。"""

    __slots__ = ("name", "kind", "lineno", "end_lineno", "origin", "body",
                 "has_docstring", "node", "source_path", "cls")

    def __init__(self, name, kind, lineno, end_lineno, origin, body,
                 has_docstring, node, source_path, cls=None):
        self.name = name
        self.kind = kind
        self.lineno = lineno
        self.end_lineno = end_lineno
        self.origin = origin
        self.body = body
        self.has_docstring = has_docstring
        self.node = node
        self.source_path = source_path
        self.cls = cls              # 所属类名（lambda 取其所在函数所属的类）

    def __repr__(self):
        return "<%s %s @%s:%d>" % (self.kind, self.name, self.origin, self.lineno)


def extract_functions(path):
    """按固定口径抽出文件里的全部「函数」，按出现顺序返回。"""
    src = io.open(path, encoding="utf-8").read()
    lines = src.splitlines()
    tree = ast.parse(src)
    origin = (os.path.relpath(path, REPO_ROOT) if path.startswith(REPO_ROOT)
              else os.path.basename(path))

    out = []

    def walk(node, cls):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                walk(child, child.name)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                body, has_doc = _body_source(lines, child)
                if has_doc:
                    body = _strip_docstring(body, child)
                out.append(Fn(child.name, "def", child.lineno, child.end_lineno,
                              origin, body, has_doc, child, path, cls))
                walk(child, cls)
            elif isinstance(child, ast.Lambda):
                seg = ast.get_source_segment(src, child) or ""
                out.append(Fn("lambda", "lambda", child.lineno, child.end_lineno,
                              origin, textwrap.dedent(seg), False, child, path, cls))
                walk(child, cls)
            else:
                walk(child, cls)

    walk(tree, None)
    out.sort(key=lambda f: f.lineno)
    return out


def new_package_files():
    files = []
    for dirpath, dirnames, filenames in os.walk(NEW_PACKAGE):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for name in sorted(filenames):
            if name.endswith(".py"):
                files.append(os.path.join(dirpath, name))
    return sorted(files)


# ---------------------------------------------------------------------------
# 归一化与比对
# ---------------------------------------------------------------------------


def rewrite_counts(text):
    return OrderedDict((rx.pattern, len(rx.findall(text))) for rx, _ in _REWRITES)


def normalise(text):
    for rx, repl in _REWRITES:
        text = rx.sub(repl, text)
    return text.strip()


def pair_by_name(src_fns, new_fns):
    """按名配对。同名有多条时，**先配逐字节相同的，剩下的按函数体大小降序配**。

    为什么不能「按出现顺序」：源里 `INPUT_TYPES` / `process` 各有 2 条 ——
    未注册基类 `LatentKSamplerUpscaler` 一条、注册节点 `LatentKSamplerUpscalerYOLO` 一条；
    新包只剩注册节点那一条。按出现顺序配会把**基类的桩**配到新聚合类上，
    于是「差异」变成 4 个且落空的是主 process —— 全错。

    为什么按**大小降序**：同名重复对应的是「一个桩 + 一个真实现」，
    桩小、真实现大；大小降序天然把大的配大的。若同名两条大小相同（如
    `_patched_load` 112 vs 112），先做的「逐字节相同优先」会把它们直接钉在一起。

    配对规则只用来**选配对**，不参与判定：判定永远是「归一化后逐字节相等」。
    """
    def groups(fns):
        g = OrderedDict()
        for f in fns:
            g.setdefault(f.name, []).append(f)
        return g

    sg, ng = groups(src_fns), groups(new_fns)
    pairs, used_n = [], set()

    for name, s_group in sg.items():
        n_group = ng.get(name, [])
        remaining_s, remaining_n = list(s_group), list(n_group)

        # 1) 逐字节相同的先钉死
        for s in list(remaining_s):
            hit = next((n for n in remaining_n
                        if normalise(n.body) == normalise(s.body)), None)
            if hit is not None:
                remaining_s.remove(s)
                remaining_n.remove(hit)
                pairs.append((s, hit))
                used_n.add(id(hit))

        # 2) 剩下的按函数体大小降序配
        remaining_s.sort(key=lambda f: -len(f.body))
        remaining_n.sort(key=lambda f: -len(f.body))
        for i in range(min(len(remaining_s), len(remaining_n))):
            pairs.append((remaining_s[i], remaining_n[i]))
            used_n.add(id(remaining_n[i]))

    pairs.sort(key=lambda p: p[0].lineno)
    paired_s = {id(s) for s, _ in pairs}
    src_unpaired = [f for f in src_fns if id(f) not in paired_s]
    new_unpaired = [f for f in new_fns if id(f) not in used_n]
    return pairs, src_unpaired, new_unpaired


# ---------------------------------------------------------------------------
# 语句级
# ---------------------------------------------------------------------------


def leaf_statements(fn_node):
    """叶子语句（定义见模块 docstring），按源码顺序返回语句节点。docstring 已排除。"""
    out = []
    for stmt in ast.walk(fn_node):
        if not isinstance(stmt, ast.stmt):
            continue
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if _is_docstring_stmt(stmt):
            continue
        if any(isinstance(x, ast.stmt)
               for f in _STMT_LIST_FIELDS
               for x in (getattr(stmt, f, None) or [])):
            continue
        out.append(stmt)
    out.sort(key=lambda s: (s.lineno, s.col_offset))
    return out


def stmt_key(stmt):
    try:
        return normalise(ast.unparse(stmt))
    except Exception:                                       # noqa: BLE001
        return "<unparse-failed>"


def find_function(path, name):
    """按名取唯一函数；同名多于一个时取最大的（并把这个规则说出来）。"""
    src = io.open(path, encoding="utf-8").read()
    cands = [n for n in ast.walk(ast.parse(src))
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]
    if not cands:
        return None, src, None
    cands.sort(key=lambda n: n.end_lineno - n.lineno, reverse=True)
    return cands[0], src, len(cands)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                                       # noqa: BLE001
        pass

    # `_OUT` 是模块级全局，**每次进入 main 都要清空**：
    # 不清的话第二次 in-process 调用会把整份报告**再追加一遍**
    # （脚本自己跑是单次，但「可重跑」包括被当模块导入后重复调用——
    #  复审正是这样复验本脚本的，所以这条不是假想）。
    del _OUT[:]

    emit("=" * 78)
    emit("拆分行为保持 —— 对拍报告（由脚本生成，勿手改）")
    emit("生成：python -X utf8 docs/superpowers/verification/"
         "verify_split_behaviour_preservation.py")
    emit("=" * 78)
    emit("")
    emit("源（外部、只读）：%s" % SOURCE_FILE)
    emit("新包：            %s" % os.path.relpath(NEW_PACKAGE, REPO_ROOT))
    emit("")

    failure = None
    if not os.path.isfile(SOURCE_FILE):
        emit("!!! 源文件不存在 —— 本脚本无法复跑（见脚本 docstring 第 2 节）。")
        emit("!!! 这**不是**产物坏了：已提交的本报告就是证据；")
        emit("!!! 要复跑请把那份 5633 行源文件放回上面的路径。")
        _flush(1)
        return 1

    with io.open(SOURCE_FILE, "rb") as fh:
        blob = fh.read()
    emit("源指纹：md5=%s  bytes=%d  lines=%d"
         % (hashlib.md5(blob).hexdigest(), len(blob),
            len(blob.decode("utf-8").splitlines())))
    emit("")

    src_fns = extract_functions(SOURCE_FILE)
    new_fns = []
    for path in new_package_files():
        new_fns.extend(extract_functions(path))

    # ---------------- §1 计数 ----------------
    emit("-" * 78)
    emit("§1 函数计数（口径：def + async def + lambda，含嵌套）")
    emit("-" * 78)
    emit("源   ：def %d + lambda %d = %d"
         % (sum(1 for f in src_fns if f.kind == "def"),
            sum(1 for f in src_fns if f.kind == "lambda"), len(src_fns)))
    emit("新包 ：def %d + lambda %d = %d"
         % (sum(1 for f in new_fns if f.kind == "def"),
            sum(1 for f in new_fns if f.kind == "lambda"), len(new_fns)))
    emit("")

    # ---------------- §2 函数级 ----------------
    emit("-" * 78)
    emit("§2 函数级：按名配对 + 三种获批改写后逐字节比对")
    emit("-" * 78)
    pairs, src_unpaired, new_unpaired = pair_by_name(src_fns, new_fns)

    src_names = {f.name for f in src_fns}
    new_names = {f.name for f in new_fns}
    missing_names = sorted(src_names - new_names)
    emit("源函数名去重 %d 个；新包中**没有同名函数**的 = %d 个 %s"
         % (len(src_names), len(missing_names), missing_names or ""))
    emit("")

    identical, differing = [], []
    for s, n in pairs:
        (identical if normalise(s.body) == normalise(n.body) else differing).append((s, n))

    emit("配对成立 %d 对 → 逐字节相同 %d，不同 %d"
         % (len(pairs), len(identical), len(differing)))
    emit("")
    emit("不同的条目（逐个说明理由；理由必须成立，否则就是回归）：")
    for s, n in differing:
        reason = EXPECTED_DIFF_FUNCTIONS.get(s.name)
        emit("  * %-16s 源 @%-5d  →  %s:%d"
             % (s.name, s.lineno, n.origin, n.lineno))
        emit("      %s" % (reason if reason else "*** 未登记 —— 需人工确认是否回归 ***"))
        if reason is None:
            failure = "未登记的函数级差异：%s" % s.name
    emit("")

    # 源侧落空 = 新包里没有可配对的同函数（同名条目在新包中已被配对用完）。
    emit("源侧落空 %d 个（同名条目在新包中已被配对用完）：" % len(src_unpaired))
    for f in src_unpaired:
        emit("  - %s.%s 源 @%d（body %d 字节）"
             % (f.cls or "<module>", f.name, f.lineno, len(f.body)))
    classes = sorted({f.cls for f in src_unpaired})
    if not src_unpaired:
        emit("    ★ 源侧落空 **0 个** —— 源的每一条函数都在新包里找到了同名对应，")
        emit("      也就是说没有任何源函数被漏搬（比「只剩 2 条未注册基类的桩」更强）。")
        emit("      为什么不再只剩那 2 条：T8 起新包的 4 个 node_*.py 也提供了")
        emit("      `process` / `INPUT_TYPES` / `__init__` 同名条目，按大小降序配对时")
        emit("      把那两条桩接走了。名字对上了不等于语义对上了 —— 所以单体那两处")
        emit("      由下面 §2b 按 (类名, 函数名) 定点复核，不吃这套配对规则。")
    elif classes == ["LatentKSamplerUpscaler"]:
        emit("    ★ 落空的两条**都**属于 `LatentKSamplerUpscaler` —— 源里那个")
        emit("      **未被注册**的基类。`base_node.py` 的 docstring 明写：")
        emit("      「仅保留 upscale_latent —— 源类 LatentKSamplerUpscaler 的")
        emit("       INPUT_TYPES 与 process 从未被调用（该基类未注册），不再移植。」")
        emit("      → 不是漏搬，是有意丢弃，且有据可查。")
    else:
        emit("    *** 落空的类不是预期的 LatentKSamplerUpscaler：%s ***" % classes)
        failure = failure or "源侧落空的不只是未注册基类的桩：%s" % classes
    emit("")

    doc_added = [(s, n) for s, n in pairs if n.has_docstring and not s.has_docstring]
    emit("新包侧新增 docstring 的配对 %d 对（第 3 种获批改写，两侧都已剥掉再比）："
         % len(doc_added))
    for s, n in doc_added:
        emit("  - %s → %s" % (s.name, n.origin))
    emit("")

    emit("新包侧未配对 %d 个（拆分新增 / 改名 / 新功能的产物）：" % len(new_unpaired))
    for f in new_unpaired:
        emit("  + %-20s %s @%d" % (f.name, f.origin, f.lineno))
    emit("")

    # ---------------- §2b 单体控件表的定点复核 ----------------
    #
    # ★ 为什么必须补这一节：T8 起 4 个 node_*.py 各自声明了自己的 INPUT_TYPES，
    #   源那两条 INPUT_TYPES 于是被「按名配对」配到某个**子集**上（§2 里作为
    #   已登记差异列出）。代价是：**「单体的控件表一个字节都没动」这句话在 §2
    #   里再没有证据**。而它恰好是本次拆分最要紧的一条不变量 —— 4 个节点的控件
    #   是从单体逐字抄的（brief §一），抄错一个字，用户的界面就变了。
    #   所以这里按 (类名, 函数名) 精确配对复核，不经按名配对、不参与大小排序，
    #   也**不可登记豁免**：它必须逐字节相等。
    emit("-" * 78)
    emit("§2b 定点复核：单体注册类的 INPUT_TYPES vs 源同名函数（不经按名配对）")
    emit("-" * 78)
    _SRC_NODE_CLS = "LatentKSamplerUpscalerYOLO"
    _NEW_NODE_CLS = "QianhuiLatentYOLO"
    src_it = [f for f in src_fns if f.name == "INPUT_TYPES" and f.cls == _SRC_NODE_CLS]
    new_it = [f for f in new_fns if f.name == "INPUT_TYPES" and f.cls == _NEW_NODE_CLS]
    emit("%-42s：%d 条" % ("源 %s.INPUT_TYPES" % _SRC_NODE_CLS, len(src_it)))
    emit("%-42s：%d 条" % ("新 node.py %s.INPUT_TYPES" % _NEW_NODE_CLS, len(new_it)))
    if len(src_it) != 1 or len(new_it) != 1:
        emit("    *** 定位不到唯一的一条 —— 这处证据失效，需人工确认 ***")
        failure = failure or ("§2b 定位失败：源 %d 条 / 新 %d 条"
                              % (len(src_it), len(new_it)))
    else:
        it_same = normalise(src_it[0].body) == normalise(new_it[0].body)
        emit("    逐字节比对（三种获批改写 + 剥 docstring 后）：%s"
             % ("**相同**" if it_same else "**不同**"))
        if not it_same:
            emit("    *** 单体控件的声明被改动了 —— 界面会变，而且 4 个节点是照它抄的 ***")
            failure = failure or "§2b：单体 INPUT_TYPES 与源不再逐字节相同"
    emit("")

    # 改写次数统计走**两侧各自提取出的全部函数**（不限于配对上的那些）：
    # 承接 process() 的 7 个阶段函数本来就是「新包侧未配对」，
    # 只统计配对会把它们身上 100% 的改写漏掉，报出来的数字就没有意义。
    # ⚠️ 口径边界：`rewrite_counts` 吃的是**函数体**文本，
    #    函数**之外**的模块级语句不在扫描范围内 —— 所以下面的标签
    #    必须写「提取出的函数体内」，不能写「整个源文件」。
    rewrite_used, rewrite_in_source = Counter(), Counter()
    for f in new_fns:
        for k, v in rewrite_counts(f.body).items():
            rewrite_used[k] += v
    for f in src_fns:
        for k, v in rewrite_counts(f.body).items():
            rewrite_in_source[k] += v
    emit("三种获批改写的命中次数（两侧**提取出的函数体内**，非整个文件）：")
    for k in rewrite_counts(""):
        emit("    %-14s 新 %5d / 源 %5d" % (k, rewrite_used[k], rewrite_in_source[k]))
    emit("    源侧全 0 = 改写方向只有「新 → 源」一种，源文件里根本没有这三种形态，")
    emit("    所以「改写」不可能在源侧掩盖任何差异。")
    emit("")

    # ---------------- §3 语句级 ----------------
    emit("-" * 78)
    # 数字**不写死**：T13 删 prepare_run 后会变 10，T4/T12 也可能增减。
    # 这是打印给用户看的输出，写死就会在某次改动后印出一个错的数。
    emit("§3 语句级：源 process() 的叶子语句 vs 承接它的 %d 个函数"
         % len(PROCESS_SPLIT_TARGETS))
    emit("-" * 78)
    src_proc, src_text, n_same = find_function(SOURCE_FILE, "process")
    emit("源 process()：同名 %d 个，取最大者 @line %d-%d（%d 行）= 主节点"
         % (n_same, src_proc.lineno, src_proc.end_lineno,
            src_proc.end_lineno - src_proc.lineno + 1))
    src_leaves = leaf_statements(src_proc)
    emit("    全部语句 %d 条；**叶子语句** %d 条"
         % (sum(1 for x in ast.walk(src_proc) if isinstance(x, ast.stmt)),
            len(src_leaves)))
    emit("")

    new_leaves = []
    for rel, fname in PROCESS_SPLIT_TARGETS:
        path = os.path.join(NEW_PACKAGE, rel.replace("/", os.sep))
        fn, _text, cnt = find_function(path, fname)
        got = leaf_statements(fn)
        new_leaves.extend(got)
        emit("    %-34s %-18s 叶子 %4d 条" % (rel, fname, len(got)))
    emit("    承接口合计 叶子 %d 条" % len(new_leaves))
    emit("")

    sk = Counter(stmt_key(s) for s in src_leaves)
    nk = Counter(stmt_key(s) for s in new_leaves)
    matched = sum((sk & nk).values())
    emit("多重集比对：源 %d 条，新版 %d 条，**逐字命中 %d 条**（%.1f%%）"
         % (len(src_leaves), len(new_leaves), matched,
            100.0 * matched / max(1, len(src_leaves))))
    emit("")

    only_src = sk - nk
    only_new = nk - sk
    lost_total = sum(only_src.values())          # ★ 给结论行用：源侧丢失的**实际**条数
    emit("源多出 %d 条（即「新版没承接的」，**每一条都必须有理由**）："
         % sum(only_src.values()))
    for text, cnt in sorted(only_src.items()):
        why = None
        for needle, reason in EXPECTED_STMT_LOSSES:
            if needle in text:
                why = reason
                break
        emit("  x%d  %s" % (cnt, text.replace("\n", " / ")[:120]))
        emit("      %s" % (why if why else "*** 未登记 —— 可能是真回归 ***"))
        if why is None:
            failure = failure or "未登记的语句级丢失：%s" % text[:80]
    emit("")
    emit("新版多出 %d 条（拆分脚手架 / 新功能 / 获批改写产物）："
         % sum(only_new.values()))
    for text, cnt in sorted(only_new.items()):
        emit("  x%d  %s" % (cnt, text.replace("\n", " / ")[:120]))
    emit("")
    preset_related = sum(c for t, c in only_new.items()
                         if "presets" in t or "apply_preset" in t or "NO_PRESET" in t)
    emit("注：新版多出的 %d 条里，与「配置预设」后端有关（含 presets 兜底）的 %d 条，"
         % (sum(only_new.values()), preset_related))
    emit("    其余 %d 条全是拆分脚手架（把局部变量搬上 ctx/cfg、把巨型 process 切成阶段调用）。"
         % (sum(only_new.values()) - preset_related))
    emit("    另外：`配置预设` 这个**控件本身在源文件里就已存在**")
    emit("    （源 nodes.py:1372-1373「V6.6 配置预设」），新版只是补上了它的后端 ——")
    emit("    所以 `INPUT_TYPES` 能逐字节相同，不是漏改。")
    emit("")

    # ---------------- §4 与复审数字核对 ----------------
    emit("-" * 78)
    emit("§4 与整分支复审所报数字的核对")
    emit("-" * 78)
    emit("  复审所报                                | 本次实算")
    emit("  源函数 105                              | %d   %s"
         % (len(src_fns), "一致" if len(src_fns) == 105 else "**不一致**"))
    emit("  新包函数 112                            | %d   %s"
         % (len(new_fns), "一致" if len(new_fns) == 112 else "**不一致**"))
    emit("  源函数缺失 0                            | 按名缺失 %d（按同名条目落空 %d）"
         % (len(missing_names), len(src_unpaired)))
    emit("  按名配对 112 → 逐字节相同 109，不同 3   | 配对 %d → 相同 %d，不同 %d"
         % (len(pairs), len(identical), len(differing)))
    emit("  process() 叶子语句 902 → 命中 901       | 叶子 %d → 命中 %d"
         % (len(src_leaves), matched))
    emit("")
    emit("复审所报「不同的 3 个」= %s"
         % "、".join(sorted(EXPECTED_DIFF_FUNCTIONS)))
    emit("本次实算的「不同」    = %s"
         % ("、".join(sorted(s.name for s, _ in differing)) or "（无）"))
    emit("")
    emit("★ T8 起上表的「新包函数」必然变大（复审所报 112 → 本次实算 %d）：新增了"
         % len(new_fns))
    emit("  4 个 node_*.py 及其私有类与私有方法，**分母变大不是差异变大**。")
    emit("  源侧的数字一位没变（源函数 105 → %d，源侧落空 0，源 md5 与复审时相同）——"
         % len(src_fns))
    emit("  「不同」的**名字集合**变大了，是因为 4 个节点各自也声明了同名函数")
    emit("  （INPUT_TYPES / process），它们与源的同名条目配对后必然不同，逐条理由")
    emit("  都在上表里。")
    emit("")

    emit("=" * 78)
    if failure:
        emit("结论：**有未登记的差异 —— 需要人工判断**：%s" % failure)
        code = 1
    else:
        emit("结论：函数级「不同」恰为已登记的 %d 个；语句级源侧丢失 %d 处全部有登记"
             "（登记表 %d 条 —— 表比实际多出的是**尚未生效**的："
             % (len(EXPECTED_DIFF_FUNCTIONS), lost_total, len(EXPECTED_STMT_LOSSES)))
        emit("      单体退役（T13）之前，那几条对应的源语句仍由旧承接者匹配着，"
             "所以现在还没被用到）；")
        emit("      单体注册类的 INPUT_TYPES 与源逐字节相同（§2b）；")
        emit("      其余全部逐字相同。拆分**未发现未登记的行为变化**。")
        code = 0
    emit("=" * 78)

    _flush(code)
    return code


def _flush(exit_code):
    payload = "\n".join(_OUT) + "\n"
    try:
        with io.open(REPORT_FILE, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(payload)
    except Exception as exc:                                # noqa: BLE001
        # ★ 写报告失败必须**强制非零退出**，不能用上面的 exit_code：
        #   exit_code 只反映「有没有未登记的差异」。若写入失败（目录只读 /
        #   磁盘满 / 权限）却退出 0，自动化重跑就分不出「报告已重建」
        #   和「报告根本没产出」—— 本脚本存在的意义就是产出那份报告，
        #   产出不了却报成功，正是本项目一路在猎的静默失败形状。
        print("[报告写入失败] %s" % exc)
        sys.exit(1)
    # 显示路径的解析**单独一个 try**，不与写入共用：
    # 两者同源（都从 _HERE 推出）故恒同盘、正常情况下不会抛；
    # 但一旦并进上面的 try，任何显示层异常都会打「[报告写入失败]」，
    # 把一次**已经成功**的写入误报成失败。
    try:
        shown = os.path.relpath(REPORT_FILE, REPO_ROOT)
    except ValueError:
        shown = REPORT_FILE
    print("")
    print("[报告已写入] %s" % shown)
    sys.exit(exit_code)


if __name__ == "__main__":
    sys.exit(main())
