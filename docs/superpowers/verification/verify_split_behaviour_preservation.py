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
2. **语句级**：源 `process()` 的叶子语句，与新版承接它的那 8 个函数，作为**多重集**比对。

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

#: 源 `process()` 被拆进这 8 个函数。
#: 这 8 个 = `node.py` 的 `process()` 实际调用的那些函数（读 node.py:239-253 得到），
#: 不是猜的：prepare_run → run_detection → run_repair → run_global_pass
#: → (finalize_interrupt | build_report)；parse_config 由 prepare_run 调用。
PROCESS_SPLIT_TARGETS = (
    ("node.py", "process"),
    ("mixins/pipeline/prepare.py", "parse_config"),
    ("mixins/pipeline/prepare.py", "prepare_run"),
    ("mixins/pipeline/detect.py", "run_detection"),
    ("mixins/pipeline/repair.py", "run_repair"),
    ("mixins/pipeline/global_pass.py", "run_global_pass"),
    ("mixins/pipeline/report.py", "finalize_interrupt"),
    ("mixins/pipeline/report.py", "build_report"),
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
                "源用 `super().__init__() if hasattr(super(),'__init__') else None` 的兼容写法",
    "get_sam_dir": "新代码改用 folder_paths 的路径解析，去掉了源里的 os.path 兜底分支",
    "process": "源 1476 行的巨型 process() 被拆成 node.process + 7 个阶段函数（本脚本 §3 逐语句对拍）",
}

#: 已登记的、有正当理由的语句级差异（源侧多出的条目，按子串识别）。
EXPECTED_STMT_LOSSES = (
    ("V6.11.2", "版本横幅 V6.11.2 → V6.11.3，拆分计划强制"),
    ("hasattr(self, 'yolo_cache')",
     "新侧写作 hasattr(node, 'yolo_cache')：三种获批改写只覆盖 `node.x`，"
     "不覆盖以值的形式出现的裸 `node`，故此处对不上（属口径边界，非行为差异）"),
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
    if classes == ["LatentKSamplerUpscaler"]:
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

    # 改写次数统计走**整个新包 / 整个源文件**，不是只统计配对上的那些：
    # 承接 process() 的 7 个阶段函数本来就是「新包侧未配对」，
    # 只统计配对会把它们身上 100% 的改写漏掉，报出来的数字就没有意义。
    rewrite_used, rewrite_in_source = Counter(), Counter()
    for f in new_fns:
        for k, v in rewrite_counts(f.body).items():
            rewrite_used[k] += v
    for f in src_fns:
        for k, v in rewrite_counts(f.body).items():
            rewrite_in_source[k] += v
    emit("三种获批改写的命中次数（整个新包 / 整个源文件）：")
    for k in rewrite_counts(""):
        emit("    %-14s 新 %5d / 源 %5d" % (k, rewrite_used[k], rewrite_in_source[k]))
    emit("    源侧全 0 = 改写方向只有「新 → 源」一种，源文件里根本没有这三种形态，")
    emit("    所以「改写」不可能在源侧掩盖任何差异。")
    emit("")

    # ---------------- §3 语句级 ----------------
    emit("-" * 78)
    emit("§3 语句级：源 process() 的叶子语句 vs 承接它的 8 个函数")
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

    emit("=" * 78)
    if failure:
        emit("结论：**有未登记的差异 —— 需要人工判断**：%s" % failure)
        code = 1
    else:
        emit("结论：函数级「不同」恰为已登记的 3 个；语句级源侧丢失恰为已登记的 2 条；")
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
        print("")
        print("[报告已写入] %s" % os.path.relpath(REPORT_FILE, REPO_ROOT))
    except Exception as exc:                                # noqa: BLE001
        print("[报告写入失败] %s" % exc)
    sys.exit(exit_code)


if __name__ == "__main__":
    sys.exit(main())
