# 千绘 · 4 节点控件精简 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 去掉千绘 4 个既有节点（角色定制/肢体动作师/场景光效大师/战斗指挥官）控件标签的数字前缀，并删除用户指定的 8 个控件，同时把删除 `模型类型`/`环境总权重` 后的输出统一为纯标签。

**Architecture:** 两个任务，都是对现有单文件的原地修改。Task 1 是 67 个参数键名的重命名（键名即控件标签，也是 `kwargs` 的键，必须同步替换全部引用）；Task 2 删除 8 个控件并收敛权重分支。因为 4 个节点只依赖 `re`/`json`/`os`/`random`/`time`，**不依赖 ComfyUI/torch/GPU**，所以两者都能用单元测试自动验证。

**Tech Stack:** Python 3.14、unittest、ComfyUI 节点 API（仅 `INPUT_TYPES` 结构约定）。

## Global Constraints

- 4 个节点文件：`nodes/character_node.py`、`nodes/pose_node.py`、`nodes/scene_node.py`、`nodes/combat_node.py`。**不改其他节点文件**。
- 节点 class 名与注册名**不变**（`QianhuiCharacterCreator` / `QianhuiActionMaster` / `QianhuiSceneMaster` / `QianhuiCombatMaster`）。
- 控件**顺序不变**（顺序由 `INPUT_TYPES` 书写顺序决定，与键名无关）。
- 改名规则：只去掉开头的「数字 + 可选小写字母 + 下划线」，其余原样保留（含 emoji、内层下划线、斜杠）。
- 被删控件的功能**不做替代**，相关输出段一并去掉。
- `character_node.py` 的**内置权重不动**（237、239、291、300、353 行的 `:1.2`/`:1.35`/`:1.1` 等）——该节点没有 `模型类型` 控件，这些权重与「模型类型切换」无关。唯一例外是 381 行随 `31_构图范围` 删除的 `f"({shot_tag}:1.1)"`。
- 测试用 `unittest`，不用 pytest。运行：`python -X utf8 tests/test_node_widgets.py` 或 `python -X utf8 -m unittest discover -s tests -p "test_*.py"`。
- **所有测试命令必须带 `-X utf8`**：本机控制台是 gbk，而 `nodes/character_node.py:61` 在导入时打印 `⏳`，不加会以 `UnicodeEncodeError` 失败（表现为 `FAILED (errors=N)`，不是断言失败）。
- 每个 commit 结尾加 `Co-Authored-By: Claude Code <noreply@anthropic.com>`。
- 测试前先确认既有测试仍全绿（基线）：`python -X utf8 -m unittest discover -s tests -p "test_*.py"` → 21 个用例 OK。

## 待删控件清单

| 节点 | 删除的控件 |
|---|---|
| `character_node.py` | `31_构图范围`、`32_Lora触发词`、`33_手动补充` |
| `pose_node.py` | `06_手动补充`、`09_模型类型` |
| `scene_node.py` | `07_手动补充`、`08_环境总权重` |
| `combat_node.py` | `07_模型类型`、`06_补充指令` |

---

## Task 1: 键名去掉数字前缀（67 个）

**Files:**
- Modify: `nodes/character_node.py`（34 个键）
- Modify: `nodes/pose_node.py`（14 个键）
- Modify: `nodes/scene_node.py`（11 个键）
- Modify: `nodes/combat_node.py`（8 个键）
- Create: `tests/test_node_widgets.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - 4 个文件里 67 个 `INPUT_TYPES` 键名与**全部** `kwargs` 引用改为无前缀形式
  - `tests/test_node_widgets.py`：`load_node(module_file: str) -> module`、`all_keys(cls) -> list[str]`、`default_kwargs(cls) -> dict`、常量 `OLD_KEYS: dict[str, set[str]]`。Task 2 复用这些辅助函数。

- [ ] **步骤 1：写测试装载器与三个辅助函数**

`tests/test_node_widgets.py`：

```python
# -*- coding: utf-8 -*-
"""4 节点控件精简的回归测试。

这 4 个节点只依赖 re/json/os/random/time，无 ComfyUI 依赖，可按路径直载。
"""
import importlib.util
import io
import os
import re
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODES_DIR = os.path.join(ROOT, "nodes")

FILES = {
    "character": ("character_node.py", "QianhuiCharacterCreator", "generate_character"),
    "pose": ("pose_node.py", "QianhuiActionMaster", "generate_pose_v8"),
    "scene": ("scene_node.py", "QianhuiSceneMaster", "generate_scene_v6"),
    "combat": ("combat_node.py", "QianhuiCombatMaster", "generate_combat"),
}

PREFIX_RE = re.compile(r"^\d+[a-z]?_")


def load_node(module_file: str):
    """按路径装载节点模块（无包内相对导入，可直接 exec）。"""
    path = os.path.join(NODES_DIR, module_file)
    name = "qh_" + os.path.splitext(module_file)[0]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def node_class(key: str):
    module_file, cls_name, _ = FILES[key]
    return getattr(load_node(module_file), cls_name)


def gen_func(key: str):
    module_file, _, fn_name = FILES[key]
    return getattr(load_node(module_file), fn_name)


def all_keys(cls) -> list:
    spec = cls.INPUT_TYPES()
    keys = []
    for section in ("required", "optional"):
        keys.extend(spec.get(section, {}).keys())
    return keys


def _first_real(options) -> str:
    """下拉默认取第一个非占位选项，使生成结果非空、断言才有意义。"""
    for o in options:
        if isinstance(o, str) and o and "(不指定)" not in o:
            return o
    return options[0] if options else ""


def default_kwargs(cls) -> dict:
    """从 INPUT_TYPES 构造一份默认 kwargs：下拉取首个非占位项，其余取 default。"""
    kw = {}
    spec = cls.INPUT_TYPES()
    for section in ("required", "optional"):
        for name, decl in spec.get(section, {}).items():
            typ = decl[0]
            opts = decl[1] if len(decl) > 1 and isinstance(decl[1], dict) else {}
            kw[name] = _first_real(typ) if isinstance(typ, list) else opts.get("default", "")
    return kw


# 67 个旧键名，用于断言源码中无残留引用
OLD_KEYS = {
    "character_node.py": {
        "00_工作安全策略", "01_身份", "02_年龄与人设", "03_种族", "04_体型",
        "05_肤色", "06_胸部", "07_发型", "08_发色", "09_头发长度", "10_瞳色",
        "11_瞳孔样式", "12_面部特征", "13_表情", "14_套装选择", "15_套装颜色",
        "16_上装款式", "17_上装颜色", "18_下装款式", "19_下装颜色", "20_服装材质",
        "21_腿部特征", "22_袜饰", "23_袜饰颜色", "24_鞋子", "25_鞋子颜色",
        "26_头饰", "27_面饰", "28_颈饰", "29_手部饰品", "30_特殊饰品",
        "31_构图范围", "32_Lora触发词", "33_手动补充",
    },
    "pose_node.py": {
        "00_🎲随机动作模式", "07_🎲随机表情开关", "05_特殊视角", "01_全身基础",
        "02_头部/视线", "03_👐双手动作", "03_👈左手动作", "03_👉右手动作",
        "04_🦵双脚动作", "04_🦵左脚动作", "04_🦵右脚动作", "08_表情细节",
        "06_手动补充", "09_模型类型",
    },
    "scene_node.py": {
        "01_地点选择", "02a_季节", "02b_天气", "02c_时间", "03_光效_整体氛围",
        "04_光效_独立光源", "05_画面画质风格", "06_全局衣物质感", "07_手动补充",
        "08_环境总权重", "09_氛围句",
    },
    "combat_node.py": {
        "00_🔥激战模式", "01_🗡️武器装备", "02_👊攻击姿态", "03_🩸战损程度",
        "04_✨技能特效", "05_📸战斗运镜", "07_模型类型", "06_补充指令",
    },
}


class TestNoNumericPrefix(unittest.TestCase):
    def test_all_keys_have_no_prefix(self):
        for key in FILES:
            bad = [k for k in all_keys(node_class(key)) if PREFIX_RE.match(k)]
            self.assertEqual(bad, [], f"{key} 仍有带前缀的键: {bad}")

    def test_source_has_no_old_key_leftovers(self):
        """源码中不得残留任何旧键名字符串（防漏改 kwargs 引用）"""
        for module_file, old in OLD_KEYS.items():
            src = io.open(os.path.join(NODES_DIR, module_file), encoding="utf-8").read()
            left = sorted(k for k in old if f'"{k}"' in src)
            self.assertEqual(left, [], f"{module_file} 残留旧键名: {left}")

    def test_key_count_unchanged(self):
        """只改名不增删，键数应与改动前一致"""
        expected = {"character": 34, "pose": 14, "scene": 11, "combat": 8}
        for key, n in expected.items():
            self.assertEqual(len(all_keys(node_class(key))), n, f"{key} 键数变化")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **步骤 2：跑测试，确认失败**

Run: `python -X utf8 tests/test_node_widgets.py`
Expected: FAIL——`test_all_keys_have_no_prefix` 报出 67 个带前缀的键。

- [ ] **步骤 3：执行脚本化改名**

键名在代码里始终以带引号的字符串出现（`INPUT_TYPES` 的键与 `kwargs` 的取值都是同一形式），因此**带引号精确替换**可一次覆盖所有引用，且不会误伤子串。

Run:
```bash
python -X utf8 - <<'PY'
import io, re, sys

PREFIX = re.compile(r"^\d+[a-z]?_")

RENAME = {
 "character_node.py": ["00_工作安全策略","01_身份","02_年龄与人设","03_种族","04_体型",
  "05_肤色","06_胸部","07_发型","08_发色","09_头发长度","10_瞳色","11_瞳孔样式",
  "12_面部特征","13_表情","14_套装选择","15_套装颜色","16_上装款式","17_上装颜色",
  "18_下装款式","19_下装颜色","20_服装材质","21_腿部特征","22_袜饰","23_袜饰颜色",
  "24_鞋子","25_鞋子颜色","26_头饰","27_面饰","28_颈饰","29_手部饰品","30_特殊饰品",
  "31_构图范围","32_Lora触发词","33_手动补充"],
 "pose_node.py": ["00_🎲随机动作模式","07_🎲随机表情开关","05_特殊视角","01_全身基础",
  "02_头部/视线","03_👐双手动作","03_👈左手动作","03_👉右手动作","04_🦵双脚动作",
  "04_🦵左脚动作","04_🦵右脚动作","08_表情细节","06_手动补充","09_模型类型"],
 "scene_node.py": ["01_地点选择","02a_季节","02b_天气","02c_时间","03_光效_整体氛围",
  "04_光效_独立光源","05_画面画质风格","06_全局衣物质感","07_手动补充",
  "08_环境总权重","09_氛围句"],
 "combat_node.py": ["00_🔥激战模式","01_🗡️武器装备","02_👊攻击姿态","03_🩸战损程度",
  "04_✨技能特效","05_📸战斗运镜","07_模型类型","06_补充指令"],
}

for fname, keys in RENAME.items():
    path = f"nodes/{fname}"
    src = io.open(path, encoding="utf-8").read()
    total = 0
    for old in keys:
        new = PREFIX.sub("", old)
        needle, repl = f'"{old}"', f'"{new}"'
        n = src.count(needle)
        src = src.replace(needle, repl)
        total += n
        print(f"{fname}: {old} -> {new}  ({n} 处)")
    io.open(path, "w", encoding="utf-8", newline="\n").write(src)
    print(f"{fname}: 共替换 {total} 处\n")
PY
```

Expected: 每个文件打印逐键替换处数；`character_node.py` 约 70+ 处，其余合计约 40 处。某键处数为 0 说明该键只在 `INPUT_TYPES` 里出现一次——正常。

- [ ] **步骤 4：跑测试，确认通过**

Run: `python -X utf8 tests/test_node_widgets.py`
Expected: `OK`，3 个用例全过。

若 `test_source_has_no_old_key_leftovers` 失败，说明某个引用用了单引号 `'09_模型类型'` 或变量拼接。按失败信息定位，手工补齐——**不要改测试**。

- [ ] **步骤 5：跑全量测试，确认无回归**

Run: `python -X utf8 -m unittest discover -s tests -p "test_*.py"`
Expected: `OK`，24 个用例（原 21 + 新 3）。

- [ ] **步骤 6：手工核对界面**

重启 ComfyUI，确认 4 个节点的控件标签不再有数字前缀，**顺序与改前一致**，且能正常出图。

- [ ] **步骤 7：提交**

```bash
git add nodes/character_node.py nodes/pose_node.py nodes/scene_node.py nodes/combat_node.py tests/test_node_widgets.py
git commit -m "refactor: 4 节点控件标签去掉数字前缀（67 个键名 + 全部 kwargs 引用）

键名即控件标签，故必须同步改 kwargs 取值；顺序不变。
新增 tests/test_node_widgets.py 守卫键名与残留引用。

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

## Task 2: 删除 8 个控件 + 输出纯标签

**Files:**
- Modify: `nodes/character_node.py`（删 3 个控件）
- Modify: `nodes/pose_node.py`（删 2 个控件 + 5 处 `is_anima` 收敛）
- Modify: `nodes/scene_node.py`（删 2 个控件 + 权重滑杆移除）
- Modify: `nodes/combat_node.py`（删 2 个控件 + 4 处 `is_anima` 收敛）
- Modify: `tests/test_node_widgets.py`（追加测试）

**Interfaces:**
- Consumes: Task 1 的 `load_node` / `node_class` / `gen_func` / `all_keys` / `default_kwargs` / `NODES_DIR`
- Produces: 4 个节点输出**不含** `:数字)` 权重语法；源码不含 `is_anima`

- [ ] **步骤 1：追加失败测试**

在 `tests/test_node_widgets.py` 的 `if __name__ == "__main__":` 之前插入：

```python
DELETED = {
    "character": {"构图范围", "Lora触发词", "手动补充"},
    "pose": {"手动补充", "模型类型"},
    "scene": {"手动补充", "环境总权重"},
    "combat": {"模型类型", "补充指令"},
}

# 权重语法：形如 (tag:1.2) / (combat mode:2) / (a, b:1.1)
WEIGHT_RE = re.compile(r"\(\s*[^()]*:\s*\d+(?:\.\d+)?\s*\)")


class TestDeletedWidgets(unittest.TestCase):
    def test_deleted_widgets_absent(self):
        for key, gone in DELETED.items():
            keys = set(all_keys(node_class(key)))
            left = sorted(gone & keys)
            self.assertEqual(left, [], f"{key} 仍存在待删控件: {left}")

    def test_default_kwargs_call_does_not_raise(self):
        """用 INPUT_TYPES 默认值调用生成函数，不应抛 KeyError（防漏删引用）"""
        for key in FILES:
            cls, fn = node_class(key), gen_func(key)
            out = fn(cls(), **default_kwargs(cls))
            self.assertIsInstance(out, tuple)
            self.assertEqual(len(out), 1)


class TestPureTagOutput(unittest.TestCase):
    def _run(self, key, **override):
        cls, fn = node_class(key), gen_func(key)
        kw = default_kwargs(cls)
        kw.update(override)
        (out,) = fn(cls(), **kw)
        return out

    def test_no_weight_syntax_in_output(self):
        for key in FILES:
            out = self._run(key)
            self.assertTrue(out, f"{key} 输出为空，断言无意义——检查 default_kwargs 取到的是否为占位项")
            m = WEIGHT_RE.search(out)
            self.assertIsNone(m, f"{key} 输出仍含权重语法: {m.group(0) if m else ''}")

    def test_combat_no_anima_branch_left(self):
        src = io.open(os.path.join(NODES_DIR, "combat_node.py"), encoding="utf-8").read()
        self.assertNotIn("is_anima", src)

    def test_pose_no_anima_branch_left(self):
        src = io.open(os.path.join(NODES_DIR, "pose_node.py"), encoding="utf-8").read()
        self.assertNotIn("is_anima", src)

    def test_pose_underscores_converted_to_spaces(self):
        """统一转空格：手动模式下，全身基础选到带下划线的项时输出不应有下划线"""
        cls = node_class("pose")
        spec = cls.INPUT_TYPES()["required"]
        options = spec["全身基础"][0]
        with_us = [o for o in options if "_" in o]
        if not with_us:
            self.skipTest("动作库中无带下划线的选项")
        out = self._run("pose", **{"🎲随机动作模式": False, "全身基础": with_us[0]})
        self.assertNotIn("_", out, f"下划线未转换: {out!r}")

    def test_scene_output_has_no_weight_wrapper(self):
        """环境总权重删除后，输出不应被 (...) 包裹"""
        out = self._run("scene")
        self.assertTrue(out, "场景输出为空，断言无意义")
        self.assertFalse(out.startswith("(") and out.endswith(")"),
                         f"场景输出仍被权重包裹: {out!r}")

    def test_combat_and_pose_nonempty(self):
        """纯标签化后输出仍应有内容（防误删整段）"""
        for key in ("pose", "combat"):
            out = self._run(key)
            self.assertTrue(out, f"{key} 输出为空")
```

- [ ] **步骤 2：跑测试，确认失败**

Run: `python -X utf8 tests/test_node_widgets.py`
Expected: FAIL——`test_deleted_widgets_absent` 报出 8 个仍存在的控件；`test_*_no_anima_branch_left` 因 `is_anima` 仍在而失败。

- [ ] **步骤 3：改 `character_node.py`**

删除 `INPUT_TYPES` 中这 3 行（原 145-147 行区域）：

```python
                "构图范围": (get_list(COMPOSITION_DATA, "shot_type"),),
                "Lora触发词": ("STRING", {"default": "", "placeholder": "输入Lora触发词 (会自动融合)"}),
                "手动补充": ("STRING", {"default": "", "multiline": True, "placeholder": "在这里输入自定义 Tag..."}),
```

删除变量读取（原 247 行）：`lora_trigger = kwargs.get("Lora触发词", "").strip()`

删除其使用（原 265-266 行）：

```python
        if lora_trigger:
            core_tags.append(lora_trigger)
```

删除 7 段（原 376-385 行）全部内容——`shot_tag` 与 `manual` 的读取与追加：

```python
        # ==========================================
        # 7. 构图与补充
        # ==========================================
        shot_tag = extract(kwargs["构图范围"])
        if shot_tag:
            parts.append(f"({shot_tag}:1.1)")

        manual = kwargs.get("手动补充", "").strip()
        if manual:
            parts.append(manual)
```

若 `COMPOSITION_DATA` 因此在文件中不再被引用，一并删除它的定义与加载（用 `grep -n COMPOSITION_DATA` 确认）。

- [ ] **步骤 4：改 `pose_node.py`**

删除 `INPUT_TYPES` 中这两行：

```python
                "手动补充": ("STRING", {"default": "", "multiline": False, "placeholder": "额外补充 tag..."}),
                "模型类型": (["SDXL", "anima (权重2~5)"],),
```

删除 `is_anima` 定义（166 行）与 `w_global`（167 行）：

```python
        is_anima = "anima" in kwargs.get("模型类型", "SDXL")
        w_global = ":2)" if is_anima else ":1.2)"
```

把 310 行的加权拼接改为纯标签：

```python
        if parts["global"]: final_list.append(parts["global"])   # 核心动作加权
```

291 行改为：

```python
            parts["expression"] = "neutral expression"
```

298-302 行去掉条件，无条件转空格：

```python
        # anima 规范: 标签用空格代替下划线 (score 系列除外, 本节点无 score 标签)
        for k in ("view", "global", "head", "expression"):
            parts[k] = parts[k].replace("_", " ")
        parts["hands"] = [h.replace("_", " ") for h in parts["hands"]]
        parts["legs"] = [l.replace("_", " ") for l in parts["legs"]]
```

324 行改为：

```python
            prompt_string += ", detailed fingers"
```

删除 `manual` 相关（原 297 行注释、304 行）：

```python
        parts["manual"] = kwargs.get("手动补充", "").strip()
```

及其在组装处的追加（原 318 行 `if parts["manual"]: final_list.append(parts["manual"])`）与 `parts` 初始化里的 `"manual": ""` 条目。

最后更新模块 docstring（第 6 行）里「模型类型: SDXL 用 :1.2 轻强调; anima 按官方权重规范 2~5」的描述，改为「输出纯标签，不含权重；下划线统一转空格」。

- [ ] **步骤 5：改 `scene_node.py`**

删除 `INPUT_TYPES` 中这两行：

```python
                "手动补充": ("STRING", {"default": "", "multiline": False, "placeholder": "如: fire, explosion, underwater"}),
                "环境总权重": ("FLOAT", {"default": 1.1, "min": 0.5, "max": 1.6, "step": 0.1, "display": "slider"}),
```

删除变量读取（原 347 行）：`manual = kwargs.get("手动补充", "").strip()`

删除其追加（原 369 行）：`if manual: parts.append(manual)`

把权重段（原 371-384 行）改为：

```python
        content = ", ".join(parts)

        # 纯标签输出（原 08_环境总权重 滑杆已移除）
        if not content:
            return ("",)
        return (content,)
```

最后更新模块 docstring（第 10 行）里「权重滑杆设 1.0 即纯标签输出」的描述，改为「输出纯标签」。

- [ ] **步骤 6：改 `combat_node.py`**

删除 `INPUT_TYPES` 中这两行：

```python
                "模型类型": (["SDXL", "anima (权重2~5)"],),
                "补充指令": ("STRING", {"default": "", "multiline": False, "placeholder": "如: explosion background, glowing eyes"}),
```

删除 `is_anima` 定义（138 行）：

```python
        is_anima = "anima" in kwargs.get("模型类型", "SDXL")
```

172 行改为：

```python
                slot_camera = "motion blur, depth of field, dynamic angle"
```

187-190 行改为：

```python
        ordered_parts.append("combat mode, dynamic pose, intense atmosphere")
```

197 行改为：

```python
        if slot_weapon: ordered_parts.append(slot_weapon)
```

204 行改为：

```python
        if slot_vfx: ordered_parts.append(slot_vfx)
```

删除 `manual_add` 相关（原 210 行）：

```python
        manual_add = kwargs.get("补充指令", "").strip()
```

及其追加处（用 `grep -n manual_add` 找到）。

最后更新模块 docstring（第 7 行）里「模型类型: SDXL 用 :1.2/:1.1 权重习惯; anima 按官方权重规范 2~5」的描述，改为「输出纯标签，不含权重」。

- [ ] **步骤 7：跑测试，确认通过**

Run: `python -X utf8 tests/test_node_widgets.py`
Expected: `OK`，11 个用例全过（Task 1 的 3 个 + 本任务 8 个）。

若 `test_no_weight_syntax_in_output` 在 `character` 上失败，**先确认失败的是不是内置权重**（如 `(rating: safe:1.2)`、`(medium breasts:1.35)`）——那些**不在本次改动范围**，应把断言收窄为只检查本次改动的 3 个节点（`pose`/`scene`/`combat`），而不是去改 character 的内置权重。

- [ ] **步骤 8：跑全量测试**

Run: `python -X utf8 -m unittest discover -s tests -p "test_*.py"`
Expected: `OK`，32 个用例（原 21 + 新增 11）。

- [ ] **步骤 9：手工核对界面与输出**

重启 ComfyUI：
- 4 个节点界面不再有那 8 个控件，其余控件顺序不变
- 肢体动作师输出形如 `spread eagle, looking at viewer`（纯标签、无权重、无下划线）
- 战斗指挥官输出形如 `combat mode, dynamic pose, intense atmosphere, katana, ...`
- 场景光效大师输出不被 `(...)` 包裹
- 角色定制仍保留其内置权重（如 `(medium breasts:1.35)`），仅少了构图/触发词/手动补充三段

- [ ] **步骤 10：提交**

```bash
git add nodes/character_node.py nodes/pose_node.py nodes/scene_node.py nodes/combat_node.py tests/test_node_widgets.py
git commit -m "feat: 4 节点删除 8 个控件，权重改纯标签输出

- 角色定制 删 构图范围/Lora触发词/手动补充
- 肢体动作师 删 手动补充/模型类型；5 处 is_anima 收敛为纯标签，
  下划线统一转空格（原 anima 行为）
- 场景光效大师 删 手动补充/环境总权重；恒输出纯标签
- 战斗指挥官 删 模型类型/补充指令；4 处 is_anima 收敛为纯标签
- character 的内置权重不动

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

## Task 3: 文档同步

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: Task 1、2 的最终界面与输出形态
- Produces: 与代码一致的 README

- [ ] **步骤 1：更新节点列表描述**

README「节点列表」表中，这几行的描述需要改（`grep -n "模型类型\|权重滑杆" README.md` 定位）：

| 节点 | 描述中的问题片段 | 改为 |
|---|---|---|
| 千绘 · 肢体动作师 | 「模型类型切换 SDXL（`:1.2` 轻强调）/ anima（权重 2 起步、标签空格化，符合 anima 官方规范）」 | 「输出纯标签（不加权重），标签下划线统一转空格」 |
| 千绘 · 战斗指挥官 | 「模型类型切换 SDXL / anima（权重 2 起步且最多 2 个）」 | 「输出纯标签（不加权重）」 |
| 千绘 · 场景光效大师 | 「权重滑杆全局加权（1.0 纯标签）」 | 「输出纯标签（原权重滑杆已移除）」 |
| 千绘 · 角色定制 | 若提到「构图范围 / Lora 触发词 / 手动补充」 | 删除相应描述 |

- [ ] **步骤 2：补一条迁移说明**

「注意事项」补：

```markdown
- 4 个提示词节点（角色定制/肢体动作师/场景光效大师/战斗指挥官）的控件标签已去掉数字前缀，
  并移除了部分控件（各节点的「手动补充」、肢体与战斗的「模型类型」、场景的「环境总权重」、
  角色的「构图范围/Lora触发词」）。节点注册名未变，但旧 workflow 的控件值可能因控件减少而错位，
  建议重新检查一遍参数。
- 肢体动作师、战斗指挥官、场景光效大师改为**纯标签输出**（不含 `(tag:1.2)` 权重）。
  需要加权重时，在后续的提示词节点里自行书写。
```

- [ ] **步骤 3：提交**

```bash
git add README.md
git commit -m "docs: README 同步 4 节点控件精简（去数字前缀、删控件、纯标签输出）

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

## 完成标准

- [ ] `python -X utf8 -m unittest discover -s tests -p "test_*.py"` 全绿，用例数 ≥ 32
- [ ] 4 个节点的 `INPUT_TYPES` 键名无数字前缀（正则 `^\d+[a-z]?_` 无匹配）
- [ ] 8 个被删控件在界面上消失，其余控件顺序不变
- [ ] `pose_node.py` / `combat_node.py` 源码不含 `is_anima`
- [ ] `pose`/`scene`/`combat` 输出不含 `:数字)` 权重语法
- [ ] `character_node.py` 的内置权重（`:1.2`/`:1.35`/`:1.1`）保持不变
- [ ] 节点注册名与 class 名未变（旧 workflow 的节点仍能识别）
