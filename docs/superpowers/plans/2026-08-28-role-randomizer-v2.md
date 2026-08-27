# 千绘 · 角色随机师 V2 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给角色随机师节点增加分类筛选（女/男/兽类/机械）与手动指定（先选作品→角色联动→指定角色或作品内随机）。

**Architecture:** 纯逻辑层在 `nodes/role_name_node.py`（仅标准库，可脱离 ComfyUI 测试）；新增两个数据文件（角色分类标签、作品中文名）；`__init__.py` 注册 WEB_DIRECTORY 与 `/qianhui_role_data` 数据路由；前端 `js/role_random.js` 实现作品→角色下拉联动，JS 失败时后端退化为全量静态列表。

**Tech Stack:** Python 3（标准库 + unittest）、ComfyUI 前端扩展（JS）、git。

## Global Constraints

- `nodes/role_name_node.py` 只允许标准库 import（json/os/random/re/time），保持可独立测试
- 数据文件编码统一 UTF-8（无 BOM）；`role_category.json` 值集合仅限 `{female, male, beast, robot}`；`work_cn.json` 键 = 版权 tag
- 哨兵值固定为 `（未指定）`（全角括号）；"抽取分类"可选值固定 `不限/女/男/兽类·福瑞/机械·人造人/兽娘/机械娘`
- 作品类型可选值：`🎲随机类型/🎮只出游戏/📺只出动漫/🎯指定作品`（沿用现有单 emoji 风格，不加新表情）
- 展示格式：作品 = `中文名 (tag)`，角色 = `中文名 (danbooru名)`；中文名缺失回退英文原名
- 输出：`("角色名(英文)", "角色名(中文)")`，转义规则与 V1 完全一致；空数据/空池返回 `("(无角色数据)"/"(无匹配角色)", 同值)` 双元组（修复 V1 单元组隐患）
- 抽取优先级（仅模式=🎯指定作品 时读取指定作品/指定角色）：
  1. 角色已指定且属于所选作品 → 直接输出（跳过分类筛选）；角色已指定但不在所选作品内 → 按作品内随机；作品未指定 → 直接输出该角色
  2. 角色未指定、作品已指定 → 作品内随机（受抽取分类约束）；过滤后空 → `(无匹配角色)`
  3. 都未指定 → 回退 🎲随机类型 逻辑（含分类约束）
- 其他模式（非🎯指定作品）：忽略指定作品/指定角色值（残留值不触发输出）
- **测试断言注意**：所有角色英文输出经 `escape_name` 转换，`hatsune_miku` → `hatsune miku`（下划线变空格），断言必须用转义后形式（正则 `r"^hatsune miku\b"` 等）
- 测试运行目录均为插件根目录：`python -X utf8 -m unittest discover -s tests -p "test_*.py" -v`
- 测试加载模块方式：`importlib.util.spec_from_file_location` 按文件路径加载（nodes/ 无 __init__.py，包导入不可靠）

---

### Task 1: 生成数据文件（role_category.json + work_cn.json）+ 质检脚本与测试

**Files:**
- Create: `data/role_category.json`（角色→标签）
- Create: `data/work_cn.json`（作品tag→中文名）
- Create: `tools/check_role_data.py`（质检脚本）
- Create: `tests/test_role_data.py`

**Interfaces:**
- Produces: `data/role_category.json` — `{"danbooru名": ["female"]}`，键集 == role_owner.json 键集（1629 条）；标签 ∈ `{female, male, beast, robot}`；未知角色值为 `[]`。`data/work_cn.json` — `{"版权tag": "中文名"}`，覆盖 445 个真实作品 tag 的 ≥90%

**标注规则（生成数据时必须遵守）：**
- female/male = 明确性别；伪娘归 male；扶他归 female；无性别生物只标 beast 或留空
- beast = 兽类外形：兽娘、兽人/福瑞、龙娘、狐/猫/犬/狼/兔/牛/羊/鼠娘、原型为动物的角色；长兽耳/兽尾而主体人形的（猫娘、狐娘）算 beast
- robot = 机械娘、人造人(android)、赛博格；武装/防具附带机械零件不算
- 可叠加：兽娘 = `["female","beast"]`；机械娘 = `["female","robot"]`
- 不认识/无法确定的留 `[]`，切忌乱标

- [ ] **Step 1: 写质检脚本 `tools/check_role_data.py`**

```python
# -*- coding: utf-8 -*-
"""role_category.json / work_cn.json 质量检查"""
import collections, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
ALLOWED = {"female", "male", "beast", "robot"}

def main():
    owner = json.load(open(os.path.join(DATA, "role_owner.json"), encoding="utf-8"))
    names = list(owner.keys())

    try:
        cat = json.load(open(os.path.join(DATA, "role_category.json"), encoding="utf-8"))
    except FileNotFoundError:
        print("FAIL: role_category.json 不存在"); return 1
    bad_keys = set(cat) - set(owner)
    bad_vals = {k: v for k, v in cat.items() if not set(v) <= ALLOWED}
    labeled = sum(1 for k in names if cat.get(k))
    counts = collections.Counter()
    for v in cat.values():
        counts.update(v)
    print(f"角色总数: {len(names)}  标注条目: {len(cat)}")
    print(f"无效键: {len(bad_keys)}  无效标签值: {len(bad_vals)}")
    print(f"有标签角色: {labeled} ({labeled/len(names)*100:.1f}%)")
    print(f"分布: 女 {counts['female']}  男 {counts['male']}  兽类 {counts['beast']}  机械 {counts['robot']}")
    print(f"未标注(未知): {len(names) - labeled}")

    try:
        wcn = json.load(open(os.path.join(DATA, "work_cn.json"), encoding="utf-8"))
    except FileNotFoundError:
        print("FAIL: work_cn.json 不存在"); return 1
    bad_wkeys = set(wcn) - set(owner.values())
    print(f"作品数: {len(set(owner.values()))}  work_cn 条目: {len(wcn)}  无效键: {len(bad_wkeys)}")

    ok = not (bad_keys or bad_vals or bad_wkeys or labeled < len(names) * 0.85)
    print("OK 合格" if ok else "FAIL 存在不合格项")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: 写测试 `tests/test_role_data.py`**

```python
# -*- coding: utf-8 -*-
"""数据文件结构校验（unittest，纯标准库）"""
import json, os, unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALLOWED = {"female", "male", "beast", "robot"}

class TestRoleData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.owner = json.load(open(os.path.join(ROOT, "data", "role_owner.json"), encoding="utf-8"))

    def test_category_keys_and_values(self):
        cat = json.load(open(os.path.join(ROOT, "data", "role_category.json"), encoding="utf-8"))
        self.assertEqual(set(cat), set(self.owner))  # 全量覆盖
        for v in cat.values():
            self.assertIsInstance(v, list)
            self.assertTrue(set(v) <= ALLOWED, f"非法标签: {v}")

    def test_category_coverage(self):
        cat = json.load(open(os.path.join(ROOT, "data", "role_category.json"), encoding="utf-8"))
        labeled = sum(1 for k in self.owner if cat.get(k))
        self.assertGreaterEqual(labeled, len(self.owner) * 0.85, "有标签覆盖率不足 85%")

    def test_gender_ratio_sane(self):
        cat = json.load(open(os.path.join(ROOT, "data", "role_category.json"), encoding="utf-8"))
        fem = sum(1 for v in cat.values() if "female" in v)
        mal = sum(1 for v in cat.values() if "male" in v)
        self.assertGreater(fem, mal)
        self.assertGreaterEqual(fem, 800)

    def test_beast_robot_counts(self):
        cat = json.load(open(os.path.join(ROOT, "data", "role_category.json"), encoding="utf-8"))
        self.assertGreaterEqual(sum(1 for v in cat.values() if "beast" in v), 100)
        self.assertGreaterEqual(sum(1 for v in cat.values() if "robot" in v), 30)

    def test_work_cn(self):
        wcn = json.load(open(os.path.join(ROOT, "data", "work_cn.json"), encoding="utf-8"))
        self.assertLessEqual(set(wcn), set(self.owner.values()))  # 键必须是真实作品 tag
        self.assertGreaterEqual(len(wcn), 0.9 * len(set(self.owner.values())))

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: 先跑确认失败**

Run: `cd "E:\comfuyui 插件\comfyui-qianhui" && python -X utf8 -m unittest tests.test_role_data -v`
Expected: FAIL（两个数据文件尚不存在）

- [ ] **Step 4: 生成 role_category.json**

过程：读取 `data/role_names.json` 全部 1629 角色（danbooru 名）＋ `data/role_names_cn.json` 中文名，按上文"标注规则"逐条标注（基于知识库判定，冷门不认识的留 `[]`），写入 `data/role_category.json`（UTF-8，`ensure_ascii=False`，键按读取顺序，值按 `female/male/beast/robot` 固定次序）。

- [ ] **Step 5: 生成 work_cn.json**

过程：收集 `data/role_owner.json` 全部作品 tag（445 个），按通行中文译名逐条标注（`genshin_impact → 原神` 等；无通行译名用通俗译名），写入 `data/work_cn.json`（UTF-8，`ensure_ascii=False`）。

- [ ] **Step 6: 质检 + 测试**

Run: `cd "E:\comfuyui 插件\comfyui-qianhui" && python -X utf8 tools\check_role_data.py && python -X utf8 -m unittest tests.test_role_data -v`
Expected: 输出 `OK 合格`；测试全 PASS。不合格则补标/修正后重跑。

- [ ] **Step 7: 提交**

```bash
git add data/role_category.json data/work_cn.json tools/check_role_data.py tests/test_role_data.py
git commit -m "feat: 角色分类标签+作品中文名数据、质检脚本与测试"
```

---

### Task 2: role_name_node.py 数据层（分类/中文名加载、展示与解析、过滤助手）

**Files:**
- Modify: `nodes/role_name_node.py`（在 `load_cn_map` 之后追加新常量与函数，不动 `load_role_data/_type_of/escape_name/load_cn_map`）
- Create: `tests/test_role_node_helpers.py`

**Interfaces:**
- Consumes: V1 的 `DATA_DIR`/`ROLE_OWNER_PATH`/`ROLE_NAMES_PATH`/`ROLE_CN_PATH`、`escape_name`
- Produces（供 Task 3/4/5 用）：
  - `SENTINEL = "（未指定）"`
  - `CATEGORY_TAGS = {"不限": None, "女": ("female",), "男": ("male",), "兽类·福瑞": ("beast",), "机械·人造人": ("robot",), "兽娘": ("female","beast"), "机械娘": ("female","robot")}`
  - `CATEGORY_PATH = os.path.join(DATA_DIR, "role_category.json")`、`WORK_CN_PATH = os.path.join(DATA_DIR, "work_cn.json")`
  - `load_category_map() -> dict`（角色→set，缓存，文件缺失返回 {}）
  - `load_work_cn() -> dict`（缓存，缺失返回 {}）
  - `load_owner_map() -> dict`（角色→版权 tag，缓存，缺失返回 {}）
  - `load_heat() -> dict`（角色→post_count，缓存）
  - `work_display(tag) -> str`、`role_display(name) -> str`
  - `parse_work_display(v) -> str|None`、`parse_role_display(v) -> str|None`（哨兵→None；取最后一个 `" ("` 拆分，无则整串）
  - `filter_by_category(names, cat_key, cats=None) -> list`（"不限"原样返回；否则要求所需标签全命中；cats 可注入便于测试）

- [ ] **Step 1: 写失败测试 `tests/test_role_node_helpers.py`**

```python
# -*- coding: utf-8 -*-
"""role_name_node.py 数据层助手单测（按文件路径加载）"""
import importlib.util, os, unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    "role_name_node", os.path.join(ROOT, "nodes", "role_name_node.py"))
rnn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rnn)

class TestHelpers(unittest.TestCase):
    def test_sentinel_and_category_tags(self):
        self.assertEqual(rnn.SENTINEL, "（未指定）")
        self.assertEqual(rnn.CATEGORY_TAGS["兽娘"], ("female", "beast"))
        self.assertIsNone(rnn.CATEGORY_TAGS["不限"])

    def test_load_maps_have_data(self):
        cat = rnn.load_category_map()
        self.assertIn("female", cat.get("hatsune_miku", []))   # 初音未来必须是女性
        wcn = rnn.load_work_cn()
        self.assertEqual(wcn.get("genshin_impact"), "原神")

    def test_display_roundtrip_all_roles(self):
        """全量 角色/作品 展示->解析往返一致（关键正确性网）"""
        for name, cop in rnn.load_owner_map().items():
            self.assertEqual(cop, rnn.parse_work_display(rnn.work_display(cop)), cop)
            disp = rnn.role_display(name)
            self.assertEqual(name, rnn.parse_role_display(disp), disp)

    def test_display_parse(self):
        self.assertIsNone(rnn.parse_work_display(rnn.SENTINEL))
        self.assertIsNone(rnn.parse_role_display(rnn.SENTINEL))
        self.assertEqual(rnn.parse_work_display(rnn.work_display("genshin_impact")), "genshin_impact")
        # 含消歧义括号的角色名
        self.assertEqual(rnn.parse_role_display("阿尔托莉雅 (artoria_pendragon_(fate))"),
                         "artoria_pendragon_(fate)")
        # 无中文名回退纯英文（无 " (" 时整串原样）
        self.assertEqual(rnn.parse_role_display("hatsune_miku"), "hatsune_miku")

    def test_filter_by_category(self):
        names = ["a", "b", "c", "d"]
        cats = {"a": {"female"}, "b": {"male", "beast"}, "c": {"female", "beast"}, "d": set()}
        self.assertEqual(rnn.filter_by_category(names, "不限", cats), ["a", "b", "c", "d"])
        self.assertEqual(rnn.filter_by_category(names, "女", cats), ["a", "c"])
        self.assertEqual(rnn.filter_by_category(names, "兽类·福瑞", cats), ["b", "c"])
        self.assertEqual(rnn.filter_by_category(names, "兽娘", cats), ["c"])
        self.assertEqual(rnn.filter_by_category(names, "机械娘", cats), [])

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd "E:\comfuyui 插件\comfyui-qianhui" && python -X utf8 -m unittest tests.test_role_node_helpers -v`
Expected: FAIL（AttributeError: module has no attribute 'SENTINEL' 等）

- [ ] **Step 3: 实现数据层**（追加在 `load_cn_map` 之后）

```python
CATEGORY_PATH = os.path.join(DATA_DIR, "role_category.json")
WORK_CN_PATH = os.path.join(DATA_DIR, "work_cn.json")

SENTINEL = "（未指定）"

# 抽取分类 -> 所需角色标签（"不限"不过滤）；"兽娘"=女+兽类, "机械娘"=女+机械
CATEGORY_TAGS = {
    "不限": None,
    "女": ("female",),
    "男": ("male",),
    "兽类·福瑞": ("beast",),
    "机械·人造人": ("robot",),
    "兽娘": ("female", "beast"),
    "机械娘": ("female", "robot"),
}

_CAT_CACHE = None
_WORK_CN_CACHE = None
_OWNER_CACHE = None
_HEAT_CACHE = None

def load_category_map():
    """角色名 -> 标签 set; 文件缺失返回 {} (模块级缓存)"""
    global _CAT_CACHE
    if _CAT_CACHE is None:
        try:
            if os.path.exists(CATEGORY_PATH):
                with open(CATEGORY_PATH, "r", encoding="utf-8") as f:
                    _CAT_CACHE = {k: set(v) for k, v in json.load(f).items()}
            else:
                _CAT_CACHE = {}
        except Exception:
            _CAT_CACHE = {}
    return _CAT_CACHE

def load_work_cn():
    """作品 tag -> 中文名; 文件缺失返回 {} (模块级缓存)"""
    global _WORK_CN_CACHE
    if _WORK_CN_CACHE is None:
        try:
            if os.path.exists(WORK_CN_PATH):
                with open(WORK_CN_PATH, "r", encoding="utf-8") as f:
                    _WORK_CN_CACHE = json.load(f)
            else:
                _WORK_CN_CACHE = {}
        except Exception:
            _WORK_CN_CACHE = {}
    return _WORK_CN_CACHE

def load_owner_map():
    """角色名 -> 版权 tag (模块级缓存)"""
    global _OWNER_CACHE
    if _OWNER_CACHE is None:
        try:
            with open(ROLE_OWNER_PATH, "r", encoding="utf-8") as f:
                _OWNER_CACHE = json.load(f)
        except Exception:
            _OWNER_CACHE = {}
    return _OWNER_CACHE

def load_heat():
    """角色名 -> post_count (模块级缓存, 用于热门排序)"""
    global _HEAT_CACHE
    if _HEAT_CACHE is None:
        try:
            with open(ROLE_NAMES_PATH, "r", encoding="utf-8") as f:
                _HEAT_CACHE = {r["name"]: r["post_count"]
                               for r in json.load(f).get("role_names", [])}
        except Exception:
            _HEAT_CACHE = {}
    return _HEAT_CACHE

def work_display(tag):
    """作品下拉展示: 中文名 (tag); 无中文名回退 tag"""
    cn = load_work_cn().get(tag)
    return f"{cn} ({tag})" if cn else tag

def role_display(name):
    """角色下拉展示: 中文名 (danbooru名); 无中文名回退原名"""
    cn = load_cn_map().get(name, "")
    return f"{cn} ({name})" if cn else name

def parse_work_display(v):
    """作品展示值 -> tag; SENTINEL/空 -> None（取最后一个 ' (' 拆分）"""
    if v in (None, SENTINEL):
        return None
    i = v.rfind(" (")
    return v[i + 2:].strip() if i != -1 else v.strip()

def parse_role_display(v):
    """角色展示值 -> danbooru 名; SENTINEL/空 -> None"""
    if v in (None, SENTINEL):
        return None
    i = v.rfind(" (")
    return v[i + 2:].strip() if i != -1 else v.strip()

def filter_by_category(names, cat_key, cats=None):
    """按抽取分类过滤: 不限 返回原列表; 否则要求所需标签全命中"""
    cats = cats if cats is not None else load_category_map()
    need = CATEGORY_TAGS.get(cat_key)
    if not need:
        return list(names)
    need = set(need)
    return [n for n in names if need.issubset(cats.get(n, set()))]
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd "E:\comfuyui 插件\comfyui-qianhui" && python -X utf8 -m unittest tests.test_role_node_helpers -v`
Expected: PASS（含全量 roundtrip 1629 条不抛错）

- [ ] **Step 5: 提交**

```bash
git add nodes/role_name_node.py tests/test_role_node_helpers.py
git commit -m "feat: 角色随机师数据层（分类/作品中文名/展示解析/过滤）"
```

---

### Task 3: 改写 generate_role_name + INPUT_TYPES（随机逻辑 + 手动模式）

**Files:**
- Modify: `nodes/role_name_node.py`（替换 `QianhuiRandomCharacter` 类体，保留 `escape_name` 静态方法原样）
- Create: `tests/test_role_logic.py`

**Interfaces:**
- Consumes: Task 2 全部助手；V1 的 `load_role_data()`/`_type_of()`/`escape_name()`
- Produces: `INPUT_TYPES`（5 下拉：作品类型/抽取分类/模型类型/指定作品/指定角色）；`generate_role_name(作品类型, 抽取分类="不限", 模型类型="SDXL", 指定作品=SENTINEL, 指定角色=SENTINEL) -> (str, str)`；`_output(role, cop, 模型类型) -> (str, str)`

**测试数据池（本任务用例统一使用）：**
```python
GROUPS = {
    "genshin_impact": ["hu_tao_(genshin_impact)", "mona_(genshin_impact)", "venti_(genshin_impact)"],
    "vocaloid": ["hatsune_miku", "kaito_(vocaloid)"],
    "dandadan": ["momoko_ayase"],
}
CATS = {
    "hu_tao_(genshin_impact)": {"female"}, "mona_(genshin_impact)": {"female"},
    "venti_(genshin_impact)": {"male"}, "hatsune_miku": {"female"},
    "kaito_(vocaloid)": {"male"}, "momoko_ayase": {"female"},
}
```
注意：英文输出经 `escape_name` 后：`hu_tao_(genshin_impact)` → `hu tao (genshin_impact)`；`hatsune_miku` → `hatsune miku`（SDXL 下划线转空格）。断言一律用转义后前缀，见 Global Constraints。

- [ ] **Step 1: 写失败测试 `tests/test_role_logic.py`**

```python
# -*- coding: utf-8 -*-
"""generate_role_name 逻辑单测（mock 依赖函数，无真实数据随机性）"""
import importlib.util, os, unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    "role_name_node", os.path.join(ROOT, "nodes", "role_name_node.py"))
rnn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rnn)

GROUPS = {
    "genshin_impact": ["hu_tao_(genshin_impact)", "mona_(genshin_impact)", "venti_(genshin_impact)"],
    "vocaloid": ["hatsune_miku", "kaito_(vocaloid)"],
    "dandadan": ["momoko_ayase"],
}
CATS = {
    "hu_tao_(genshin_impact)": {"female"},
    "mona_(genshin_impact)": {"female"},
    "venti_(genshin_impact)": {"male"},
    "hatsune_miku": {"female"},
    "kaito_(vocaloid)": {"male"},
    "momoko_ayase": {"female"},
}
OWNER = {}
for _cop, _rs in GROUPS.items():
    for _r in _rs:
        OWNER[_r] = _cop
CN = {"hu_tao_(genshin_impact)": "胡桃", "hatsune_miku": "初音未来", "momoko_ayase": "莫摩"}

class TestLogic(unittest.TestCase):
    def setUp(self):
        p = mock.patch.object(rnn, "load_role_data", return_value=GROUPS)
        p.start(); self.addCleanup(p.stop)
        p = mock.patch.object(rnn, "load_category_map", return_value=CATS)
        p.start(); self.addCleanup(p.stop)
        p = mock.patch.object(rnn, "load_owner_map", return_value=OWNER)
        p.start(); self.addCleanup(p.stop)
        p = mock.patch.object(rnn, "load_cn_map", return_value=CN)
        p.start(); self.addCleanup(p.stop)

    def _node(self):
        return rnn.QianhuiRandomCharacter()

    def test_empty_data(self):
        with mock.patch.object(rnn, "load_role_data", return_value={}):
            self.assertEqual(self._node().generate_role_name("🎲随机类型"),
                             ("(无角色数据)", "(无角色数据)"))

    def test_manual_role_skips_filter(self):
        # 指定角色跳过分类筛选: 选男性类目仍能出初音
        en, cn = self._node().generate_role_name(
            "🎯指定作品", 抽取分类="男",
            指定角色=rnn.role_display("hatsune_miku"))
        self.assertRegex(en, r"^hatsune miku")

    def test_manual_role_emits(self):
        en, cn = self._node().generate_role_name(
            "🎯指定作品", 指定作品=rnn.work_display("genshin_impact"),
            指定角色=rnn.role_display("venti_(genshin_impact)"))
        self.assertRegex(en, r"^venti")

    def test_manual_role_not_in_work_falls_to_work_random(self):
        en, cn = self._node().generate_role_name(
            "🎯指定作品", 指定作品=rnn.work_display("genshin_impact"),
            指定角色=rnn.role_display("hatsune_miku"))   # 初音不属于原神
        self.assertRegex(en, r"^(hu tao|mona|venti)\b")  # 变成原神内随机

    def test_work_random_respects_filter(self):
        en, cn = self._node().generate_role_name(
            "🎯指定作品", 抽取分类="女", 指定作品=rnn.work_display("genshin_impact"))
        self.assertRegex(en, r"^(hu tao|mona)\b")        # 排除温迪(男)

    def test_work_random_empty_after_filter(self):
        en, cn = self._node().generate_role_name(
            "🎯指定作品", 抽取分类="机械娘", 指定作品=rnn.work_display("vocaloid"))
        self.assertEqual(en, "(无匹配角色)")

    def test_fallback_when_nothing_specified(self):
        en, cn = self._node().generate_role_name("🎯指定作品", 抽取分类="不限")
        # 回退随机: 输出必是池内角色（转义后前缀匹配）
        self.assertRegex(en, r"^(hu tao|mona|venti|hatsune miku|kaito|momoko)\b")

    def test_random_mode_ignores_leftovers(self):
        en, cn = self._node().generate_role_name(
            "📺只出动漫", 指定角色=rnn.role_display("hatsune_miku"))  # 残留值
        self.assertRegex(en, r"^momoko")  # 只出动漫 = dandadan 的莫摩

    def test_filter_in_game_mode(self):
        en, cn = self._node().generate_role_name("🎮只出游戏", 抽取分类="男")
        self.assertRegex(en, r"^(venti|kaito)\b")

    def test_escape_unchanged(self):
        e = rnn.QianhuiRandomCharacter.escape_name
        self.assertEqual(e("hu_tao_(genshin_impact)", "SDXL"), "hu tao (genshin_impact)")
        self.assertEqual(e("hu_tao_(genshin_impact)", "anima (括号转义)"),
                         "hu tao \\(genshin_impact\\)")

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd "E:\comfuyui 插件\comfyui-qianhui" && python -X utf8 -m unittest tests.test_role_logic -v`
Expected: FAIL（TypeError: generate_role_name() 不接收新参数）

- [ ] **Step 3: 实现**（替换类体——`escape_name` 静态方法原样保留，另加 `_output` 方法）

```python
    @classmethod
    def INPUT_TYPES(s):
        works = sorted(set(load_owner_map().values()))
        work_opts = [SENTINEL] + [work_display(w) for w in works]
        names = sorted(load_owner_map().keys(),
                       key=lambda n: (0 - load_heat().get(n, 0), n))
        role_opts = [SENTINEL] + [role_display(n) for n in names]
        return {
            "required": {
                "作品类型": (["🎲随机类型", "🎮只出游戏", "📺只出动漫", "🎯指定作品"],),
                "抽取分类": (list(CATEGORY_TAGS.keys()),),
                "模型类型": (["SDXL", "anima (括号转义)"],),
                "指定作品": (work_opts,),
                "指定角色": (role_opts,),
            },
        }

    def generate_role_name(self, 作品类型, 抽取分类="不限", 模型类型="SDXL",
                           指定作品=SENTINEL, 指定角色=SENTINEL):
        groups = load_role_data()
        if not groups:
            return ("(无角色数据)", "(无角色数据)")
        cats = load_category_map()

        if 作品类型 == "🎯指定作品":
            role_name = parse_role_display(指定角色)
            work_tag = parse_work_display(指定作品)
            if role_name:
                if work_tag:
                    if role_name in groups.get(work_tag, []):
                        return self._output(role_name, work_tag, 模型类型)
                    role_name = None   # 角色不在所选作品 -> 走作品内随机
                else:
                    return self._output(role_name, None, 模型类型)
            if work_tag and work_tag in groups:
                pool = filter_by_category(groups[work_tag], 抽取分类, cats)
                if not pool:
                    return ("(无匹配角色)", "(无匹配角色)")
                return self._output(random.choice(pool), work_tag, 模型类型)
            作品类型 = "🎲随机类型"   # 都未指定 -> 回退随机

        # 按类型聚合（V1 加权随机 ≈ 类型内角色均匀抽取）
        by_type = {"game": {}, "anime": {}, "other": {}}
        for cop, roles in groups.items():
            by_type[_type_of(cop)][cop] = roles

        if 作品类型 == "🎲随机类型":
            weights = {t: sum(len(r) for r in pool.values()) for t, pool in by_type.items()}
            total = sum(weights.values())
            if total == 0:
                return ("(无角色数据)", "(无角色数据)")
            roll = random.random() * total
            acc = 0
            chosen_type = "other"
            for t, w in weights.items():
                acc += w
                if roll < acc:
                    chosen_type = t
                    break
        else:
            chosen_type = "game" if 作品类型 == "🎮只出游戏" else "anime"

        pool = by_type.get(chosen_type, {})
        names = [n for rs in pool.values() for n in rs]
        names = filter_by_category(names, 抽取分类, cats)
        if not names:
            return ("(无匹配角色)", "(无匹配角色)")
        return self._output(random.choice(names), None, 模型类型)

    def _output(self, role, cop, 模型类型):
        """统一输出: (英文提示词名, 中文显示名)；复刻 V1 格式。
        cop 未知时从 owner 查询; 无括号角色追加 (版权tag) 消歧义并同样转义。"""
        en = self.escape_name(role, 模型类型)
        cop = cop or load_owner_map().get(role, "")
        if "(" not in role and cop:
            suffix = f"({cop})"
            if "anima" in 模型类型:
                suffix = suffix.replace("(", r"\(").replace(")", r"\)")
            en = en + " " + suffix
        cn = load_cn_map().get(role, "")
        cn_display = f"{cn}({cop})" if cn else en
        return (en, cn_display)
```

（`IS_CHANGED(s, **kwargs): return time.time()`、`FUNCTION`/`RETURN_TYPES`/`RETURN_NAMES`/`CATEGORY` 均保持原值。）

- [ ] **Step 4: 跑测试确认通过**

Run: `cd "E:\comfuyui 插件\comfyui-qianhui" && python -X utf8 -m unittest discover -s tests -p "test_*.py" -v`
Expected: 全部 PASS

- [ ] **Step 5: 真实数据冒烟（依赖 Task 1 数据）**

Run: `cd "E:\comfuyui 插件\comfyui-qianhui" && python -X utf8 -c "import importlib.util; s=importlib.util.spec_from_file_location('r','nodes/role_name_node.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); n=m.QianhuiRandomCharacter(); print(n.generate_role_name('🎯指定作品', 抽取分类='女', 指定作品=m.work_display('genshin_impact'))); print(n.generate_role_name('🎲随机类型', 抽取分类='兽娘')); print(n.generate_role_name('🎮只出游戏', 抽取分类='机械娘'))"`
Expected: 每行输出 2 元组；原神+女 必为女性角色；无异常

- [ ] **Step 6: 提交**

```bash
git add nodes/role_name_node.py tests/test_role_logic.py
git commit -m "feat: 角色随机师随机逻辑+手动指定模式（V2 核心）"
```

---

### Task 4: build_payload（前端联动数据）

**Files:**
- Modify: `nodes/role_name_node.py`（文件末尾追加）
- Test: `tests/test_role_logic.py` 追加 `TestPayload`

**Interfaces:**
- Consumes: load_owner_map/load_cn_map/load_work_cn/load_heat
- Produces: `build_payload() -> dict`：`{"works": [[tag, cn_or_null], ...], "roles_by_work": {tag: [[danbooru名, cn_or_null], ...]}}`（works 按 tag 升序，roles 按 post_count 降序）

- [ ] **Step 1: 追加测试（`tests/test_role_logic.py` 末尾）**

```python
class TestPayload(unittest.TestCase):
    def setUp(self):
        p = mock.patch.object(rnn, "load_owner_map", return_value=OWNER)
        p.start(); self.addCleanup(p.stop)
        p = mock.patch.object(rnn, "load_cn_map", return_value=CN)
        p.start(); self.addCleanup(p.stop)
        p = mock.patch.object(rnn, "load_work_cn", return_value={"genshin_impact": "原神"})
        p.start(); self.addCleanup(p.stop)
        p = mock.patch.object(rnn, "load_heat",
                              return_value={"hu_tao_(genshin_impact)": 5000,
                                            "mona_(genshin_impact)": 3000,
                                            "venti_(genshin_impact)": 4000,
                                            "hatsune_miku": 9999, "kaito_(vocaloid)": 1,
                                            "momoko_ayase": 100})
        p.start(); self.addCleanup(p.stop)

    def test_payload_shape(self):
        p = rnn.build_payload()
        self.assertEqual(len(p["works"]), 3)
        wmap = {w[0]: w[1] for w in p["works"]}
        self.assertEqual(wmap["genshin_impact"], "原神")
        self.assertIsNone(wmap["vocaloid"])   # 未提供中文名 -> null
        self.assertEqual(len(p["roles_by_work"]["genshin_impact"]), 3)
        names = [r[0] for r in p["roles_by_work"]["genshin_impact"]]
        self.assertEqual(names, ["hu_tao_(genshin_impact)", "venti_(genshin_impact)",
                                 "mona_(genshin_impact)"])   # 按热度降序

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 跑测试确认失败**（AttributeError: 无 build_payload）

- [ ] **Step 3: 实现**（文件末尾追加）

```python
def build_payload():
    """前端联动数据: 作品列表 + 每作品角色列表（均含中文名, 无则 null）"""
    owner = load_owner_map()
    cn = load_cn_map()
    wcn = load_work_cn()
    works = sorted(set(owner.values()))
    roles_by_work = {}
    for tag in works:
        rs = [n for n, w in owner.items() if w == tag]
        rs.sort(key=lambda n: (0 - load_heat().get(n, 0), n))
        roles_by_work[tag] = [[n, cn.get(n) or None] for n in rs]
    return {"works": [[t, wcn.get(t)] for t in works],
            "roles_by_work": roles_by_work}
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd "E:\comfuyui 插件\comfyui-qianhui" && python -X utf8 -m unittest tests.test_role_logic -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add nodes/role_name_node.py && git commit -m "feat: build_payload 前端联动数据接口"
```

---

### Task 5: __init__.py——WEB_DIRECTORY + /qianhui_role_data 路由 + V2 显示名

**Files:**
- Modify: `__init__.py`（文件尾部、`__all__` 之前加路由块；第 8 条显示名改 `"千绘 · 角色随机师 (V2)"`）

**Interfaces:**
- Consumes: `role_name_node.build_payload`（Task 4）
- Produces: `GET /qianhui_role_data` → JSON；`WEB_DIRECTORY = "./js"`

- [ ] **Step 1: 修改 __init__.py**

`__all__ = [...]` 之前追加：

```python
# 8b. 前端联动数据路由（注册失败不影响其余节点）
WEB_DIRECTORY = "./js"
try:
    from aiohttp import web
    from server import PromptServer
    from .nodes.role_name_node import build_payload

    @PromptServer.instance.routes.get("/qianhui_role_data")
    async def qianhui_role_data(request):
        return web.json_response(build_payload())
except Exception:
    pass
```

并把第 8 条显示名改为：

```python
NODE_DISPLAY_NAME_MAPPINGS["QianhuiRandomCharacter"] = "千绘 · 角色随机师 (V2)"
```

- [ ] **Step 2: 语法冒烟**

Run: `cd "E:\comfuyui 插件\comfyui-qianhui" && python -X utf8 -c "import py_compile; py_compile.compile('__init__.py', doraise=True); print('ok')"`（`from server import PromptServer` 在无 ComfyUI 环境会失败——由 try/except 兜底，本命令只验证语法）
Expected: 打印 ok

- [ ] **Step 3: 提交**

```bash
git add __init__.py && git commit -m "feat: WEB_DIRECTORY + 前端数据路由 + 随机师 V2 显示名"
```

---

### Task 6: js/role_random.js 前端联动 + README 更新

**Files:**
- Create: `js/role_random.js`
- Modify: `README.md`（随机师节点段）
- Test: `node --check js/role_random.js`

**Interfaces:**
- Consumes: `GET /qianhui_role_data`（Task 5）
- Produces: ComfyUI 前端扩展 `qianhui.role_random`；「指定作品」变化 → 「指定角色」过滤为该作品角色（保留首项"（未指定）"）并重置；拉取失败静默退出（全量静态列表兜底）

**tag 解析注意（已按含括号 tag 设计）**：展示串如 `Fate系列 (fate_(series))`，取 `lastIndexOf(" (")` 后 `slice` 再剥**最末一个** `)`（display 闭合符），内部括号保留：`tag = cur.slice(i + 2).replace(/\)$/, "").trim()`。

- [ ] **Step 1: 编写 `js/role_random.js`**

```javascript
// 千绘 · 角色随机师 —— 作品→角色下拉联动
(function () {
    "use strict";
    const SENTINEL = "（未指定）";
    let roleData = null;

    function roleLabel(entry) {
        const [name, cn] = entry;
        return cn ? cn + " (" + name + ")" : name;
    }

    app.registerExtension({
        name: "qianhui.role_random",
        async beforeRegisterNodeDef(nodeType, nodeData, app) {
            if (nodeData.name !== "QianhuiRandomCharacter") return;
            const orig = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const ret = orig ? orig.apply(this, arguments) : undefined;
                const node = this;
                if (!roleData) {
                    fetch("/qianhui_role_data")
                        .then(function (r) { return r.json(); })
                        .then(function (d) {
                            roleData = d;
                            if (node._syncRoleList) node._syncRoleList();
                        })
                        .catch(function () { }); // 拉取失败: 保持后端全量静态列表
                }
                const workWidget = this.widgets.find(function (w) { return w.name === "指定作品"; });
                const roleWidget = this.widgets.find(function (w) { return w.name === "指定角色"; });
                if (workWidget && roleWidget) {
                    // 作品变化 -> 同步角色列表（callback 内 this 是 widget, 故用闭包 node）
                    workWidget.callback = (function (prev, nd) {
                        return function (v) {
                            const r = prev ? prev.call(this, v) : undefined;
                            if (nd._syncRoleList) nd._syncRoleList(v);
                            return r;
                        };
                    })(workWidget.callback, node);
                }
                return ret;
            };
            nodeType.prototype._syncRoleList = function (workValue) {
                if (!roleData) return;
                const workWidget = this.widgets.find(function (w) { return w.name === "指定作品"; });
                const roleWidget = this.widgets.find(function (w) { return w.name === "指定角色"; });
                if (!workWidget || !roleWidget) return;
                const cur = workValue !== undefined ? workValue : workWidget.value;
                const byWork = roleData.roles_by_work || {};
                let entries = [];
                let tag = null;
                if (cur && cur !== SENTINEL) {
                    const i = cur.lastIndexOf(" (");
                    tag = i === -1 ? cur : cur.slice(i + 2).replace(/\)$/, "").trim();
                }
                if (tag && byWork[tag]) {
                    entries = byWork[tag];
                } else {
                    for (const w of roleData.works || []) {
                        if (byWork[w[0]]) for (const e of byWork[w[0]]) entries.push(e);
                    }
                }
                roleWidget.options.values = [SENTINEL].concat(entries.map(roleLabel));
                if (roleWidget.value !== SENTINEL ||
                    !roleWidget.options.values.includes(roleWidget.value)) {
                    roleWidget.value = SENTINEL;
                    if (typeof roleWidget.callback === "function") roleWidget.callback(roleWidget.value);
                }
            };
        },
    });
})();
```

- [ ] **Step 2: 静态检查**

Run: `cd "E:\comfuyui 插件\comfyui-qianhui" && node --check js/role_random.js`
Expected: 无输出（语法通过）

- [ ] **Step 3: README 更新**（如 README 有随机师段落则补充 V2 能力描述；无则加一行节点简介）

- [ ] **Step 4: 提交**

```bash
git add js/role_random.js README.md && git commit -m "feat: 作品→角色下拉联动 JS 扩展 + README"
```

---

### Task 7: 全量回归验证

**Files:** 无新文件

- [ ] **Step 1: 全量单测 + 质检**

Run: `cd "E:\comfuyui 插件\comfyui-qianhui" && python -X utf8 -m unittest discover -s tests -p "test_*.py" -v && python -X utf8 tools\check_role_data.py`
Expected: 全部 PASS + `OK 合格`

- [ ] **Step 2: 真实数据全模式冒烟**

```bash
cd "E:\comfuyui 插件\comfyui-qianhui" && python -X utf8 -c "
import importlib.util
s = importlib.util.spec_from_file_location('r', 'nodes/role_name_node.py')
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
n = m.QianhuiRandomCharacter()
print('随机:', n.generate_role_name('🎲随机类型'))
print('游戏女:', n.generate_role_name('🎮只出游戏', 抽取分类='女'))
print('动漫男:', n.generate_role_name('📺只出动漫', 抽取分类='男'))
print('兽娘:', n.generate_role_name('🎲随机类型', 抽取分类='兽娘'))
print('机械:', n.generate_role_name('🎲随机类型', 抽取分类='机械·人造人'))
print('指定作品原神女:', n.generate_role_name('🎯指定作品', 抽取分类='女', 指定作品=m.work_display('genshin_impact')))
print('指定角色:', n.generate_role_name('🎯指定作品', 指定作品=m.work_display('genshin_impact'), 指定角色=m.role_display('hu_tao_(genshin_impact)')))
print('跳过筛选:', n.generate_role_name('🎯指定作品', 抽取分类='男', 指定角色=m.role_display('hatsune_miku')))
p = m.build_payload()
print('payload作品数:', len(p['works']))
print('示例:', p['works'][0], p['roles_by_work'][p['works'][0][0]][:2])
"
```

Expected: 分类语义正确（游戏女=女性、动漫男=男性、兽娘/机械=对应类）；指定原神+女 → 纯女性；指定胡桃精确输出；初音+男筛选仍输出初音（跳过筛选）；payload 结构可用。控制台中文显示乱码属终端编码问题，不影响正确性。

- [ ] **Step 3: 人工抽查 50 个标注**

以 `tools/check_role_data.py` 分布为基线，随机抽 50 个角色核对 female/male/beast/robot 是否符合常识；发现错标修正 `role_category.json` 并重跑 Step 1。

- [ ] **Step 4: 收尾提交（如抽查有修正）**

```bash
git add data/role_category.json && git commit -m "fix: 角色分类标注抽查修正"
```

---

## Self-Review 记录

- **Spec 覆盖**：数据层（Task1）、UI 5 下拉+抽取分类映射（Task2/3）、JS 联动+回退（Task5/6）、判定优先级 3 层（Task3）、输出格式/转义与 V1 一致（Task3 `_output`）、错误处理（数据缺失/过滤空/角色不在作品内 → Task3 分支、Task1 质检）、/qianhui_role_data 路由（Task5）、build_payload（Task4）、README（Task6）。
- **类型一致性**：`SENTINEL`/`CATEGORY_TAGS`/`load_*`/`parse_*`/`filter_by_category`/`build_payload` 均在 Task2/4 定义，Task3/5 使用处签名一致；`_output(role, cop, 模型类型)` 三处调用一致。
- **已知风险**：指定作品模式+B 端 `by_type` 空 → 已有 `(无角色数据)` 兜底；JS 在旧 ComfyUI（widget 无 value setter 兼容层）下抛错会被 onNodeCreated try/catch 隔离（fetch 后 `.then` 中 `this._syncRoleList` 已防御）。
