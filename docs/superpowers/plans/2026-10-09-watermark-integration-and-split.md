# 千绘 · 水印插件整合 + Latent YOLO 细粒度拆分 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `水印/` 插件的 3 个节点整合进 `comfyui-qianhui`，并把 5633 行的 `水印/nodes.py` 拆成最大约 730 行的职责单一模块，同时把失效的 `配置预设` 控件真正接上。

**Architecture:** 水印侧按「算法层 → 节点层」拆成 `nodes/qianhui_watermark/`（7 文件）；Latent 修复侧按「支撑类外移 + 巨型类切 6 个 Mixin + `process()` 切 5 个阶段函数」拆成 `nodes/qianhui_latent_yolo/`（20 文件），跨阶段状态显式收进 `YoloRunContext`。拆分为纯结构变更，唯一行为变更是接上 `配置预设`。

**Tech Stack:** Python 3.14、ComfyUI 节点 API、PyTorch、NumPy、Pillow、reedsolo、unittest。

## Global Constraints

- 注册名只保留千绘前缀：`QianhuiLatentYOLO`、`QianhuiWatermarkAdd`、`QianhuiWatermarkExtract`。旧名 `LatentKSamplerUpscalerYOLO` / `WatermarkAdd` / `WatermarkExtract` 及全部中文别名一律**不注册**。
- class 名 = 注册 key（千绘既有约定）。
- 新增分类：`千绘/修复`、`千绘/水印`。
- 版本号统一为 `V6.11.3`（源码 docstring 已是此值，注册显示名与启动打印原为 V6.11.2）。
- 水印模块依赖 `reedsolo`，写入新建的 `requirements.txt`。
- 源目录 `E:\comfuyui 插件\水印\` **保留不动**，只读取、不修改、不删除。
- 拆分期间**不改方法体文本**：只改所在文件、`self` 前缀（原本就是 `self.`）、缩进层级语义。原注释（含 `# V6.11.3修复：` 等）随代码搬走，不删不改写。
- YOLO/水印节点的输入参数名保持原样（YOLO 用中文参数名，水印用英文参数名）。
- 测试用 `unittest`，不用 pytest。运行方式：`python -X utf8 tests/xxx.py` 或 `python -X utf8 -m unittest discover -s tests -p "test_*.py"`。
- **所有测试命令必须带 `-X utf8`**：本机控制台是 gbk，而 `nodes/character_node.py:61` 在导入时打印 `⏳`，不加会以 `UnicodeEncodeError` 失败（表现为 `FAILED (errors=N)`，不是断言失败）。
- 每个 commit 结尾加 `Co-Authored-By: Claude Code <noreply@anthropic.com>`。
- 只有 `algorithms.py`（水印侧无）保持**无包内导入**，以便单测按路径直载；其余模块可用相对导入。

## 关键源文件坐标

拆分依据（`E:\comfuyui 插件\水印\nodes.py`，5633 行）：

| 内容 | 行范围 |
|---|---|
| `LatentAlgorithms`（10 个 `@staticmethod`） | 104-533 |
| `YOLOModelManager` | 534-613 |
| `SAMModelManager` | 614-723 |
| `YOLOModelCache` | 724-1159 |
| `ControlNetHelper` | 1160-1234 |
| `LatentKSamplerUpscaler`（仅 `upscale_latent` 是活代码） | 1239-1317（活：1266-1283） |
| `LatentKSamplerUpscalerYOLO` | 1322-5607 |
| ├ `INPUT_TYPES` | 1332-1516 |
| ├ `RETURN_*` / `FUNCTION` / `CATEGORY` | 1512-1515 |
| ├ `__init__` | 1517-1540 |
| ├ 检测组 | 1541-1788, 1889-2346 |
| ├ 手骨架提取 `_extract_hand_skeleton` | 1789-1888 |
| ├ SAM 组 | 2347-2946 |
| ├ 几何组 | 2947-3178 |
| ├ 遮罩组 | 3179-3370 |
| ├ 策略组 | 3371-3701 |
| ├ 重绘组 | 3702-4130 |
| └ `process()` | 4131-5607 |
| 注册 + 启动打印 | 5609-5633 |

`process()` 内部阶段边界：参数解析 4131-4240、准备 4240-4386、检测主循环 4387-4862、区域精修 4864-5406、全图降噪+时序平滑 5407-5481、预览与报告 5483-5607。

---

## Task 0: 采集基准图（用户执行，不可省略）

**Files:**
- Create: `docs/superpowers/baseline/`（存输出）

**Interfaces:**
- Consumes: 无
- Produces: `docs/superpowers/baseline/A..E` 各场景输出图 + `检测信息` 文本。Task 3、4、6 用它做回归比对。

这是本计划唯一需要人工执行的任务，也是 Task 3/4/6 的验证前提。**必须在任何代码改动之前完成。**

- [ ] **步骤 1：搭一组覆盖主要路径的工作流**

在 ComfyUI 里用原 `水印` 插件（节点名 `LatentKSamplerUpscalerYOLO`）搭一条链，固定 `seed`，`配置预设` 一律选 **`[不使用预设]`**。跑 5 个场景：

| 场景 | 关键设置 |
|---|---|
| A | `启用YOLO`=开、`启用SAM精细化`=关、不接 CN、`工作模式`=`仅修复(保持输入Latent)`、`开启裁剪精修`=开 |
| B | A + `启用SAM精细化`=开 |
| C | A + 接 `CN模型`/`CN图像`，并接 `手骨架CN模型` |
| D | A + `工作模式`=`修复后全图降噪`，`全图降噪强度`=0.25 |
| E | A + `开启裁剪精修`=关（走全图 mask 修复分支） |

- [ ] **步骤 2：保存输出**

每个场景保存 4 项到 `docs/superpowers/baseline/<场景>/`：
- 输出 Latent 解码后的图（`检测预览` 已含预览图，另存一份最终 latent 出图）
- `检测预览` 输出图
- `检测遮罩` 输出图
- `检测信息` 文本（复制到 `info.txt`）

命名示例：`docs/superpowers/baseline/A/preview.png`、`A/mask.png`、`A/info.txt`。

- [ ] **步骤 3：提交基准**

```bash
git add docs/superpowers/baseline
git commit -m "test: 采集 Latent YOLO 修复基准输出（5 场景）

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

## Task 1: 整合骨架

**Files:**
- Create: `nodes/qianhui_watermark/__init__.py`（占位，Task 2 填充）
- Create: `nodes/qianhui_latent_yolo/__init__.py`（占位，Task 3-6 填充）
- Create: `requirements.txt`
- Modify: `__init__.py`（在现有第 9 个加载块之后追加两个块）

**Interfaces:**
- Consumes: 无
- Produces: 包路径 `nodes.qianhui_watermark`、`nodes.qianhui_latent_yolo`；`requirements.txt`。Task 2-6 往里填实现。

- [ ] **步骤 1：建两个包目录并写占位 `__init__.py`**

```bash
mkdir -p nodes/qianhui_watermark nodes/qianhui_latent_yolo
```

`nodes/qianhui_watermark/__init__.py`：

```python
"""千绘 · 水印模块（可见水印 + DCT-QIM 隐形水印）。"""
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
```

`nodes/qianhui_latent_yolo/__init__.py`：

```python
"""千绘 · Latent 智能修复模块（YOLO 检测 + SAM 精修 + ControlNet）。"""
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
```

- [ ] **步骤 2：写 `requirements.txt`**

```
reedsolo
```

- [ ] **步骤 3：在 `__init__.py` 追加两个加载块**

在现有第 9 个加载块（`QianhuiMultiLoRAGrid`）之后、`# 8b. 前端联动数据路由` 之前插入：

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
    NODE_CLASS_MAPPINGS["QianhuiWatermarkAdd"] = QianhuiWatermarkAdd
    NODE_DISPLAY_NAME_MAPPINGS["QianhuiWatermarkAdd"] = "千绘 · 水印添加"
    NODE_CLASS_MAPPINGS["QianhuiWatermarkExtract"] = QianhuiWatermarkExtract
    NODE_DISPLAY_NAME_MAPPINGS["QianhuiWatermarkExtract"] = "千绘 · 水印提取"
except Exception as e:
    print(f"❌ 千绘 水印模块加载失败: {e}")
```

- [ ] **步骤 4：验证骨架不破坏现有节点**

Run:
```bash
python -X utf8 - <<'PY'
import ast, io
tree = ast.parse(io.open("__init__.py", encoding="utf-8").read())
tries = [n for n in tree.body if isinstance(n, ast.Try)]
print("语法 OK, 顶层 try 块:", len(tries), "个")
# 构成：9 个既有节点加载块 + 1 个路由块(8b) + 2 个新增块(11/12) = 12
assert len(tries) >= 12, f"try 块数不足，加载隔离被破坏: {len(tries)}"
# 每个 try 都必须有 except，保证单模块失败不中断其余节点
for t in tries:
    assert t.handlers, "存在无 except 的 try 块"
print("加载隔离 OK")
PY
```
Expected: `语法 OK, 顶层 try 块: 12 个` 与 `加载隔离 OK`。

因为占位包还没有 `QianhuiLatentYOLO` 等符号，加载会走 except 分支并打印 ❌——**这是预期的**，Task 2、3 填入实现后消失。这也顺手验证了「单模块失败不影响其余节点」。

- [ ] **步骤 5：提交**

```bash
git add nodes/qianhui_watermark nodes/qianhui_latent_yolo requirements.txt __init__.py
git commit -m "feat: 千绘整合骨架（水印/Latent修复包目录 + 依赖清单 + 注册块）

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

## Task 2: 水印模块拆分

**Files:**
- Create: `nodes/qianhui_watermark/fonts.py`、`dct.py`、`payload.py`、`invisible.py`、`visible.py`、`nodes.py`
- Modify: `nodes/qianhui_watermark/__init__.py`
- Create: `tests/_loader.py`、`tests/test_watermark.py`、`tests/test_latent_algorithms.py`
- Create: `tests/assets/test_cn_wm.png`

**Interfaces:**
- Consumes: `nodes.qianhui_watermark` 包（Task 1 建）
- Produces:
  - `nodes/qianhui_watermark/__init__.py` 导出 `QianhuiWatermarkAdd`、`QianhuiWatermarkExtract`、`NODE_CLASS_MAPPINGS`、`NODE_DISPLAY_NAME_MAPPINGS`
  - `dct.py`：`_DCT_D`、`_dct8(block: np.ndarray) -> np.ndarray`、`_idct8(coef) -> np.ndarray`、`_get_block_q(variance: float, embed: bool) -> float`
  - `payload.py`：`_SYNC`、`_RS`、`_require_reedsolo(feature: str) -> None`、`_encode_payload(text: str) -> list[int]`、`_decode_payload(bits: list[int]) -> str`
  - `invisible.py`：`_QIM_REPEAT`、`_embed_invisible(img_rgb, text) -> np.ndarray`、`_extract_invisible(img_rgb) -> str`
  - `visible.py`：`_hex_to_rgb(color: str) -> tuple[int,int,int]`、`_make_text_tile(...)`、`_apply_visible_watermark(...) -> np.ndarray`
  - `fonts.py`：`_load_font(font_size: int) -> ImageFont.FreeTypeFont`
  - `tests/_loader.py`：`load_package(pkg_name: str, pkg_dir: str) -> module`（支持包内相对导入）

- [ ] **步骤 1：先写测试装载器**

`tests/_loader.py`：

```python
# -*- coding: utf-8 -*-
"""按路径装载包/模块，绕开需要 ComfyUI 运行时的包导入。

spec_from_file_location 配合 submodule_search_locations 可构造真正的包，
使包内 `from .dct import ...` 相对导入正常工作。
"""
import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_package(pkg_name: str, pkg_dir: str):
    """把 pkg_dir 当作名为 pkg_name 的包装载并注册进 sys.modules。"""
    init_py = os.path.join(pkg_dir, "__init__.py")
    spec = importlib.util.spec_from_file_location(
        pkg_name, init_py, submodule_search_locations=[pkg_dir]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[pkg_name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_module(mod_name: str, path: str):
    """按路径装载单个模块（要求该模块无包内相对导入）。"""
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def pkgs_dir(*parts: str) -> str:
    return os.path.join(ROOT, "nodes", *parts)
```

- [ ] **步骤 2：写失败测试（水印算法层）**

`tests/test_watermark.py` 开头：

```python
# -*- coding: utf-8 -*-
"""水印模块单测（算法层 + 节点层）。搬运自 水印/test_watermark.py 的 17 个用例。"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _loader import load_package, pkgs_dir  # noqa: E402

WM = load_package("qh_watermark", pkgs_dir("qianhui_watermark"))


class TestDct(unittest.TestCase):
    def test_dct_roundtrip(self):
        """DCT 后 IDCT 应还原原始块（误差 < 1e-9）"""
        from qh_watermark.dct import _dct8, _idct8
        rng = np.random.default_rng(42)
        block = rng.random((8, 8)) * 255
        reconstructed = _idct8(_dct8(block))
        self.assertTrue(
            np.allclose(block, reconstructed, atol=1e-9),
            f"DCT round-trip error: max={np.abs(block - reconstructed).max()}",
        )

    def test_adaptive_q(self):
        """块方差分档应返回 25 / 50 / 80，且提取阈值比嵌入阈值宽松"""
        from qh_watermark.dct import _get_block_q
        self.assertEqual(_get_block_q(50.0, embed=True), 25.0)
        self.assertEqual(_get_block_q(400.0, embed=True), 50.0)
        self.assertEqual(_get_block_q(2000.0, embed=True), 80.0)
        # 提取侧死区更宽：120 在嵌入侧是 50，在提取侧仍是 25
        self.assertEqual(_get_block_q(120.0, embed=True), 50.0)
        self.assertEqual(_get_block_q(120.0, embed=False), 25.0)


class TestPayload(unittest.TestCase):
    def test_payload_roundtrip_ascii(self):
        from qh_watermark.payload import _encode_payload, _decode_payload
        text = "© Alice 2026"
        self.assertEqual(_decode_payload(_encode_payload(text)), text)

    def test_payload_roundtrip_cjk(self):
        from qh_watermark.payload import _encode_payload, _decode_payload
        text = "版权所有 张三"
        self.assertEqual(_decode_payload(_encode_payload(text)), text)

    def test_decode_no_sync(self):
        from qh_watermark.payload import _decode_payload
        self.assertEqual(_decode_payload([0, 1] * 64), "")

    def test_rs_roundtrip(self):
        """RS 编码能纠正 2 字节错误"""
        from qh_watermark.payload import _encode_payload, _decode_payload
        bits = _encode_payload("hello")
        flipped = bits[:]
        for i in (70, 71, 72):          # 翻一个字节内的 3 个 bit
            flipped[i] ^= 1
        self.assertEqual(_decode_payload(flipped), "hello")


class TestInvisible(unittest.TestCase):
    def _img(self, h=256, w=256, seed=0):
        rng = np.random.default_rng(seed)
        return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)

    def test_invisible_roundtrip(self):
        from qh_watermark.invisible import _embed_invisible, _extract_invisible
        img = self._img()
        out = _embed_invisible(img, "版权所有 2026")
        self.assertEqual(out.shape, img.shape)
        self.assertEqual(_extract_invisible(out), "版权所有 2026")

    def test_invisible_no_watermark(self):
        from qh_watermark.invisible import _extract_invisible
        # 纯噪声图不应解出有效水印
        self.assertEqual(_extract_invisible(self._img(seed=7)), "")

    def test_invisible_psnr(self):
        """嵌入后 PSNR 应高于 30dB（肉眼不可见）"""
        from qh_watermark.invisible import _embed_invisible
        img = self._img(h=512, w=512)
        out = _embed_invisible(img, "版权")
        mse = np.mean((img.astype(np.float64) - out.astype(np.float64)) ** 2)
        psnr = 10 * np.log10(255.0 ** 2 / max(mse, 1e-12))
        self.assertGreater(psnr, 30.0, f"PSNR 过低: {psnr:.2f}dB")

    def test_invisible_capacity_error(self):
        """图像太小应抛 ValueError"""
        from qh_watermark.invisible import _embed_invisible
        with self.assertRaises(ValueError):
            _embed_invisible(self._img(h=64, w=64), "x" * 500)


class TestVisible(unittest.TestCase):
    def _img(self, h=128, w=128):
        return np.full((h, w, 3), 128, dtype=np.uint8)

    def test_visible_text_shape(self):
        from qh_watermark.visible import _apply_visible_watermark
        out = _apply_visible_watermark(
            self._img(), "文字", "测试", 0.5, 40, -30, 24, "#FF0000", None
        )
        self.assertEqual(out.shape, (128, 128, 3))
        self.assertEqual(out.dtype, np.uint8)

    def test_visible_invalid_color_fallback(self):
        """非法颜色串应回退白色且不抛异常"""
        from qh_watermark.visible import _hex_to_rgb
        self.assertEqual(_hex_to_rgb("#GGGGGG"), (255, 255, 255))
        self.assertEqual(_hex_to_rgb("#FF0000"), (255, 0, 0))

    def test_visible_image_wm_missing_fallback(self):
        """图片模式但未接图 → 回退文字模式"""
        from qh_watermark.visible import _apply_visible_watermark
        out = _apply_visible_watermark(
            self._img(), "图片", "回退", 0.5, 40, 0, 24, "#FFFFFF", None
        )
        self.assertEqual(out.shape, (128, 128, 3))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **步骤 3：跑测试，确认失败**

Run: `python -X utf8 tests/test_watermark.py`
Expected: FAIL——`FileNotFoundError` 或 `ModuleNotFoundError: No module named 'qh_watermark.dct'`（因为模块还没建）。

- [ ] **步骤 4：按层拆出 6 个模块**

把 `水印/watermark_nodes.py` 的内容按下面切分搬入，**方法体逐字不动**，只调整导入头：

| 新文件 | 搬入内容（源行） | 需要的导入头 |
|---|---|---|
| `fonts.py` | `_FONT_CANDIDATES`(14-28)、`_load_font`(31-39) | `import os` / `from PIL import ImageFont` |
| `dct.py` | `_make_dct_matrix`(43-51)、`_DCT_D`(53)、`_dct8`(56-58)、`_idct8`(61-63)、`_get_block_q`(80-102) | `import numpy as np` |
| `payload.py` | `_SYNC`(68)、`_RS`(69)、`_require_reedsolo`(72-77)、`_int_to_bits`(105-107)、`_bits_to_int`(110-115)、`_encode_payload`(118-132)、`_decode_payload`(135-153)，以及 reedsolo 的 try/except 导入(6-11) | 见下 |
| `invisible.py` | `_QIM_REPEAT`(158)、`_embed_invisible`(161-243)、`_extract_invisible`(246-291) | `import numpy as np` / `from PIL import Image` / `from .dct import _dct8, _idct8, _get_block_q` / `from .payload import _encode_payload, _decode_payload, _require_reedsolo` |
| `visible.py` | `_hex_to_rgb`(296-308)、`_make_text_tile`(311-325)、`_apply_visible_watermark`(328-380) | `import numpy as np` / `from PIL import Image, ImageDraw` / `from .fonts import _load_font` |
| `nodes.py` | `WatermarkAdd`(385-461)、`WatermarkExtract`(464-497) | `import torch` / `import numpy as np` / `from .invisible import _embed_invisible, _extract_invisible` / `from .visible import _apply_visible_watermark` |

`payload.py` 的导入头（保持软失败语义）：

```python
try:
    from reedsolo import RSCodec
    _REEDSOLO_IMPORT_ERROR = None
except ImportError as exc:
    RSCodec = None
    _REEDSOLO_IMPORT_ERROR = exc

_SYNC = 0xDEADBEEF
_RS = RSCodec(4) if RSCodec is not None else None
```

- [ ] **步骤 5：改类名与分类**

在 `nodes/qianhui_watermark/nodes.py` 中：
- `class WatermarkAdd` → `class QianhuiWatermarkAdd`
- `class WatermarkExtract` → `class QianhuiWatermarkExtract`
- 两个类的 `CATEGORY = "image/watermark"` → `CATEGORY = "千绘/水印"`
- 文件末尾加：

```python
NODE_CLASS_MAPPINGS = {
    "QianhuiWatermarkAdd": QianhuiWatermarkAdd,
    "QianhuiWatermarkExtract": QianhuiWatermarkExtract,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "QianhuiWatermarkAdd": "千绘 · 水印添加",
    "QianhuiWatermarkExtract": "千绘 · 水印提取",
}
```

- [ ] **步骤 6：写包 `__init__.py`**

```python
"""千绘 · 水印模块（可见水印 + DCT-QIM 隐形水印）。"""
from .nodes import (
    QianhuiWatermarkAdd,
    QianhuiWatermarkExtract,
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)

__all__ = [
    "QianhuiWatermarkAdd",
    "QianhuiWatermarkExtract",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
```

- [ ] **步骤 7：跑测试，确认通过**

Run: `python -X utf8 tests/test_watermark.py`
Expected: `OK`，13 个用例全过。

- [ ] **步骤 8：补节点层测试**

在 `tests/test_watermark.py` 的 `if __name__` 之前追加（节点层需要 torch）：

```python
class TestNodes(unittest.TestCase):
    """节点层测试。

    输入、断言、边界值**照搬源测试** `水印/test_watermark.py` 中同名用例，
    只把导入方式从 `from watermark_nodes import X` 改为包路径 `qh_watermark.nodes`。
    不要重写输入或收紧边界——源测试的边界是实测出来的，凭空收紧会变成不可满足的断言。
    """

    def test_node_watermarkadd_text(self):
        import torch
        from qh_watermark.nodes import QianhuiWatermarkAdd
        node = QianhuiWatermarkAdd()
        # ComfyUI IMAGE tensor: (B, H, W, C) float32 0-1
        img = torch.zeros(1, 64, 64, 3)
        original, watermarked = node.apply_watermark(
            image=img, watermark_type="文字", text="TEST © 2026",
            opacity=0.5, spacing=30, rotation=-30, font_size=16, color="#FFFFFF",
        )
        self.assertEqual(tuple(original.shape), (1, 64, 64, 3))
        self.assertEqual(tuple(watermarked.shape), (1, 64, 64, 3))
        # 原图应未被修改
        self.assertTrue(torch.equal(original, img), "original image was modified")

    def test_node_watermarkadd_invisible(self):
        import torch
        from qh_watermark.nodes import QianhuiWatermarkAdd
        node = QianhuiWatermarkAdd()
        rng = np.random.default_rng(7)
        arr = rng.random((1, 512, 512, 3)).astype(np.float32)
        img = torch.from_numpy(arr)
        original, watermarked = node.apply_watermark(
            image=img, watermark_type="隐形文字", text="copyright",
            opacity=0.3, spacing=100, rotation=-30, font_size=36, color="#FFFFFF",
        )
        self.assertFalse(torch.equal(original, watermarked),
                         "invisible watermark had no effect")
        diff = (original - watermarked).abs().max().item()
        # 边界 0.40 照搬源测试；实测约 0.21
        # （源测试行尾注释里的 "actual max ~0.07" 是失效的实测记录，不要照搬）
        self.assertLess(diff, 0.40, f"Invisible watermark caused large pixel diff: {diff}")

    def test_node_watermarkextract_roundtrip(self):
        import torch
        from qh_watermark.nodes import QianhuiWatermarkAdd, QianhuiWatermarkExtract
        add = QianhuiWatermarkAdd()
        extract = QianhuiWatermarkExtract()
        rng = np.random.default_rng(99)
        arr = rng.random((1, 256, 256, 3)).astype(np.float32)
        img = torch.from_numpy(arr)
        text = "hello 水印"
        _orig, watermarked = add.apply_watermark(
            image=img, watermark_type="隐形文字", text=text,
            opacity=0.3, spacing=100, rotation=-30, font_size=36, color="#FFFFFF",
        )
        (extracted_text,) = extract.extract_watermark(image=watermarked)
        self.assertEqual(extracted_text, text)

    def test_node_watermarkextract_no_wm(self):
        import torch
        from qh_watermark.nodes import QianhuiWatermarkExtract
        extract = QianhuiWatermarkExtract()
        img = torch.zeros(1, 128, 128, 3)
        (result,) = extract.extract_watermark(image=img)
        self.assertEqual(result, "")
```

⚠ **算法层的 13 个用例同理**：源文件里已存在的用例，其输入、断言、边界值一律照搬，只改导入方式与组织成 `unittest` 类。**不要凭空收紧任何边界**——源测试的边界是实测校准过的（例：隐形水印像素差边界是 0.40 而非 0.2）。若某条断言在照搬后失败，说明移植出错，应查移植而不是改断言。

- [ ] **步骤 9：跑全部水印测试**

Run: `python -X utf8 tests/test_watermark.py`
Expected: `OK`，17 个用例全过。

- [ ] **步骤 10：搬入测试素材**

```bash
mkdir -p tests/assets
cp "../水印/test_cn_wm.png" tests/assets/
```

- [ ] **步骤 11：提交**

```bash
git add nodes/qianhui_watermark tests/_loader.py tests/test_watermark.py tests/assets
git commit -m "feat: 水印模块整合进千绘（拆 6 模块 + 17 个单测搬入并适配）

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

## Task 3: Latent 修复 —— 支撑类外移 + 巨型类切 Mixin（`process()` 暂不动）

**Files:**
- Create: `nodes/qianhui_latent_yolo/algorithms.py`、`model_managers.py`、`model_cache.py`、`controlnet.py`、`base_node.py`、`node.py`
- Create: `nodes/qianhui_latent_yolo/mixins/__init__.py`、`detection.py`、`sam.py`、`geometry.py`、`masks.py`、`policy.py`、`inpaint.py`
- Modify: `nodes/qianhui_latent_yolo/__init__.py`
- Create: `tests/test_latent_algorithms.py`

**Interfaces:**
- Consumes: `nodes.qianhui_latent_yolo` 包（Task 1 建）
- Produces:
  - `__init__.py` 导出 `QianhuiLatentYOLO` + `NODE_*_MAPPINGS`
  - `algorithms.py`：`LatentAlgorithms`，含 10 个 `@staticmethod`（`latent_aware_resize`、`_mild_sharpen`、`gaussian_feather`、`frequency_aware_color_match`、`simple_color_match`、`smart_soft_blend`、`poisson_blend_latent`、`mpb_blend_latent`、`simple_blend`）。**该文件不得有包内相对导入**（供单测直载）。
  - `model_managers.py`：`YOLOModelManager`、`SAMModelManager`
  - `model_cache.py`：`YOLOModelCache(max_cached_models=3, vram_threshold_mb=500)`
  - `controlnet.py`：`ControlNetHelper`、`extract_hand_skeleton(image_np, bbox, target_h, target_w)`
  - `base_node.py`：`LatentKSamplerUpscaler`（仅 `upscale_latent(samples, method, factor)`）
  - `node.py`：`QianhuiLatentYOLO`，MRO = `(DetectionMixin, SamMixin, GeometryMixin, MasksMixin, PolicyMixin, InpaintMixin, LatentKSamplerUpscaler)`
  - 各 mixin：只定义方法，无 `__init__`、无类属性

- [ ] **步骤 1：写 `algorithms.py` 的失败测试**

`tests/test_latent_algorithms.py`：

```python
# -*- coding: utf-8 -*-
"""LatentAlgorithms 纯函数单测（只依赖 torch/numpy，无需 ComfyUI）。"""
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _loader import load_module, pkgs_dir  # noqa: E402

ALG = load_module(
    "qh_algorithms", os.path.join(pkgs_dir("qianhui_latent_yolo"), "algorithms.py")
)
LA = ALG.LatentAlgorithms


class TestLatentAwareResize(unittest.TestCase):
    def test_shape(self):
        x = torch.randn(1, 4, 16, 16)
        out = LA.latent_aware_resize(x, 32, 32)
        self.assertEqual(tuple(out.shape), (1, 4, 32, 32))

    def test_same_size_is_identity(self):
        x = torch.randn(1, 4, 16, 16)
        out = LA.latent_aware_resize(x, 16, 16)
        self.assertTrue(torch.allclose(out, x, atol=1e-6))


class TestGaussianFeather(unittest.TestCase):
    def test_range_and_shape(self):
        mask = torch.zeros(1, 1, 32, 32)
        mask[:, :, 12:20, 12:20] = 1.0
        out = LA.gaussian_feather(mask, sigma=2.0)
        self.assertEqual(tuple(out.shape), tuple(mask.shape))
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0 + 1e-6)

    def test_center_higher_than_edge(self):
        """中心权重应不低于边缘（羽化是向外衰减）"""
        mask = torch.zeros(1, 1, 32, 32)
        mask[:, :, 14:18, 14:18] = 1.0
        out = LA.gaussian_feather(mask, sigma=2.0)
        self.assertGreaterEqual(float(out[0, 0, 16, 16]), float(out[0, 0, 0, 0]))


class TestBlends(unittest.TestCase):
    def _pair(self, h=8, w=8):
        src = torch.zeros(1, 4, h, w)
        dst = torch.ones(1, 4, h, w)
        return src, dst

    def test_simple_blend_mask_zero_keeps_dst(self):
        src, dst = self._pair()
        mask = torch.zeros(1, 1, 8, 8)
        out = LA.simple_blend(src, dst, mask)
        self.assertTrue(torch.allclose(out, dst, atol=1e-5))

    def test_simple_blend_mask_one_takes_src(self):
        src, dst = self._pair()
        mask = torch.ones(1, 1, 8, 8)
        out = LA.simple_blend(src, dst, mask)
        self.assertTrue(torch.allclose(out, src, atol=1e-5))

    def test_poisson_blend_shape(self):
        src, dst = self._pair()
        mask = torch.zeros(1, 1, 8, 8)
        mask[:, :, 2:6, 2:6] = 1.0
        out = LA.poisson_blend_latent(src, dst, mask, iterations=3)
        self.assertEqual(tuple(out.shape), tuple(dst.shape))

    def test_mpb_blend_shape(self):
        src, dst = self._pair()
        mask = torch.zeros(1, 1, 8, 8)
        mask[:, :, 2:6, 2:6] = 1.0
        out = LA.mpb_blend_latent(src, dst, mask, keep=0.5)
        self.assertEqual(tuple(out.shape), tuple(dst.shape))

    def test_smart_soft_blend_shape(self):
        src, dst = self._pair()
        mask = torch.zeros(1, 1, 8, 8)
        mask[:, :, 2:6, 2:6] = 1.0
        out = LA.smart_soft_blend(src, dst, mask)
        self.assertEqual(tuple(out.shape), tuple(dst.shape))


class TestColorMatch(unittest.TestCase):
    def test_simple_color_match_shape(self):
        src = torch.rand(1, 4, 8, 8)
        ref = torch.rand(1, 4, 8, 8)
        out = LA.simple_color_match(src, ref)
        self.assertEqual(tuple(out.shape), tuple(src.shape))

    def test_frequency_aware_color_match_shape(self):
        src = torch.rand(1, 4, 8, 8)
        ref = torch.rand(1, 4, 8, 8)
        out = LA.frequency_aware_color_match(src, ref, strength=0.5)
        self.assertEqual(tuple(out.shape), tuple(src.shape))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **步骤 2：跑测试，确认失败**

Run: `python -X utf8 tests/test_latent_algorithms.py`
Expected: FAIL——`FileNotFoundError`（`algorithms.py` 尚未创建）。

- [ ] **步骤 3：搬入 `algorithms.py`**

把源 104-533 行（`LatentAlgorithms` 整个类）搬入 `nodes/qianhui_latent_yolo/algorithms.py`。导入头按该类实际用到的符号设：

```python
"""Latent 空间核心算法（插值/羽化/色差修复/融合）。本模块不依赖 ComfyUI。"""
import math

import numpy as np
import torch
import torch.nn.functional as F
```

搬完后运行 `python -X utf8 -c "import ast,io; ast.parse(io.open('nodes/qianhui_latent_yolo/algorithms.py',encoding='utf-8').read())"` 确认语法。若 `LatentAlgorithms` 用到未列出的符号（如 `gc`），补上对应 import——**判据是文件里出现的名字，不是猜测**。

- [ ] **步骤 4：跑测试，确认通过**

Run: `python -X utf8 tests/test_latent_algorithms.py`
Expected: `OK`，12 个用例全过。若有断言失败，说明方法在搬运中被动过——对照源文件修正，**不要改断言**。

- [ ] **步骤 5：搬入其余支撑模块**

| 新文件 | 搬入源行 | 需要的导入 |
|---|---|---|
| `model_managers.py` | 534-723（`YOLOModelManager` + `SAMModelManager`） | `glob`/`os`/`folder_paths` |
| `model_cache.py` | 724-1159（`YOLOModelCache`） | `gc`/`os`/`torch`/`comfy.model_management` |
| `controlnet.py` | 1160-1234（`ControlNetHelper`） | `os`/`hp`/`torch`/`torch.nn.functional as F`/`numpy as np`/`comfy.controlnet`/`folder_paths` |
| `base_node.py` | 1266-1283（仅 `upscale_latent`） | `torch`/`torch.nn.functional as F`/`from .algorithms import LatentAlgorithms` |

**`base_node.py` 只搬 `upscale_latent`**。源 1242-1265 的 `INPUT_TYPES` 与 1284-1317 的 `process` 从未被调用（该基类未注册），不再移植。

**`_extract_hand_skeleton`（1789-1888）归 `inpaint.py`，保持实例方法原样不动**——它虽不引用 `self`，但唯一调用点在同属精修流程的 5093 行，放同一个 Mixin 可完全避免改造调用点，符合「只改所在文件、不改方法体」的约束。（早期草案曾计划挪到 `controlnet.py` 并改成模块级函数，会违反该约束，已废弃。）

- [ ] **步骤 6：切 6 个 Mixin**

`nodes/qianhui_latent_yolo/mixins/__init__.py`：

```python
"""QianhuiLatentYOLO 的方法按职责切分成的 6 个 Mixin。"""
```

各 Mixin 文件的内容 = 源文件中对应方法**逐字搬入**（含原注释），导入头按实际用到的符号设：

| Mixin 文件 | 方法（源行号） |
|---|---|
| `detection.py` | `_get_ultralytics_version`(1541)、`_check_yolo_capabilities`(1552)、`_load_yolo_model`(1591)、`_decode_latent_to_image`(1745)、`_upscale_image_for_yolo`(1750)、`_detect_with_multi_models`(1889)、`_detect_with_yolo`(1969)、`_detect_single_enhanced`(2181) |
| `sam.py` | `_refine_with_sam`(2347)、`_unload_sam_model`(2839) |
| `geometry.py` | `_nms_detections_enhanced`(2947)、`_iou_xyxy`(2978)、`_wiou_xyxy`(2993)、`_match_class_name`(3058)、`_filter_detections_by_target`(3069)、`_bbox_area`(3073)、`_bbox_aspect`(3078)、`_assign_lr_group`(3084)、`_filter_detections_geometry`(3090)、`_sort_detections`(3107)、`_limit_detections_per_class`(3114)、`_expand_bbox_xyxy`(3124)、`_merge_overlapping_detections`(3134) |
| `masks.py` | `_create_detection_mask`(3179)、`_calculate_ema_attention`(3260)、`_draw_detection_preview`(3329) |
| `policy.py` | `_resolve_model_path`(3371)、`_resolve_sam_model_path`(3381)、`_parse_target_classes`(3392)、`_auto_params_for_det`(3398)、`_get_preset_class_rules`(3451)、`_parse_class_rules`(3476)、`_match_rule`(3513)、`_get_rule_for_det`(3524)、`_priority_sort_detections`(3539)、`_apply_rule_limits`(3569)、`_build_det_policy`(3592) |
| `inpaint.py` | `_extract_hand_skeleton`(1789)、`_decode_for_detection`(3702)、`_crop_latent`(3722)、`_build_crop_mask`(3740)、`_maybe_apply_controlnet`(3766)、`_gc_if_needed`(3780)、`_encode_hand_foot_prompts`(3818)、`_merge_conditioning`(3834)、`_get_enhanced_conditioning_simple`(3854)、`_prepare_inpaint_latent`(3872)、`_inpaint_sample`(3932)、`_get_crop_inpaint_mask`(4030) |

每个 Mixin 的类名：`DetectionMixin`、`SamMixin`、`GeometryMixin`、`MasksMixin`、`PolicyMixin`、`InpaintMixin`。类体只有方法，**不加 `__init__`**。

**已核对：6 个 Mixin 之间方法名无交集**，MRO 不会遮蔽。跨 Mixin 调用一律 `self.xxx(...)`，Python 在实例上按 MRO 解析，与原先同类内调用等价。

- [ ] **步骤 7：写 `node.py`**

```python
"""千绘 · Latent 智能修复节点（QianhuiLatentYOLO）。

由 6 个 Mixin 聚合而成，MRO 顺序：
Detection → Sam → Geometry → Masks → Policy → Inpaint → LatentKSamplerUpscaler
基类置于末位以保证 upscale_latent 可解析。
"""
import os
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
import folder_paths
import comfy.samplers
import comfy.sample
import comfy.utils
import comfy.model_management
import comfy.controlnet
import nodes  # ComfyUI 根目录的 nodes.py，取 nodes.KSampler()

from .algorithms import LatentAlgorithms
from .model_managers import YOLOModelManager, SAMModelManager
from .model_cache import YOLOModelCache
from .controlnet import ControlNetHelper
from .base_node import LatentKSamplerUpscaler
from .mixins.detection import DetectionMixin
from .mixins.sam import SamMixin
from .mixins.geometry import GeometryMixin
from .mixins.masks import MasksMixin
from .mixins.policy import PolicyMixin
from .mixins.inpaint import InpaintMixin

try:
    from comfy.model_management import InterruptProcessingException
except ImportError:
    class InterruptProcessingException(Exception):
        pass


class QianhuiLatentYOLO(DetectionMixin, SamMixin, GeometryMixin, MasksMixin,
                       PolicyMixin, InpaintMixin, LatentKSamplerUpscaler):
    # INPUT_TYPES(1332-1516) / RETURN_* / FUNCTION / CATEGORY / __init__(1517-1540) 原样搬入
    ...
```

- `INPUT_TYPES`（1332-1516）与 `RETURN_TYPES`/`RETURN_NAMES`/`FUNCTION`/`CATEGORY`（1512-1515）原样搬入。
- `CATEGORY = "Latent/批量处理"` → `CATEGORY = "千绘/修复"`。
- `__init__`（1517-1540）原样搬入，但把

  ```python
  super().__init__() if hasattr(super(), '__init__') else None
  ```

  简化为

  ```python
  super().__init__()
  ```

- **源文件末尾的「节点注册 + 启动打印」块（5609-5633）不移植**，理由如下：
  - 注册部分（5609-5617）由本任务步骤 8 的包 `__init__.py` 取代，重复注册会互相覆盖。
  - 启动横幅打印（5619-5633）是水印插件的自有风格；千绘的根 `__init__.py` 已统一打印加载状态（`✅ 千绘节点加载成功: ...`），再打一个 15 行横幅不符合千绘既有约定。
- `process()` 本次**不动**：先把 4131-5607 整体搬进 `node.py` 类体内（Task 4 再切）。

#### 版本号处理规则（本任务及后续任务统一适用）

用户可见的版本标识统一为 `V6.11.3`：

| 位置 | 原值 | 改为 |
|---|---|---|
| 包 `__init__.py` 的显示名（本任务步骤 8） | — | 已含 `V6.11.3` |
| `process()` 内 4296 行的运行时 print `[Latent YOLO智能处理+ControlNet V6.11.2] 批量:` | `V6.11.2` | `V6.11.3` |
| 源 5618 的显示名、5622 的启动 print | `V6.11.2` | 不移植（见上） |

⚠ **不要动代码注释里形如 `# V6.11.2: 确保latent在GPU上` 的记录。** 那些是「该修复引入于哪个版本」的历史变更记录（全文件约 20 处，如 1774、3958、4830、4869 行），改了就等于篡改历史。本规则**只**改用户可见的版本标识字符串。

- [ ] **步骤 8：写包 `__init__.py`**

```python
"""千绘 · Latent 智能修复模块（YOLO 检测 + SAM 精修 + ControlNet）。"""
from .node import QianhuiLatentYOLO

NODE_CLASS_MAPPINGS = {
    "QianhuiLatentYOLO": QianhuiLatentYOLO,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "QianhuiLatentYOLO": "千绘 · Latent智能修复 (YOLO+SAM+CN V6.11.3)",
}

__all__ = ["QianhuiLatentYOLO", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
```

- [ ] **步骤 9：静态健全性检查**

Run:
```bash
python - <<'PY'
import ast, io, os
base = "nodes/qianhui_latent_yolo"
smoke = {
    "algorithms.py": ["math", "numpy", "torch"],
    "model_managers.py": ["glob", "os", "folder_paths"],
    "model_cache.py": ["gc", "os", "torch", "comfy.model_management"],
    "controlnet.py": ["os", "numpy", "torch", "comfy.controlnet", "folder_paths"],
    "base_node.py": ["torch"],
    "node.py": ["torch", "numpy", "folder_paths", "comfy.samplers", "nodes"],
}
for rel, _ in smoke.items():
    p = os.path.join(base, rel)
    tree = ast.parse(io.open(p, encoding="utf-8").read())
    defined = {n.name for n in ast.walk(tree) if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
    print(f"{rel:24s} 顶层定义 {len(defined):3d} 个")
# 校验 algorithms.py 无包内相对导入（单测直载的前提）
tree = ast.parse(io.open(os.path.join(base, "algorithms.py"), encoding="utf-8").read())
rels = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and (n.level or 0) > 0]
assert not rels, f"algorithms.py 不应有相对导入: {rels}"
print("algorithms.py 无相对导入 OK")
PY
```
Expected: 各文件打印定义数，末行 `algorithms.py 无相对导入 OK`。出现 `SyntaxError` 或断言失败即定位修正。

- [ ] **步骤 10：跑全部单测**

Run: `python -X utf8 -m unittest discover -s tests -p "test_*.py"`
Expected: `OK`，不少于 21（原有）+ 17（水印）+ 12（算法）= 50 个用例。

- [ ] **步骤 11：用户跑基准图比对（首次）**

用户在 ComfyUI 里重启加载千绘，用与 Task 0 **完全相同的 seed 与参数**重跑场景 A-E，与 `docs/superpowers/baseline/` 逐像素比对。

Expected: 5 个场景的图与 `检测信息` 与基准**完全一致**。

⚠ 注意：此时节点名已从 `LatentKSamplerUpscalerYOLO` 变为 `QianhuiLatentYOLO`，需在节点图中重新选择该节点，但**控件值要保持与基准一致**。

不一致 → 定位到对应 Mixin 文件排查，优先怀疑：漏搬方法、`controlnet.py` 的 `extract_hand_skeleton` 改造引入的差异、MRO 遮蔽。

- [ ] **步骤 12：提交**

```bash
git add nodes/qianhui_latent_yolo tests/test_latent_algorithms.py
git commit -m "refactor: Latent 修复拆分 —— 支撑类外移 + 巨型类切 6 个 Mixin

process() 本次未拆分（Task 4 处理），最大文件降至约 1500 行。

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

## Task 4: `process()` 阶段化

**Files:**
- Create: `nodes/qianhui_latent_yolo/mixins/context.py`
- Create: `nodes/qianhui_latent_yolo/mixins/pipeline/__init__.py`、`prepare.py`、`detect.py`、`repair.py`、`global_pass.py`、`report.py`
- Modify: `nodes/qianhui_latent_yolo/node.py`（`process` 瘦身为阶段编排）

**Interfaces:**
- Consumes: `QianhuiLatentYOLO` 的 6 个 Mixin（Task 3）
- Produces:
  - `context.py`：`YoloConfig`（字段名 = 原局部变量名）、`YoloRunContext`（`cfg: YoloConfig` 加跨阶段运行时字段）
  - `pipeline/prepare.py`：`parse_config(kwargs) -> YoloConfig`、`prepare_run(node, kwargs, Latent, VAE, ...) -> YoloRunContext`
  - `pipeline/detect.py`：`run_detection(node, ctx) -> None`（就地写 `ctx`）
  - `pipeline/repair.py`：`run_repair(node, ctx) -> None`
  - `pipeline/global_pass.py`：`run_global_pass(node, ctx) -> None`
  - `pipeline/report.py`：`build_report(node, ctx) -> tuple`（返回 `({"samples": ...}, preview, masks, info)`）

- [ ] **步骤 1：用扫描确定跨阶段变量集（不靠推断）**

Run:
```bash
python - <<'PY'
import ast, io, collections
SRC = r"E:\comfuyui 插件\水印\nodes.py"
# process() 的阶段边界（源文件行号，1-based，闭区间）
PHASES = [
    ("prepare_a", 4131, 4240),
    ("prepare_b", 4240, 4386),
    ("detect",    4387, 4862),
    ("repair",    4864, 5406),
    ("global",    5407, 5481),
    ("report",    5483, 5607),
]
lines = io.open(SRC, encoding="utf-8").read().splitlines()
def grab(a, b):
    return "\n".join(lines[a-1:b])

defs, uses = {}, {}
for name, a, b in PHASES:
    tree = ast.parse("\n".join("    " + l for l in grab(a, b).splitlines()[1:]) if False else grab(a, b).lstrip())
    d, u = set(), set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            (d if isinstance(n.ctx, ast.Store) else u).add(n.id)
    defs[name], uses[name] = d, u

order = [p[0] for p in PHASES]
cross = collections.OrderedDict()
for i, src in enumerate(order):
    for j in range(i+1, len(order)):
        dst = order[j]
        shared = (defs[src] & uses[dst])
        if shared:
            cross[f"{src} -> {dst}"] = sorted(shared)
for k, v in cross.items():
    print(f"\n{k}  ({len(v)} 个)")
    print("  " + ", ".join(v))
print("\n合计跨阶段变量:", sum(len(v) for v in cross.values()))
PY
```
Expected: 打印每对相邻/跨阶段共享的变量名清单。**这份清单就是 `YoloRunContext` 需要承载的字段集**——以脚本输出为准，不要凭记忆增删。

- [ ] **步骤 2：写 `context.py`**

`YoloConfig` 的字段名**逐字沿用原局部变量名**（含中文名如 `放大倍数`、`降噪`、`步数`），使搬运成为机械替换 `局部名` → `cfg.局部名`。

```python
"""process() 阶段间的状态容器。

YoloConfig: 99 个 kwargs 解析结果，字段名与原局部变量名逐字一致，
           使阶段搬运成为机械替换（局部名 → cfg.局部名）。
YoloRunContext: 跨阶段运行时状态 + cfg。
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import torch


@dataclass
class YoloConfig:
    """99 个 kwargs 解析结果。字段名 = 原局部变量名，逐字不改写。

    分组（与原文件注释一致）：ControlNet / 低显存性能 / 工作模式 /
    YOLO核心 / 高级检测 / 视频 / EMA / Tile / 合并过滤 / 智能策略 /
    手脚增强 / 骨架CN / 优先级规则 / 遮罩 / 修复采样 / 裁剪精修 /
    二次检测 / 融合。
    """
    # 字段由步骤 3 的脚本生成后粘贴于此，不要手写


@dataclass
class YoloRunContext:
    cfg: YoloConfig
    # 字段集**完全由步骤 1 的扫描输出决定**，不要预列：
    #   扫描输出里「前段定义、后段读取」的每个变量名 → 一个 ctx 字段
    #   扫描输出里没有的名字 → 不得出现（否则是死字段，审查会打回）
    # 参考类型映射：tensor 列表用 List[torch.Tensor] + field(default_factory=list)，
    #   标量用 Optional[...] = None 或 0 / 0.0，进度条等外部对象用 Any = None。
```

**不要照抄任何"看起来合理"的字段名**——例如 `frames`、`previews`、`masks` 这类名字是早期草案的猜测，源码里的真实变量名以扫描输出为准（很可能是 `out_latents`、`frame_previews`、`frame_masks` 之类）。字段名必须与源码局部变量名逐字一致，否则机械替换会全线断掉。
```

- [ ] **步骤 3：用脚本生成 `YoloConfig` 的 99 个字段并粘贴**

**不要手写这些字段**——从源码抽取，避免转录错误。把下面脚本存为仓库根的 `gen_yoloconfig.py`（用完即删，不入库）：

```python
import ast, io

SRC = r"E:\comfuyui 插件\水印\nodes.py"
lines = io.open(SRC, encoding="utf-8").read().splitlines()
seg = "\n".join(lines[4134:4240])          # 源 4135-4240：kwargs 解析区
tree = ast.parse(seg)


def unwrap(v):
    """剥掉 bool()/float()/int()/str() 外层包裹"""
    while (isinstance(v, ast.Call) and isinstance(v.func, ast.Name)
           and v.func.id in ("bool", "float", "int", "str")):
        v = v.args[0]
    return v


def ann(d):
    if isinstance(d, bool):
        return "bool"
    if isinstance(d, int):
        return "int"
    if isinstance(d, float):
        return "float"
    if isinstance(d, str):
        return "str"
    return "Any"


rows = []
for n in ast.walk(tree):
    if not isinstance(n, ast.Assign) or len(n.targets) != 1:
        continue
    tgt = n.targets[0]
    if not isinstance(tgt, ast.Name):
        continue
    v = unwrap(n.value)
    key = default = None
    if (isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute)
            and v.func.attr == "get"):
        key = ast.literal_eval(v.args[0])
        default = ast.literal_eval(v.args[1]) if len(v.args) > 1 else None
    elif isinstance(v, ast.Subscript):
        key = ast.literal_eval(v.slice)
    if key is None:
        continue            # 非 kwargs 派生的赋值：归属由步骤 1 的扫描决定
    rows.append((n.lineno, tgt.id, key, default))

rows.sort()
print("# 共 %d 个字段" % len(rows))
for _, var, key, default in rows:
    print("    %s: %s = %r   # 键: %s" % (var, ann(default), default, key))
```

Run: `python -X utf8 gen_yoloconfig.py`
Expected: 打印约 99 行字段声明。把输出**逐行粘贴**到 `YoloConfig` 类体内，替换掉 `# 字段由步骤 3 的脚本生成后粘贴于此，不要手写` 那一行。

⚠ **dataclass 字段顺序约束**：无默认值的字段（来自裸 `kwargs["键"]` 下标读取）必须排在有默认值字段**之前**，否则报 `TypeError: non-default argument follows default argument`。粘贴后用步骤 6 的语法检查确认。

⚠ **不要**把 `has_cn`、`perf_start_time`、`sam_model_path` 这类**派生值**放进 `YoloConfig`——它们不是 kwargs 直读，归属由步骤 1 的扫描结果决定（进 `ctx` 或留作局部变量）。

- [ ] **步骤 4：切 5 个阶段文件**

按 `process()` 的源行边界搬入，**保持原语句顺序与缩进层级语义**，不合并、不重排、不顺手优化；原注释一律搬走：

| 文件 | 源行 | 搬入内容 |
|---|---|---|
| `pipeline/prepare.py` | 4131-4386 | 99 个 `kwargs.get` → `YoloConfig` 字段；规则预设处理；模型列表解析；缓存设备决策；`ksampler` 准备 |
| `pipeline/detect.py` | 4387-4862 | 逐帧主循环前半：预处理 → 全图预降噪 → VAE 解码 → YOLO 检测 → 检测后处理 → 二次检测精修 → SAM 精修 → 中间变量清理 |
| `pipeline/repair.py` | 4864-5406 | `3) 区域精修`：逐目标裁剪 → 条件编码 → 局部重绘 → ControlNet → 融合贴回 |
| `pipeline/global_pass.py` | 5407-5481 | `4) 修复后全图降噪` + 视频模式时序平滑 + 帧历史维护 |
| `pipeline/report.py` | 5483-5607 | 预览图解码、遮罩汇总、`检测信息` 组装、性能统计、最终 return |

搬运规则：阶段函数内的局部变量若在步骤 1 的跨阶段清单里，改为 `ctx.<名字>`；其余保持为普通局部变量。

- [ ] **步骤 5：`node.py` 的 `process` 瘦身为编排**

```python
    def process(self, Latent, 模型, VAE, 正面提示, 负面提示, seed, 步数,
                CFG, 采样器, 调度器, 降噪, 放大算法, 放大倍数, **kwargs):
        ctx = prepare_run(self, kwargs, Latent=Latent, VAE=VAE,
                          正面提示=正面提示, 负面提示=负面提示, seed=seed,
                          放大算法=放大算法, 放大倍数=放大倍数)
        run_detection(self, ctx)
        run_repair(self, ctx)
        run_global_pass(self, ctx)
        return build_report(self, ctx)
```

- [ ] **步骤 6：静态健全性检查**

Run:
```bash
python - <<'PY'
import ast, io, os, re
base = "nodes/qianhui_latent_yolo"
for rel in ["mixins/context.py"] + [f"mixins/pipeline/{m}.py" for m in
            ("prepare", "detect", "repair", "global_pass", "report")]:
    p = os.path.join(base, rel)
    src = io.open(p, encoding="utf-8").read()
    ast.parse(src)
    print(f"{rel:36s} {len(src.splitlines()):5d} 行")
# 确认 process 已瘦身
src = io.open(os.path.join(base, "node.py"), encoding="utf-8").read()
tree = ast.parse(src)
for n in ast.walk(tree):
    if isinstance(n, ast.FunctionDef) and n.name == "process":
        print("process() 行数:", n.end_lineno - n.lineno + 1)
PY
```
Expected: 各文件行数打印；`process()` 行数 ≤ 15。
若 `process()` 仍接近 1500 行，说明阶段未真正搬出。

- [ ] **步骤 7：跑全部单测**

Run: `python -X utf8 -m unittest discover -s tests -p "test_*.py"`
Expected: `OK`，仍是 50 个用例。

- [ ] **步骤 8：用户跑基准图比对（关键检查点）**

用户重启 ComfyUI，同 seed 同参数重跑场景 A-E，与基准逐像素 + `检测信息` 逐字符比对。

Expected: **5 个场景全部完全一致**。

⚠ 这一步是本计划风险最高的验证点——`process()` 拆分必然改动代码文本（局部变量 → `ctx.` 引用），无法用"逐字节一致"证明等价。不一致时按阶段二分：先注释掉 `run_global_pass` 与 `run_repair` 留 `detect`，逐段收窄。

- [ ] **步骤 9：提交**

```bash
git add nodes/qianhui_latent_yolo/mixins/context.py nodes/qianhui_latent_yolo/mixins/pipeline nodes/qianhui_latent_yolo/node.py
git commit -m "refactor: process() 阶段化 —— YoloConfig/YoloRunContext + 5 个阶段函数

1477 行单方法拆成 5 个 100-570 行的阶段函数，跨阶段状态显式收进 ctx。

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

## Task 5: 落地 `配置预设`

**Files:**
- Create: `nodes/qianhui_latent_yolo/presets.py`
- Create: `js/latent_yolo_presets.js`
- Modify: `nodes/qianhui_latent_yolo/mixins/pipeline/prepare.py`（`parse_config` 末尾应用预设）
- Modify: `nodes/qianhui_latent_yolo/__init__.py`（导出预设表查询接口）
- Modify: `__init__.py`（新增 `/qianhui_preset_data` 路由）
- Modify: `tests/test_latent_algorithms.py`（追加预设单测）

**Interfaces:**
- Consumes: `YoloConfig`（Task 4）
- Produces:
  - `presets.py`：`PRESETS: dict[str, dict[str, object]]`（5 档）、`NO_PRESET = "[不使用预设]"`、`CUSTOM = "自定义"`、`apply_preset(cfg, name) -> None`（就地改 `cfg`）
  - `tests` 可断言 `apply_preset` 的字段覆盖
  - HTTP `GET /qianhui_preset_data` → `{"presets": {...}, "labels": [...]}`

- [ ] **步骤 1：写失败测试**

在 `tests/test_latent_algorithms.py` 末尾追加：

```python
class TestPresets(unittest.TestCase):
    def _cfg(self):
        from dataclasses import fields
        pres = load_module(
            "qh_presets",
            os.path.join(pkgs_dir("qianhui_latent_yolo"), "presets.py"),
        )
        return pres

    def test_five_presets_defined(self):
        pres = self._cfg()
        self.assertEqual(
            sorted(pres.PRESETS.keys()),
            sorted(["快速模式(低质量)", "平衡模式(推荐)", "质量模式(慢速)",
                    "极致质量(超慢)", "省显存模式"]),
        )

    def test_no_preset_is_noop(self):
        pres = self._cfg()
        cfg = type("C", (), {})()
        cfg.区域步数 = 999
        pres.apply_preset(cfg, pres.NO_PRESET)
        self.assertEqual(cfg.区域步数, 999)
        pres.apply_preset(cfg, pres.CUSTOM)
        self.assertEqual(cfg.区域步数, 999)

    def test_fast_preset_overrides(self):
        pres = self._cfg()
        cfg = type("C", (), {})()
        for k in pres.PRESETS["快速模式(低质量)"]:
            setattr(cfg, k, None)
        pres.apply_preset(cfg, "快速模式(低质量)")
        self.assertEqual(cfg.区域步数, 12)
        self.assertEqual(cfg.修复轮次, 1)
        self.assertFalse(cfg.多尺度检测)

    def test_extreme_is_heaviest(self):
        pres = self._cfg()
        fast = pres.PRESETS["快速模式(低质量)"]
        extreme = pres.PRESETS["极致质量(超慢)"]
        self.assertLess(fast["区域步数"], extreme["区域步数"])
        self.assertLessEqual(fast["修复轮次"], extreme["修复轮次"])

    def test_missing_field_does_not_raise(self):
        """cfg 缺字段时只覆盖存在的键，不抛异常"""
        pres = self._cfg()
        cfg = type("C", (), {})()
        pres.apply_preset(cfg, "快速模式(低质量)")
```

- [ ] **步骤 2：跑测试，确认失败**

Run: `python -X utf8 tests/test_latent_algorithms.py`
Expected: FAIL——`FileNotFoundError`（`presets.py` 未创建）。

- [ ] **步骤 3：写 `presets.py`**

```python
"""配置预设表（后端唯一来源，前端 JS 通过 /qianhui_preset_data 拉取）。

覆盖 19 个参数 × 5 档。不纳入预设的参数表达用户意图，见设计文档 2.6。
"""

NO_PRESET = "[不使用预设]"
CUSTOM = "自定义"

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

    NO_PRESET 与 CUSTOM 均为「不覆盖」；cfg 缺字段时静默跳过该键。
    """
    if not name or name in (NO_PRESET, CUSTOM):
        return
    table = PRESETS.get(name)
    if not table:
        return
    for key, value in table.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)


def as_payload() -> dict:
    """供 /qianhui_preset_data 路由返回。"""
    return {"presets": PRESETS, "labels": PRESET_LABELS}
```

- [ ] **步骤 4：跑测试，确认通过**

Run: `python -X utf8 tests/test_latent_algorithms.py`
Expected: `OK`，17 个用例全过（原 12 + 预设 5）。

- [ ] **步骤 5：在 `parse_config` 末尾接上预设**

在 `pipeline/prepare.py` 的 `parse_config` 返回前插入：

```python
    from ...presets import apply_preset
    apply_preset(cfg, getattr(cfg, "配置预设", None) or "[不使用预设]")
```

**相对导入层级按实际目录调整**（`prepare.py` 在 `mixins/pipeline/` 下，指向 `qianhui_latent_yolo/presets.py` 需三层 `.`）。若层级不确定，改用绝对式 `from nodes.qianhui_latent_yolo.presets import apply_preset` 会失败（`nodes/` 是命名空间包且父包 `__init__` 依赖 ComfyUI），所以**必须用相对导入**，写完后用步骤 6 验证。

- [ ] **步骤 6：验证导入层级正确**

Run:
```bash
python - <<'PY'
import ast, io
src = io.open("nodes/qianhui_latent_yolo/mixins/pipeline/prepare.py", encoding="utf-8").read()
tree = ast.parse(src)
for n in ast.walk(tree):
    if isinstance(n, ast.ImportFrom) and "presets" in (n.module or ""):
        print("level =", n.level, "module =", n.module)
        assert n.level == 3, f"预期 level=3（pipeline → mixins → qianhui_latent_yolo），实际 {n.level}"
print("预设导入层级 OK")
PY
```
Expected: `level = 3` 且 `预设导入层级 OK`。

- [ ] **步骤 7：在包 `__init__.py` 导出预设载荷**

```python
from .presets import as_payload as preset_payload
```

并加入 `__all__`。

- [ ] **步骤 8：加 `/qianhui_preset_data` 路由**

在 `__init__.py` 现有 `# 8b. 前端联动数据路由` 块之后追加（沿用同样的静默失败模式）：

```python
# 8c. 配置预设数据路由（注册失败不影响节点）
try:
    from aiohttp import web
    from server import PromptServer
    from .nodes.qianhui_latent_yolo import preset_payload

    @PromptServer.instance.routes.get("/qianhui_preset_data")
    async def qianhui_preset_data(request):
        return web.json_response(preset_payload())
except Exception:
    pass
```

- [ ] **步骤 9：写前端联动 `js/latent_yolo_presets.js`**

```javascript
// 千绘 · Latent智能修复 —— 配置预设联动
// 兼容写法：老前端提供全局 app；新前端(>=0.31) 以 ES module 执行扩展，
// 裸 app 标识符会 ReferenceError，需动态 import 兜底。
(async function () {
    "use strict";
    const WIDGET = "配置预设";
    let presetData = null;

    async function getApp() {
        if (typeof app !== "undefined") return app;
        if (globalThis.app) return globalThis.app;
        try {
            const mod = await import("../../scripts/app.js");
            return mod.app || globalThis.app || null;
        } catch (e) {
            return null;
        }
    }

    function applyToNode(node, name) {
        if (!presetData) return;
        const table = (presetData.presets || {})[name];
        if (!table) return;
        for (const key of Object.keys(table)) {
            const w = node.widgets.find(function (x) { return x.name === key; });
            if (!w) continue;
            w.value = table[key];
            if (typeof w.callback === "function") w.callback(w.value);
        }
    }

    const appRef = await getApp();
    if (!appRef) return;

    appRef.registerExtension({
        name: "qianhui.latent_yolo_presets",
        async beforeRegisterNodeDef(nodeType, nodeData) {
            if (nodeData.name !== "QianhuiLatentYOLO") return;
            const orig = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const ret = orig ? orig.apply(this, arguments) : undefined;
                const node = this;
                if (!presetData) {
                    fetch("/qianhui_preset_data")
                        .then(function (r) { return r.json(); })
                        .then(function (d) { presetData = d; })
                        .catch(function () { });   // 拉取失败：后端兜底仍生效
                }
                const w = this.widgets.find(function (x) { return x.name === WIDGET; });
                if (w) {
                    w.callback = (function (prev, nd) {
                        return function (v) {
                            const r = prev ? prev.call(this, v) : undefined;
                            applyToNode(nd, v);
                            return r;
                        };
                    })(w.callback, node);
                }
                return ret;
            };
        },
    });
})();
```

- [ ] **步骤 10：用户验证两条生效路径**

浏览器路径：重启 ComfyUI，在 `千绘 · Latent智能修复` 节点上把 `配置预设` 从 `[不使用预设]` 切到 `快速模式(低质量)`，确认 `区域步数` 控件显示值从 20 变为 12（**看得见**）。

API 路径：用 `[不使用预设]` 存一份工作流，改为 `快速模式(低质量)` 后用 API 格式提交（或脚本调用），确认后端生效——即运行日志/`检测信息` 反映的是预设值。

- [ ] **步骤 11：跑全部单测 + 提交**

Run: `python -X utf8 -m unittest discover -s tests -p "test_*.py"`
Expected: `OK`，不少于 55 个用例。

```bash
git add nodes/qianhui_latent_yolo/presets.py nodes/qianhui_latent_yolo/__init__.py \
        nodes/qianhui_latent_yolo/mixins/pipeline/prepare.py js/latent_yolo_presets.js \
        __init__.py tests/test_latent_algorithms.py
git commit -m "feat: 接上失效的 配置预设 控件（后端兜底 + 前端联动 + 数据路由）

预设表后端唯一定义，JS 走 /qianhui_preset_data 拉取；API 无头调用亦生效。

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

## Task 6: 文档同步

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: Task 1-5 的最终结构
- Produces: 与代码一致的 README

- [ ] **步骤 1：更新来源插件表与节点列表**

在 README「由四个历史插件整合而成」表中加一行：

```markdown
| 水印插件 | Latent YOLO 智能修复（含 SAM/ControlNet）、水印添加、水印提取 |
```

「节点列表」表追加三行：

```markdown
| 千绘 · Latent智能修复 | `QianhuiLatentYOLO` | YOLO 多尺度检测 + Tile 分块 + Seg 分割 + SAM 精修 + ControlNet + 手脚增强；`配置预设` 五档一键调速/调质 |
| 千绘 · 水印添加 | `QianhuiWatermarkAdd` | 可见水印（文字/图片平铺）与 DCT-QIM 隐形水印；依赖 `reedsolo` |
| 千绘 · 水印提取 | `QianhuiWatermarkExtract` | 提取 DCT-QIM 隐形水印文本 |
```

- [ ] **步骤 2：更新目录结构树与依赖说明**

「目录结构」补：

```
│   ├── qianhui_latent_yolo/ # Latent 智能修复（支撑模块 + 6 Mixin + 5 阶段函数）
│   └── qianhui_watermark/   # 水印添加 / 提取
├── requirements.txt     # Python 依赖（reedsolo）
```

「注意事项」补一条：

```markdown
- `Latent 智能修复` 与 `水印` 模块需要额外依赖：`pip install reedsolo`（隐形水印的 Reed-Solomon 纠错）。未安装时可见水印仍可用，隐形水印会报出带安装指引的错误。
- 整合自水印插件的 3 个节点**不保留旧注册名**（`LatentKSamplerUpscalerYOLO`、`WatermarkAdd`、`WatermarkExtract` 及中文别名），旧 workflow 需重新选择节点。
```

- [ ] **步骤 3：提交**

```bash
git add README.md
git commit -m "docs: README 补入整合的 3 个节点、目录结构与 reedsolo 依赖说明

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

## 完成标准

- [ ] `python -X utf8 -m unittest discover -s tests -p "test_*.py"` 全绿，且用例数 ≥ 55
- [ ] ComfyUI 启动日志中出现 `千绘 · Latent智能修复 (YOLO+SAM+CN V6.11.3)`、`千绘 · 水印添加`、`千绘 · 水印提取`，无 ❌
- [ ] 用户基准图比对：场景 A-E 全部与 `docs/superpowers/baseline/` 完全一致
- [ ] `配置预设` 在浏览器与 API 两条路径下均生效
- [ ] `nodes/qianhui_latent_yolo/` 下最大文件 ≤ 800 行
- [ ] 旧注册名（含中文别名）在节点菜单中不再出现
