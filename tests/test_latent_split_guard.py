# -*- coding: utf-8 -*-
"""节点拆分守卫：跨节点载荷的不变量（T1）与阶段模块 cfg 读取的闭合（T2）。

这组用例护的是拆分工程的**地基**。4 个节点之间传的检测数据都装在
`DetectionPayload` 里，而它的两条不变量（帧数对齐、帧号对齐）一旦破了，
全局降噪节点的视频时序平滑会**静默**给出错误画面 —— 不报错、不崩，
只是结果不对。所以这里把不变量钉死在测试里。

`payload.py` 按硬约束**没有包内相对导入**，因此可以和 algorithms.py 一样
用 `_loader.load_module` 按路径直载，不需要桩 ComfyUI 运行时。

`TestCfgReadClosure` 那一组是拆分的**第二层地基**：4 个节点的控件表是从
「哪个阶段读哪些 `cfg.<字段>`」推出来的，推错了的症状是某个旋钮在界面上
消失而没有任何测试变红，所以先用 AST 把字段表 ↔ 解析器写入 ↔ 阶段读取
↔ 控件这四张表双向闭合起来。
"""
import ast
import io
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _loader import ROOT, load_module  # noqa: E402
from _loader import pkgs_dir  # noqa: E402
from test_latent_algorithms import _context_key_map, _latent_node_widget_decls  # noqa: E402

payload = load_module(
    "qh_payload", os.path.join(ROOT, "nodes", "qianhui_latent_yolo", "payload.py")
)


def _frame(i, detections=None, processed=True):
    """构造一帧最小可用的检测帧。"""
    return payload.DetectionFrame(
        i=i,
        processed=processed,
        detections=[] if detections is None else detections,
        pixel_h=512,
        pixel_w=512,
        latent_h=64,
        latent_w=64,
    )


class TestPayloadTypeName(unittest.TestCase):
    def test_type_name_is_the_registered_type_string(self):
        """TYPE_NAME 是节点 class 里写死的注册串，改一个字就断线。

        前端连线和 ComfyUI 的图校验都按这个字符串匹配，所以它必须是
        `QIANHUI_DETECT`，且不能是空串或空白（空串会让所有自定义类型
        互相串味：任何节点都能接任何东西，接线错误不再被拦下）。
        """
        self.assertEqual(payload.TYPE_NAME, "QIANHUI_DETECT")
        self.assertTrue(payload.TYPE_NAME.strip(), "TYPE_NAME 不能是空白串")


class TestIndexAlignment(unittest.TestCase):
    def test_frame_index_alignment_is_asserted(self):
        """两条不变量各自真的会拦人 —— 数不对拦一次，序不对再拦一次。"""
        # 情况 A：帧数比 batch_size 少（中间节点丢了一帧）
        short = payload.DetectionPayload(
            frames=[_frame(0), _frame(1)], batch_size=3
        )
        with self.assertRaises(ValueError) as ctx:
            list(payload.iter_frames_in_order(short))
        msg = str(ctx.exception)
        self.assertIn("2", msg, f"错误消息要带上实际帧数，实际是：{msg}")
        self.assertIn("3", msg, f"错误消息要带上声明的 batch_size，实际是：{msg}")

        # 情况 B：数量对、但帧号错位（顺序被打乱过）
        swapped = payload.DetectionPayload(
            frames=[_frame(1), _frame(0)], batch_size=2
        )
        with self.assertRaises(ValueError) as ctx:
            list(payload.iter_frames_in_order(swapped))
        msg = str(ctx.exception)
        self.assertIn("1", msg, f"错误消息要带上出错的帧号，实际是：{msg}")

    def test_iter_frames_in_order_yields_in_index_order(self):
        """顺序产出，且产出的是**原对象**而不是副本。

        `assertIs` 而非 `assertEqual`：如果哪天有人在 yield 之前套一层
        `list()` / `copy()` / 重建 dataclass，值还能相等，但下游对帧的
        追加改动上游就看不见了 —— 那正是「静默失效」的形态。
        """
        frames = [_frame(0), _frame(1), _frame(2)]
        p = payload.DetectionPayload(frames=frames, batch_size=3)
        out = list(payload.iter_frames_in_order(p))
        self.assertEqual([f.i for f in out], [0, 1, 2])
        for idx, got in enumerate(out):
            self.assertIs(got, p.frames[idx])

    def test_detections_pass_through_by_reference(self):
        """检测记录必须按引用穿过载荷，不能在中途被拷贝。"""
        records = [{"class_name": "person", "class_id": 0, "confidence": 0.9}]
        p = payload.DetectionPayload(frames=[_frame(0, records)], batch_size=1)
        got = next(iter(payload.iter_frames_in_order(p)))
        self.assertIs(got.detections, records)

        # 证明没有拷贝：往原 list 里追加，通过载荷也看得见
        records.append({"class_name": "cat", "class_id": 1, "confidence": 0.5})
        self.assertEqual(len(p.frames[0].detections), 2)
        self.assertIs(p.frames[0].detections[1], records[1])


class TestPayloadDefaults(unittest.TestCase):
    def test_payload_defaults(self):
        """默认值 + `default_factory`（经典可变默认参数陷阱）。

        若 `stage_flags` / `perf` 写成 `= {}`，两个实例会**共享同一个 dict**：
        A 阶段记的耗时会在 B 阶段里冒出来，且这种污染只在同批次多图时出现，
        单图测试永远绿。所以这里专门验证两个实例不是同一个对象。
        """
        a = payload.DetectionPayload(frames=[], batch_size=0)
        b = payload.DetectionPayload(frames=[], batch_size=0)
        self.assertIs(a.interrupted, False)
        self.assertEqual(a.stage_flags, {})
        self.assertEqual(a.perf, {})
        self.assertIsNot(
            a.stage_flags, b.stage_flags, "stage_flags 用了可变默认参数"
        )
        self.assertIsNot(a.perf, b.perf, "perf 用了可变默认参数")


class TestSlots(unittest.TestCase):
    def test_frames_have_no_dict(self):
        """两个容器都带 `slots=True` —— 写错字段名当场 AttributeError。

        与 test_latent_algorithms.py 对 YoloConfig 的写法同形。
        """
        frame = _frame(0)
        p = payload.DetectionPayload(frames=[frame], batch_size=1)
        for obj, name in ((frame, "DetectionFrame"), (p, "DetectionPayload")):
            self.assertFalse(
                hasattr(obj, "__dict__"),
                f"{name} 应带 @dataclass(slots=True)；一旦有了 __dict__，"
                f"写错字段名就会静默新建属性、真字段不动",
            )
            with self.assertRaises(AttributeError):
                obj.definitely_not_a_field = 1


# ==================== 阶段模块的 cfg 访问（ast 解析，不 import） ====================
#
# 为什么不 import：这些模块 `import comfy.*`，还要 ComfyUI 运行时；静态解析
# 与 test_latent_algorithms.py 的 _latent_node_widget_decls 同一理由。
#
# ★ 必须区分 Load 与 Store：prepare.py 的 99 处是「写」（parse_config 给 cfg 赋值），
#   各阶段模块的才是「读」。不区分的话两个方向会算成一个。

PIPELINE_DIR = os.path.join(pkgs_dir("qianhui_latent_yolo"), "mixins", "pipeline")
STAGE_MODULES = ("prepare.py", "detect.py", "repair.py", "global_pass.py", "report.py")

# 6 个冻结计数 (reads, writes)。
# ★ 改动这些数字**必须在同一个提交里说明为什么** —— 它们不是「待填的期望值」，
#   而是反空转的绊线：某个阶段模块被改名/搬走时并集变小，用例 2/3 会因为
#   「并集 ⊆ 字段」而**空转通过**，只有这里会让「读到 0 个」当场变红。
#   预期会触发它的改动：T4 的 SAM3 分支、T12 的 apply_preset 新签名、
#   T13 删 prepare_run —— 每次都要有理由，不是无脑改数字。
EXPECTED_CFG_ACCESS = {
    "prepare.py": (48, 99),  # (reads, writes)
    "detect.py": (37, 0),
    "repair.py": (48, 0),
    "global_pass.py": (7, 0),
    "report.py": (6, 0),
}

# 有控件、但没有对应 YoloConfig 字段的 14 个名字（13 个必填 socket/参数 + 配置预设）。
# ★ 这是**故意的绊线**：以后新增控件时，要么给它一个 YoloConfig 字段 +
#   `# 键:` 注释，要么**显式**把这个名字加进本常量 —— 两个都不做，用例 6 就红。
# 注意 步数 / CFG / 降噪 出现在这里是**对的**：它们被 repair.py 当函数参数用
# （未检测时回退），不是 YoloConfig 字段，所以没有 `# 键:` 注释。
WIDGETS_WITHOUT_CONFIG_FIELD = frozenset((
    "Latent", "VAE", "模型", "正面提示", "负面提示", "seed", "步数", "CFG",
    "降噪", "采样器", "调度器", "放大倍数", "放大算法", "配置预设",
))


def _cfg_access(module_path):
    """AST 取一个模块里对 cfg 字段的访问，拆成 (reads, writes) 两个 set。

    只认两种基底：`cfg.<字段>`（prepare.py 的形参）与 `ctx.cfg.<字段>`（各阶段
    从上下文取）。两者之外的一律不算 —— 免得把 `foo.cfg` 之类误收进来。
    """
    with io.open(module_path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    reads, writes = set(), set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        base = node.value
        is_bare_cfg = isinstance(base, ast.Name) and base.id == "cfg"
        is_ctx_cfg = isinstance(base, ast.Attribute) and base.attr == "cfg"
        if not (is_bare_cfg or is_ctx_cfg):
            continue
        (reads if isinstance(node.ctx, ast.Load) else writes).add(node.attr)
    return reads, writes


def _yolo_config_fields():
    """context.py 里 YoloConfig 的字段名集合（ast，不 import）。

    与 test_latent_algorithms.py:_context_key_map 同样的定位法：
    找 ClassDef 名为 YoloConfig，取它 body 里 AnnAssign 的 target.id。
    """
    path = os.path.join(pkgs_dir("qianhui_latent_yolo"), "mixins", "context.py")
    with io.open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    cls = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "YoloConfig":
            cls = node
            break
    assert cls is not None, "context.py 里找不到 YoloConfig 类"
    return {
        st.target.id
        for st in cls.body
        if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name)
    }


def _stage_cfg_access():
    """{模块文件名: (reads, writes)}，键就用 STAGE_MODULES 里那几个名字。"""
    return {
        name: _cfg_access(os.path.join(PIPELINE_DIR, name)) for name in STAGE_MODULES
    }


class TestCfgReadClosure(unittest.TestCase):
    """cfg 读取的静态守卫：字段表 ↔ 解析器写入 ↔ 阶段读取 ↔ 控件，四张表双向闭合。

    拆成 4 个节点之前先把网建好：拆完之后每个节点的控件表要跟这张网对账，
    否则「某个旋钮在界面上消失」不会有任何测试变红，只能靠肉眼看出来。

    前 5 条不碰 node.py，拆完之后依然成立；后 2 条依赖 node.py，T13 删单体时
    预期要跟着改，所以刻意分成独立方法、不揉成一条。
    """

    def test_every_config_field_is_written_by_parse_config(self):
        """prepare.py 的 `cfg.<字段> = ...` 与 YoloConfig 字段表严格一一对应。

        同时钉住两个方向：解析器漏接一个字段（字段多出来），
        以及写了一个不存在的字段（拼写错）。
        """
        _, writes = _cfg_access(os.path.join(PIPELINE_DIR, "prepare.py"))
        fields = _yolo_config_fields()
        self.assertEqual(
            writes,
            fields,
            "prepare.py 的 cfg 赋值集合必须与 YoloConfig 字段表严格一一对应：\n"
            f"  写了但没有对应字段（拼写错？）：{sorted(writes - fields)}\n"
            f"  有字段但解析器没接（界面上没这个旋钮？）：{sorted(fields - writes)}",
        )

    def test_every_read_field_is_a_declared_config_field(self):
        """5 个阶段读到的字段都必须真在 YoloConfig 里 —— 这是**拼写错**的网。

        `ctx.cfg.confidnce` 这类错字在带 `slots=True` 的类上会在运行期抛
        AttributeError，但只在该分支被走到时才炸；静态这里当场红。
        """
        reads = set()
        for stage_reads, _ in _stage_cfg_access().values():
            reads |= stage_reads
        fields = _yolo_config_fields()
        self.assertEqual(
            reads - fields,
            set(),
            "阶段模块读到了 YoloConfig 里不存在的字段（拼写错？）："
            f"{sorted(reads - fields)}",
        )

    def test_every_config_field_is_read_by_some_stage(self):
        """每个字段都必须被某个阶段读到 —— 这是**死字段**的网。

        声明了、解析器也接了、`# 键:` 注释也写了，但没有任何阶段用它：
        界面上有个旋钮其实什么都没干，而且不会报错。
        """
        reads = set()
        for stage_reads, _ in _stage_cfg_access().values():
            reads |= stage_reads
        fields = _yolo_config_fields()
        self.assertEqual(
            fields - reads,
            set(),
            "这些字段从未被任何阶段读取（界面上的旋钮是死的）："
            f"{sorted(fields - reads)}",
        )

    def test_every_config_field_has_a_key_comment(self):
        """字段 ↔ 控件的桥：context.py 里每个字段都要有 `# 键: <控件名>` 注释。

        T8~T11 的「每节点字段 ⊆ 其控件」就靠 _context_key_map() 把字段名翻成
        控件名，少一处注释就有一个字段在拆分后找不到自己的控件。
        """
        fields = _yolo_config_fields()
        key_map = _context_key_map()
        self.assertEqual(
            set(key_map),
            fields,
            "YoloConfig 字段表与 `# 键:` 注释表必须完全一致：\n"
            f"  有字段但缺 `# 键:` 注释：{sorted(fields - set(key_map))}\n"
            f"  有注释但没有对应字段：{sorted(set(key_map) - fields)}",
        )

    def test_every_mapped_widget_exists_and_is_unique(self):
        """`# 键:` 指到的控件必须真在 node.py 控件表里，且不能两个字段指同一个。

        第二条防的是「复制粘贴注释时把新字段的注释也写成旧控件名」——
        那样两个字段抢一个旋钮，其中一个永远改不动。
        """
        mapped = list(_context_key_map().values())
        widgets = _latent_node_widget_decls()
        counts = {}
        for name in mapped:
            counts[name] = counts.get(name, 0) + 1
        dupes = sorted(name for name, n in counts.items() if n > 1)

        self.assertEqual(
            set(mapped) - set(widgets),
            set(),
            "`# 键:` 指到了 node.py 里不存在的控件："
            f"{sorted(set(mapped) - set(widgets))}",
        )
        self.assertEqual(dupes, [], f"多个字段映射到同一个控件名：{dupes}")

    def test_widgets_without_a_config_field_are_exactly_the_known_set(self):
        """★ 故意的绊线：没有 cfg 字段的控件**精确等于**那 14 个已知名字。

        以后新增控件时二选一：要么给它一个 YoloConfig 字段 + `# 键:` 注释，
        要么**显式**把这名字加进 WIDGETS_WITHOUT_CONFIG_FIELD 常量；
        两个都不做，这条就红。拆成 4 个节点后，本用例由每节点各自的
        socket/控件表接替（T8~T11）。
        """
        widgets = set(_latent_node_widget_decls())
        mapped = set(_context_key_map().values())
        for name in mapped:
            widgets.discard(name)
        known = set(WIDGETS_WITHOUT_CONFIG_FIELD)
        self.assertEqual(
            widgets,
            known,
            "没有 YoloConfig 字段的控件集合变了。新增控件请二选一：\n"
            "  (a) 在 context.py 的 YoloConfig 里加字段，并在同一行写 `# 键: <控件名>`；\n"
            "  (b) 确认它本来就不该有字段，再显式加进 WIDGETS_WITHOUT_CONFIG_FIELD。\n"
            f"  多出来的（两边都没做）：{sorted(widgets - known)}\n"
            f"  少了的（已配好字段/注释，或控件被改名）：{sorted(known - widgets)}",
        )

    def test_cfg_access_counts_are_frozen(self):
        """★ 反空转的网：6 个计数写死，阶段模块被改名/搬走时「读到 0 个」当场红。

        用例 2/3 是「并集 ⊆ 字段」/「字段 ⊆ 并集」的单向包含断言：某个阶段模块
        一旦解析不到（改名、搬走），并集变小，它们会**空转通过**。这条把每个
        模块的 (reads, writes) 计数钉死，让空转变成红。

        ★ 改动这些数字**必须在同一个提交里说明为什么**。
        预期会触发它的改动：T4 的 SAM3 分支、T12 的 apply_preset 新签名、
        T13 删 prepare_run —— 每次都要有理由，不是无脑改数字。
        """
        actual = {
            name: (len(reads), len(writes))
            for name, (reads, writes) in _stage_cfg_access().items()
        }
        self.assertEqual(
            actual,
            EXPECTED_CFG_ACCESS,
            "各阶段模块的 cfg 访问计数变了，每项都是 (reads, writes)。变动时：\n"
            "  1) 先确认是真改了读取逻辑，而不是模块被改名/搬走、或解析失手；\n"
            "  2) 在**同一个提交**里说明为什么，并同步更新 EXPECTED_CFG_ACCESS。",
        )


if __name__ == "__main__":
    unittest.main()
