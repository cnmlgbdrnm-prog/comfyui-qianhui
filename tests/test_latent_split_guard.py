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

`TestSlimContexts` 那一组是拆分的**第三层地基**（T6）：4 个精简 Context 的
字段归属。它护的是「静默失效」里最阴的一种 —— 某个 `ctx.` 字段**在 4 个
Context 里都没有家**，那个节点就会在运行期 `AttributeError`；`slots=True`
会当场炸，但**只在该分支被走到时才炸**，界面上旋钮还在、改了没用、平时不报错。
所以这里把「字段归属」用 AST 与阶段模块的真实读取双向闭合起来。
"""
import ast
import io
import os
import sys
import types
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
STAGE_MODULES = ("prepare.py", "detect.py", "repair.py",
                 "global_pass.py", "report.py", "report_bits.py",
                 "sam_refine.py")

# 7 个冻结计数 (reads, writes)。
#
# ★ 口径：数的是**去重后的字段名个数**（`{字段名}` 的 len），**不是** AST 访问点个数。
#   两者的差别很大 —— 例如 repair.py 磁盘上有 114 处 cfg 读取，但只有 48 个不同的字段名。
#   所以「删掉一处重复读取」这类改动**不会**让这里变红（那是刻意的：重复读取的增减
#   不改变字段归属，而字段归属才是本文件要护的东西）。
#
# ★ 改动这些数字**必须在同一个提交里说明为什么** —— 它们不是「待填的期望值」，
#   而是反空转的绊线：某个阶段模块被改名/搬走时并集变小，用例 2/3 会因为
#   「并集 ⊆ 字段」而**空转通过**，只有这里会让「读到 0 个」当场变红。
#   预期会触发它的改动：T4 的 SAM3 分支、T12 的 apply_preset 新签名、
#   T13 删 prepare_run —— 每次都要有理由，不是无脑改数字。
EXPECTED_CFG_ACCESS = {
    "prepare.py": (48, 99),  # (reads, writes)
    # T4：SAM 精修块整段搬进了 sam_refine.py，带走了块内**仅有**的两个 cfg 读取
    # （enable_sam 在 :367 的 if 条件里、sam_conf_threshold 在 :408 的调用里；
    # 机检确认这两个字段在 detect.py 的 SAM 块**之外**一处都没读）。整条管道的
    # 读取一个没少、只是换了文件 —— 非行为性迁移，在此同步期望值。
    "detect.py": (35, 0),
    # T4 新增：不登记的话新文件完全没守卫看着（它手里有那 2 个 cfg 读取）。
    # （并集仍覆盖全部 99 个字段，故用例 2/3 不受影响，只有本条需要更新。）
    "sam_refine.py": (2, 0),
    "repair.py": (48, 0),
    "global_pass.py": (7, 0),
    # T3：6 个 cfg 读取（work_mode / blend_mode / use_latent_aware / enable_yolo /
    # enable_crop / enable_profiling）随「预览 / 遮罩 / 检测信息」三段搬进了
    # report_bits.py，故 report.py 只剩 finalize_interrupt 里的 use_latent_aware。
    # 整条管道的读取一个没少、只是换了文件 —— 非行为性迁移，在此同步期望值。
    "report.py": (1, 0),
    # T3 新增：不登记的话新文件完全没守卫看着。
    # （并集仍覆盖全部字段，故用例 2/3 不受影响，只有本条需要更新。）
    "report_bits.py": (6, 0),
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

# 99 条「字段 ↔ 控件」配对快照，从 _context_key_map() 的当前输出生成（按字段名排序）。
#
# 这 99 条配对是 hand-written 的（context.py 里每个字段旁的 `# 键:` 注释），今天没有任何
# 自动化手段能判断「结果缓存设备」是否**语义上**就是 cache_device_mode —— 所以这里钉住的
# 是**当前这份配对**，不是它的正确性。它是绊线：改动必须写理由
# （改字段名/控件名/新增字段是正当的；**成对交换两处注释**是事故）。
# ★ 不要声称它证明了配对是对的。它证明的是「配对没有变」。
#
# 为什么需要它（复审 C-1）：用例 4 只断言**键集合**相等、用例 5 只断言**值集合** ⊆ 控件表
# 且互不重复。成对交换两处 `# 键:` 注释时这三个不变量全部保持、三条断言全绿 ——
# 而 _context_key_map() 正是 T8~T11 把字段翻成控件的唯一来源，交换后 4 个节点会把旋钮
# 挂到错误的字段上（改 A 旋钮、B 字段变），全量测试一条都不红。本条断言填的就是那个洞。
CONTEXT_KEY_SNAPSHOT = {
    "CN图像": "CN图像",
    "CN开始": "CN开始",
    "CN强度": "CN强度",
    "CN模型": "CN模型",
    "CN结束": "CN结束",
    "adaptive_dn": "按置信度自适应降噪",
    "attention_focus": "注意力聚焦强度",
    "bbox_expand2_px": "合并后二次扩张像素",
    "bbox_expand2_ratio": "合并后二次扩张比例",
    "blend_mode": "融合模式",
    "cache_device_mode": "结果缓存设备",
    "clarity_strength": "清晰度恢复强度",
    "class_rules_preset": "类别规则预设",
    "class_rules_text": "类别规则",
    "clear_each": "每帧清理缓存",
    "color_strength": "色差修复强度",
    "conf_high_scale": "自适应降噪_高置信缩放",
    "conf_low_scale": "自适应降噪_低置信缩放",
    "confidence": "置信度",
    "confidence2": "YOLO模型2置信度",
    "confidence3": "YOLO模型3置信度",
    "confidence4": "YOLO模型4置信度",
    "crop_cfg": "裁剪CFG",
    "crop_denoise": "裁剪降噪",
    "crop_max_edge": "裁剪最大边",
    "crop_pad": "裁剪扩展",
    "crop_rounds": "裁剪轮次",
    "crop_steps": "裁剪步数",
    "crop_upscale": "裁剪放大倍率",
    "crop_use_cn": "裁剪使用ControlNet",
    "custom_priority": "自定义优先级关键词",
    "decay": "逐次降噪递减",
    "enable_crop": "开启裁剪精修",
    "enable_ema_attention": "启用EMA注意力",
    "enable_multi_model": "启用多模型",
    "enable_profiling": "启用性能分析",
    "enable_sam": "启用SAM精细化",
    "enable_yolo": "启用YOLO",
    "expand": "遮罩扩展",
    "feather": "遮罩羽化",
    "final_denoise_strength": "全图降噪强度",
    "fix_rounds": "修复轮次",
    "frame_window": "参考帧窗口",
    "hand_foot_cfg_boost": "手脚CFG加成",
    "hand_foot_cn_strength": "手脚CN强度",
    "hand_foot_denoise": "手脚降噪下限",
    "hand_foot_denoise_decay": "手脚降噪衰减",
    "hand_foot_denoise_min": "手脚衰减下限",
    "hand_foot_enhance": "手脚增强模式",
    "hand_foot_extra_rounds": "手脚额外轮次",
    "hand_skeleton_cn_model": "手骨架CN模型",
    "inference_precision": "推理精度",
    "low_vram": "低显存模式",
    "lr_protect": "左右分组保护",
    "manual_path": "YOLO路径",
    "max_area_ratio": "过滤最大面积比",
    "max_aspect": "过滤最大长宽比",
    "max_targets": "最多目标数",
    "merge_boxes": "合并重叠框",
    "merge_iou": "合并IOU阈值",
    "merge_same_class": "合并仅同类",
    "min_area_ratio": "过滤最小面积比",
    "min_aspect": "过滤最小长宽比",
    "mpb_color_preserve": "MPB颜色保持",
    "multi_scale": "多尺度检测",
    "no_detect_mode": "未检测时处理",
    "per_class_limit": "每类最多目标数",
    "poisson_iters": "Poisson迭代次数",
    "priority_mode": "修复优先级",
    "refine_det": "二次检测精修",
    "refine_max": "二次检测最多目标",
    "refine_pad": "二次检测扩展",
    "region_cfg": "区域CFG",
    "region_denoise": "区域降噪",
    "region_steps": "区域步数",
    "sam_conf_threshold": "SAM置信度阈值",
    "sam_model_manual": "SAM模型手动路径",
    "sam_model_selection": "SAM模型选择",
    "seg_feather": "Seg边缘羽化",
    "skeleton_cn_end": "骨架CN结束",
    "skeleton_cn_strength": "骨架CN强度",
    "smart_strategy": "智能策略",
    "sort_mode": "排序方式",
    "target_classes_str": "目标类别",
    "temporal_strength": "时序一致性强度",
    "tile_enabled": "启用Tile检测",
    "tile_overlap": "Tile重叠",
    "tile_size": "Tile大小",
    "use_latent_aware": "使用Latent感知插值",
    "use_seg": "使用Seg分割",
    "use_wiou": "使用WIoU优化",
    "video_mode": "视频模式",
    "wiou_alpha": "WIoU权重参数",
    "work_mode": "工作模式",
    "yolo_imgsz": "YOLO推理尺寸",
    "yolo_model2": "YOLO模型2",
    "yolo_model3": "YOLO模型3",
    "yolo_model4": "YOLO模型4",
    "yolo_model_selection": "YOLO模型",
}


def _cfg_access(module_path):
    """AST 取一个模块里对 cfg 字段的访问，拆成 (reads, writes) 两个 set。

    只认两种基底：`cfg.<字段>`（prepare.py 的形参）与 `ctx.cfg.<字段>`（各阶段
    从上下文取）。两者之外的一律不算 —— 免得把 `foo.cfg` 之类误收进来。

    ★ 第二条基底必须**连基底对象一起认**（`... and base.value.id == "ctx"`）：
    只写 `base.attr == "cfg"` 会把 `self.cfg.x` / `payload.cfg.x` / `node.cfg.x`
    一并收进来 —— 那些与 cfg 读取无关的写法会造出指名 Fake 字段的**假红**，
    还会顶翻用例 7 的冻结计数，逼人为了无关改动去动绊线数字。
    """
    with io.open(module_path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    reads, writes = set(), set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        base = node.value
        is_bare_cfg = isinstance(base, ast.Name) and base.id == "cfg"
        is_ctx_cfg = (
            isinstance(base, ast.Attribute)
            and base.attr == "cfg"
            and isinstance(base.value, ast.Name)
            and base.value.id == "ctx"
        )
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

    T13 删单体时要改**哪几条** —— 按**方法名**列，不要按编号：
    本类的方法顺序与简报里的编号**并不一致**（`test_context_key_pairing_is_pinned`
    是复审后补的，插在中间），照位置数会数出相反的结论。

      - 依赖 `node.py` 的两条（实测：删掉 node.py 只红这两条）：
          `test_every_mapped_widget_exists_and_is_unique`
          `test_widgets_without_a_config_field_are_exactly_the_known_set`
        T13 删单体时预期要跟着改。
      - 依赖 `STAGE_MODULES` 计数的一条，与 node.py 无关：
          `test_cfg_access_counts_are_frozen`
        T4/T12/T13 改阶段读取逻辑时预期要跟着改。
      - 其余各条都不碰 node.py，拆完之后依然成立。

    刻意分成独立方法、不揉成一条，就是为了让 T13 能按条改。
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
        """7 个阶段读到的字段都必须真在 YoloConfig 里 —— 这是**拼写错**的网。

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

    def test_context_key_pairing_is_pinned(self):
        """★ 绊线：99 条「字段 ↔ 控件」配对逐条钉死（不止键集合与值集合）。

        用例 4 只断言**键集合**相等，用例 5 只断言**值集合** ⊆ 控件表且互不重复。
        **成对交换两处 `# 键:` 注释**时，这三个不变量全部保持，那三条断言全绿 ——
        而 `_context_key_map()` 正是 T8~T11 把字段翻成控件的唯一来源，交换后
        4 个节点会把旋钮挂到错误的字段上（改 A 旋钮、B 字段变），全量测试一条都不红。
        本条断言填的就是那个洞（复审 C-1）。

        ★ 它钉住的是**当前这份配对**，不是它的正确性：这 99 条注释是 hand-written 的，
        今天没有任何自动化手段能判断「结果缓存设备」是否**语义上**就是
        `cache_device_mode`。它证明的是「配对没有变」，不是「配对是对的」。

        实测另有一条路**走不通**（别去试）：context.py 按功能分组、node.py 把 CN 系
        放在 optional 靠后，所以「按声明顺序一一对应」这条结构判据不成立。
        """
        actual = _context_key_map()
        self.assertEqual(
            actual,
            CONTEXT_KEY_SNAPSHOT,
            "字段 ↔ 控件 的配对变了。逐类诊断：\n"
            f"  快照里有、代码里没有：{sorted(set(CONTEXT_KEY_SNAPSHOT) - set(actual))}\n"
            f"  代码里有、快照里没有：{sorted(set(actual) - set(CONTEXT_KEY_SNAPSHOT))}\n"
            f"  两侧都在但控件名不同（配对被换）："
            f"{sorted(k for k in set(actual) & set(CONTEXT_KEY_SNAPSHOT) if actual[k] != CONTEXT_KEY_SNAPSHOT[k])}\n"
            "★ 若是有意改动（改字段名/控件名、新增字段），在**同一个提交**里说明为什么\n"
            "  并同步 CONTEXT_KEY_SNAPSHOT；若是成对交换了两处 `# 键:` 注释 —— 那是事故，\n"
            "  拆出来的节点会把旋钮挂到错误的字段上。",
        )

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
        """★ 反空转的网：7 个计数写死，阶段模块被改名/搬走时「读到 0 个」当场红。

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


# ==================== 4 个节点共享的模型缓存单例（T5） ====================
#
# 这条护的是计划里的「静默失效 a」：拆成 4 个节点后若各持一份 YOLOModelCache，
# pipeline 里的代码（repair.py / sam_refine.py）会伸手进 `node.yolo_cache` 去卸模型
# 整理显存，而 load_model 发生在 detect.py —— 于是修复节点卸的是**自己那个空缓存**，
# 检测节点加载的模型再没人卸。**不报错、不崩，只是显存一直涨。**
#
# 为什么不 import 整个包：`model_cache.py` 模块级只 `import gc / os / torch /
# comfy.model_management`（ultralytics 等都在函数体内懒导入），且导入期**不调用**
# comfy 的任何东西，所以一个空壳桩就够 —— 与 test_import_smoke.py 同一套路，
# 且同样只回收**自己注入**的那两个名字，不动装载前就在 sys.modules 里的东西。
_STUB_MODULES = ("comfy", "comfy.model_management")
_STUB_SNAPSHOT = None
model_cache = None


def setUpModule():
    """注入最小的 comfy 桩，再按路径装载 model_cache（它没有包内相对导入）。"""
    global _STUB_SNAPSHOT, model_cache
    _STUB_SNAPSHOT = set(sys.modules)
    comfy_stub = sys.modules.get("comfy")
    if comfy_stub is None:
        comfy_stub = types.ModuleType("comfy")
        sys.modules["comfy"] = comfy_stub
    if "comfy.model_management" not in sys.modules:
        mm_stub = types.ModuleType("comfy.model_management")
        sys.modules["comfy.model_management"] = mm_stub
        comfy_stub.model_management = mm_stub
    model_cache = load_module(
        "qh_model_cache",
        os.path.join(pkgs_dir("qianhui_latent_yolo"), "model_cache.py"),
    )


def tearDownModule():
    """只回收本模块注入的名字，别动装载前就在 sys.modules 里的东西。"""
    if _STUB_SNAPSHOT is None:
        return
    for name in _STUB_MODULES:
        if name not in _STUB_SNAPSHOT:
            sys.modules.pop(name, None)


class TestSharedModelCache(unittest.TestCase):
    """T5：4 个拆分节点必须共用同一个 YOLOModelCache 实例。"""

    def test_get_shared_cache_returns_the_same_object(self):
        """`get_shared_cache()` 恒返回同一个对象 —— 这是「显存不再一直涨」的全部依据。

        若缓存被去掉、每次都 `YOLOModelCache(...)` 新建，`assertIs` 当场红
        （那条变异及对照组见 task-5 报告）。

        ★ 本条护的是**单例本身**；「node.py 是否真的改用了它」不在射程内 ——
        `node.py` 若被改回自己 new 一个，本条仍绿。那个方向由 T8~T11 的 4 个节点
        各自接线时暴露，见 task-5 报告的 concern。
        """
        first = model_cache.get_shared_cache()
        self.assertIs(
            first, model_cache.get_shared_cache(),
            "get_shared_cache() 每次返回新对象 —— 4 个节点又各持一份缓存了："
            "修复节点会去卸自己那个空缓存，检测节点加载的模型再没人卸（显存一直涨）",
        )
        self.assertIsInstance(first, model_cache.YOLOModelCache)


# ---------------- 节点「真的用了」那个单例（T5，静态） ----------------
#
# 上面那条护的是**单例本身**；这里护的是**节点有没有真的用它**。两条缺一不可：
# `node.py` 若被改回自己 `new` 一个 YOLOModelCache，单例依然存在、上面那条依然绿，
# 而「静默失效 a」已经复活 —— 修复节点卸的是自己那个空缓存，
# 检测节点加载的模型再没人卸（不报错、不崩，只是显存一直涨）。
#
# ★ 只做**静态** AST 解析、**不 import** 目标模块：`node.py` `import comfy.samplers`，
#   需要 ComfyUI 运行时（那一格由 test_import_smoke.py 的桩负责，各司其职）。

NODE_PY = os.path.join(pkgs_dir("qianhui_latent_yolo"), "node.py")

_SHARED_CACHE_SINGLETON = "get_shared_cache"
_CACHE_ATTR = "yolo_cache"


def _yolo_cache_assignments_in_init(tree, class_name, attr):
    """<class_name>.__init__ 里所有 `self.<attr> = ...` → [(行号, 右侧 unparse)]。

    找不到类或 `__init__` 时返回 None（与「找到了但没有赋值」的 [] 区分开）。
    """
    for node in ast.walk(tree):
        if not (isinstance(node, ast.ClassDef) and node.name == class_name):
            continue
        for sub in node.body:
            if not (isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and sub.name == "__init__"):
                continue
            out = []
            for st in ast.walk(sub):
                if isinstance(st, ast.Assign):
                    targets, value = st.targets, st.value
                elif isinstance(st, ast.AnnAssign):
                    targets, value = [st.target], st.value
                else:
                    continue
                if value is None:
                    continue
                for t in targets:
                    if (isinstance(t, ast.Attribute) and t.attr == attr
                            and isinstance(t.value, ast.Name) and t.value.id == "self"):
                        out.append((st.lineno, ast.unparse(value)))
            out.sort()
            return out
    return None


def _direct_cache_constructions(tree, cache_class):
    """整个模块里 `<cache_class>(...)` 的直接构造点行号（绕过单例的那种）。"""
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if isinstance(f, ast.Name) and f.id == cache_class:
            hits.append(node.lineno)
        elif isinstance(f, ast.Attribute) and f.attr == cache_class:
            hits.append(node.lineno)
    return sorted(hits)


def _assert_node_uses_shared_cache(case, path, class_name,
                                   attr=_CACHE_ATTR,
                                   singleton=_SHARED_CACHE_SINGLETON,
                                   cache_class="YOLOModelCache"):
    """断言 <path> 的 <class_name>.__init__ 通过模块级单例取 `attr`。

    ★ 这是给 **T8~T11 的 4 个节点**准备的：每个节点各调一次即可
      （换 path / class_name），别写死成只能测单体。

    三条不变量，每条都对应一个具体退化：
      1. 这类赋值**恰好一处** —— 多一处就可能有人把共享单例覆盖掉；
      2. 右侧 unparse 恰好是 `get_shared_cache()` —— 防被换回直接构造；
      3. 整个文件里**没有** `YOLOModelCache(...)` 的直接构造点 ——
         防绕过单例另起一份缓存（「静默失效 a」的原形）。

    `path` 由调用方给（测试读模块级 NODE_PY），好让变异对照组指向仓外副本。
    """
    with io.open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())

    assigns = _yolo_cache_assignments_in_init(tree, class_name, attr)
    case.assertIsNotNone(
        assigns,
        "%s 里找不到 %s.__init__（或它没有 self.%s = ... 赋值）"
        % (path, class_name, attr),
    )
    case.assertEqual(
        len(assigns), 1,
        "%s.__init__ 里 self.%s 有 %d 处赋值（应恰好 1 处）：%s\n"
        "  多一处就可能把共享单例覆盖掉 —— 那和「各持一份缓存」是同一个病。"
        % (class_name, attr, len(assigns), assigns),
    )
    case.assertEqual(
        assigns[0][1], singleton + "()",
        "%s.__init__ 的 self.%s 赋的不是 %s()（第 %d 行是 %r）。\n"
        "  改回 `%s(...)` 会让 4 个节点各持一份缓存 —— 修复节点卸的是自己那个"
        "**空**缓存，检测节点加载的模型再没人卸：不报错，只是显存一直涨。"
        % (path, attr, singleton, assigns[0][0], assigns[0][1], cache_class),
    )
    direct = _direct_cache_constructions(tree, cache_class)
    case.assertEqual(
        direct, [],
        "%s 里仍有直接构造 %s(...) 的行 %s —— 那会绕过共享单例。"
        % (path, cache_class, direct),
    )


class TestNodesUseSharedCache(unittest.TestCase):
    """T5：节点 `__init__` 必须走模块级单例，而不是各 new 一个缓存。

    「单例存在」与「节点用了它」是两件事：`TestSharedModelCache` 护前者（运行时），
    本类护后者（静态）。只留前者的话，node.py 被改回自己 new 一个也不会有测试红。
    """

    def test_qianhui_latent_yolo_init_uses_the_shared_cache(self):
        """★ 绊线：node.py 改回自己 `new` 缓存 / 多赋一处 / 绕过单例，本条都当场红。

        两个变异对照组（改回直接构造、在 `__init__` 末尾再赋一处）见 task-5 报告。
        """
        _assert_node_uses_shared_cache(self, NODE_PY, "QianhuiLatentYOLO")


# ==================== 4 个精简 Context 的字段归属（T6） ====================
#
# 拆成 4 个节点后每个节点只该看见自己那部分字段。这既是隔离，也是一张网：
# 字段的归属由「哪个阶段真的读它」机检得来，不是手抄的清单 —— 手抄会漏，
# 而漏掉的字段在运行期**只在该分支被走到时**才炸（`slots=True` 让写未声明属性
# 当场 AttributeError，但读一个不存在的槽同样只在那一刻才炸）。
#
# ★ 三条不变量（各自有能杀它的变异，见 task-6 报告 §6）：
#   1. 每个 Context 的 slots ⊇ 它那一组来源的读取集（少了 → 运行期炸）
#   2. ★★ 四个 Context 的并集 == 所有阶段读到的 `ctx.` 字段并集（应为 50）
#      —— 这是最重要的一条：任一字段在 4 个 Context 里**都没家** ⇒ 当场红。
#   3. 反向：Context 里没有谁读的 slot = 死字段。

# 每个精简 Context 的**来源**：它归哪个节点的哪些阶段函数读的 `ctx.` 字段。
# `func=None` 表示整个模块（那几个模块只有一个入口函数）。
# 归属依据：`finalize_interrupt`（中断回填）与 `build_preview` 归节点 A，
# `build_masks` 归 C、`build_info` 归 D；`report.py:build_report` 那个编排壳是
# 过渡态（T13 拆掉），它的读取是这四组的子集，**不单独喂给任何 Context**。
CONTEXT_SOURCES = {
    "DetectCtx": (
        ("detect.py", None),
        ("report.py", "finalize_interrupt"),
        ("report_bits.py", "build_preview"),
    ),
    "SamCtx": (("sam_refine.py", None),),
    "RepairCtx": (
        ("repair.py", None),
        ("report_bits.py", "build_masks"),
    ),
    "GlobalCtx": (
        ("global_pass.py", None),
        ("report_bits.py", "build_info"),
    ),
}

# T6 机检实测的字段数与并集：41 / 3 / 28 / 27，并集 50 = YoloRunContext 的字段数。
#
# ★ 这是**反空转**的绊线：某个来源模块被改名/搬走时 `_ctx_reads` 会算出空集，
#   下面的包含式断言（差集 ⊆ 空集）会**空转通过**，只有这里会让「读到 0 个」当场红。
#   预期会触发它的改动：T8~T11 各节点接线（读法变了）、T13 删 prepare_run/单体。
#   改动时先确认是真改了归属，并在**同一个提交**里说明为什么。
EXPECTED_CTX_FIELD_COUNTS = {
    "DetectCtx": 41,
    "SamCtx": 3,
    "RepairCtx": 28,
    "GlobalCtx": 27,
}
EXPECTED_CTX_UNION = 50

CONTEXT_PY = os.path.join(pkgs_dir("qianhui_latent_yolo"), "mixins", "context.py")


def _ctx_reads(module_path, func=None):
    """AST 取 `ctx.<字段>` 的 **Load** 集合；func=None 表示整个模块。

    与 `_cfg_access` 的区别：那个取的是 **cfg** 字段（`cfg.x` / `ctx.cfg.x`），
    这个取的是 **ctx** 字段（`ctx.x`）。两者互不覆盖，是两套网。

    ★ 同样**不 import**（这些模块需要 comfy 运行时），只 `ast.parse`。
      也**必须区分 Load 与 Store**：`prepare.py` 那一堆是「写」（填 ctx 字段），
      各阶段模块的才是「读」。不区分的话两个方向会算成一个。

    ★ `ctx.cfg` **算**一次 `cfg` 槽的读取：`ctx.cfg.<旋钮>` 的 AST 里，
      内层 `ctx.cfg` 就是一个独立的 Load 节点，而 `cfg` 在 `YoloRunContext` 里
      也是一个真槽（4 个阶段都要它）。把它滤掉会让并集少 1、`cfg` 看起来
      「4 个 Context 里都没有家」—— 那是**假红**。

    ★ 只认基底是**裸名 `ctx`** 的属性；`self.ctx.x` / `payload.ctx.x` 一律不算 ——
      与 `_cfg_access` 对 `self.cfg` 的处理同一理由（免得造出指名 Fake 字段的假红）。
    """
    with io.open(module_path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    if func is None:
        roots = [tree]
    else:
        roots = [
            n for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name == func
        ]
        assert roots, "%s 里找不到函数 %s（改名/搬走了？）" % (module_path, func)
    out = set()
    for root in roots:
        for node in ast.walk(root):
            if not isinstance(node, ast.Attribute):
                continue
            if not isinstance(node.ctx, ast.Load):
                continue
            if isinstance(node.value, ast.Name) and node.value.id == "ctx":
                out.add(node.attr)
    return out


def _context_classdefs():
    """context.py 里的 {类名: ClassDef}（ast，不 import）。"""
    with io.open(CONTEXT_PY, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    return {n.name: n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}


def _slim_context_slots():
    """{类名: set(字段名)}，从 context.py 的 ClassDef 取 AnnAssign 目标。"""
    defs = _context_classdefs()
    out = {}
    for name in CONTEXT_SOURCES:
        cls = defs.get(name)
        assert cls is not None, "context.py 里找不到类 %s（T6 的 4 个精简 Context）" % name
        out[name] = {
            st.target.id
            for st in cls.body
            if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name)
        }
    return out


def _context_source_reads(class_name):
    """某个 Context 那一组来源读到的 `ctx.` 字段并集。"""
    reads = set()
    for module, func in CONTEXT_SOURCES[class_name]:
        reads |= _ctx_reads(os.path.join(PIPELINE_DIR, module), func)
    return reads


def _all_stage_ctx_reads():
    """7 个阶段模块读到的 `ctx.` 字段并集（并集 2 的右端）。"""
    reads = set()
    for name in STAGE_MODULES:
        reads |= _ctx_reads(os.path.join(PIPELINE_DIR, name))
    return reads


def _is_slots_dataclass(cls_node):
    """ClassDef 的装饰器里有没有 `@dataclass(... slots=True ...)`。"""
    for dec in cls_node.decorator_list:
        if not isinstance(dec, ast.Call):
            continue
        f = dec.func
        name = f.id if isinstance(f, ast.Name) else getattr(f, "attr", None)
        if name != "dataclass":
            continue
        for kw in dec.keywords:
            if kw.arg == "slots":
                return isinstance(kw.value, ast.Constant) and kw.value.value is True
    return False


class TestSlimContexts(unittest.TestCase):
    """T6：4 个精简 Context 的字段归属（三条不变量 + 反空转绊线 + slots 机制）。

    拆成 4 个节点后共用的那个 50 字段 `YoloRunContext` 会被 T13 撤掉，
    这 4 个类接管。字段归属错了的症状是：界面上旋钮还在、改了没用、不报错，
    只在那条分支被走到时 AttributeError —— 所以先在这里闭合。
    """

    def test_each_context_covers_its_sources(self):
        """不变量 1：每个 Context 的 slots ⊇ 它那一组来源读到的 `ctx.` 字段。

        少了就是一个运行期的 AttributeError（`slots=True` 会当场炸，
        但只在该分支被走到时才炸）—— 静态这里当场红。
        """
        slots = _slim_context_slots()
        for name in sorted(CONTEXT_SOURCES):
            missing = _context_source_reads(name) - slots[name]
            self.assertEqual(
                missing, set(),
                "%s 少了这些槽：%s\n"
                "  它们在 %s 里被读到，却没在 %s 里声明 —— "
                "运行期会在那条分支上 AttributeError（界面上看不出问题）。"
                % (name, sorted(missing), CONTEXT_SOURCES[name], name),
            )

    def test_every_slot_is_read_by_its_sources(self):
        """不变量 3（反向）：Context 里没有谁读的 slot = **死字段**，也应报。

        与 T2 用例 3（`test_every_config_field_is_read_by_some_stage`）同一思路：
        声明了却没人读的槽，只会让下一个人以为「这个字段有用」。
        """
        slots = _slim_context_slots()
        for name in sorted(CONTEXT_SOURCES):
            dead = slots[name] - _context_source_reads(name)
            self.assertEqual(
                dead, set(),
                "%s 里这些槽没有任何来源读过（死字段）：%s\n"
                "  它的来源是 %s。要么归属写错了，要么该字段根本不该在这。"
                % (name, sorted(dead), CONTEXT_SOURCES[name]),
            )

    def test_context_union_covers_every_ctx_field_read_by_any_stage(self):
        """★★ 不变量 2：4 个 Context 的并集 == 所有阶段读到的 `ctx.` 字段并集（= 50）。

        这是**最重要**的一条：任一字段在 4 个 Context 里**都没家** ⇒ 当场红。
        这正是「静默失效」的形状 —— 那个节点会在运行期 AttributeError，
        但只在该分支被走到时才炸；界面上旋钮还在、改了没用。

        反向（并集里有、阶段没读）与不变量 3 重叠，这里一并断言，
        消息里分开列出两个方向，好一眼看出是「漏装」还是「多装」。
        """
        slots = _slim_context_slots()
        union = set()
        for members in slots.values():
            union |= members
        stage_reads = _all_stage_ctx_reads()

        self.assertEqual(
            stage_reads - union, set(),
            "这些 ctx 字段被阶段读到，却在 4 个 Context 里**都没有家**：%s\n"
            "  ⇒ 拆出来的节点会在运行期 AttributeError（界面上旋钮还在，"
            "改了没用、不报错）。" % sorted(stage_reads - union),
        )
        self.assertEqual(
            union - stage_reads, set(),
            "这些槽在某个 Context 里，却没有任何阶段读它：%s"
            % sorted(union - stage_reads),
        )

    def test_context_field_counts_are_frozen(self):
        """★ 反空转的网：4 个字段数与并集写死。

        上面的包含式断言是「差集为空」：某个来源模块一旦解析不到（改名、搬走），
        `_ctx_reads` 算出空集，它们会**空转通过**。这条把计数钉死，
        让「读到 0 个」当场红。

        ★ 改动这些数字**必须在同一个提交里说明为什么**
        （预期触发者：T8~T11 各节点接线、T13 删 prepare_run/单体）。
        """
        slots = _slim_context_slots()
        actual = {name: len(members) for name, members in slots.items()}
        self.assertEqual(
            actual, EXPECTED_CTX_FIELD_COUNTS,
            "4 个精简 Context 的字段数变了（应为 41/3/28/27）。变动时：\n"
            "  1) 先确认是真改了归属，而不是模块被改名/搬走、或解析失手；\n"
            "  2) 在**同一个提交**里说明为什么，并同步 EXPECTED_CTX_FIELD_COUNTS。",
        )
        union = set()
        for members in slots.values():
            union |= members
        self.assertEqual(
            len(union), EXPECTED_CTX_UNION,
            "4 个 Context 的并集应为 %d（= YoloRunContext 的字段数）。"
            % EXPECTED_CTX_UNION,
        )

    def test_slim_contexts_are_slots_dataclasses(self):
        """4 个类都带 `@dataclass(slots=True)` —— 这是「防笔误」的全部依据。

        去掉 `slots=True` 不报错、不改结果，只是把「写错字段名」从
        AttributeError 降级成**静默新建属性**：真字段一动不动，而拆出来的
        4 个节点正是靠这层保护接线的。所以这条也得有人看着。
        """
        defs = _context_classdefs()
        for name in sorted(CONTEXT_SOURCES):
            cls = defs.get(name)
            self.assertIsNotNone(cls, "context.py 里找不到类 %s" % name)
            self.assertTrue(
                _is_slots_dataclass(cls),
                "%s 的装饰器不是 @dataclass(slots=True)（第 %d 行）。\n"
                "  没有 slots，写错字段名会静默新建属性、真字段不动 —— "
                "这正是本计划已经踩过一次的 Critical 形状。"
                % (name, cls.lineno),
            )


if __name__ == "__main__":
    unittest.main()
