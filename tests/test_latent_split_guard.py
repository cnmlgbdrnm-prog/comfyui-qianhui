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
import re
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

#: 4 个拆出来的节点文件。T13 删掉 prepare_run 之后，**一批 cfg 读取与那唯一一处
#: cfg 写（`ctx.cfg.enable_yolo = False`）搬到了它们身上** —— 不把它们算进来，
#: 不变量 2 会**空转**（并集看着闭合、其实少了 4 个文件的读取），而空转比变红更坏。
NODE_MODULES = ("node_detect.py", "node_sam.py", "node_repair.py", "node_global.py")


def _mod_path(name):
    """模块文件名 → 磁盘路径。

    pipeline 阶段模块住在 `mixins/pipeline/`，4 个 `node_*.py` 住在包根 ——
    两处来源放同一个网里，所以路径解析收成一个函数，别在调用点各写一遍。
    """
    if name in NODE_MODULES:
        return os.path.join(pkgs_dir("qianhui_latent_yolo"), name)
    return os.path.join(PIPELINE_DIR, name)


STAGE_MODULES = ("prepare.py", "detect.py", "repair.py",
                 "global_pass.py", "report_bits.py",
                 "sam_refine.py") + NODE_MODULES

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
    # ★ T13：`prepare.py` 的 reads 48 → **0**。那 48 次 cfg 读取**全部**发生在
    #   `prepare_run` 体内（`parse_config` 只写不读，99 次），`prepare_run` 已随单体退役。
    #   **这不是覆盖率的损失**：那 48 处读取已被 4 个节点的 `_prepare` / `process` 接手
    #   （见下面 node_*.py 那 4 条），所以「并集覆盖全部 99 个字段」仍是闭合的。
    #   本条从 48 变 0 正是反 no-op 绊线在**通知**「读的人换地方了」，不是放宽。
    "prepare.py": (0, 99),  # (reads, writes)
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
    # report_bits.py。整条管道的读取一个没少、只是换了文件。
    "report_bits.py": (6, 0),
    # ==================== T13：4 个节点各自的那份读取 ====================
    #
    # 删掉 prepare_run 之后，它原来读的那批 cfg 分散到了 4 个 node_*.py ——
    # 这 4 条**必须**登记：不登记的话「并集覆盖全部字段」那两条断言会因为
    # 少了这 4 个文件的读取而**空转通过**（看着是绿的，其实网漏了）。
    # 数字是机检的**去重字段数**（不是 AST 访问点个数），口径与上面一致。
    "node_detect.py": (27, 2),   # 写：class_rules_text（规则预设展开）、enable_yolo（路径无效时禁用）
    "node_sam.py": (4, 0),
    "node_repair.py": (18, 1),   # 写：class_rules_text
    "node_global.py": (17, 4),   # 写：blend_mode / enable_crop / enable_yolo / use_latent_aware（stage_flags 回填）
}
# ★ 上一版里的 `report.py` 两条（`(1, 0)`）在 T13 被删掉了：
#   它的 `finalize_interrupt` 整段搬进了 node_detect.py 的 `_finalize_interrupt`，
#   那唯一一处 `use_latent_aware` 读取随之过去（机检确认 node_detect.py 里有它），
#   所以**整条管道的读取集合一个字段都没少**。删条目不是放宽 —— 是那个文件没了。

# 有控件、但没有对应 YoloConfig 字段的 15 个名字（14 个必填 socket/参数 + 配置预设）。
# ★ 这是**故意的绊线**：以后新增控件时，要么给它一个 YoloConfig 字段 +
#   `# 键:` 注释，要么**显式**把这个名字加进本常量 —— 两个都不做，用例 6 就红。
# 注意 步数 / CFG / 降噪 出现在这里是**对的**：它们被 repair.py 当函数参数用
# （未检测时回退），不是 YoloConfig 字段，所以没有 `# 键:` 注释。
#
# ★ T13 从 14 → 15：多出来的是 `检测数据`（节点间的自定义 socket，类型
#   `QIANHUI_DETECT`）。它**本来就不该有 YoloConfig 字段** —— 它不是旋钮，
#   是 4 个节点之间传的逐帧记录；旧单体里它不存在（单体一个节点干完全部活），
#   所以这一条是拆分本身带来的、按 (b) 显式登记。
#   判据没变：这张表 = 「4 个节点控件表里、没有 cfg 字段的那些」的精确集合。
WIDGETS_WITHOUT_CONFIG_FIELD = frozenset((
    "Latent", "VAE", "模型", "正面提示", "负面提示", "seed", "步数", "CFG",
    "降噪", "采样器", "调度器", "放大倍数", "放大算法", "配置预设", "检测数据",
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
    return {name: _cfg_access(_mod_path(name)) for name in STAGE_MODULES}


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
        # ★ T13 起「写方」不止 parse_config：那唯一一处 cfg 写
        #   （YOLO 路径无效时 `ctx.cfg.enable_yolo = False`）在 prepare_run 里，
        #   而 prepare_run 已删、该行随之外移进 node_detect.py。
        #   只盯 prepare.py 会变成「98 vs 99」的假红；放宽成并集后这条断言守的
        #   东西没变：每个字段都有人写，且写了的都得是真字段。
        writes = set()
        for _name in ("prepare.py",) + NODE_MODULES:
            _, _w = _cfg_access(_mod_path(_name))
            writes |= _w
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
        """★ 故意的绊线：没有 cfg 字段的控件**精确等于**那 15 个已知名字。

        以后新增控件时二选一：要么给它一个 YoloConfig 字段 + `# 键:` 注释，
        要么**显式**把这名字加进 WIDGETS_WITHOUT_CONFIG_FIELD 常量；
        两个都不做，这条就红。

        ★ T13：参照物从单体 `node.py` 换成 **4 个节点的并集**（14 → 15 个名字，
          多的是 `检测数据` 那个节点间 socket —— 理由写在常量的注释里）。
          这条的**语义**没变：它仍然是「有哪些控件不该有 cfg 字段」的唯一清单，
          仍然是「新控件必须二选一」的那道闸门。
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
# ★ 只做**静态** AST 解析、**不 import** 目标模块：4 个节点都 `import comfy.samplers`，
#   需要 ComfyUI 运行时（那一格由 test_import_smoke.py 的桩负责，各司其职）。

#: T13：原 `NODE_PY`（指向已删的 `node.py`）随单体退役去掉。
#: 现在路径由调用方逐节点给出 —— `_assert_node_uses_shared_cache(case, path, class_name)`
#: 的第 2 个参数就是为此留的（好让变异对照组指向仓外副本）。
def split_node_path(mod_name):
    """`node_<stage>.py` 的绝对路径（按需构造，不缓存 —— 变异对照组会换根）。"""
    return os.path.join(pkgs_dir("qianhui_latent_yolo"), mod_name + ".py")


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

    `path` 由调用方给（测试传 `split_node_path(mod_name)`），好让变异对照组指向仓外副本。
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
    本类护后者（静态）。只留前者的话，某个节点被改回自己 new 一个也不会有测试红。

    ★ T13：单体退役后本类改盯**4 个节点**（各一条子用例）。原先那条只测单体
    （`NODE_PY` + `QianhuiLatentYOLO`），随 `node.py` 一起删 —— 但它护的不变量
    对 4 个节点同样成立，所以是**改指**不是**取消**。`_assert_node_uses_shared_cache`
    本来就是为这 4 个节点准备的（见其 docstring）。
    """

    #: (模块名, 类名) —— 4 个节点各自一份 `__init__`。
    NODES = (("node_detect", "QianhuiYoloDetect"),
             ("node_sam", "QianhuiSamRefine"),
             ("node_repair", "QianhuiRegionRepair"),
             ("node_global", "QianhuiGlobalPass"))

    def test_every_split_node_init_uses_the_shared_cache(self):
        """★ 绊线：任一节点的 `__init__` 改回自己 `new` 缓存 / 多赋一处 / 绕过单例，当场红。

        两个变异对照组（改回直接构造、在 `__init__` 末尾再赋一处）见 task-5 报告。
        """
        for mod_name, cls_name in self.NODES:
            with self.subTest(node=cls_name):
                _assert_node_uses_shared_cache(self, split_node_path(mod_name), cls_name)


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
        # T13：report.py 没了，`finalize_interrupt` 整段搬进 node_detect.py 的
        # `_finalize_interrupt`（逐字搬），故来源换成它 —— 实测并集仍是 50、差集 []。
        ("node_detect.py", "_finalize_interrupt"),
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
# ★ 它钉的是**「两边一起长」**：给某个 Context 加一个字段、**同时**在它的来源模块里读它，
#   用例 1/2/3 会**全绿**（集合双向闭合，没破），只有这里红
#   —— 因为那个字段 `YoloRunContext` 从来没有过，并集也不再是 50。
#   （订正：早先这里写「模块被改名 ⇒ `_ctx_reads` 算空集 ⇒ 包含式断言空转通过」，
#     那个理由是**错的**：真·改名会让 `io.open` 抛 FileNotFoundError；
#     把基底名换掉只红用例 2/3，用例 4 不红。详见报告 §修复轮 1。）
#   预期会触发它的改动：T8~T11 各节点接线（读法变了）、T13 删 prepare_run/单体。
#   改动时先确认是真改了归属，并在**同一个提交**里说明为什么。
EXPECTED_CTX_FIELD_COUNTS = {
    "DetectCtx": 41,
    "SamCtx": 3,
    "RepairCtx": 28,
    # ★ T13：27 → 26。删掉 `global_pass.py` 那段死代码（`if ctx.preview_image is None:
    #   ... _decode_latent_to_image(ctx.VAE, ...)`）之后，`VAE` 在 GlobalCtx 里
    #   **再没有读者** —— 而全图降噪节点本来就不声明 VAE 输入，这个槽本就不该在。
    #   是 `test_every_slot_is_read_by_its_sources`（每个槽都要有来源读）把它揪出来的。
    #   四个类的**并集仍是 50**：`VAE` 由 DetectCtx 持有并读（检测节点解码预览图）。
    "GlobalCtx": 26,
}
EXPECTED_CTX_UNION = 50

# ★★ T13 交接（T6 在下面那条用例的 docstring 里就预告过）：
#   `YoloRunContext` 是「4 个精简 Context 切自哪块原件」的参照物。单体退役时它被删，
#   参照物就没了 —— 所以这里把它**冻成一张表**（从活的类机检出来的
#   {字段名: (注解 dump, 默认值 dump)}，50 条）。
#   这就是那条用例的 docstring 说的「同 CONTEXT_KEY_SNAPSHOT 的做法」，
#   **不是**把这道网丢掉：它护的是「切出来的每一片是不是和原件逐字一样」。
#   ★ 只对**同名字段**生效（与原参照物同口径）：某个 Context 独有的字段不在此表约束内。
#   ★ 它证明的是「与原件的这份拷贝一致」，不是「默认值在语义上正确」—— 与原用例同。
# YoloRunContext 的冻结快照（50 个字段），按字段名排序。
YOLO_RUN_CONTEXT_FIELD_SNAPSHOT = {
    'CFG': ('float', '8.0'),
    'VAE': ('Any', 'None'),
    '_decoded_np': ('Any', 'None'),
    'all_dets': ('List[Any]', 'field(default_factory=list)'),
    'base_latent': ('Any', 'None'),
    'base_neg': ('Any', 'None'),
    'base_pos': ('Any', 'None'),
    'batch_size': ('int', '0'),
    'cfg': ('YoloConfig', None),
    'class_rules': ('Any', 'None'),
    'detection_mask': ('Any', 'None'),
    'detections': ('List[Any]', 'field(default_factory=list)'),
    'frame_history': ('Any', 'None'),
    'has_cn': ('bool', 'False'),
    'has_hand_skeleton_cn': ('bool', 'False'),
    'i': ('int', '0'),
    'interrupted': ('bool', 'False'),
    'ksampler': ('Any', 'None'),
    'latent_for_detection': ('Any', 'None'),
    'latent_h': ('int', '0'),
    'latent_w': ('int', '0'),
    'neg_full': ('Any', 'None'),
    'original_samples': ('Any', 'None'),
    'out_device': ('Any', 'None'),
    'out_latents': ('Any', 'None'),
    'out_masks': ('List[torch.Tensor]', 'field(default_factory=list)'),
    'out_previews': ('List[torch.Tensor]', 'field(default_factory=list)'),
    'pbar': ('Any', 'None'),
    'perf_start_time': ('Any', 'None'),
    'pixel_h': ('int', '0'),
    'pixel_w': ('int', '0'),
    'pos_full': ('Any', 'None'),
    'preview_image': ('Any', 'None'),
    'processed_count': ('int', '0'),
    'processed_latent': ('Any', 'None'),
    'sam_model_path': ('Optional[str]', 'None'),
    'samples': ('Any', 'None'),
    'seed': ('int', '0'),
    'target_classes': ('Any', 'None'),
    'yolo_confidences_to_use': ('List[float]', 'field(default_factory=list)'),
    'yolo_models_to_use': ('List[str]', 'field(default_factory=list)'),
    '放大倍数': ('float', '1.0'),
    '放大算法': ('str', "'latent_aware'"),
    '模型': ('Any', 'None'),
    '正面提示': ('Any', 'None'),
    '步数': ('int', '20'),
    '调度器': ('str', "'normal'"),
    '负面提示': ('Any', 'None'),
    '采样器': ('str', "'euler'"),
    '降噪': ('float', '1.0'),
}

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


def _field_specs(cls_node):
    """ClassDef → `{字段名: (注解 unparse, 默认值 unparse 或 None)}`。

    `None` 表示该字段没有默认值（如 `cfg: YoloConfig`）。
    用 `ast.unparse` 而不是取源码切片：它对缩进/换行/行尾注释免疫，
    只留下**语义**等价的那部分（`int` vs `float`、`20` vs `21`、
    `field(default_factory=list)` vs `field(default_factory=set)` 都能分辨）。
    """
    return {
        st.target.id: (
            ast.unparse(st.annotation),
            None if st.value is None else ast.unparse(st.value),
        )
        for st in cls_node.body
        if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name)
    }


def _context_source_reads(class_name):
    """某个 Context 那一组来源读到的 `ctx.` 字段并集。"""
    reads = set()
    for module, func in CONTEXT_SOURCES[class_name]:
        reads |= _ctx_reads(_mod_path(module), func)
    return reads


def _all_stage_ctx_reads():
    """7 个阶段模块读到的 `ctx.` 字段并集（并集 2 的右端）。"""
    reads = set()
    for name in STAGE_MODULES:
        reads |= _ctx_reads(_mod_path(name))
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

    def test_context_fields_are_same_source_as_yolo_run_context(self):
        """★ 同名字段的**注解与默认值**必须与 `YoloRunContext` 逐字一致。

        字段名对上还不够：`步数: int = 20` 被改成 `= 21`、`interrupted` 从 `False`
        改成 `True`、`default_factory=list` 换成 `set` —— 三种退化都**编译得过、
        跑得动、界面上看不出来**，只是运行期行为变了。简报步骤 3 把「默认值也要照抄」
        点名为最要紧的一处，而用例 1/2/3 只比**字段名集合**，对默认值一言不发。
        这条填的就是那个洞。

        注解**顺手一起比**（同一趟遍历，零成本）：`int` 写成 `float` 同样只在这条网里现形。

        ★ 参照物取的是 `YoloRunContext` 的 `{名: (注解 dump, 默认值 dump)}` —— 也就是
          **拆分的源头**（这 4 个类本来就是从它切出来的）。于是这条网问的是
          「切出来的每一片，是不是和原件一模一样」，而不是「值看起来对不对」。

        ★ **刻意不比「声明顺序」**：顺序只在「按位置传参构造」时才有行为差异，
          而 4 个节点一律**关键字构造**（`DetectCtx(cfg=..., 模型=...)`）；
          真正危险的错序 —— 无默认值字段排在有默认值字段之后 —— Python 在
          **类创建时**就抛 `TypeError`，是当场崩，不是静默失效。故不设此网。

        ★ T13 交接：`YoloRunContext` 会被删，本用例的参照物也就没了。届时把参照物
          改为本文件内**冻结的** `{名: (注解 dump, 默认值 dump)}` 快照
          （同 `CONTEXT_KEY_SNAPSHOT` 的做法），**别把这道网静静丢掉**。
        """
        defs = _context_classdefs()
        base = dict(YOLO_RUN_CONTEXT_FIELD_SNAPSHOT)

        # ★ 原件还在时**额外**核一次「快照 == 活的类」：这样这张表是从原件机检来的、
        #   不是手抄的（手抄的表现在也能通过，但那是巧合）。
        #   单体现在已删 ⇒ 这一段自动失效（不再是「跳过」，是那段代码不存在了）。
        base_node = defs.get("YoloRunContext")
        if base_node is not None:
            live = _field_specs(base_node)
            self.assertEqual(
                live, base,
                "冻结快照与**活的** YoloRunContext 不一致 —— 快照被人手改过，"
                "或 context.py 的字段注解/默认值在这之后动过。\n"
                "  快照是「切自哪块原件」的唯一证据，请重新机检生成，不要手改。",
            )

        for name in sorted(CONTEXT_SOURCES):
            cls = defs.get(name)
            self.assertIsNotNone(cls, "context.py 里找不到类 %s" % name)
            specs = _field_specs(cls)
            bad = [
                "    %s.%s: 快照是 %r，这里是 %r" % (name, f, base[f], spec)
                for f, spec in sorted(specs.items())
                if f in base and base[f] != spec
            ]
            self.assertEqual(
                bad, [],
                "%s 里这些字段的**注解或默认值**与 YoloRunContext 的冻结快照不一致：\n"
                "%s\n"
                "  ⇒ 默认值不同会**改运行期行为**：编译得过、跑得动、界面上看不出来。\n"
                "  字段名逐字一致还不够 —— 它必须与原件（冻结快照）是**同源的一份拷贝**。"
                % (name, "\n".join(bad)),
            )

    def test_context_field_counts_are_frozen(self):
        """★ 钉「两边一起长」：4 个字段数与并集写死。

        它护的是**集合双向闭合之外**的那一类退化：给某个 Context 加一个字段、
        **同时**在它的来源模块里读它 —— 用例 1（slots ⊇ 读集）、2（并集 == 读集）、
        3（slots ⊆ 读集）会**全绿**，因为两边一起长了、闭合没破。
        可那字段是 `YoloRunContext` 从来没有的，并集也不再是 50 —— 只有这条会红。

        （订正：本用例的说明早先写成「来源模块被改名 ⇒ `_ctx_reads` 算出空集 ⇒
          包含式断言空转通过」。那个理由是**错的**：真·改名会让 `io.open` 抛
          `FileNotFoundError`；把基底名换掉只红用例 2/3，本用例不红。
          详见报告 §修复轮 1 的 P2 / P5。）

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



# ==================== 4 个拆分节点的控件表（T8~T14） ====================
#
# 拆完之后「某个旋钮在界面上消失」是**不会让任何测试变红**的那一类故障：
# 控件少了、用户改不了、后端读的是声明默认值、什么都不报。所以拆出来的 4 个节点
# 各要一张与它的阶段模块**双向对账**过的控件表。
#
# ★ 唯一判据：`widget-ownership-table.md` 的「读到它的阶段模块」列 —— 节点 X 的
#   控件 = 字段被 X 的阶段模块读到的那些控件。**不是**「模块的读取条数」。
#
# ★ 15 个「只被 prepare.py 读」的字段没有阶段读者，它们的归属是**人指定**的
#   （prepare-split-design.md §1）：它们被 prepare 消费成 `ctx.*` 之后再被阶段读。
#   `PREPARE_ONLY_OWNER` 就是那份指定的机读化。
#
# ★ `STAGE_FLAG_FIELDS`：节点 D 的「检测信息」要复述 4 个值，而它在自己那份 cfg 里
#   读不到它们 —— 计划明文要求**从 payload.stage_flags 快照取，不复制控件**。
#   所以这 4 个字段对 D 不产生控件要求。

SPLIT_NODE_DIR = pkgs_dir("qianhui_latent_yolo")

#: ★ 冻结的「4 个节点控件声明」快照（114 条：控件名 → `ast.unparse(声明)`）。
#
# 为什么需要它（终审 C-附3 ①②）：
#   T13 把 `_latent_node_widget_decls()` 的数据源换成了「4 个节点 INPUT_TYPES 的并集」，
#   而 `test_the_four_nodes_together_cover_every_widget_of_the_old_node` 原本拿它当
#   「旧单体」的参照物 ⇒ 等号两边成了同一份东西，那条守卫退化成**同义反复**
#   （终审实测：删 `node_global` 的 `工作模式`、删 `node_detect` 的 `二次检测精修`
#    都不会让它红）。这里把参照物换成本文件内**冻结**的 114 条 —— 守卫重新有了独立性。
#
# 同时补上 C-附3② 那个洞：**单节点独有控件的绝对值**此前无人钉
#   （实测：把 `node_sam` 独有的 `SAM置信度阈值` 默认值 0.8 → 0.9，套件全绿 ——
#    `test_shared_widget_declarations_are_identical_across_nodes` 只查「同名的相等」，
#    看不见只出现在一个节点里的控件被动过）。
#   本快照存的是**整条声明**：类型、default、min/max/step、tooltip、options 顺序。
#
# ★ 这是**冻结契约**，不是「当前值的转抄」：改它必须与「为什么改」同一次提交，
#   并且先问一句「用户界面真的该变吗」。生成方式与 `CONTEXT_KEY_SNAPSHOT` 一致 ——
#   从 `_node_widget_decls()` 逐条机检 dump，**不要手抄**。
#
# ★ 为什么是 **114** 条而不是源单体的 113 条：源侧那 113 条 = 本快照 **减去**
#   `检测数据`（4 个节点之间的自定义 socket，类型 `QIANHUI_DETECT`，源单体里
#   当然没有它）。快照要跟**并集**做**双向**比对（少了红、多了也红），
#   所以参照物必须是并集的完整契约 114 条；拿 113 条做双向会永远多出 `检测数据`。
#   「源的 113 条逐字都在并集里、只多 `检测数据`」这另一半由对拍脚本 §2b 独立把守
#   （它比的是仓外只读源文件 `水印/nodes.py`，与这里的冻结快照互为独立复算）。
SPLIT_WIDGET_DECL_SNAPSHOT = {
    'CFG': "('FLOAT', {'default': 8.0, 'min': 0.0, 'max': 100.0, 'step': 0.5})",
    'CN图像': "('IMAGE',)",
    'CN开始': "('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.05})",
    'CN强度': "('FLOAT', {'default': 0.7, 'min': 0.0, 'max': 2.0, 'step': 0.05, 'tooltip': 'ControlNet强度，建议0.6-0.8（过高会丢失纹理细节）'})",
    'CN模型': "('CONTROL_NET',)",
    'CN结束': "('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 1.0, 'step': 0.05})",
    'Latent': "('LATENT',)",
    'MPB颜色保持': "('FLOAT', {'default': 0.5, 'min': 0.0, 'max': 1.0, 'step': 0.05, 'tooltip': 'MPB模式下的颜色保持强度，0=完全融合，1=完全保持'})",
    'Poisson迭代次数': "('INT', {'default': 40, 'min': 0, 'max': 100, 'step': 5})",
    'SAM模型手动路径': "('STRING', {'default': '', 'tooltip': '手动输入SAM模型路径（选择[手动输入路径]时使用）'})",
    'SAM模型选择': "(SAMModelManager.get_model_list(), {'default': SAMModelManager.get_model_list()[0], 'tooltip': '选择SAM模型（支持v1/v2/v2.1，自动检测版本）'})",
    'SAM置信度阈值': "('FLOAT', {'default': 0.8, 'min': 0.0, 'max': 1.0, 'step': 0.05, 'tooltip': 'SAM分割置信度阈值，提高可获得更精确的分割'})",
    'Seg边缘羽化': "('FLOAT', {'default': 3.0, 'min': 0.0, 'max': 10.0, 'step': 0.5})",
    'Tile大小': "('INT', {'default': 640, 'min': 320, 'max': 1280, 'step': 64})",
    'Tile重叠': "('FLOAT', {'default': 0.3, 'min': 0.1, 'max': 0.5, 'step': 0.05})",
    'VAE': "('VAE',)",
    'WIoU权重参数': "('FLOAT', {'default': 3.0, 'min': 1.0, 'max': 10.0, 'step': 0.5, 'tooltip': 'WIoU聚焦参数，越大越关注困难样本'})",
    'YOLO推理尺寸': "('INT', {'default': 768, 'min': 320, 'max': 1280, 'step': 32, 'tooltip': 'YOLO imgsz参数，提高尺寸可提升小目标检测精度'})",
    'YOLO模型': "(yolo_models if yolo_models else ['[手动输入路径]'], {'default': yolo_models[0] if yolo_models else '[手动输入路径]'})",
    'YOLO模型2': "(['[不使用]'] + (yolo_models if yolo_models else []), {'default': '[不使用]', 'tooltip': '多模型模式：第2个检测模型'})",
    'YOLO模型2置信度': "('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.05, 'tooltip': 'YOLO模型2的独立置信度（0=使用主置信度）'})",
    'YOLO模型3': "(['[不使用]'] + (yolo_models if yolo_models else []), {'default': '[不使用]', 'tooltip': '多模型模式：第3个检测模型'})",
    'YOLO模型3置信度': "('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.05, 'tooltip': 'YOLO模型3的独立置信度（0=使用主置信度）'})",
    'YOLO模型4': "(['[不使用]'] + (yolo_models if yolo_models else []), {'default': '[不使用]', 'tooltip': '多模型模式：第4个检测模型'})",
    'YOLO模型4置信度': "('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.05, 'tooltip': 'YOLO模型4的独立置信度（0=使用主置信度）'})",
    'YOLO路径': "('STRING', {'default': '', 'multiline': False, 'placeholder': '手动输入模型完整路径'})",
    'seed': "('INT', {'default': 0, 'min': 0, 'max': 18446744073709551615})",
    '二次检测扩展': "('INT', {'default': 32, 'min': 0, 'max': 256, 'step': 8})",
    '二次检测最多目标': "('INT', {'default': 6, 'min': 1, 'max': 50, 'step': 1})",
    '二次检测精修': "('BOOLEAN', {'default': False})",
    '低显存模式': "('BOOLEAN', {'default': False})",
    '使用Latent感知插值': "('BOOLEAN', {'default': True})",
    '使用Seg分割': "('BOOLEAN', {'default': True, 'tooltip': '使用YOLO-seg模型的分割mask（需要seg模型）'})",
    '使用WIoU优化': "('BOOLEAN', {'default': True, 'tooltip': '使用Weighted IoU提高边界框定位精度10-15%'})",
    '修复优先级': "(['按置信度', '按面积', '脸优先', '手优先', '脚优先', '自定义优先级'], {'default': '脸优先'})",
    '修复轮次': "('INT', {'default': 1, 'min': 1, 'max': 6, 'step': 1})",
    '全图降噪强度': "('FLOAT', {'default': 0.25, 'min': 0.05, 'max': 0.5, 'step': 0.05, 'tooltip': '修复后全图降噪的强度，建议0.2-0.35'})",
    '区域CFG': "('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 100.0, 'step': 0.5})",
    '区域步数': "('INT', {'default': 0, 'min': 0, 'max': 10000, 'step': 1})",
    '区域降噪': "('FLOAT', {'default': 0.65, 'min': 0.0, 'max': 1.0, 'step': 0.05, 'tooltip': '提高降噪强度可获得更彻底的修复效果'})",
    '参考帧窗口': "('INT', {'default': 2, 'min': 1, 'max': 5, 'step': 1, 'tooltip': '用于时序一致性的参考帧数量'})",
    '合并IOU阈值': "('FLOAT', {'default': 0.35, 'min': 0.0, 'max': 0.95, 'step': 0.05})",
    '合并仅同类': "('BOOLEAN', {'default': True})",
    '合并后二次扩张像素': "('INT', {'default': 0, 'min': 0, 'max': 256, 'step': 8})",
    '合并后二次扩张比例': "('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.05})",
    '合并重叠框': "('BOOLEAN', {'default': True})",
    '启用EMA注意力': "('BOOLEAN', {'default': False, 'tooltip': '启用多尺度注意力机制，聚焦关键区域'})",
    '启用SAM精细化': "('BOOLEAN', {'default': False, 'tooltip': '使用SAM模型精细化分割（支持SAM/SAM2/SAM2.1）'})",
    '启用Tile检测': "('BOOLEAN', {'default': True, 'tooltip': '大图分块检测，提高召回率'})",
    '启用YOLO': "('BOOLEAN', {'default': True})",
    '启用多模型': "('BOOLEAN', {'default': False, 'tooltip': '启用多模型分批检测，降低显存压力，提高检测准确率'})",
    '启用性能分析': "('BOOLEAN', {'default': False, 'tooltip': '显示各步骤耗时统计，帮助定位性能瓶颈'})",
    '多尺度检测': "('BOOLEAN', {'default': True, 'tooltip': '使用多个尺度检测同一图像，提高准确度（略慢）'})",
    '工作模式': "(['仅修复(保持输入Latent)', '修复后全图降噪'], {'default': '仅修复(保持输入Latent)'})",
    '左右分组保护': "('BOOLEAN', {'default': True})",
    '开启裁剪精修': "('BOOLEAN', {'default': True})",
    '手脚CFG加成': "('FLOAT', {'default': 1.2, 'min': 1.0, 'max': 2.0, 'step': 0.1, 'tooltip': '手脚区域CFG增强倍数（1.2-1.5效果最佳，过高会导致过度饱和）'})",
    '手脚CN强度': "('FLOAT', {'default': 0.7, 'min': 0.3, 'max': 1.0, 'step': 0.05, 'tooltip': '手脚区域ControlNet强度（0.5-0.8效果最佳，1.0可能丢失自然纹理）'})",
    '手脚增强模式': '(\'BOOLEAN\', {\'default\': True, \'tooltip\': "手脚区域自动提高降噪、扩展mask、增加padding/steps（建议在prompt中添加\'perfect hands, detailed fingers\'以获得更好效果）"})',
    '手脚衰减下限': "('FLOAT', {'default': 0.4, 'min': 0.2, 'max': 0.8, 'step': 0.05, 'tooltip': '多轮修复时降噪的最低值（防止后续轮次降噪过低）'})",
    '手脚降噪下限': "('FLOAT', {'default': 0.8, 'min': 0.5, 'max': 1.0, 'step': 0.05, 'tooltip': '手脚区域的最低降噪值（V6.10提升至0.80，参考HandRefiner最佳实践）'})",
    '手脚降噪衰减': "('FLOAT', {'default': 0.85, 'min': 0.5, 'max': 1.0, 'step': 0.05, 'tooltip': '多轮修复时降噪的衰减系数（每轮 denoise *= 衰减系数）'})",
    '手脚额外轮次': "('INT', {'default': 1, 'min': 0, 'max': 3, 'step': 1, 'tooltip': '手脚区域额外修复轮次'})",
    '手骨架CN模型': "('CONTROL_NET', {'tooltip': '手部骨架专用ControlNet模型（推荐: control_openpose / control_dw_pose）。提供此模型后，修手时自动用MediaPipe提取骨架注入，避免生成多余手指'})",
    '按置信度自适应降噪': "('BOOLEAN', {'default': True})",
    '排序方式': "(['置信度', '面积'], {'default': '置信度'})",
    '推理精度': "(['FP32(默认)', 'FP16(省显存)', 'BF16(平衡)'], {'default': 'FP32(默认)', 'tooltip': 'FP16可节省40%显存并提速，BF16在支持的GPU上更稳定'})",
    '放大倍数': "('FLOAT', {'default': 1.0, 'min': 1.0, 'max': 4.0, 'step': 0.25})",
    '放大算法': "(['nearest', 'bilinear', 'bicubic', 'area', 'latent_aware'],)",
    '时序一致性强度': "('FLOAT', {'default': 0.3, 'min': 0.0, 'max': 1.0, 'step': 0.05, 'tooltip': '帧间平滑强度，越大越平滑但可能损失细节'})",
    '智能策略': "(['关闭', '人像(脸/手/脚)', '通用(按面积)'], {'default': '人像(脸/手/脚)'})",
    '最多目标数': "('INT', {'default': 0, 'min': 0, 'max': 200, 'step': 1})",
    '未检测时处理': "(['保持原样', '全图重绘'], {'default': '保持原样'})",
    '检测数据': "('QIANHUI_DETECT',)",
    '模型': "('MODEL',)",
    '正面提示': "('CONDITIONING',)",
    '步数': "('INT', {'default': 20, 'min': 1, 'max': 10000})",
    '每帧清理缓存': "('BOOLEAN', {'default': False})",
    '每类最多目标数': "('INT', {'default': 0, 'min': 0, 'max': 50, 'step': 1})",
    '注意力聚焦强度': "('FLOAT', {'default': 1.5, 'min': 1.0, 'max': 3.0, 'step': 0.1, 'tooltip': '注意力聚焦系数，越大越集中于高置信度区域'})",
    '清晰度恢复强度': "('FLOAT', {'default': 0.3, 'min': 0.0, 'max': 1.0, 'step': 0.05, 'tooltip': '修复后的细节和清晰度恢复强度，可减少朦胧感'})",
    '目标类别': "('STRING', {'default': '', 'multiline': False, 'placeholder': '留空=全部 | 支持部分匹配: foot,hand,face'})",
    '类别规则': "('STRING', {'default': '', 'multiline': True, 'placeholder': '自定义规则格式:\\nface: steps=20,denoise=0.4,limit=4\\nhand: denoise=0.75,limit=2'})",
    '类别规则预设': "(['[不使用预设]', '人像优化(face高质量)', '手部优化(hand高降噪)', '脚部优化(foot高降噪)', '人像全面(face+hand+foot)', '自定义规则'], {'default': '[不使用预设]', 'tooltip': '快速应用预设规则模板'})",
    '结果缓存设备': "(['自动', 'GPU', 'CPU'], {'default': '自动'})",
    '置信度': "('FLOAT', {'default': 0.25, 'min': 0.0, 'max': 1.0, 'step': 0.05, 'tooltip': 'YOLO模型1的检测置信度阈值'})",
    '自定义优先级关键词': "('STRING', {'default': 'face,hand,foot', 'multiline': False})",
    '自适应降噪_低置信缩放': "('FLOAT', {'default': 0.85, 'min': 0.1, 'max': 1.5, 'step': 0.05})",
    '自适应降噪_高置信缩放': "('FLOAT', {'default': 1.0, 'min': 0.1, 'max': 1.5, 'step': 0.05})",
    '色差修复强度': "('FLOAT', {'default': 0.85, 'min': 0.0, 'max': 1.0, 'step': 0.05, 'tooltip': '提高色差修复强度可更好地平衡修复区域和原始区域的色彩'})",
    '融合模式': "(['Poisson融合', 'MPB增强融合', '频率感知', '简单混合', '智能柔和融合'], {'default': '智能柔和融合', 'tooltip': '智能柔和融合：多层次渐进羽化+色彩对齐+细节恢复，柔化色差和朦胧感'})",
    '裁剪CFG': "('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 100.0, 'step': 0.5})",
    '裁剪使用ControlNet': "('BOOLEAN', {'default': True})",
    '裁剪扩展': "('INT', {'default': 64, 'min': 0, 'max': 512, 'step': 8})",
    '裁剪放大倍率': "('FLOAT', {'default': 1.6, 'min': 1.0, 'max': 4.0, 'step': 0.1})",
    '裁剪最大边': "('INT', {'default': 512, 'min': 128, 'max': 2048, 'step': 64})",
    '裁剪步数': "('INT', {'default': 20, 'min': 1, 'max': 200, 'step': 1, 'tooltip': '增加步数可提升修复质量，推荐18-25步'})",
    '裁剪轮次': "('INT', {'default': 2, 'min': 1, 'max': 4, 'step': 1})",
    '裁剪降噪': "('FLOAT', {'default': 0.6, 'min': 0.0, 'max': 1.0, 'step': 0.05, 'tooltip': '提高降噪可获得更高质量的修复，推荐0.55-0.70'})",
    '视频模式': "('BOOLEAN', {'default': False, 'tooltip': '启用视频模式，优化帧间一致性'})",
    '调度器': '(comfy.samplers.KSampler.SCHEDULERS,)',
    '负面提示': "('CONDITIONING',)",
    '过滤最大长宽比': "('FLOAT', {'default': 10.0, 'min': 1.0, 'max': 50.0, 'step': 0.5})",
    '过滤最大面积比': "('FLOAT', {'default': 1.0, 'min': 0.05, 'max': 1.0, 'step': 0.01})",
    '过滤最小长宽比': "('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 5.0, 'step': 0.05})",
    '过滤最小面积比': "('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 0.5, 'step': 0.01})",
    '逐次降噪递减': "('BOOLEAN', {'default': True})",
    '遮罩扩展': "('INT', {'default': 32, 'min': 0, 'max': 256, 'step': 8})",
    '遮罩羽化': "('INT', {'default': 24, 'min': 0, 'max': 128, 'step': 4, 'tooltip': '增加羽化可让边缘更柔和自然，建议20-32'})",
    '配置预设': "(['[不使用预设]', '快速模式(低质量)', '平衡模式(推荐)', '质量模式(慢速)', '极致质量(超慢)', '省显存模式', '自定义'], {'default': '[不使用预设]', 'tooltip': '一键应用预设配置，快速适配不同场景'})",
    '采样器': '(comfy.samplers.KSampler.SAMPLERS,)',
    '降噪': "('FLOAT', {'default': 0.75, 'min': 0.0, 'max': 1.0, 'step': 0.01})",
    '骨架CN强度': "('FLOAT', {'default': 0.65, 'min': 0.1, 'max': 1.5, 'step': 0.05, 'tooltip': '手骨架ControlNet引导强度（0.5-0.75通常效果最好）'})",
    '骨架CN结束': "('FLOAT', {'default': 0.7, 'min': 0.0, 'max': 1.0, 'step': 0.05, 'tooltip': '骨架CN在扩散过程的结束步比例（0.7=前70%步有骨架约束，后30%自由细化）'})",
}

#: (节点短名, 文件名, 类名) —— 顺序就是链条顺序。
SPLIT_NODES = (
    ("detect", "node_detect.py", "QianhuiYoloDetect"),
    ("sam", "node_sam.py", "QianhuiSamRefine"),
    ("repair", "node_repair.py", "QianhuiRegionRepair"),
    ("global", "node_global.py", "QianhuiGlobalPass"),
)

#: 每个拆出来的节点**承接哪些 pipeline 阶段模块**（决定它的字段集）。
NODE_STAGE_MODULES = {
    "detect": (("detect.py", None),),
    "sam": (("sam_refine.py", None),),
    "repair": (("repair.py", None),),
    # `build_info` 归 D（「检测信息」由它产出）。
    "global": (("global_pass.py", None), ("report_bits.py", "build_info")),
}

#: 15 个「只被 prepare.py 读」的字段 → 归哪个节点。
PREPARE_ONLY_OWNER = {
    "target_classes_str": ("detect",),
    "yolo_model_selection": ("detect",),
    "manual_path": ("detect",),
    "yolo_model2": ("detect",),
    "yolo_model3": ("detect",),
    "yolo_model4": ("detect",),
    "confidence2": ("detect",),
    "confidence3": ("detect",),
    "confidence4": ("detect",),
    "enable_multi_model": ("detect",),
    "class_rules_preset": ("detect", "repair"),
    "class_rules_text": ("detect", "repair"),
    "sam_model_selection": ("sam",),
    "sam_model_manual": ("sam",),
    "cache_device_mode": ("global",),
}

#: 节点**读了自己没有控件的** cfg 字段 —— 逐条、带原因（终审 M-2 / M-3）。
#:
#: 为什么需要这张表：`_node_stage_fields()` 原先只统计**pipeline 阶段模块**的读取，
#: 于是「节点自己的 `_prepare` / `process` 里读了一个它不拥有的 cfg 字段」这类越界
#: **完全看不见**（终审 M-3 实测：已有的 `color_strength` 就是这么漏的）。
#: 把 node 文件的读取并进口径之后，下面这两条会当场变红 —— 所以在此显式豁免。
#:
#: ★ 两条都由终审**逐行核实过：无数值影响**，因此不为此改行为（改了反而偏离单体）：
#:   · `hand_skeleton_cn_model`（M-2）：节点 A 用它派生 `ctx.has_hand_skeleton_cn`，
#:     而该值只喂给 `detect.py:134` 那句 `_decoded_np = decoded_original.numpy()`。
#:     真正读 `_decoded_np` 的是 `repair.py:240`，节点 C 已用**同一个源**
#:     （`frame.decoded`）自己重绑过 ⇒ A 侧那条分支恒不执行，是个**静默死分支**。
#:   · `color_strength`（M-3）：节点 A 的启动横幅打印它。A 不拥有这个控件 ⇒
#:     用户改的是节点 C 上的值，A 的横幅永远显示声明默认值 —— **纯日志差异**。
#:
#: ★ 新增任何**其它**「读了自己没有的控件」都会红 —— 那正是这条口径的意义。
UNOWNED_CFG_READS = {
    ("detect", "hand_skeleton_cn_model"):
        "M-2：只喂给 detect.py:134 的 `_decoded_np`，而那个变量在节点 A 恒不被读"
        "（节点 C 用同一个 `frame.decoded` 自己重绑）。死分支，无数值影响。",
    ("detect", "color_strength"):
        "M-3：节点 A 的启动横幅打印它（纯日志）。A 不声明该控件 ⇒ 改的是 C 上的值，"
        "A 永远打印声明默认值。无数值影响。",
}

#: 节点 D 从 `payload.stage_flags` 取的值 → 它不需要对应控件。
STAGE_FLAG_FIELDS = frozenset(("blend_mode", "use_latent_aware", "enable_yolo", "enable_crop"))

#: D 从 `stage_flags` 取的**非 cfg** 键（它落在 `ctx.放大倍数` 这个槽上，不是 cfg 字段）。
STAGE_FLAG_NON_CFG_KEYS = frozenset(("放大倍数",))

#: 计划 §四个节点的名册 里写死的返回契约（类名 == 注册名）。
NODE_RETURN_CONTRACT = {
    "QianhuiYoloDetect": (("IMAGE", "QIANHUI_DETECT", "LATENT"),
                          ("检测预览", "检测数据", "检测用Latent")),
    "QianhuiSamRefine": (("QIANHUI_DETECT",), ("检测数据",)),
    "QianhuiRegionRepair": (("LATENT", "MASK"), ("修复后Latent", "检测遮罩")),
    "QianhuiGlobalPass": (("LATENT", "STRING"), ("Latent", "检测信息")),
}


def _split_node_path(fname):
    return os.path.join(SPLIT_NODE_DIR, fname)


def _parse(path):
    with io.open(path, encoding="utf-8") as fh:
        return ast.parse(fh.read())


def _find_class(tree, class_name):
    for n in ast.walk(tree):
        if isinstance(n, ast.ClassDef) and n.name == class_name:
            return n
    return None


def _input_types_sections(path, class_name):
    """AST 取 `<class_name>.INPUT_TYPES()` 里 return 的那个 dict 的 {required/optional: Dict}。

    与 `test_latent_algorithms._latent_node_widget_decls` 同一套定位法（找同时含
    `required` 与 `optional` 两个字面量键的 dict），只是不写死文件与类名。
    """
    tree = _parse(path)
    cls = _find_class(tree, class_name)
    assert cls is not None, "%s 里找不到类 %s" % (path, class_name)
    fn = None
    for sub in ast.walk(cls):
        if isinstance(sub, ast.FunctionDef) and sub.name == "INPUT_TYPES":
            fn = sub
            break
    assert fn is not None, "%s 的 %s 没有 INPUT_TYPES()" % (path, class_name)

    root = None
    for sub in ast.walk(fn):
        if not isinstance(sub, ast.Return) or not isinstance(sub.value, ast.Dict):
            continue
        keys = {k.value for k in sub.value.keys if isinstance(k, ast.Constant)}
        if {"required", "optional"} <= keys:
            root = sub.value
            break
    assert root is not None, (
        "%s 的 INPUT_TYPES() 里找不到同时含 required/optional 的 dict 字面量" % path)

    out = {}
    for k, v in zip(root.keys, root.values):
        if isinstance(k, ast.Constant) and k.value in ("required", "optional"):
            out[k.value] = v
    assert set(out) == {"required", "optional"}, "%s 的 INPUT_TYPES 少了 required/optional" % path
    return out


def _node_widget_decls(path, class_name):
    """{控件名: `ast.unparse(声明)`}（required 与 optional 合并）。

    比的是**声明的规范化源码**而不是某个「类别串」：类别串对 COMBO 是整张 options 表，
    对 INT 只是一句 "INT"，退化成「类型对了就算相等」—— 而计划 §验证③ 要的正是
    「同名共享控件的**定义**逐个相等」（默认值、min/max/step、tooltip 全都要一样）。
    """
    sections = _input_types_sections(path, class_name)
    out = {}
    for section in ("required", "optional"):
        node = sections[section]
        for k, v in zip(node.keys, node.values):
            assert isinstance(k, ast.Constant), "%s 里有个非字面量控件名" % path
            name = k.value
            assert name not in out, (
                "%s 的控件 %r 同时出现在 required 与 optional 里" % (path, name))
            out[name] = ast.unparse(v)
    return out


def _node_widgets(path, class_name):
    return set(_node_widget_decls(path, class_name))


def _split_node_file(node_key):
    """节点短名 → 文件名（`SPLIT_NODES` 是唯一权威）。"""
    for key, fname, _cls in SPLIT_NODES:
        if key == node_key:
            return fname
    raise KeyError(node_key)


def _node_stage_fields(node_key):
    """节点承接的阶段模块读到的 cfg 字段 ∪ 归它的那 15 个 prepare-only 字段
    ∪ **它自己文件里的** cfg 读取。

    ★ 终审 M-3：最后这一项是补上的口径缺口。原先只统计 pipeline 阶段模块，
      于是「节点自己的 `_prepare` / `process` 里读了一个它不拥有的 cfg 字段」
      这类越界看不见（实例：`node_detect.py` 的横幅打印 `ctx.cfg.color_strength`）。
      补上之后，已存在的两条越界由 `UNOWNED_CFG_READS` 逐条豁免 —— 新增的会红。
    """
    fields = set()
    for module, func in NODE_STAGE_MODULES[node_key]:
        reads, _ = _cfg_access(os.path.join(PIPELINE_DIR, module))
        fields |= reads
    for field, owners in PREPARE_ONLY_OWNER.items():
        if node_key in owners:
            fields.add(field)
    # ★ M-3：把节点**自己那个文件**的 cfg 读取并进来（`_prepare` / `process` 都在里面）。
    node_reads, _ = _cfg_access(_split_node_path(_split_node_file(node_key)))
    exempt = {f for (nk, f) in UNOWNED_CFG_READS if nk == node_key}
    fields |= node_reads - exempt
    return fields - STAGE_FLAG_FIELDS


def _class_assigns(path, class_name):
    """类体里的赋值 `{名字: unparse}`（RETURN_TYPES / FUNCTION / CATEGORY …）。"""
    cls = _find_class(_parse(path), class_name)
    assert cls is not None, "%s 里找不到类 %s" % (path, class_name)
    out = {}
    for st in cls.body:
        if isinstance(st, ast.Assign):
            for t in st.targets:
                if isinstance(t, ast.Name):
                    out[t.id] = ast.unparse(st.value)
    return out


def _stage_flag_keys_written():
    """4 个节点里**写进** `stage_flags` 的字符串键（两种写法都认，终审 M8）。

    · `DetectionPayload(..., stage_flags={"k": ..., ...})` —— 关键字 + dict 字面量
      （节点 A 建 payload 时就是这么写的）
    · `payload.stage_flags["k"] = ...`                    —— 下标赋值（节点 C 那三处）
    """
    keys = set()
    for _key, fname, _cls in SPLIT_NODES:
        tree = _parse(_split_node_path(fname))
        for n in ast.walk(tree):
            if isinstance(n, ast.Call):
                for kw in n.keywords:
                    if kw.arg == "stage_flags" and isinstance(kw.value, ast.Dict):
                        for k in kw.value.keys:
                            if isinstance(k, ast.Constant) and isinstance(k.value, str):
                                keys.add(k.value)
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    if (isinstance(t, ast.Subscript)
                            and isinstance(t.value, ast.Attribute)
                            and t.value.attr == "stage_flags"
                            and isinstance(t.slice, ast.Constant)
                            and isinstance(t.slice.value, str)):
                        keys.add(t.slice.value)
    return keys


def _stage_flag_keys_read_by_global():
    """节点 D 从 `payload.stage_flags` **读**的字符串键（终审 M8）。

    做法：先在 `node_global.py` 里找「值里出现 `stage_flags` 的那个赋值」拿到本地名
    （今天是 `flags = 检测数据.stage_flags or {}`），再收集 `flags.get("k", ...)`
    第一个实参的字面量。

    ★ 认不出本地名就**断言失败**：静默返回空集会让「读的键都有生产者」变成永真，
      那就成了一条假网。
    """
    tree = _parse(_split_node_path("node_global.py"))
    names = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and "stage_flags" in ast.unparse(n.value):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
    assert names, ("node_global.py 里找不到「从 stage_flags 取一份到本地名」的赋值 ——"
                   " M8 的口径失效，别让它静默通过")
    keys = set()
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "get" and isinstance(n.func.value, ast.Name)
                and n.func.value.id in names and n.args
                and isinstance(n.args[0], ast.Constant)
                and isinstance(n.args[0].value, str)):
            keys.add(n.args[0].value)
    assert keys, ("node_global.py 里读不到任何 stage_flags 键 —— 口径失效"
                  "（D 至少读 4 个 cfg 标志 + 放大倍数）")
    return keys


def _assign_targets_in_the_frame_loop(fname, cls_name, loop_iter_substr):
    """`process()` 里 `for ... in <含 loop_iter_substr>...:` 那个循环体内的赋值目标（unparse）。

    含循环体内**嵌套**块（`with` / `if` / `try`）里的赋值 —— 逐帧重绑就藏在那里。
    """
    cls = _find_class(_parse(_split_node_path(fname)), cls_name)
    assert cls is not None, "%s 里找不到类 %s" % (fname, cls_name)
    fn = next((s for s in ast.walk(cls)
               if isinstance(s, ast.FunctionDef) and s.name == "process"), None)
    assert fn is not None, "%s 的 %s 没有 process()" % (fname, cls_name)
    loop = next((n for n in ast.walk(fn) if isinstance(n, ast.For)
                 and loop_iter_substr in ast.unparse(n.iter)), None)
    assert loop is not None, (
        "%s.process 里找不到 `for ... in %s…` 那个循环 —— 口径失效，别让它静默通过"
        % (fname, loop_iter_substr))
    out = []
    for n in ast.walk(loop):
        if isinstance(n, ast.Assign):
            out.extend(ast.unparse(t) for t in n.targets)
        elif isinstance(n, ast.AnnAssign) and n.value is not None:
            out.append(ast.unparse(n.target))
    return out


class TestSplitNodeWidgets(unittest.TestCase):
    """4 个节点的控件表：字段覆盖、共享控件同源、整体不漏、契约一致。"""

    def test_each_node_field_set_is_covered_by_its_widgets(self):
        """★ 每节点「它读的 cfg 字段」必须全部在它的控件表里。

        少了就是那个旋钮在本节点上**不存在**：用户改不了、后端读声明默认值、
        界面看着正常、什么都不报 —— 正是本计划猎的那一类静默失效。
        """
        key_map = _context_key_map()
        for node_key, fname, cls_name in SPLIT_NODES:
            fields = _node_stage_fields(node_key)
            widgets = _node_widgets(_split_node_path(fname), cls_name)
            missing = sorted(key_map[f] for f in fields if key_map[f] not in widgets)
            self.assertEqual(
                missing, [],
                "%s（%s）少了这些控件：%s\n"
                "  它们对应的 cfg 字段被本节点的阶段模块读到，却没有对应的控件 ——\n"
                "  那个旋钮在本节点上不存在，用户改不了、后端读的是声明默认值。"
                % (cls_name, fname, missing),
            )

    def test_shared_widget_declarations_are_identical_across_nodes(self):
        """★ 同名共享控件的**定义**必须逐个相等（计划 §验证③）。

        「跨节点控件两边各放一份，用户手工保持一致」（决策 3）的代价是漂移：
        两个节点同名控件的默认值/范围不同时，选哪个节点就得到哪套行为，
        而没有任何一条测试会红。这里把「手工保持一致」从「值」收窄成
        「只在运行时选择上可以不同」。
        """
        decls = {
            node_key: _node_widget_decls(_split_node_path(fname), cls_name)
            for node_key, fname, cls_name in SPLIT_NODES
        }
        names = {}
        for node_key, table in decls.items():
            for name in table:
                names.setdefault(name, []).append(node_key)

        bad = []
        for name, owners in sorted(names.items()):
            if len(owners) < 2:
                continue
            specs = {owner: decls[owner][name] for owner in owners}
            if len(set(specs.values())) > 1:
                bad.append("    %s（出现在 %s）：\n%s" % (
                    name, "、".join(owners),
                    "\n".join("        %-8s %s" % (o, s) for o, s in sorted(specs.items()))))
        self.assertEqual(
            bad, [],
            "同名共享控件的声明在 4 个节点之间不一致：\n%s\n"
            "  ⇒ 用户在哪个节点上改，就得到哪一套行为。必须逐字相等\n"
            "    （类型名、默认值、min/max/step、tooltip、options 顺序）。" % "\n".join(bad),
        )

    def test_the_four_nodes_together_cover_every_widget_of_the_old_node(self):
        """★ 反「旋钮消失」：4 个节点的控件并集必须**恰好**等于冻结的 114 条。

        上一条只保证「同名的相等」，这一条保证「一个都没丢、也没偷偷多」。

        ★★ 终审 C-附3①：本条原先拿 `set(_latent_node_widget_decls())` 当参照物 ——
          T13 把那个函数的数据源换成「4 个节点并集」之后，等号两边成了**同一份东西**，
          本条退化成同义反复（删 `工作模式` / `二次检测精修` 都不红）。
          现在参照物换成模块级**冻结快照** `SPLIT_WIDGET_DECL_SNAPSHOT`，
          独立复算的关系重新成立。
        """
        union = set()
        for node_key, fname, cls_name in SPLIT_NODES:
            union |= _node_widgets(_split_node_path(fname), cls_name)
        frozen = set(SPLIT_WIDGET_DECL_SNAPSHOT)
        self.assertEqual(
            sorted(frozen - union), [],
            "这些控件在冻结快照里有、4 个节点里**一个都没有**（界面上的旋钮消失了）：%s"
            % sorted(frozen - union),
        )
        self.assertEqual(
            sorted(union - frozen), [],
            "4 个节点里出现了快照没有的控件（新增界面元素要同时更新 "
            "SPLIT_WIDGET_DECL_SNAPSHOT 并说明为什么）：%s" % sorted(union - frozen),
        )

    def test_every_widget_declaration_matches_the_frozen_snapshot(self):
        """★ 每个控件的**整条声明**必须与冻结快照逐字相同（终审 C-附3②）。

        `test_shared_widget_declarations_are_identical_across_nodes` 只查「同名的相等」，
        对**只出现在一个节点里**的控件一言不发 —— 终审的变异电池实测：把 `node_sam`
        独有的 `SAM置信度阈值` 默认值 `0.8 → 0.9`，整个套件**全绿**。
        这条把 114 条的绝对值钉死：类型、default、min/max/step、tooltip、options 顺序。
        """
        union = {}
        for node_key, fname, cls_name in SPLIT_NODES:
            for name, decl in _node_widget_decls(_split_node_path(fname), cls_name).items():
                union.setdefault(name, decl)
        frozen = SPLIT_WIDGET_DECL_SNAPSHOT
        missing = sorted(set(frozen) - set(union))
        added = sorted(set(union) - set(frozen))
        changed = sorted(n for n in set(union) & set(frozen) if union[n] != frozen[n])
        self.assertEqual(
            (missing, added, changed), ([], [], []),
            "控件声明的冻结快照被动了：\n"
            "  消失：%s\n  新增：%s\n"
            "  改动：%s\n"
            "  ⇒ 这直接改用户界面 —— 字段变少 = 旋钮没了；默认值 / min / max / step /\n"
            "     tooltip / options 变了 = 行为或提示变了。确认是有意为之，再更新\n"
            "     SPLIT_WIDGET_DECL_SNAPSHOT（并在同一次提交里说明为什么）。"
            % (missing, added,
               [(n, frozen[n], union[n]) for n in changed]),
        )

    # ---------------- 终审 M8 / M6 / M7 补的三条网 ----------------

    def test_every_stage_flag_read_by_node_d_is_written_by_some_node(self):
        """★ 终审 M8：节点 D 从 `payload.stage_flags` 读的**每个键**都要有生产者。

        那 4 个「不复制控件」的值（融合模式 / Latent感知插值 / YOLO / 裁剪精修）
        由节点 C 写进 `payload.stage_flags`、D 在 `_prepare` 里用
        `flags.get(k, ctx.cfg.k)` **覆盖**掉自己的声明默认值，
        好让 `检测信息` 复述的是各阶段**实际用的**值。

        删掉 C 那三处写入 ⇒ D 静默退回声明默认值 ⇒ 检测信息报的不是实际值，
        **而既有守卫一条都不会红**（终审的变异电池实测：守卫层全绿，只有
        `test_split_matches_golden` 红 5 例）。这条把「读的键 ⊆ 写的键」钉住。
        """
        written = _stage_flag_keys_written()
        read = _stage_flag_keys_read_by_global()
        self.assertEqual(
            sorted(read - written), [],
            "节点 D 从 payload.stage_flags 读了这些键，但 4 个节点里**没有任何地方写过**：%s\n"
            "  ⇒ D 会静默退回它自己的声明默认值，`检测信息` 复述的就不是实际用的值。\n"
            "     （写入口在节点 A 的 `stage_flags={...}` 与节点 C 的 "
            "`payload.stage_flags[k] = ...`）" % sorted(read - written),
        )

    def test_stage_flag_exemption_tables_match_what_d_actually_reads(self):
        """`STAGE_FLAG_FIELDS` / `STAGE_FLAG_NON_CFG_KEYS` 必须与「D 真的读了什么」对齐。

        这两张表是 `_node_stage_fields()` 的**豁免依据**（D 读它们但不需要对应控件）。
        表一旦与代码脱节就会**悄悄放宽** G1：D 多读一个 cfg 字段而表没跟上，
        那个字段就同时不受「控件覆盖」与「stage_flags 生产」两条网约束。
        所以这里把「表 == 实际读的键」直接钉死。
        """
        read = _stage_flag_keys_read_by_global()
        self.assertEqual(
            read, set(STAGE_FLAG_FIELDS) | set(STAGE_FLAG_NON_CFG_KEYS),
            "D 实际读的 stage_flags 键与两张豁免表对不上：\n"
            "  实际读: %s\n  表里 cfg 侧: %s\n  表里非 cfg 侧: %s\n"
            "  ⇒ 表脱节 = G1 被悄悄放宽。" % (sorted(read), sorted(STAGE_FLAG_FIELDS),
                                             sorted(STAGE_FLAG_NON_CFG_KEYS)),
        )

    def test_the_frame_loop_rebinds_the_per_frame_scratch_slots(self):
        """★ 终审 M6 / M7：帧循环里必须**逐帧重绑**的两个 ctx 槽。

        这两处「少一次赋值」在守卫层完全看不见，而行为上是真差异：

        · **M6** `node_repair` 的 `ctx.detection_mask`：`repair.py` 只在
          `if ctx.detections:` 里设它 ⇒ 删掉这处复位，**无检测帧会把上一帧的遮罩**
          append 进 `ctx.out_masks`（终审实测：红 2 个无检测场景）。
        · **M7** `node_detect` 的 `ctx.out_latents[i:i+1]`：承载张量是
          `torch.empty(...)`（`detect.py:41-42`）且整块喂给节点 C ⇒ 删掉填充
          等于**把未初始化内存传给下游**（终审实测：红 30 个子用例）。

        判据取的是「循环体内存在这样的赋值目标」，而不是「值对不对」——
        值那一层由 `test_split_matches_golden` 管，两条网互补。
        """
        repair_targets = _assign_targets_in_the_frame_loop(
            "node_repair.py", "QianhuiRegionRepair", "iter_frames_in_order")
        self.assertIn(
            "ctx.detection_mask", repair_targets,
            "node_repair.process 的帧循环里没有 `ctx.detection_mask` 的**逐帧复位**。\n"
            "  ⇒ 无检测帧会把上一帧的遮罩 append 进 `ctx.out_masks`（M6）。\n"
            "  实测到的赋值目标：%s" % sorted(repair_targets),
        )
        detect_targets = _assign_targets_in_the_frame_loop(
            "node_detect.py", "QianhuiYoloDetect", "range(")
        self.assertTrue(
            any(t.startswith("ctx.out_latents[") for t in detect_targets),
            "node_detect.process 的帧循环里没有 `ctx.out_latents[...] = ...` 的填充。\n"
            "  ⇒ `torch.empty` 分配的承载张量会把**未初始化内存**传给下游（M7）。\n"
            "  实测到的赋值目标：%s" % sorted(detect_targets),
        )

    def test_every_node_offers_the_same_preset_dropdown(self):
        """4 个节点各自的 `配置预设` 都必须逐字等于 `presets.PRESET_LABELS`。

        T12 把预设表按节点拆了，但**档位清单**仍只有一份权威（`presets.py`）。
        某个节点手抄漏一档 ⇒ 那一档在它的界面上根本不存在，而所有测试都绿。
        """
        pres = load_module("qh_presets_for_nodes",
                           os.path.join(SPLIT_NODE_DIR, "presets.py"))
        for node_key, fname, cls_name in SPLIT_NODES:
            sections = _input_types_sections(_split_node_path(fname), cls_name)
            combo = None
            for k, v in zip(sections["optional"].keys, sections["optional"].values):
                if isinstance(k, ast.Constant) and k.value == "配置预设":
                    combo = v
            self.assertIsNotNone(combo, "%s 的 optional 里没有 配置预设" % fname)
            options = [elt.value for elt in combo.elts[0].elts] if isinstance(combo, ast.Tuple) else None
            self.assertEqual(
                options, pres.PRESET_LABELS,
                "%s 的配置预设档位与 presets.PRESET_LABELS 不一致：\n  少：%s\n  多：%s"
                % (fname, sorted(set(pres.PRESET_LABELS) - set(options or [])),
                   sorted(set(options or []) - set(pres.PRESET_LABELS))),
            )

    def test_node_contract_matches_the_plan(self):
        """CATEGORY / FUNCTION / RETURN_TYPES / RETURN_NAMES 逐个对上计划名册。

        这些字符串是 ComfyUI 的**外部接口**：改一个字母，老 workflow 的连线
        就匹配不上了，而它是运行时才表现出来的（构建期什么都不报）。
        """
        for node_key, fname, cls_name in SPLIT_NODES:
            assigns = _class_assigns(_split_node_path(fname), cls_name)
            self.assertEqual(assigns.get("CATEGORY"), "'千绘/修复'", "%s 的 CATEGORY 不对" % fname)
            self.assertEqual(assigns.get("FUNCTION"), "'process'", "%s 的 FUNCTION 不对" % fname)
            types, names = NODE_RETURN_CONTRACT[cls_name]
            self.assertEqual(
                assigns.get("RETURN_TYPES"),
                "(" + ", ".join("'%s'" % t for t in types) + ("," if len(types) == 1 else "") + ")",
                "%s 的 RETURN_TYPES 与计划名册不符" % fname)
            self.assertEqual(
                assigns.get("RETURN_NAMES"),
                "(" + ", ".join("'%s'" % n for n in names) + ("," if len(names) == 1 else "") + ")",
                "%s 的 RETURN_NAMES 与计划名册不符" % fname)


class TestSplitNodePayloadOrder(unittest.TestCase):
    """payload 的顺序契约：遍历只能走 `iter_frames_in_order`（计划明文）。"""

    def test_no_node_touches_payload_frames_directly(self):
        """★ 4 个节点里**一个** `.frames` 属性访问都不许有。

        `payload.frames` 是裸 list，直接遍历就绕开了「帧号与下标对齐」那道断言；
        而全局降噪节点的视频时序平滑**依赖顺序**：顺序一旦错乱它不报错、
        只是给出看似正常的错误画面。所以这里从根上禁止出现这个属性名。
        """
        hits = {}
        for node_key, fname, cls_name in SPLIT_NODES:
            lines = [n.lineno for n in ast.walk(_parse(_split_node_path(fname)))
                     if isinstance(n, ast.Attribute) and n.attr == "frames"]
            if lines:
                hits[fname] = sorted(lines)
        self.assertEqual(
            hits, {},
            "这些节点直接碰了 `...frames`（必须走 payload.iter_frames_in_order）：%s" % hits,
        )

    def test_every_payload_consumer_uses_the_single_entrypoint(self):
        """B / C / D 都要**真的**调 `iter_frames_in_order` —— 只禁止 `.frames` 不够：

        有人把遍历整个删掉（例如 `for frame in []`）同样不会碰 `.frames`，
        而那个节点会静默什么都不做。
        """
        for node_key, fname, cls_name in SPLIT_NODES:
            if node_key == "detect":
                continue                     # 节点 A 是 payload 的**生产方**，不遍历
            calls = [
                n for n in ast.walk(_parse(_split_node_path(fname)))
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name) and n.func.id == "iter_frames_in_order"
            ]
            self.assertTrue(
                calls,
                "%s 没有调用 iter_frames_in_order —— payload 的遍历入口只有这一个" % fname,
            )


class TestSplitNodesUseSharedCache(unittest.TestCase):
    """T5：4 个拆出来的节点各自都必须用模块级单例，而不是各 new 一个缓存。"""

    def test_each_split_node_init_uses_the_shared_cache(self):
        """逐节点调一次 `_assert_node_uses_shared_cache`（它就是为这一步准备的）。

        各持一份的后果：修复节点伸手卸的是自己那个**空**缓存，检测节点加载的模型
        再没人卸 —— 不报错、不崩，只是显存一直涨。
        """
        for node_key, fname, cls_name in SPLIT_NODES:
            _assert_node_uses_shared_cache(self, _split_node_path(fname), cls_name)


# ==================== SAM 3 分支（T9 的追加需求） ====================
#
# 用户后续明确要求支持 SAM 3 ⇒ 这条盖过了「leaf mixin / model_managers 冻结」的原计划。
# 它有两处**顺序**是正确性的一部分，而顺序错了都不会报错、只会给出错误行为：
#
#   ① `detect_sam_version` 里，sam3 的关键词表必须排在 sam2 的**前面**。
#      SAM 3 沿用 Hiera 骨干，官方权重名是 `sam3_hiera_l.pt` —— 名字里**带 `hiera`**。
#      顺序反了，sam3 权重被判成 sam2 ⇒ 拿 `build_sam2` + 加载一个形状不匹配的
#      checkpoint，报错出现在一个与真正原因毫无关系的地方。
#   ② `_refine_with_sam` 的加载分派里，sam3 分支必须是 sam2 分支的**同一串**
#      （elif），不能另起一个 if 挂在后面 —— 那样 sam3 分支是死代码。
#
# 两条都只做 AST 静态解析（这两个模块都 import comfy 运行时，不能直接载）。

_MODEL_MANAGERS_PY = os.path.join(SPLIT_NODE_DIR, "model_managers.py")
_SAM_MIXIN_PY = os.path.join(SPLIT_NODE_DIR, "mixins", "sam.py")


def _funcdef(path, name):
    for n in ast.walk(_parse(path)):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
            return n
    return None


def _keyword_probe_line(fn, keywords):
    """函数体里「用这张关键词表判定版本」的位置（行号）；找不到返回 None。

    判据是**表里的元素**而不是变量名：变量名可以改，`'sam3'` 这个字面量是行为本身。
    两种写法都认，因为两种都常见：
      * 先赋值 `x_keywords = [...]`，再 `if any(kw in filename for kw in x_keywords)`；
      * 直接把表内联进 `if any(kw in filename for kw in [...])`。

    取 min(赋值行, 判定行)：赋值的顺序就是判定的顺序（表赋值完下一句才是 if），
    内联写法没有赋值行、只剩判定行 —— 两种写法下「谁在前」都得到同一答案。
    """
    want = set(keywords)
    lines, var_names = [], set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Assign) and isinstance(n.value, (ast.List, ast.Tuple)):
            vals = {e.value for e in n.value.elts if isinstance(e, ast.Constant)}
            if vals == want:
                lines.append(n.lineno)
                for t in n.targets:
                    if isinstance(t, ast.Name):
                        var_names.add(t.id)
    for n in ast.walk(fn):
        if not isinstance(n, ast.If):
            continue
        lists = [x for x in ast.walk(n.test) if isinstance(x, (ast.List, ast.Tuple))]
        inline = any(
            {e.value for e in l.elts if isinstance(e, ast.Constant)} == want
            for l in lists
        )
        refs = {x.id for x in ast.walk(n.test) if isinstance(x, ast.Name)} & var_names
        if inline or refs:
            lines.append(n.lineno)
    return min(lines) if lines else None


# ==================== T12：预设按节点分表 ====================
#
# 分表之后「一键套预设」才有意义：每个节点只有一部分控件，整张表写下去会让
# **本节点没有的旋钮**也被改 —— 用户看不见、也不报错。
# 这三条网盯的都是同一类静默失效：
#   ① 某个键**任何**节点都没有 ⇒ 那个旋钮对预设完全免疫（切了没反应）；
#   ② 某个键挂在**不声明它**的节点上 ⇒ 前端 `widgets.find` 找不到、静默跳过；
#   ③ 前端那张**静态**闸门表（JS 里的 NODE_OF）与后端的表漂移 ⇒ 联动整个不注册。

_PRESETS_PY = os.path.join(SPLIT_NODE_DIR, "presets.py")


def _presets():
    return load_module("qh_presets_for_partition", _PRESETS_PY)


class TestPresetTablesArePartitioned(unittest.TestCase):
    """NODE_KEYS 的分表：并集完整、每键都在本节点的控件表里、payload 形状对。"""

    def test_preset_key_partition_covers_every_field_of_key(self):
        """★ NODE_KEYS 的并集必须**恰好**等于 `FIELD_OF` 的键集合（19 个）。

        少了谁，那个旋钮在**任何**节点上都对预设免疫：用户切档位、面板纹丝不动、
        后端也不写 —— 这是本计划猎的那类静默失效里最隐蔽的一种
        （因为它只在「恰好是那个旋钮」时表现出来）。
        多了谁，就是把 `FIELD_OF` 之外的名字挂上了表 —— 那就是拼写错。
        """
        pres = _presets()
        union = set()
        for keys in pres.NODE_KEYS.values():
            union |= set(keys)
        self.assertEqual(
            sorted(set(pres.FIELD_OF) - union), [],
            "这些键**任何**节点都没有 —— 那个旋钮对预设完全免疫（切了没反应）：%s"
            % sorted(set(pres.FIELD_OF) - union),
        )
        self.assertEqual(
            sorted(union - set(pres.FIELD_OF)), [],
            "这些键不在 FIELD_OF 里 —— 拼写错？%s" % sorted(union - set(pres.FIELD_OF)),
        )
        # 预设表里出现过的键也必须全覆盖（FIELD_OF 是它的超集，这里显式再钉一次，
        # 免得将来「FIELD_OF 有、PRESETS 没用」的键把并集撑大而看不出）
        used = set()
        for table in pres.PRESETS.values():
            used |= set(table)
        self.assertEqual(
            sorted(used - union), [],
            "预设表用到、但没有任何节点承接的键：%s" % sorted(used - union),
        )

    def test_every_node_preset_key_is_a_widget_of_that_node(self):
        """★ 每个节点表里的键，必须真的是**该节点自己**的控件。

        不成立时的症状：前端 `node.widgets.find(x => x.name === key)` 找不到，
        `continue` 掉、不报错 —— 预设看着应用了，那个参数其实没写。
        后端也一样：`apply_preset` 会写进 cfg，但那个字段本节点根本读不到。
        """
        pres = _presets()
        for node_key, fname, cls_name in SPLIT_NODES:
            widgets = _node_widgets(_split_node_path(fname), cls_name)
            stray = sorted(set(pres.NODE_KEYS[node_key]) - widgets)
            self.assertEqual(
                stray, [],
                "%s（%s）的预设表里有不属于它的控件：%s\n"
                "  ⇒ 前端找不到 widget 会静默跳过，后端写了也没人读。"
                % (cls_name, fname, stray),
            )

    def test_payload_is_split_per_node(self):
        """`as_payload()` 的形状：`presets[节点短名][档位名] = {控件名: 值}`，
        且每档**只含本节点的键**。这是前端 `applyToNode` 的取数契约。"""
        pres = _presets()
        payload = pres.as_payload()
        self.assertEqual(sorted(payload["nodes"]), sorted(pres.NODE_OF))
        self.assertEqual(sorted(payload["presets"]), sorted(pres.NODE_KEYS))
        self.assertEqual(payload["labels"], pres.PRESET_LABELS)
        for node_key, per_node in payload["presets"].items():
            self.assertEqual(sorted(per_node), sorted(pres.PRESETS),
                             "每个节点都要有**全部**档位（空的档位会让下拉框选了个寂寞）")
            allowed = set(pres.NODE_KEYS[node_key])
            for tier, table in per_node.items():
                self.assertTrue(table, "「%s」在 %s 上是空表" % (tier, node_key))
                self.assertEqual(
                    sorted(set(table) - allowed), [],
                    "「%s」在 %s 上带了不属于它的键：%s"
                    % (tier, node_key, sorted(set(table) - allowed)),
                )
                # 值必须与总表一致（分表只能是**子集**，不许另抄一份值）
                for k, v in table.items():
                    self.assertEqual(v, pres.PRESETS[tier][k],
                                     "「%s」· %s 的值与总表不一致" % (tier, k))

    def test_node_of_matches_the_four_registered_names(self):
        """`NODE_OF` 的键 == 4 个注册名，值 == 4 个节点短名。"""
        pres = _presets()
        self.assertEqual(sorted(pres.NODE_OF),
                         sorted(cls for _, _, cls in SPLIT_NODES))
        self.assertEqual(sorted(pres.NODE_OF.values()),
                         sorted(k for k, _, _ in SPLIT_NODES))

    def test_js_gate_table_matches_the_backend(self):
        """★ 前端那张**静态**闸门表必须与 `presets.NODE_OF` 同构。

        为什么 JS 里必须另有一张：`beforeRegisterNodeDef` 跑在
        `/qianhui_preset_data` 回来**之前**（那一刻表恒为 null），
        所以「要不要给这个节点挂联动」只能用静态表决定 —— 用 payload 决定
        会让整条联动永不注册（预设面板在、选了什么都不动、不报错）。
        代价就是这张表会和后端漂移；本条把它钉住。

        ★ 只做正则文本匹配，**不跑 node**（CI 上不一定有 node，也不该有）。
        """
        with io.open(os.path.join(ROOT, "js", "latent_yolo_presets.js"),
                     encoding="utf-8") as fh:
            src = fh.read()
        block = re.search(r"const NODE_OF = \{(.*?)\};", src, re.S)
        self.assertIsNotNone(block, "js/latent_yolo_presets.js 里找不到静态 NODE_OF 表")
        pairs = re.findall(r"(\w+)\s*:\s*\"(\w+)\"", block.group(1))
        self.assertTrue(pairs, "静态 NODE_OF 表解析成空 —— 形状变了，请同步本条正则")
        self.assertEqual(
            dict(pairs), _presets().NODE_OF,
            "前端静态闸门表与后端 presets.NODE_OF 不一致：\n"
            "  前端 %s\n  后端 %s\n"
            "  ⇒ 症状是「预设面板在，但选了什么都不动」，而且不报错。\n"
            "  （注意这不是「自动同步」的：前端那张必须静态，理由见本用例 docstring）"
            % (dict(pairs), _presets().NODE_OF),
        )


class TestSam3DispatchOrder(unittest.TestCase):
    """两处「顺序即正确性」的 SAM 3 绊线。"""

    def test_sam3_is_detected_before_sam2(self):
        """★ `detect_sam_version` 里 sam3 的关键词表必须排在 sam2 的表之前。

        反例（真会发生）：按「从新到旧」以外的某种直觉把 sam2 的表放前面，
        `sam3_hiera_l.pt` 先命中 `hiera` → 判成 sam2 → 用 SAM2 的加载器装
        SAM3 的权重。现象是「装了 SAM3 就报一条看不懂的错」，不是「识别不出 SAM3」。
        """
        fn = _funcdef(_MODEL_MANAGERS_PY, "detect_sam_version")
        self.assertIsNotNone(fn, "model_managers.py 里没有 detect_sam_version")
        sam3 = _keyword_probe_line(fn, ["sam3", "sam_3", "sam-3"])
        sam2 = _keyword_probe_line(fn, ["sam2", "hiera", "sam_2", "sam-2"])
        self.assertIsNotNone(sam3, "找不到 sam3 的关键词判定")
        self.assertIsNotNone(sam2, "找不到 sam2 的关键词判定")
        self.assertLess(
            sam3, sam2,
            "sam3 的关键词判定在第 %d 行、sam2 的在第 %d 行 —— sam3 必须在前。\n"
            "  反了的话 `sam3_hiera_l.pt` 会先命中 sam2 的 `hiera` 关键词被判成 sam2，\n"
            "  于是拿 SAM2 的加载器去装 SAM3 的权重：报错发生在与真正原因无关的地方。"
            % (sam3, sam2),
        )

    def test_sam3_load_branch_is_before_sam2_in_the_same_chain(self):
        """★ 加载分派里 sam3 必须与 sam2 是**同一串**（sam3 在前），不能另起一个 if。

        另起一个 `if` 挂在 sam2 之后 ⇒ sam3 分支永远不可达（sam2 那条先命中），
        而代码看着「两种都支持」。
        """
        fn = _funcdef(_SAM_MIXIN_PY, "_refine_with_sam")
        self.assertIsNotNone(fn, "mixins/sam.py 里没有 _refine_with_sam")

        outer = None
        for n in ast.walk(fn):
            if (isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
                    and any(isinstance(c, ast.Constant) and c.value == "sam3"
                            for c in n.test.comparators)):
                outer = n
                break
        self.assertIsNotNone(outer, "sam.py 里找不到 `sam_version == \"sam3\"` 的分支")

        inner = [n for n in outer.orelse if isinstance(n, ast.If)]
        self.assertEqual(
            len(inner), 1,
            "sam3 的那条 `if` 之后不是**恰好一条** elif（.orelse 里有 %d 条 if）——\n"
            "  说明它没和 sam2 串在同一条链上。" % len(inner),
        )
        self.assertTrue(
            isinstance(inner[0].test, ast.Compare)
            and any(isinstance(c, ast.Constant) and c.value == "sam2"
                    for c in inner[0].test.comparators),
            "sam3 的 elif 链上的下一条不是 `sam_version == \"sam2\"`（第 %d 行）—— "
            "SAM1 与 SAM2 两条路的先后被改动了。" % inner[0].lineno,
        )

    def test_sam3_adapter_keeps_the_shared_predictor_contract(self):
        """★ `_SAM3BoxPredictor` 必须提供下游循环真正调的那两个方法/关键字。

        下游那段推理循环（SAM1/SAM2 共用）调的是 `set_image(image_np)` 与
        `predict(box=..., multimask_output=True)`。适配层少一个方法 / 少一个关键字，
        就是「选了 SAM3 就崩」—— 而所有测试都绿（本机没装 sam3）。
        """
        cls = _find_class(_parse(_SAM_MIXIN_PY), "_SAM3BoxPredictor")
        self.assertIsNotNone(cls, "sam.py 里没有 _SAM3BoxPredictor 适配层")

        methods = {}
        for st in cls.body:
            if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods[st.name] = st
        for want in ("set_image", "predict"):
            self.assertIn(
                want, methods,
                "_SAM3BoxPredictor 少了 %s() —— 下游循环会 AttributeError" % want,
            )

        for arg in ("box", "multimask_output"):
            self.assertIn(
                arg, {a.arg for a in methods["predict"].args.args},
                "`_SAM3BoxPredictor.predict` 没有形参 %r —— 下游是按关键字调的，"
                "少了它当场 TypeError" % arg,
            )

        # 返回三元组：下游写的是 `masks, scores, _ = ...`。
        returns = [n for n in ast.walk(methods["predict"]) if isinstance(n, ast.Return)]
        self.assertTrue(returns, "_SAM3BoxPredictor.predict 里找不到 return")
        ok = any(isinstance(r.value, ast.Tuple) and len(r.value.elts) == 3 for r in returns)
        self.assertTrue(
            ok,
            "`_SAM3BoxPredictor.predict` 没有一条 return 是三元组 —— "
            "下游写的是 `masks, scores, _ = predictor.predict(...)`。",
        )


if __name__ == "__main__":
    unittest.main()
