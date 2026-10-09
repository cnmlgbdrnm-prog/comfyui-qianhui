# -*- coding: utf-8 -*-
"""节点拆分守卫：跨节点载荷 payload.py 的不变量。

这组用例护的是拆分工程的**地基**：4 个节点之间传的检测数据都装在
`DetectionPayload` 里，而它的两条不变量（帧数对齐、帧号对齐）一旦破了，
全局降噪节点的视频时序平滑会**静默**给出错误画面 —— 不报错、不崩，
只是结果不对。所以这里把不变量钉死在测试里。

`payload.py` 按硬约束**没有包内相对导入**，因此可以和 algorithms.py 一样
用 `_loader.load_module` 按路径直载，不需要桩 ComfyUI 运行时。

本文件只放 payload 用例；cfg 读取守卫属于 T2，不在这里。
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _loader import ROOT, load_module  # noqa: E402

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


if __name__ == "__main__":
    unittest.main()
