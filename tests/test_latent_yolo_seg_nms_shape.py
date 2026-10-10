# -*- coding: utf-8 -*-
"""NMS 合并 seg 遮罩的**形状安全**回归测试（V6.11.4）。

背景（真事故，2026-10-10）：
    1024×1280 竖图 + seg 模型 + 多尺度检测时，翻转TTA 分支的 seg 遮罩停在
    letterbox 分辨率，与已规范化到原图尺寸的遮罩混进同一轮 NMS：

        ValueError: operands could not be broadcast together with shapes
                    (1280,1024) (768,640)

    整个检测节点抛异常、整条链挂掉。源头修在
    `mixins/detection.py` 的翻转后重采样；这里护的是 `mixins/geometry.py`
    的兜底——任何未来的形状不一致都必须**退化成「不合并」而不是崩**。

★ 为什么只测 geometry 这一半：
    `detection.py` 那段重采样埋在 `_detect_with_yolo` 里面，要真 YOLO 权重
    才跑得到；端到端那套（`test_latent_split_e2e.py`）按设计把这个函数打了桩，
    所以那一半只能靠真机复现工作流压（`_wf_gen/repro_seg.json`）。这里护住
    能纯函数化护住的那一半：**形状不对绝不能抛异常**。
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _loader import load_module, pkgs_dir  # noqa: E402

GEO = load_module(
    "qh_geometry", os.path.join(pkgs_dir("qianhui_latent_yolo", "mixins"), "geometry.py")
)


def _has_cv2():
    try:
        import cv2  # noqa: F401
        return True
    except Exception:
        return False


class _Node(GEO.GeometryMixin):
    """GeometryMixin 是纯 mixin，没有可实例化的基类，自己搭一个空的。"""


def _det(conf, bbox, seg):
    return {"confidence": conf, "class_id": 0, "bbox": list(bbox),
            "class_name": "foot", "seg_mask": seg}


#: 事故现场的两种形状：原图空间 (h=1280, w=1024) vs letterbox 空间 (768, 640)
FULL = (1280, 1024)
LB = (768, 640)


class TestSegMaskShapeMismatch(unittest.TestCase):
    def setUp(self):
        self.node = _Node()

    def test_mismatched_shapes_do_not_raise(self):
        """★ 这是这条测试存在的唯一理由：形状不一致时**不许抛**。"""
        a = np.zeros(FULL, dtype=np.float32)
        a[100:200, 100:200] = 1.0
        b = np.zeros(LB, dtype=np.float32)
        b[10:40, 10:40] = 1.0
        dets = [_det(0.9, (0, 0, 100, 100), a), _det(0.5, (0, 0, 90, 90), b)]

        keep = self.node._nms_detections_enhanced(dets, iou_thresh=0.5)

        self.assertEqual(len(keep), 1, "两条重叠同类框应合并成一条")
        self.assertEqual(keep[0]["seg_mask"].shape, FULL,
                         "合并结果必须留在**保留框**的形状上")

    def test_mismatched_shapes_align_and_merge_with_cv2(self):
        """有 cv2 时：小遮罩被重采样到保留框尺寸后逐点 max。"""
        if not _has_cv2():
            self.skipTest("测试环境没有 cv2 —— 走的是「退化成不合并」那条分支")
        import cv2

        a = np.zeros(FULL, dtype=np.float32)
        a[100:200, 100:200] = 1.0
        b = np.zeros(LB, dtype=np.float32)
        b[10:40, 10:40] = 1.0
        dets = [_det(0.9, (0, 0, 100, 100), a), _det(0.5, (0, 0, 90, 90), b)]

        keep = self.node._nms_detections_enhanced(dets, iou_thresh=0.5)
        expect = np.maximum(a, cv2.resize(b, (FULL[1], FULL[0]),
                                          interpolation=cv2.INTER_LINEAR))
        self.assertTrue(np.array_equal(keep[0]["seg_mask"], expect))
        # 反向绊线：重采样后的 b 必须真的往结果里贡献了非零像素，
        # 否则「合并」是假动作（比如 b 被 resize 成全零），本用例就白跑了。
        self.assertGreater(float(np.count_nonzero(expect - a)), 0.0,
                           "对齐后的 b 一点非零都没贡献 ⇒ 这个用例没压到合并路径")

    def test_mismatched_shapes_degrade_without_cv2(self):
        """没有 cv2 时：退化成「不合并」，保留框自己的遮罩原样留下。"""
        if _has_cv2():
            self.skipTest("测试环境有 cv2 —— 走的是「对齐后合并」那条分支")

        a = np.zeros(FULL, dtype=np.float32)
        a[100:200, 100:200] = 1.0
        dets = [_det(0.9, (0, 0, 100, 100), a),
                _det(0.5, (0, 0, 90, 90), np.zeros(LB, dtype=np.float32))]

        keep = self.node._nms_detections_enhanced(dets, iou_thresh=0.5)
        self.assertIs(keep[0]["seg_mask"], a, "应当原样保留，而不是换成一个空遮罩")

    def test_matched_shapes_merge_bit_exact(self):
        """形状一致时走原语句：逐点 max，一个 bit 都不许变。"""
        a = np.zeros(FULL, dtype=np.float32)
        a[100:200, 100:200] = 1.0
        b = np.zeros(FULL, dtype=np.float32)
        b[300:400, 300:400] = 0.5
        dets = [_det(0.9, (0, 0, 100, 100), a), _det(0.5, (0, 0, 90, 90), b)]

        keep = self.node._nms_detections_enhanced(dets, iou_thresh=0.5)
        self.assertTrue(np.array_equal(keep[0]["seg_mask"], np.maximum(a, b)))

    def test_none_seg_mask_is_untouched(self):
        """任一侧没有 seg_mask：不合并、不报错（原行为）。"""
        a = np.zeros(FULL, dtype=np.float32)
        dets = [_det(0.9, (0, 0, 100, 100), a), _det(0.5, (0, 0, 90, 90), None)]

        keep = self.node._nms_detections_enhanced(dets, iou_thresh=0.5)
        self.assertTrue(np.array_equal(keep[0]["seg_mask"], a))

    def test_empty_input(self):
        self.assertEqual(self.node._nms_detections_enhanced([], iou_thresh=0.5), [])


if __name__ == "__main__":
    unittest.main()
