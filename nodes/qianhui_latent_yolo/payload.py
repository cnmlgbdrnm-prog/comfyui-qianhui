# -*- coding: utf-8 -*-
"""跨节点的「检测数据」载荷。

这里定义的是 4 个节点之间传的东西：检测记录 list[dict] 必须从
YOLO检测 → SAM精修 → 区域修复 → 全局降噪 一路带下去，而 ComfyUI 没有
能装它的标准类型（LATENT/IMAGE/MASK 都装不下 bbox+class+confidence+seg_mask），
所以用一个自定义类型 QIANHUI_DETECT。

★ 不变量：frames[i].i == i，len(frames) == batch_size。
  所有遍历必须走 iter_frames_in_order()，不要直接 for frame in payload.frames ——
  全局降噪节点的视频时序平滑依赖顺序（见计划文件 §静默失效 c）。

★ 本模块不得有任何包内相对导入（即不要写「from」＋点号开头的模块名）。
  这不是风格偏好：测试要像 algorithms.py 那样用 ``_loader.load_module`` 按路径
  直载本文件，不需要桩出整个 ComfyUI 运行时。加一行相对导入，那一整套守卫
  用例就全部装载失败，而失败原因看起来会像「测试坏了」而不是「引入了耦合」。
  允许的顶层导入：``dataclasses`` / ``typing`` / ``collections.abc``。
"""
from dataclasses import dataclass, field
from typing import Any

TYPE_NAME = "QIANHUI_DETECT"


@dataclass(slots=True)
class DetectionFrame:
    """单帧的检测结果与定位信息。"""

    i: int                      # 帧号 —— 所有 seed 偏移的唯一来源
    processed: bool             # False = 中断补帧，下游必须跳过全部分支
    detections: list            # 检测记录 list[dict]，见下方键表
    pixel_h: int
    pixel_w: int
    latent_h: int
    latent_w: int
    decoded: Any = None         # decoded_original: CPU tensor [H,W,3] float32 ∈ [0,1]
    det_image_size: Any = None  # (det_h, det_w)，_upscale_image_for_yolo 的输出尺寸


# detections 每条记录的键（沿用单体，不要新增也不要删）：
#   class_name: str, class_id: int, confidence: float,
#   bbox: list[float],              # 像素空间
#   seg_mask: Any | None,           # np.ndarray(H,W) float32，像素空间
#   seg_mask_shape: tuple | None,
#   sam_score: float,               # 仅 SAM 成功时存在
#   source_model: str               # 仅多模型路径存在


@dataclass(slots=True)
class DetectionPayload:
    """4 个节点之间传递的整批检测数据。"""

    frames: list                # 索引对齐：frames[i].i == i
    batch_size: int
    stage_flags: dict = field(default_factory=dict)   # 各阶段实际用到的值，供「检测信息」复述
    perf: dict = field(default_factory=dict)          # {"detect":秒, "sam":秒, "repair":秒, "global":秒}
    interrupted: bool = False


def iter_frames_in_order(payload):
    """按帧号升序产出帧，并断言索引对齐。

    这是**唯一**允许的遍历入口：全局降噪节点的视频时序平滑依赖顺序，
    而 payload 由检测节点按序生成、中途节点不得改序。顺序一旦错乱，
    时序平滑会静默给出错误结果，所以这里当场断言而不是靠约定。

    用 ``raise`` 而不是 ``assert``：``python -O`` 会把 assert 整条删掉，
    而这条不变量的失败是**静默的坏结果**（时序平滑算出看似正常的画面），
    不是崩溃，所以它必须活过 -O。

    这里只校验、不拷贝：``yield`` 的就是 ``payload.frames`` 里的原对象，
    检测记录也仍是各帧那个同一个 list。跨节点传递靠引用，
    拷贝一份（哪怕是浅拷贝）会让下游的追加改动在上游看不见。
    """
    frames = payload.frames

    # 断言 1：帧数与声明的 batch_size 一致
    if len(frames) != payload.batch_size:
        raise ValueError(
            f"帧数与 batch_size 不一致：len(frames)={len(frames)}, "
            f"batch_size={payload.batch_size}"
        )

    # 断言 2：每帧的帧号与它在列表里的下标一致
    for idx, frame in enumerate(frames):
        if frame.i != idx:
            raise ValueError(
                f"帧号错位：下标 {idx} 处的帧 frame.i={frame.i}，"
                f"必须满足 frames[i].i == i"
            )

    yield from frames
