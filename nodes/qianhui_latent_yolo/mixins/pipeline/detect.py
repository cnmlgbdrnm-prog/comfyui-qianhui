"""process() 阶段 2：逐帧检测（源 4387-4862，主循环前半）。"""
import comfy.model_management
import numpy as np
import torch
import torch.nn.functional as F

from ...algorithms import LatentAlgorithms
from .sam_refine import refine_with_sam


def run_detection(node, ctx) -> None:
    """预处理 → 全图预降噪 → VAE 解码 → YOLO 检测 → 检测后处理 → SAM 精修。"""
    # V6.10: 检查中断信号，允许用户随时停止
    comfy.model_management.throw_exception_if_processing_interrupted()

    if ctx.pbar:
        ctx.pbar.update(1)
    print(f"\n  [{ctx.i+1}/{ctx.batch_size}] 处理中...")
    single_sample = ctx.samples[ctx.i:ctx.i+1]

    # V6.11.2: 确保samples在GPU上
    if torch.cuda.is_available() and single_sample.device.type != 'cuda':
        print(f"    [设备警告] samples在{single_sample.device}，移动到GPU")
        single_sample = single_sample.to('cuda')
        # 同步更新原samples
        ctx.samples = ctx.samples.to('cuda')

    # 放大（V5.1：支持latent_aware模式）
    if ctx.放大倍数 > 1.0:
        if ctx.放大算法 == "latent_aware" or ctx.cfg.use_latent_aware:
            target_h = int(single_sample.shape[2] * ctx.放大倍数)
            target_w = int(single_sample.shape[3] * ctx.放大倍数)
            single_sample = LatentAlgorithms.latent_aware_resize(single_sample, target_h, target_w)
        else:
            single_sample = node.upscale_latent(single_sample, ctx.放大算法, ctx.放大倍数)

    ctx.latent_h, ctx.latent_w = single_sample.shape[2], single_sample.shape[3]
    ctx.pixel_h, ctx.pixel_w = ctx.latent_h * 8, ctx.latent_w * 8

    if ctx.out_latents is None:
        ctx.out_latents = torch.empty((ctx.batch_size, single_sample.shape[1], ctx.latent_h, ctx.latent_w), device=ctx.out_device, dtype=single_sample.dtype)

    ctx.base_pos = ctx.正面提示
    ctx.base_neg = ctx.负面提示

    ctx.pos_full, ctx.neg_full = ctx.base_pos, ctx.base_neg
    if ctx.has_cn:
        ctx.pos_full, ctx.neg_full = node._maybe_apply_controlnet(
            ctx.base_pos, ctx.base_neg, True, ctx.cfg.CN模型, ctx.cfg.CN图像, ctx.cfg.CN强度, ctx.cfg.CN开始, ctx.cfg.CN结束, ctx.pixel_h, ctx.pixel_w, ctx.i
        )

    # 1) base_latent初始化
    ctx.base_latent = single_sample

    # V6.11.2: 确保base_latent在GPU上
    if torch.cuda.is_available() and ctx.base_latent.device.type != 'cuda':
        print(f"    [设备警告] base_latent在{ctx.base_latent.device}，移动到GPU")
        ctx.base_latent = ctx.base_latent.to('cuda')

    # 1.5) YOLO预处理：先进行全图降噪得到清晰图片
    # 原理：原始latent包含大量噪声，直接解码会得到模糊图片
    # 解决：先做一次全图降噪，得到清晰的基础图像供YOLO检测
    # 优势：YOLO能准确识别目标，然后对这些目标做高质量区域精修
    if ctx.cfg.enable_yolo and ctx.yolo_models_to_use and ctx.降噪 > 0.0:
        # 检测专用降噪上限 0.45：保留手部/脚部结构，防止KSampler"创意重绘"导致目标消失
        # 用户设 0.75 时不影响最终修复质量，只影响检测预处理图像
        detect_denoise = min(float(ctx.降噪), 0.45)
        print(f"    [1/3] 全图降噪 (检测用={detect_denoise}, 修复用={ctx.降噪}) 生成清晰基础图像...")

        # V6.10.1: 添加异常处理和内存管理
        try:
            # 清理CUDA缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 确保base_latent连续存储
            if hasattr(ctx.base_latent, 'contiguous'):
                ctx.base_latent = ctx.base_latent.contiguous()

            pre_latent = {"samples": ctx.base_latent}
            pre_result = ctx.ksampler.sample(
                ctx.模型, ctx.seed + ctx.i, ctx.步数, ctx.CFG, ctx.采样器, ctx.调度器,
                ctx.pos_full, ctx.neg_full, pre_latent, denoise=detect_denoise
            )
            # 用降噪后的清晰latent进行YOLO检测
            ctx.latent_for_detection = pre_result[0]["samples"]
            print(f"    [2/3] YOLO检测中...")

        except Exception as e:
            print(f"    [1/3] 全图降噪失败，使用原始latent: {e}")
            ctx.latent_for_detection = ctx.base_latent
            print(f"    [2/3] YOLO检测中...")
            # 清理CUDA缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        # 如果不启用YOLO或降噪为0，直接使用原始latent
        ctx.latent_for_detection = ctx.base_latent

    # V6.11.3: 不再强制latent移到GPU——ComfyUI的VAE.decode()内部会自行处理设备放置
    # 强制.to('cuda')反而会在SDXL占满显存时造成额外压力，导致模型换出卡顿

    # 2) YOLO检测
    ctx.detection_mask = torch.zeros((ctx.pixel_h, ctx.pixel_w), dtype=torch.float32)
    ctx.detections = []
    ctx.preview_image = None
    ctx._decoded_np = None  # V6.11.3: 原图numpy，供手骨架提取使用

    if ctx.cfg.enable_yolo and ctx.yolo_models_to_use:
        # V6.11.3: VAE解码前显式释放显存，避免SDXL(~5GB)换出时卡顿
        # SDXL采样完成后不再需要占用VRAM，释放后VAE可以直接加载而无需走PCIe换出
        print(f"    [显存] 采样完成，释放显存准备VAE解码...")
        try:
            comfy.model_management.soft_empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            import gc as _gc_vae; _gc_vae.collect()
            if torch.cuda.is_available():
                _free_mb = (torch.cuda.get_device_properties(0).total_memory
                            - torch.cuda.memory_allocated(0)) / 1024**2
                print(f"    [显存] 清理后可用: {_free_mb:.0f}MB")
        except Exception as _vae_mem_err:
            print(f"    [显存] 释放警告（继续）: {_vae_mem_err}")

        # 解码降噪后的latent（而不是原始latent）
        decoded_original = node._decode_latent_to_image(ctx.VAE, ctx.latent_for_detection)[0]

        # V6.11.3: decode完成后立刻移到CPU，释放VRAM给后续SDXL修复阶段
        if hasattr(decoded_original, 'is_cuda') and decoded_original.is_cuda:
            decoded_original = decoded_original.cpu()

        # V6.11.3: 保存原图numpy供手骨架提取（需要时才转换，zero-copy）
        if ctx.has_hand_skeleton_cn:
            ctx._decoded_np = decoded_original.numpy() if isinstance(decoded_original, torch.Tensor) else decoded_original
        else:
            ctx._decoded_np = None

        # 上采样图片以提高 YOLO 检测质量
        decoded_det = node._upscale_image_for_yolo(decoded_original, min_size=1024)

        det_h, det_w = decoded_det.shape[0], decoded_det.shape[1]

        # V6.9：支持多模型分批检测（每个模型独立置信度）
        if len(ctx.yolo_models_to_use) > 1:
            # 多模型模式
            raw = node._detect_with_multi_models(
                ctx.yolo_models_to_use, decoded_det, ctx.yolo_confidences_to_use,  # V6.9: 传入置信度列表
                imgsz=ctx.cfg.yolo_imgsz,
                tile_enabled=ctx.cfg.tile_enabled,
                tile_size=ctx.cfg.tile_size,
                tile_overlap=ctx.cfg.tile_overlap,
                use_seg=ctx.cfg.use_seg,
                target_classes=ctx.target_classes,
                multi_scale=ctx.cfg.multi_scale,
                precision=ctx.cfg.inference_precision
            )
        else:
            # 单模型模式（兼容旧版）
            yolo_model = node.yolo_cache.load_model(ctx.yolo_models_to_use[0], precision=ctx.cfg.inference_precision) if hasattr(node, 'yolo_cache') else node._load_yolo_model(ctx.yolo_models_to_use[0])
            model_confidence = ctx.yolo_confidences_to_use[0] if ctx.yolo_confidences_to_use else ctx.cfg.confidence
            raw = node._detect_with_yolo(
                yolo_model, decoded_det, model_confidence,  # V6.9: 使用对应置信度
                imgsz=ctx.cfg.yolo_imgsz,
                tile_enabled=ctx.cfg.tile_enabled,
                tile_size=ctx.cfg.tile_size,
                tile_overlap=ctx.cfg.tile_overlap,
                use_seg=ctx.cfg.use_seg,
                target_classes=ctx.target_classes,
                multi_scale=ctx.cfg.multi_scale
            )
        # 无结果时降低置信度自动重试一次（解决边界置信度漏检问题）
        if not raw:
            retry_conf = max(0.10, ctx.cfg.confidence * 0.6)
            print(f"    [YOLO] 未检测到目标，降低置信度至 {retry_conf:.2f} 自动重试...")
            try:
                if len(ctx.yolo_models_to_use) > 1:
                    raw = node._detect_with_multi_models(
                        ctx.yolo_models_to_use, decoded_det,
                        [max(0.10, c * 0.6) for c in ctx.yolo_confidences_to_use],
                        imgsz=ctx.cfg.yolo_imgsz, tile_enabled=ctx.cfg.tile_enabled,
                        tile_size=ctx.cfg.tile_size, tile_overlap=ctx.cfg.tile_overlap,
                        use_seg=ctx.cfg.use_seg, target_classes=ctx.target_classes,
                        multi_scale=ctx.cfg.multi_scale, precision=ctx.cfg.inference_precision
                    )
                else:
                    raw = node._detect_with_yolo(
                        yolo_model, decoded_det, retry_conf,
                        imgsz=ctx.cfg.yolo_imgsz, tile_enabled=ctx.cfg.tile_enabled,
                        tile_size=ctx.cfg.tile_size, tile_overlap=ctx.cfg.tile_overlap,
                        use_seg=ctx.cfg.use_seg, target_classes=ctx.target_classes,
                        multi_scale=ctx.cfg.multi_scale
                    )
                if raw:
                    print(f"    [YOLO] 重试成功，检测到 {len(raw)} 个目标（置信度={retry_conf:.2f}）")
                else:
                    print(f"    [YOLO] 重试仍无结果")
            except Exception as retry_err:
                print(f"    [YOLO] 重试失败: {retry_err}")

        if raw:
            dets = raw  # 已经在检测时过滤了target_classes

            # 如果检测图片被上采样，需要缩放坐标回原始尺寸
            if det_h != ctx.pixel_h or det_w != ctx.pixel_w:
                scale_x = ctx.pixel_w / det_w
                scale_y = ctx.pixel_h / det_h
                print(f"    [YOLO] 缩放检测坐标: {det_w}x{det_h} -> {ctx.pixel_w}x{ctx.pixel_h}")

                for det in dets:
                    # 缩放边界框
                    if "bbox" in det:
                        x1, y1, x2, y2 = det["bbox"]
                        det["bbox"] = [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]

                    # 缩放 seg_mask（V6.9优化：更鲁棒的形状处理）
                    if det.get("seg_mask") is not None:
                        try:
                            with torch.no_grad():
                                seg = det["seg_mask"]

                                # 确保seg是numpy数组
                                if not isinstance(seg, np.ndarray):
                                    seg = np.array(seg)

                                # 确保是2D数组 (H, W) - 使用循环确保完全降维
                                while seg.ndim > 2:
                                    if seg.shape[0] == 1:
                                        seg = seg.squeeze(0)  # 移除大小为1的第一维
                                    else:
                                        seg = seg[0]  # 取第一个通道

                                # 转换为tensor并resize
                                seg_t = torch.from_numpy(seg.astype(np.float32))

                                # 确保是4D张量 (1, 1, H, W)
                                if seg_t.ndim == 2:
                                    seg_t = seg_t.unsqueeze(0).unsqueeze(0)
                                elif seg_t.ndim == 3:
                                    seg_t = seg_t.unsqueeze(0)

                                # Resize到目标尺寸
                                seg_t = F.interpolate(seg_t, size=(ctx.pixel_h, ctx.pixel_w), mode="bilinear", align_corners=False)

                                # 降维到2D并转回numpy
                                seg_resized = seg_t.squeeze().cpu().numpy()

                                # 最终确保是2D
                                while seg_resized.ndim > 2:
                                    if seg_resized.shape[0] == 1:
                                        seg_resized = seg_resized.squeeze(0)
                                    else:
                                        seg_resized = seg_resized[0]

                                det["seg_mask"] = seg_resized
                                del seg_t

                        except Exception as e:
                            # seg mask缩放失败，移除mask但保留检测框
                            det["seg_mask"] = None
                            print(f"    [警告] Seg mask缩放失败（已移除mask）: {e}")

            print(f"    [过滤] 原始检测数: {len(dets)} | 几何过滤参数: 最小面积比={ctx.cfg.min_area_ratio}, 最大面积比={ctx.cfg.max_area_ratio}, 最大长宽比={ctx.cfg.max_aspect}")
            dets = node._filter_detections_geometry(dets, ctx.pixel_w, ctx.pixel_h,
                                                   min_area_ratio=ctx.cfg.min_area_ratio, max_area_ratio=ctx.cfg.max_area_ratio,
                                                   min_aspect=ctx.cfg.min_aspect, max_aspect=ctx.cfg.max_aspect)
            print(f"    [过滤] 几何过滤后: {len(dets)}")
            dets = node._sort_detections(dets, sort_mode=ctx.cfg.sort_mode)
            dets = node._limit_detections_per_class(dets, per_class_limit=ctx.cfg.per_class_limit)
            print(f"    [过滤] 类别限制后(limit={ctx.cfg.per_class_limit}): {len(dets)}")
            if ctx.cfg.max_targets and ctx.cfg.max_targets > 0 and len(dets) > ctx.cfg.max_targets:
                dets = dets[:ctx.cfg.max_targets]
            print(f"    [过滤] 最多目标限制后(max={ctx.cfg.max_targets}): {len(dets)}")

            if ctx.cfg.merge_boxes and len(dets) > 1:
                dets = node._merge_overlapping_detections(dets, iou_threshold=ctx.cfg.merge_iou,
                                                          same_class=ctx.cfg.merge_same_class, lr_protect=ctx.cfg.lr_protect, image_width=ctx.pixel_w,
                                                          use_wiou=ctx.cfg.use_wiou, wiou_alpha=ctx.cfg.wiou_alpha)
            print(f"    [过滤] 合并重叠后: {len(dets)}")

            if (ctx.cfg.bbox_expand2_px and ctx.cfg.bbox_expand2_px > 0) or (ctx.cfg.bbox_expand2_ratio and ctx.cfg.bbox_expand2_ratio > 0.0):
                for d in dets:
                    d["bbox"] = node._expand_bbox_xyxy(d.get("bbox",[0,0,0,0]), ctx.pixel_w, ctx.pixel_h,
                                                      extra_px=ctx.cfg.bbox_expand2_px, extra_ratio=ctx.cfg.bbox_expand2_ratio)

            dets = node._sort_detections(dets, sort_mode=ctx.cfg.sort_mode)

            # 二次检测精修（V6.0更新 / V6.5支持多模型）
            if ctx.cfg.refine_det and dets and (len(dets) <= ctx.cfg.refine_max):
                # V6.5: 获取用于二次检测的模型（使用第一个模型或根据检测来源选择）
                refine_model = None
                if len(ctx.yolo_models_to_use) == 1:
                    # 单模型模式：使用已加载的模型
                    refine_model = yolo_model if 'yolo_model' in locals() else None
                else:
                    # 多模型模式：使用第一个模型进行二次检测
                    if hasattr(node, 'yolo_cache'):
                        refine_model = node.yolo_cache.load_model(ctx.yolo_models_to_use[0], precision=ctx.cfg.inference_precision)

                if refine_model is None:
                    print(f"    [警告] 二次检测：无法获取模型，跳过")
                else:
                    refined = []
                    ref_conf = max(0.05, float(ctx.cfg.confidence) * 0.5)
                    # 计算坐标缩放比例（从 pixel 坐标到 det 坐标）
                    scale_to_det_x = det_w / ctx.pixel_w if ctx.pixel_w != det_w else 1.0
                    scale_to_det_y = det_h / ctx.pixel_h if ctx.pixel_h != det_h else 1.0

                    for dd in dets:
                        try:
                            # bbox 坐标是 pixel 尺寸，需要缩放到 det 尺寸用于裁剪
                            bx1, by1, bx2, by2 = [float(v) for v in dd.get("bbox", [0, 0, 0, 0])]
                            bx1_det = bx1 * scale_to_det_x
                            by1_det = by1 * scale_to_det_y
                            bx2_det = bx2 * scale_to_det_x
                            by2_det = by2 * scale_to_det_y

                            cx1 = max(0, int(bx1_det - ctx.cfg.refine_pad)); cy1 = max(0, int(by1_det - ctx.cfg.refine_pad))
                            cx2 = min(det_w, int(bx2_det + ctx.cfg.refine_pad)); cy2 = min(det_h, int(by2_det + ctx.cfg.refine_pad))
                            if (cx2 - cx1) < 8 or (cy2 - cy1) < 8:
                                refined.append(dd)
                                continue

                            crop_img = decoded_det[cy1:cy2, cx1:cx2, :]
                            ref_dets = node._detect_single_enhanced(refine_model, crop_img, ref_conf, ctx.cfg.yolo_imgsz, ctx.cfg.use_seg, [dd.get("class_name","")])
                            if ref_dets:
                                best = max(ref_dets, key=lambda x: float(x.get("confidence", 0.0)))
                                rx1, ry1, rx2, ry2 = [float(v) for v in best.get("bbox", [0, 0, 0, 0])]
                                dd = dd.copy()
                                # 新坐标：从 det 坐标转回 pixel 坐标
                                new_x1 = (cx1 + rx1) / scale_to_det_x
                                new_y1 = (cy1 + ry1) / scale_to_det_y
                                new_x2 = (cx1 + rx2) / scale_to_det_x
                                new_y2 = (cy1 + ry2) / scale_to_det_y
                                dd["bbox"] = [new_x1, new_y1, new_x2, new_y2]
                                # 合并seg mask（已经是 pixel 尺寸）
                                if best.get("seg_mask") is not None:
                                    try:
                                        seg = best["seg_mask"]
                                        # 确保seg是2D数组
                                        if seg.ndim > 2:
                                            seg = seg.squeeze()
                                        # 缩放 seg_mask 到 pixel 尺寸
                                        with torch.no_grad():
                                            seg_t = torch.from_numpy(seg).unsqueeze(0).unsqueeze(0)
                                            seg_t = F.interpolate(seg_t, size=(ctx.pixel_h, ctx.pixel_w), mode="bilinear", align_corners=False)
                                            dd["seg_mask"] = seg_t.squeeze().cpu().numpy()
                                            del seg_t
                                    except Exception as e_seg:
                                        # seg mask处理失败，但保留检测框
                                        dd["seg_mask"] = None
                            else:
                                # V6.11.3修复：二次检测无结果时保留原始检测框，而非静默丢弃
                                # 原bug：此处缺少else分支，导致refine无结果时dd不被append
                                print(f"    [二次检测] {dd.get('class_name','')} 裁剪区域无结果，保留原始框")
                            refined.append(dd)
                        except Exception:
                            refined.append(dd)
                    dets = refined
                    print(f"    [过滤] 二次检测精修后: {len(dets)}")

            dets = node._priority_sort_detections(dets, ctx.cfg.priority_mode, ctx.cfg.custom_priority)
            dets = node._apply_rule_limits(dets, ctx.class_rules, default_per_class_limit=ctx.cfg.per_class_limit, max_targets=ctx.cfg.max_targets)
            print(f"    [过滤] 规则限制后(最终): {len(dets)}")

            dets = refine_with_sam(node, ctx, dets, decoded_det)

            ctx.detections = dets

        if ctx.detections:
            ctx.preview_image = node._draw_detection_preview(decoded_original, ctx.detections)
            print(f"    [3/3] 检测到 {len(ctx.detections)} 个目标，开始区域精修...")
        else:
            ctx.preview_image = decoded_original
            print(f"    [2/3] 未检测到目标")

        # 检测完成后立即清理中间变量，释放显存
        del decoded_det, decoded_original
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    else:
        ctx.preview_image = node._decode_latent_to_image(ctx.VAE, ctx.base_latent)[0]
        ctx.latent_for_detection = ctx.base_latent
