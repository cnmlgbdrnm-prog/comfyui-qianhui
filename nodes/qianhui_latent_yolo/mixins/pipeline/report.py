"""process() 阶段 5：中断收尾与最终报告（源 5502-5606）。"""
import time

import torch
import torch.nn.functional as F

from ..algorithms import LatentAlgorithms
from ..context import YoloRunContext


def finalize_interrupt(node, ctx) -> None:
    """中断收尾：未处理批次回填原图（源 5502-5528，原 except 子句体）。"""
    # V6.10.7: 捕获中断异常，返回已处理的结果而不是黑图
    ctx.interrupted = True
    print(f"\n{'='*60}")
    print(f"[中断] 用户停止处理")
    print(f"[中断] 已完成: {ctx.processed_count}/{ctx.batch_size} 张")
    print(f"[中断] 返回已处理的结果（未处理的部分保持原图）")
    print(f"{'='*60}")

    # 对于未处理的批次，使用原始输入latent
    if ctx.out_latents is not None:
        for j in range(ctx.processed_count, ctx.batch_size):
            # 如果latent尺寸不同（因为有放大），需要调整原始latent
            orig_sample = ctx.original_samples[j:j+1]
            if ctx.放大倍数 > 1.0:
                if ctx.放大算法 == "latent_aware" or ctx.cfg.use_latent_aware:
                    target_h = ctx.out_latents.shape[2]
                    target_w = ctx.out_latents.shape[3]
                    orig_sample = LatentAlgorithms.latent_aware_resize(orig_sample, target_h, target_w)
                else:
                    orig_sample = node.upscale_latent(orig_sample, ctx.放大算法, ctx.放大倍数)
            ctx.out_latents[j:j+1] = orig_sample.detach().to(ctx.out_device)

            # 为未处理的批次创建空preview和mask
            empty_preview = torch.zeros((1, 64, 64, 3))
            ctx.out_previews.append(empty_preview)
            ctx.out_masks.append(torch.zeros((64, 64)))


def build_report(node, ctx) -> tuple:
    """预览/遮罩汇总、检测信息组装、性能统计、最终 return（源 5530-5606）。"""
    # 合并输出
    # V6.10.7: 处理中断导致out_latents为None的情况
    if ctx.out_latents is None:
        # 如果在第一张图处理前就中断了，返回原始输入（可能需要放大）
        print(f"[中断] 未完成任何处理，返回原始输入")
        if ctx.放大倍数 > 1.0:
            if ctx.放大算法 == "latent_aware" or ctx.cfg.use_latent_aware:
                target_h = int(ctx.samples.shape[2] * ctx.放大倍数)
                target_w = int(ctx.samples.shape[3] * ctx.放大倍数)
                ctx.out_latents = LatentAlgorithms.latent_aware_resize(ctx.original_samples, target_h, target_w)
            else:
                ctx.out_latents = node.upscale_latent(ctx.original_samples, ctx.放大算法, ctx.放大倍数)
        else:
            ctx.out_latents = ctx.original_samples.clone()

        # 创建空的preview和mask
        for _ in range(ctx.batch_size):
            ctx.out_previews.append(torch.zeros((1, 64, 64, 3)))
            ctx.out_masks.append(torch.zeros((64, 64)))

    final_samples = ctx.out_latents.to(ctx.samples.device) if ctx.out_device != ctx.samples.device else ctx.out_latents
    final_preview = torch.cat(ctx.out_previews, dim=0) if ctx.out_previews else torch.zeros((ctx.batch_size, 64, 64, 3))

    max_h = max(m.shape[0] for m in ctx.out_masks) if ctx.out_masks else 64
    max_w = max(m.shape[1] for m in ctx.out_masks) if ctx.out_masks else 64
    masks_resized = []
    for m in ctx.out_masks:
        if m.shape[0] != max_h or m.shape[1] != max_w:
            mm = F.interpolate(m.unsqueeze(0).unsqueeze(0), size=(max_h, max_w), mode="bilinear", align_corners=False).squeeze()
        else:
            mm = m
        masks_resized.append(mm)
    final_masks = torch.stack(masks_resized, dim=0) if masks_resized else torch.zeros((ctx.batch_size, max_h, max_w))

    # V6.10.7: 根据是否中断显示不同信息
    if ctx.interrupted:
        info_lines = [
            f"[已中断] 处理: {ctx.processed_count}/{ctx.batch_size} 张",
            f"工作模式: {ctx.cfg.work_mode}",
            f"融合模式: {ctx.cfg.blend_mode} | Latent感知插值: {'ON' if ctx.cfg.use_latent_aware else 'OFF'}",
            f"放大: {ctx.放大倍数}x | 输出latent: {tuple(final_samples.shape)}",
            f"YOLO: {'ON' if ctx.cfg.enable_yolo else 'OFF'} | 裁剪精修: {'ON' if ctx.cfg.enable_crop else 'OFF'}",
            f"检测总数(使用后): {len(ctx.all_dets)}",
            f"[提示] 未处理的图像已使用原图填充",
        ]
    else:
        info_lines = [
            f"处理完成: {ctx.batch_size} 张",
            f"工作模式: {ctx.cfg.work_mode}",
            f"融合模式: {ctx.cfg.blend_mode} | Latent感知插值: {'ON' if ctx.cfg.use_latent_aware else 'OFF'}",
            f"放大: {ctx.放大倍数}x | 输出latent: {tuple(final_samples.shape)}",
            f"YOLO: {'ON' if ctx.cfg.enable_yolo else 'OFF'} | 裁剪精修: {'ON' if ctx.cfg.enable_crop else 'OFF'}",
            f"检测总数(使用后): {len(ctx.all_dets)}",
        ]
    if ctx.all_dets:
        class_counts = {}
        for d in ctx.all_dets:
            name = d.get("class_name","unknown")
            class_counts[name] = class_counts.get(name, 0) + 1
        for k,v in class_counts.items():
            info_lines.append(f"  - {k}: {v}")

    info = "\n".join(info_lines)

    # V6.6: 性能统计
    if ctx.cfg.enable_profiling and ctx.perf_start_time is not None:
        total_time = time.time() - ctx.perf_start_time
        info_lines.append(f"\n[性能统计] 总耗时: {total_time:.2f}秒 | 平均每帧: {total_time/ctx.batch_size:.2f}秒")
        # 显示显存信息
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            info_lines.append(f"[性能统计] 显存: 已分配 {allocated:.0f}MB | 已保留 {reserved:.0f}MB")
        info = "\n".join(info_lines)

    print(f"\n{'='*60}\n{info}\n{'='*60}\n")
    return ({"samples": final_samples}, final_preview, final_masks, info)
