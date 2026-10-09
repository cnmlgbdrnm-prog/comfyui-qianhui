"""process() 阶段 5：中断收尾与最终报告（源 5502-5606）。"""
import torch

from ...algorithms import LatentAlgorithms
from ..context import YoloRunContext
from .report_bits import build_preview, build_masks, build_info


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
    final_preview = build_preview(ctx)
    final_masks = build_masks(ctx)
    info = build_info(ctx, final_samples)

    print(f"\n{'='*60}\n{info}\n{'='*60}\n")
    return ({"samples": final_samples}, final_preview, final_masks, info)
