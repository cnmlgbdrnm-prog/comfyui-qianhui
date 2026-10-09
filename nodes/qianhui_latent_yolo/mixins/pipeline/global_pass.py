"""process() 阶段 4：逐帧修复后全图降噪 + 时序平滑 + 输出累积（源 5407-5500）。"""
import torch


def run_global_pass(node, ctx) -> None:
    """4) 修复后全图降噪 + 视频时序平滑 + 帧历史维护 + 逐帧输出累积。"""
    # ★ V6.1: 修复后全图降噪（可选）
    if ctx.cfg.work_mode == "修复后全图降噪":
        print(f"    [全图降噪] 执行全图降噪，强度={ctx.cfg.final_denoise_strength}")

        # V6.10.1: 优化报错修复 - 添加异常处理和内存管理
        try:
            # 清理CUDA缓存，防止OOM错误
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 确保processed_latent在正确的设备上
            if hasattr(ctx.processed_latent, 'device'):
                ctx.processed_latent = ctx.processed_latent.contiguous()

            single_latent = {"samples": ctx.processed_latent}
            res = ctx.ksampler.sample(
                ctx.模型, ctx.seed + ctx.i + 99999, ctx.步数, ctx.CFG, ctx.采样器, ctx.调度器,
                ctx.pos_full, ctx.neg_full, single_latent, denoise=float(ctx.cfg.final_denoise_strength)
            )
            ctx.processed_latent = res[0]["samples"]
            print(f"    [全图降噪] 完成")

        except RuntimeError as e:
            error_msg = str(e)
            if 'out of memory' in error_msg.lower() or 'cuda' in error_msg.lower():
                print(f"    [全图降噪] CUDA内存错误，跳过全图降噪: {e}")
                # 强制清理缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                import gc
                gc.collect()
                print(f"    [全图降噪] 已跳过，保持修复后的结果")
            else:
                print(f"    [全图降噪] 运行时错误，跳过全图降噪: {e}")

        except Exception as e:
            print(f"    [全图降噪] 发生异常，跳过全图降噪: {e}")
            print(f"    [全图降噪] 已跳过，保持修复后的结果")
            import traceback
            traceback.print_exc()

    # V6.7: 视频模式时序平滑
    if ctx.cfg.video_mode and ctx.frame_history and ctx.cfg.temporal_strength > 0:
        # 使用历史帧加权平均进行平滑
        weights = []
        historical_latents = []

        # 收集历史帧
        for hist_idx in range(min(ctx.cfg.frame_window, len(ctx.frame_history))):
            hist_latent = ctx.frame_history[-(hist_idx + 1)]
            # 距离越近权重越大
            weight = (1.0 / (hist_idx + 2)) ** 1.5
            weights.append(weight)
            historical_latents.append(hist_latent)

        if historical_latents:
            # 归一化权重
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]

            # 计算历史帧的加权平均
            hist_avg = torch.zeros_like(ctx.processed_latent)
            for hist_lat, w in zip(historical_latents, weights):
                hist_avg += hist_lat * w

            # 将当前帧与历史平均混合
            ctx.processed_latent = ctx.processed_latent * (1.0 - ctx.cfg.temporal_strength) + hist_avg * ctx.cfg.temporal_strength

    # 保存到历史（仅保留最近的帧）
    if ctx.cfg.video_mode:
        ctx.frame_history.append(ctx.processed_latent.detach().clone())
        if len(ctx.frame_history) > ctx.cfg.frame_window:
            ctx.frame_history.pop(0)

    ctx.out_latents[ctx.i:ctx.i+1] = ctx.processed_latent.detach().to(ctx.out_device)

    ctx.out_previews.append(ctx.preview_image.unsqueeze(0))

    ctx.out_masks.append(ctx.detection_mask)
    ctx.all_dets.extend(ctx.detections)

    # V6.10.7: 记录已完成数量
    ctx.processed_count = ctx.i + 1

    # 清理批次中的临时变量
    del ctx.preview_image, ctx.detection_mask, ctx.detections, ctx.processed_latent, ctx.base_latent

    if (ctx.i + 1) % 2 == 0:
        node._gc_if_needed(ctx.cfg.low_vram, ctx.cfg.clear_each)
        # 每处理2个批次清理一次 CUDA 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
