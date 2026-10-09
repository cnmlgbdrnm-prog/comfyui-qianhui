"""从 build_report 搬出来的三段：预览 / 遮罩 / 检测信息。

拆分后「节点 A（检测）产预览 / 节点 C（修复）产遮罩 / 节点 D（全局）产检测
信息」，这三段必须先能单独取用，故从 build_report 里整段搬出。本次是**纯搬家**：
语句逐字保留原样，行为不变。
"""
import time

import torch
import torch.nn.functional as F


def build_preview(ctx) -> torch.Tensor:
    """检测预览汇总（源 build_report L64）。"""
    final_preview = torch.cat(ctx.out_previews, dim=0) if ctx.out_previews else torch.zeros((ctx.batch_size, 64, 64, 3))
    return final_preview


def build_masks(ctx) -> torch.Tensor:
    """检测遮罩汇总：统一尺寸后堆叠（源 build_report L66-75）。"""
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
    return final_masks


def build_info(ctx, final_samples) -> str:
    """检测信息组装 + 性能统计（源 build_report L77-116）。"""
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
    return info
