"""process() 阶段 3：逐帧区域精修（源 4863-5406）。"""
import comfy.model_management
import torch
import torch.nn.functional as F

from ...algorithms import LatentAlgorithms
from ..context import YoloRunContext
from ...controlnet import ControlNetHelper


def run_repair(node, ctx) -> None:
    """3) 区域精修：逐目标裁剪 → 条件编码 → 局部重绘 → ControlNet → 融合贴回。"""

    # 3) 区域精修
    # 使用预处理降噪后的latent作为基础（如果进行了预处理）
    # 这样后续的区域修复是在清晰的基础上进行的
    ctx.processed_latent = ctx.latent_for_detection

    # V6.11.2: 确保latent在GPU上（SAM后可能导致tensor留在CPU）
    if torch.cuda.is_available() and ctx.processed_latent.device.type != 'cuda':
        print(f"    [设备] latent在{ctx.processed_latent.device}，移动到GPU...")
        ctx.processed_latent = ctx.processed_latent.to('cuda')

    if ctx.detections:
        masks_for_union = []
        for d in ctx.detections:
            pol = node._build_det_policy(d, ctx.pixel_w, ctx.pixel_h, ctx.cfg.smart_strategy, ctx.cfg.expand, ctx.cfg.feather, ctx.cfg.region_denoise, ctx.cfg.crop_pad, ctx.cfg.crop_upscale, ctx.cfg.crop_max_edge, ctx.cfg.crop_steps, float(ctx.CFG if ctx.cfg.crop_cfg <= 0.0 else ctx.cfg.crop_cfg), ctx.cfg.adaptive_dn, ctx.cfg.conf_low_scale, ctx.cfg.conf_high_scale, ctx.class_rules, ctx.cfg.hand_foot_enhance, ctx.cfg.hand_foot_denoise, ctx.cfg.hand_foot_extra_rounds, ctx.cfg.hand_foot_cfg_boost, ctx.cfg.hand_foot_cn_strength, ctx.cfg.hand_foot_denoise_decay, ctx.cfg.hand_foot_denoise_min)
            m_expand, m_feather = pol['mask_expand'], pol['mask_feather']
            masks_for_union.append((d, m_expand, m_feather))

        max_expand = max(x[1] for x in masks_for_union) if masks_for_union else ctx.cfg.expand
        max_feather = max(x[2] for x in masks_for_union) if masks_for_union else ctx.cfg.feather
        ctx.detection_mask = node._create_detection_mask(ctx.detections, ctx.pixel_h, ctx.pixel_w, max_expand, max_feather, seg_feather=ctx.cfg.seg_feather)

        # V6.7: 应用 EMA 注意力机制
        if ctx.cfg.enable_ema_attention:
            attention_map = node._calculate_ema_attention(ctx.detections, ctx.pixel_h, ctx.pixel_w, ctx.cfg.attention_focus)
            # 将注意力图应用到检测遮罩（增强高置信度区域）
            ctx.detection_mask = ctx.detection_mask * attention_map

        # ========== 裁剪精修 V5.1 ==========
        if ctx.cfg.enable_crop:
            # V6.10.5: 修复前彻底清理显存+碎片整理
            if getattr(node, '_low_vram_mode', False):
                print(f"    [低显存模式] 修复前清理显存...")
                # 确保YOLO缓存已清空
                if hasattr(node, 'yolo_cache') and len(node.yolo_cache.models) > 0:
                    while len(node.yolo_cache.models) > 0:
                        node.yolo_cache._unload_lru_model()
                # 确保SAM已卸载
                node._unload_sam_model()
                # 请求ComfyUI释放非必要模型
                comfy.model_management.soft_empty_cache()
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    # V6.10.5: 同步并进行碎片整理
                    torch.cuda.synchronize()
                    gc.collect()
                    torch.cuda.empty_cache()

                    # 检查碎片化程度
                    reserved = torch.cuda.memory_reserved(0) / 1024**2
                    allocated = torch.cuda.memory_allocated(0) / 1024**2
                    free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)
                    frag = (reserved - allocated) / reserved * 100 if reserved > 0 else 0

                    print(f"    [低显存模式] 清理后可用显存: {free_mem/1024**2:.0f}MB (碎片化: {frag:.1f}%)")

                    # 如果碎片化严重，再次整理
                    if frag > 30:
                        print(f"    [低显存模式] 碎片化较高，执行深度整理...")
                        gc.collect()
                        torch.cuda.empty_cache()
                        if hasattr(node, 'yolo_cache'):
                            node.yolo_cache.defragment_vram()

            # V6.11.3修复：低显存清理后显式重载主模型到GPU
            # 原因：soft_empty_cache/gc/empty_cache可能使ComfyUI模型管理认为模型需要重载
            # 若不提前reload，第一个crop的_inpaint_sample会触发5GB PCIe传输，显示为0%GPU占用
            # （与SAM路径的load_model_gpu保持一致，见line 4676）
            try:
                print(f"    [模型] 显存清理后重新确认主模型在GPU...")
                _load_fn = getattr(comfy.model_management, 'load_model_gpu', None) or \
                           (lambda m: comfy.model_management.load_models_gpu([m]))
                _load_fn(ctx.模型)
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated(0) / 1024**2
                    print(f"    [模型] 确认完成，已分配显存: {allocated:.0f}MB")
                    try:
                        model_device = next(ctx.模型.model.parameters()).device
                        print(f"    [模型] 主模型设备: {model_device}")
                    except Exception:
                        pass
            except Exception as _reload_err:
                print(f"    [警告] 主模型确认失败: {_reload_err}（将在采样时重试）")

            cur = ctx.processed_latent.clone()

            # V6.11.2: 确保latent在GPU上（SAM后模型重载可能导致latent留在CPU）
            if torch.cuda.is_available():
                if cur.device.type != 'cuda':
                    print(f"      [设备] latent在{cur.device}，移动到GPU...")
                    cur = cur.to('cuda')
                    print(f"      [设备] latent已移动到{cur.device}")

            # V6.11.3: 精修开始前做一次轻量 CUDA 上下文检查（移除 synchronize）
            # 原代码有2次 synchronize，相当于在整批修复开始前就额外阻塞2次
            if torch.cuda.is_available():
                try:
                    torch.cuda.set_device(0)
                    # 只分配小张量检测上下文状态，不做 synchronize
                    test_tensor = torch.zeros(1, device='cuda')
                    del test_tensor

                except RuntimeError as cuda_err:
                    error_msg = str(cuda_err).lower()
                    if 'illegal memory' in error_msg or 'cuda error' in error_msg:
                        print(f"    [CUDA严重错误] 检测到CUDA上下文损坏: {cuda_err}")
                        print(f"    [CUDA严重错误] 尝试恢复CUDA上下文...")
                        try:
                            import gc
                            gc.collect()
                            torch.cuda.empty_cache()
                            torch.cuda.set_device(0)
                            print(f"    [CUDA] 上下文恢复成功")
                        except Exception as recovery_err:
                            print(f"    [CUDA严重错误] 无法恢复，建议重启ComfyUI: {recovery_err}")
                    else:
                        print(f"    [CUDA警告] 检测到错误: {cuda_err}")
                        import gc
                        gc.collect()
                        torch.cuda.empty_cache()

            for pass_i in range(max(1, ctx.cfg.crop_rounds)):
                # V6.10: 检查中断信号（裁剪轮次循环）
                comfy.model_management.throw_exception_if_processing_interrupted()

                dn = float(ctx.cfg.crop_denoise)
                if pass_i > 0 and ctx.cfg.decay:
                    dn = float(max(0.0, dn * (0.75 ** pass_i)))

                for di, d in enumerate(ctx.detections):
                    # V6.10: 检查中断信号（遍历每个检测目标）
                    comfy.model_management.throw_exception_if_processing_interrupted()

                    # V6.10.5: 低显存模式下，每个目标处理前检查显存和碎片化
                    if getattr(node, '_low_vram_mode', False) and torch.cuda.is_available():
                        reserved = torch.cuda.memory_reserved(0)
                        allocated = torch.cuda.memory_allocated(0)
                        total = torch.cuda.get_device_properties(0).total_memory
                        free_mem = total - reserved
                        free_mb = free_mem / 1024**2
                        frag = (reserved - allocated) / reserved * 100 if reserved > 0 else 0

                        # 如果可用显存低于1GB 或 碎片化超过50%，主动清理
                        if free_mb < 1024 or frag > 50:
                            reason = f"显存不足({free_mb:.0f}MB)" if free_mb < 1024 else f"碎片化过高({frag:.0f}%)"
                            print(f"      [显存警告] 目标{di+1}处理前{reason}，执行整理...")
                            import gc
                            torch.cuda.synchronize()
                            gc.collect()
                            torch.cuda.empty_cache()
                            comfy.model_management.soft_empty_cache()
                            # 再次检查
                            free_mem = total - torch.cuda.memory_reserved(0)
                            reserved_new = torch.cuda.memory_reserved(0)
                            allocated_new = torch.cuda.memory_allocated(0)
                            frag_new = (reserved_new - allocated_new) / reserved_new * 100 if reserved_new > 0 else 0
                            print(f"      [显存警告] 清理后: {free_mem/1024**2:.0f}MB可用, 碎片化{frag_new:.1f}%")

                    bx1, by1, bx2, by2 = d.get("bbox",[0,0,0,0])
                    pol = node._build_det_policy(d, ctx.pixel_w, ctx.pixel_h, ctx.cfg.smart_strategy, ctx.cfg.expand, ctx.cfg.feather, dn, ctx.cfg.crop_pad, ctx.cfg.crop_upscale, ctx.cfg.crop_max_edge, ctx.cfg.crop_steps, float(ctx.CFG if ctx.cfg.crop_cfg <= 0.0 else ctx.cfg.crop_cfg), ctx.cfg.adaptive_dn, ctx.cfg.conf_low_scale, ctx.cfg.conf_high_scale, ctx.class_rules, ctx.cfg.hand_foot_enhance, ctx.cfg.hand_foot_denoise, ctx.cfg.hand_foot_extra_rounds, ctx.cfg.hand_foot_cfg_boost, ctx.cfg.hand_foot_cn_strength, ctx.cfg.hand_foot_denoise_decay, ctx.cfg.hand_foot_denoise_min)
                    m_expand, m_feather = pol['mask_expand'], pol['mask_feather']
                    dn_use = pol['denoise']
                    pad_use = int(pol['crop_pad'])
                    upscale_use = float(pol['crop_upscale'])
                    max_edge_use = int(pol['crop_max_edge'])
                    step_use = int(pol['steps'])
                    cfg_use = float(pol['cfg'])
                    local_rounds = int(pol.get('local_rounds', 1))
                    is_hand_foot = pol.get('is_hand_foot', False)
                    # V6.11 新增：提取手脚优化参数
                    cn_strength_override = pol.get('cn_strength_override', None)
                    denoise_decay = float(pol.get('denoise_decay', 0.75))
                    denoise_min = float(pol.get('denoise_min', 0.0))

                    # 打印手脚增强信息
                    if is_hand_foot and ctx.cfg.hand_foot_enhance:
                        print(f"      [{d.get('class_name','')}] ★手脚增强: denoise={dn_use:.2f}, rounds={local_rounds}, cfg={cfg_use:.1f}, CN强度={cn_strength_override if cn_strength_override else 'default'}")

                    crop_lat, (lx1, ly1, lx2, ly2) = node._crop_latent(cur, bx1, by1, bx2, by2, pad_use, ctx.pixel_w, ctx.pixel_h)
                    if crop_lat is None:
                        continue
                    crop_lat = crop_lat.clone()

                    # ★ 先定义ox_px/oy_px，后续ControlNet和mask都要用
                    ox_px = float(lx1) * 8.0
                    oy_px = float(ly1) * 8.0

                    crop_lh, crop_lw = crop_lat.shape[2], crop_lat.shape[3]
                    crop_ph, crop_pw = crop_lh*8, crop_lw*8

                    target_pw = int(round(crop_pw * float(upscale_use)))
                    target_ph = int(round(crop_ph * float(upscale_use)))
                    max_edge = int(max_edge_use)
                    if max_edge > 0:
                        s = min(1.0, float(max_edge) / float(max(target_pw, target_ph)))
                        target_pw = max(64, int(round(target_pw * s)))
                        target_ph = max(64, int(round(target_ph * s)))
                    target_lw = max(8, int(round(target_pw / 8)))
                    target_lh = max(8, int(round(target_ph / 8)))

                    # V6.0：使用Latent感知插值放大裁剪
                    if ctx.cfg.use_latent_aware and ((target_lh != crop_lh) or (target_lw != crop_lw)):
                        crop_lat_big = LatentAlgorithms.latent_aware_resize(crop_lat, target_lh, target_lw)
                    elif (target_lh != crop_lh) or (target_lw != crop_lw):
                        crop_lat_big = F.interpolate(crop_lat, size=(target_lh, target_lw), mode="bilinear", align_corners=False)
                    else:
                        crop_lat_big = crop_lat

                    # 裁剪conditioning
                    pos_crop, neg_crop = ctx.base_pos, ctx.base_neg

                    if ctx.cfg.crop_use_cn and ctx.has_cn:
                        if ctx.cfg.CN图像.shape[0] > 1 and ctx.i < ctx.cfg.CN图像.shape[0]:
                            ctrl_full = ctx.cfg.CN图像[ctx.i:ctx.i+1]
                        else:
                            ctrl_full = ctx.cfg.CN图像[0:1]
                        cx1p = int(max(0, ox_px)); cy1p = int(max(0, oy_px))
                        cx2p = int(min(ctx.pixel_w, float(lx2)*8.0)); cy2p = int(min(ctx.pixel_h, float(ly2)*8.0))
                        ctrl_crop = ctrl_full[:, cy1p:cy2p, cx1p:cx2p, :]
                        ctrl_crop = ControlNetHelper.prepare_control_image(ctrl_crop, target_lh*8, target_lw*8)
                        # V6.11: 手脚区域使用专用CN强度（0.5-0.8效果更好，避免过度约束）
                        cn_strength_use = cn_strength_override if cn_strength_override is not None else ctx.cfg.CN强度
                        pos_crop, neg_crop = ControlNetHelper.apply_controlnet(
                            ctx.base_pos, ctx.base_neg, ctx.cfg.CN模型, ctrl_crop, cn_strength_use, ctx.cfg.CN开始, ctx.cfg.CN结束
                        )

                    # V6.11.3: 手骨架ControlNet（仅手部区域，自动提取MediaPipe骨架避免多指）
                    if is_hand_foot and ctx.has_hand_skeleton_cn and ctx._decoded_np is not None:
                        try:
                            skeleton_np = node._extract_hand_skeleton(
                                ctx._decoded_np, d.get("bbox", [bx1, by1, bx2, by2]),
                                target_lh * 8, target_lw * 8
                            )
                            if skeleton_np is not None:
                                skeleton_t = torch.from_numpy(skeleton_np).unsqueeze(0)  # [1,H,W,3]
                                pos_crop, neg_crop = ControlNetHelper.apply_controlnet(
                                    pos_crop, neg_crop,
                                    ctx.cfg.hand_skeleton_cn_model, skeleton_t,
                                    ctx.cfg.skeleton_cn_strength, 0.0, ctx.cfg.skeleton_cn_end
                                )
                                print(f"      [手骨架] ✓ 骨架CN注入成功，强度={ctx.cfg.skeleton_cn_strength:.2f} 结束={ctx.cfg.skeleton_cn_end:.1f}")
                            else:
                                print(f"      [手骨架] MediaPipe未检测到手部关键点，跳过骨架CN")
                        except Exception as _sk_err:
                            print(f"      [手骨架] 骨架提取失败（跳过）: {_sk_err}")

                    # ========== V6.0: Inpaint采样（mask外不漂移） ==========
                    step_use = int(step_use)
                    cfg_use = float(cfg_use)
                    current_crop = crop_lat_big
                    original_crop = crop_lat_big.clone()  # 保存原始用于边缘混合

                    # 获取inpaint mask（latent级）- 增加羽化
                    l_expand = max(2, m_expand // 8)
                    l_feather = max(2.0, float(m_feather) / 8.0)  # 增大羽化值

                    inpaint_mask = node._get_crop_inpaint_mask(
                        d, target_lh, target_lw, (ox_px, oy_px),
                        expand_latent=l_expand, feather_latent=l_feather, use_seg=ctx.cfg.use_seg
                    )

                    # V6.7: 对裁剪区域应用EMA注意力
                    if ctx.cfg.enable_ema_attention:
                        # 为单个检测创建注意力权重
                        crop_attention = node._calculate_ema_attention([d], target_lh * 8, target_lw * 8, ctx.cfg.attention_focus)
                        # 下采样到latent尺寸
                        crop_attention_t = crop_attention.unsqueeze(0).unsqueeze(0)
                        crop_attention_lat = F.interpolate(crop_attention_t, size=(target_lh, target_lw), mode="bilinear", align_corners=False)
                        crop_attention_lat = crop_attention_lat.squeeze()
                        # V6.10.2修复：确保crop_attention_lat与inpaint_mask在同一设备上，避免CUDA非法内存访问
                        crop_attention_lat = crop_attention_lat.to(inpaint_mask.device)
                        # 应用注意力权重
                        inpaint_mask = inpaint_mask * crop_attention_lat

                    # 为边缘混合准备更大羽化的mask
                    blend_mask = node._get_crop_inpaint_mask(
                        d, target_lh, target_lw, (ox_px, oy_px),
                        expand_latent=l_expand + 1, feather_latent=l_feather * 1.5, use_seg=ctx.cfg.use_seg
                    )

                    # V6.7: 对blend mask也应用EMA注意力
                    if ctx.cfg.enable_ema_attention:
                        # V6.10.2修复：确保crop_attention_lat与blend_mask在同一设备上
                        crop_attention_lat = crop_attention_lat.to(blend_mask.device)
                        blend_mask = blend_mask * crop_attention_lat

                    blend_mask_4d = blend_mask.unsqueeze(0).unsqueeze(0).expand_as(current_crop).to(current_crop.device)
                    # V6.10性能优化：crop_mask_4d移到循环外，避免重复创建
                    crop_mask_4d = inpaint_mask.unsqueeze(0).unsqueeze(0).expand_as(current_crop).to(current_crop.device)

                    for local_i in range(max(1, local_rounds)):
                        # V6.10: 检查中断信号（本地精修循环）
                        comfy.model_management.throw_exception_if_processing_interrupted()

                        # V6.11: 渐进式降噪（手脚区域使用优化的衰减策略）
                        dn_local = float(dn_use)
                        if local_i > 0 and ctx.cfg.decay:
                            # 使用可配置的衰减系数（手脚区域用denoise_decay，其他用0.75）
                            decay_factor = denoise_decay if is_hand_foot else 0.75
                            dn_local = float(dn_local * (decay_factor ** local_i))
                            # 应用下限（防止降噪过低导致无效果）
                            dn_local = float(max(denoise_min, dn_local))

                        # V6.11.2: 采样前确保所有tensor在GPU上
                        if torch.cuda.is_available():
                            if current_crop.device.type != 'cuda':
                                current_crop = current_crop.to('cuda')
                            if inpaint_mask.device.type != 'cuda':
                                inpaint_mask = inpaint_mask.to('cuda')

                        # V6.10：启用noise_mask精确噪点注入（只在mask区域加噪，mask外完全保持原样）
                        print(f"      [V6.11.3] 采样 pass={pass_i+1} target={di+1}/{len(ctx.detections)} local={local_i+1} steps={step_use} denoise={dn_local:.2f}")
                        sampled_crop = node._inpaint_sample(
                            ctx.ksampler, ctx.模型, ctx.seed + ctx.i + (pass_i*1000) + di + (local_i*100),
                            step_use, cfg_use, ctx.采样器, ctx.调度器,
                            pos_crop, neg_crop, current_crop, inpaint_mask, denoise=float(dn_local)
                        )
                        print(f"      [V6.11.3] 采样完成 pass={pass_i+1} target={di+1}")

                        # V6.10优化：noise_mask已确保mask外不变，仅需轻量融合或直接使用
                        # 后混合（V6.9：新增智能柔和融合）
                        if ctx.cfg.blend_mode == "智能柔和融合":
                            # 智能柔和融合：集成色彩对齐+细节恢复+清晰度增强
                            try:
                                current_crop = LatentAlgorithms.smart_soft_blend(
                                    sampled_crop, current_crop, crop_mask_4d[:, :1, :, :],
                                    color_strength=ctx.cfg.color_strength,
                                    clarity_strength=ctx.cfg.clarity_strength
                                )
                            except Exception as e:
                                print(f"    [警告] 智能柔和融合失败: {e}，回退到简单混合")
                                current_crop = current_crop * (1.0 - crop_mask_4d) + sampled_crop * crop_mask_4d
                        elif ctx.cfg.blend_mode == "Poisson融合" and ctx.cfg.poisson_iters > 0:
                            # Poisson融合：先色差修复，再梯度域融合
                            if ctx.cfg.color_strength > 0:
                                try:
                                    sampled_crop = LatentAlgorithms.frequency_aware_color_match(
                                        sampled_crop, current_crop, crop_mask_4d[:, :1, :, :],
                                        strength=ctx.cfg.color_strength
                                    )
                                except Exception as e:
                                    print(f"    [警告] 色差修复失败: {e}")
                                    pass
                            try:
                                current_crop = LatentAlgorithms.poisson_blend_latent(
                                    sampled_crop, current_crop, crop_mask_4d, iterations=ctx.cfg.poisson_iters
                                )
                            except Exception as e:
                                print(f"    [警告] Poisson融合失败: {e}，回退到简单混合")
                                current_crop = current_crop * (1.0 - crop_mask_4d) + sampled_crop * crop_mask_4d
                        elif ctx.cfg.blend_mode == "MPB增强融合":
                            # MPB融合：先色差修复，再MPB混合
                            if ctx.cfg.color_strength > 0:
                                try:
                                    sampled_crop = LatentAlgorithms.frequency_aware_color_match(
                                        sampled_crop, current_crop, crop_mask_4d[:, :1, :, :],
                                        strength=ctx.cfg.color_strength
                                    )
                                except Exception as e:
                                    print(f"    [警告] 色差修复失败: {e}")
                                    pass
                            try:
                                current_crop = LatentAlgorithms.mpb_blend_latent(
                                    sampled_crop, current_crop, crop_mask_4d, color_preserve=ctx.cfg.mpb_color_preserve
                                )
                            except Exception as e:
                                print(f"    [警告] MPB融合失败: {e}，回退到简单混合")
                                current_crop = current_crop * (1.0 - crop_mask_4d) + sampled_crop * crop_mask_4d
                        else:
                            # 简单混合/频率感知：先色差修复，再简单混合
                            if ctx.cfg.color_strength > 0:
                                try:
                                    sampled_crop = LatentAlgorithms.frequency_aware_color_match(
                                        sampled_crop, current_crop, crop_mask_4d[:, :1, :, :],
                                        strength=ctx.cfg.color_strength
                                    )
                                except Exception as e:
                                    print(f"    [警告] 色差修复失败: {e}")
                                    pass
                            try:
                                current_crop = current_crop * (1.0 - crop_mask_4d) + sampled_crop * crop_mask_4d
                            except Exception as e:
                                print(f"    [警告] 简单混合失败: {e}，保持原crop")
                                pass

                        # 清理临时变量
                        del sampled_crop

                    # V5.1：使用Latent感知插值缩回
                    blended = current_crop
                    if blended.shape[2] != crop_lh or blended.shape[3] != crop_lw:
                        if ctx.cfg.use_latent_aware:
                            blended = LatentAlgorithms.latent_aware_resize(blended, crop_lh, crop_lw)
                        else:
                            blended = F.interpolate(blended, size=(crop_lh, crop_lw), mode="bilinear", align_corners=False)

                    cur[:, :, ly1:ly2, lx1:lx2] = blended

                    del crop_lat, crop_lat_big, current_crop, original_crop, blended

                    # V6.11.3: 合并双重 empty_cache 为一次（原来每个目标后调用2次）
                    # empty_cache 本身是重操作（重新整理显存分配信息）
                    if getattr(node, '_low_vram_mode', False) or ctx.cfg.low_vram:
                        import gc
                        gc.collect()
                        node._gc_if_needed(ctx.cfg.low_vram, ctx.cfg.clear_each)  # 内部已含 empty_cache
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        if getattr(node, '_low_vram_mode', False):
                            free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)
                            print(f"      [目标{di+1}/{len(ctx.detections)}完成] 可用显存: {free_mem/1024**2:.0f}MB")

                # V6.10.5: 每轮结束后清理显存（pass_i循环结束）
                if getattr(node, '_low_vram_mode', False):
                    node._gc_if_needed(True, True)
                    if torch.cuda.is_available():
                        free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)
                        print(f"    [第{pass_i+1}轮完成] 可用显存: {free_mem/1024**2:.0f}MB")

            ctx.processed_latent = cur

        else:
            # 全图mask修复（无裁剪精修时）
            try:
                latent_mask = F.interpolate(ctx.detection_mask.unsqueeze(0).unsqueeze(0),
                                            size=(ctx.latent_h, ctx.latent_w),
                                            mode="bilinear", align_corners=False)
                # 确保是2D
                while latent_mask.dim() > 2:
                    latent_mask = latent_mask.squeeze(0)
                latent_mask = latent_mask.clamp(0.0, 1.0)
            except Exception as e:
                print(f"    [警告] latent_mask处理失败: {e}")
                latent_mask = torch.ones((ctx.latent_h, ctx.latent_w), dtype=torch.float32)

            step_use = int(ctx.步数 if ctx.cfg.region_steps <= 0 else ctx.cfg.region_steps)
            cfg_use = float(ctx.CFG if ctx.cfg.region_cfg <= 0.0 else ctx.cfg.region_cfg)

            cur = ctx.processed_latent

            # V6.10性能优化：安全创建4D mask（避免重复计算）
            try:
                latent_mask_4d = latent_mask.unsqueeze(0).unsqueeze(0)
                if latent_mask_4d.shape[2:] != cur.shape[2:]:
                    latent_mask_4d = F.interpolate(latent_mask_4d, size=cur.shape[2:], mode="bilinear", align_corners=False)
                latent_mask_4d = latent_mask_4d.expand_as(cur).to(cur.device)
            except Exception as e:
                print(f"    [警告] latent_mask_4d创建失败: {e}")
                latent_mask_4d = torch.ones_like(cur)

            for r in range(max(1, ctx.cfg.fix_rounds)):
                # V6.10: 检查中断信号（全图mask修复循环）
                comfy.model_management.throw_exception_if_processing_interrupted()

                dn = float(ctx.cfg.region_denoise)
                if r > 0 and ctx.cfg.decay:
                    dn = float(max(0.0, dn * (0.75 ** r)))

                # V6.10：启用noise_mask精确噪点注入（全图mask修复）
                original = cur.clone()
                sampled = node._inpaint_sample(
                    ctx.ksampler, ctx.模型, ctx.seed + ctx.i + r*1000,
                    step_use, cfg_use, ctx.采样器, ctx.调度器,
                    ctx.pos_full, ctx.neg_full, cur, latent_mask, denoise=float(dn)
                )

                # 后混合（V6.9：新增智能柔和融合）
                if ctx.cfg.blend_mode == "智能柔和融合":
                    # 智能柔和融合：集成色彩对齐+细节恢复+清晰度增强
                    try:
                        cur = LatentAlgorithms.smart_soft_blend(
                            sampled, original, latent_mask_4d[:, :1, :, :],
                            color_strength=ctx.cfg.color_strength,
                            clarity_strength=ctx.cfg.clarity_strength
                        )
                    except Exception as e:
                        print(f"    [警告] 智能柔和融合失败: {e}，回退到简单混合")
                        cur = original * (1.0 - latent_mask_4d) + sampled * latent_mask_4d
                elif ctx.cfg.blend_mode == "Poisson融合" and ctx.cfg.poisson_iters > 0:
                    # Poisson融合：先色差修复，再梯度域融合
                    if ctx.cfg.color_strength > 0:
                        try:
                            sampled = LatentAlgorithms.frequency_aware_color_match(
                                sampled, original, latent_mask_4d[:, :1, :, :], strength=ctx.cfg.color_strength
                            )
                        except Exception:
                            pass
                    cur = LatentAlgorithms.poisson_blend_latent(sampled, original, latent_mask_4d, iterations=ctx.cfg.poisson_iters)
                elif ctx.cfg.blend_mode == "MPB增强融合":
                    # MPB融合：先色差修复，再MPB混合
                    if ctx.cfg.color_strength > 0:
                        try:
                            sampled = LatentAlgorithms.frequency_aware_color_match(
                                sampled, original, latent_mask_4d[:, :1, :, :], strength=ctx.cfg.color_strength
                            )
                        except Exception:
                            pass
                    cur = LatentAlgorithms.mpb_blend_latent(sampled, original, latent_mask_4d, color_preserve=ctx.cfg.mpb_color_preserve)
                else:
                    # 简单混合/频率感知：先色差修复，再简单混合
                    if ctx.cfg.color_strength > 0:
                        try:
                            sampled = LatentAlgorithms.frequency_aware_color_match(
                                sampled, original, latent_mask_4d[:, :1, :, :], strength=ctx.cfg.color_strength
                            )
                        except Exception:
                            pass
                    cur = original * (1.0 - latent_mask_4d) + sampled * latent_mask_4d

                # V6.10.5: 每轮全图修复后清理
                if getattr(node, '_low_vram_mode', False) or ctx.cfg.low_vram:
                    del original, sampled
                    node._gc_if_needed(True, True)

                if ctx.cfg.low_vram:
                    node._gc_if_needed(ctx.cfg.low_vram, ctx.cfg.clear_each)
                    # V6.10: 显式清理CUDA缓存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            ctx.processed_latent = cur

    else:
        # 没检测到目标
        if ctx.cfg.no_detect_mode == "全图重绘":
            # V6.10.1: 添加异常处理和内存管理
            try:
                # 清理CUDA缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                single_latent = {"samples": ctx.processed_latent}
                res = ctx.ksampler.sample(
                    ctx.模型, ctx.seed + ctx.i, ctx.步数, ctx.CFG, ctx.采样器, ctx.调度器,
                    ctx.pos_full, ctx.neg_full, single_latent, denoise=float(ctx.降噪)
                )
                ctx.processed_latent = res[0]["samples"]

            except Exception as e:
                print(f"    [全图重绘] 失败，保持原latent: {e}")
                # 清理CUDA缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
