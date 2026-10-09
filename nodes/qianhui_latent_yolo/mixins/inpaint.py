"""精修方法 Mixin：裁剪工具、手脚提示词增强与真正的 Inpaint 采样。"""
import gc
import math

import numpy as np
import torch
import torch.nn.functional as F
import comfy.model_management

from ..algorithms import LatentAlgorithms
from ..controlnet import ControlNetHelper


class InpaintMixin:
    def _extract_hand_skeleton(self, image_np, bbox, target_h, target_w):
        """
        V6.11.3: 用 MediaPipe Hands 从原图提取手部21关键点，渲染为 OpenPose 兼容的骨架图。
        用于手部修复时注入 ControlNet，避免生成多余手指。

        Args:
            image_np: float32 numpy array, shape (H, W, 3), 值域 [0, 1], RGB
            bbox: [x1, y1, x2, y2] 手部检测框（像素坐标）
            target_h, target_w: 输出骨架图的尺寸（与修复区域像素尺寸一致）
        Returns:
            numpy array (target_h, target_w, 3) float32 [0,1]，或 None（检测失败/未安装mediapipe）
        """
        try:
            import mediapipe as mp
            import numpy as np
            import cv2
        except ImportError:
            print("      [手骨架] mediapipe 未安装，跳过骨架引导（pip install mediapipe 可启用）")
            return None

        import numpy as np
        import cv2

        h_img, w_img = image_np.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]

        # 扩大裁剪区域（10% padding），让 MediaPipe 更容易检测到完整手部
        pad_x = max(20, int((x2 - x1) * 0.15))
        pad_y = max(20, int((y2 - y1) * 0.15))
        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(w_img, x2 + pad_x)
        cy2 = min(h_img, y2 + pad_y)
        crop_h, crop_w = cy2 - cy1, cx2 - cx1

        if crop_h < 20 or crop_w < 20:
            return None

        hand_region = image_np[cy1:cy2, cx1:cx2]
        # float[0,1] → uint8 RGB（MediaPipe 要求 uint8）
        hand_uint8 = (np.clip(hand_region, 0.0, 1.0) * 255).astype(np.uint8)

        # 用于渲染骨架的黑色画布（与 crop 同尺寸）
        skeleton_canvas = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)

        mp_hands = mp.solutions.hands

        # 手指颜色（OpenPose 风格）
        FINGER_COLORS = {
            0:  (255, 255, 255),   # 手腕
            1:  (255, 153,  51),   # 大拇指 (orange)
            5:  ( 51, 255,  51),   # 食指 (green)
            9:  ( 51, 255, 255),   # 中指 (cyan)
            13: ( 51,  51, 255),   # 无名指 (blue)
            17: (255,  51, 255),   # 小指 (magenta)
        }

        def _finger_color(idx):
            """根据关键点索引返回所属手指颜色"""
            for base in sorted(FINGER_COLORS.keys(), reverse=True):
                if idx >= base:
                    return FINGER_COLORS[base]
            return (200, 200, 200)

        detected = False
        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.25,
            min_tracking_confidence=0.25
        ) as hands:
            results = hands.process(hand_uint8)
            if results.multi_hand_landmarks:
                detected = True
                for hand_landmarks in results.multi_hand_landmarks:
                    lms = hand_landmarks.landmark

                    # 画连接线
                    for (a, b) in mp_hands.HAND_CONNECTIONS:
                        xa = int(lms[a].x * crop_w)
                        ya = int(lms[a].y * crop_h)
                        xb = int(lms[b].x * crop_w)
                        yb = int(lms[b].y * crop_h)
                        color = _finger_color(min(a, b))
                        cv2.line(skeleton_canvas, (xa, ya), (xb, yb), color, 2)

                    # 画关节点
                    for idx, lm in enumerate(lms):
                        x = int(lm.x * crop_w)
                        y = int(lm.y * crop_h)
                        cv2.circle(skeleton_canvas, (x, y), 4, (255, 255, 255), -1)
                        cv2.circle(skeleton_canvas, (x, y), 2, _finger_color(idx), -1)

        if not detected:
            return None

        # 缩放到目标尺寸（修复区域像素大小）
        skeleton_resized = cv2.resize(skeleton_canvas, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        return skeleton_resized.astype(np.float32) / 255.0

    def _decode_for_detection(self, vae, latent_1x, yolo_max_edge: int):
        _, _, lh, lw = latent_1x.shape
        ph, pw = lh*8, lw*8
        max_edge = int(yolo_max_edge or 0)
        if max_edge <= 0 or max(ph, pw) <= max_edge:
            decoded = self._decode_latent_to_image(vae, latent_1x)[0]
            return decoded, 1.0

        scale_det = float(max_edge) / float(max(ph, pw))
        new_ph = max(64, int(round(ph * scale_det)))
        new_pw = max(64, int(round(pw * scale_det)))
        new_lh = max(8, int(round(new_ph / 8)))
        new_lw = max(8, int(round(new_pw / 8)))

        lat_small = F.interpolate(latent_1x, size=(new_lh, new_lw), mode="bilinear", align_corners=False)
        decoded_small = self._decode_latent_to_image(vae, lat_small)[0]
        scale_back = 1.0 / scale_det
        return decoded_small, scale_back

    # -------------------- 裁剪工具 --------------------
    def _crop_latent(self, latent_1x, x1p, y1p, x2p, y2p, pad_px, img_w, img_h):
        _, c, lh, lw = latent_1x.shape
        x1p = max(0.0, float(x1p) - float(pad_px))
        y1p = max(0.0, float(y1p) - float(pad_px))
        x2p = min(float(img_w), float(x2p) + float(pad_px))
        y2p = min(float(img_h), float(y2p) + float(pad_px))

        lx1 = int(max(0, math.floor(x1p / 8.0)))
        ly1 = int(max(0, math.floor(y1p / 8.0)))
        lx2 = int(min(lw, math.ceil(x2p / 8.0)))
        ly2 = int(min(lh, math.ceil(y2p / 8.0)))

        if lx2 <= lx1 or ly2 <= ly1:
            return None, (0,0,0,0)

        crop = latent_1x[:, :, ly1:ly2, lx1:lx2]
        return crop, (lx1, ly1, lx2, ly2)

    def _build_crop_mask(self, crop_lh, crop_lw, bbox_px, crop_origin_px, feather_px=8):
        """V5.1：使用真高斯羽化构建裁剪mask"""
        x1, y1, x2, y2 = [float(v) for v in bbox_px]
        ox, oy = [float(v) for v in crop_origin_px]

        rx1 = max(0.0, x1 - ox)
        ry1 = max(0.0, y1 - oy)
        rx2 = max(0.0, x2 - ox)
        ry2 = max(0.0, y2 - oy)

        lx1 = int(max(0, math.floor(rx1 / 8.0)))
        ly1 = int(max(0, math.floor(ry1 / 8.0)))
        lx2 = int(min(crop_lw, math.ceil(rx2 / 8.0)))
        ly2 = int(min(crop_lh, math.ceil(ry2 / 8.0)))

        m = torch.zeros((crop_lh, crop_lw), dtype=torch.float32)
        if lx2 > lx1 and ly2 > ly1:
            m[ly1:ly2, lx1:lx2] = 1.0

        feather = int(max(0, feather_px))
        if feather > 0 and m.sum() > 0:
            sigma = max(0.5, float(feather) / 3.0)
            m = LatentAlgorithms.gaussian_feather(m, sigma=sigma)

        return m

    def _maybe_apply_controlnet(self, pos, neg, has_cn, CN模型, CN图像, strength, start_p, end_p, target_h, target_w, index_i):
        if not has_cn:
            return pos, neg
        if CN图像 is None or CN模型 is None:
            return pos, neg
        if strength == 0:
            return pos, neg
        if CN图像.shape[0] > 1 and index_i < CN图像.shape[0]:
            ctrl_img = CN图像[index_i:index_i+1]
        else:
            ctrl_img = CN图像[0:1]
        ctrl_img = ControlNetHelper.prepare_control_image(ctrl_img, target_h, target_w)
        return ControlNetHelper.apply_controlnet(pos, neg, CN模型, ctrl_img, strength, start_p, end_p)

    def _gc_if_needed(self, low_vram: bool, clear_each: bool):
        """
        V6.10.5增强：智能显存清理（含碎片整理）
        """
        if clear_each or low_vram or getattr(self, '_low_vram_mode', False):
            try:
                gc.collect()
            except Exception:
                pass

            if torch.cuda.is_available():
                try:
                    # V6.10.5: 同步所有CUDA操作后再清理（减少碎片）
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

                    # 低显存模式下检查碎片化
                    if getattr(self, '_low_vram_mode', False):
                        reserved = torch.cuda.memory_reserved(0) / 1024**2
                        allocated = torch.cuda.memory_allocated(0) / 1024**2
                        if reserved > 0:
                            frag = (reserved - allocated) / reserved * 100
                            # 碎片化超过40%时进行整理
                            if frag > 40:
                                gc.collect()
                                torch.cuda.empty_cache()
                except Exception:
                    pass

            try:
                comfy.model_management.soft_empty_cache()
            except Exception:
                pass

    # ============================================================
    # V6.4 手脚提示词增强
    # ============================================================
    
    def _encode_hand_foot_prompts(self, clip, pos_prompt, neg_prompt):
        """编码手脚区域的增强提示词"""
        from nodes import CLIPTextEncode
        encoder = CLIPTextEncode()
        
        pos_cond = None
        neg_cond = None
        
        if pos_prompt and pos_prompt.strip():
            pos_cond = encoder.encode(clip, pos_prompt.strip())[0]
        
        if neg_prompt and neg_prompt.strip():
            neg_cond = encoder.encode(clip, neg_prompt.strip())[0]
        
        return pos_cond, neg_cond

    def _merge_conditioning(self, base_cond, extra_cond, weight=1.0):
        """合并conditioning（将extra_cond添加到base_cond）"""
        if extra_cond is None:
            return base_cond
        if base_cond is None:
            return extra_cond
        
        # ComfyUI的conditioning是列表，每个元素是(tensor, dict)
        merged = []
        for c in base_cond:
            merged.append(c)
        for c in extra_cond:
            if weight != 1.0:
                tensor, meta = c
                tensor = tensor * weight
                merged.append((tensor, meta))
            else:
                merged.append(c)
        return merged

    def _get_enhanced_conditioning_simple(self, base_pos, base_neg, 
                                          hand_foot_cond_pos, hand_foot_cond_neg):
        """获取增强的conditioning（简化版，不区分手脚）"""
        pos_enhanced = base_pos
        neg_enhanced = base_neg
        
        if hand_foot_cond_pos is not None:
            pos_enhanced = self._merge_conditioning(base_pos, hand_foot_cond_pos, weight=1.3)
        
        if hand_foot_cond_neg is not None:
            neg_enhanced = self._merge_conditioning(base_neg, hand_foot_cond_neg, weight=1.5)
        
        return pos_enhanced, neg_enhanced

    # ============================================================
    # V6.0 核心：真正的Inpaint采样
    # ============================================================
    
    def _prepare_inpaint_latent(self, latent_samples, mask):
        """
        准备inpaint latent（带noise_mask）- V6.10性能优化版

        关键：noise_mask告诉采样器只在mask=1的区域加噪
        mask=0的区域完全保持原样，不会漂移

        noise_mask格式要求: (batch, height, width) - 3D tensor
        """
        try:
            b, c, lh, lw = latent_samples.shape
            device = latent_samples.device

            # mask转tensor（直接在目标设备上）
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask).to(device)
            elif not isinstance(mask, torch.Tensor):
                mask = torch.tensor(mask, device=device)

            # 确保在正确设备上并转换类型
            mask = mask.to(device=device, dtype=torch.float32)

            # 快速标准化mask维度到3D (B, H, W)
            ndim = mask.dim()
            if ndim == 2:  # (H, W) -> (1, H, W)
                mask = mask.unsqueeze(0)
            elif ndim == 4:  # (B, C, H, W) -> (B, H, W)
                mask = mask.squeeze(1) if mask.shape[1] == 1 else mask[:, 0, :, :]
            elif ndim > 4:  # 过度维度，强制压缩
                mask = mask.view(-1, mask.shape[-2], mask.shape[-1])[:1]

            # 获取当前mask尺寸
            mask_h, mask_w = mask.shape[-2:]

            # 缩放到latent尺寸（仅在必要时）
            if mask_h != lh or mask_w != lw:
                # 临时添加通道维度进行插值
                mask = F.interpolate(
                    mask.unsqueeze(1) if mask.dim() == 3 else mask,
                    size=(lh, lw),
                    mode="bilinear",
                    align_corners=False
                ).squeeze(1)

            # batch维度对齐（使用expand避免内存拷贝）
            if mask.shape[0] != b:
                mask = mask[:1].expand(b, lh, lw)

            # 确保值在0-1范围（in-place操作）
            mask.clamp_(0.0, 1.0)

            return {
                "samples": latent_samples,
                "noise_mask": mask
            }
        except Exception as e:
            print(f"    [警告] _prepare_inpaint_latent异常: {e}")
            print(f"    [警告] 回退到普通采样模式")
            return {"samples": latent_samples}

    def _inpaint_sample(self, ksampler, model, seed, steps, cfg, sampler_name, scheduler,
                        positive, negative, latent_samples, mask, denoise):
        """
        V6: 真正的inpaint采样
        通过noise_mask，mask外区域不加噪不去噪，完全保持原样
        V6.10.8: 增强CUDA错误检测和恢复，检测到损坏时直接返回原样
        """
        # V6.10.8: 检查是否已标记CUDA损坏（避免重复尝试）
        if getattr(self, '_cuda_corrupted', False):
            print(f"    [跳过] CUDA已损坏，返回原图")
            return latent_samples

        try:
            # V6.11.3: 仅做轻量的 CUDA 可用性检查（移除 synchronize，避免每次采样都阻塞）
            # synchronize 会强制 CPU 等待 GPU 完成所有队列操作，30 轮修复 = 30 次阻塞
            if torch.cuda.is_available():
                try:
                    # 只分配一个小张量来测试 CUDA 上下文是否正常，不做 synchronize
                    test = torch.zeros(1, device='cuda')
                    del test
                except RuntimeError as cuda_test_err:
                    print(f"    [CUDA损坏] 采样前检测到CUDA错误: {cuda_test_err}")
                    print(f"    [CUDA损坏] 跳过采样，返回原图")
                    self._cuda_corrupted = True
                    return latent_samples

            # V6.11.2: 确保latent在GPU上
            if torch.cuda.is_available() and latent_samples.device.type != 'cuda':
                latent_samples = latent_samples.to('cuda')

            # V6.11.3: 仅在模型不在 GPU 时才调用 load_model_gpu（避免每次都进入 ComfyUI 锁）
            try:
                _needs_load = True
                try:
                    _needs_load = (next(model.model.parameters()).device.type != 'cuda')
                except Exception:
                    pass  # 获取不到设备则保守地触发 load
                if _needs_load:
                    print(f"      [采样] 主模型不在GPU，重新加载...")
                    _load_fn = getattr(comfy.model_management, 'load_model_gpu', None) or \
                               (lambda m: comfy.model_management.load_models_gpu([m]))
                    _load_fn(model)
                    print(f"      [采样] 主模型已加载到GPU")
            except Exception as load_err:
                print(f"    [警告] 加载模型到GPU失败: {load_err}")

            inpaint_latent = self._prepare_inpaint_latent(latent_samples, mask)

            # V6.11.2: 确保inpaint_latent的samples在GPU上
            if torch.cuda.is_available() and inpaint_latent["samples"].device.type != 'cuda':
                inpaint_latent["samples"] = inpaint_latent["samples"].to('cuda')
                if "noise_mask" in inpaint_latent and inpaint_latent["noise_mask"] is not None:
                    inpaint_latent["noise_mask"] = inpaint_latent["noise_mask"].to('cuda')

            print(f"      [采样] latent设备: {inpaint_latent['samples'].device}, 开始ksampler.sample()...")
            result = ksampler.sample(
                model, seed, steps, cfg, sampler_name, scheduler,
                positive, negative, inpaint_latent, denoise=denoise
            )
            print(f"      [采样] ksampler.sample()完成，返回结果")

            return result[0]["samples"]

        except RuntimeError as e:
            error_msg = str(e).lower()
            if 'cuda' in error_msg or 'illegal memory' in error_msg:
                print(f"    [CUDA错误] inpaint采样失败: {e}")
                # 标记CUDA损坏，避免后续继续尝试
                self._cuda_corrupted = True
                print(f"    [CUDA错误] 已标记CUDA损坏，后续采样将跳过")
                print(f"    [CUDA错误] 返回原图（建议重启ComfyUI）")
                return latent_samples
            else:
                print(f"    [警告] inpaint采样失败: {e}")
                print(f"    [警告] 回退到普通采样")
                try:
                    result = ksampler.sample(
                        model, seed, steps, cfg, sampler_name, scheduler,
                        positive, negative, {"samples": latent_samples}, denoise=denoise
                    )
                    return result[0]["samples"]
                except Exception as e2:
                    print(f"    [错误] 普通采样也失败: {e2}")
                    return latent_samples

        except Exception as e:
            print(f"    [警告] inpaint采样失败: {e}")
            print(f"    [警告] 回退到普通采样")
            try:
                result = ksampler.sample(
                    model, seed, steps, cfg, sampler_name, scheduler,
                    positive, negative, {"samples": latent_samples}, denoise=denoise
                )
                return result[0]["samples"]
            except Exception as e2:
                print(f"    [错误] 普通采样也失败: {e2}")
                return latent_samples

    def _get_crop_inpaint_mask(self, det, crop_lh, crop_lw, crop_origin_px, expand_latent=2, feather_latent=1, use_seg=True):
        """
        获取裁剪区域的inpaint mask（latent级）
        返回: 2D tensor (crop_lh, crop_lw)
        """
        import numpy as np
        
        ox_px, oy_px = crop_origin_px
        crop_ph, crop_pw = crop_lh * 8, crop_lw * 8
        
        seg = det.get("seg_mask") if use_seg else None
        mask = None
        
        if seg is not None:
            try:
                # V6.9优化：更安全的seg mask裁剪
                # 确保seg是2D数组
                if not isinstance(seg, np.ndarray):
                    seg = np.array(seg)

                while seg.ndim > 2:
                    seg = seg.squeeze() if seg.shape[0] == 1 else seg[0]

                # 从全图seg裁剪
                full_h, full_w = seg.shape[:2]
                x1 = int(max(0, ox_px))
                y1 = int(max(0, oy_px))
                x2 = int(min(full_w, ox_px + crop_pw))
                y2 = int(min(full_h, oy_px + crop_ph))

                if x2 > x1 and y2 > y1:
                    crop_seg = seg[y1:y2, x1:x2]
                    if crop_seg.size > 0:
                        # 缩放到latent尺寸
                        seg_t = torch.from_numpy(crop_seg.astype(np.float32))
                        if seg_t.ndim == 2:
                            seg_t = seg_t.unsqueeze(0).unsqueeze(0)
                        seg_t = F.interpolate(seg_t, size=(crop_lh, crop_lw), mode="bilinear", align_corners=False)
                        mask = seg_t.squeeze()  # squeeze所有维度到2D
                        # 确保是2D
                        while mask.ndim > 2:
                            mask = mask.squeeze() if mask.shape[0] == 1 else mask[0]
            except Exception as e:
                print(f"    [警告] seg mask裁剪失败（使用bbox）: {e}")
                mask = None
        
        # bbox fallback（seg失败或不存在时）
        if mask is None or mask.sum() < 1.0:
            bx1, by1, bx2, by2 = det["bbox"]
            # 转为crop内的latent坐标
            lx1 = int(max(0, (bx1 - ox_px) / 8))
            ly1 = int(max(0, (by1 - oy_px) / 8))
            lx2 = int(min(crop_lw, (bx2 - ox_px) / 8))
            ly2 = int(min(crop_lh, (by2 - oy_px) / 8))
            
            mask = torch.zeros((crop_lh, crop_lw), dtype=torch.float32)
            if lx2 > lx1 and ly2 > ly1:
                mask[ly1:ly2, lx1:lx2] = 1.0
        
        # 安全检查：如果mask仍然全0，创建中心区域mask
        if mask.sum() < 1.0:
            print(f"    [警告] mask为空，使用中心区域fallback")
            mask = torch.zeros((crop_lh, crop_lw), dtype=torch.float32)
            margin_h = max(1, crop_lh // 4)
            margin_w = max(1, crop_lw // 4)
            mask[margin_h:-margin_h, margin_w:-margin_w] = 1.0
        
        # 确保mask是2D
        while mask.dim() > 2:
            mask = mask.squeeze(0)
        while mask.dim() < 2:
            mask = mask.unsqueeze(0)
        
        # latent级膨胀
        if expand_latent > 0 and mask.sum() > 0:
            try:
                mask = mask.unsqueeze(0).unsqueeze(0)
                k = expand_latent * 2 + 1
                mask = F.max_pool2d(mask, kernel_size=k, stride=1, padding=expand_latent)
                mask = mask.squeeze(0).squeeze(0)
            except Exception:
                pass
        
        # latent级羽化
        if feather_latent > 0 and mask.sum() > 0:
            try:
                mask = LatentAlgorithms.gaussian_feather(mask, sigma=float(feather_latent))
            except Exception:
                pass
        
        # 最终确保2D并clamp
        while mask.dim() > 2:
            mask = mask.squeeze(0)
        mask = mask.clamp(0.0, 1.0)
        
        return mask.cpu().contiguous()

    # ============================================================
    # 核心处理函数 V5.1 - 整合所有优化算法
    # ============================================================
