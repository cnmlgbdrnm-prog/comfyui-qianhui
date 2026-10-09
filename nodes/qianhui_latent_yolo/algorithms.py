"""Latent 空间核心算法（插值/羽化/色差修复/融合）。本模块不依赖 ComfyUI。"""
import math

import numpy as np
import torch
import torch.nn.functional as F


# ============================================================
# 核心算法优化模块 - V5.1新增
# ============================================================

class LatentAlgorithms:
    """Latent空间高级算法集合 - 极致优化"""
    
    # -------------------- 1. Latent感知插值 --------------------
    @staticmethod
    def latent_aware_resize(latent: torch.Tensor, target_h: int, target_w: int, 
                            mode: str = "adaptive") -> torch.Tensor:
        """
        Latent感知插值 - 保留语义结构
        
        原理：latent的4个通道有不同的语义含义，分别处理后合并
        - 通道0-1：主要结构信息，用bicubic保持锐度
        - 通道2-3：细节/纹理信息，用area防止摩尔纹（缩小时）
        """
        if latent.shape[2] == target_h and latent.shape[3] == target_w:
            return latent
        
        if mode == "simple":
            return F.interpolate(latent, size=(target_h, target_w), 
                               mode="bilinear", align_corners=False)
        
        b, c, h, w = latent.shape
        
        # 结构通道：bicubic（更锐利）
        struct_channels = latent[:, :2, :, :]
        struct_resized = F.interpolate(struct_channels, size=(target_h, target_w),
                                       mode="bicubic", align_corners=False)
        
        # 细节通道：根据缩放方向选择
        detail_channels = latent[:, 2:, :, :]
        if target_h * target_w < h * w:
            # 缩小：用area避免锯齿
            detail_resized = F.interpolate(detail_channels, size=(target_h, target_w),
                                          mode="area")
        else:
            # 放大：用bilinear+轻微锐化
            detail_resized = F.interpolate(detail_channels, size=(target_h, target_w),
                                          mode="bilinear", align_corners=False)
            # 轻微锐化
            if target_h >= 16 and target_w >= 16:
                detail_resized = LatentAlgorithms._mild_sharpen(detail_resized, strength=0.12)
        
        return torch.cat([struct_resized, detail_resized], dim=1)
    
    @staticmethod
    def _mild_sharpen(x: torch.Tensor, strength: float = 0.12) -> torch.Tensor:
        """轻微锐化 - 拉普拉斯增强
        V6.11.3: 向量化实现，用 groups conv2d 一次处理所有通道，消除 per-channel 循环。
        """
        C = x.shape[1]
        # groups=C：每通道独立卷积，等价于原来的 for 循环，但只调用一次 GPU kernel
        kernel = torch.tensor([
            [0., -1., 0.],
            [-1., 4., -1.],
            [0., -1., 0.]
        ], dtype=x.dtype, device=x.device).view(1, 1, 3, 3).expand(C, 1, 3, 3).contiguous()
        padded = F.pad(x, (1, 1, 1, 1), mode='replicate')
        edge = F.conv2d(padded, kernel, groups=C)
        return x + edge * strength
    
    # -------------------- 2. 真高斯羽化 --------------------
    @staticmethod
    def gaussian_feather(mask: torch.Tensor, sigma: float, 
                        truncate: float = 4.0) -> torch.Tensor:
        """
        真正的高斯羽化（可分离卷积实现，效率高）
        """
        if sigma <= 0:
            return mask
        
        squeeze_back = 0
        original_shape = mask.shape
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
            squeeze_back = 2
        elif mask.dim() == 3:
            mask = mask.unsqueeze(0)
            squeeze_back = 1
        
        radius = int(truncate * sigma + 0.5)
        kernel_size = 2 * radius + 1
        
        x = torch.arange(kernel_size, dtype=mask.dtype, device=mask.device) - radius
        gaussian_1d = torch.exp(-x**2 / (2 * sigma**2))
        gaussian_1d = gaussian_1d / gaussian_1d.sum()
        
        kernel_h = gaussian_1d.view(1, 1, 1, -1)
        kernel_v = gaussian_1d.view(1, 1, -1, 1)
        
        # 水平
        padded = F.pad(mask, (radius, radius, 0, 0), mode='replicate')
        blurred = F.conv2d(padded, kernel_h)
        
        # 垂直
        padded = F.pad(blurred, (0, 0, radius, radius), mode='replicate')
        blurred = F.conv2d(padded, kernel_v)
        
        result = torch.clamp(blurred, 0.0, 1.0)
        
        if squeeze_back == 2:
            result = result.squeeze(0).squeeze(0)
        elif squeeze_back == 1:
            result = result.squeeze(0)
        
        return result
    
    # -------------------- 3. 频率感知色差修复 --------------------
    @staticmethod
    def frequency_aware_color_match(src: torch.Tensor, ref: torch.Tensor, 
                                    mask: torch.Tensor, strength: float = 0.6) -> torch.Tensor:
        """
        频率感知色差修复
        
        原理：
        1. 分离低频（整体色调）和高频（细节纹理）
        2. 只对低频做统计匹配，保留src的高频细节
        3. 在mask边缘区域渐进混合
        """
        if strength <= 0:
            return src

        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        if mask.shape[1] != src.shape[1]:
            mask = mask.expand(-1, src.shape[1], -1, -1)

        # 确保所有张量在同一设备上（修复CUDA内存访问错误）
        mask = mask.to(src.device)
        ref = ref.to(src.device)

        # 边缘权重：mask*(1-mask) 在羽化边缘最大
        edge_weight = (mask * (1.0 - mask) * 4.0).clamp(0.0, 1.0)
        
        # 低频分离
        kernel_size = min(31, min(src.shape[2], src.shape[3]) // 4)
        if kernel_size < 3:
            kernel_size = 3
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        padding = kernel_size // 2
        
        src_low = F.avg_pool2d(F.pad(src, (padding,)*4, mode='replicate'), 
                               kernel_size, stride=1)
        ref_low = F.avg_pool2d(F.pad(ref, (padding,)*4, mode='replicate'), 
                               kernel_size, stride=1)
        
        # 高频
        src_high = src - src_low
        
        # 边缘区域统计匹配
        eps = 1e-6
        edge_sum = edge_weight.sum(dim=(2, 3), keepdim=True).clamp(min=eps)
        
        src_low_mean = (src_low * edge_weight).sum(dim=(2, 3), keepdim=True) / edge_sum
        ref_low_mean = (ref_low * edge_weight).sum(dim=(2, 3), keepdim=True) / edge_sum
        
        src_low_var = ((src_low - src_low_mean)**2 * edge_weight).sum(dim=(2, 3), keepdim=True) / edge_sum
        ref_low_var = ((ref_low - ref_low_mean)**2 * edge_weight).sum(dim=(2, 3), keepdim=True) / edge_sum
        
        src_low_std = torch.sqrt(src_low_var + eps)
        ref_low_std = torch.sqrt(ref_low_var + eps)
        
        scale = (ref_low_std / src_low_std.clamp(min=eps)).clamp(0.5, 2.0)  # 限制缩放范围
        adjusted_low = (src_low - src_low_mean) * scale + ref_low_mean
        
        adjusted = adjusted_low + src_high
        
        return src * (1.0 - strength) + adjusted * strength
    
    # -------------------- 3.5 简单均值色差修复（更强） --------------------
    @staticmethod
    def simple_color_match(src: torch.Tensor, ref: torch.Tensor, 
                           mask: torch.Tensor, strength: float = 0.8) -> torch.Tensor:
        """
        简单的均值+方差匹配 - 更直接有效
        
        在整个mask区域内匹配src和ref的均值和方差
        """
        if strength <= 0:
            return src
        
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        if mask.shape[1] != src.shape[1]:
            mask = mask.expand(-1, src.shape[1], -1, -1)

        # 确保所有张量在同一设备上（修复CUDA内存访问错误）
        mask = mask.to(src.device)
        ref = ref.to(src.device)

        eps = 1e-6
        mask_sum = mask.sum(dim=(2, 3), keepdim=True).clamp(min=eps)
        
        # 在mask区域计算均值
        src_mean = (src * mask).sum(dim=(2, 3), keepdim=True) / mask_sum
        ref_mean = (ref * mask).sum(dim=(2, 3), keepdim=True) / mask_sum
        
        # 在mask区域计算标准差
        src_var = ((src - src_mean)**2 * mask).sum(dim=(2, 3), keepdim=True) / mask_sum
        ref_var = ((ref - ref_mean)**2 * mask).sum(dim=(2, 3), keepdim=True) / mask_sum
        
        src_std = torch.sqrt(src_var + eps)
        ref_std = torch.sqrt(ref_var + eps)
        
        # 标准化src，再按ref的分布缩放（限制缩放范围防止色差过矫）
        scale = (ref_std / src_std.clamp(min=eps)).clamp(0.5, 2.0)
        adjusted = (src - src_mean) * scale + ref_mean
        
        # 只在mask区域应用，mask外保持原样
        result = src * (1.0 - mask * strength) + adjusted * (mask * strength)
        
        return result

    # -------------------- 3.6 智能柔和融合（V6.9新增）--------------------
    @staticmethod
    def smart_soft_blend(src: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor,
                        color_strength: float = 0.85, clarity_strength: float = 0.3) -> torch.Tensor:
        """
        智能柔和融合 - 专门解决色差和朦胧感问题

        特点：
        1. 多层次渐进羽化：创建3层不同强度的羽化mask
        2. 色彩空间对齐：匹配均值和方差，保持色彩一致性
        3. 细节恢复：提取并强化高频细节
        4. 清晰度增强：局部对比度增强，减少朦胧感

        Args:
            src: 修复后的图像
            ref: 原始参考图像
            mask: 修复区域mask
            color_strength: 色彩对齐强度 (0-1)
            clarity_strength: 清晰度恢复强度 (0-1)
        """
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        if mask.shape[1] != src.shape[1]:
            mask = mask.expand(-1, src.shape[1], -1, -1)

        # 确保所有张量在同一设备上（修复CUDA内存访问错误）
        mask = mask.to(src.device)
        ref = ref.to(src.device)
        eps = 1e-6

        # 步骤1: 多层次渐进羽化
        # 创建3层mask：核心区域、过渡区域、边缘区域
        mask_core = mask.clamp(0.7, 1.0)  # 核心：保留修复结果
        mask_edge = (mask * (1.0 - mask) * 4.0).clamp(0.0, 1.0)  # 边缘：重点融合
        mask_transition = (mask * 0.5).clamp(0.0, 1.0)  # 过渡：平滑过渡

        # 步骤2: 色彩空间对齐（在整个mask区域）
        if color_strength > 0:
            mask_sum = mask.sum(dim=(2, 3), keepdim=True).clamp(min=eps)

            # 计算均值
            src_mean = (src * mask).sum(dim=(2, 3), keepdim=True) / mask_sum
            ref_mean = (ref * mask).sum(dim=(2, 3), keepdim=True) / mask_sum

            # 计算标准差
            src_var = ((src - src_mean)**2 * mask).sum(dim=(2, 3), keepdim=True) / mask_sum
            ref_var = ((ref - ref_mean)**2 * mask).sum(dim=(2, 3), keepdim=True) / mask_sum

            src_std = torch.sqrt(src_var + eps)
            ref_std = torch.sqrt(ref_var + eps)

            # 色彩对齐（限制缩放范围，防止极端色差矫正导致溢出）
            scale = (ref_std / src_std.clamp(min=eps)).clamp(0.5, 2.0)
            src_aligned = (src - src_mean) * scale + ref_mean

            # 应用色彩对齐（核心区域较弱，边缘区域较强）
            color_weight = mask_edge * color_strength + mask_transition * (color_strength * 0.5)
            src = src * (1.0 - color_weight) + src_aligned * color_weight

        # 步骤3: 细节恢复（增强修复区域的高频细节，不引入旧图高频避免雾化）
        if clarity_strength > 0:
            # 低频分离（使用小卷积核）
            kernel_size = 7
            padding = kernel_size // 2

            src_low = F.avg_pool2d(F.pad(src, (padding,)*4, mode='replicate'),
                                  kernel_size, stride=1)

            # 只提取新图高频，在mask核心区放大高频细节（不混入旧图高频）
            src_high = src - src_low
            # 在mask区域内增强高频：让修复结果更清晰，而不是把旧内容混入
            detail_boost = mask * clarity_strength * 0.5
            src = src_low + src_high * (1.0 + detail_boost)

        # 步骤4: 局部对比度增强（减少朦胧感）
        if clarity_strength > 0:
            # 使用更大的卷积核计算局部平均
            kernel_size_large = 15
            padding_large = kernel_size_large // 2

            src_local_mean = F.avg_pool2d(F.pad(src, (padding_large,)*4, mode='replicate'),
                                         kernel_size_large, stride=1)

            # 计算局部对比度
            local_contrast = src - src_local_mean

            # 在mask区域内增强对比度
            contrast_weight = mask * clarity_strength * 0.5
            src = src + local_contrast * contrast_weight

        # 步骤5: 最终混合
        # 修复原方案的外推问题（原 blend_weight 在 mask=1.0 时超过1.0，导致数值溢出和颜色失真）
        # 直接用 mask 作为权重：核心区=1.0(全用修复结果)，边缘平滑过渡到0.0(全用原图)
        blend_weight = mask.clamp(0.0, 1.0)
        result = src * blend_weight + ref * (1.0 - blend_weight)

        return result

    # -------------------- 4. Poisson梯度域融合 --------------------
    @staticmethod
    def poisson_blend_latent(src: torch.Tensor, dst: torch.Tensor, 
                             mask: torch.Tensor, iterations: int = 40) -> torch.Tensor:
        """
        Poisson风格融合 - 梯度域无缝贴合
        
        原理：迭代求解使结果梯度接近src，边界匹配dst
        """
        if iterations <= 0:
            return dst * (1.0 - mask) + src * mask

        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        if mask.shape[1] != src.shape[1]:
            mask = mask.expand(-1, src.shape[1], -1, -1)

        # 确保所有张量在同一设备上（修复CUDA内存访问错误）
        mask = mask.to(src.device)
        dst = dst.to(src.device)

        # 拉普拉斯核
        laplacian_kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=src.dtype, device=src.device).view(1, 1, 3, 3)
        
        # src的拉普拉斯
        src_lap = []
        for c in range(src.shape[1]):
            ch = src[:, c:c+1, :, :]
            lap = F.conv2d(F.pad(ch, (1,1,1,1), mode='replicate'), laplacian_kernel)
            src_lap.append(lap)
        src_laplacian = torch.cat(src_lap, dim=1)
        
        # Jacobi迭代
        result = dst.clone()
        
        for _ in range(iterations):
            padded = F.pad(result, (1,1,1,1), mode='replicate')
            neighbors = (
                padded[:, :, 1:-1, :-2] +
                padded[:, :, 1:-1, 2:] +
                padded[:, :, :-2, 1:-1] +
                padded[:, :, 2:, 1:-1]
            ) / 4.0
            
            guided = neighbors - src_laplacian / 4.0
            result = dst * (1.0 - mask) + guided * mask
        
        return result

    # -------------------- 5. MPB增强融合（V6.6新增） --------------------
    @staticmethod
    def mpb_blend_latent(src: torch.Tensor, dst: torch.Tensor,
                         mask: torch.Tensor, color_preserve: float = 0.5) -> torch.Tensor:
        """
        Modified Poisson Blending - 改进的泊松融合

        特点：更快的收敛速度 + 精确的颜色控制
        color_preserve: 颜色保持参数 (0=完全融合, 1=完全保持源颜色)
        """
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        if mask.shape[1] != src.shape[1]:
            mask = mask.expand(-1, src.shape[1], -1, -1)

        # 确保所有张量在同一设备上（修复CUDA内存访问错误）
        mask = mask.to(src.device)
        dst = dst.to(src.device)

        # MPB算法：结合梯度融合和颜色保持
        # 1. 标准泊松融合部分（简化为少量迭代）
        result = dst.clone()
        laplacian_kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=src.dtype, device=src.device).view(1, 1, 3, 3)

        # 只迭代10次（MPB比标准Poisson快）
        for _ in range(10):
            padded = F.pad(result, (1,1,1,1), mode='replicate')
            neighbors = (
                padded[:, :, 1:-1, :-2] +
                padded[:, :, 1:-1, 2:] +
                padded[:, :, :-2, 1:-1] +
                padded[:, :, 2:, 1:-1]
            ) / 4.0
            result = dst * (1.0 - mask) + neighbors * mask

        # 2. 颜色保持调整
        if color_preserve > 0:
            result = result * (1.0 - color_preserve * mask) + src * (color_preserve * mask)

        return result

    # -------------------- 6. 简单混合（兼容模式） --------------------
    @staticmethod
    def simple_blend(src: torch.Tensor, dst: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """简单alpha混合"""
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        if mask.shape[1] != src.shape[1]:
            mask = mask.expand(-1, src.shape[1], -1, -1)

        # 确保所有张量在同一设备上（修复CUDA内存访问错误）
        mask = mask.to(src.device)
        dst = dst.to(src.device)

        return dst * (1.0 - mask) + src * mask
