"""基础节点：Latent 空间批量 K 采样与放大。

仅保留 upscale_latent —— 源类 LatentKSamplerUpscaler 的 INPUT_TYPES 与 process
从未被调用（该基类未注册），不再移植。
"""
import torch
import torch.nn.functional as F

from .algorithms import LatentAlgorithms


class LatentKSamplerUpscaler:
    """Latent空间K采样器与放大 - 支持批量处理"""

    def upscale_latent(self, samples, method, factor):
        height = int(samples.shape[2] * factor)
        width = int(samples.shape[3] * factor)
        
        if method == "latent_aware":
            return LatentAlgorithms.latent_aware_resize(samples, height, width, mode="adaptive")
        
        # area和nearest模式不支持align_corners参数
        if method in ["nearest", "area"]:
            upscaled = torch.nn.functional.interpolate(
                samples, size=(height, width), mode=method
            )
        else:
            upscaled = torch.nn.functional.interpolate(
                samples, size=(height, width), mode=method, align_corners=False
            )
        return upscaled
    
