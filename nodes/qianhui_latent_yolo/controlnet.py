"""ControlNet 辅助类（conditioning 注入与控制图预处理）。"""
import os

import numpy as np
import torch
import torch.nn.functional as F
import comfy.controlnet
import folder_paths


# ============================================================
# ControlNet 辅助类
# ============================================================

class ControlNetHelper:
    """ControlNet处理辅助类"""
    
    @staticmethod
    def apply_controlnet(positive, negative, control_net, image, strength, start_percent=0.0, end_percent=1.0):
        """应用ControlNet到conditioning"""
        if control_net is None or image is None:
            return positive, negative
        
        if strength == 0:
            return positive, negative
        
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        c_net = control_net.copy().set_cond_hint(image, strength, (start_percent, end_percent))
        
        out_positive = []
        for conditioning in positive:
            c = conditioning[0].clone()
            keys = conditioning[1].copy()
            
            control_hint = keys.get("control", None)
            if control_hint is not None:
                c_net_copy = c_net.copy()
                c_net_copy.set_previous_controlnet(control_hint)
                keys["control"] = c_net_copy
            else:
                keys["control"] = c_net
            
            out_positive.append([c, keys])
        
        out_negative = []
        for conditioning in negative:
            c = conditioning[0].clone()
            keys = conditioning[1].copy()
            
            control_hint = keys.get("control", None)
            if control_hint is not None:
                c_net_copy = c_net.copy()
                c_net_copy.set_previous_controlnet(control_hint)
                keys["control"] = c_net_copy
            else:
                keys["control"] = c_net
            
            out_negative.append([c, keys])
        
        return out_positive, out_negative

    @staticmethod
    def prepare_control_image(image, target_height, target_width):
        """准备控制图像，调整到目标尺寸"""
        if image is None:
            return None
        
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        b, h, w, c = image.shape
        
        if h != target_height or w != target_width:
            image = image.permute(0, 3, 1, 2)
            image = torch.nn.functional.interpolate(
                image, size=(target_height, target_width),
                mode='bilinear', align_corners=False
            )
            image = image.permute(0, 2, 3, 1)
        
        return image
