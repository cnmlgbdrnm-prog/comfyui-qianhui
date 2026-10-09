"""遮罩方法 Mixin：检测掩码生成、EMA 注意力计算与检测预览绘制。"""
import torch
import torch.nn.functional as F

from ..algorithms import LatentAlgorithms


class MasksMixin:
    def _create_detection_mask(self, detections, height, width, expand=32, feather=16, seg_feather=3.0):
        """
        V6.9优化：更安全的seg mask处理，参考Ultralytics官方方式
        优先使用seg mask，fallback到bbox
        """
        import numpy as np
        mask = np.zeros((height, width), dtype=np.float32)
        if not detections:
            return torch.from_numpy(mask)

        has_any_seg = any(det.get("seg_mask") is not None for det in detections)

        for det in detections:
            seg_mask = det.get("seg_mask", None)

            if seg_mask is not None:
                try:
                    # 确保seg_mask是numpy数组
                    if not isinstance(seg_mask, np.ndarray):
                        seg_mask = np.array(seg_mask)

                    # 确保是2D (H, W)
                    while seg_mask.ndim > 2:
                        seg_mask = seg_mask.squeeze() if seg_mask.shape[0] == 1 else seg_mask[0]

                    # 如果mask尺寸与目标不匹配，进行resize
                    if seg_mask.shape[0] != height or seg_mask.shape[1] != width:
                        # 转为torch tensor进行resize
                        seg_t = torch.from_numpy(seg_mask.astype(np.float32))

                        # 确保是4D张量 (1, 1, H, W)
                        if seg_t.ndim == 2:
                            seg_t = seg_t.unsqueeze(0).unsqueeze(0)
                        elif seg_t.ndim == 3:
                            seg_t = seg_t.unsqueeze(0)

                        # Resize到目标尺寸
                        seg_t = F.interpolate(seg_t, size=(height, width), mode="bilinear", align_corners=False)
                        seg_mask = seg_t.squeeze().cpu().numpy()

                    # 确保值在[0, 1]范围
                    seg_mask = np.clip(seg_mask, 0.0, 1.0)

                    # 膨胀seg mask（expand）
                    if expand > 0:
                        seg_t = torch.from_numpy(seg_mask.astype(np.float32))
                        if seg_t.ndim == 2:
                            seg_t = seg_t.unsqueeze(0).unsqueeze(0)
                        kernel_size = expand * 2 + 1
                        seg_t = F.max_pool2d(seg_t, kernel_size=kernel_size, stride=1, padding=expand)
                        seg_mask = seg_t.squeeze().cpu().numpy()

                    mask = np.maximum(mask, seg_mask)

                except Exception as e:
                    # seg mask处理失败，回退到bbox
                    print(f"    [警告] Seg mask处理失败，回退到bbox: {e}")
                    x1, y1, x2, y2 = det["bbox"]
                    x1 = max(0, int(x1) - int(expand))
                    y1 = max(0, int(y1) - int(expand))
                    x2 = min(width, int(x2) + int(expand))
                    y2 = min(height, int(y2) + int(expand))
                    mask[y1:y2, x1:x2] = 1.0
            else:
                # Fallback到bbox
                x1, y1, x2, y2 = det["bbox"]
                x1 = max(0, int(x1) - int(expand))
                y1 = max(0, int(y1) - int(expand))
                x2 = min(width, int(x2) + int(expand))
                y2 = min(height, int(y2) + int(expand))
                mask[y1:y2, x1:x2] = 1.0
        
        mask = torch.from_numpy(mask)
        
        # 羽化
        sigma = seg_feather if has_any_seg else max(1.0, float(feather) / 3.0)
        if sigma > 0 and mask.sum() > 0:
            mask = LatentAlgorithms.gaussian_feather(mask, sigma=sigma)
        
        return mask

    def _calculate_ema_attention(self, detections, height, width, focus_strength=1.5):
        """
        EMA (Efficient Multi-Scale Attention) 注意力机制 - V6.7 新增

        基于检测置信度和空间位置计算注意力权重，增强关键区域

        Args:
            detections: 检测结果列表
            height, width: 图像尺寸
            focus_strength: 聚焦强度 (1.0-3.0)，越大越聚焦高置信度区域

        Returns:
            attention_map: torch.Tensor [H, W]，值域 [0, 1]
        """
        import numpy as np

        if not detections:
            return torch.ones((height, width), dtype=torch.float32)

        # 初始化注意力图（基础值0.3，确保背景也有适度权重）
        attention = np.full((height, width), 0.3, dtype=np.float32)

        # 多尺度处理：为每个检测生成注意力响应
        for det in detections:
            conf = float(det.get('confidence', 0.5))
            x1, y1, x2, y2 = det.get("bbox", [0, 0, width, height])

            # 边界保护
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(width, int(x2))
            y2 = min(height, int(y2))

            if x2 <= x1 or y2 <= y1:
                continue

            # 计算置信度权重（使用指数函数增强高置信度）
            # focus_strength 控制聚焦强度
            conf_weight = conf ** (1.0 / max(0.5, focus_strength))

            # 中心点和半径
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            radius_x = (x2 - x1) / 2.0
            radius_y = (y2 - y1) / 2.0

            # 创建二维高斯分布作为注意力响应
            # 使用网格计算距离
            y_grid, x_grid = np.ogrid[0:height, 0:width]

            # 归一化距离（椭圆形区域）
            dx = (x_grid - cx) / max(1.0, radius_x)
            dy = (y_grid - cy) / max(1.0, radius_y)
            dist_sq = dx * dx + dy * dy

            # 高斯响应（sigma=1.5 确保覆盖检测框并有适度衰减）
            gaussian_response = np.exp(-dist_sq / (2.0 * 1.5 ** 2))

            # 应用置信度权重
            weighted_response = gaussian_response * conf_weight

            # 累加到注意力图（使用max保证高置信度区域不被稀释）
            attention = np.maximum(attention, weighted_response)

        # 归一化到 [0.5, 1.0] 范围（确保所有区域都有基础权重）
        attention = attention * 0.5 + 0.5

        return torch.from_numpy(attention).float()

    def _draw_detection_preview(self, image, detections):
        import numpy as np
        preview=image.cpu().numpy().copy() if isinstance(image, torch.Tensor) else image.copy()
        if preview.max()<=1.0: preview=(preview*255).astype(np.uint8)
        
        h, w = preview.shape[:2]
        thickness = 2
        
        for det in detections:
            # 边界裁剪，防止越界
            x1 = max(0, min(w - 1, int(det["bbox"][0])))
            y1 = max(0, min(h - 1, int(det["bbox"][1])))
            x2 = max(0, min(w, int(det["bbox"][2])))
            y2 = max(0, min(h, int(det["bbox"][3])))
            
            # 确保有效区域
            if x2 <= x1 or y2 <= y1:
                continue
            
            # 上边
            t_y2 = min(y1 + thickness, y2)
            if t_y2 > y1:
                preview[y1:t_y2, x1:x2] = [255, 0, 0]
            
            # 下边
            t_y1 = max(y2 - thickness, y1)
            if y2 > t_y1:
                preview[t_y1:y2, x1:x2] = [255, 0, 0]
            
            # 左边
            t_x2 = min(x1 + thickness, x2)
            if t_x2 > x1:
                preview[y1:y2, x1:t_x2] = [255, 0, 0]
            
            # 右边
            t_x1 = max(x2 - thickness, x1)
            if x2 > t_x1:
                preview[y1:y2, t_x1:x2] = [255, 0, 0]
        
        preview = preview.astype(np.float32) / 255.0
        return torch.from_numpy(preview)
