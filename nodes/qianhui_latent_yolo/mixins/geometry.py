"""检测框几何方法 Mixin：NMS、IoU/WIoU、按目标筛选与分组。"""


class GeometryMixin:
    def _nms_detections_enhanced(self, detections, iou_thresh=0.5):
        """增强版NMS，支持seg mask合并"""
        import numpy as np
        if not detections:
            return []
        
        detections = sorted(detections, key=lambda d: d["confidence"], reverse=True)
        keep = []
        suppressed = [False] * len(detections)
        
        for i, det_i in enumerate(detections):
            if suppressed[i]:
                continue
            keep.append(det_i)
            
            for j in range(i + 1, len(detections)):
                if suppressed[j]:
                    continue
                if det_i["class_id"] != detections[j]["class_id"]:
                    continue
                
                iou = self._iou_xyxy(det_i["bbox"], detections[j]["bbox"])
                if iou >= iou_thresh:
                    suppressed[j] = True
                    # 合并seg mask
                    if det_i.get("seg_mask") is not None and detections[j].get("seg_mask") is not None:
                        # V6.11.4：形状不一致时先对齐，绝不硬崩。历史上翻转TTA 的
                        # letterbox 遮罩与已规范化的原图遮罩混进同一轮 NMS，直接
                        # ValueError: operands could not be broadcast (1280,1024)(768,640)，
                        # 整个检测节点挂掉。修在源头（detection.py 翻转后重采样）之外，
                        # 这里再兜一层，任何未来的形状不一致都退化成「不合并」而不是崩。
                        a, b = det_i["seg_mask"], detections[j]["seg_mask"]
                        if a.shape != b.shape:
                            try:
                                import cv2
                                b = cv2.resize(np.asarray(b, dtype=np.float32),
                                               (a.shape[1], a.shape[0]),
                                               interpolation=cv2.INTER_LINEAR)
                            except Exception:
                                b = None
                        if b is not None:
                            det_i["seg_mask"] = np.maximum(a, b)
        
        return keep

    @staticmethod
    def _iou_xyxy(box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        inter_x1 = max(ax1, bx1); inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2); inter_y2 = min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1); inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area <= 0.0: return 0.0
        area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
        area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
        union = area_a + area_b - inter_area
        if union <= 0.0: return 0.0
        return float(inter_area / union)

    @staticmethod
    def _wiou_xyxy(box_a, box_b, alpha=3.0):
        """
        Weighted IoU (WIoU) 计算 - V6.7 新增

        基于原论文：https://arxiv.org/abs/2301.10051
        特点：通过动态聚焦机制，更关注困难样本的边界框

        Args:
            box_a, box_b: [x1, y1, x2, y2]
            alpha: 聚焦参数，越大越关注困难样本（推荐 3.0）

        Returns:
            WIoU 值 (0-1)
        """
        # 标准 IoU 计算
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1); inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2); inter_y2 = min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        if inter_area <= 0.0:
            return 0.0

        area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
        area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
        union = area_a + area_b - inter_area

        if union <= 0.0:
            return 0.0

        iou = inter_area / union

        # WIoU 权重计算
        # 计算中心点距离
        center_a_x = (ax1 + ax2) / 2.0
        center_a_y = (ay1 + ay2) / 2.0
        center_b_x = (bx1 + bx2) / 2.0
        center_b_y = (by1 + by2) / 2.0

        # 最小外接框
        c_x1 = min(ax1, bx1); c_y1 = min(ay1, by1)
        c_x2 = max(ax2, bx2); c_y2 = max(ay2, by2)
        c_w = c_x2 - c_x1
        c_h = c_y2 - c_y1
        c_diag = (c_w ** 2 + c_h ** 2) ** 0.5

        if c_diag <= 0.0:
            return iou

        # 中心点距离
        dist = ((center_a_x - center_b_x) ** 2 + (center_a_y - center_b_y) ** 2) ** 0.5

        # WIoU 权重（距离越远，权重越小）
        weight = 1.0 / (1.0 + (dist / c_diag) ** alpha)

        # Weighted IoU
        wiou = iou * weight

        return float(wiou)

    @staticmethod
    def _match_class_name(cls_name: str, target_classes):
        if not target_classes: return True
        if not cls_name: return False
        cls_name_lower = cls_name.lower()
        for tc in target_classes:
            tc_lower = str(tc).lower().strip()
            if not tc_lower: continue
            if tc_lower in cls_name_lower or cls_name_lower in tc_lower: return True
            if tc_lower.rstrip('s') == cls_name_lower.rstrip('s'): return True
        return False

    def _filter_detections_by_target(self, detections, target_classes):
        if not target_classes: return detections
        return [d for d in detections if self._match_class_name(d.get("class_name",""), target_classes)]

    def _bbox_area(self, bbox):
        x1, y1, x2, y2 = [float(v) for v in (bbox or [0,0,0,0])]
        w = max(0.0, x2-x1); h = max(0.0, y2-y1)
        return w*h

    def _bbox_aspect(self, bbox):
        x1, y1, x2, y2 = [float(v) for v in (bbox or [0,0,0,0])]
        w = max(1e-6, x2-x1); h = max(1e-6, y2-y1)
        r1 = w/h; r2 = h/w
        return float(r1 if r1>=r2 else r2)

    def _assign_lr_group(self, bbox, image_width):
        if not image_width or image_width <= 0: return "C"
        x1, _, x2, _ = [float(v) for v in (bbox or [0,0,0,0])]
        cx = (x1+x2)*0.5
        return "L" if cx < (float(image_width)*0.5) else "R"

    def _filter_detections_geometry(self, detections, image_w, image_h,
                                   min_area_ratio=0.0, max_area_ratio=1.0,
                                   min_aspect=0.0, max_aspect=10.0):
        if not detections: return []
        img_area = max(1.0, float(image_w)*float(image_h))
        out=[]
        for d in detections:
            bbox=d.get("bbox",[0,0,0,0])
            area_ratio = self._bbox_area(bbox)/img_area
            aspect = self._bbox_aspect(bbox)
            if min_area_ratio>0.0 and area_ratio<float(min_area_ratio): continue
            if max_area_ratio<1.0 and area_ratio>float(max_area_ratio): continue
            if min_aspect>0.0 and aspect<float(min_aspect): continue
            if max_aspect>0.0 and aspect>float(max_aspect): continue
            out.append(d)
        return out

    def _sort_detections(self, detections, sort_mode="置信度"):
        if not detections: return []
        mode=(sort_mode or "置信度").strip()
        if mode=="面积":
            return sorted(detections, key=lambda d: float(self._bbox_area(d.get("bbox",[0,0,0,0]))), reverse=True)
        return sorted(detections, key=lambda d: float(d.get("confidence",0.0)), reverse=True)

    def _limit_detections_per_class(self, detections, per_class_limit=0):
        if not detections or not per_class_limit or per_class_limit<=0: return detections or []
        out=[]; seen={}
        for d in detections:
            cls=d.get("class_id", d.get("class_name","unknown"))
            cnt=seen.get(cls,0)
            if cnt>=per_class_limit: continue
            out.append(d); seen[cls]=cnt+1
        return out

    def _expand_bbox_xyxy(self, bbox, image_w, image_h, extra_px=0, extra_ratio=0.0):
        if bbox is None: return [0.0,0.0,0.0,0.0]
        x1,y1,x2,y2=[float(v) for v in bbox]
        w=max(0.0,x2-x1); h=max(0.0,y2-y1)
        extra=float(extra_px)+float(max(w,h)*float(extra_ratio or 0.0))
        if extra<=0.0: return [x1,y1,x2,y2]
        x1=max(0.0,x1-extra); y1=max(0.0,y1-extra)
        x2=min(float(image_w),x2+extra); y2=min(float(image_h),y2+extra)
        return [x1,y1,x2,y2]

    def _merge_overlapping_detections(self, detections, iou_threshold=0.35, same_class=True, lr_protect=False, image_width=None,
                                      use_wiou=False, wiou_alpha=3.0):
        """
        V6.7: 支持 WIoU 优化的边界框合并
        """
        if not detections or len(detections)<=1: return detections or []
        dets=sorted(detections, key=lambda d: float(d.get("confidence",0.0)), reverse=True)
        boxes=[d.get("bbox",[0,0,0,0]) for d in dets]
        cls_ids=[d.get("class_id",-1) for d in dets]
        lr_groups=None
        if lr_protect and image_width:
            lr_groups=[self._assign_lr_group(b, float(image_width)) for b in boxes]
        n=len(dets)
        adj=[[] for _ in range(n)]
        for i in range(n):
            for j in range(i+1,n):
                if same_class and cls_ids[i]!=cls_ids[j]: continue
                if lr_groups is not None and lr_groups[i]!=lr_groups[j]: continue
                # V6.7: 使用 WIoU 或标准 IoU
                if use_wiou:
                    similarity = self._wiou_xyxy(boxes[i], boxes[j], alpha=wiou_alpha)
                else:
                    similarity = self._iou_xyxy(boxes[i], boxes[j])
                if similarity >= float(iou_threshold):
                    adj[i].append(j); adj[j].append(i)
        visited=[False]*n
        merged=[]
        for i in range(n):
            if visited[i]: continue
            stack=[i]; visited[i]=True; comp=[i]
            while stack:
                u=stack.pop()
                for v in adj[u]:
                    if not visited[v]:
                        visited[v]=True; stack.append(v); comp.append(v)
            x1=min(boxes[k][0] for k in comp); y1=min(boxes[k][1] for k in comp)
            x2=max(boxes[k][2] for k in comp); y2=max(boxes[k][3] for k in comp)
            best_k=max(comp, key=lambda k: float(dets[k].get("confidence",0.0)))
            base=dets[best_k].copy()
            base["bbox"]=[float(x1),float(y1),float(x2),float(y2)]
            base["merged_count"]=len(comp)
            merged.append(base)
        merged.sort(key=lambda d: float(d.get("confidence",0.0)), reverse=True)
        return merged
