"""策略方法 Mixin：模型路径解析、类别规则解析与逐检测框参数生成。"""
import os

from ..model_managers import YOLOModelManager, SAMModelManager


class PolicyMixin:
    def _resolve_model_path(self, yolo_model_selection, manual_path):
        selected_path = YOLOModelManager.get_model_path(yolo_model_selection)
        if selected_path == "MANUAL_INPUT" or not selected_path:
            path = manual_path.strip() if manual_path else ""
        else:
            path = selected_path
        if path and os.path.isfile(path):
            return path
        return ""

    def _resolve_sam_model_path(self, sam_model_selection, manual_path):
        """解析SAM模型路径"""
        selected_path = SAMModelManager.get_model_path(sam_model_selection)
        if selected_path == "MANUAL_INPUT" or not selected_path:
            path = manual_path.strip() if manual_path else ""
        else:
            path = selected_path
        if path and os.path.isfile(path):
            return path
        return ""

    def _parse_target_classes(self, target_str):
        if not target_str or not target_str.strip(): return None
        classes=[c.strip() for c in target_str.split(",") if c.strip()]
        return classes if classes else None

    # -------------------- 智能策略 --------------------

    def _auto_params_for_det(self, det, img_w, img_h, strategy, base_expand, base_feather, base_denoise):
        cls = (det.get("class_name","") or "").lower()
        area_ratio = self._bbox_area(det.get("bbox",[0,0,0,0])) / max(1.0, float(img_w)*float(img_h))

        mask_expand = int(base_expand)
        mask_feather = int(base_feather)
        denoise = float(base_denoise)
        crop_extra = 0
        local_rounds = 1  # 新增：局部修复轮次
        crop_upscale_boost = 1.0  # 新增：裁剪放大倍率加成

        if strategy == "人像(脸/手/脚)":
            if any(k in cls for k in ["face","head","eye","mouth","nose"]):
                mask_expand = int(base_expand * 1.25 + 12)
                mask_feather = int(base_feather * 1.15 + 6)
                denoise = float(base_denoise * 0.75)
                crop_extra = 24
                local_rounds = 1
            elif any(k in cls for k in ["hand","finger","palm"]):
                # ★ 手部：高降噪 + 多轮修复 + 放大看细节
                mask_expand = int(base_expand * 1.3 + 16)
                mask_feather = int(base_feather * 1.2 + 8)
                denoise = float(min(1.0, base_denoise * 1.25 + 0.1))  # 提高降噪！
                crop_extra = 32
                local_rounds = 2  # 手部2轮
                crop_upscale_boost = 1.2  # 放大看细节
            elif any(k in cls for k in ["foot","feet","toe","ankle","leg","knee"]):
                # ★ 脚部：高降噪 + 多轮修复 + 放大看细节
                mask_expand = int(base_expand * 1.3 + 16)
                mask_feather = int(base_feather * 1.2 + 8)
                denoise = float(min(1.0, base_denoise * 1.25 + 0.1))  # 提高降噪！
                crop_extra = 32
                local_rounds = 2  # 脚部2轮
                crop_upscale_boost = 1.2  # 放大看细节
        elif strategy == "通用(按面积)":
            if area_ratio < 0.01:
                mask_expand = int(base_expand * 1.30 + 16)
                mask_feather = int(base_feather * 1.25 + 8)
                denoise = float(base_denoise * 0.85)
                crop_extra = 24
            elif area_ratio > 0.12:
                mask_expand = int(base_expand * 0.85 + 4)
                mask_feather = int(base_feather * 0.90)
                denoise = float(min(1.0, base_denoise * 1.10))
                crop_extra = 12

        mask_expand = max(0, min(256, mask_expand))
        mask_feather = max(0, min(128, mask_feather))
        denoise = max(0.0, min(1.0, denoise))
        crop_extra = max(0, min(256, int(crop_extra)))
        return mask_expand, mask_feather, denoise, crop_extra, local_rounds, crop_upscale_boost

    # -------------------- 类别规则预设模板 (V6.5) --------------------
    def _get_preset_class_rules(self, preset_name: str) -> str:
        """
        获取预设的类别规则模板

        Args:
            preset_name: 预设名称

        Returns:
            规则文本字符串
        """
        presets = {
            "人像优化(face高质量)": """face: steps=20,denoise=0.45,limit=4""",

            "手部优化(hand高降噪)": """hand: denoise=0.75,limit=2""",

            "脚部优化(foot高降噪)": """foot: denoise=0.75,limit=2""",

            "人像全面(face+hand+foot)": """face: steps=20,denoise=0.45,limit=4
hand: denoise=0.75,limit=2
foot: denoise=0.75,limit=2""",
        }

        return presets.get(preset_name, "")

    # -------------------- 类别规则解析 --------------------
    def _parse_class_rules(self, rules_text: str):
        if not rules_text or not str(rules_text).strip():
            return []
        rules = []
        for raw_line in str(rules_text).splitlines():
            line = raw_line.strip()
            if not line or line.startswith('#') or line.startswith('//'):
                continue
            line = line.split('//', 1)[0].strip()
            line = line.split('#', 1)[0].strip()
            if not line or ':' not in line:
                continue
            left, right = line.split(':', 1)
            keys = [k.strip().lower() for k in left.split('|') if k.strip()]
            if not keys:
                continue
            params = {}
            for part in right.split(','):
                part = part.strip()
                if not part or '=' not in part:
                    continue
                k, v = [x.strip().lower() for x in part.split('=', 1)]
                if not k:
                    continue
                try:
                    if v.endswith('px'):
                        v = v[:-2]
                    if '.' in v:
                        val = float(v)
                    else:
                        val = int(v)
                except Exception:
                    continue
                params[k] = val
            rules.append({'keys': keys, 'params': params})
        return rules

    def _match_rule(self, cls_name: str, rule_keys):
        if not rule_keys:
            return False
        name = (cls_name or '').lower()
        for k in rule_keys:
            if k == 'default':
                continue
            if k and (k in name or name in k):
                return True
        return False

    def _get_rule_for_det(self, det, rules):
        if not rules:
            return None
        cls = (det.get('class_name', '') or '').lower()
        matched = None
        default_rule = None
        for r in rules:
            if any(k == 'default' for k in r.get('keys', [])):
                default_rule = r
                continue
            if self._match_rule(cls, r.get('keys', [])):
                matched = r
                break
        return matched or default_rule

    def _priority_sort_detections(self, detections, mode: str, custom_keywords: str = ''):
        if not detections:
            return []
        mode = (mode or '按置信度').strip()
        def cls_lower(d):
            return (d.get('class_name', '') or '').lower()

        if mode == '按面积':
            return sorted(detections, key=lambda d: float(self._bbox_area(d.get('bbox', [0,0,0,0]))), reverse=True)
        if mode == '按置信度':
            return sorted(detections, key=lambda d: float(d.get('confidence', 0.0)), reverse=True)

        if mode == '自定义优先级':
            pri = [x.strip().lower() for x in (custom_keywords or '').split(',') if x.strip()]
        elif mode == '手优先':
            pri = ['hand', 'finger', 'palm', 'arm', 'wrist', 'face', 'head', 'foot', 'shoe', 'leg']
        elif mode == '脚优先':
            pri = ['foot', 'feet', 'shoe', 'leg', 'knee', 'hand', 'face', 'head']
        else:
            pri = ['face', 'head', 'eye', 'mouth', 'nose', 'hand', 'finger', 'palm', 'foot', 'shoe', 'leg']

        def rank(d):
            name = cls_lower(d)
            for idx, kw in enumerate(pri):
                if kw and kw in name:
                    return idx
            return len(pri) + 1

        return sorted(detections, key=lambda d: (rank(d), -float(d.get('confidence', 0.0))))

    def _apply_rule_limits(self, detections, rules, default_per_class_limit: int = 0, max_targets: int = 0):
        if not detections:
            return []
        out = []
        seen = {}
        for d in detections:
            cls_key = d.get('class_id', d.get('class_name', 'unknown'))
            rule = self._get_rule_for_det(d, rules)
            limit = None
            if rule:
                limit = rule.get('params', {}).get('limit', None)
            if limit is None:
                limit = default_per_class_limit
            if limit and int(limit) > 0:
                cnt = seen.get(cls_key, 0)
                if cnt >= int(limit):
                    continue
                seen[cls_key] = cnt + 1
            out.append(d)
            if max_targets and int(max_targets) > 0 and len(out) >= int(max_targets):
                break
        return out

    def _build_det_policy(self, det, img_w, img_h,
                          smart_strategy: str,
                          base_expand: int, base_feather: int,
                          base_denoise: float,
                          base_crop_pad: int, base_crop_upscale: float, base_crop_max_edge: int,
                          base_crop_steps: int, base_cfg: float,
                          adaptive_denoise: bool, conf_low_scale: float, conf_high_scale: float,
                          rules,
                          # V6.1 手脚增强参数
                          hand_foot_enhance: bool = True,
                          hand_foot_denoise: float = 0.70,
                          hand_foot_extra_rounds: int = 1,
                          # V6.11 新增手脚优化参数
                          hand_foot_cfg_boost: float = 1.2,
                          hand_foot_cn_strength: float = 0.7,
                          hand_foot_denoise_decay: float = 0.85,
                          hand_foot_denoise_min: float = 0.40):
        m_expand, m_feather, dn, crop_extra, auto_rounds, upscale_boost = self._auto_params_for_det(
            det, img_w, img_h, smart_strategy, base_expand, base_feather, base_denoise
        )
        pad = int(base_crop_pad + crop_extra)
        upscale = float(base_crop_upscale * upscale_boost)  # 应用放大倍率加成
        max_edge = int(base_crop_max_edge)
        steps = int(base_crop_steps)
        cfg = float(base_cfg)
        rounds = max(1, auto_rounds)  # 使用自动轮次

        # ★ V6.10 手脚增强模式（优化修复质量）
        cls = (det.get("class_name", "") or "").lower()
        is_hand_foot = any(k in cls for k in ["hand", "finger", "palm", "foot", "feet", "toe", "ankle"])

        if hand_foot_enhance and is_hand_foot:
            # 强制使用更高降噪（0.80，参考HandRefiner最佳实践）
            dn = max(dn, float(hand_foot_denoise))
            # 增加轮次（V6.11.1: 限制最多2轮，防止多目标时采样爆炸）
            rounds = min(2, max(rounds, 1 + int(hand_foot_extra_rounds)))
            # 增加mask扩展（建议30px+，避免边缘瑕疵）
            m_expand = max(m_expand, max(30, int(base_expand * 1.5)))
            # 增加mask羽化（24-32px，柔和过渡）
            m_feather = max(m_feather, max(24, int(base_feather * 1.5)))
            # 增加pad确保充足上下文（手臂、连接部分）
            pad = max(pad, int(base_crop_pad * 1.5 + 48))
            # 增加步数（V6.11.1: 限制最多25步，平衡质量和速度）
            steps = min(25, max(steps, int(base_crop_steps * 1.25)))
            # 适度增加放大倍率
            upscale = max(upscale, min(2.0, float(base_crop_upscale) * 1.2))

        if adaptive_denoise and not (hand_foot_enhance and is_hand_foot):
            # 手脚增强模式下不再降低降噪
            conf = float(det.get('confidence', 0.0))
            t = max(0.0, min(1.0, (conf - 0.05) / 0.95))
            scale = float(conf_low_scale) * (1.0 - t) + float(conf_high_scale) * t
            dn = max(0.0, min(1.0, float(dn) * float(scale)))

        rule = self._get_rule_for_det(det, rules)
        if rule:
            p = rule.get('params', {}) or {}
            if 'expand' in p: m_expand = int(p['expand'])
            if 'feather' in p: m_feather = int(p['feather'])
            if 'denoise' in p: dn = float(p['denoise'])
            if 'pad' in p: pad = int(p['pad'])
            if 'upscale' in p: upscale = float(p['upscale'])
            if 'max_edge' in p: max_edge = int(p['max_edge'])
            if 'steps' in p: steps = int(p['steps'])
            if 'cfg' in p: cfg = float(p['cfg'])
            if 'rounds' in p: rounds = int(max(1, min(4, int(p['rounds']))))

        m_expand = max(0, min(256, int(m_expand)))
        m_feather = max(0, min(128, int(m_feather)))
        pad = max(0, min(512, int(pad)))
        upscale = max(1.0, min(4.0, float(upscale)))
        max_edge = max(0, min(4096, int(max_edge)))
        steps = max(1, min(200, int(steps)))
        cfg = max(0.0, min(100.0, float(cfg)))
        dn = max(0.0, min(1.0, float(dn)))

        # V6.11: 手脚区域的CFG和CN强度调整
        cfg_final = cfg
        cn_strength_override = None  # None表示使用默认值
        denoise_decay = 0.75  # 默认衰减
        denoise_min = 0.0  # 默认无下限

        if hand_foot_enhance and is_hand_foot:
            # 应用CFG加成
            cfg_final = float(cfg) * float(hand_foot_cfg_boost)
            cfg_final = max(0.0, min(100.0, cfg_final))
            # 设置CN强度覆盖
            cn_strength_override = float(hand_foot_cn_strength)
            # 设置渐进式降噪参数
            denoise_decay = float(hand_foot_denoise_decay)
            denoise_min = float(hand_foot_denoise_min)

        return {
            'mask_expand': m_expand,
            'mask_feather': m_feather,
            'denoise': dn,
            'crop_pad': pad,
            'crop_upscale': upscale,
            'crop_max_edge': max_edge,
            'steps': steps,
            'cfg': cfg_final,
            'local_rounds': rounds,
            'is_hand_foot': is_hand_foot,  # 标记是否是手脚
            # V6.11 新增
            'cn_strength_override': cn_strength_override,  # 手脚区域CN强度覆盖
            'denoise_decay': denoise_decay,  # 多轮降噪衰减系数
            'denoise_min': denoise_min,  # 降噪下限
        }

    # -------------------- 低显存检测 --------------------
