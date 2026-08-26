"""
千绘 · 角色模块 (角色定制节点 QianhuiCharacterCreator)

模块职责：基于 data/ 目录下的 JSON 素材库，按身份/年龄/种族/体型/发型/
服装/饰品/构图等 30+ 维度组合生成角色提示词。
数据流：data/*.json -> 选项列表 -> 用户选择 -> 提取英文 Tag -> 组合输出 STRING。
外部依赖：无（仅标准库）。年龄选项仅保留成年角色类别。
"""

"""
Qianhui Character Creator (源自 Gemini Character Creator V36.3)
优化内容：
- 删除安全模式下胸部屏蔽逻辑
- 删除默认模式下的强制穿衣逻辑
- 性能优化：减少重复计算、预编译正则
- 代码结构优化
"""

import re
import json
import os

# ==========================================
# 📂 数据加载器
# ==========================================

# 预编译正则表达式（性能优化）
_EXTRACT_PATTERN = re.compile(r'\((.*?)\)')

def load_json_data(filename):
    """加载 data 目录下的 json 文件"""
    # 节点位于 nodes/ 子目录, 数据在插件根目录
    curr_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(curr_dir, "data", f"{filename}.json")

    if not os.path.exists(data_path):
        print(f"⚠️ 警告: 未找到数据文件 {data_path}")
        return {}

    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ 读取 {filename}.json 失败: {e}")
        return {}


def get_list(data_dict, key):
    """安全获取列表 (过滤 "---" 分组标题行, 防止中文标题被当作选项输出)"""
    items = data_dict.get(key, ["(不指定)"])
    if not isinstance(items, list):
        return ["(不指定)"]
    cleaned = [x for x in items if not str(x).startswith("---")]
    return cleaned if cleaned else ["(不指定)"]


# ==========================================
# 📦 数据初始化
# ==========================================

print("⏳ 正在加载千绘 Character Data...")

COLORS_DATA = load_json_data("colors")
COLORS = COLORS_DATA if isinstance(COLORS_DATA, list) else [
    "(不指定)", "🔴 红 (red)", "⚫ 黑 (black)", "⚪ 白 (white)"
]

IDENTITY_DATA = load_json_data("identity")
BODY_DATA = load_json_data("body")
HEAD_DATA = load_json_data("head")
CLOTHING_DATA = load_json_data("clothing")
ACCESSORIES_DATA = load_json_data("accessories")
COMPOSITION_DATA = load_json_data("composition")
ENHANCEMENT_DATA = load_json_data("enhancements")

AGE_ENHANCEMENT = ENHANCEMENT_DATA.get("age_enhancement", {})
RACE_ENHANCEMENT = ENHANCEMENT_DATA.get("race_enhancement", {})

print("✅ 千绘 Character Data 加载完成 (V36.3)")


# ==========================================
# 🎨 角色创建节点
# ==========================================

class QianhuiCharacterCreator:
    """千绘 角色创建器 (原 V36.3)"""

    # 年龄映射表（优化：避免重复字符串匹配）
    AGE_IDENTITY_MAP = {
        "milf": "1milf",
        "mature female": "1 mature female",
        "adult": "1adult",
        "elder": "1elder",
    }

    # 机器人种族关键词
    ROBOT_KEYWORDS = frozenset(["robot", "android", "cyborg"])

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "00_工作安全策略": (["🟢 安全模式 (SFW)", "🔞 绅士模式 (NSFW)", "⚪ 默认/无限制 (Default)"],),
                # --- 1. 核心 ---
                "01_身份": (get_list(IDENTITY_DATA, "identity"),),
                "02_年龄与人设": (get_list(IDENTITY_DATA, "age"),),
                "03_种族": (get_list(IDENTITY_DATA, "race"),),
                # --- 2. 身体 ---
                "04_体型": (get_list(BODY_DATA, "body_type"),),
                "05_肤色": (get_list(BODY_DATA, "skin_color"),),
                "06_胸部": (get_list(BODY_DATA, "breast_size"),),
                # --- 3. 头部 ---
                "07_发型": (get_list(HEAD_DATA, "hair_style"),),
                "08_发色": (COLORS,),
                "09_头发长度": (get_list(HEAD_DATA, "hair_length"),),
                "10_瞳色": (get_list(HEAD_DATA, "eye_color"),),
                "11_瞳孔样式": (get_list(HEAD_DATA, "pupil_style"),),
                "12_面部特征": (get_list(HEAD_DATA, "face_features"),),
                "13_表情": (get_list(HEAD_DATA, "expression"),),
                # --- 4. 服装与材质 ---
                "14_套装选择": (get_list(CLOTHING_DATA, "suit_style"),),
                "15_套装颜色": (COLORS,),
                "16_上装款式": (get_list(CLOTHING_DATA, "top_style"),),
                "17_上装颜色": (COLORS,),
                "18_下装款式": (get_list(CLOTHING_DATA, "bottom_style"),),
                "19_下装颜色": (COLORS,),
                "20_服装材质": (get_list(CLOTHING_DATA, "fabric_material"),),
                # --- 5. 细节 ---
                "21_腿部特征": (get_list(CLOTHING_DATA, "leg_feature"),),
                "22_袜饰": (get_list(CLOTHING_DATA, "socks_style"),),
                "23_袜饰颜色": (COLORS,),
                "24_鞋子": (get_list(CLOTHING_DATA, "shoes_style"),),
                "25_鞋子颜色": (COLORS,),
                # --- 6. 饰品 ---
                "26_头饰": (get_list(ACCESSORIES_DATA, "acc_head"),),
                "27_面饰": (get_list(ACCESSORIES_DATA, "acc_face"),),
                "28_颈饰": (get_list(ACCESSORIES_DATA, "acc_neck"),),
                "29_手部饰品": (get_list(ACCESSORIES_DATA, "acc_hand"),),
                "30_特殊饰品": (get_list(ACCESSORIES_DATA, "acc_special"),),
                # --- 7. 扩展 ---
                "31_构图范围": (get_list(COMPOSITION_DATA, "shot_type"),),
                "32_Lora触发词": ("STRING", {"default": "", "placeholder": "输入Lora触发词 (会自动融合)"}),
                "33_手动补充": ("STRING", {"default": "", "multiline": True, "placeholder": "在这里输入自定义 Tag..."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Character_Prompt",)
    FUNCTION = "generate_character"
    CATEGORY = "千绘/角色"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    # ==========================================
    # 辅助方法
    # ==========================================

    @staticmethod
    def _extract(text):
        """提取括号内的英文内容; 无括号时仅当文本含英文才输出, 中文标题不泄漏"""
        if not text or "(不指定" in text:
            return ""
        match = _EXTRACT_PATTERN.search(text)
        if match:
            return match.group(1)
        return text if re.search(r"[A-Za-z]", text) else ""

    @staticmethod
    def _wrap(text):
        """包裹括号"""
        return f"({text})" if text else ""

    @staticmethod
    def _combine(color_sel, item_sel, material_sel=None):
        """组合颜色、物品、材质"""
        extract = QianhuiCharacterCreator._extract

        c = extract(color_sel)
        i = extract(item_sel)
        m = extract(material_sel) if material_sel else ""

        if not i:
            return ""

        i_lower = i.lower()
        if "nude" in i_lower:
            return "nude"
        if "bare" in i_lower:
            return i

        attrs = []
        if c:
            attrs.append(c)
        if m:
            attrs.append(m)
        attrs.append(i)

        return f"({' '.join(attrs)})"

    def _get_age_identity(self, raw_age):
        """获取年龄对应的身份标识"""
        for keyword, identity in self.AGE_IDENTITY_MAP.items():
            if keyword in raw_age:
                return identity
        return ""

    def _is_robot_race(self, raw_race):
        """判断是否为机器人种族"""
        if not raw_race:
            return False
        race_lower = raw_race.lower()
        return any(kw in race_lower for kw in self.ROBOT_KEYWORDS)

    # ==========================================
    # 主生成方法
    # ==========================================

    def generate_character(self, **kwargs):
        parts = []
        extract = self._extract
        wrap = self._wrap
        combine = self._combine

        # ==========================================
        # 0. 安全策略
        # ==========================================
        safety_opt = kwargs["00_工作安全策略"]
        is_nsfw_mode = "🔞" in safety_opt

        if "🟢" in safety_opt:
            parts.append("(rating: safe:1.2), (rating: general:1.2)")
        elif is_nsfw_mode:
            parts.append("(nsfw:1.2), (rating: explicit:1.2), (uncensored:1.1)")

        # ==========================================
        # 1. 核心身份
        # ==========================================
        raw_identity = extract(kwargs["01_身份"])
        raw_age = extract(kwargs["02_年龄与人设"])
        raw_race = extract(kwargs["03_种族"])
        lora_trigger = kwargs.get("32_Lora触发词", "").strip()

        main_identity = raw_identity or "1girl"

        # 年龄身份转换
        age_identity = self._get_age_identity(raw_age)
        if age_identity and "1girl" in main_identity:
            main_identity = age_identity

        # 年龄额外标签
        age_extra_tags = []
        if raw_age in AGE_ENHANCEMENT:
            age_extra_tags.extend(AGE_ENHANCEMENT[raw_age])
        elif raw_age and raw_age not in main_identity:
            age_extra_tags.append(raw_age)

        # 构建核心标签
        core_tags = []
        if lora_trigger:
            core_tags.append(lora_trigger)
        core_tags.append(main_identity)
        core_tags.extend(age_extra_tags)

        # 种族处理
        if raw_race:
            if raw_race in RACE_ENHANCEMENT:
                core_tags.extend(RACE_ENHANCEMENT[raw_race])
            else:
                core_tags.append(raw_race)

        # Solo 标签
        is_baby = raw_age and ("baby" in raw_age or "toddler" in raw_age)
        if not is_baby:
            exclude_patterns = ["no humans", "2girls", "2boys", "1boy, 1girl"]
            if not any(p in main_identity for p in exclude_patterns):
                core_tags.append("solo")

        parts.append(f"({', '.join(core_tags)})")

        # ==========================================
        # 2. 身体
        # ==========================================
        body_tag = extract(kwargs["04_体型"])
        if body_tag:
            parts.append(f"({body_tag}:1.35)")

        parts.append(wrap(extract(kwargs["05_肤色"])))

        # 胸部 - 直接输出，不做任何限制
        breast = extract(kwargs["06_胸部"])
        if breast:
            is_robot = self._is_robot_race(raw_race)
            if is_robot and is_nsfw_mode:
                parts.append(f"({breast}), (organic breasts:1.1), (human skin breasts:1.1)")
            else:
                parts.append(wrap(breast))

        # ==========================================
        # 3. 头发
        # ==========================================
        hair_parts = []
        h_color = extract(kwargs["08_发色"])
        h_style = extract(kwargs["07_发型"])
        h_len = extract(kwargs["09_头发长度"])

        if h_color:
            hair_parts.append(f"{h_color} hair")
        if h_len:
            hair_parts.append(h_len)
        if h_style:
            hair_parts.append(h_style)

        if hair_parts:
            parts.append(f"({', '.join(hair_parts)})")

        parts.append(wrap(extract(kwargs["26_头饰"])))

        # ==========================================
        # 4. 脸部
        # ==========================================
        eyes_parts = []
        e_color = extract(kwargs["10_瞳色"])
        e_pupil = extract(kwargs["11_瞳孔样式"])
        if e_color:
            eyes_parts.append(e_color)
        if e_pupil:
            eyes_parts.append(e_pupil)
        if eyes_parts:
            parts.append(f"({', '.join(eyes_parts)})")

        parts.append(wrap(extract(kwargs["12_面部特征"])))
        parts.append(wrap(extract(kwargs["13_表情"])))
        parts.append(wrap(extract(kwargs["27_面饰"])))

        # ==========================================
        # 5. 服装
        # ==========================================
        material_tag = kwargs["20_服装材质"]
        parts.append(wrap(extract(kwargs["28_颈饰"])))

        suit = combine(kwargs["15_套装颜色"], kwargs["14_套装选择"], material_tag)
        is_suit_nude = (suit == "nude")

        if is_suit_nude:
            parts.append("nude")
            if self._is_robot_race(raw_race) and is_nsfw_mode:
                parts.append("(flesh pussy:1.2), (organic torso:1.1)")
        elif suit:
            parts.append(suit)
        else:
            top = combine(kwargs["17_上装颜色"], kwargs["16_上装款式"], material_tag)
            bottom = combine(kwargs["19_下装颜色"], kwargs["18_下装款式"], material_tag)

            if top:
                parts.append(top)
            if bottom:
                parts.append(bottom)
            # 已删除：默认模式下的 (wearing clothes:1.1) 逻辑

        parts.append(wrap(extract(kwargs["29_手部饰品"])))

        # ==========================================
        # 6. 腿脚饰品
        # ==========================================
        parts.append(wrap(extract(kwargs["21_腿部特征"])))
        parts.append(combine(kwargs["23_袜饰颜色"], kwargs["22_袜饰"]))
        parts.append(combine(kwargs["25_鞋子颜色"], kwargs["24_鞋子"]))
        parts.append(wrap(extract(kwargs["30_特殊饰品"])))

        # ==========================================
        # 7. 构图与补充
        # ==========================================
        shot_tag = extract(kwargs["31_构图范围"])
        if shot_tag:
            parts.append(f"({shot_tag}:1.1)")

        manual = kwargs.get("33_手动补充", "").strip()
        if manual:
            parts.append(manual)

        # ==========================================
        # 输出
        # ==========================================
        final_prompt = ", ".join(p for p in parts if p)
        return (final_prompt,)


# ==========================================
# 🔌 节点注册
# ==========================================

NODE_CLASS_MAPPINGS = {
    "QianhuiCharacterCreator": QianhuiCharacterCreator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QianhuiCharacterCreator": "千绘 · 角色定制",
}
