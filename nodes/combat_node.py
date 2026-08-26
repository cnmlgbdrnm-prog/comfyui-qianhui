"""
千绘 · 战斗模块 (战斗指挥官 QianhuiCombatMaster)

模块职责：从 combat_data/ 目录加载武器/攻击姿态/战损/特效/运镜数据池，
支持随机激战与手动指令两种模式，按 氛围-运镜-武器-招式-特效-战损 顺序组装战斗提示词。
武器-招式联动: WEAPON_ACTION_GROUPS 按武器分组抽取, 避免"弓箭+武士刀招式"错配。
模型类型: SDXL 用 :1.2/:1.1 权重习惯; anima 按官方权重规范 2~5,
只给氛围与武器加权(最多2个), 特效/保底运镜去权重纯标签。
数据流：combat_data/*.json -> 选项/随机抽取 -> 提取英文 Tag -> 拼接输出 STRING。
外部依赖：无（仅标准库）。
"""

import random
import re
import json
import os
import time

# ==========================================
# ⚔️ 博士的战斗指挥官 (Combat Master V1)
# 专用于生成高张力战斗、特效、武器交互
# ==========================================

# 路径定义 (节点位于 nodes/ 子目录, 数据在插件根目录)
CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(CURRENT_DIR, "combat_data")

# 武器与招式组映射: weapons 条目中文名 -> actions 条目编号前缀
# (actions 池按武器分组, 编号 01~19, 随机模式需联动抽取避免"弓箭+武士刀招式"错配)
WEAPON_ACTION_GROUPS = {
    "武士刀": "01", "巨剑": "02", "巨型镰刀": "03", "长枪": "04", "弓箭": "05",
    "双匕首": "06", "双枪": "07", "重型火炮": "08", "魔法法杖": "09", "机械拳套": "10",
    "狙击枪": "11", "锁链": "12", "软鞭": "13", "巨型战锤": "14", "龙之利爪": "15",
    "中国剑": "16", "陌刀": "17", "中国刀": "18", "护手双钩": "19",
}

# 键名与文件名的映射关系 (战斗特化版)
FILE_MAPPING = {
    "weapons": "01_weapons.json",
    "actions": "02_actions.json",
    "damage": "03_damage.json",
    "vfx": "04_vfx.json",
    "camera": "05_camera.json"
}

def load_combat_data():
    """读取 combat_data 文件夹下的 JSON"""
    data = {
        "weapons": ["(不指定)", "⚔️  katana (武士刀)"],
        "actions": ["(不指定)", "👊 punching (出拳)"],
        "damage": ["(不指定)", "✨ pristine (无伤)"],
        "vfx": ["(不指定)"],
        "camera": ["(不指定)"]
    }

    if not os.path.exists(DATA_DIR):
        print(f"⚠️ [千绘Combat] 警告: 找不到文件夹 {DATA_DIR}，请创建并放入JSON文件。")
        return data

    for key, filename in FILE_MAPPING.items():
        file_path = os.path.join(DATA_DIR, filename)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                    if isinstance(content, list):
                        data[key] = content
            except Exception as e:
                 print(f"❌ [千绘Combat] 读取失败: {filename} - {e}")
    return data

# 初始化加载
COMBAT_DATA = load_combat_data()

class QianhuiCombatMaster:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        global COMBAT_DATA
        COMBAT_DATA = load_combat_data()

        return {
            "required": {
                "00_🔥激战模式": ("BOOLEAN", {"default": False, "label_on": "🔥 随机激战 (Chaos)", "label_off": "🔒 手动指令 (Tactical)"}),
                "01_🗡️武器装备": (COMBAT_DATA["weapons"],),
                # 过滤 "---" 分组标题行, 手动模式下也不可选中
                "02_👊攻击姿态": ([x for x in COMBAT_DATA["actions"] if not x.startswith("---")],),
                "03_🩸战损程度": (COMBAT_DATA["damage"],),
                "04_✨技能特效": (COMBAT_DATA["vfx"],),
                "05_📸战斗运镜": (COMBAT_DATA["camera"],),
                "07_模型类型": (["SDXL", "anima (权重2~5)"],),
                "06_补充指令": ("STRING", {"default": "", "multiline": False, "placeholder": "如: explosion background, glowing eyes"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Combat_Prompt",)
    FUNCTION = "generate_combat"
    CATEGORY = "千绘/战斗"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        if kwargs.get("00_🔥激战模式", False):
            return time.time()
        return False

    def generate_combat(self, **kwargs):
        global COMBAT_DATA

        # --- 工具函数 ---
        def extract(text):
            if not text or "(不指定)" in text: return ""
            # 提取括号内的英文 tag
            matches = re.findall(r'\((.*?)\)', text)
            valid_tags = [m.strip() for m in matches if re.search(r"[A-Za-z]", m)]
            return ", ".join(valid_tags) if valid_tags else ""

        def get_random(key):
            """从数据池中随机抽取一个非空的选项 (排除 "(不指定)" 占位与 "---" 分组标题行)"""
            pool = [x for x in COMBAT_DATA[key]
                    if "(不指定)" not in x and not x.startswith("---")]
            if not pool: return ""
            return extract(random.choice(pool))

        def get_random_full(key):
            """随机抽取完整条目(含中文标题), 供武器-招式组联动使用"""
            pool = [x for x in COMBAT_DATA[key]
                    if "(不指定)" not in x and not x.startswith("---")]
            if not pool: return ""
            return random.choice(pool)

        # --- 1. 数据准备 ---
        is_random = kwargs["00_🔥激战模式"]
        # 模型类型: anima 权重规范 2~5, 只给氛围与武器加权(最多2个),
        # 特效/保底运镜去权重纯标签, 符合 anima"权重少而精"规范
        is_anima = "anima" in kwargs.get("07_模型类型", "SDXL")

        # 定义槽位
        slot_camera = ""
        slot_weapon = ""
        slot_action = ""
        slot_vfx = ""
        slot_damage = ""

        if is_random:
            # === 随机模式 (Chaos Mode) ===
            # 武器-招式联动: 先抽武器完整条目(含中文名), 招式从该武器对应组中抽取
            weapon_full = get_random_full("weapons")
            slot_weapon = extract(weapon_full)

            action_group = ""
            for wname, group_id in WEAPON_ACTION_GROUPS.items():
                if wname in weapon_full:
                    action_group = group_id
                    break
            if action_group:
                group_pool = [x for x in COMBAT_DATA["actions"]
                              if not x.startswith("---")
                              and x.startswith(f"{action_group}.")]
                slot_action = extract(random.choice(group_pool)) if group_pool else ""
            else:
                slot_action = get_random("actions")

            slot_damage = get_random("damage") if random.random() > 0.5 else "" # 50%概率战损
            slot_vfx = get_random("vfx")
            slot_camera = get_random("camera")

            # 随机模式下，如果没有随机到运镜，给一个保底的动态运镜
            if not slot_camera:
                slot_camera = "motion blur, depth of field, dynamic angle" if is_anima else "motion blur, depth of field, (dynamic angle:1.3)"
        else:
            # === 手动模式 (Tactical Mode) ===
            slot_weapon = extract(kwargs["01_🗡️武器装备"])
            slot_action = extract(kwargs["02_👊攻击姿态"])
            slot_damage = extract(kwargs["03_🩸战损程度"])
            slot_vfx = extract(kwargs["04_✨技能特效"])
            slot_camera = extract(kwargs["05_📸战斗运镜"])

        # --- 2. 核心优化：按逻辑顺序组装 ---
        # 目标顺序：视角 -> 武器 -> 招式 -> 特效 -> 战损

        ordered_parts = []

        # [0] 基础氛围 (Global Context) - 永远在最前
        if is_anima:
            ordered_parts.append("(combat mode:2), dynamic pose, intense atmosphere")
        else:
            ordered_parts.append("(combat mode:1.2), dynamic pose, intense atmosphere")

        # [1] 视角 (View/Camera) - 决定构图
        if slot_camera: ordered_parts.append(slot_camera)

        # [2] 武器 (Weapon) - 决定主体持有的物品
        # 增加权重，确保武器不消失 (anima 下同为 2 起步)
        if slot_weapon: ordered_parts.append(f"({slot_weapon}:{'2' if is_anima else '1.2'})")

        # [3] 招式 (Action) - 紧跟武器，确保是"用这个武器"做的动作
        if slot_action: ordered_parts.append(slot_action)

        # [4] 特效 (VFX) - 动作产生的光影/粒子
        # SDXL 加权让特效更明显; anima 下纯标签(权重留给氛围与武器)
        if slot_vfx: ordered_parts.append(slot_vfx if is_anima else f"({slot_vfx}:1.1)")

        # [5] 战损 (Damage) - 状态叠加
        if slot_damage: ordered_parts.append(slot_damage)

        # [6] 手动补充指令 (Manual Add) - 最后补充细节
        manual_add = kwargs.get("06_补充指令", "").strip()
        if manual_add: ordered_parts.append(manual_add)

        # --- 3. 生成最终字符串 ---
        final_prompt = ", ".join([p for p in ordered_parts if p])

        return (final_prompt,)
