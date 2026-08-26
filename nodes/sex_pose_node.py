"""
千绘 · 姿势模块 (姿势大师 QianhuiSexMaster)

模块职责：从 configs/ 目录加载姿势分类与配套动作数据池，按智能随机/
手动指定两种模式组合主姿势、视角、手部/足部动作、表情、液体与外观细节，
智能区分插入/非插入行为并附加对应标签。
模型类型: SDXL 用 :1.1/:1.2/:1.25 权重习惯; anima 按官方权重规范 2~5,
只给核心姿势与行为加权(最多2个), 细节/外观/基础标签去权重纯标签,
避免权重标签超过 anima 规范上限(最多4个)。
数据流：configs/*.json -> 数据池 -> 选择/随机 -> 行为判定 -> 组合输出 STRING。
外部依赖：无（仅标准库）；ConfigLoader 带 mtime 缓存热更新。
"""

import random
import re
import time
import json
import os
import glob

# ==========================================
# 🔞 性爱姿势大师 V40.0 · 性能优化版
# ==========================================
# 修复内容:
# - 修复 parts 变量未初始化的致命bug
# - 修复 val_pussy_select/val_hair_select/manual_text 变量未定义bug
# - 优化关键词匹配性能（预编译集合）
# - 优化正则表达式（预编译）
# - 精简重复代码
# ==========================================


class ConfigLoader:
    """配置加载器 - 带缓存和热更新"""

    _data_cache = {}
    _poses_all_cache = ["(不指定)"]
    _last_load_time = 0
    # 节点位于 nodes/ 子目录, 数据在插件根目录
    _config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")

    _pose_categories = [
        "poses_lying", "poses_kneeling", "poses_ontop", "poses_standing",
        "poses_oral", "poses_paizuri", "poses_feet", "poses_anal",
        "poses_special", "poses_showing"
    ]

    @classmethod
    def get_latest_mtime(cls):
        if not os.path.exists(cls._config_dir):
            return 0
        files = glob.glob(os.path.join(cls._config_dir, "*.json"))
        if not files:
            return 0
        return max(os.path.getmtime(f) for f in files)

    @classmethod
    def ensure_data(cls):
        current_mtime = cls.get_latest_mtime()
        if current_mtime > cls._last_load_time or not cls._data_cache:
            cls._load_from_disk(current_mtime)
        return cls._data_cache, cls._poses_all_cache

    @classmethod
    def _load_from_disk(cls, current_mtime):
        data = {}
        keys_to_ensure = cls._pose_categories + [
            "hand_actions", "foot_actions", "motion",
            "views", "expressions", "fluids",
            "pussy_types", "pubic_hair"
        ]
        for key in keys_to_ensure:
            data[key] = ["(不指定)"]

        if not os.path.exists(cls._config_dir):
            try:
                os.makedirs(cls._config_dir)
            except Exception:
                pass
            cls._data_cache = data
            cls._poses_all_cache = ["(不指定)"]
            return

        loaded_count = 0
        json_files = glob.glob(os.path.join(cls._config_dir, "*.json"))

        for path in json_files:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                    for k, v in content.items():
                        if k in data and isinstance(data[k], list) and isinstance(v, list):
                            if data[k] == ["(不指定)"]:
                                data[k] = v
                            else:
                                # 使用集合去重，性能更好
                                existing = set(data[k])
                                data[k].extend(x for x in v if x not in existing)
                        else:
                            data[k] = v
                    loaded_count += 1
            except Exception as e:
                print(f"❌ 读取配置文件 {os.path.basename(path)} 失败: {e}")

        poses_all = ["(不指定)"]
        for cat in cls._pose_categories:
            # 合并时跳过各池自带的 "(不指定)" 占位项, 避免下拉框出现 10+ 个重复项
            for item in data.get(cat, []):
                if "(不指定)" not in item:
                    poses_all.append(item)

        cls._data_cache = data
        cls._poses_all_cache = poses_all
        cls._last_load_time = current_mtime
        print(f"✅ 千绘姿势大师 V40.0: 已成功加载 {loaded_count} 个配置文件")


class QianhuiSexMaster:
    """千绘 姿势生成节点"""

    # 预编译正则表达式（性能优化）
    _EXTRACT_PATTERN = re.compile(r'\((.*?)\)')

    # 预定义非插入行为关键词集合（性能优化：O(1)查找）
    _NON_PENETRATIVE_KEYWORDS = frozenset([
        # 乳交类
        "paizuri", "titfuck", "breast_sex", "乳交", "squeezing",
        # 足交类
        "footjob", "feet_sex", "足交", "soles", "foot",
        # 手交类
        "handjob", "manual_stimulation", "手交", "fingering", "rubbing", "stroking",
        # 磨豆腐类
        "tribadism", "scissoring", "grinding", "磨豆腐",
        # 展示类
        "showing", "spreading", "presenting", "展示", "exposing", "lift", "open",
        # 口交类
        "oral", "fellatio", "blowjob", "cunnilingus", "lick", "suck", "mouth", "bj", "口交", "深喉", "throat",
        # 其他
        "69", "six_nine", "facesitting", "face_sitting", "on_face", "smothering"
    ])

    # 行为类型关键词映射（用于智能匹配, 仅匹配英文 tag）
    # 注意: 关键词必须精确, 宽泛词(feet/breast/oral/lick/tongue/fingering)会误判,
    # 如"feet on bed"配足交、"breast smother"配乳交——真条目均自带精确词, 不受影响
    _BEHAVIOR_KEYWORDS = {
        "paizuri": ["paizuri", "titfuck"],
        "footjob": ["footjob", "soles", "feet_sex"],
        "handjob": ["handjob", "manual_stimulation"],
        "fellatio": ["fellatio", "blowjob", "mouth"],
        "cunnilingus": ["cunnilingus"],
        "showing": ["showing", "exposed", "display"],
    }

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        current_data, current_poses = ConfigLoader.ensure_data()

        return {"required": {
            "模式选择": (["🎲 智能随机模式 (逻辑锁V4)", "🔒 手动指定模式"],),
            "全局权重": ("FLOAT", {"default": 1.25, "min": 0.5, "max": 2.0, "step": 0.05}),

            "主姿势": (current_poses,),
            "视角": (current_data.get("views", ["(不指定)"]),),
            "动作强度": (current_data.get("motion", ["(不指定)"]),),
            "👋 手部动作": (current_data.get("hand_actions", ["(不指定)"]),),
            "🦶 足部细节": (current_data.get("foot_actions", ["(不指定)"]),),
            "面部表情": (current_data.get("expressions", ["(不指定)"]),),
            "液体细节": (current_data.get("fluids", ["(不指定)"]),),

            "🍑 下体外观": (current_data.get("pussy_types", ["(不指定)"]),),
            "🎲 随机包含外观": ("BOOLEAN", {"default": True}),

            "🌲 毛发样式": (current_data.get("pubic_hair", ["(不指定)"]),),
            "🎲 随机包含毛发": ("BOOLEAN", {"default": True}),

            "模型类型": (["SDXL", "anima (权重2~5)"],),
            "手动补充": ("STRING", {"default": "", "multiline": True}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("NSFW_Prompt",)
    FUNCTION = "run"
    CATEGORY = "千绘/姿势"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        mode = kwargs.get("模式选择", "")
        if "随机" in mode:
            return float(time.time())
        return False

    def _extract(self, text: str) -> str:
        """从文本中提取括号内的内容"""
        if not text or not isinstance(text, str) or "(不指定)" in text:
            return ""
        matches = self._EXTRACT_PATTERN.findall(text)
        if matches:
            return matches[-1]
        return text

    def _find_match(self, keyword: str, source_list: list) -> str:
        """在列表中查找包含关键词的项"""
        if not keyword or not source_list:
            return ""
        for item in source_list:
            if isinstance(item, str) and keyword in item:
                return item
        return ""

    def _check_penetrative(self, pose_text: str) -> bool:
        """判断是否为插入式性行为（优化版：使用集合查找）"""
        if not pose_text or not isinstance(pose_text, str) or "(不指定)" in pose_text:
            return False

        p_lower = pose_text.lower()

        # 使用集合的O(1)查找，而不是遍历列表
        for keyword in self._NON_PENETRATIVE_KEYWORDS:
            if keyword in p_lower:
                return False

        return True

    def _detect_behavior_type(self, pose_text: str, tag_pose: str) -> str:
        """检测具体行为类型（仅基于英文 tag, 避免中文标题"口交/乳交"前缀误伤）"""
        tag_lower = tag_pose.lower() if tag_pose else ""
        if not tag_lower:
            return ""

        for behavior_type, keywords in self._BEHAVIOR_KEYWORDS.items():
            if any(kw in tag_lower for kw in keywords):
                return behavior_type

        return ""

    def _safe_random_choice(self, data: dict, key: str, prob: float, is_random_mode: bool, filter_func=None) -> str:
        """安全的随机选择；filter_func 可选按条目内容过滤（如排除 gore）"""
        if not is_random_mode:
            return ""
        lst = data.get(key, [])
        if not lst:
            return ""
        if filter_func:
            lst = [x for x in lst if filter_func(x)]
            if not lst:
                return ""
        if random.random() < prob:
            return random.choice(lst)
        return ""

    def run(self, **kw):
        try:
            DATA, _ = ConfigLoader.ensure_data()

            # ==========================================
            # 🛡️ 变量获取（修复：确保所有变量都有定义）
            # ==========================================
            mode = kw.get("模式选择", "🎲 智能随机模式 (逻辑锁V4)")
            w = kw.get("全局权重", 1.25)
            is_random_mode = "随机" in mode
            # 模型类型: anima 权重规范 2~5 起步, 全局权重低于 2 时提升到 2;
            # 权重标签少而精(官方规范最多 4 个), 细节/外观类去权重纯标签
            is_anima = "anima" in kw.get("模型类型", "SDXL")
            w_eff = max(w, 2.0) if is_anima else w

            selected_pose_full = kw.get("主姿势", "")
            selected_view_full = kw.get("视角", "")
            selected_hand_full = kw.get("👋 手部动作", "")
            selected_foot_full = kw.get("🦶 足部细节", "")

            # 外观选项
            opt_pussy = kw.get("🎲 随机包含外观", True)
            opt_hair = kw.get("🎲 随机包含毛发", True)

            # 🔧 修复：这些变量之前未定义
            val_pussy_select = kw.get("🍑 下体外观", "")
            val_hair_select = kw.get("🌲 毛发样式", "")
            manual_text = kw.get("手动补充", "").strip()

            # ==========================================
            # 🧠 随机逻辑
            # ==========================================
            pose_cat = ""

            if is_random_mode:
                roll = random.random()

                # 定义概率分布和对应的姿势池
                # 乳交/足交/肛交/特殊独立成档，避免原合并池(59项挤15%)素材利用率过低
                probability_map = [
                    (0.18, "poses_lying", "lying"),
                    (0.36, "poses_kneeling", "kneeling"),
                    (0.50, "poses_ontop", "ontop"),
                    (0.60, "poses_standing", "standing"),
                    (0.70, "poses_oral", "oral"),
                    (0.80, "poses_showing", "showing"),
                    (0.85, "poses_paizuri", "paizuri"),
                    (0.90, "poses_feet", "feet"),
                    (0.95, "poses_anal", "anal"),
                ]

                for threshold, pool_key, cat_name in probability_map:
                    if roll < threshold:
                        # 过滤"(不指定)"占位项, 避免随机抽到空姿势
                        pool = [x for x in DATA.get(pool_key, []) if "(不指定)" not in x]
                        if pool:
                            selected_pose_full = random.choice(pool)
                            pose_cat = cat_name
                            break

                # 如果没有命中前面的概率 (0.95~1.0)，使用特殊池
                if not pose_cat:
                    pool = [x for x in DATA.get("poses_special", []) if "(不指定)" not in x]
                    if pool:
                        selected_pose_full = random.choice(pool)
                        pose_cat = "special"

                # 配套动作
                self._assign_matching_actions(
                    pose_cat, selected_pose_full, DATA,
                    lambda v: setattr(self, '_view', v),
                    lambda h: setattr(self, '_hand', h),
                    lambda f: setattr(self, '_foot', f)
                )

                # 应用配套动作
                if hasattr(self, '_view'):
                    selected_view_full = self._find_match(self._view, DATA.get("views", []))
                    delattr(self, '_view')
                if hasattr(self, '_hand'):
                    selected_hand_full = self._find_match(self._hand, DATA.get("hand_actions", []))
                    delattr(self, '_hand')
                if hasattr(self, '_foot'):
                    selected_foot_full = self._find_match(self._foot, DATA.get("foot_actions", []))
                    delattr(self, '_foot')

            # --- 提取 Tag ---
            tag_pose = self._extract(selected_pose_full)
            tag_view = self._extract(selected_view_full)
            tag_hand = self._extract(selected_hand_full)
            tag_foot = self._extract(selected_foot_full)

            # 辅助元素处理
            i_motion = kw.get("动作强度", "")
            i_fluid = kw.get("液体细节", "")
            i_expr = kw.get("面部表情", "")

            # 运动池专用动作(骑乘摇摆/后入猛撞/深喉抽送)按姿势类别过滤，
            # 避免"正常位+深喉抽送""口交+骑乘摇摆"类跨姿势冲突
            motion_exclude = ["cowgirl bouncing", "ass slapping", "deepthroat"]
            if pose_cat in ("oral", "paizuri"):
                motion_exclude.remove("deepthroat")          # 口交/乳交保留深喉类动作
            if pose_cat == "ontop":
                motion_exclude.remove("cowgirl bouncing")    # 骑乘保留骑乘摇摆
            if pose_cat in ("kneeling", "anal"):
                motion_exclude.remove("ass slapping")        # 后入/肛交保留后入猛撞
            motion_filter = (lambda x: not any(w in x for w in motion_exclude)) if motion_exclude else None

            val_motion = i_motion if (not is_random_mode or i_motion) else self._safe_random_choice(DATA, "motion", 0.5, is_random_mode, motion_filter)
            val_fluid = i_fluid if (not is_random_mode or i_fluid) else self._safe_random_choice(DATA, "fluids", 0.6, is_random_mode, lambda x: "gore" not in x)
            val_expr = i_expr if (not is_random_mode or i_expr) else self._safe_random_choice(DATA, "expressions", 0.8, is_random_mode)

            tag_motion = self._extract(val_motion)
            tag_fluid = self._extract(val_fluid)
            tag_expr = self._extract(val_expr)

            # ==========================================
            # 🛡️ 智能组装
            # ==========================================

            # 🔧 修复：初始化 parts 列表（这是之前的致命bug）
            parts = []

            # 1. 基础标签 (anima 下纯标签, 权重留给核心姿势与行为)
            if is_anima:
                base_tags = ["nsfw", "rating:explicit", "uncensored", "best quality", "correct anatomy"]
            else:
                base_tags = [
                    "(nsfw:1.2)",
                    "(rating:explicit:1.2)",
                    "(uncensored:1.1)",
                    "(best quality:1.1)",
                    "(correct anatomy:1.1)"
                ]
            parts.extend(base_tags)

            # 2. 姿势本体 (anima 权重 2 起步, 由 w_eff 控制)
            if tag_pose:
                parts.append(f"({tag_pose}:{w_eff:g})")

            # 3. 核心行为分支
            is_penetration = self._check_penetrative(selected_pose_full)

            if is_penetration:
                # 检测肛交姿势，使用对应标签，避免"肛交配 vaginal sex"
                if "anal" in selected_pose_full.lower():
                    parts.append(f"(sex, intercourse, anal sex, penetration:{'2' if is_anima else '1.1'})")
                    parts.append(f"(detailed anus, detailed penis:{'2' if is_anima else '1.1'})")
                else:
                    parts.append(f"(sex, intercourse, vaginal sex, penetration:{'2' if is_anima else '1.1'})")
                    parts.append(f"(detailed pussy, detailed penis:{'2' if is_anima else '1.1'})")
            else:
                # 非插入行为：智能匹配具体类型
                behavior = self._detect_behavior_type(selected_pose_full, tag_pose)

                if is_anima:
                    behavior_tags = {
                        "paizuri": ["(paizuri, breast sex, squeezing breasts:2)", "detailed penis"],
                        "footjob": ["(footjob, feet sex, soles:2)"],
                        "handjob": ["(handjob, penis grab:2)"],
                        "fellatio": ["(fellatio, blowjob, oral sex, cheek bulge:2)"],
                        "cunnilingus": ["(cunnilingus, oral sex:2)"],
                        "showing": ["nudity, exposed, detailed pussy"],
                    }
                else:
                    behavior_tags = {
                        "paizuri": ["(paizuri, breast sex, squeezing_breasts:1.2)", "(detailed penis:1.1)"],
                        "footjob": ["(footjob, feet sex, soles:1.2)"],
                        "handjob": ["(handjob, penis_grab:1.2)"],
                        "fellatio": ["(fellatio, blowjob, oral sex, cheek_bulge:1.2)"],
                        "cunnilingus": ["(cunnilingus, oral sex:1.2)"],
                        "showing": ["(nudity, exposed, detailed pussy:1.1)"],
                    }

                if behavior in behavior_tags:
                    parts.extend(behavior_tags[behavior])

            # 4. 其他细节
            detail_tags = [
                (tag_view, w),
                (tag_hand, w),
                (tag_foot, w),
                (tag_motion, w),
                (tag_expr, w),
                (tag_fluid, w),
            ]

            for tag, weight in detail_tags:
                if tag:
                    # anima 下细节为纯标签(无括号无权重, 避免被解析为括号标签/权重语法)
                    parts.append(tag if is_anima else f"({tag}:{weight})")

            # ==========================================
            # 5. 外观特征
            # ==========================================
            tag_pussy_final = ""
            tag_hair_final = ""

            if "手动" in mode:
                # 手动模式：使用下拉菜单的值
                tag_pussy_final = self._extract(val_pussy_select)
                tag_hair_final = self._extract(val_hair_select)
            else:
                # 随机模式：根据开关决定是否随机
                if opt_pussy:
                    pt_list = DATA.get('pussy_types', [])
                    if pt_list:
                        tag_pussy_final = self._extract(random.choice(pt_list))

                if opt_hair:
                    ph_list = DATA.get('pubic_hair', [])
                    if ph_list:
                        tag_hair_final = self._extract(random.choice(ph_list))

            if tag_pussy_final:
                parts.append(tag_pussy_final if is_anima else f"({tag_pussy_final}:{w})")
            if tag_hair_final:
                parts.append(tag_hair_final if is_anima else f"({tag_hair_final}:{w})")

            # 6. 手动补充
            if manual_text:
                parts.append(manual_text)

            return (", ".join(parts),)

        except Exception as e:
            import traceback
            print(f"❌ 千绘姿势大师 运行时错误: {e}")
            print(traceback.format_exc())
            return (f"ERROR: {str(e)}",)

    def _assign_matching_actions(self, pose_cat: str, pose_full: str, data: dict,
                                  set_view, set_hand, set_foot):
        """根据姿势类别分配配套动作"""
        if pose_cat in ["lying", "kneeling", "ontop", "standing"]:
            views_pool = data.get("views", [])
            if views_pool:
                set_view(random.choice(views_pool))
            set_hand(random.choice(["搂紧脖子", "抓紧床单", "托起乳房", "引导插入"]))
            # 姿势本体已描述腿部形态(抱头/架肩/盘腰/抬腿等)时,
            # 只给兼容性最高的"蜷趾高潮"(脚趾细节与腿部形态无关), 避免"踝抱头+盘腰"类冲突
            pose_lower = pose_full.lower()
            leg_shaped = any(w in pose_lower for w in [
                "ankles behind head", "legs on shoulders", "legs wrapped", "pinned legs",
                "legs up", "one leg up", "leg lifted", "raised leg", "legs hanging",
                "holding knees", "high split", "feet on bed",
            ])
            if leg_shaped:
                set_foot("蜷趾高潮")
            else:
                set_foot(random.choice(["盘在腰上", "双腿架肩", "蜷趾高潮"]))
        elif pose_cat == "oral":
            # 匹配口交专用视角条目(views 池"跪姿仰视"), 而非第一条 POV
            set_view("跪姿仰视")
            set_hand("抓乱头发")
        elif pose_cat == "showing":
            set_view("特写")
            set_hand("双手掰穴")
        elif pose_cat in ("special", "feet", "paizuri"):
            if "足交" in pose_full:
                set_foot("展示脚底")   # foot_actions 池实际条目名
            if "乳交" in pose_full:
                set_hand("托起乳房")
