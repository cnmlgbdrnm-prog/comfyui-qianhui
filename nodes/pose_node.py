"""
千绘 · 动作模块 (肢体动作师 QianhuiActionMaster)

模块职责：从 pose_data/ 目录读取视角/全身/头部/左右手/左右腿/表情等 JSON
数据池，支持随机模式与手动模式，按 视角-全身-头-手-腿-表情 顺序组装动作提示词。
模型类型: SDXL 用 :1.2 轻强调; anima 按官方权重规范 2~5 (核心动作 :2,
细节/表情不加权, 权重少而精最多4个)。标签均为小写空格形式, 符合 anima 标签规范。
数据流：pose_data/*.json -> 选项/随机抽取 -> 提取英文 Tag -> 按模型类型加权拼接。
外部依赖：无（仅标准库）。
"""

import random
import re
import json
import os
import time

# ==========================================
# 💃 博士的动作姿态数据库 (V8.0 Split-Limb)
# 更新日志:
# 1. 移除全局左右侧重，改为精准的 左/右/双 独立控制
# 2. 优化输出顺序逻辑：视角-全身-手-脚-表情
# 3. 随机模式下增加 单侧/双侧 互斥逻辑
# ==========================================

# 路径定义 (节点位于 nodes/ 子目录, 数据在插件根目录)
CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(CURRENT_DIR, "pose_data")

# 键名与文件名的映射关系 (新版拆分结构)
# 注意：hands_left/hands_right、legs_left/legs_right 为完全镜像池，
# 修改一侧时必须同步修改另一侧，保持条目一一对应。
FILE_MAPPING = {
    "global": "01_global.json",
    "head": "02_head.json",
    "view": "05_view.json",
    "expressions": "08_expressions.json",
    # 手部拆分
    "hands_dual": "03_hands_dual.json",   # 双手动作
    "hands_left": "03_hands_left.json",   # 左手动作
    "hands_right": "03_hands_right.json", # 右手动作
    # 腿部拆分
    "legs_dual": "04_legs_dual.json",     # 双腿/整体腿部
    "legs_left": "04_legs_left.json",     # 左腿
    "legs_right": "04_legs_right.json"    # 右腿
}

def load_pose_data():
    """读取 pose_data 文件夹下的多个 JSON 文件"""

    # 默认兜底数据，防崩溃
    data = {key: ["(不指定)"] for key in FILE_MAPPING.keys()}
    # 给global加个默认值
    data["global"] = ["(不指定)", "🧍 站立 (standing)"]

    if not os.path.exists(DATA_DIR):
        print(f"⚠️ [千绘Pose] 警告: 找不到文件夹 {DATA_DIR}")
        return data

    # 遍历映射表读取文件
    for key, filename in FILE_MAPPING.items():
        file_path = os.path.join(DATA_DIR, filename)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                    if isinstance(content, list):
                        data[key] = content
                    else:
                        print(f"❌ [千绘Pose] 格式错误: {filename} 应该是一个列表 []")
            except Exception as e:
                 print(f"❌ [千绘Pose] 读取失败: {filename} - {e}")
        else:
             # 静默或提示缺失，V8版本文件较多，提示一下方便用户补全
             print(f"ℹ️ [千绘Pose] 未找到 {filename}，该部位将无选项")

    return data

# 初始化加载
POSE_DATA = load_pose_data()

class QianhuiActionMaster:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        # 运行时重新加载，方便热更
        global POSE_DATA
        POSE_DATA = load_pose_data()

        return {
            "required": {
                "00_🎲随机动作模式": ("BOOLEAN", {"default": False, "label_on": "🎲 智能随机 (Auto)", "label_off": "🔒 手动指定 (Manual)"}),
                "07_🎲随机表情开关": ("BOOLEAN", {"default": False, "label_on": "🤪 随机表情 (ON)", "label_off": "😐 锁定表情 (OFF)"}),

                # 基础层
                "05_特殊视角": (POSE_DATA["view"],),
                "01_全身基础": (POSE_DATA["global"],),
                "02_头部/视线": (POSE_DATA["head"],),

                # 手部层 (拆分)
                "03_👐双手动作": (POSE_DATA["hands_dual"],),
                "03_👈左手动作": (POSE_DATA["hands_left"],),
                "03_👉右手动作": (POSE_DATA["hands_right"],),

                # 腿部层 (拆分)
                "04_🦵双脚动作": (POSE_DATA["legs_dual"],),
                "04_🦵左脚动作": (POSE_DATA["legs_left"],),
                "04_🦵右脚动作": (POSE_DATA["legs_right"],),

                # 细节层
                "08_表情细节": (POSE_DATA["expressions"],),
                "06_手动补充": ("STRING", {"default": "", "multiline": False, "placeholder": "额外补充 tag..."}),
                "09_模型类型": (["SDXL", "anima (权重2~5)"],),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Pose_Prompt",)
    FUNCTION = "generate_pose_v8"
    CATEGORY = "千绘/动作"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        # 随机动作模式 或 随机表情开关 任一开启时每次强制刷新，
        # 否则 ComfyUI 会缓存结果导致"随机"不生效
        if kwargs.get("00_🎲随机动作模式", False) or kwargs.get("07_🎲随机表情开关", False):
            return time.time()
        return False

    def generate_pose_v8(self, **kwargs):
        global POSE_DATA

        # --- 工具函数 ---
        def extract(text):
            """提取括号内英文"""
            if not text or "(不指定)" in text: return ""
            matches = re.findall(r'\((.*?)\)', text)
            for m in matches:
                if re.search(r"[A-Za-z]", m): return m.strip()
            return ""

        def pick_random(key_name, filter_func=None):
            """从指定key的池子里随机抽取一个非空项；filter_func 可选按条目标题过滤"""
            pool = [x for x in POSE_DATA.get(key_name, []) if "(不指定)" not in x]
            if filter_func:
                pool = [x for x in pool if filter_func(x)]
            if not pool: return ""
            return extract(random.choice(pool))

        # --- 初始化变量 ---
        parts = {
            "view": "",
            "global": "",
            "head": "",
            "hands": [],
            "legs": [],
            "expression": "",
            "manual": ""
        }

        is_auto = kwargs["00_🎲随机动作模式"]
        # 模型类型: SDXL 权重习惯 (:1.2 轻强调); anima 权重规范 2~5,
        # 核心动作 2 起步, 细节/表情不再加权(anima 官方规范: 权重少而精, 最多4个)
        is_anima = "anima" in kwargs.get("09_模型类型", "SDXL")
        w_global = ":2)" if is_anima else ":1.2)"

        # ===========================
        # 1. 视角 & 全身 (View & Global)
        # ===========================
        if is_auto:
            # 随机视角
            parts["view"] = pick_random("view")
            # 随机全身
            parts["global"] = pick_random("global")
            if not parts["global"]: parts["global"] = "standing"
        else:
            parts["view"] = extract(kwargs["05_特殊视角"])
            parts["global"] = extract(kwargs["01_全身基础"])

        # 简单的状态推断，用于辅助决策 (可选)
        g_lower = parts["global"].lower()

        # ===========================
        # 2. 手部动作 (Hands Logic)
        # ===========================
        if is_auto:
            # 逻辑：50%概率双手联动(dual)，50%概率左右手独立(split)
            # 如果全身动作已包含持物/抱/交叉等手部占用词，跳过随机手部避免叠加冲突
            # (jojo/praise 覆盖固定手位的梗动作，pocket/glove 覆盖手部占用，
            #  sleep/naruto 覆盖趴睡/忍者跑这类手部被占用的全身姿势)
            if not any(x in g_lower for x in ["cross", "hug", "reach", "hold",
                                               "drink", "eat", "brush", "cook", "book",
                                               "phone", "guitar", "violin", "cigarette",
                                               "pet", "gift", "dry",
                                               "spread_eagle", "lap_pillow",
                                               "jojo", "praise", "pocket", "glove",
                                               "sleep", "naruto", "game"]):
                if random.random() < 0.4:
                    # 抽取双手
                    h_dual = pick_random("hands_dual")
                    if h_dual: parts["hands"].append(h_dual)
                else:
                    # 分别抽取左右手
                    if random.random() < 0.7: # 70%概率有左手动作
                        h_left = pick_random("hands_left")
                        if h_left: parts["hands"].append(h_left)
                    if random.random() < 0.7: # 70%概率有右手动作
                        h_right = pick_random("hands_right")
                        if h_right: parts["hands"].append(h_right)
        else:
            # 手动模式：全部读取
            h_dual = extract(kwargs["03_👐双手动作"])
            h_left = extract(kwargs["03_👈左手动作"])
            h_right = extract(kwargs["03_👉右手动作"])

            if h_dual: parts["hands"].append(h_dual)
            if h_left: parts["hands"].append(h_left)
            if h_right: parts["hands"].append(h_right)

        # ===========================
        # 3. 腿部动作 (Legs Logic)
        # ===========================
        if is_auto:
            # 如果全身动作包含 sit, kneel, run, lying 等强腿部动作，跳过随机腿部
            # (lying/sleep/hang/prone/crawl 覆盖躺卧类，curl/float/fly/fall/slide
            #  覆盖蜷缩/漂浮/飞行/坠落/侧滑类，foot/step/stomp/one_leg 覆盖单脚平衡、
            #  调整鞋、抬脚逼近、踩踏类脚部占用，避免与独立腿部动作叠加冲突)
            if not any(x in g_lower for x in ["sit", "kneel", "squat", "run", "jump", "walk",
                                               "lying", "sleep", "hang", "prone", "crawl",
                                               "bridge", "arch", "curl", "float", "fly",
                                               "fall", "slide", "foot", "step", "stomp",
                                               "one_leg", "bending", "game"]):
                if random.random() < 0.3: # 腿部动作概率稍低
                    # 抽取双腿：进入此分支的全身均为站立类活动（站姿/生活/梗/视角类，
                    # 坐/躺/蹲/动/蜷类已被上方跳过词拦截），固定只抽站姿腿，
                    # 避免"弹吉他+坐姿抱膝"类跨姿势冲突
                    l_dual = pick_random("legs_dual", lambda x: "站姿" in x)
                    if l_dual: parts["legs"].append(l_dual)
                else:
                    # 分别抽取 (通常腿部左右独立动作较少，主要是抬腿/迈步)
                    # 过滤强动作(踩踏/膝撞/踩人): 踩踏类 global 已被上方跳过词拦截,
                    # 普通站姿下抽到强动作违和, 强动作仅保留在手动模式可选
                    strong_leg = ["stepping_on_viewer", "stomping", "knee_strike", "flying_kick"]
                    if random.random() < 0.5:
                        l_left = pick_random("legs_left",
                                             lambda x: not any(w in x for w in strong_leg))
                        if l_left: parts["legs"].append(l_left)
                    if random.random() < 0.5:
                        l_right = pick_random("legs_right",
                                              lambda x: not any(w in x for w in strong_leg))
                        if l_right: parts["legs"].append(l_right)
                    # 冲突防护：一侧为强单腿动作(踢/踩踏/踩脸/膝撞)时，只保留该侧
                    if len(parts["legs"]) == 2:
                        conflict_words = ["flying_kick", "stepping_on_viewer", "stomping", "knee_strike"]
                        if any(w in parts["legs"][0] for w in conflict_words) or \
                           any(w in parts["legs"][1] for w in conflict_words):
                            parts["legs"] = [parts["legs"][0]]
        else:
            # 手动模式
            l_dual = extract(kwargs["04_🦵双脚动作"])
            l_left = extract(kwargs["04_🦵左脚动作"])
            l_right = extract(kwargs["04_🦵右脚动作"])

            if l_dual: parts["legs"].append(l_dual)
            if l_left: parts["legs"].append(l_left)
            if l_right: parts["legs"].append(l_right)

        # ===========================
        # 4. 头部 & 表情 (Head & Expr)
        # ===========================
        if is_auto:
            # 全身为饮食/抽烟/刷牙等嘴部占用动作时，head 池排除嘴部动作避免"叼着糖喝奶茶"类冲突
            mouth_busy = any(x in g_lower for x in ["drink", "eat", "smoke", "cigarette", "brush"])
            if mouth_busy:
                mouth_items = ["toast", "cigarette", "lollipop", "hair_tie", "biting"]
                parts["head"] = pick_random("head", lambda x: not any(w in x for w in mouth_items))
            else:
                parts["head"] = pick_random("head")
        else:
            parts["head"] = extract(kwargs["02_头部/视线"])

        if kwargs["07_🎲随机表情开关"]:
            parts["expression"] = pick_random("expressions")
        else:
            parts["expression"] = extract(kwargs["08_表情细节"])

        if not parts["expression"]:
            # anima 下纯标签(权重1.0无意义), SDXL 保留权重写法
            parts["expression"] = "neutral expression" if is_anima else "(neutral expression:1.0)"

        # ===========================
        # 5. 组装输出 (Assembly)
        # ===========================
        # anima 规范: 标签用空格代替下划线 (score 系列除外, 本节点无 score 标签);
        # 手动补充为用户输入, 不转换, 由用户按模型习惯自行书写
        if is_anima:
            for k in ("view", "global", "head", "expression"):
                parts[k] = parts[k].replace("_", " ")
            parts["hands"] = [h.replace("_", " ") for h in parts["hands"]]
            parts["legs"] = [l.replace("_", " ") for l in parts["legs"]]

        parts["manual"] = kwargs.get("06_手动补充", "").strip()

        # 核心顺序: View -> Global -> Head -> Hands -> Legs -> Expression -> Manual
        final_list = []

        if parts["view"]: final_list.append(parts["view"])
        if parts["global"]: final_list.append(f"({parts['global']}{w_global}") # 核心动作加权
        if parts["head"]: final_list.append(parts["head"])

        # 扩展手脚列表
        final_list.extend(parts["hands"])
        final_list.extend(parts["legs"])

        final_list.append(parts["expression"])
        if parts["manual"]: final_list.append(parts["manual"])

        prompt_string = ", ".join(final_list)

        # 自动补充细节tag (anima 下纯标签不加权, 符合权重少而精规范)
        if any(x in prompt_string for x in ["hand", "finger", "hold", "reach"]):
            prompt_string += ", detailed fingers" if is_anima else ", (detailed fingers:1.1)"

        return (prompt_string,)
