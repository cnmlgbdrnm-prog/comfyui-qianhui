"""
千绘 · 剧情模块 (剧情编排机 + 剧情播放器)

模块职责：
- QianhuiStoryScheduler: 从 story_scripts 目录读取剧本 JSON 或手写文本，
  按自动连播/手动锁定模式逐段输出当前段 Tag 与页面引导。
- QianhuiStoryComposer: 将剧情/人设/全局设定/手动补充四路输入清洗、
  加权、去重后拼接为最终提示词。
数据流：story_scripts/*.json -> 段落列表 -> 当前段输出；四路 STRING -> Final_Prompt。
副作用：story_scripts 目录缺失时自动创建。
"""

import json
import os
import random
import re

# ==========================================
# 📂 路径配置 (节点位于 nodes/ 子目录, 数据在插件根目录)
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORY_DIR = os.path.join(CURRENT_DIR, "story_scripts")

if not os.path.exists(STORY_DIR):
    try:
        os.makedirs(STORY_DIR)
    except:
        pass

def get_story_files():
    if not os.path.exists(STORY_DIR): return ["(文件夹缺失)"]
    files = [f for f in os.listdir(STORY_DIR) if f.lower().endswith(".json")]
    return files if files else ["(无json文件)"]

# ==========================================
# 🛠️ 权重安全阀 (V9.1 优化版)
# ==========================================
def apply_safety_clamp(text_input):
    """
    1. 仅针对 (tag:number) 格式且 number > 1.5 的情况进行降权。
    2. 不再暴力删除括号，保留用户的原始分组习惯 ((tag)) 或 (tag)。
    3. 列表转字符串，换行转逗号。
    """
    if not text_input: return ""

    # 格式标准化
    if isinstance(text_input, list): text_input = ",".join([str(x) for x in text_input])
    if not isinstance(text_input, str): return str(text_input)

    # 基础清洗
    clean_text = text_input.replace("\n", ",")
    tags = [t.strip() for t in clean_text.split(',') if t.strip()]
    processed_tags = []

    # 正则：精准匹配 (内容:数字) 的格式
    # 避免误伤普通括号，如 (white dress)
    weight_pattern = re.compile(r"^\((.+):([0-9.]+)\)$")

    for tag in tags:
        match = weight_pattern.match(tag)
        if match:
            content = match.group(1)
            try:
                weight = float(match.group(2))
                # 逻辑：如果权重大于 1.5，强制降为 1.5
                if weight > 1.5:
                    processed_tags.append(f"({content}:1.5)")
                else:
                    # 权重安全或过小，保持原样 (保留原格式)
                    processed_tags.append(tag)
            except ValueError:
                # 数字解析失败，保持原样
                processed_tags.append(tag)
        else:
            # 没有命中权重格式（例如普通tag、带括号的组合tag），直接保留，不做任何修改
            processed_tags.append(tag)

    return ", ".join(processed_tags)

# ==========================================
# 节点 1: 剧情播放器 (V14·智能归零版)
# ==========================================
class QianhuiStoryScheduler:
    def __init__(self):
        # 初始化内部状态
        self._counter = 0
        self._last_file = "" # 记忆上一次选的文件名

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "外挂剧本选择": (get_story_files(),),
                "临时手写剧本": ("STRING", {"multiline": True, "default": "", "placeholder": "兼容手写模式..."}),
                "播放模式": (["自动连播 (内部计数)", "手动锁定 (指定页码)"],),
                "手动页码": ("INT", {"default": 1, "min": 1, "max": 9999}),
                "重置进度": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("当前段落(Tag)", "进度信息", "页面引导(Guide)")
    FUNCTION = "schedule_story"
    CATEGORY = "千绘/剧情"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("nan")

    def schedule_story(self, 外挂剧本选择, 临时手写剧本, 播放模式, 手动页码, 重置进度):
        # 1. 确保实例变量存在 (兼容热重载)
        if not hasattr(self, "_counter"): self._counter = 0
        if not hasattr(self, "_last_file"): self._last_file = ""

        # ★★★ 智能重置逻辑 (V14 核心) ★★★
        # 检查：如果当前选的文件 和 上次不一样
        if 外挂剧本选择 != self._last_file:
            print(f"🔄 [千绘 剧情V14] 检测到剧本切换: {self._last_file} -> {外挂剧本选择}")
            print("🔄 计数器已自动归零！")
            self._counter = 0          # 归零
            self._last_file = 外挂剧本选择 # 记住新文件

        # 手动强制重置
        if 重置进度:
            self._counter = 0

        # 2. 读取内容
        content_list = []
        loaded_from_file = False

        if 外挂剧本选择 and not 外挂剧本选择.startswith("("):
            file_path = os.path.join(STORY_DIR, 外挂剧本选择)
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        for item in data:
                            if isinstance(item, dict):
                                tag = item.get("tags", "")
                                guide = item.get("guide", "（无引导）")
                                content_list.append((tag, guide))
                            else:
                                content_list.append((str(item), "（旧版数据）"))
                    loaded_from_file = True
                except: pass

        if not loaded_from_file:
            text = 临时手写剧本.strip()
            if text:
                lines = text.splitlines()
                content_list = [(l.strip(), "") for l in lines if l.strip()]

        total = len(content_list)
        if total == 0:
            return ("", "空", "")

        # 3. 计数与播放
        if "自动" in 播放模式:
            idx = self._counter % total
            self._counter += 1
            status = "▶️ [自动]"
        else:
            idx = max(0, min(手动页码 - 1, total - 1))
            status = "🔒 [锁定]"

        current_tag, current_guide = content_list[idx]
        progress_info = f"{status} 第 {idx+1} 页 / 共 {total} 页"

        print(f"🎬 [千绘 剧情V14] {progress_info} | {os.path.basename(self._last_file)}")

        return (current_tag, progress_info, current_guide)

# ==========================================
# 节点 2: 剧情编排机 V9.1 (排序优化版)
# ==========================================
class QianhuiStoryComposer:
    def __init__(self): pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "剧情_自动加权": ("BOOLEAN", {"default": True, "label_on": "🔥 剧情优先 (Auto 1.5)", "label_off": "🎚️ 手动调节"}),
                "剧情_手动权重": ("FLOAT", {"default": 1.2, "min": 1.0, "max": 1.6, "step": 0.05, "display": "slider"}),
                "手动补充": ("STRING", {"multiline": True, "default": ""}),
                "智能去重": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "剧情_连接": ("STRING", {"forceInput": True, "default": ""}),
                "人设_连接": ("STRING", {"forceInput": True, "default": ""}),
                "全局设定_连接": ("STRING", {"forceInput": True, "default": ""}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Final_Prompt",)
    FUNCTION = "compose_story_v9"
    CATEGORY = "千绘/剧情"

    def compose_story_v9(self, 剧情_自动加权, 剧情_手动权重, 手动补充, 智能去重, 剧情_连接="", 人设_连接="", 全局设定_连接=""):
        # 1. 单独清洗各部分 (应用安全阀)
        safe_story = apply_safety_clamp(剧情_连接)
        safe_char = apply_safety_clamp(人设_连接)
        safe_global = apply_safety_clamp(全局设定_连接)
        safe_manual = apply_safety_clamp(手动补充)

        # 2. 剧情权重处理 (确保不超过 1.5)
        final_story_block = safe_story
        if safe_story:
            if 剧情_自动加权:
                final_story_block = f"({safe_story}:1.5)"
            else:
                if 剧情_手动权重 != 1.0:
                    # 确保手动权重也不超过 1.5
                    actual_weight = min(剧情_手动权重, 1.5)
                    final_story_block = f"({safe_story}:{actual_weight:.2f})"

        # 3. 严格定义输出顺序：剧情 -> 人设 -> 全局 -> 手动
        raw_parts = [final_story_block, safe_char, safe_global, safe_manual]

        # 过滤空内容
        valid_parts = [str(p) for p in raw_parts if p]

        # 4. 智能去重 (保持顺序)
        if 智能去重:
            seen = set()
            clean_list = []
            # 先拼接成完整字符串，保持顺序
            full_temp = ",".join(valid_parts).replace("，", ",")

            for tag in full_temp.split(','):
                tag = tag.strip()
                if not tag: continue

                # 提取核心词进行比对 (忽略权重和括号)
                core_word = re.sub(r'[\(\):0-9.]', '', tag).lower().strip()

                if core_word not in seen:
                    seen.add(core_word)
                    clean_list.append(tag)

            final_string = ", ".join(clean_list)
        else:
            final_string = ", ".join(valid_parts)

        return (final_string,)
