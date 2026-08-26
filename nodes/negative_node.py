"""
千绘 · 智能负面提示词节点 (QianhuiAutoNegative)

模块职责：接收正面提示词输入，按需附加通用负面词与正/负面 Embeddings，
并统一应用负面权重，输出组合后的正面/负面提示词。
数据流：输入 STRING + Embeddings 选项 -> 组合加权 -> 输出两路 STRING。
外部依赖：torch（仅导入类型约定）、folder_paths（获取 embeddings 列表）。
"""

import torch
import folder_paths
import os

class QianhuiAutoNegative:
    """
    ComfyUI 自定义节点 - 纯净版 (无自动检测)
    功能：
    1. 接收外部文本输入 (forceInput)。
    2. 通用负面词 (可开关)。
    3. 支持正面/负面 Embeddings 嵌入。
    4. 负面整体权重滑块 (控制 通用负面词 + 负面嵌入 的强度)。

    *已移除：自动根据正面词生成负面词的逻辑*
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        try:
            # 尝试获取 embeddings 列表，如果失败则为空列表 (防止非 Comfy 环境报错)
            embeddings_list = folder_paths.get_filename_list("embeddings")
        except:
            embeddings_list = []

        embeddings_options = ["None"] + embeddings_list

        return {
            "required": {
                # 强制输入端口
                "正面提示词输入": ("STRING", {"forceInput": True}),

                "启用通用负面词": ("BOOLEAN", {"default": True}),

                # 权重滑块：控制所有负面内容的权重
                "负面整体权重": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider"
                }),
            },
            "optional": {
                # 正面嵌入 (不受负面权重影响)
                "正面嵌入_1": (embeddings_options,),
                "正面嵌入_2": (embeddings_options,),
                "正面嵌入_3": (embeddings_options,),

                # 负面嵌入 (受负面权重影响)
                "负面嵌入_1": (embeddings_options,),
                "负面嵌入_2": (embeddings_options,),
                "负面嵌入_3": (embeddings_options,),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("正面提示词(含嵌入)", "负面提示词(全权重+嵌入)")
    FUNCTION = "process_keywords"
    CATEGORY = "千绘/提示词"

    def process_keywords(self, 正面提示词输入, 启用通用负面词, 负面整体权重,
                         正面嵌入_1="None", 正面嵌入_2="None", 正面嵌入_3="None",
                         负面嵌入_1="None", 负面嵌入_2="None", 负面嵌入_3="None"):

        # 防止输入为 None 的情况
        if 正面提示词输入 is None:
            正面提示词输入 = ""

        # --- 辅助函数：给列表中的 Tag 加权 ---
        def apply_weight(tag_list, weight):
            # 如果权重为 1.0，直接返回原列表，保持 Prompt 简洁
            if weight == 1.0:
                return tag_list
            weighted_list = []
            for tag in tag_list:
                w_str = round(weight, 1)
                weighted_list.append(f"({tag}:{w_str})")
            return weighted_list

        # --- 辅助函数：格式化 Embedding 字符串 ---
        def format_embedding(emb_name):
            if emb_name and emb_name != "None":
                # 修复核心：去除文件后缀 (如 .pt, .safetensors)
                # os.path.splitext("bad_hands.pt")[0] -> "bad_hands"
                clean_name = os.path.splitext(emb_name)[0]
                return f"embedding:{clean_name}"
            return None

        # ==========================================
        # 1. 处理正面 (组合输入+嵌入)
        # ==========================================
        pos_embeddings = []
        for emb in [正面嵌入_1, 正面嵌入_2, 正面嵌入_3]:
            res = format_embedding(emb)
            if res: pos_embeddings.append(res)

        pos_parts = [正面提示词输入] + pos_embeddings
        # 过滤空字符串，确保格式整洁
        final_positive = ", ".join([p for p in pos_parts if p])

        # ==========================================
        # 2. 准备负面数据
        # ==========================================

        # A. 负面嵌入原始列表
        raw_neg_embeddings = []
        for emb in [负面嵌入_1, 负面嵌入_2, 负面嵌入_3]:
            res = format_embedding(emb)
            if res: raw_neg_embeddings.append(res)

        # B. 通用保底负面词原始列表
        raw_base_negatives = [
            "lowres", "bad anatomy", "bad hands", "text", "error",
            "missing fingers", "extra digit", "fewer digits", "cropped",
            "worst quality", "low quality", "normal quality", "jpeg artifacts",
            "signature", "watermark", "username", "blurry", "artist name"
        ]

        if not 启用通用负面词:
            raw_base_negatives = []

        # ==========================================
        # 3. 统一加权处理
        # ==========================================

        # 对通用词和嵌入分别进行加权处理
        weighted_base = apply_weight(raw_base_negatives, 负面整体权重)
        weighted_embeddings = apply_weight(raw_neg_embeddings, 负面整体权重)

        # ==========================================
        # 4. 组合输出
        # ==========================================

        # 顺序：通用负面 -> 负面嵌入
        all_neg_parts = weighted_base + weighted_embeddings
        final_negative = ", ".join(all_neg_parts)

        return (final_positive, final_negative)

# ==========================================
# 关键修复：添加节点映射
# 没有这部分，ComfyUI 无法识别节点
# ==========================================
NODE_CLASS_MAPPINGS = {
    "QianhuiAutoNegative": QianhuiAutoNegative,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QianhuiAutoNegative": "千绘 · 智能负面提示词",
}
