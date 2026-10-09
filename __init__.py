"""
千绘 (comfyui-qianhui) — ComfyUI 综合提示词插件

由五个历史插件整合而成：
- comfyui-FMC (智能负面提示词)
- ComfyUIcustom_nodes (剧情/角色/场景系列)
- confyui-pose (动作/姿势/战斗系列)
- commfyui-loraxy (多 LoRA XY 网格测试)
- 水印插件 (Latent YOLO 智能修复 / 水印添加 / 水印提取)

模块职责：统一注册全部 13 个节点（13 类，剧情模块含编排/调度两节点）到
NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS，注册名统一带 Qianhui 前缀。
单个模块加载失败不影响其余节点。
"""

import sys
import traceback

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

def load_nodes(module_name, import_dict, file_label=""):
    try:
        module = __import__(module_name, globals(), locals(), level=1)
        for class_obj, display_name in import_dict.items():
            class_name = class_obj.__name__
            NODE_CLASS_MAPPINGS[class_name] = class_obj
            NODE_DISPLAY_NAME_MAPPINGS[class_name] = display_name
        print(f"✅ 千绘节点加载成功: {file_label}")
    except Exception as e:
        print(f"❌ 千绘节点加载失败 [{file_label}]: {e}")

# 1. 负面提示词模块
try:
    from .nodes.negative_node import QianhuiAutoNegative
    NODE_CLASS_MAPPINGS["QianhuiAutoNegative"] = QianhuiAutoNegative
    NODE_DISPLAY_NAME_MAPPINGS["QianhuiAutoNegative"] = "千绘 · 智能负面提示词"
except Exception as e:
    print(f"❌ 千绘 负面提示词模块加载失败: {e}")

# 2. 剧情模块
try:
    from .nodes.story_node import QianhuiStoryComposer, QianhuiStoryScheduler
    NODE_CLASS_MAPPINGS.update({
        "QianhuiStoryComposer": QianhuiStoryComposer,
        "QianhuiStoryScheduler": QianhuiStoryScheduler,
    })
    NODE_DISPLAY_NAME_MAPPINGS.update({
        "QianhuiStoryComposer": "千绘 · 剧情编排机 (V8)",
        "QianhuiStoryScheduler": "千绘 · 剧情播放器 (V13.3)",
    })
except Exception as e:
    print(f"❌ 千绘 剧情模块加载失败: {e}")

# 3. 角色模块
try:
    from .nodes.character_node import QianhuiCharacterCreator
    NODE_CLASS_MAPPINGS["QianhuiCharacterCreator"] = QianhuiCharacterCreator
    NODE_DISPLAY_NAME_MAPPINGS["QianhuiCharacterCreator"] = "千绘 · 角色定制 (V36)"
except Exception as e:
    print(f"❌ 千绘 角色模块加载失败: {e}")

# 4. 场景模块
try:
    from .nodes.scene_node import QianhuiSceneMaster
    NODE_CLASS_MAPPINGS["QianhuiSceneMaster"] = QianhuiSceneMaster
    NODE_DISPLAY_NAME_MAPPINGS["QianhuiSceneMaster"] = "千绘 · 场景光效大师 (V6.8·Krea2画风扩容版)"
except Exception as e:
    print(f"❌ 千绘 场景模块加载失败: {e}")

# 5. 动作模块
try:
    from .nodes.pose_node import QianhuiActionMaster
    NODE_CLASS_MAPPINGS["QianhuiActionMaster"] = QianhuiActionMaster
    NODE_DISPLAY_NAME_MAPPINGS["QianhuiActionMaster"] = "千绘 · 肢体动作师 (V8)"
except Exception as e:
    print(f"❌ 千绘 动作模块加载失败: {e}")

# 6. 姿势模块
try:
    from .nodes.sex_pose_node import QianhuiSexMaster
    NODE_CLASS_MAPPINGS["QianhuiSexMaster"] = QianhuiSexMaster
    NODE_DISPLAY_NAME_MAPPINGS["QianhuiSexMaster"] = "千绘 · 姿势大师 (V39)"
except Exception as e:
    print(f"❌ 千绘 姿势模块加载失败: {e}")

# 7. 战斗模块
try:
    from .nodes.combat_node import QianhuiCombatMaster
    NODE_CLASS_MAPPINGS["QianhuiCombatMaster"] = QianhuiCombatMaster
    NODE_DISPLAY_NAME_MAPPINGS["QianhuiCombatMaster"] = "千绘 · 战斗指挥官 (V1)"
except Exception as e:
    print(f"❌ 千绘 战斗模块加载失败: {e}")

# 8. 角色随机模块
try:
    from .nodes.role_name_node import QianhuiRandomCharacter
    NODE_CLASS_MAPPINGS["QianhuiRandomCharacter"] = QianhuiRandomCharacter
    NODE_DISPLAY_NAME_MAPPINGS["QianhuiRandomCharacter"] = "千绘 · 角色随机师 (V2)"
except Exception as e:
    print(f"❌ 千绘 角色随机模块加载失败: {e}")

# 9. 多 LoRA XY 网格模块
try:
    from .nodes.lora_grid_node import QianhuiMultiLoRAGrid
    NODE_CLASS_MAPPINGS["QianhuiMultiLoRAGrid"] = QianhuiMultiLoRAGrid
    NODE_DISPLAY_NAME_MAPPINGS["QianhuiMultiLoRAGrid"] = "千绘 · 多LoRA XY网格 (V2.6)"
except Exception as e:
    print(f"❌ 千绘 多LoRA XY网格模块加载失败: {e}")

# 11. Latent 智能修复模块
try:
    from .nodes.qianhui_latent_yolo import QianhuiLatentYOLO
    NODE_CLASS_MAPPINGS["QianhuiLatentYOLO"] = QianhuiLatentYOLO
    NODE_DISPLAY_NAME_MAPPINGS["QianhuiLatentYOLO"] = "千绘 · Latent智能修复 (YOLO+SAM+CN V6.11.3)"
except Exception as e:
    print(f"❌ 千绘 Latent修复模块加载失败: {e}")

# 12. 水印模块
try:
    from .nodes.qianhui_watermark import QianhuiWatermarkAdd, QianhuiWatermarkExtract
    NODE_CLASS_MAPPINGS["QianhuiWatermarkAdd"] = QianhuiWatermarkAdd
    NODE_DISPLAY_NAME_MAPPINGS["QianhuiWatermarkAdd"] = "千绘 · 水印添加"
    NODE_CLASS_MAPPINGS["QianhuiWatermarkExtract"] = QianhuiWatermarkExtract
    NODE_DISPLAY_NAME_MAPPINGS["QianhuiWatermarkExtract"] = "千绘 · 水印提取"
except Exception as e:
    print(f"❌ 千绘 水印模块加载失败: {e}")

# 8b. 前端联动数据路由（注册失败不影响其余节点）
WEB_DIRECTORY = "./js"
try:
    from aiohttp import web
    from server import PromptServer
    from .nodes.role_name_node import build_payload

    @PromptServer.instance.routes.get("/qianhui_role_data")
    async def qianhui_role_data(request):
        return web.json_response(build_payload())
except Exception:
    pass

# 8c. 配置预设数据路由（注册失败不影响节点）
try:
    from aiohttp import web
    from server import PromptServer
    from .nodes.qianhui_latent_yolo.presets import as_payload as preset_payload

    @PromptServer.instance.routes.get("/qianhui_preset_data")
    async def qianhui_preset_data(request):
        return web.json_response(preset_payload())
except Exception:
    pass

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
