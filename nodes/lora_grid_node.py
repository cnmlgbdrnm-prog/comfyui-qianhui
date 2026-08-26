"""
千绘 · 多 LoRA XY 网格测试节点 (QianhuiMultiLoRAGrid, 由 commfyui-loraxy SimpleMultiLoRAGrid V2.6 整合)

模块职责：批量对比 LoRA 效果 —— 行 = LoRA (最多 6 个 + No LoRA 基线行),
列 = 统一权重滑杆 (X轴), 逐格挂载 LoRA 采样并拼图输出, 底部绘制
自动溯源信息 (底模名/采样参数/正向与负向提示词)。全手动选择, 无随机逻辑。

数据流：MODEL/CLIP/VAE/CONDITIONING/LATENT 输入 -> 逐行逐列加载 LoRA ->
common_ksampler 采样 -> VAE 解码 -> 拼图并绘制信息栏 -> 输出 IMAGE。
隐藏输入 prompt/unique_id 用于穿透 LoRA 堆栈自动溯源底模名称。

文本渲染：全部文本经 load_cjk_font 加载中文字体(CJK_FONTS 回退链, Arial 无
中文身形会渲染为口口)；侧栏文件名取 basename 并按像素宽度截断(double-width
感知, 替代原按字符数 len 估算)。

外部依赖：ComfyUI 运行时 (nodes / folder_paths / comfy.sd / comfy.utils /
comfy.model_management / server)、torch、numpy、Pillow。
用法注意：网格行数受权重数量约束, 权重列表解析失败回退 [1.0]。
"""

import nodes
import folder_paths
import comfy.sd
import comfy.utils
import comfy.model_management
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import server

# ==========================================
# 多 LoRA XY 网格测试节点 (V2.6 - 终极完全体)
# Update Log:
# - V2.6: 新增 "自动溯源底模名称" (Auto-Trace Checkpoint Name)
#         现在可以穿透 LoRA 堆栈，自动找到最上游的模型文件名。
# ==========================================

# 中文字体回退链：Arial 无 CJK 字形, 中文会渲染成空框(口口)。
# 优先加载系统带中文覆盖的字体, 逐级回退; 全部失败用 Pillow 默认字体。
CJK_FONTS = ("msyh.ttc", "simhei.ttf", "simsun.ttc", "arial.ttf")


def load_cjk_font(size):
    """加载支持中文渲染的字体，按 CJK_FONTS 顺序回退。"""
    for name in CJK_FONTS:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()

class QianhuiMultiLoRAGrid:
    @classmethod
    def INPUT_TYPES(s):
        try:
            lora_list = folder_paths.get_filename_list("loras")
        except:
            lora_list = []

        lora_list = ["None"] + lora_list

        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),

                # --- 统一权重控制 (X轴) ---
                "global_weights": ("STRING", {
                    "default": "0.6, 0.8, 1.0, 1.2",
                    "multiline": False,
                    "placeholder": "X轴: 填入权重，如 0.6, 0.8, 1.0"
                }),

                # --- LoRA 选择 (Y轴) ---
                "lora_1": (lora_list, ),
                "lora_2": (lora_list, ),
                "lora_3": (lora_list, ),
                "lora_4": (lora_list, ),
                "lora_5": (lora_list, ),
                "lora_6": (lora_list, ),

                # --- 布局与预览 ---
                "font_size": ("INT", {"default": 40, "min": 20, "max": 200}),
                "show_preview": ("BOOLEAN", {"default": True, "label_on": "开启实时预览", "label_off": "关闭"}),

                # --- 采样设置 ---
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20}),
                "cfg": ("FLOAT", {"default": 8.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                "denoise": ("FLOAT", {"default": 1.0}),
            },
            # --- 隐藏输入：用于获取图谱信息 ---
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run_xy_grid"
    CATEGORY = "千绘/网格"

    # --- 辅助绘图函数 ---
    def create_text_image(self, text, font_size, orientation='horizontal'):
        font = load_cjk_font(font_size)

        temp_img = Image.new("RGB", (1, 1))
        draw = ImageDraw.Draw(temp_img)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0] + 20
        text_h = bbox[3] - bbox[1] + 20

        img = Image.new("RGB", (text_w, text_h), "white")
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), text, fill="black", font=font)

        if orientation == 'vertical':
            return img.rotate(90, expand=True)
        return img

    def truncate_by_width(self, text, font, max_width):
        """按像素宽度截断文本（中文双宽字符正确度量），末尾加省略号。"""
        if font.getlength(text) <= max_width:
            return text
        ellipsis = "…"
        max_content = max_width - font.getlength(ellipsis)
        # 逐字符累计, 保证 内容+省略号 不超宽
        cutoff = 0
        for i in range(1, len(text) + 1):
            if font.getlength(text[:i]) <= max_content:
                cutoff = i
            else:
                break
        return text[:cutoff] + ellipsis

    def wrap_text(self, text, font, max_width):
        """按像素宽度换行——中文/全角字符占双宽, 用 getlength 逐字符累计,
        与 textwrap 的近似字符数估算相比能正确容纳 CJK。"""
        lines = []
        if not text: return lines
        for para in text.split('\n'):
            cur = ""
            for ch in para:
                if font.getlength(cur + ch) <= max_width:
                    cur += ch
                else:
                    if cur: lines.append(cur)
                    # 单字符超宽时跳过, 避免死循环
                    cur = ch if font.getlength(ch) <= max_width else ""
            if cur: lines.append(cur)
        return lines

    def send_preview(self, img_tensor):
        try:
            from nodes import PreviewImage
            preview_node = PreviewImage()
            preview_node.save_images(img_tensor)
        except:
            pass

    # --- 侦探函数 A: 找提示词 ---
    def find_upstream_text(self, input_link, full_prompt_data):
        if not input_link or not isinstance(input_link, list): return ""
        node_id = str(input_link[0])
        node = full_prompt_data.get(node_id)
        if not node: return ""

        inputs = node.get("inputs", {})

        # 1. 尝试直接找 text
        if "text" in inputs:
            val = inputs["text"]
            if isinstance(val, str): return val
            if isinstance(val, list): return self.find_upstream_text(val, full_prompt_data)

        # 2. 尝试找 text_g / text_l (SDXL)
        for key in ["text_g", "text_l"]:
            if key in inputs:
                val = inputs[key]
                if isinstance(val, str): return val
                if isinstance(val, list): return self.find_upstream_text(val, full_prompt_data)
        return f"(Unknown from Node {node_id})"

    # --- 侦探函数 B: 找 Checkpoint (V2.6 新增) ---
    def find_upstream_checkpoint(self, input_link, full_prompt_data, depth=0):
        # 防止死循环
        if depth > 20: return "Unknown (Max Depth)"

        if not input_link or not isinstance(input_link, list): return "Unknown Model"

        node_id = str(input_link[0])
        node = full_prompt_data.get(node_id)
        if not node: return "Unknown Node"

        inputs = node.get("inputs", {})

        # 1. 直接命中：如果是 Loader，通常有 ckpt_name 控件
        #    注意：API JSON 里，控件的值直接存在 inputs 字典里，key 就是控件名
        if "ckpt_name" in inputs:
            val = inputs["ckpt_name"]
            if isinstance(val, str): return val # 找到了！

        # 2. 间接命中：如果是 UNETLoader (比如 GGUF)，可能有 unet_name
        if "unet_name" in inputs:
             val = inputs["unet_name"]
             if isinstance(val, str): return val

        # 3. 递归回溯：如果是 LoraLoader 或 Reroute，它们会有一个 model 输入
        #    我们需要穿过它们继续往上找
        if "model" in inputs:
            upstream_link = inputs["model"]
            if isinstance(upstream_link, list):
                return self.find_upstream_checkpoint(upstream_link, full_prompt_data, depth+1)

        return f"Unknown (Node {node_id})"

    def run_xy_grid(self, model, clip, vae, positive, negative, latent_image,
                    global_weights, lora_1, lora_2, lora_3, lora_4, lora_5, lora_6,
                    font_size, show_preview, seed, steps, cfg, sampler_name, scheduler, denoise,
                    prompt=None, unique_id=None):

        # --- 自动侦探逻辑 ---
        pos_text = "N/A"
        neg_text = "N/A"
        ckpt_name = "Unknown Model"

        if prompt and unique_id:
            my_node = prompt.get(str(unique_id))
            if my_node:
                my_inputs = my_node.get("inputs", {})

                # 1. 抓 Prompt
                if "positive" in my_inputs and isinstance(my_inputs["positive"], list):
                    pos_text = self.find_upstream_text(my_inputs["positive"], prompt)
                if "negative" in my_inputs and isinstance(my_inputs["negative"], list):
                    neg_text = self.find_upstream_text(my_inputs["negative"], prompt)

                # 2. 抓 Checkpoint (通过 model 接口逆流而上)
                if "model" in my_inputs and isinstance(my_inputs["model"], list):
                    ckpt_name = self.find_upstream_checkpoint(my_inputs["model"], prompt)

        # 1. 准备 LoRA 列表
        lora_inputs = [lora_1, lora_2, lora_3, lora_4, lora_5, lora_6]
        active_loras = [name for name in lora_inputs if name != "None"]
        process_list = ["__NO_LORA__"] + active_loras

        print(f"🚀 [Grid V2.6] Auto-Detect:\nModel: {ckpt_name}\nPos: {pos_text[:15]}...\nNeg: {neg_text[:15]}...")

        # 解析权重 X轴
        w_text = global_weights.replace("，", ",").replace(";", ",").replace(" ", "")
        weights = []
        for v in w_text.split(","):
            if v:
                try: weights.append(float(v))
                except: pass
        if not weights: weights = [1.0]

        # 预加载 LoRA
        lora_dicts = {}
        for name in active_loras:
            path = folder_paths.get_full_path("loras", name)
            if path:
                try:
                    lora_dicts[name] = comfy.utils.load_torch_file(path)
                except:
                    print(f"❌ LoRA损坏: {name}")

        grid_matrix = []

        # 2. 生成循环
        for row_idx, lora_name in enumerate(process_list):
            row_imgs = []
            is_no_lora = (lora_name == "__NO_LORA__")

            if not is_no_lora and lora_name not in lora_dicts:
                for _ in weights: row_imgs.append(Image.new("RGB", (512, 512), "black"))
                grid_matrix.append(row_imgs)
                continue

            display_name = "No LoRA" if is_no_lora else lora_name
            print(f"   ► 行 {row_idx+1}/{len(process_list)}: {display_name}")

            for col_idx, w in enumerate(weights):
                comfy.model_management.soft_empty_cache()

                # A. 挂载
                if is_no_lora:
                    model_lora, clip_lora = model, clip
                else:
                    try:
                        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora_dicts[lora_name], w, w)
                    except:
                        row_imgs.append(Image.new("RGB", (512, 512), "red"))
                        continue

                # B. 采样
                current_latent = {"samples": latent_image["samples"].clone()}
                try:
                    common_res = nodes.common_ksampler(
                        model=model_lora, seed=seed, steps=steps, cfg=cfg,
                        sampler_name=sampler_name, scheduler=scheduler,
                        positive=positive, negative=negative,
                        latent=current_latent, denoise=denoise
                    )
                    samples = common_res[0]["samples"]
                    img_tensor = vae.decode(samples)
                    img_tensor = torch.clamp(img_tensor, 0.0, 1.0)

                    if show_preview: self.send_preview(img_tensor)

                    pil_img = Image.fromarray(np.clip(255. * img_tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
                    row_imgs.append(pil_img)
                except Exception as e:
                    print(f"E: {e}")
                    row_imgs.append(Image.new("RGB", (512, 512), "black"))

            grid_matrix.append(row_imgs)

        # 3. 拼图与信息绘制
        if not grid_matrix or not grid_matrix[0]: return (torch.zeros((1, 512, 512, 3)),)

        sample_w, sample_h = grid_matrix[0][0].size

        # 基础尺寸
        header_h = font_size + 40
        sidebar_w = font_size + 40

        font_main = load_cjk_font(font_size)
        font_small = load_cjk_font(int(font_size * 0.7))
        font_prompt = load_cjk_font(int(font_size * 0.6))

        content_w = sample_w * len(weights)
        total_w = sidebar_w + content_w

        # --- 底部信息栏计算 ---
        padding = 20
        line_height = int(font_size * 0.8)

        # 清理文本
        pos_text_clean = pos_text.strip()
        neg_text_clean = neg_text.strip()
        ckpt_name_clean = ckpt_name.replace("\\", "/").split("/")[-1] # 只显示文件名，不显示长路径

        # 组合第一行参数信息
        info_line_1 = f"Model: {ckpt_name_clean}"
        info_line_2 = f"Sampler: {sampler_name} ({scheduler}) | Steps: {steps} | CFG: {cfg} | Seed: {seed} | Denoise: {denoise}"

        pos_lines = self.wrap_text(f"Positive: {pos_text_clean}", font_prompt, total_w - 40)
        neg_lines = self.wrap_text(f"Negative: {neg_text_clean}", font_prompt, total_w - 40)

        footer_h = padding * 2
        footer_h += line_height * 3 # 模型名 + 采样参数 + 间距
        footer_h += len(pos_lines) * line_height + (padding if pos_lines else 0)
        footer_h += len(neg_lines) * line_height + (padding if neg_lines else 0)

        total_h = header_h + (sample_h * len(process_list)) + footer_h

        canvas = Image.new("RGB", (total_w, total_h), "white")
        draw = ImageDraw.Draw(canvas)

        # --- 绘图 ---
        for i, w in enumerate(weights):
            label = f"w = {w}"
            txt_img = self.create_text_image(label, font_size, 'horizontal')
            x = sidebar_w + (i * sample_w) + (sample_w - txt_img.width) // 2
            y = (header_h - txt_img.height) // 2
            canvas.paste(txt_img, (x, y))

        for i, name in enumerate(process_list):
            if name == "__NO_LORA__":
                d_name = "No LoRA"
            else:
                # 只显示 basename（去路径、去扩展名），否则路径前段挤掉文件名
                d_name = name.replace("\\", "/").rsplit("/", 1)[-1].rsplit('.', 1)[0]
                # 按侧栏像素宽度截断（CJK 按双宽度量, 代替字符数估算）
                d_name = self.truncate_by_width(d_name, font_main, sidebar_w - 20)
            txt_img = self.create_text_image(d_name, font_size, 'vertical')
            x = (sidebar_w - txt_img.width) // 2
            y = header_h + (i * sample_h) + (sample_h - txt_img.height) // 2
            canvas.paste(txt_img, (x, y))

        for r, row_imgs in enumerate(grid_matrix):
            for c, img in enumerate(row_imgs):
                canvas.paste(img, (sidebar_w + c * sample_w, header_h + r * sample_h))

        # --- 绘制底部信息 ---
        current_y = header_h + (sample_h * len(process_list)) + padding

        draw.line([(20, current_y), (total_w - 20, current_y)], fill="black", width=2)
        current_y += 20

        # 1. 模型名 (加粗/大一点，深蓝)
        draw.text((20, current_y), info_line_1, fill="#000080", font=font_small)
        current_y += line_height * 1.2

        # 2. 采样参数 (黑色)
        draw.text((20, current_y), info_line_2, fill="black", font=font_small)
        current_y += line_height * 1.5

        # 3. Prompt
        if pos_lines:
            for line in pos_lines:
                draw.text((20, current_y), line, fill="#006400", font=font_prompt)
                current_y += line_height
            current_y += 10

        if neg_lines:
            for line in neg_lines:
                draw.text((20, current_y), line, fill="#8B0000", font=font_prompt)
                current_y += line_height

        print("✅ XY Grid (V2.6 终极版) 完成！")

        final_tensor = torch.from_numpy(np.array(canvas).astype(np.float32) / 255.0).unsqueeze(0)
        return (final_tensor,)

NODE_CLASS_MAPPINGS = {
    "QianhuiMultiLoRAGrid": QianhuiMultiLoRAGrid
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "QianhuiMultiLoRAGrid": "千绘 · 多LoRA XY网格"
}
