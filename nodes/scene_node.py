"""
千绘 · 场景模块 (场景/光效大师 QianhuiSceneMaster)

模块职责：内置场景/季节/天气/时间/光效/画质风格/衣物质感数据库，
按用户选择提取英文 Tag 并智能拼接环境提示词，全手动选择无随机逻辑。
anima 适配：画面画质风格 内置完整质量体系 —— 官方完整顶配(人工评分
masterpiece/best quality + 美学评分三叠 + meta highres/absurdres/newest)与
美学评分 5 档(score_9+8+7 / score_9 / score_8 / score_7 / score_6)；
氛围句 提供单句环境氛围自然语言(Hybrid 格式第二层, 与标签逗号混排)，
权重滑杆设 1.0 即纯标签输出, 符合官方"硬标签层+质量词+氛围句"规范, 无需模型类型区分。
Krea 2 适配：画面画质风格 内置 Krea 2 画风词库 —— Krea 2 (Qwen3VL 编码器)
以自然语言风格短语为主, 含官方 Moodboard 画风(电影微距/巨构/粗粝写实、霓虹黑色电影、
霓虹 VHS、霓虹剪影)与官方风格 LoRA 触发词(如 neon noir fog style / emerald haze style),
以及 Krea2 Me 正向质量词, 供 Krea 2 系列模型用户选择。
数据流：内置 SCENE_DATA 选项 -> 提取 (括号内英文) -> 拼接 -> 加权输出 STRING。
外部依赖：无（仅标准库）。
"""

import re

# ==========================================
# 🌍 博士的场景与光影数据库 (V6.8 Krea2 画风扩容)
# ==========================================

SCENE_DATA = {
    # 1. 场景 (Location) - 保持 V6.6 全量
    "location": [
        "(不指定)",
        # --- 🏠 室内/生活 (Indoor Life) ---
        "🏠 凌乱卧室 (indoors, messy bedroom, unmade bed, clothes on floor, morning light)",
        "🏠 唯美闺房 (indoors, girly bedroom, pastel colors, stuffed toys, frills)",
        "🏠 现代客厅 (indoors, luxury living room, modern furniture, floor-to-ceiling windows, city view)",
        "🏠 厨房/料理台 (indoors, kitchen, cooking utensils, refrigerator, food on table)",
        "🏠 传统和室 (indoors, tatami, sliding doors, japanese room, kotatsu)",
        "🏠 浴室/浴缸 (indoors, bathroom, bathtub, steam, water droplets, tiles)",
        "🏠 阳台/天台 (outdoors, balcony, railing, laundry, flower pot)",

        # --- 🏫 学校/青春 (School Life) ---
        "🏫 教室 (indoors, classroom, school desks, blackboard, sunlight through window)",
        "🏫 走廊 (indoors, school hallway, lockers, polished floor, perspective)",
        "🏫 体育馆/仓库 (indoors, gym storage, gym mats, volleyball, dusty)",
        "🏫 保健室 (indoors, school infirmary, white curtain, bed)",
        "🏫 游泳池 (outdoors, school pool, water surface, chain-link fence, blue sky)",
        "🏫 图书馆 (indoors, library, bookshelves, books, quiet atmosphere)",

        # --- 🏙️ 城市/街头 (Urban) ---
        "🏙️ 繁华涩谷 (outdoors, shibuya crossing, crowd, skyscrapers, advertisement screens)",
        "🏙️ 深夜便利店 (indoors, convenience store, shelves, fluorescent light, night)",
        "🏙️ 黑暗小巷 (outdoors, dark alley, trash bags, graffiti, wet ground, brick wall)",
        "🏙️ 电车内部 (indoors, train interior, handrails, priority seat, windows)",
        "🏙️ 废弃工厂 (indoors, abandoned factory, rust, broken glass, industrial pipes, debris)",
        "🏙️ 摩天大楼顶层 (outdoors, skyscraper rooftop, cityscape, night view, railing)",
        "🏙️ 酒吧/夜店 (indoors, bar counter, alcohol bottles, dim lighting, neon ambiance)",

        # --- 🚽 密闭/特殊 (Confined/Special) ---
        "🚽 公共厕所 (indoors, public toilet, tile wall, mirror, sinks, cold lighting)",
        "🚽 厕所隔间 (indoors, toilet stall, cramped, graffiti, privacy)",
        "👗 更衣室 (indoors, locker room, lockers, benches, changing clothes)",
        "🏥 病房 (indoors, hospital room, hospital bed, medical equipment, white sterile)",
        "⛓️ 地牢/监禁 (indoors, dungeon, stone wall, chains, bars, dark)",
        "📦 狭窄储物间 (indoors, storage room, boxes, shelves, cramped space)",

        # --- 🏰 奇幻/古典 (Fantasy/Historical) ---
        "🏰 欧式皇宫 (indoors, throne room, red carpet, chandelier, marble columns)",
        "🏰 魔法塔图书馆 (indoors, wizard tower, magic books, floating objects, mystical)",
        "🏰 中世纪酒馆 (indoors, medieval tavern, wooden tables, barrels, candle light)",
        "🏰 古代遗迹 (outdoors, ancient ruins, broken pillars, vines, moss, history)",
        "🏰 哥特式教堂 (indoors, cathedral, stained glass, pews, holy atmosphere)",
        "🏰 东方庭院 (outdoors, japanese garden, pond, bamboo, stone lantern, cherry blossoms)",

        # --- 🌲 自然/生态 (Nature) ---
        "🌲 原始森林 (outdoors, deep forest, huge trees, ferns, sunbeams, nature)",
        "🌲 花海 (outdoors, flower field, blooming flowers, petals, vast sky)",
        "🌲 热带海滩 (outdoors, beach, ocean, white sand, palm trees, summer)",
        "🌲 露天温泉 (outdoors, onsen, steam, rocks, snow, relaxing)",
        "🌲 洞穴/地底 (indoors, cave, stalactites, underground lake, bioluminescence)",
        "🌲 荒凉沙漠 (outdoors, desert, dunes, dry, sun, cactus)",
        "🌲 雪山之巅 (outdoors, snowy mountain, peak, clouds, cold, frozen)",

        # --- 🌌 科幻/赛博 (Sci-Fi/Cyberpunk) ---
        "🌌 赛博贫民窟 (outdoors, cyberpunk city, neon signs, rain, cables, high tech low life)",
        "🌌 太空站走廊 (indoors, spaceship interior, sci-fi corridor, led lights, metallic)",
        "🌌 月球表面 (outdoors, moon surface, earth in sky, stars, crater)",
        "🌌 赛博黑客房 (indoors, computer room, multiple screens, servers, cables, messy)",
        "🌌 虚空/纯色 (simple background, white background, void, minimalist)"
    ],

    # 2a. 季节 (Season)
    "season": [
        "(不指定)",
        "🌸 春季 (spring, cherry blossoms, pink petals, vibrant)",
        "🌻 夏季 (summer, lush green, harsh sunlight, cicadas, cumulonimbus clouds)",
        "🍁 秋季 (autumn, red maple leaves, fallen leaves, orange palette)",
        "❄️ 冬季 (winter, snow, cold breath, frozen, white)",
        "🍂 末世/枯萎 (dead trees, wasteland, grey sky, desolation)"
    ],

    # 2b. 天气 (Weather)
    "weather": [
        "(不指定)",
        "☀️ 晴朗 (clear sky, blue sky, sunny)",
        "☁️ 多云 (cloudy sky, overcast, diffused light)",
        "🌧️ 雨天 (raining, wet ground, splashing, umbrella)",
        "⛈️ 暴风雨 (storm, lightning, heavy rain, dark clouds)",
        "❄️ 下雪 (snowing, snowflakes, blizzard)",
        "🌫️ 浓雾 (thick fog, mist, haze, atmospheric perspective)",
        "🌬️ 强风 (strong wind, blowing hair, flying debris)",
        "🌅 极光 (aurora borealis, night sky, magical)"
    ],

    # 2c. 时间 (Time)
    "time": [
        "(不指定)",
        "🌅 日出 (sunrise, morning glow, lens flare)",
        "☀️ 正午 (noon, bright daylight, hard shadows)",
        "🌇 日落 (sunset, orange sky, silhouette, evening)",
        "🌙 深夜 (night, dark, moon, stars)",
        "📸 黄金时刻 (golden hour lighting, warm atmosphere, soft light)",
        "📸 蓝调时刻 (blue hour lighting, cold atmosphere, twilight)"
    ],

    # 3. 光效组合 (Lighting Combo)
    "lighting_combo": [
        "(不指定)",
        # --- 电影/氛围 ---
        "🎬 电影感: 诺兰风 (cinematic lighting, teal and orange, dramatic contrast, masterpiece)",
        "🎬 电影感: 黑色电影 (film noir, chiaroscuro, black and white, high contrast, dramatic shadows)",
        "🎬 电影感: 王家卫风 (neon ambiance, moody, saturated, cinematic, motion blur)",
        "🎬 电影感: 韦斯安德森 (flat lighting, pastel colors, symmetry, minimal set design)",

        # --- 专业摄影/人像 ---
        "💡 人像: 柔光影棚 (studio lighting, softbox, beauty dish, flattering, professional)",
        "💡 人像: 伦勃朗光 (rembrandt lighting, triangle of light, artistic shadows, depth)",
        "💡 人像: 蝴蝶光 (butterfly lighting, glamour lighting, symmetrical shadows)",
        "💡 人像: 分割光 (split lighting, half face in shadow, dramatic)",
        "💡 人像: 高调摄影 (high key lighting, bright, white background, cheerful, no shadows)",
        "💡 人像: 低调摄影 (low key lighting, dark background, rim light, mysterious)",

        # --- 特殊氛围 ---
        "😈 赛博: 霓虹光污染 (neon lighting, cyberpunk atmosphere, purple and blue lights, glow)",
        "😈 赛博: 故障艺术 (glitch art lighting, chromatic aberration, digital noise)",
        "🌞 自然: 丁达尔圣光 (volumetric lighting, god rays, sunbeams through mist, divine)",
        "🌞 自然: 树影斑驳 (komorebi, dappled sunlight, leaf shadows, nature)",
        "🧚‍♀️ 奇幻: 仙境辉光 (ethereal lighting, bloom, soft glow, magical particles, dreamlike)",
        "👻 恐怖: 阴森烛光 (dimly lit, candle light, eerie shadows, horror theme, lighting from below)",
        "🚀 科幻: 医疗舱冷光 (clinical lighting, cold white, sterile, fluorescent, futuristic)",
        "🔥 战争: 火光映照 (firelight, warm glow, dynamic lighting, embers, burning)",
        "🌊 水下: 深海焦散 (underwater, caustics, refracting light, blue tint, bubbles)",
        "🕺 派对: 俱乐部激光 (club lighting, laser lights, strobe, colorful, dynamic)"
    ],

    # 4. 独立光效 (Single Source)
    "lighting_single": [
        "(不指定)",
        "✨ 轮廓光/背光 (rim lighting, backlighting, hair glow, halo)",
        "✨ 侧光 (side lighting, dramatic shadows, volume)",
        "✨ 顶光/聚光灯 (top lighting, spotlight, stage effect)",
        "✨ 底光/鬼光 (lighting from below, ominous, scary)",
        "✨ 强硬直射光 (hard light, sharp shadows, sunlight)",
        "✨ 柔和漫射光 (soft light, diffused, cloudy day)",
        "✨ 生物荧光 (bioluminescence, glowing particles, magical)",
        "✨ 镜头光晕 (lens flare, anamorphic flare, bright)",
        "✨ 丁达尔光柱 (god rays, light shafts, visible beam)",
        "✨ 窗户投影 (window shadow, blind shadow, light pattern)",
        "✨ 强光手电 (flashlight beam, cone of light, exploration)",
        "✨ 手机/屏幕光 (screen glow, face illuminated by phone, blue light)"
    ],

    # 5. 画质/风格 (Visual Style) - 🔥 修复：黑白漫画去墨水化
    "visual_style": [
        "(不指定)",
        # --- 评分词 (anima 完整质量体系:
        #   1) 人工评分 6级 masterpiece/best quality/good quality/normal quality/low/worst
        #   2) 美学评分 9级 score_9~score_1 (唯一允许下划线的标签, 正面取高分档,
        #      低分档 score_5 以下用于负面提示词)
        #   3) meta: highres/absurdres 分辨率, newest/year 年份
        #   官方完整正向为 人工评分+美学评分+meta 混合, 第一档为官方示例整条) ---
        "🏆 anima官方顶配 (masterpiece, best quality, score_9, score_8, score_7, highres, absurdres, newest)",
        "🏆 anima顶级: score_9+8+7 (score_9, score_8, score_7)",
        "🏆 anima高质: score_9 (score_9)",
        "🏆 anima优质: score_8 (score_8)",
        "🏆 anima推荐: score_7 (score_7)",
        "🏆 anima标准: score_6 (score_6)",

        # --- 摄影风格 ---
        "📸 极致画质 (masterpiece, best quality, 8k, ultra detailed, highres)",
        "📸 真实摄影 (photorealistic, raw photo, dslr, 85mm, depth of field, bokeh)",
        "📸 复古胶片 (film grain, vintage, polaroid, kodak portra, analog aesthetic)",
        "📸 鱼眼镜头/GoPro (fisheye lens, distorted perspective, action shot, wide angle)",
        "📸 航拍/无人机 (aerial view, drone shot, bird's eye view, vast scale)",

        # --- 传统艺术 ---
        "🎨 厚涂油画 (impasto, oil painting, heavy brushstrokes, textured canvas)",
        "🎨 水彩风格 (watercolor, wet on wet, soft edges, artistic)",
        "🎨 素描线稿 (sketch, charcoal, pencil drawing, rough lines)",
        "🎨 浮世绘 (ukiyo-e, japanese traditional art, woodblock print, flat color)",
        "🎨 新艺术运动 (art nouveau, alphonse mucha style, intricate details, flowing lines)",
        "🎨 印象派 (impressionism, claude monet style, light and color)",

        # --- 现代/数字 ---
        "🖌️ 赛璐璐/动漫 (anime style, cel shading, vibrant colors, flat color)",
        # 🔥 修复点：移除 'ink'，改为 'lineart, clean lines'，保留网点纸
        "🖌️ 黑白漫画 (monochrome, manga style, greyscale, screentones, lineart, clean lines)",
        "👾 像素艺术 (pixel art, 16-bit, retro game style)",
        "👾 低多边形 (low poly, 3d model, sharp edges, minimalist)",
        "👾 故障艺术 (glitch art, datamosh, vhs artifacts, distortion)",
        "📼 蒸汽波 (vaporwave, retro 80s, neon grid, pink and cyan)",

        # --- 渲染引擎 ---
        "🧱 虚幻引擎5 (unreal engine 5, ray tracing, 3d render, hyperrealistic)",
        "🧱 Octane渲染 (octane render, physically based rendering, cinematic)",
        "🗿 粘土定格 (claymation, stop motion, plastic texture, handcrafted)",

        # --- Krea 2 官方画风 (Krea 2 Moodboard 派生, 自然语言风格短语, 非标签堆叠) ---
        "🎬 Krea2官方: 电影微距写实 (extreme macro portraiture, hyper-realistic skin texture, chiaroscuro lighting, golden hour tones, shallow depth of field, cinematic film grain)",
        "🎬 Krea2官方: 电影巨构写实 (cinematic scale, megastructure, hard sci-fi realism, high-contrast lighting, intricate greeble textures, atmospheric perspective, monumental decay)",
        "🎬 Krea2官方: 电影粗粝写实 (cinematic realism, desaturated color palette, raw skin texture, shallow depth of field, moody atmospheric lighting, editorial portraiture, high-detail photography)",
        "🎬 Krea2官方: 电影黑色电影 (high-contrast studio portrait, dramatic gel lighting, bichromatic color palette, sharp rim light, glossy skin texture, editorial fashion photography)",
        "📼 Krea2官方: 霓虹VHS黑 (lo-fi vhs aesthetic, chromatic aberration, 90s retro anime style, heavy analog film grain, cyberpunk atmosphere, saturated magenta and blue lighting)",
        "🎬 Krea2官方: 霓虹剪影 (sharp rim lighting, high-contrast silhouette, neo-noir aesthetic, vibrant neon backlighting, dramatic chiaroscuro)",

        # --- Krea 2 风格 LoRA 触发词 (Krea-2-Turbo 基座, 整句触发词, 权重 1.0-1.25) ---
        "🧪 Krea2LoRA: 霓虹雾 (neon noir fog style)",
        "🧪 Krea2LoRA: 霓虹辉光 (neon noir glow style)",
        "🧪 Krea2LoRA: 霓虹散景 (glowing neon bokeh style)",
        "🧪 Krea2LoRA: 电蓝辉光 (electric blue neon glow style)",
        "🧪 Krea2LoRA: 绯红霓虹漫画 (crimson neon manga)",
        "🧪 Krea2LoRA: 深霓虹水粉夜 (dark neon gouache night style)",
        "🧪 Krea2LoRA: 绯雾辉光 (foggy crimson glow style)",
        "🧪 Krea2LoRA: 翡翠雾霭 (emerald haze style)",
        "🧪 Krea2LoRA: 金色散景 (golden bokeh glow style)",

        # --- Krea2 Me 正向质量词 (SeaArt Krea2 Me 推荐正向词, 用逗号短语注入) ---
        "🏆 Krea2质量词 (masterpiece, photorealistic, highly detailed scenery, cinematic lighting, raw photo, epic landscape, 8k resolution, intricate details)"
    ],

    # 6. 衣物/全局材质 (Clothing Texture)
    "clothing_texture": [
        "(不指定)",
        # --- 基础面料 ---
        "🧶 棉质/日常 (cotton fabric, matte finish, soft texture)",
        "🧶 丹宁/牛仔 (denim texture, visible stitching, rough fabric)",
        "🧶 针织/毛衣 (ribbed sweater, knitted fabric, wool texture, cozy)",
        # --- 丝滑光泽 ---
        "✨ 丝绸/缎面 (silk, satin, shiny fabric, smooth, flowing)",
        "✨ 天鹅绒 (velvet, rich texture, soft sheen)",
        # --- 恋物/特殊 ---
        "🧤 乳胶/胶衣 (latex, rubber, highly reflective, polished, fetish)",
        "🧤 漆皮/亮皮 (patent leather, shiny leather, reflective)",
        "🧤 皮革 (leather texture, textured leather, matte leather)",
        "🧤 PVC/塑料 (pvc, transparent plastic, synthetic, raincoat)",
        # --- 运动/机能 ---
        "🤸 氨纶/紧身 (spandex, lycra, stretchy, tight fit, gym wear)",
        "🤸 网眼/透气 (mesh fabric, sportswear, see-through)",
        "🤸 碳纤维/机甲 (carbon fiber, metallic texture, armor plate, hard surface)",
        # --- 诱惑/透视 ---
        "🪶 蕾丝 (lace, intricate pattern, sheer)",
        "🪶 雪纺/薄纱 (chiffon, translucent, see-through fabric)",
        "🪶 渔网 (fishnet, grid pattern)",
        "🪶 湿身 (wet clothes, clinging to body, translucent when wet, sweat)",
        # --- 战损/状态 ---
        "🍂 破损/褴褛 (torn clothes, ripped, damaged, ragged)",
        "🍂 脏污/油迹 (dirty clothes, oil stains, grime, messy)"
    ],

    # 7. 氛围句 (Natural Language Layer) - anima Hybrid 格式第二层
    # 官方规范: 硬标签锚定主体结构, 自然语言供给构图/光影/氛围;
    # 纯环境氛围句(不含人物), 与标签逗号混排输出
    "atmosphere": [
        "(不指定)",
        "🌫️ 雨夜霓虹 (rain glistens on the pavement under flickering neon signs, wet reflections stretching into the distance)",
        "🌫️ 温暖夕阳 (warm golden sunlight pours through the window, long shadows stretching across the room)",
        "🌫️ 幽静月光 (cool silver moonlight spills across the room, soft highlights tracing the furniture)",
        "🌫️ 丁达尔森林 (sunbeams pierce through the canopy, dust motes floating in the light, gentle mist rising)",
        "🌫️ 静谧雪夜 (snow falls silently, covering the ground in a soft white blanket, cold blue tones)",
        "🌫️ 深海焦散 (refracted light shimmers through the water, caustics dancing, bubbles drifting upward)",
        "🌫️ 极光苍穹 (aurora borealis paints the night sky in green and purple ribbons, stars gleaming)",
        "🌫️ 烛光摇曳 (flickering candlelight casts dancing shadows on the walls, warm intimate glow)",
        "🌫️ 赛博雨巷 (holographic advertisements flicker in the rain-soaked streets, steam rising from vents)",
        "🌫️ 温泉晨雾 (steam rises from the water surface, mingling with the crisp morning air)",
        "🌫️ 车水马龙 (city traffic lights blur into long-exposure bokeh streaks, dynamic motion energy)",
        "🌫️ 空旷寂寥 (vast empty space stretches to the horizon, a sense of silence and solitude)"
    ]
}

class QianhuiSceneMaster:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # --- 场景 ---
                "地点选择": (SCENE_DATA["location"],),

                # --- 时空 ---
                "季节": (SCENE_DATA["season"],),
                "天气": (SCENE_DATA["weather"],),
                "时间": (SCENE_DATA["time"],),

                # --- 光效 ---
                "光效_整体氛围": (SCENE_DATA["lighting_combo"],),
                "光效_独立光源": (SCENE_DATA["lighting_single"],),

                # --- 质感 ---
                "画面画质风格": (SCENE_DATA["visual_style"],),
                "全局衣物质感": (SCENE_DATA["clothing_texture"],),

                # --- 控制 ---
                "手动补充": ("STRING", {"default": "", "multiline": False, "placeholder": "如: fire, explosion, underwater"}),
                "环境总权重": ("FLOAT", {"default": 1.1, "min": 0.5, "max": 1.6, "step": 0.1, "display": "slider"}),
                "氛围句": (SCENE_DATA["atmosphere"],),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Environment_Prompt",)
    FUNCTION = "generate_scene_v6"
    CATEGORY = "千绘/场景"

    # 每次运行都强制刷新，防止随机数或条件被缓存
    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("nan")

    def generate_scene_v6(self, **kwargs):

        def extract(text):
            if not text or "(不指定)" in text: return ""
            # 正则提取括号内的内容，丢弃中文标题
            match = re.search(r'\((.*?)\)', text)
            return match.group(1) if match else text

        parts = []

        # 1. 提取所有 Tag
        loc = extract(kwargs["地点选择"])
        season = extract(kwargs["季节"])
        weather = extract(kwargs["天气"])
        time_of_day = extract(kwargs["时间"])
        combo = extract(kwargs["光效_整体氛围"])
        single = extract(kwargs["光效_独立光源"])
        visual = extract(kwargs["画面画质风格"])
        texture = extract(kwargs["全局衣物质感"])
        atmo = extract(kwargs["氛围句"])
        manual = kwargs.get("手动补充", "").strip()

        # 2. 智能拼接 (构建画面层级)
        # 地点是基础
        if loc: parts.append(loc)

        # 季节/天气/时间 是环境修饰
        if season: parts.append(season)
        if weather: parts.append(weather)
        if time_of_day: parts.append(time_of_day)

        # 光效渲染
        if combo: parts.append(combo)
        if single: parts.append(single)

        # 风格和材质放在后面 (通常作为 LoRA 或 Checkpoint 的触发引导)
        if visual: parts.append(visual)
        if texture: parts.append(texture)

        # 氛围句 (anima Hybrid 自然语言层, 单句环境氛围, 与标签逗号混排)
        if atmo: parts.append(atmo)

        if manual: parts.append(manual)

        content = ", ".join(parts)
        weight = kwargs["环境总权重"]

        # 3. 权重逻辑
        # 权重滑杆全局加权 (权重1.0即纯标签); anima 用户权重设 1.0
        # 并在 画面画质风格 选 score_7/score_9, 即纯标签+官方评分词
        if not content:
            return ("",)

        if weight == 1.0:
            return (content,)
        else:
            # 全局加权
            return (f"({content}:{weight:g})",)

# 注册代码
NODE_CLASS_MAPPINGS = {
    "QianhuiSceneMaster": QianhuiSceneMaster,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QianhuiSceneMaster": "千绘 · 场景光效大师 (V6.8·Krea2画风扩容版)",
}
