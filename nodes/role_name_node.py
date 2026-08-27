"""
千绘 · 角色随机模块 (角色随机师 QianhuiRandomCharacter)

模块职责：两级随机——先按模式抽取作品类型(游戏/动漫/其他)，
再随机作品，最后从该作品的角色池随机抽一个 Danbooru 高频角色名
(post_count >= 2000)。角色名按模型类型转义：SDXL 普通括号、
anima 括号加反斜杠(hu_tao_(genshin_impact) -> hu tao\\(genshin_impact\\))
防提示词权重语法误解析；主名下划线转空格、括号消歧义保留下划线。
数据流：data/role_owner.json(角色->版权) + data/role_names.json(高频角色)
-> 按版权分组 -> 按模式随机 -> 按模型类型转义输出 STRING。
外部依赖：无（仅标准库）。
"""

import json
import os
import random
import re
import time

# 路径定义 (节点位于 nodes/ 子目录, 数据在插件根目录)
CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(CURRENT_DIR, "data")
ROLE_OWNER_PATH = os.path.join(DATA_DIR, "role_owner.json")
ROLE_NAMES_PATH = os.path.join(DATA_DIR, "role_names.json")
ROLE_CN_PATH = os.path.join(DATA_DIR, "role_names_cn.json")

# 版权 tag -> 作品类型 (以 Danbooru 实际版权标签为准; 未标注默认归入"其他")
GAME_TAGS = {
    # 米哈游系
    "genshin_impact", "honkai:_star_rail", "honkai_(series)", "honkai_3rd",
    "zenless_zone_zero", "mihoyo",
    # 国产/手游系
    "blue_archive", "arknights", "girls'_frontline", "goddess_of_victory:_nikke",
    "wuthering_waves", "snowbreak", "reverse:1999", "punishing_gray_raven",
    "alchemy_stars", "counter_side", "epic_seven", "brown_dust", "last_origin",
    "destiny_child", "girl_cafe_gun", "nightmare_sea", "granblue_fantasy",
    "princess_connect!", "love_live!", "bang_dream!", "gakuen_idolmaster",
    # 舰娘/马娘/偶像
    "azur_lane", "kantai_collection", "umamusume", "umamusume_pretty_derby",
    "idolmaster", "idol_master_cinderella_girls", "idol_master_million_live",
    "idol_master_shiny_colors", "idol_master_side_m",
    # Fate 系 (含小说/游戏混合, 统一按游戏)
    "fate/grand_order", "fate/stay_night", "fate/zero", "fate/apocrypha",
    "fate/extra", "fate/hollow_ataraxia", "fate_(series)", "fate/kaleid_liner_prisma_illya",
    "tsukihime", "mahou_tsukai_no_yoru", "melty_blood",
    # 东方系
    "touhou", "double_spoiler", "cookie_(touhou)", "legacy_of_lunatic_kingdom",
    # 任天堂/主机
    "animal_crossing", "mario_&_luigi_rpg", "mario_(series)", "splatoon_(series)",
    "kirby_(series)", "pokemon", "the_legend_of_zelda", "metroid", "fire_emblem",
    "xenoblade_chronicles_(series)", "kid_icarus", "arms_(game)", "super_mario",
    # 日系游戏
    "omori", "deltarune", "undertale", "nier:automata", "ragnarok_online",
    "needy_girl_overdose", "doki_doki_literature_club", "a_hat_in_time",
    "hollow_knight", "disgaea", "helltaker", "senran_kagura", "heaven_burns_red",
    "atelier_(series)", "monster_hunter_(series)", "crisis_core:_final_fantasy_vii",
    "doom_(series)", "devil_may_cry_(series)", "mega_man_(series)", "mega_man_(classic)",
    "fatal_fury", "darkstalkers", "yu-gi-oh!", "sonic_(series)", "to_heart_(series)",
    "amagami", "little_busters!", "science_adventure", "choujigen_game_neptune",
    "shantae_(series)", "guilty_gear", "blazblue", "king_of_fighters", "street_fighter",
    "final_fantasy", "kingdom_hearts", "persona", "shin_megami_tensei", "tales_series",
    "legend_of_zelda", "darksouls", "elden_ring", "monster_hunter",
    # Vtuber/音源
    "hololive", "nijisanji", "indie_virtual_youtuber", "vocaloid", "project_diva",
    "magical_mirai_(vocaloid)", "voiceroid", "synthesizer_v", "utau", "cevio",
    # 欧美游戏
    "league_of_legends", "valorant", "overwatch", "world_of_warcraft", "diablo",
    "starcraft", "minecraft", "terraria", "fortnite", "apex_legends",
    # 其他
    "neurosama", "galaxy_angel", "monster_girl_encyclopedia", "sakura_wars",
    "dragalia_lost", "osu", "neural_cloud", "krankenhaus", "diamond_wa_kudakenai",
    # 抓取核对后补充 (2026-08-17 全量汉化核对时发现归类缺失)
    "project_sekai", "resident_evil", "cho_kaguya-hime!", "limbus_company",
    "project_moon", "infinity_nikki", "densetsu.exe", "awoke_(love_live!)",
    "yume_nikki", "ib_(kouri)", "mogeko_castle", "satsuriku_no_tenshi",
    "hello_charlotte",
}

ANIME_TAGS = {
    # 新番/热门
    "sousou_no_frieren", "frieren", "dandadan", "solo_leveling", "tower_of_god",
    "oshi_no_ko", "bocchi_the_rock!", "chainsaw_man", "spy_x_family",
    "jujutsu_kaisen", "demon_slayer", "kimetsu_no_yaiba", "attack_on_titan",
    "shingeki_no_kyojin", "solo_leveling_anime",
    # 长篇/经典
    "naruto", "naruto_(series)", "one_piece", "dragon_ball", "bleach",
    "detective_conan", "gintama", "jojo", "jojo_no_kimyou_na_bouken",
    "battle_tendency", "fullmetal_alchemist", "hunter_x_hunter",
    "neon_genesis_evangelion", "evangelion:_3.0_you_can_(not)_redo",
    "code_geass", "steins_gate", "death_note", "sailor_moon",
    "bishoujo_senshi_sailor_moon", "cardcaptor_sakura", "ranma_1/2",
    "urusei_yatsura", "city_hunter", "macross", "getter_robo", "gundam",
    # 异世界/轻改
    "re:zero_kara_hajimeru_isekai_seikatsu", "konosuba",
    "kono_subarashii_sekai_ni_shukufuku_wo!", "overlord", "tensura",
    "mushoku_tensei", "tate_no_yuusha_no_nariagari", "shield_hero",
    "made_in_abyss", "no_game_no_life", "log_horizon", "dungeon_meshi",
    "that_time_i_got_reincarnated", "kumo_desu_ga",
    # 校园/恋爱/日常
    "kaguya-sama_wa_kokurasetai_~tensai-tachi_no_renai_zunousen~",
    "kaguya_sama", "horimiya", "komi-san_wa_komyushou_desu", "toradora!",
    "nisekoi", "saekano", "bunny_girl_senpai", "oregairu",
    "yahari_ore_no_seishun_lovecome_wa_machigatteiru.",
    "classroom_of_the_elite", "hyouka", "k-on!", "lucky_star",
    "suzumiya_haruhi_no_yuuutsu", "suzumiya_haruhi_no_shoushitsu",
    "clannad", "angel_beats!", "anohana", "your_lie_in_april", "charlotte",
    "sakurasou", "pet_girl_of_sakurasou", "gochuumon_wa_usagi_desu_ka?",
    "yuruyuri", "yurucamp", "machikado_mazoku", "sono_bisque_doll_wa_koi_wo_suru",
    "ijiranaide_nagatoro-san", "boku_wa_tomodachi_ga_sukunai",
    "ore_no_imouto_ga_konna_ni_kawaii_wake_ga_nai", "eromanga_sensei",
    "high_school_dxd", "to_love_ru", "date_a_live", "zero_no_tsukaima",
    "guilty_crown", "shakugan_no_shana", "kobayashi-san_chi_no_maidragon",
    "onii-chan_wa_oshimai!", "watashi_ga_motenai_no_wa_dou_kangaetemo_omaera_ga_warui!",
    # 奇幻/战斗/黑深残
    "sword_art_online", "darling_in_the_franxx", "kill_la_kill", "promare",
    "gurren_lagann", "tengen_toppa_gurren_lagann", "devilman", "berserk",
    "claymore", "goblin_slayer", "vinland_saga", "dorohedoro", "fire_force",
    "black_clover", "one_punch_man", "mob_psycho_100", "my_hero_academia",
    "boku_no_hero_academia", "haikyuu", "kuroko_no_basket", "tokyo_ghoul",
    "tokyo_revengers", "danganronpa_(series)", "danganronpa:_trigger_happy_havoc",
    "puella_magi", "mahou_shoujo_madoka_magica",
    "magia_record:_mahou_shoujo_madoka_magica_gaiden", "dungeon_ni_deai_wo_motomeru_no_wa_machigatteiru_darou_ka",
    # 治愈/其他
    "girls_und_panzer", "strike_witches", "strike_witches:_aurora_no_majo",
    "lycoris_recoil", "kemono_friends", "monogatari_(series)", "kizumonogatari",
    "panty_&_stocking_with_garterbelt", "gridman_universe", "chiikawa",
    "the_hundred_line_-last_defense_academy-", "gegege_no_kitarou",
    "axis_powers_hetalia", "kiratto_pri_chan", "bungou_stray_dogs",
    "black_rock_shooter", "spice_and_wolf", "ghost_in_the_shell",
    "lyrical_nanoha", "toaru_kagaku_no_railgun", "toaru_majutsu_no_index",
    "go-toubun_no_hanayome", "saibou_shinkyoku", "senki_zesshou_symphogear",
    "shoujo_shuumatsu_ryokou", "umineko_no_naku_koro_ni", "higurashi_no_naku_koro_ni",
    "houseki_no_kuni", "ansatsu_kyoushitsu", "hibike!_euphonium",
    "little_witch_academia", "kara_no_kyoukai", "osomatsu-san",
    "kusuriya_no_hitorigoto", "poptepipic", "rozen_maiden",
    "chuunibyou_demo_koi_ga_shitai!", "ganbare!_nakamura-kun!!", "rwby",
    "alien_stage", "grandmaster_of_demonic_cultivation", "omniscient_reader's_viewpoint",
    "arcane:_league_of_legends", "cyberpunk:_edgerunners", "star_wars_visions",
    "adventure_time", "steven_universe", "avatar_the_last_airbender",
    "the_legend_of_korra", "my_little_pony", "cars_(movie)", "spirited_away",
    "howls_moving_castle", "princess_mononoke", "your_name", "kimi_no_na_wa",
    "violet_evergarden", "made_in_abyss_anime", "blacksad_(comic)",
    "marvel", "deadpool_(series)", "batman_(series)", "avengers_(series)",
    "american_psycho", "dokidoki!_precure", "futari_wa_precure",
    "heartcatch_precure!", "hirogaru_sky!_precure", "meitantei_precure!",
    "precure", "shinryaku!_ikamusume", "gin_ni_ii_made",
    # 抓取核对后补充 (2026-08-17 全量汉化核对时发现归类缺失)
    "girls_band_cry", "one-punch_man", "the_amazing_digital_circus",
}

def load_role_data():
    """加载角色与归属, 按版权分组: {copyright_tag: [角色名...]}"""
    if not os.path.exists(ROLE_OWNER_PATH) or not os.path.exists(ROLE_NAMES_PATH):
        return {}
    with open(ROLE_NAMES_PATH, "r", encoding="utf-8") as f:
        roles = json.load(f).get("role_names", [])
    with open(ROLE_OWNER_PATH, "r", encoding="utf-8") as f:
        owner = json.load(f)
    groups = {}
    for role in roles:
        tag = role["name"]
        cop = owner.get(tag, "")
        first = cop.split()[0] if cop else ""
        if not first:
            continue
        groups.setdefault(first, []).append(tag)
    return groups

def _type_of(cop_tag):
    """版权 tag -> 作品类型"""
    if cop_tag in GAME_TAGS:
        return "game"
    if cop_tag in ANIME_TAGS:
        return "anime"
    return "other"

_CN_CACHE = None

def load_cn_map():
    """加载角色中文名映射 (全量汉化, 缺失时回退英文)"""
    global _CN_CACHE
    if _CN_CACHE is None:
        try:
            if os.path.exists(ROLE_CN_PATH):
                with open(ROLE_CN_PATH, "r", encoding="utf-8") as f:
                    _CN_CACHE = json.load(f)
            else:
                _CN_CACHE = {}
        except Exception:
            _CN_CACHE = {}
    return _CN_CACHE


CATEGORY_PATH = os.path.join(DATA_DIR, "role_category.json")
WORK_CN_PATH = os.path.join(DATA_DIR, "work_cn.json")

SENTINEL = "（未指定）"

# 抽取分类 -> 所需角色标签（"不限"不过滤）；"兽娘"=女+兽类, "机械娘"=女+机械
CATEGORY_TAGS = {
    "不限": None,
    "女": ("female",),
    "男": ("male",),
    "兽类·福瑞": ("beast",),
    "机械·人造人": ("robot",),
    "兽娘": ("female", "beast"),
    "机械娘": ("female", "robot"),
}

_CAT_CACHE = None
_WORK_CN_CACHE = None
_OWNER_CACHE = None
_HEAT_CACHE = None


def load_category_map():
    """角色名 -> 标签 set; 文件缺失返回 {} (模块级缓存)"""
    global _CAT_CACHE
    if _CAT_CACHE is None:
        try:
            if os.path.exists(CATEGORY_PATH):
                with open(CATEGORY_PATH, "r", encoding="utf-8") as f:
                    _CAT_CACHE = {k: set(v) for k, v in json.load(f).items()}
            else:
                _CAT_CACHE = {}
        except Exception:
            _CAT_CACHE = {}
    return _CAT_CACHE


def load_work_cn():
    """作品 tag -> 中文名; 文件缺失返回 {} (模块级缓存)"""
    global _WORK_CN_CACHE
    if _WORK_CN_CACHE is None:
        try:
            if os.path.exists(WORK_CN_PATH):
                with open(WORK_CN_PATH, "r", encoding="utf-8") as f:
                    _WORK_CN_CACHE = json.load(f)
            else:
                _WORK_CN_CACHE = {}
        except Exception:
            _WORK_CN_CACHE = {}
    return _WORK_CN_CACHE


def load_owner_map():
    """角色名 -> 版权串(首段为作品 tag) (模块级缓存)"""
    global _OWNER_CACHE
    if _OWNER_CACHE is None:
        try:
            with open(ROLE_OWNER_PATH, "r", encoding="utf-8") as f:
                _OWNER_CACHE = json.load(f)
        except Exception:
            _OWNER_CACHE = {}
    return _OWNER_CACHE


def load_heat():
    """角色名 -> post_count (模块级缓存, 用于热门排序)"""
    global _HEAT_CACHE
    if _HEAT_CACHE is None:
        try:
            with open(ROLE_NAMES_PATH, "r", encoding="utf-8") as f:
                _HEAT_CACHE = {r["name"]: r["post_count"]
                               for r in json.load(f).get("role_names", [])}
        except Exception:
            _HEAT_CACHE = {}
    return _HEAT_CACHE


def work_display(tag):
    """作品下拉展示: 中文名 (tag); 无中文名回退 tag"""
    cn = load_work_cn().get(tag)
    return f"{cn} ({tag})" if cn else tag


def role_display(name):
    """角色下拉展示: 中文名 (danbooru名); 无中文名回退原名"""
    cn = load_cn_map().get(name, "")
    return f"{cn} ({name})" if cn else name


def parse_work_display(v):
    """作品展示值 -> tag; SENTINEL/空 -> None（取最后一个 ' (' 拆分, 剥一个展示追加的 ')'）"""
    if v in (None, SENTINEL):
        return None
    i = v.rfind(" (")
    if i == -1:
        return v.strip()
    t = v[i + 2:].strip()
    if t.endswith(")"):
        t = t[:-1].strip()
    return t


def parse_role_display(v):
    """角色展示值 -> danbooru 名; SENTINEL/空 -> None"""
    if v in (None, SENTINEL):
        return None
    i = v.rfind(" (")
    if i == -1:
        return v.strip()
    t = v[i + 2:].strip()
    if t.endswith(")"):
        t = t[:-1].strip()
    return t


def filter_by_category(names, cat_key, cats=None):
    """按抽取分类过滤: 不限 返回原列表; 否则要求所需标签全命中"""
    cats = cats if cats is not None else load_category_map()
    need = CATEGORY_TAGS.get(cat_key)
    if not need:
        return list(names)
    need = set(need)
    return [n for n in names if need.issubset(cats.get(n, set()))]


class QianhuiRandomCharacter:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        works = sorted({v.split()[0] for v in load_owner_map().values()})
        work_opts = [SENTINEL] + [work_display(w) for w in works]
        names = sorted(load_owner_map().keys(),
                       key=lambda n: (0 - load_heat().get(n, 0), n))
        role_opts = [SENTINEL] + [role_display(n) for n in names]
        return {
            "required": {
                "作品类型": (["🎲随机类型", "🎮只出游戏", "📺只出动漫", "🎯指定作品"],),
                "抽取分类": (list(CATEGORY_TAGS.keys()),),
                "模型类型": (["SDXL", "anima (括号转义)"],),
                "指定作品": (work_opts,),
                "指定角色": (role_opts,),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("角色名(英文)", "角色名(中文)")
    FUNCTION = "generate_role_name"
    CATEGORY = "千绘/角色"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return time.time()

    @staticmethod
    def escape_name(name, model_type="anima (括号转义)"):
        """按模型类型转义角色名

        SDXL: hu_tao_(genshin_impact) -> hu tao (genshin_impact) (普通括号)
        anima: hu_tao_(genshin_impact) -> hu tao\\(genshin_impact\\) (括号加反斜杠)
        hina_(swimsuit)_(blue_archive) -> hina (swimsuit) (blue_archive) / hina\\(swimsuit\\)\\(blue_archive\\)
        规则: 主名与括号间/外下划线转空格; 括号内下划线保留(作品名约定写法);
        anima 对所有括号(含嵌套)做反斜杠转义, 防提示词权重语法误解析。
        """
        idx = name.find("(")
        if idx == -1:
            return name.replace("_", " ")
        main = name[:idx].replace("_", " ").rstrip()
        # 括号间/外的下划线是 tag 分隔符: SDXL 转空格, anima 删除(括号紧贴);
        # 括号内下划线保留(作品名约定写法)
        dis = name[idx:]
        sep = "" if "anima" in model_type else " "
        out = []
        in_paren = False
        for ch in dis:
            if ch == "(":
                in_paren = True
            elif ch == ")":
                in_paren = False
            if ch == "_" and not in_paren:
                ch = sep
            out.append(ch)
        dis = "".join(out)
        if "anima" in model_type:
            return main + dis.replace("(", r"\(").replace(")", r"\)")
        return main + " " + dis

    def generate_role_name(self, 作品类型, 抽取分类="不限", 模型类型="SDXL",
                           指定作品=SENTINEL, 指定角色=SENTINEL):
        groups = load_role_data()
        if not groups:
            return ("(无角色数据)", "(无角色数据)")
        cats = load_category_map()

        if 作品类型 == "🎯指定作品":
            role_name = parse_role_display(指定角色)
            work_tag = parse_work_display(指定作品)
            if role_name:
                if work_tag:
                    if role_name in groups.get(work_tag, []):
                        return self._output(role_name, work_tag, 模型类型)
                    role_name = None   # 角色不在所选作品 -> 走作品内随机
                else:
                    return self._output(role_name, None, 模型类型)
            if work_tag and work_tag in groups:
                pool = filter_by_category(groups[work_tag], 抽取分类, cats)
                if not pool:
                    return ("(无匹配角色)", "(无匹配角色)")
                return self._output(random.choice(pool), work_tag, 模型类型)
            作品类型 = "🎲随机类型"   # 都未指定 -> 回退随机类型

        # 按类型聚合（V1 加权随机 ≈ 类型内角色均匀抽取）
        by_type = {"game": {}, "anime": {}, "other": {}}
        for cop, roles in groups.items():
            by_type[_type_of(cop)][cop] = roles

        if 作品类型 == "🎲随机类型":
            weights = {t: sum(len(r) for r in pool.values()) for t, pool in by_type.items()}
            total = sum(weights.values())
            if total == 0:
                return ("(无角色数据)", "(无角色数据)")
            roll = random.random() * total
            acc = 0
            chosen_type = "other"
            for t, w in weights.items():
                acc += w
                if roll < acc:
                    chosen_type = t
                    break
        else:
            chosen_type = "game" if 作品类型 == "🎮只出游戏" else "anime"

        pool = by_type.get(chosen_type, {})
        names = [n for rs in pool.values() for n in rs]
        names = filter_by_category(names, 抽取分类, cats)
        if not names:
            return ("(无匹配角色)", "(无匹配角色)")
        return self._output(random.choice(names), None, 模型类型)

    def _output(self, role, cop, 模型类型):
        """统一输出: (英文提示词名, 中文显示名)；复刻 V1 格式。
        cop 未知时从 owner 首tag 查询; 无括号角色追加 (版权tag) 消歧义并同样转义。"""
        en = self.escape_name(role, 模型类型)
        if not cop:
            cop = load_owner_map().get(role, "").split()[0]
        if "(" not in role and cop:
            suffix = f"({cop})"
            if "anima" in 模型类型:
                suffix = suffix.replace("(", r"\(").replace(")", r"\)")
            en = en + " " + suffix
        cn = load_cn_map().get(role, "")
        cn_display = f"{cn}({cop})" if cn else en
        return (en, cn_display)

def build_payload():
    """前端联动数据: 作品列表 + 每作品角色列表（均含中文名, 无则 null）"""
    owner = load_owner_map()
    cn = load_cn_map()
    wcn = load_work_cn()
    works = sorted({v.split()[0] for v in owner.values()})
    roles_by_work = {}
    for tag in works:
        rs = [n for n, w in owner.items() if w.split()[0] == tag]
        rs.sort(key=lambda n: (0 - load_heat().get(n, 0), n))
        roles_by_work[tag] = [[n, cn.get(n) or None] for n in rs]
    return {"works": [[t, wcn.get(t)] for t in works],
            "roles_by_work": roles_by_work}
