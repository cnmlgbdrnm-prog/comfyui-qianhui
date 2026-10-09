# -*- coding: utf-8 -*-
"""4 节点控件精简的回归测试。

这 4 个节点只依赖 re/json/os/random/time，无 ComfyUI 依赖，可按路径直载。
"""
import importlib.util
import io
import os
import re
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODES_DIR = os.path.join(ROOT, "nodes")

FILES = {
    "character": ("character_node.py", "QianhuiCharacterCreator", "generate_character"),
    "pose": ("pose_node.py", "QianhuiActionMaster", "generate_pose_v8"),
    "scene": ("scene_node.py", "QianhuiSceneMaster", "generate_scene_v6"),
    "combat": ("combat_node.py", "QianhuiCombatMaster", "generate_combat"),
}

PREFIX_RE = re.compile(r"^\d+[a-z]?_")


def load_node(module_file: str):
    """按路径装载节点模块（无包内相对导入，可直接 exec）。"""
    path = os.path.join(NODES_DIR, module_file)
    name = "qh_" + os.path.splitext(module_file)[0]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def node_class(key: str):
    module_file, cls_name, _ = FILES[key]
    return getattr(load_node(module_file), cls_name)


def gen_func(key: str):
    module_file, _, fn_name = FILES[key]
    return getattr(load_node(module_file), fn_name)


def all_keys(cls) -> list:
    spec = cls.INPUT_TYPES()
    keys = []
    for section in ("required", "optional"):
        keys.extend(spec.get(section, {}).keys())
    return keys


def _first_real(options) -> str:
    """下拉默认取第一个非占位选项，使生成结果非空、断言才有意义。"""
    for o in options:
        if isinstance(o, str) and o and "(不指定)" not in o:
            return o
    return options[0] if options else ""


def default_kwargs(cls) -> dict:
    """从 INPUT_TYPES 构造一份默认 kwargs：下拉取首个非占位项，其余取 default。"""
    kw = {}
    spec = cls.INPUT_TYPES()
    for section in ("required", "optional"):
        for name, decl in spec.get(section, {}).items():
            typ = decl[0]
            opts = decl[1] if len(decl) > 1 and isinstance(decl[1], dict) else {}
            kw[name] = _first_real(typ) if isinstance(typ, list) else opts.get("default", "")
    return kw


# 67 个旧键名，用于断言源码中无残留引用
OLD_KEYS = {
    "character_node.py": {
        "00_工作安全策略", "01_身份", "02_年龄与人设", "03_种族", "04_体型",
        "05_肤色", "06_胸部", "07_发型", "08_发色", "09_头发长度", "10_瞳色",
        "11_瞳孔样式", "12_面部特征", "13_表情", "14_套装选择", "15_套装颜色",
        "16_上装款式", "17_上装颜色", "18_下装款式", "19_下装颜色", "20_服装材质",
        "21_腿部特征", "22_袜饰", "23_袜饰颜色", "24_鞋子", "25_鞋子颜色",
        "26_头饰", "27_面饰", "28_颈饰", "29_手部饰品", "30_特殊饰品",
        "31_构图范围", "32_Lora触发词", "33_手动补充",
    },
    "pose_node.py": {
        "00_🎲随机动作模式", "07_🎲随机表情开关", "05_特殊视角", "01_全身基础",
        "02_头部/视线", "03_👐双手动作", "03_👈左手动作", "03_👉右手动作",
        "04_🦵双脚动作", "04_🦵左脚动作", "04_🦵右脚动作", "08_表情细节",
        "06_手动补充", "09_模型类型",
    },
    "scene_node.py": {
        "01_地点选择", "02a_季节", "02b_天气", "02c_时间", "03_光效_整体氛围",
        "04_光效_独立光源", "05_画面画质风格", "06_全局衣物质感", "07_手动补充",
        "08_环境总权重", "09_氛围句",
    },
    "combat_node.py": {
        "00_🔥激战模式", "01_🗡️武器装备", "02_👊攻击姿态", "03_🩸战损程度",
        "04_✨技能特效", "05_📸战斗运镜", "07_模型类型", "06_补充指令",
    },
}


class TestNoNumericPrefix(unittest.TestCase):
    def test_all_keys_have_no_prefix(self):
        for key in FILES:
            bad = [k for k in all_keys(node_class(key)) if PREFIX_RE.match(k)]
            self.assertEqual(bad, [], f"{key} 仍有带前缀的键: {bad}")

    def test_source_has_no_old_key_leftovers(self):
        """源码中不得残留任何旧键名——含单引号、字符串拼接、注释与 docstring"""
        # 注意：这里必须用裸 `k in src`，不能写成 f'"{k}"' in src。
        # 带引号的写法只覆盖双引号字面量，漏掉单引号、拼接和注释里的旧名，
        # 而本用例的承诺是「源码中不残留」——覆盖面必须与承诺一致。
        for module_file, old in OLD_KEYS.items():
            src = io.open(os.path.join(NODES_DIR, module_file), encoding="utf-8").read()
            left = sorted(k for k in old if k in src)
            self.assertEqual(left, [], f"{module_file} 残留旧键名: {left}")

    def test_key_count_unchanged(self):
        """只改名不增删，键数应与改动前一致"""
        expected = {"character": 34, "pose": 14, "scene": 11, "combat": 8}
        for key, n in expected.items():
            self.assertEqual(len(all_keys(node_class(key))), n, f"{key} 键数变化")


if __name__ == "__main__":
    unittest.main()
