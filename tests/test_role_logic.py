# -*- coding: utf-8 -*-
"""generate_role_name 逻辑单测（mock 依赖函数，无真实数据随机性）"""
import importlib.util, os, unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    "role_name_node", os.path.join(ROOT, "nodes", "role_name_node.py"))
rnn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rnn)

GROUPS = {
    "genshin_impact": ["hu_tao_(genshin_impact)", "mona_(genshin_impact)", "venti_(genshin_impact)"],
    "vocaloid": ["hatsune_miku", "kaito_(vocaloid)"],
    "dandadan": ["momoko_ayase"],
}
CATS = {
    "hu_tao_(genshin_impact)": {"female"},
    "mona_(genshin_impact)": {"female"},
    "venti_(genshin_impact)": {"male"},
    "hatsune_miku": {"female"},
    "kaito_(vocaloid)": {"male"},
    "momoko_ayase": {"female"},
}
OWNER = {}
for _cop, _rs in GROUPS.items():
    for _r in _rs:
        OWNER[_r] = _cop
CN = {"hu_tao_(genshin_impact)": "胡桃", "hatsune_miku": "初音未来", "momoko_ayase": "莫摩"}


class TestLogic(unittest.TestCase):
    def setUp(self):
        p = mock.patch.object(rnn, "load_role_data", return_value=GROUPS)
        p.start(); self.addCleanup(p.stop)
        p = mock.patch.object(rnn, "load_category_map", return_value=CATS)
        p.start(); self.addCleanup(p.stop)
        p = mock.patch.object(rnn, "load_owner_map", return_value=OWNER)
        p.start(); self.addCleanup(p.stop)
        p = mock.patch.object(rnn, "load_cn_map", return_value=CN)
        p.start(); self.addCleanup(p.stop)

    def _node(self):
        return rnn.QianhuiRandomCharacter()

    def test_empty_data(self):
        with mock.patch.object(rnn, "load_role_data", return_value={}):
            self.assertEqual(self._node().generate_role_name("🎲随机类型"),
                             ("(无角色数据)", "(无角色数据)"))

    def test_manual_role_skips_filter(self):
        # 指定角色跳过分类筛选: 选男性类目仍能出初音
        en, cn = self._node().generate_role_name(
            "🎯指定作品", 抽取分类="男",
            指定角色=rnn.role_display("hatsune_miku"))
        self.assertRegex(en, r"^hatsune miku")

    def test_manual_role_emits(self):
        en, cn = self._node().generate_role_name(
            "🎯指定作品", 指定作品=rnn.work_display("genshin_impact"),
            指定角色=rnn.role_display("venti_(genshin_impact)"))
        self.assertRegex(en, r"^venti")

    def test_manual_role_not_in_work_falls_to_work_random(self):
        en, cn = self._node().generate_role_name(
            "🎯指定作品", 指定作品=rnn.work_display("genshin_impact"),
            指定角色=rnn.role_display("hatsune_miku"))   # 初音不属于原神
        self.assertRegex(en, r"^(hu tao|mona|venti)\b")  # 变成原神内随机

    def test_work_random_respects_filter(self):
        en, cn = self._node().generate_role_name(
            "🎯指定作品", 抽取分类="女", 指定作品=rnn.work_display("genshin_impact"))
        self.assertRegex(en, r"^(hu tao|mona)\b")        # 排除温迪(男)

    def test_work_random_empty_after_filter(self):
        en, cn = self._node().generate_role_name(
            "🎯指定作品", 抽取分类="机械娘", 指定作品=rnn.work_display("vocaloid"))
        self.assertEqual(en, "(无匹配角色)")

    def test_fallback_when_nothing_specified(self):
        en, cn = self._node().generate_role_name("🎯指定作品", 抽取分类="不限")
        # 回退随机: 输出必是池内角色（转义后前缀匹配）
        self.assertRegex(en, r"^(hu tao|mona|venti|hatsune miku|kaito|momoko)\b")

    def test_random_mode_ignores_leftovers(self):
        en, cn = self._node().generate_role_name(
            "📺只出动漫", 指定角色=rnn.role_display("hatsune_miku"))  # 残留值
        self.assertRegex(en, r"^momoko")  # 只出动漫 = dandadan 的莫摩

    def test_filter_in_game_mode(self):
        en, cn = self._node().generate_role_name("🎮只出游戏", 抽取分类="男")
        self.assertRegex(en, r"^(venti|kaito)\b")

    def test_escape_unchanged(self):
        e = rnn.QianhuiRandomCharacter.escape_name
        self.assertEqual(e("hu_tao_(genshin_impact)", "SDXL"), "hu tao (genshin_impact)")
        self.assertEqual(e("hu_tao_(genshin_impact)", "anima (括号转义)"),
                         "hu tao\\(genshin_impact\\)")


class TestPayload(unittest.TestCase):
    def setUp(self):
        p = mock.patch.object(rnn, "load_owner_map", return_value=OWNER)
        p.start(); self.addCleanup(p.stop)
        p = mock.patch.object(rnn, "load_cn_map", return_value=CN)
        p.start(); self.addCleanup(p.stop)
        p = mock.patch.object(rnn, "load_work_cn", return_value={"genshin_impact": "原神"})
        p.start(); self.addCleanup(p.stop)
        p = mock.patch.object(rnn, "load_heat",
                              return_value={"hu_tao_(genshin_impact)": 5000,
                                            "mona_(genshin_impact)": 3000,
                                            "venti_(genshin_impact)": 4000,
                                            "hatsune_miku": 9999, "kaito_(vocaloid)": 1,
                                            "momoko_ayase": 100})
        p.start(); self.addCleanup(p.stop)

    def test_payload_shape(self):
        p = rnn.build_payload()
        self.assertEqual(len(p["works"]), 3)
        wmap = {w[0]: w[1] for w in p["works"]}
        self.assertEqual(wmap["genshin_impact"], "原神")
        self.assertIsNone(wmap["vocaloid"])   # 未提供中文名 -> null
        self.assertEqual(len(p["roles_by_work"]["genshin_impact"]), 3)
        names = [r[0] for r in p["roles_by_work"]["genshin_impact"]]
        self.assertEqual(names, ["hu_tao_(genshin_impact)", "venti_(genshin_impact)",
                                 "mona_(genshin_impact)"])   # 按热度降序


if __name__ == "__main__":
    unittest.main()
