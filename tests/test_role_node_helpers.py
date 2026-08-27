# -*- coding: utf-8 -*-
"""role_name_node.py 数据层助手单测（按文件路径加载）"""
import importlib.util, os, unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    "role_name_node", os.path.join(ROOT, "nodes", "role_name_node.py"))
rnn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rnn)


class TestHelpers(unittest.TestCase):
    def test_sentinel_and_category_tags(self):
        self.assertEqual(rnn.SENTINEL, "（未指定）")
        self.assertEqual(rnn.CATEGORY_TAGS["兽娘"], ("female", "beast"))
        self.assertIsNone(rnn.CATEGORY_TAGS["不限"])

    def test_load_maps_have_data(self):
        cat = rnn.load_category_map()
        self.assertIn("female", cat.get("hatsune_miku", []))   # 初音未来必须是女性
        wcn = rnn.load_work_cn()
        self.assertEqual(wcn.get("genshin_impact"), "原神")

    def test_display_roundtrip_all_roles(self):
        """全量 角色/作品 展示->解析往返一致（关键正确性网）"""
        for name, cop in rnn.load_owner_map().items():
            self.assertEqual(cop.split()[0], rnn.parse_work_display(rnn.work_display(cop.split()[0])), cop)
            disp = rnn.role_display(name)
            self.assertEqual(name, rnn.parse_role_display(disp), disp)

    def test_display_parse(self):
        self.assertIsNone(rnn.parse_work_display(rnn.SENTINEL))
        self.assertIsNone(rnn.parse_role_display(rnn.SENTINEL))
        self.assertEqual(rnn.parse_work_display(rnn.work_display("genshin_impact")), "genshin_impact")
        # 含消歧义括号的角色名
        self.assertEqual(rnn.parse_role_display("阿尔托莉雅 (artoria_pendragon_(fate))"),
                         "artoria_pendragon_(fate)")
        # 无中文名回退纯英文（无 " (" 时整串原样）
        self.assertEqual(rnn.parse_role_display("hatsune_miku"), "hatsune_miku")

    def test_filter_by_category(self):
        names = ["a", "b", "c", "d"]
        cats = {"a": {"female"}, "b": {"male", "beast"}, "c": {"female", "beast"}, "d": set()}
        self.assertEqual(rnn.filter_by_category(names, "不限", cats), ["a", "b", "c", "d"])
        self.assertEqual(rnn.filter_by_category(names, "女", cats), ["a", "c"])
        self.assertEqual(rnn.filter_by_category(names, "兽类·福瑞", cats), ["b", "c"])
        self.assertEqual(rnn.filter_by_category(names, "兽娘", cats), ["c"])
        self.assertEqual(rnn.filter_by_category(names, "机械娘", cats), [])


if __name__ == "__main__":
    unittest.main()
