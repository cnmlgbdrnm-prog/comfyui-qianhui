# -*- coding: utf-8 -*-
"""数据文件结构校验（unittest，纯标准库）"""
import json, os, unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALLOWED = {"female", "male", "beast", "robot"}

class TestRoleData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(os.path.join(ROOT, "data", "role_owner.json"), encoding="utf-8") as f:
            cls.owner = json.load(f)

    def test_category_keys_and_values(self):
        with open(os.path.join(ROOT, "data", "role_category.json"), encoding="utf-8") as f:
            cat = json.load(f)
        self.assertEqual(set(cat), set(self.owner))  # 全量覆盖
        for v in cat.values():
            self.assertIsInstance(v, list)
            self.assertTrue(set(v) <= ALLOWED, f"非法标签: {v}")

    def test_category_coverage(self):
        with open(os.path.join(ROOT, "data", "role_category.json"), encoding="utf-8") as f:
            cat = json.load(f)
        labeled = sum(1 for k in self.owner if cat.get(k))
        self.assertGreaterEqual(labeled, len(self.owner) * 0.85, "有标签覆盖率不足 85%")

    def test_gender_ratio_sane(self):
        with open(os.path.join(ROOT, "data", "role_category.json"), encoding="utf-8") as f:
            cat = json.load(f)
        fem = sum(1 for v in cat.values() if "female" in v)
        mal = sum(1 for v in cat.values() if "male" in v)
        self.assertGreater(fem, mal)
        self.assertGreaterEqual(fem, 800)

    def test_beast_robot_counts(self):
        with open(os.path.join(ROOT, "data", "role_category.json"), encoding="utf-8") as f:
            cat = json.load(f)
        self.assertGreaterEqual(sum(1 for v in cat.values() if "beast" in v), 100)
        self.assertGreaterEqual(sum(1 for v in cat.values() if "robot" in v), 20)

    def test_work_cn(self):
        with open(os.path.join(ROOT, "data", "work_cn.json"), encoding="utf-8") as f:
            wcn = json.load(f)
        firsts = {v.split()[0] for v in self.owner.values()}  # 首tag = 实际作品键
        self.assertLessEqual(set(wcn), firsts)  # 键必须是真实作品首 tag
        self.assertGreaterEqual(len(wcn), 0.9 * len(firsts))

if __name__ == "__main__":
    unittest.main()
