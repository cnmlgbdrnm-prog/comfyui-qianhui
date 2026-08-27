# -*- coding: utf-8 -*-
"""role_category.json / work_cn.json 质量检查"""
import collections, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
ALLOWED = {"female", "male", "beast", "robot"}

def main():
    owner = json.load(open(os.path.join(DATA, "role_owner.json"), encoding="utf-8"))
    names = list(owner.keys())

    try:
        cat = json.load(open(os.path.join(DATA, "role_category.json"), encoding="utf-8"))
    except FileNotFoundError:
        print("FAIL: role_category.json 不存在")
        return 1
    bad_keys = set(cat) - set(owner)
    bad_vals = {k: v for k, v in cat.items() if not set(v) <= ALLOWED}
    labeled = sum(1 for k in names if cat.get(k))
    counts = collections.Counter()
    for v in cat.values():
        counts.update(v)
    print(f"角色总数: {len(names)}  标注条目: {len(cat)}")
    print(f"无效键: {len(bad_keys)}  无效标签值: {len(bad_vals)}")
    print(f"有标签角色: {labeled} ({labeled/len(names)*100:.1f}%)")
    print(f"分布: 女 {counts['female']}  男 {counts['male']}  兽类 {counts['beast']}  机械 {counts['robot']}")
    print(f"未标注(未知): {len(names) - labeled}")

    try:
        wcn = json.load(open(os.path.join(DATA, "work_cn.json"), encoding="utf-8"))
    except FileNotFoundError:
        print("FAIL: work_cn.json 不存在")
        return 1
    firsts = {v.split()[0] for v in owner.values()}  # 角色分组实际以首 tag 为作品
    bad_wkeys = set(wcn) - firsts
    print(f"作品数: {len(firsts)}  work_cn 条目: {len(wcn)}  无效键: {len(bad_wkeys)}")

    ok = not (bad_keys or bad_vals or bad_wkeys or labeled < len(names) * 0.85)
    print("OK 合格" if ok else "FAIL 存在不合格项")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
