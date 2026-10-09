# -*- coding: utf-8 -*-
"""按路径装载包/模块，绕开需要 ComfyUI 运行时的包导入。

spec_from_file_location 配合 submodule_search_locations 可构造真正的包，
使包内 `from .dct import ...` 相对导入正常工作。
"""
import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_package(pkg_name: str, pkg_dir: str):
    """把 pkg_dir 当作名为 pkg_name 的包装载并注册进 sys.modules。"""
    init_py = os.path.join(pkg_dir, "__init__.py")
    spec = importlib.util.spec_from_file_location(
        pkg_name, init_py, submodule_search_locations=[pkg_dir]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[pkg_name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_module(mod_name: str, path: str):
    """按路径装载单个模块（要求该模块无包内相对导入）。"""
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def pkgs_dir(*parts: str) -> str:
    return os.path.join(ROOT, "nodes", *parts)
