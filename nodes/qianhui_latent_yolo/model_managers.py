"""YOLO / SAM 模型文件管理与扫描（目录发现、缓存与版本识别）。"""
import glob
import os

import folder_paths

# 源文件 `水印/nodes.py` 就放在插件根目录，所以它那句
# `os.path.dirname(__file__)` 指的就是**插件根**，`models/sam` 即
# `<插件根>/models/sam`。搬进 `nodes/qianhui_latent_yolo/` 后 `__file__`
# 深了两层，直接照搬会静默指向 `nodes/qianhui_latent_yolo/models/sam`
# ——一个永远不会存在的地方。这里补回两层，恢复原有语义。
# （这是搬运中唯一一处**必须**改动的路径解析：字节对拍看不出这种错，
#   因为方法体一个字没变，变的是 `__file__` 本身指到哪。）
_PLUGIN_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# YOLO模型管理
# ============================================================

class YOLOModelManager:
    """YOLO模型管理器"""
    
    SUPPORTED_EXTENSIONS = ('.pt', '.onnx', '.engine', '.torchscript')
    MODEL_DIRS = [
        "models/yolo",
        "models/ultralytics",
        "custom_nodes/yolo_models",
        "models/detection",
    ]
    
    _cached_models = None
    _last_scan_time = 0
    
    @classmethod
    def get_model_dir(cls):
        base_path = folder_paths.models_dir
        yolo_dir = os.path.join(base_path, "yolo")
        if not os.path.exists(yolo_dir):
            try:
                os.makedirs(yolo_dir)
            except:
                pass
        return yolo_dir
    
    @classmethod
    def scan_models(cls, force_refresh=False):
        import time
        current_time = time.time()
        if not force_refresh and cls._cached_models and (current_time - cls._last_scan_time) < 5:
            return cls._cached_models
        
        models = {}
        comfy_root = os.path.dirname(folder_paths.models_dir)
        
        for rel_dir in cls.MODEL_DIRS:
            full_dir = os.path.join(comfy_root, rel_dir)
            if os.path.exists(full_dir):
                cls._scan_directory(full_dir, models, rel_dir)
        
        yolo_dir = cls.get_model_dir()
        if os.path.exists(yolo_dir):
            cls._scan_directory(yolo_dir, models, "models/yolo")
        
        if not models:
            models["[无可用模型]"] = ""
        models["[手动输入路径]"] = "MANUAL_INPUT"
        
        cls._cached_models = models
        cls._last_scan_time = current_time
        return models
    
    @classmethod
    def _scan_directory(cls, directory, models_dict, display_prefix):
        for ext in cls.SUPPORTED_EXTENSIONS:
            pattern = os.path.join(directory, f"**/*{ext}")
            for filepath in glob.glob(pattern, recursive=True):
                rel_path = os.path.relpath(filepath, directory)
                display_name = f"{display_prefix}/{rel_path}"
                models_dict[display_name] = filepath
    
    @classmethod
    def get_model_list(cls):
        models = cls.scan_models()
        return list(models.keys())
    
    @classmethod
    def get_model_path(cls, selection):
        if selection == "[手动输入路径]":
            return "MANUAL_INPUT"
        if selection == "[无可用模型]":
            return ""
        models = cls.scan_models()
        return models.get(selection, "")


# ============================================================
# SAM 模型管理器
# ============================================================

class SAMModelManager:
    """
    SAM模型管理器

    支持：
    - SAM v1 (segment-anything): vit_b, vit_l, vit_h
    - SAM 2 / SAM 2.1: sam2_hiera_tiny, sam2_hiera_small, sam2_hiera_base_plus, sam2_hiera_large
    """

    _cached_models = None
    _last_scan_time = 0
    CACHE_DURATION = 30  # 缓存30秒
    SUPPORTED_EXTENSIONS = ['.pt', '.pth']

    @classmethod
    def get_sam_dir(cls):
        """获取SAM模型目录"""
        # 尝试多个可能的目录
        possible_dirs = [
            "models/sam",
            "models/SAM",
            os.path.join(_PLUGIN_ROOT, "models", "sam"),
        ]

        for d in possible_dirs:
            if os.path.exists(d) and os.path.isdir(d):
                return d

        # 如果都不存在，返回默认路径
        return "models/sam"

    @classmethod
    def scan_models(cls):
        """扫描SAM模型目录"""
        import time
        current_time = time.time()

        # 使用缓存
        if cls._cached_models and (current_time - cls._last_scan_time) < cls.CACHE_DURATION:
            return cls._cached_models

        models = {}
        sam_dir = cls.get_sam_dir()

        if os.path.exists(sam_dir):
            for ext in cls.SUPPORTED_EXTENSIONS:
                pattern = os.path.join(sam_dir, f"**/*{ext}")
                for filepath in glob.glob(pattern, recursive=True):
                    filename = os.path.basename(filepath)
                    # 移除扩展名作为显示名称
                    display_name = os.path.splitext(filename)[0]
                    models[display_name] = filepath

        if not models:
            models["[无可用SAM模型]"] = ""

        models["[手动输入路径]"] = "MANUAL_INPUT"

        cls._cached_models = models
        cls._last_scan_time = current_time
        return models

    @classmethod
    def get_model_list(cls):
        """获取模型列表"""
        models = cls.scan_models()
        return list(models.keys())

    @classmethod
    def get_model_path(cls, selection):
        """根据选择获取模型路径"""
        if selection == "[手动输入路径]":
            return "MANUAL_INPUT"
        if selection == "[无可用SAM模型]":
            return ""
        models = cls.scan_models()
        return models.get(selection, "")

    @classmethod
    def detect_sam_version(cls, model_path):
        """
        检测SAM模型版本

        ★ 判定的**顺序**是这段代码的一部分，别按「从新到旧」重排成别的样子：

        SAM 3 的官方权重名形如 `sam3_hiera_l.pt`、`sam3.1_hiera_large.pt` ——
        SAM 3 沿用 Hiera 骨干，名字里**带 `hiera`**。若先跑下面那条 sam2 的
        关键词表，`sam3_hiera_l.pt` 会命中 `hiera` 被判成 **sam2**，
        于是拿 SAM2 的 `build_sam2` + `SAM2ImagePredictor` 去装 SAM 3 的权重 ——
        报错发生在 `build_sam2` 内部的某个无关位置（配置/形状不匹配），
        用户看到的是一条与真正原因毫无关系的报错。所以 sam3 必须**排在最前**。

        Returns:
            "sam1" - SAM v1 (segment-anything)
            "sam2" - SAM 2 / SAM 2.1
            "sam3" - SAM 3
        """
        if not model_path or not os.path.exists(model_path):
            return "sam1"  # 默认

        filename = os.path.basename(model_path).lower()

        # SAM 3特征：包含sam3 —— ★ 必须排在 sam2 的 `hiera` 之前（见 docstring）
        sam3_keywords = ['sam3', 'sam_3', 'sam-3']
        if any(kw in filename for kw in sam3_keywords):
            return "sam3"

        # SAM 2特征：包含sam2、hiera等关键词
        sam2_keywords = ['sam2', 'hiera', 'sam_2', 'sam-2']
        if any(kw in filename for kw in sam2_keywords):
            return "sam2"

        # SAM v1特征：vit_b, vit_l, vit_h
        sam1_keywords = ['vit_b', 'vit_l', 'vit_h', 'sam_vit']
        if any(kw in filename for kw in sam1_keywords):
            return "sam1"

        # 默认识别为SAM v1
        return "sam1"
