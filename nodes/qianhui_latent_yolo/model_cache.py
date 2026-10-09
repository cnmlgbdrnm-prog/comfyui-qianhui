"""YOLO 模型缓存管理器（LRU 缓存、显存自动适配与卸载）。"""
import gc
import os

import torch
import comfy.model_management


# ============================================================
# YOLO 模型缓存管理器（支持多模型分批调用）
# ============================================================

class YOLOModelCache:
    """
    YOLO 模型缓存管理器 V1.1

    功能：
    - 支持多个 YOLO 模型的加载和缓存
    - 自动显存管理，避免 OOM
    - LRU 策略，自动卸载不常用模型
    - 分批调用，降低显存压力
    - V6.10.5: 智能低显存模式，自动适配8GB及以下显存GPU
    """

    def __init__(self, max_cached_models=3, vram_threshold_mb=500):
        """
        初始化模型缓存管理器

        Args:
            max_cached_models: 最多同时缓存的模型数量（默认3个）
            vram_threshold_mb: 显存阈值（MB），低于此值时触发模型卸载
        """
        self.models = {}  # {model_path: model_obj}
        self.model_access_order = []  # LRU 列表
        self._printed_classes = {}  # {model_path: bool}

        # V6.10.5: 智能显存检测，自动调整缓存策略
        self.low_vram_mode = False
        total_vram = 0
        if torch.cuda.is_available():
            try:
                total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**2
                gpu_name = torch.cuda.get_device_name(0)

                if total_vram <= 8192:  # 8GB及以下
                    self.low_vram_mode = True
                    self.max_cached = 1  # 只缓存1个模型
                    self.vram_threshold_mb = 1500  # 保留1.5GB空闲
                    print(f"[YOLO缓存] 检测到低显存GPU: {gpu_name} ({total_vram:.0f}MB)")
                    print(f"[YOLO缓存] 已启用低显存模式: 最大缓存1个模型, 保留1.5GB空闲")
                elif total_vram <= 12288:  # 12GB
                    self.max_cached = 2
                    self.vram_threshold_mb = 1000
                    print(f"[YOLO缓存] 检测到中等显存GPU: {gpu_name} ({total_vram:.0f}MB)")
                else:  # >12GB
                    self.max_cached = max_cached_models
                    self.vram_threshold_mb = vram_threshold_mb
                    print(f"[YOLO缓存] 检测到高显存GPU: {gpu_name} ({total_vram:.0f}MB)")
            except Exception as e:
                print(f"[YOLO缓存] 显存检测失败: {e}，使用默认配置")
                self.max_cached = max_cached_models
                self.vram_threshold_mb = vram_threshold_mb
        else:
            self.max_cached = max_cached_models
            self.vram_threshold_mb = vram_threshold_mb

        self.total_vram = total_vram

    def get_vram_info(self):
        """获取显存信息（MB）"""
        if not torch.cuda.is_available():
            return {"total": 0, "used": 0, "free": 0, "fragmentation": 0}

        try:
            total = torch.cuda.get_device_properties(0).total_memory / 1024**2
            reserved = torch.cuda.memory_reserved(0) / 1024**2
            allocated = torch.cuda.memory_allocated(0) / 1024**2
            free = total - reserved

            # V6.10.5: 计算碎片化程度
            # 碎片化 = (reserved - allocated) / reserved * 100
            # 表示已预留但未使用的内存比例
            fragmentation = 0
            if reserved > 0:
                fragmentation = (reserved - allocated) / reserved * 100

            return {
                "total": total,
                "allocated": allocated,
                "reserved": reserved,
                "free": free,
                "fragmentation": fragmentation
            }
        except:
            return {"total": 0, "used": 0, "free": 0, "fragmentation": 0}

    def defragment_vram(self):
        """
        V6.10.5: 显存碎片整理

        通过完全释放CUDA缓存来减少碎片化
        当碎片化程度高时调用此函数
        """
        if not torch.cuda.is_available():
            return

        try:
            import gc

            # 获取整理前状态
            before = self.get_vram_info()

            # 1. 同步所有CUDA操作
            torch.cuda.synchronize()

            # 2. 清理Python垃圾
            gc.collect()

            # 3. 释放所有缓存的显存
            torch.cuda.empty_cache()

            # 4. 如果PyTorch版本支持，重置峰值统计
            if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                torch.cuda.reset_peak_memory_stats()

            # 5. 再次同步并清理
            torch.cuda.synchronize()
            gc.collect()

            # 获取整理后状态
            after = self.get_vram_info()

            # 打印整理结果
            freed = before['reserved'] - after['reserved']
            if freed > 100:  # 只有释放超过100MB才打印
                print(f"[显存整理] 释放碎片: {freed:.0f}MB (碎片化: {before['fragmentation']:.1f}% -> {after['fragmentation']:.1f}%)")

            return freed
        except Exception as e:
            print(f"[显存整理] 失败: {e}")
            return 0

    def check_cuda_state(self):
        """
        V6.10.6: 检查CUDA状态并尝试恢复

        在关键操作前调用，防止CUDA错误累积导致崩溃
        返回: True表示CUDA正常，False表示不可用
        """
        if not torch.cuda.is_available():
            return False

        try:
            # 尝试简单的CUDA操作来检测状态
            test_tensor = torch.zeros(1, device='cuda')
            del test_tensor

            # 同步确保没有pending错误
            torch.cuda.synchronize()
            return True

        except RuntimeError as e:
            error_msg = str(e).lower()
            if 'illegal memory access' in error_msg or 'cuda error' in error_msg:
                print(f"[CUDA] 检测到错误状态，尝试恢复...")
                try:
                    # 尝试恢复CUDA状态
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                    # 再次测试
                    test_tensor = torch.zeros(1, device='cuda')
                    del test_tensor
                    torch.cuda.synchronize()

                    print(f"[CUDA] 状态已恢复")
                    return True
                except Exception as e2:
                    print(f"[CUDA] 无法恢复: {e2}")
                    print(f"[CUDA] 建议重启ComfyUI")
                    return False
            return False
        except Exception as e:
            print(f"[CUDA] 状态检查失败: {e}")
            return False

    def _unload_lru_model(self):
        """卸载最久未使用的模型"""
        if not self.model_access_order:
            return

        # 获取最久未使用的模型路径
        oldest_path = self.model_access_order.pop(0)

        if oldest_path in self.models:
            print(f"[YOLO缓存] 卸载模型: {os.path.basename(oldest_path)}")
            try:
                # V6.10.5: 更彻底的模型卸载
                model = self.models[oldest_path]

                # V6.10.6: 移除危险的model.cpu()调用
                # 在CUDA操作进行中时移动模型到CPU会导致非法内存访问
                # 正确做法：先同步CUDA，再删除引用，最后清理缓存
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                del self.models[oldest_path]
                del model

                if oldest_path in self._printed_classes:
                    del self._printed_classes[oldest_path]

                # V6.10.5: 同步后清理，减少碎片
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"[YOLO缓存] 卸载失败: {e}")

    def _check_and_cleanup(self):
        """检查显存并清理模型"""
        vram_info = self.get_vram_info()

        # V6.10.5: 检测碎片化程度
        if vram_info.get("fragmentation", 0) > 30:  # 碎片化超过30%
            print(f"[YOLO缓存] 检测到显存碎片化({vram_info['fragmentation']:.1f}%)，执行整理...")
            self.defragment_vram()
            vram_info = self.get_vram_info()

        # V6.10.5: 低显存模式下更积极清理
        if self.low_vram_mode:
            # 先执行gc和empty_cache
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            vram_info = self.get_vram_info()

        # 如果显存不足，卸载模型
        while vram_info["free"] < self.vram_threshold_mb and len(self.models) > 0:
            print(f"[YOLO缓存] 显存不足 ({vram_info['free']:.0f}MB < {self.vram_threshold_mb}MB)，触发清理")
            self._unload_lru_model()
            vram_info = self.get_vram_info()

        # 如果缓存数量超过限制，卸载最旧的模型
        while len(self.models) >= self.max_cached:
            self._unload_lru_model()

    def load_model(self, model_path, precision="FP32(默认)"):
        """
        加载 YOLO 模型（带缓存）

        Args:
            model_path: 模型文件路径
            precision: 推理精度 ("FP32(默认)", "FP16(省显存)", "BF16(平衡)")

        Returns:
            YOLO 模型对象
        """
        # 如果已缓存，更新访问顺序并返回
        if model_path in self.models:
            # 更新 LRU
            if model_path in self.model_access_order:
                self.model_access_order.remove(model_path)
            self.model_access_order.append(model_path)

            print(f"[YOLO缓存] 命中缓存: {os.path.basename(model_path)}")
            return self.models[model_path]

        # 检查并清理显存
        self._check_and_cleanup()

        # 加载新模型
        print(f"[YOLO缓存] 加载新模型: {os.path.basename(model_path)}")

        # 临时 patch torch.load
        _original_torch_load = torch.load
        def _patched_load(*args, **kwargs):
            if 'weights_only' not in kwargs:
                kwargs['weights_only'] = False
            return _original_torch_load(*args, **kwargs)

        try:
            torch.load = _patched_load

            # 导入 YOLO
            yolo_class = None
            ultralytics_version = None

            try:
                from ultralytics import YOLO
                yolo_class = YOLO
                try:
                    import ultralytics
                    ultralytics_version = ultralytics.__version__
                except:
                    ultralytics_version = "unknown"
            except ImportError:
                try:
                    from ultralytics.yolo.engine.model import YOLO
                    yolo_class = YOLO
                    ultralytics_version = "legacy"
                except ImportError:
                    try:
                        from ultralytics.models import YOLO
                        yolo_class = YOLO
                        ultralytics_version = "very_old"
                    except ImportError as ie:
                        raise RuntimeError(
                            f"ultralytics 未安装或版本不兼容\n"
                            f"  请运行: pip install ultralytics\n"
                            f"  推荐版本: ultralytics>=8.0.0\n"
                            f"  错误: {ie}"
                        )

            # 检查模型文件
            if not os.path.exists(model_path):
                raise RuntimeError(f"模型文件不存在: {model_path}")

            # 加载模型
            try:
                yolo_model = yolo_class(model_path, verbose=False)
            except TypeError:
                yolo_model = yolo_class(model_path)

            # 设置为评估模式
            if hasattr(yolo_model, 'model'):
                if hasattr(yolo_model.model, 'eval'):
                    yolo_model.model.eval()
                if hasattr(yolo_model.model, 'requires_grad_'):
                    yolo_model.model.requires_grad_(False)

            # 尝试融合层以优化推理
            try:
                if hasattr(yolo_model, 'fuse'):
                    yolo_model.fuse()
            except:
                pass

            # V6.6: 应用推理精度设置
            try:
                if hasattr(yolo_model, 'model') and hasattr(yolo_model.model, 'to'):
                    if precision == "FP16(省显存)":
                        yolo_model.model.half()  # 转换为 FP16
                        print(f"[YOLO缓存] 应用 FP16 精度（节省40%显存）")
                    elif precision == "BF16(平衡)":
                        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                            yolo_model.model.to(dtype=torch.bfloat16)
                            print(f"[YOLO缓存] 应用 BF16 精度（平衡模式）")
                        else:
                            print(f"[YOLO缓存] BF16不支持，使用FP32")
                    # FP32 是默认值，无需转换
            except Exception as e:
                print(f"[YOLO缓存] 精度转换警告: {e}，使用默认精度")

            # 缓存模型
            self.models[model_path] = yolo_model
            self.model_access_order.append(model_path)
            self._printed_classes[model_path] = False

            # 打印信息
            model_name = os.path.basename(model_path)
            print(f"[YOLO缓存] ✓ 加载成功: {model_name} (ultralytics {ultralytics_version})")

            # 检测模型类型
            try:
                if hasattr(yolo_model, 'task'):
                    task = str(yolo_model.task)
                    print(f"[YOLO缓存] 任务类型: {task}")

                # 检测架构版本
                model_type = "未知"
                model_lower = model_name.lower()
                import re

                if 'yolo11' in model_lower or 'yolo-11' in model_lower:
                    model_type = "YOLO11"
                elif 'yolov11' in model_lower:
                    model_type = "YOLOv11"
                elif 'yolov10' in model_lower or 'yolo10' in model_lower:
                    model_type = "YOLOv10"
                elif 'yolov9' in model_lower or 'yolo9' in model_lower:
                    model_type = "YOLOv9"
                elif 'yolov8' in model_lower or 'yolo8' in model_lower:
                    model_type = "YOLOv8"
                elif 'yolov7' in model_lower or 'yolo7' in model_lower:
                    model_type = "YOLOv7"
                elif 'yolov6' in model_lower or 'yolo6' in model_lower:
                    model_type = "YOLOv6"
                elif 'yolov5' in model_lower or 'yolo5' in model_lower:
                    model_type = "YOLOv5"
                else:
                    match = re.search(r'yolo[_-]?v?(\d+)', model_lower)
                    if match:
                        version = match.group(1)
                        model_type = f"YOLOv{version}"

                if model_type != "未知":
                    print(f"[YOLO缓存] 架构版本: {model_type}")
            except:
                pass

            # 打印显存信息
            vram_info = self.get_vram_info()
            if vram_info["total"] > 0:
                print(f"[YOLO缓存] 显存: {vram_info['allocated']:.0f}MB / {vram_info['total']:.0f}MB "
                      f"(已缓存 {len(self.models)} 个模型)")

            return yolo_model

        finally:
            torch.load = _original_torch_load

    def unload_all(self):
        """卸载所有模型"""
        print(f"[YOLO缓存] 卸载所有模型 ({len(self.models)} 个)")
        self.models.clear()
        self.model_access_order.clear()
        self._printed_classes.clear()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc
            gc.collect()

    def get_cached_count(self):
        """获取已缓存的模型数量"""
        return len(self.models)

    def is_model_printed(self, model_path):
        """检查模型是否已打印过类别信息"""
        return self._printed_classes.get(model_path, False)

    def set_model_printed(self, model_path, printed=True):
        """设置模型已打印类别信息"""
        self._printed_classes[model_path] = printed


#: 4 个拆分节点的**共享**模型缓存。
#:
#: 为什么必须是模块级单例而不是每个节点各一个：pipeline 里的代码会伸手进
#: `node.yolo_cache` 去卸模型 / 整理显存（repair.py、sam_refine.py），
#: 而 load_model 发生在 detect.py。各持一份的话，修复节点卸的是自己那个**空**缓存，
#: 检测节点加载的模型再没人卸 —— **不报错，只是显存一直涨**。
_SHARED_CACHE = None


def get_shared_cache():
    """取 4 个节点共用的 YOLOModelCache（首次调用时创建，之后恒返回同一个对象）。"""
    global _SHARED_CACHE
    if _SHARED_CACHE is None:
        _SHARED_CACHE = YOLOModelCache(max_cached_models=3, vram_threshold_mb=500)
    return _SHARED_CACHE
