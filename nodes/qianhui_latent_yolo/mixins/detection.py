"""YOLO 检测方法 Mixin：模型版本探测与加载、多模型批量检测、TTA 与 Tile 分块检测。"""
import os

import torch
import comfy.model_management

from ..model_cache import YOLOModelCache


class DetectionMixin:
    # -------------------- YOLO工具函数 --------------------
    # 支持的 YOLO 版本说明：
    # ✅ YOLOv5 (2020) - ultralytics-yolov5
    # ✅ YOLOv6 (2022) - 第三方实现
    # ✅ YOLOv7 (2022) - 第三方实现
    # ✅ YOLOv8 (2023) - ultralytics >= 8.0
    # ✅ YOLOv9 (2024) - ultralytics >= 8.1
    # ✅ YOLOv10 (2024) - ultralytics >= 8.2
    # ✅ YOLOv11/YOLO11 (2024) - ultralytics >= 8.3
    # ✅ YOLO-World (2024) - 开放词汇检测
    # ✅ 所有基于 ultralytics 的自定义训练模型

    def _get_ultralytics_version(self):
        """获取 ultralytics 版本"""
        try:
            import ultralytics
            version_str = ultralytics.__version__
            # 解析版本号 (例如: "8.3.0" -> (8, 3))
            parts = version_str.split('.')
            return tuple(map(int, parts[:2]))
        except:
            return (8, 3)  # 默认假设是最新版本

    def _check_yolo_capabilities(self, model):
        """
        检查 YOLO 模型的功能特性
        支持 YOLOv5-v11 及所有 ultralytics 模型
        """
        capabilities = {
            'has_seg': False,
            'has_pose': False,
            'has_obb': False,
            'has_track': False,
            'version': 'unknown'
        }

        try:
            # 检测任务类型
            task = None
            if hasattr(model, 'task'):
                task = str(model.task).lower()
            elif hasattr(model, 'model') and hasattr(model.model, 'task'):
                task = str(model.model.task).lower()

            if task:
                capabilities['has_seg'] = 'seg' in task or 'segment' in task
                capabilities['has_pose'] = 'pose' in task or 'keypoint' in task
                capabilities['has_obb'] = 'obb' in task or 'oriented' in task

            # 检测追踪能力 (YOLOv8+)
            if hasattr(model, 'track'):
                capabilities['has_track'] = True

            # 检测版本
            version = self._get_ultralytics_version()
            capabilities['version'] = f"{version[0]}.{version[1]}"

        except Exception:
            pass

        return capabilities

    def _load_yolo_model(self, model_path):
        # 确保属性存在（防止 __init__ 未被调用的情况）
        if not hasattr(self, 'yolo_model'):
            self.yolo_model = None
        if not hasattr(self, 'yolo_model_path'):
            self.yolo_model_path = None
        if not hasattr(self, '_printed_classes'):
            self._printed_classes = False

        if self.yolo_model is not None and self.yolo_model_path == model_path:
            return self.yolo_model

        # 清理旧模型释放显存
        if self.yolo_model is not None:
            try:
                del self.yolo_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass

        # 临时patch torch.load（仅在加载YOLO时生效）
        _original_torch_load = torch.load
        def _patched_load(*args, **kwargs):
            if 'weights_only' not in kwargs:
                kwargs['weights_only'] = False
            return _original_torch_load(*args, **kwargs)

        try:
            torch.load = _patched_load  # 临时替换

            # 1. 导入ultralytics（兼容多种导入方式）
            yolo_class = None
            ultralytics_version = None

            # 尝试多种导入方式
            try:
                from ultralytics import YOLO
                yolo_class = YOLO
                try:
                    import ultralytics
                    ultralytics_version = ultralytics.__version__
                except:
                    ultralytics_version = "unknown"
            except ImportError:
                # 尝试旧版本导入
                try:
                    from ultralytics.yolo.engine.model import YOLO
                    yolo_class = YOLO
                    ultralytics_version = "legacy"
                except ImportError:
                    # 再尝试更旧的版本
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

            # 2. 检查模型文件
            if not os.path.exists(model_path):
                raise RuntimeError(f"模型文件不存在: {model_path}")

            # 3. 加载模型（兼容不同参数）
            try:
                # 尝试带 verbose 参数加载（新版本）
                try:
                    self.yolo_model = yolo_class(model_path, verbose=False)
                except TypeError:
                    # 旧版本不支持 verbose 参数
                    self.yolo_model = yolo_class(model_path)

                # 设置为评估模式，减少显存占用
                if hasattr(self.yolo_model, 'model'):
                    if hasattr(self.yolo_model.model, 'eval'):
                        self.yolo_model.model.eval()
                    # 禁用梯度计算
                    if hasattr(self.yolo_model.model, 'requires_grad_'):
                        self.yolo_model.model.requires_grad_(False)

                # 尝试设置为推理模式
                try:
                    if hasattr(self.yolo_model, 'fuse'):
                        self.yolo_model.fuse()
                except:
                    pass

            except Exception as e:
                raise RuntimeError(
                    f"YOLO模型加载失败\n"
                    f"  路径: {model_path}\n"
                    f"  ultralytics版本: {ultralytics_version}\n"
                    f"  错误: {type(e).__name__}: {e}"
                )

            self.yolo_model_path = model_path
            self._printed_classes = False

            # 打印版本信息
            model_name = os.path.basename(model_path)
            print(f"[YOLO] ✓ 模型加载成功: {model_name}")
            print(f"[YOLO] ultralytics 版本: {ultralytics_version}")

            # 检测模型类型和 YOLO 架构版本
            try:
                if hasattr(self.yolo_model, 'task'):
                    task = str(self.yolo_model.task)
                    print(f"[YOLO] 模型任务: {task}")

                # 检测 YOLO 架构版本 (YOLOv5/v8/v9/v10/v11/YOLO11等)
                model_type = "未知"
                model_lower = model_name.lower()

                # 匹配各种 YOLO 版本命名
                import re
                if 'yolo11' in model_lower or 'yolo-11' in model_lower:
                    model_type = "YOLO11"
                elif 'yolov11' in model_lower or 'yolo-v11' in model_lower:
                    model_type = "YOLOv11"
                elif 'yolov10' in model_lower or 'yolo10' in model_lower or 'yolo-v10' in model_lower:
                    model_type = "YOLOv10"
                elif 'yolov9' in model_lower or 'yolo9' in model_lower or 'yolo-v9' in model_lower:
                    model_type = "YOLOv9"
                elif 'yolov8' in model_lower or 'yolo8' in model_lower or 'yolo-v8' in model_lower:
                    model_type = "YOLOv8"
                elif 'yolov7' in model_lower or 'yolo7' in model_lower:
                    model_type = "YOLOv7"
                elif 'yolov6' in model_lower or 'yolo6' in model_lower:
                    model_type = "YOLOv6"
                elif 'yolov5' in model_lower or 'yolo5' in model_lower:
                    model_type = "YOLOv5"
                else:
                    # 尝试用正则提取版本号
                    match = re.search(r'yolo[_-]?v?(\d+)', model_lower)
                    if match:
                        version = match.group(1)
                        model_type = f"YOLOv{version}"

                if model_type != "未知":
                    print(f"[YOLO] 模型架构: {model_type}")

            except:
                pass

            return self.yolo_model

        finally:
            torch.load = _original_torch_load  # 恢复原始函数

    def _decode_latent_to_image(self, vae, latent_samples):
        decoded = vae.decode(latent_samples)
        decoded = torch.clamp(decoded, 0.0, 1.0)
        return decoded

    def _upscale_image_for_yolo(self, image, min_size=1024):
        """
        上采样图片以提高 YOLO 检测质量

        参数:
            image: 图片张量 [H, W, C]，可在CPU或GPU上
            min_size: 最小边长，默认 1024
        """
        import numpy as np
        import cv2

        h, w = image.shape[0], image.shape[1]

        # 如果图片已经足够大，直接返回
        if min(h, w) >= min_size:
            return image

        # 计算放大倍数
        scale = min_size / min(h, w)
        new_h = int(h * scale)
        new_w = int(w * scale)

        print(f"    [YOLO预处理] 上采样图片: {h}x{w} -> {new_h}x{new_w} (提高检测质量)")

        # V6.11.3: 改用纯CPU的cv2.resize，不占VRAM，避免GPU上下文切换
        # image可能是GPU或CPU tensor，统一转为CPU numpy处理
        if isinstance(image, torch.Tensor):
            img_np = image.detach().cpu().numpy()
        else:
            img_np = np.array(image)

        # cv2.resize期望float32，(H, W, C)格式
        img_np = img_np.astype(np.float32)
        upscaled_np = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        upscaled_np = np.clip(upscaled_np, 0.0, 1.0)

        # 返回CPU tensor，与后续numpy转换兼容
        return torch.from_numpy(upscaled_np)

    def _detect_with_multi_models(self, model_paths, image, confidences, imgsz=640,
                                   tile_enabled=True, tile_size=640, tile_overlap=0.25,
                                   use_seg=True, target_classes=None, multi_scale=True,
                                   precision="FP32(默认)"):
        """
        多模型分批检测（V6.9 优化：支持每个模型独立置信度）

        逐个加载模型进行检测，自动管理显存，最后合并结果

        Args:
            model_paths: YOLO 模型路径列表
            confidences: 每个模型对应的置信度列表（与model_paths长度相同）
            其他参数同 _detect_with_yolo

        Returns:
            合并后的检测结果列表
        """
        if not model_paths:
            return []

        all_detections = []
        total_models = len(model_paths)

        # 确保confidences列表长度与model_paths一致
        if not isinstance(confidences, list):
            confidences = [confidences] * total_models
        elif len(confidences) < total_models:
            # 如果confidences不足，用最后一个值填充
            confidences = list(confidences) + [confidences[-1]] * (total_models - len(confidences))

        print(f"    [多模型检测] 开始分批检测，共 {total_models} 个模型")

        for idx, (model_path, conf) in enumerate(zip(model_paths, confidences), 1):
            # V6.10: 检查中断信号（多模型检测可能耗时较长）
            comfy.model_management.throw_exception_if_processing_interrupted()

            try:
                # 使用缓存管理器加载模型（自动管理显存）
                if not hasattr(self, 'yolo_cache'):
                    self.yolo_cache = YOLOModelCache(max_cached_models=3, vram_threshold_mb=500)

                model = self.yolo_cache.load_model(model_path, precision=precision)
                model_name = os.path.basename(model_path)

                print(f"    [{idx}/{total_models}] 使用模型: {model_name} | 置信度: {conf:.2f}")

                # 执行检测
                detections = self._detect_with_yolo(
                    model, image, conf,  # 使用独立置信度
                    imgsz=imgsz,
                    tile_enabled=tile_enabled,
                    tile_size=tile_size,
                    tile_overlap=tile_overlap,
                    use_seg=use_seg,
                    target_classes=target_classes,
                    multi_scale=multi_scale
                )

                if detections:
                    # 为每个检测结果添加模型来源信息
                    for det in detections:
                        det['source_model'] = model_name
                    all_detections.extend(detections)
                    print(f"    [{idx}/{total_models}] 检测到 {len(detections)} 个目标")
                else:
                    print(f"    [{idx}/{total_models}] 未检测到目标")

                # 清理临时变量
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"    [{idx}/{total_models}] 模型检测失败: {e}")
                continue

        print(f"    [多模型检测] 完成，共检测到 {len(all_detections)} 个目标")

        # 返回合并后的结果（后续会进行去重和过滤）
        return all_detections

    def _detect_with_yolo(self, model, image, confidence, imgsz=640, 
                          tile_enabled=True, tile_size=640, tile_overlap=0.25,
                          use_seg=True, target_classes=None, multi_scale=True):
        """
        V6.4增强版YOLO检测
        - 多尺度检测(TTA) - 提高准确度
        - Tile分块检测
        - Seg mask支持
        - 性能优化
        """
        import numpy as np

        # V6.10.4修复：确保正确转换tensor为numpy数组
        if isinstance(image, torch.Tensor):
            # 使用detach()确保tensor可以转换，即使requires_grad=True
            image_np = image.detach().cpu().numpy()
        else:
            image_np = image

        # 确保是numpy数组
        if not isinstance(image_np, np.ndarray):
            image_np = np.array(image_np)

        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)

        if not getattr(self, "_printed_classes", False):
            try:
                # 获取类别名称（兼容多种方式）
                names = None
                if hasattr(model, 'names'):
                    names = model.names
                elif hasattr(model, 'model') and hasattr(model.model, 'names'):
                    names = model.model.names

                if names:
                    # 兼容字典和列表格式
                    if isinstance(names, dict):
                        class_list = list(names.values())
                    else:
                        class_list = list(names)
                    print(f"    [YOLO] 模型支持的类别: {class_list[:20]}")  # 最多显示20个
                    if len(class_list) > 20:
                        print(f"    [YOLO] ... 共 {len(class_list)} 个类别")

                # 检测是否为分割模型（兼容多种方式）
                is_seg = False
                if hasattr(model, 'task'):
                    is_seg = 'seg' in str(model.task).lower()
                elif hasattr(model, 'model') and hasattr(model.model, 'task'):
                    is_seg = 'seg' in str(model.model.task).lower()
                # 也可以通过检查是否有 segment 相关属性
                elif hasattr(model, 'predictor'):
                    is_seg = 'segment' in str(type(model.predictor)).lower()

                print(f"    [YOLO] 分割模型: {'是' if is_seg else '否'}")
            except Exception as e:
                print(f"    [YOLO] 无法获取模型信息: {e}")
            self._printed_classes = True

        h, w = image_np.shape[:2]
        all_detections = []
        
        # ★ 多尺度检测(TTA) - 用不同imgsz检测同一图像
        if multi_scale:
            scales = [imgsz]
            # 添加额外尺度
            if imgsz >= 640:
                scales.append(int(imgsz * 0.75))  # 小尺度
            if imgsz <= 1024:
                scales.append(int(imgsz * 1.25))  # 大尺度
            scales = list(set(scales))  # 去重
        else:
            scales = [imgsz]
        
        # 判断是否需要tile检测
        need_tile = tile_enabled and max(h, w) > tile_size * 1.5
        
        if not need_tile:
            # 单图多尺度检测
            for scale in scales:
                dets = self._detect_single_enhanced(model, image_np, confidence, scale, use_seg, target_classes)
                all_detections.extend(dets)
        else:
            # Tile + 多尺度检测
            print(f"    [YOLO] Tile检测: {w}x{h}, tile={tile_size}, overlap={tile_overlap}, scales={scales}")
            stride = int(tile_size * (1.0 - tile_overlap))
            
            tiles = []
            for y in range(0, h, stride):
                for x in range(0, w, stride):
                    x2 = min(x + tile_size, w)
                    y2 = min(y + tile_size, h)
                    if x2 - x < tile_size // 3 or y2 - y < tile_size // 3:
                        continue
                    tiles.append((x, y, x2, y2))
            
            print(f"    [YOLO] 分块数: {len(tiles)}")
            
            for tx1, ty1, tx2, ty2 in tiles:
                # V6.10: 检查中断信号（tile检测可能有很多块）
                comfy.model_management.throw_exception_if_processing_interrupted()

                tile_img = image_np[ty1:ty2, tx1:tx2]

                # 每个tile用主尺度检测（性能优化：tile不用多尺度）
                tile_dets = self._detect_single_enhanced(model, tile_img, confidence, imgsz, use_seg, target_classes)

                for det in tile_dets:
                    bx1, by1, bx2, by2 = det["bbox"]
                    det["bbox"] = [bx1 + tx1, by1 + ty1, bx2 + tx1, by2 + ty1]

                    if det.get("seg_mask") is not None:
                        try:
                            full_mask = np.zeros((h, w), dtype=np.float32)
                            seg = det["seg_mask"]

                            # 确保seg是2D数组
                            if seg.ndim > 2:
                                seg = seg.squeeze()

                            # V6.10.3修复：增强seg_mask形状验证
                            if seg.ndim < 2:
                                raise ValueError(f"seg_mask维度不足: {seg.ndim}D，期望2D")

                            seg_h, seg_w = seg.shape[:2]
                            tile_h, tile_w = ty2 - ty1, tx2 - tx1

                            # V6.11.3性能优化：用cv2.resize替换F.interpolate
                            # cv2.resize是纯CPU操作，无GPU上下文切换开销，且速度更快
                            # 注意：cv2.resize参数是(width, height)，与numpy的(H,W)相反
                            if seg_h != tile_h or seg_w != tile_w:
                                import cv2
                                seg = cv2.resize(seg.astype(np.float32), (tile_w, tile_h),
                                                 interpolation=cv2.INTER_LINEAR)

                            # V6.10.3修复：再次验证resize后的尺寸
                            if seg.shape != (tile_h, tile_w):
                                raise ValueError(f"Resize后尺寸仍不匹配: seg={seg.shape}, tile=({tile_h}, {tile_w})")

                            full_mask[ty1:ty2, tx1:tx2] = seg
                            det["seg_mask"] = full_mask
                        except Exception as e:
                            # seg mask处理失败，移除mask但保留检测框
                            det["seg_mask"] = None
                            if tx1 == 0 and ty1 == 0:  # 只在第一个tile打印一次
                                print(f"    [YOLO] Tile seg mask处理失败 (tile={tx1},{ty1},{tx2},{ty2}, 全图={h}x{w}): {e}")

                    all_detections.append(det)
            
            # 全图也检测一次（捕获跨tile的大目标）- 用所有尺度，提高大目标召回率
            for scale in scales:
                full_dets = self._detect_single_enhanced(model, image_np, confidence, scale, use_seg, target_classes)

                # V6.10.3修复：规范化全图检测的seg_mask尺寸，避免与tile检测的mask尺寸不匹配
                for det in full_dets:
                    if det.get("seg_mask") is not None:
                        try:
                            seg = det["seg_mask"]

                            # 确保seg是2D数组
                            if seg.ndim > 2:
                                seg = seg.squeeze()

                            seg_h, seg_w = seg.shape[:2] if seg.ndim >= 2 else (0, 0)

                            # 如果seg尺寸与原图不匹配，resize到原图尺寸
                            # V6.11.3性能优化：用cv2.resize替换F.interpolate，避免GPU上下文切换
                            if seg_h != h or seg_w != w:
                                import cv2
                                seg = cv2.resize(seg.astype(np.float32), (w, h),
                                                 interpolation=cv2.INTER_LINEAR)
                                det["seg_mask"] = seg
                                print(f"    [YOLO] 全图检测seg_mask已规范化: {seg_h}x{seg_w} -> {h}x{w}")
                        except Exception as e:
                            # seg mask处理失败，移除mask但保留检测框
                            det["seg_mask"] = None
                            print(f"    [YOLO] 全图检测seg_mask规范化失败: {e}")

                all_detections.extend(full_dets)

        # 水平翻转TTA（multi_scale模式下额外增强，提高检测率）
        if multi_scale and len(all_detections) > 0:
            try:
                image_flipped = image_np[:, ::-1, :].copy()
                flip_dets = self._detect_single_enhanced(model, image_flipped, confidence, imgsz, use_seg, target_classes)
                for det in flip_dets:
                    bx1, by1, bx2, by2 = det["bbox"]
                    # 翻转坐标映射回原图
                    det["bbox"] = [w - bx2, by1, w - bx1, by2]
                    if det.get("seg_mask") is not None:
                        try:
                            det["seg_mask"] = det["seg_mask"][:, ::-1].copy()
                        except Exception:
                            det["seg_mask"] = None
                all_detections.extend(flip_dets)
            except Exception as flip_err:
                print(f"    [YOLO] 翻转TTA失败（跳过）: {flip_err}")

        # 增强版NMS去重（降低阈值到0.35，减少重叠框，同类间更严格）
        if len(all_detections) > 1:
            all_detections = self._nms_detections_enhanced(all_detections, iou_thresh=0.35)

        # 清理 CUDA 缓存，释放显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"    [YOLO] 检测完成，共 {len(all_detections)} 个目标")
        return all_detections

    def _detect_single_enhanced(self, model, image_np, confidence, imgsz, use_seg, target_classes):
        """单张图增强检测（兼容多版本ultralytics）"""
        import numpy as np
        from PIL import Image as PILImage
        detections = []

        # V6.11.3修复：统一转换为PIL Image再传给YOLO（与ComfyUI-Impact-Pack/zml插件保持一致）
        # 原因：ultralytics内部对PIL和numpy处理路径不同：
        #   - PIL输入：_single_check做RGB→BGR, preprocess再做BGR→RGB，模型收到RGB ✓
        #   - numpy输入：_single_check直接as-is，preprocess做BGR→RGB（期望输入是BGR）
        #   - 若传RGB numpy（原bug）：_single_check as-is，preprocess"BGR→RGB"实为BGR，模型收到BGR ✗
        # 直接传PIL是ultralytics官方推荐且经过验证的方式，避免通道顺序歧义
        if isinstance(image_np, torch.Tensor):
            image_np = image_np.detach().cpu().numpy()
        if not isinstance(image_np, np.ndarray):
            image_np = np.array(image_np)

        # 转为uint8 RGB numpy
        if image_np.max() <= 1.0:
            image_uint8 = (image_np * 255).astype(np.uint8)
        else:
            image_uint8 = image_np.astype(np.uint8)

        # 转为PIL Image（RGB格式），ultralytics内部会正确处理颜色空间
        pil_image = PILImage.fromarray(image_uint8, mode='RGB')

        # V6.11.3性能优化：缓存verbose支持状态，避免每次调用都try/except触发推理
        # 使用model对象本身作为缓存key（不同model实例独立缓存）
        _verbose_key = '_yolo_verbose_ok'
        if not hasattr(model, _verbose_key):
            setattr(model, _verbose_key, None)  # None=未知，True=支持，False=不支持

        # 使用 no_grad 避免梯度计算，减少显存占用
        with torch.no_grad():
            # 构建推理参数（兼容不同版本）
            predict_kwargs = {
                'conf': float(confidence),
                'imgsz': int(imgsz),
            }

            # V6.11.3性能优化：缓存verbose支持状态，避免每次调用重复try/except
            _verbose_ok = getattr(model, _verbose_key, None)
            if _verbose_ok is True:
                predict_kwargs['verbose'] = False
                results = model(pil_image, **predict_kwargs)
            elif _verbose_ok is False:
                results = model(pil_image, **predict_kwargs)
            else:
                # 首次调用：探测verbose是否支持（TypeError发生在参数校验阶段，不会重复推理）
                try:
                    predict_kwargs['verbose'] = False
                    results = model(pil_image, **predict_kwargs)
                    setattr(model, _verbose_key, True)
                except TypeError:
                    predict_kwargs.pop('verbose', None)
                    results = model(pil_image, **predict_kwargs)
                    setattr(model, _verbose_key, False)

            for result in results:
                # 兼容不同版本的 boxes 访问
                boxes = None
                if hasattr(result, 'boxes'):
                    boxes = result.boxes
                elif hasattr(result, 'pred') and len(result.pred) > 0:
                    # 旧版本可能使用 pred
                    boxes = result.pred[0] if isinstance(result.pred, list) else result.pred

                if boxes is None or len(boxes) == 0:
                    continue

                # V6.11.3性能优化：seg mask 批量一次性从GPU传到CPU
                # 原来在每个box循环内调用 masks.data[idx].cpu().numpy()，导致N次GPU同步
                # 改为在循环外统一传输，只需1次GPU→CPU同步
                masks = getattr(result, 'masks', None)
                all_masks_np = None
                if use_seg and masks is not None:
                    try:
                        if hasattr(masks, 'data') and len(masks.data) > 0:
                            all_masks_np = masks.data.cpu().numpy()  # (N, H, W) 一次传输所有mask
                        elif hasattr(masks, 'masks') and len(masks.masks) > 0:
                            all_masks_np = masks.masks.cpu().numpy()
                    except Exception as e:
                        print(f"    [YOLO] Seg mask批量提取失败，降级到逐个提取: {e}")
                        all_masks_np = None

                # 批量提取bbox和conf（一次GPU→CPU，避免per-box .cpu()调用）
                try:
                    all_xyxy = boxes.xyxy.cpu().numpy()   # (N, 4)
                    all_conf = boxes.conf.cpu().numpy()    # (N,)
                    all_cls  = boxes.cls.cpu().numpy().astype(int)  # (N,)
                    use_batch_boxes = True
                except Exception:
                    use_batch_boxes = False

                # 获取类别名称字典（在循环外一次性获取）
                names_dict = None
                if hasattr(model, 'names'):
                    names_dict = model.names
                elif hasattr(model, 'model') and hasattr(model.model, 'names'):
                    names_dict = model.model.names

                for idx, box in enumerate(boxes):
                    # 兼容不同的 box 数据结构
                    try:
                        if use_batch_boxes:
                            cls_id = int(all_cls[idx])
                            bbox   = all_xyxy[idx].tolist()
                            conf   = float(all_conf[idx])
                        else:
                            # 旧版本降级路径
                            if hasattr(box, 'cls'):
                                cls_id = int(box.cls[0]) if torch.is_tensor(box.cls) else int(box.cls)
                            else:
                                cls_id = int(box[5]) if len(box) > 5 else 0
                            if hasattr(box, 'xyxy'):
                                bbox = box.xyxy[0].cpu().numpy().tolist()
                            else:
                                bbox = box[:4].cpu().numpy().tolist() if torch.is_tensor(box) else list(box[:4])
                            if hasattr(box, 'conf'):
                                conf = float(box.conf[0]) if torch.is_tensor(box.conf) else float(box.conf)
                            else:
                                conf = float(box[4]) if len(box) > 4 else 0.0

                        cls_name = names_dict.get(cls_id, str(cls_id)) if names_dict else str(cls_id)

                        if target_classes and not self._match_class_name(cls_name, target_classes):
                            continue

                    except Exception as e:
                        if idx == 0:
                            print(f"    [YOLO] 警告: 检测框解析失败 - {e}")
                        continue

                    det = {
                        "class_name": cls_name,
                        "class_id": cls_id,
                        "confidence": conf,
                        "bbox": bbox,
                        "seg_mask": None
                    }

                    # V6.11.3性能优化：seg mask 从已批量传输的 all_masks_np 中取（无GPU同步开销）
                    if use_seg and all_masks_np is not None and idx < len(all_masks_np):
                        try:
                            seg_mask = all_masks_np[idx]  # 已经是CPU numpy，直接使用

                            # 确保seg_mask是2D数组 (H, W)
                            while seg_mask.ndim > 2:
                                seg_mask = seg_mask.squeeze() if seg_mask.shape[0] == 1 else seg_mask[0]

                            # 转换为float32并归一化到[0, 1]
                            seg_mask = seg_mask.astype(np.float32)
                            if seg_mask.max() > 1.0:
                                seg_mask = seg_mask / 255.0

                            det["seg_mask"] = seg_mask
                            det["seg_mask_shape"] = image_np.shape[:2]  # 原始图像尺寸 (H, W)

                        except Exception as e:
                            if idx == 0:
                                print(f"    [YOLO] Seg mask 提取失败: {e}")

                    detections.append(det)

        return detections
