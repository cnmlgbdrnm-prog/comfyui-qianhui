"""SAM 精修方法 Mixin：SAM v1 / SAM 2 / SAM 2.1 加载、分割精修与模型卸载。"""
import os

import torch
import comfy.model_management

from ..model_managers import SAMModelManager


class SamMixin:
    def _refine_with_sam(self, image_np, detections, model_path="", conf_threshold=0.7):
        """
        使用 SAM 精细化分割（V6.8 多版本支持）

        将 YOLO 检测框作为 prompt，使用 SAM 生成精确分割 mask
        支持：SAM v1, SAM 2, SAM 2.1（自动检测版本）
        优化：torch.no_grad加速、set_image复用、模型缓存

        Args:
            image_np: 图像 numpy 数组 [H, W, C]
            detections: YOLO 检测结果列表
            model_path: SAM模型路径（自动检测版本）
            conf_threshold: SAM 分割置信度阈值

        Returns:
            精细化后的检测结果
        """
        if not detections:
            return detections

        if not model_path or not os.path.exists(model_path):
            print(f"[SAM] 模型文件不存在: {model_path}")
            return detections

        try:
            import time
            import numpy as np

            # V6.10.4: 在SAM处理开始前检查中断信号
            comfy.model_management.throw_exception_if_processing_interrupted()

            # V6.10.8: 显式设置CUDA设备，防止设备上下文错误导致非法内存访问
            # 参考: https://github.com/facebookresearch/segment-anything/issues/490
            if torch.cuda.is_available():
                torch.cuda.set_device(0)
                torch.cuda.synchronize()

            # V6.10.6: 检查CUDA状态
            if torch.cuda.is_available():
                if not self.yolo_cache.check_cuda_state():
                    print(f"[SAM] CUDA状态异常，跳过SAM精细化")
                    return detections

            # 检测SAM版本
            sam_version = SAMModelManager.detect_sam_version(model_path)
            print(f"[SAM] 检测到模型版本: {sam_version.upper()}")

            # 检查是否需要重新加载模型
            need_reload = (
                not hasattr(self, '_sam_predictor') or
                not hasattr(self, '_sam_model_path') or
                self._sam_model_path != model_path
            )

            if need_reload:
                model_name = os.path.basename(model_path)
                print(f"[SAM] 加载模型: {model_name}")

                if sam_version == "sam2":
                    # SAM 2 / SAM 2.1
                    try:
                        from sam2.build_sam import build_sam2
                        from sam2.sam2_image_predictor import SAM2ImagePredictor
                    except ImportError:
                        print("[SAM] sam2 未安装，请安装: pip install SAM-2")
                        print("[SAM] 或从 https://github.com/facebookresearch/sam2 安装")
                        return detections

                    # SAM2需要配置文件，根据模型名称推断
                    filename = os.path.basename(model_path).lower()
                    if 'large' in filename:
                        cfg_name = "sam2_hiera_l.yaml"
                    elif 'base' in filename or 'plus' in filename:
                        cfg_name = "sam2_hiera_b+.yaml"
                    elif 'small' in filename:
                        cfg_name = "sam2_hiera_s.yaml"
                    elif 'tiny' in filename:
                        cfg_name = "sam2_hiera_t.yaml"
                    else:
                        cfg_name = "sam2_hiera_b+.yaml"  # 默认

                    # 构建完整的配置文件路径
                    try:
                        import sam2
                        sam2_dir = os.path.dirname(sam2.__file__)
                        cfg_path = os.path.join(sam2_dir, "configs", cfg_name)

                        if not os.path.exists(cfg_path):
                            # 尝试替代路径
                            cfg_path = os.path.join(sam2_dir, "sam2_configs", cfg_name)

                        if not os.path.exists(cfg_path):
                            print(f"[SAM2] 配置文件不存在: {cfg_path}")
                            print(f"[SAM2] 尝试使用配置名称: {cfg_name}")
                            cfg_path = cfg_name  # 回退到相对路径
                    except:
                        cfg_path = cfg_name

                    # 构建SAM2模型（使用checkpoint参数）
                    try:
                        print(f"[SAM2] 配置路径: {cfg_path}")
                        print(f"[SAM2] 模型路径: {model_path}")

                        # SAM2的build_sam2签名可能是: build_sam2(config_file, ckpt_path, device, ...)
                        try:
                            sam = build_sam2(cfg_path, model_path, device="cuda" if torch.cuda.is_available() else "cpu")
                        except TypeError:
                            # 旧版本API可能不支持device参数
                            sam = build_sam2(cfg_path, model_path)
                            if torch.cuda.is_available():
                                sam.to(device="cuda")

                        # V6.10.6: 禁用autocast（可能导致CUDA内存错误）
                        # autocast在某些GPU配置下不稳定，特别是低显存环境
                        self._sam_use_fp16 = False
                        if torch.cuda.is_available():
                            # 启用cudnn benchmark加速卷积
                            torch.backends.cudnn.benchmark = True
                            # V6.10.6: 不再默认启用autocast，改为FP32以提高稳定性
                            print(f"[SAM2] 使用FP32精度（稳定模式）")

                        self._sam_predictor = SAM2ImagePredictor(sam)
                        self._sam_version = "sam2"
                        print(f"[SAM2] 使用配置: {cfg_name}")

                        # V6.10.5优化: Windows系统跳过torch.compile（Triton不支持Windows）
                        self._sam_compiled = False
                        self._sam_compile_mode = "none"
                        self._sam_original_encoder = None

                        # 检测操作系统
                        import platform
                        is_windows = platform.system() == 'Windows'

                        # 检查环境变量，允许用户禁用torch.compile
                        disable_compile = os.getenv('DISABLE_SAM_COMPILE', '0') == '1'

                        if is_windows:
                            print(f"[SAM2] Windows系统检测到，跳过torch.compile（Triton不支持Windows）")
                            print(f"[SAM2] 提示：使用FP16精度可以显著加速推理")
                        elif disable_compile:
                            print(f"[SAM2] torch.compile已被环境变量禁用 (DISABLE_SAM_COMPILE=1)")
                        elif torch.cuda.is_available() and hasattr(torch, 'compile'):
                            try:
                                # 禁用CUDA graphs防止卡顿和OOM
                                if hasattr(torch._inductor, 'config') and hasattr(torch._inductor.config, 'triton'):
                                    torch._inductor.config.triton.cudagraphs = False
                                    print(f"[SAM2] 已禁用CUDA graphs防止卡顿")

                                # 保存原始编码器用于回退
                                self._sam_original_encoder = sam.image_encoder

                                # V6.10: 使用更安全的编译配置，避免卡顿
                                # 移除max-autotune（容易卡死），只使用稳定配置
                                compile_configs = [
                                    {"backend": "inductor", "mode": "reduce-overhead", "desc": "Inductor后端（安全且有加速）"},
                                    {"backend": "eager", "desc": "Eager模式（最稳定）"}
                                ]

                                for config in compile_configs:
                                    try:
                                        compile_start = time.time()
                                        print(f"[SAM2] 尝试编译: {config['desc']}")

                                        # 使用更简单的编译参数
                                        if config['backend'] == 'inductor':
                                            sam.image_encoder = torch.compile(
                                                sam.image_encoder,
                                                backend='inductor',
                                                mode=config.get('mode', 'default'),
                                                fullgraph=False,  # 避免fullgraph导致的编译卡顿
                                                dynamic=True      # 支持动态shape，减少重新编译
                                            )
                                        else:
                                            sam.image_encoder = torch.compile(sam.image_encoder, backend=config['backend'])

                                        compile_time = time.time() - compile_start
                                        print(f"[SAM2] 编译图构建成功，耗时 {compile_time:.2f}s")
                                        print(f"[SAM2] 注意：首次推理时会触发实际编译(5-30秒)，请耐心等待")
                                        self._sam_compiled = True
                                        self._sam_compile_mode = config.get('backend', 'unknown')
                                        break  # 编译成功，退出循环

                                    except Exception as e:
                                        error_msg = str(e)
                                        if 'triton' in error_msg.lower() or 'inductor' in error_msg.lower():
                                            print(f"[SAM2] {config['desc']} 编译失败（Triton问题），尝试下一个...")
                                        else:
                                            print(f"[SAM2] 编译失败: {e}，尝试下一个...")
                                        # 恢复原始编码器
                                        sam.image_encoder = self._sam_original_encoder

                                if not self._sam_compiled:
                                    print(f"[SAM2] 所有编译模式失败，使用未编译版本（性能可能较慢但稳定）")
                            except Exception as e:
                                print(f"[SAM2] 编译配置失败: {e}，使用未编译版本")
                                if self._sam_original_encoder is not None:
                                    sam.image_encoder = self._sam_original_encoder
                        else:
                            print(f"[SAM2] CUDA不可用或PyTorch版本过旧，跳过编译")

                        print(f"[SAM2] 模型加载成功")

                    except Exception as e:
                        print(f"[SAM2] 加载失败: {e}")
                        print(f"[SAM2] 请确保:")
                        print(f"  1. SAM2已正确安装: pip install git+https://github.com/facebookresearch/sam2.git")
                        print(f"  2. 模型文件完整且未损坏")
                        print(f"  3. 配置文件存在于sam2包中")
                        return detections

                else:
                    # SAM v1
                    try:
                        from segment_anything import sam_model_registry, SamPredictor
                    except ImportError:
                        print("[SAM] segment-anything 未安装，请安装: pip install segment-anything")
                        return detections

                    # 推断模型类型
                    filename = os.path.basename(model_path).lower()
                    if 'vit_h' in filename or 'huge' in filename:
                        model_type = 'vit_h'
                    elif 'vit_l' in filename or 'large' in filename:
                        model_type = 'vit_l'
                    else:
                        model_type = 'vit_b'  # 默认

                    sam = sam_model_registry[model_type](checkpoint=model_path)

                    # V6.10.6: 禁用autocast（可能导致CUDA内存错误）
                    self._sam_use_fp16 = False
                    precision_str = "FP32"
                    if torch.cuda.is_available():
                        sam.to(device="cuda")
                        # 启用cudnn benchmark加速卷积
                        torch.backends.cudnn.benchmark = True
                        precision_str = "FP32(CUDA)"
                    else:
                        precision_str = "FP32(CPU)"

                    self._sam_predictor = SamPredictor(sam)
                    self._sam_version = "sam1"
                    print(f"[SAM] 使用类型: {model_type}, {precision_str}精度（稳定模式）")

                    # V6.10.5优化: Windows系统跳过torch.compile（Triton不支持Windows）
                    self._sam_compiled = False
                    self._sam_compile_mode = "none"
                    self._sam_original_encoder = None

                    # 检测操作系统
                    import platform
                    is_windows = platform.system() == 'Windows'

                    # 检查环境变量，允许用户禁用torch.compile
                    disable_compile = os.getenv('DISABLE_SAM_COMPILE', '0') == '1'

                    if is_windows:
                        print(f"[SAM] Windows系统检测到，跳过torch.compile（Triton不支持Windows）")
                        print(f"[SAM] 提示：使用FP16精度可以显著加速推理")
                    elif disable_compile:
                        print(f"[SAM] torch.compile已被环境变量禁用 (DISABLE_SAM_COMPILE=1)")
                    elif torch.cuda.is_available() and hasattr(torch, 'compile'):
                        try:
                            # 禁用CUDA graphs防止卡顿和OOM
                            if hasattr(torch._inductor, 'config') and hasattr(torch._inductor.config, 'triton'):
                                torch._inductor.config.triton.cudagraphs = False
                                print(f"[SAM] 已禁用CUDA graphs防止卡顿")

                            # 保存原始编码器用于回退
                            self._sam_original_encoder = sam.image_encoder

                            # V6.10: 使用更安全的编译配置，避免卡顿
                            # 移除max-autotune（容易卡死），只使用稳定配置
                            compile_configs = [
                                {"backend": "inductor", "mode": "reduce-overhead", "desc": "Inductor后端（安全且有加速）"},
                                {"backend": "eager", "desc": "Eager模式（最稳定）"}
                            ]

                            for config in compile_configs:
                                try:
                                    compile_start = time.time()
                                    print(f"[SAM] 尝试编译: {config['desc']}")

                                    # 使用更简单的编译参数
                                    if config['backend'] == 'inductor':
                                        sam.image_encoder = torch.compile(
                                            sam.image_encoder,
                                            backend='inductor',
                                            mode=config.get('mode', 'default'),
                                            fullgraph=False,  # 避免fullgraph导致的编译卡顿
                                            dynamic=True      # 支持动态shape，减少重新编译
                                        )
                                    else:
                                        sam.image_encoder = torch.compile(sam.image_encoder, backend=config['backend'])

                                    compile_time = time.time() - compile_start
                                    print(f"[SAM] 编译图构建成功，耗时 {compile_time:.2f}s")
                                    print(f"[SAM] 注意：首次推理时会触发实际编译(5-30秒)，请耐心等待")
                                    self._sam_compiled = True
                                    self._sam_compile_mode = config.get('backend', 'unknown')
                                    break  # 编译成功，退出循环

                                except Exception as e:
                                    error_msg = str(e)
                                    if 'triton' in error_msg.lower() or 'inductor' in error_msg.lower():
                                        print(f"[SAM] {config['desc']} 编译失败（Triton问题），尝试下一个...")
                                    else:
                                        print(f"[SAM] 编译失败: {e}，尝试下一个...")
                                    # 恢复原始编码器
                                    sam.image_encoder = self._sam_original_encoder

                            if not self._sam_compiled:
                                print(f"[SAM] 所有编译模式失败，使用未编译版本（性能可能较慢但稳定）")
                        except Exception as e:
                            print(f"[SAM] 编译配置失败: {e}，使用未编译版本")
                            if self._sam_original_encoder is not None:
                                sam.image_encoder = self._sam_original_encoder
                    else:
                        print(f"[SAM] CUDA不可用或PyTorch版本过旧，跳过编译")

                self._sam_model_path = model_path

            # 图像预处理
            if image_np.dtype != np.uint8:
                image_np = (np.clip(image_np, 0, 1) * 255).astype(np.uint8)

            # 使用torch.no_grad()加速
            with torch.no_grad():
                start_time = time.time()

                # V6.10.5优化: 使用快速hash判断图像是否改变
                # 采样图像的关键点而不是整个图像，大幅减少hash计算时间
                def fast_image_hash(img):
                    """快速图像hash：使用形状+采样点+统计值"""
                    h, w = img.shape[:2]
                    # 采样4个角和中心的像素值
                    corners = (
                        img[0, 0].sum(),
                        img[0, w-1].sum(),
                        img[h-1, 0].sum(),
                        img[h-1, w-1].sum(),
                        img[h//2, w//2].sum()
                    )
                    # 组合形状和采样值
                    return hash((h, w, img.shape[2] if img.ndim > 2 else 1) + corners + (img.mean(),))

                image_hash = fast_image_hash(image_np)
                need_set_image = (
                    not hasattr(self, '_sam_last_image_hash') or
                    self._sam_last_image_hash != image_hash
                )

                if need_set_image:
                    # V6.10.4: 在图像编码前检查中断信号（set_image可能耗时较长）
                    comfy.model_management.throw_exception_if_processing_interrupted()

                    # set_image只在图像改变时调用
                    set_image_start = time.time()
                    try:
                        # V6.10.5: 使用autocast自动混合精度加速
                        if getattr(self, '_sam_use_fp16', False) and torch.cuda.is_available():
                            with torch.cuda.amp.autocast():
                                self._sam_predictor.set_image(image_np)
                        else:
                            self._sam_predictor.set_image(image_np)
                        set_image_time = time.time() - set_image_start
                        self._sam_last_image_hash = image_hash
                        # V6.10.5: 显示精度和编译模式信息
                        precision_info = " (FP16)" if getattr(self, '_sam_use_fp16', False) else " (FP32)"
                        compile_info = f", 编译:{self._sam_compile_mode}" if hasattr(self, '_sam_compile_mode') and self._sam_compile_mode != "none" else ""
                        print(f"[{self._sam_version.upper()}] 图像编码: {set_image_time:.2f}s{precision_info}{compile_info}")
                    except Exception as e:
                        # 如果编译后的代码执行失败，回退到未编译版本
                        error_msg = str(e)
                        if ('triton' in error_msg.lower() or 'inductor' in error_msg.lower()) and hasattr(self, '_sam_compiled') and self._sam_compiled:
                            print(f"[{self._sam_version.upper()}] 编译版本执行失败，回退到未编译版本: {e}")
                            # 回退到原始编码器
                            if hasattr(self, '_sam_original_encoder') and self._sam_original_encoder is not None:
                                self._sam_predictor.model.image_encoder = self._sam_original_encoder
                                self._sam_compiled = False
                                # 重新尝试
                                self._sam_predictor.set_image(image_np)
                                set_image_time = time.time() - set_image_start
                                self._sam_last_image_hash = image_hash
                                print(f"[{self._sam_version.upper()}] 图像编码（未编译）: {set_image_time:.2f}s")
                            else:
                                raise
                        else:
                            raise
                else:
                    set_image_time = 0.0
                    print(f"[{self._sam_version.upper()}] 使用缓存的图像编码（跳过set_image）")

                # V6.8优化: 批量处理检测框
                refined_dets = []
                num_dets = len(detections)
                predict_time = 0.0

                # 收集所有有效的bbox
                valid_boxes = []
                valid_indices = []
                for idx, det in enumerate(detections):
                    bbox = det.get("bbox", [0, 0, 0, 0])
                    if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                        valid_boxes.append(bbox)
                        valid_indices.append(idx)

                if not valid_boxes:
                    return detections

                # 批量处理（每批16个）
                batch_size = 16
                for batch_start in range(0, len(valid_boxes), batch_size):
                    # V6.10.4: 每个批次开始时检查中断信号
                    comfy.model_management.throw_exception_if_processing_interrupted()

                    batch_end = min(batch_start + batch_size, len(valid_boxes))
                    batch_boxes = valid_boxes[batch_start:batch_end]
                    batch_indices = valid_indices[batch_start:batch_end]

                    for rel_idx, (bbox, orig_idx) in enumerate(zip(batch_boxes, batch_indices)):
                        # V6.10.4: 每个检测框处理前检查中断信号
                        comfy.model_management.throw_exception_if_processing_interrupted()

                        input_box = np.array(bbox, dtype=np.float32)

                        try:
                            # 预测（SAM v1和SAM2 API相同）
                            pred_start = time.time()
                            # V6.10.5: 使用autocast自动混合精度加速
                            if getattr(self, '_sam_use_fp16', False) and torch.cuda.is_available():
                                with torch.cuda.amp.autocast():
                                    masks, scores, _ = self._sam_predictor.predict(
                                        box=input_box,
                                        multimask_output=True
                                    )
                            else:
                                masks, scores, _ = self._sam_predictor.predict(
                                    box=input_box,
                                    multimask_output=True
                                )
                            predict_time += (time.time() - pred_start)

                            det = detections[orig_idx]
                            if len(scores) > 0:
                                best_idx = np.argmax(scores)
                                best_score = scores[best_idx]

                                if best_score >= conf_threshold:
                                    refined_det = det.copy()
                                    refined_det["seg_mask"] = masks[best_idx].astype(np.float32)
                                    refined_det["sam_score"] = float(best_score)
                                    refined_dets.append(refined_det)
                                else:
                                    refined_dets.append(det)
                            else:
                                refined_dets.append(det)

                        except Exception as e:
                            print(f"[SAM] 检测{orig_idx+1}/{num_dets}处理失败: {e}")
                            refined_dets.append(detections[orig_idx])
                            continue

                # 添加跳过的检测框
                for idx, det in enumerate(detections):
                    if idx not in valid_indices:
                        refined_dets.append(det)

                elapsed_time = time.time() - start_time
                refined_count = sum(1 for d in refined_dets if "sam_score" in d)
                avg_predict_time = predict_time / len(valid_boxes) if valid_boxes else 0
                # 性能分析
                set_image_time_display = set_image_time if need_set_image else 0.0
                other_time = elapsed_time - set_image_time_display - predict_time

                print(f"[{self._sam_version.upper()}] 精细化完成: {refined_count}/{len(detections)} 个目标成功")
                print(f"  ├─ 总耗时: {elapsed_time:.2f}s ({len(detections)/max(0.01, elapsed_time):.1f} det/s)")
                if need_set_image:
                    print(f"  ├─ 图像编码: {set_image_time_display:.2f}s ({set_image_time_display/elapsed_time*100:.1f}%)")
                print(f"  ├─ 推理预测: {predict_time:.2f}s ({predict_time/elapsed_time*100:.1f}%)")
                print(f"  │  └─ 平均单个: {avg_predict_time*1000:.1f}ms/det")
                print(f"  └─ 其他开销: {other_time:.2f}s ({other_time/elapsed_time*100:.1f}%)")

                return refined_dets

        except Exception as e:
            print(f"[SAM] 精细化失败: {e}")
            import traceback
            traceback.print_exc()
            return detections

    def _unload_sam_model(self):
        """
        释放SAM模型，清理GPU内存

        V6.10性能优化：SAM使用完后立即释放，避免占用GPU内存影响后续修复
        V6.10.5：低显存模式下更积极清理 + 碎片整理
        V6.10.6：显式清理SAM内部features，防止CUDA非法内存访问
        V6.11.1：强制多轮GC + 彻底清理模型引用
        """
        try:
            # V6.11.3: 移除冗余的前置 synchronize（SAM卸载前不需要等GPU空闲）
            # 真正的 synchronize 保留在显存清理阶段（下方）

            if hasattr(self, '_sam_predictor') and self._sam_predictor is not None:
                # V6.10.6: 显式重置SAM predictor的内部状态
                # 不移动tensor到CPU（可能导致错误），而是设置为None让GC处理
                try:
                    predictor = self._sam_predictor

                    # V6.11.1: 先获取底层model引用，稍后单独清理
                    sam_model = getattr(predictor, 'model', None)

                    # 重置SAM1的内部features
                    if hasattr(predictor, 'features'):
                        predictor.features = None

                    # 重置SAM2的内部features
                    if hasattr(predictor, '_features'):
                        if isinstance(predictor._features, dict):
                            predictor._features.clear()
                        predictor._features = None

                    # 重置底层SAM模型的内部缓存
                    if sam_model is not None:
                        if hasattr(sam_model, 'image_encoder'):
                            encoder = sam_model.image_encoder
                            if hasattr(encoder, '_features'):
                                encoder._features = None
                        # V6.11.1: 清理模型自身的引用
                        if hasattr(predictor, 'model'):
                            predictor.model = None

                    # 重置图像状态标志
                    if hasattr(predictor, 'original_size'):
                        predictor.original_size = None
                    if hasattr(predictor, '_orig_hw'):
                        predictor._orig_hw = None
                    if hasattr(predictor, 'input_size'):
                        predictor.input_size = None
                    if hasattr(predictor, 'is_image_set'):
                        predictor.is_image_set = False
                    if hasattr(predictor, '_is_image_set'):
                        predictor._is_image_set = False

                    # V6.11.1: 显式删除底层模型
                    if sam_model is not None:
                        del sam_model

                except Exception as e:
                    print(f"[SAM] 重置内部状态时出错: {e}")

                # 删除predictor引用
                del self._sam_predictor
                self._sam_predictor = None

            if hasattr(self, '_sam_original_encoder') and self._sam_original_encoder is not None:
                del self._sam_original_encoder
                self._sam_original_encoder = None

            # 清理缓存
            if hasattr(self, '_sam_last_image_hash'):
                del self._sam_last_image_hash

            if hasattr(self, '_sam_model_path'):
                self._sam_model_path = None

            # V6.11.1: 多轮GC确保循环引用被清理
            import gc
            for _ in range(3):
                gc.collect()

            if torch.cuda.is_available():
                try:
                    # V6.11.3: 只保留一次 synchronize + 一次 empty_cache
                    # 原来有 3 次 synchronize + 2 次 empty_cache，每次都阻塞 CPU 等待 GPU
                    torch.cuda.synchronize()
                    gc.collect()
                    torch.cuda.empty_cache()
                except RuntimeError as cuda_err:
                    print(f"[SAM] CUDA同步出错: {cuda_err}")
                    try:
                        gc.collect()
                        torch.cuda.empty_cache()
                    except:
                        pass

            # V6.10.5: 低显存模式下请求ComfyUI清理
            if getattr(self, '_low_vram_mode', False):
                try:
                    comfy.model_management.soft_empty_cache()
                except:
                    pass

            print("[SAM] 模型已释放，GPU内存已清理")

        except Exception as e:
            print(f"[SAM] 释放模型时出错: {e}")
