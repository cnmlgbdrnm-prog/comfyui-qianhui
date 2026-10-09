"""从 run_detection 搬出来的 SAM 精修块（源 detect.py:366-469）。

拆分后 SAM 精修归「节点 B（SAM 精修）」，故先把这段从 `run_detection` 里
整段搬出来、单独取用。本次是**纯搬家**：语句逐字保留原样 —— 注释、空格、
运算符、常量、`if` 条件一律照抄，一行逻辑未改，故行为不变。
"""
import comfy.model_management
import numpy as np
import torch


def refine_with_sam(node, ctx, dets, decoded_det) -> list:
    """V6.8: SAM 精细化（可选，支持SAM/SAM2/SAM2.1）（源 detect.py:366-469）。"""
    # V6.8: SAM 精细化（可选，支持SAM/SAM2/SAM2.1）
    if ctx.cfg.enable_sam and dets and ctx.sam_model_path:
        # V6.10: 检查中断信号（SAM处理可能耗时较长）
        comfy.model_management.throw_exception_if_processing_interrupted()

        # V6.11.1: SAM加载前清理显存（保守方式，不完全卸载主模型）
        # 使用soft_empty_cache而非unload_all_models，避免主模型无法自动重载
        print(f"    [SAM] 清理显存为SAM腾出空间...")
        try:
            # 请求ComfyUI清理缓存（不会强制卸载模型）
            comfy.model_management.soft_empty_cache()
            # 清理YOLO缓存
            if hasattr(node, 'yolo_cache'):
                while len(node.yolo_cache.models) > 0:
                    node.yolo_cache._unload_lru_model()
            # 多轮GC确保清理彻底
            import gc
            for _ in range(2):
                gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                # 打印显存状态
                free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                print(f"    [SAM] 显存已清理，可用: {free_mem/1024**2:.0f}MB")
        except Exception as cleanup_err:
            print(f"    [SAM] 清理显存时出错: {cleanup_err}")

        # V6.10.8: 完整的SAM错误保护，防止CUDA错误传播到inpaint阶段
        sam_success = False
        try:
            # 转换图像格式供 SAM 使用
            decoded_for_sam = decoded_det.cpu().numpy()
            if decoded_for_sam.max() <= 1.0:
                decoded_for_sam = (decoded_for_sam * 255).astype(np.uint8)
            else:
                decoded_for_sam = decoded_for_sam.astype(np.uint8)

            # 显式同步CUDA确保之前操作完成
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            dets = node._refine_with_sam(decoded_for_sam, dets, model_path=ctx.sam_model_path, conf_threshold=ctx.cfg.sam_conf_threshold)
            sam_success = True

        except RuntimeError as sam_err:
            error_msg = str(sam_err).lower()
            if 'cuda' in error_msg or 'illegal memory' in error_msg:
                print(f"    [SAM错误] CUDA错误，跳过SAM精细化: {sam_err}")
                print(f"    [SAM错误] 将使用YOLO检测结果（无精细分割mask）")
                # 尝试恢复CUDA状态
                try:
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        # 重置CUDA设备上下文
                        torch.cuda.synchronize()
                except:
                    pass
            else:
                print(f"    [SAM错误] 运行时错误: {sam_err}")

        except Exception as sam_err:
            print(f"    [SAM错误] 异常: {sam_err}")

        finally:
            # V6.10性能优化：SAM使用完后立即释放
            try:
                node._unload_sam_model()
            except:
                pass

            # V6.11.1: SAM完成后彻底清理，为主模型重新加载腾出空间
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                    import gc
                    for _ in range(3):  # 多轮GC确保清理循环引用
                        gc.collect()
                    torch.cuda.empty_cache()
                    # 请求ComfyUI清理
                    comfy.model_management.soft_empty_cache()
                    # 打印显存状态
                    free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                    print(f"    [SAM] 释放完成，显存可用: {free_mem/1024**2:.0f}MB")
                except Exception as cleanup_err:
                    print(f"    [CUDA警告] 清理时出错: {cleanup_err}")

            # V6.11.2: 显式重新加载主模型到GPU，确保后续采样正常
            try:
                print(f"    [模型] 重新加载主模型到GPU...")
                comfy.model_management.load_model_gpu(ctx.模型)
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated(0) / 1024**2
                    print(f"    [模型] 加载完成，显存占用: {allocated:.0f}MB")
                    # V6.11.2: 检查模型实际设备
                    try:
                        model_device = next(ctx.模型.model.parameters()).device
                        print(f"    [模型] 模型实际设备: {model_device}")
                    except:
                        pass
            except Exception as load_err:
                print(f"    [模型警告] 重新加载模型时出错: {load_err}")
    return dets
