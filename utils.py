

import os
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

def setup_cuda_environment():
    
    # 常見的 CUDA 安裝路徑
    cuda_paths = [
        "/usr/local/cuda",
        "/usr/local/cuda-12.8",
        "/usr/local/cuda-12.6",
        "/usr/local/cuda-12.7",
        "/usr/local/cuda-11.8",
    ]
    
    # 查找存在的 CUDA 路徑
    cuda_path = None
    for path in cuda_paths:
        if os.path.exists(path) and os.path.isdir(path):
            lib64_path = os.path.join(path, "lib64")
            if os.path.exists(lib64_path):
                cuda_path = path
                break
    
    # 如果找到 CUDA 路徑，設置環境變量
    if cuda_path:
        lib64_path = os.path.join(cuda_path, "lib64")
        bin_path = os.path.join(cuda_path, "bin")
        
        # 設置 LD_LIBRARY_PATH
        current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        if lib64_path not in current_ld_path:
            if current_ld_path:
                os.environ["LD_LIBRARY_PATH"] = f"{lib64_path}:{current_ld_path}"
            else:
                os.environ["LD_LIBRARY_PATH"] = lib64_path
        
        # 設置 PATH（如果需要 nvcc）
        current_path = os.environ.get("PATH", "")
        if bin_path not in current_path:
            if current_path:
                os.environ["PATH"] = f"{bin_path}:{current_path}"
            else:
                os.environ["PATH"] = bin_path
        
        logger.info(f"已設置 CUDA 環境變量: {cuda_path}")
    else:
        # 嘗試從系統庫路徑查找
        try:
            import subprocess
            result = subprocess.run(
                ["ldconfig", "-p"],
                capture_output=True,
                text=True,
                timeout=5
            )
        except Exception as e:
            logger.debug(f"無法檢查系統 CUDA 庫: {e}")

def get_torchvision_weights_api() -> Tuple[bool, Optional[type]]:
    
    try:
        import torchvision
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
        
        # 檢查是否有 weights 屬性（新 API）
        if hasattr(FasterRCNN_ResNet50_FPN_Weights, 'DEFAULT'):
            logger.info("使用 torchvision weights API (新版本)")
            return True, FasterRCNN_ResNet50_FPN_Weights
        else:
            logger.info("使用 torchvision pretrained API (舊版本)")
            return False, None
    except ImportError:
        logger.warning("無法導入 torchvision，將使用舊版 API")
        return False, None
    except AttributeError:
        logger.info("torchvision 不支援 weights API，使用舊版 API")
        return False, None
    except Exception as e:
        logger.warning(f"檢查 torchvision weights API 時發生錯誤: {e}，使用舊版 API")
        return False, None
