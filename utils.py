

import os
import logging
from typing import Tuple, Optional, Any
import numpy as np
import cv2

logger = logging.getLogger(__name__)

def setup_cuda_environment():
    """設置 CUDA 環境變數。"""
    cuda_paths = [
        "/usr/local/cuda",
        "/usr/local/cuda-12.8",
        "/usr/local/cuda-12.6",
        "/usr/local/cuda-12.7",
        "/usr/local/cuda-11.8",
    ]
    
    cuda_path = next((p for p in cuda_paths if os.path.exists(os.path.join(p, "lib64"))), None)
    
    if cuda_path:
        lib64_path = os.path.join(cuda_path, "lib64")
        bin_path = os.path.join(cuda_path, "bin")
        
        for env_var, path in [("LD_LIBRARY_PATH", lib64_path), ("PATH", bin_path)]:
            current = os.environ.get(env_var, "")
            if path not in current:
                os.environ[env_var] = f"{path}:{current}" if current else path
        
        logger.info(f"已設置 CUDA 環境變數: {cuda_path}")

def get_torchvision_weights_api() -> Tuple[bool, Optional[type]]:
    """檢查 torchvision 是否支援新的 weights API。"""
    try:
        import torchvision
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
        if hasattr(FasterRCNN_ResNet50_FPN_Weights, 'DEFAULT'):
            return True, FasterRCNN_ResNet50_FPN_Weights
    except (ImportError, AttributeError):
        pass
    return False, None

def preprocess_image(img: np.ndarray, denoise_strength: int = 10, sharpen_radius: int = 5, contrast_alpha: float = 1.0) -> np.ndarray:
    """
    通用影像預處理流程：
    1. 轉灰階
    2. 銳利化 (Unsharp Masking)
    3. 降噪 (Non-local Means)
    4. 對比度增強 (CLAHE)
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # 銳利化
    k_size = int(sharpen_radius * 2 + 1)
    if k_size % 2 == 0: k_size += 1
    blurred = cv2.GaussianBlur(gray, (k_size, k_size), sharpen_radius)
    sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    # 降噪
    denoised = cv2.fastNlMeansDenoising(sharpened, h=denoise_strength)

    # 對比度增強
    clahe = cv2.createCLAHE(clipLimit=contrast_alpha * 2.0, tileGridSize=(8, 8))
    return clahe.apply(denoised)
