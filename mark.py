

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

# CUDA 環境設置（必須在導入 torch 之前）
from utils import setup_cuda_environment, get_torchvision_weights_api

setup_cuda_environment()

import torch
import torchvision
from torch.amp import autocast
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou

# 檢查 torchvision weights API 可用性
USE_WEIGHTS_API, FasterRCNN_ResNet50_FPN_Weights = get_torchvision_weights_api()

# ============================================================================
# 配置和常量
# ============================================================================

# 日誌配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DetectionConfig:
    
    
    # 基本配置
    DEFAULT_NUM_CLASSES = 4
    IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp']
    
    # 類別特定閾值
    # 調整建議：
    #   - 降低閾值可以偵測到更多物體，但可能增加誤報
    #   - 提高閾值可以減少誤報，但可能漏掉一些真實物體
    #   - 建議根據實際偵測結果逐步調整
    CATEGORY_THRESHOLDS = {
        'RFID': 0.69,      # 提高閾值以減少誤報
        'colony': 0.3,    # 提高閾值以減少誤報
        'point': 0.35    # 降低從 0.05 到 0.03，偵測更多 point
    }
    
    # 預處理配置(與 model.py 保持一致)
    PREPROCESS_DENOISE_STRENGTH = 10  # 降噪強度
    PREPROCESS_SHARPEN_RADIUS = 5    # 銳化半徑
    PREPROCESS_CONTRAST_ALPHA = 1.0   # 對比度增強係數
    
    # 二值化配置(與 model.py 保持一致)
    BINARY_PETRI_DISH_THRESHOLD = 30  # 培養皿與背景分割的閾值(黑色背景 < 30)
    BINARY_OBJECTS_THRESHOLD = 127    # 物件分割的閾值(Otsu 自動閾值或固定閾值)
    BINARY_USE_OTSU = True            # 是否使用 Otsu 自動閾值進行物件分割
    
    # RFID 遮罩配置(數據增強策略，推理時通常不需要，但保留以保持與 model.py 一致)
    RFID_MASK_ENABLED = True          # 是否啟用 RFID 遮罩處理（與 model.py 保持一致，已開啟）
    RFID_MASK_MODE = "noise"           # 填充模式: "noise" (隨機噪點) 或 "mean" (平均灰階值)
    RFID_NOISE_INTENSITY = 0.5         # 隨機噪點強度 (0.0-1.0)，僅在 mode="noise" 時使用
    
    # 繪圖配置
    DRAW_SHOW_SCORES = True            # 是否顯示信心值分數
    DRAW_LINE_WIDTH = 2                # 邊界框線寬
    DRAW_FONT_SIZE = 16                # 字體大小
    DRAW_CATEGORY_COLORS = {
        'RFID': (170, 255, 255),       # 淺青色
        'colony': (255, 255, 170),    # 淺黃色
        'point': (255, 170, 255),      # 淺洋紅色
    }
    DRAW_DEFAULT_COLORS = [
        (255, 0, 0),    # 紅色
        (0, 255, 0),    # 綠色
        (0, 0, 255),    # 藍色
        (255, 255, 0),  # 黃色
        (255, 0, 255),  # 洋紅色
        (0, 255, 255),  # 青色
    ]
# 向後兼容：保留舊的常數名稱
DEFAULT_NUM_CLASSES = DetectionConfig.DEFAULT_NUM_CLASSES
CATEGORY_THRESHOLDS = DetectionConfig.CATEGORY_THRESHOLDS
IMAGE_EXTENSIONS = DetectionConfig.IMAGE_EXTENSIONS
PREPROCESS_DENOISE_STRENGTH = DetectionConfig.PREPROCESS_DENOISE_STRENGTH
PREPROCESS_SHARPEN_RADIUS = DetectionConfig.PREPROCESS_SHARPEN_RADIUS
PREPROCESS_CONTRAST_ALPHA = DetectionConfig.PREPROCESS_CONTRAST_ALPHA
BINARY_PETRI_DISH_THRESHOLD = DetectionConfig.BINARY_PETRI_DISH_THRESHOLD
BINARY_OBJECTS_THRESHOLD = DetectionConfig.BINARY_OBJECTS_THRESHOLD
BINARY_USE_OTSU = DetectionConfig.BINARY_USE_OTSU
RFID_MASK_ENABLED = DetectionConfig.RFID_MASK_ENABLED
RFID_MASK_MODE = DetectionConfig.RFID_MASK_MODE
RFID_NOISE_INTENSITY = DetectionConfig.RFID_NOISE_INTENSITY

# ============================================================================
# 偵測器類別
# ============================================================================
class FasterRCNNDetector:
    

    def __init__(
        self,
        model_path: Optional[str] = None,
        num_classes: int = DEFAULT_NUM_CLASSES,
        device: Optional[str] = None
    ):
    
        
        self.num_classes = num_classes
        
        # 設置設備
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # RTX 5090 優化：啟用 CUDNN benchmark 和 TF32
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"GPU 設備: {gpu_name}")
            logger.info(f"GPU 記憶體: {gpu_memory:.2f} GB")
            logger.info("已啟用 CUDNN benchmark 和 TF32 加速")
        
        logger.info(f"使用設備: {self.device}")
        
        # 載入模型
        if model_path is None:
            model_path = self._find_latest_model()
        
        if model_path is None:
            raise FileNotFoundError("找不到模型檔案，請指定 model_path 或確保 run/ 目錄中有訓練好的模型")
        
        self.model_path = Path(model_path)
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()
        
        # 載入類別資訊
        self.category_names = self._load_category_names()
        
        logger.info(f"✓ 模型載入完成: {self.model_path}")
    
    def _find_latest_model(self) -> Optional[str]:
        
        run_dir = Path("run")
        if not run_dir.exists():
            return None
        
        # 尋找所有 model_* 開頭的目錄
        model_dirs = [d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("model_")]
        
        if not model_dirs:
            return None
        
        # 按目錄名稱排序（時間戳格式：YYYYMMDD_HHMMSS）
        model_dirs.sort(key=lambda x: x.name, reverse=True)
        
        # 檢查每個目錄是否有模型檔案
        for model_dir in model_dirs:
            model_file = model_dir / "model.pth"
            model_full_file = model_dir / "model_full.pth"
            
            if model_file.exists() or model_full_file.exists():
                logger.info(f"找到模型: {model_dir}")
                return str(model_dir)
        
        return None
    
    def _load_model(self) -> torch.nn.Module:
        
        logger.info(f"正在載入模型: {self.model_path}")
        
        # 載入模型架構
        if USE_WEIGHTS_API:
            model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        else:
            model = fasterrcnn_resnet50_fpn(pretrained=True)
        
        # 替換分類頭
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        
        # 載入權重
        model_file = self.model_path / "model.pth"
        model_full_file = self.model_path / "model_full.pth"
        
        if model_full_file.exists():
            logger.info(f"載入完整模型: {model_full_file}")
            # PyTorch 2.6+ 需要設置 weights_only=False 來載入完整模型
            model = torch.load(model_full_file, map_location=self.device, weights_only=False)
        elif model_file.exists():
            logger.info(f"載入模型權重: {model_file}")
            # 載入 state_dict 時也需要設置 weights_only=False（PyTorch 2.6+）
            state_dict = torch.load(model_file, map_location=self.device, weights_only=False)
            model.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(f"在 {self.model_path} 中找不到模型檔案（model.pth 或 model_full.pth）")
        
        return model
    
    def _load_category_names(self) -> Dict[int, str]:
        
        info_file = self.model_path / "training_info.json"
        category_names = {0: "background"}  # 背景類別
        
        # 嘗試從 training_info.json 獲取訓練數據路徑
        train_annotation_file = None
        if info_file.exists():
            try:
                with open(info_file, 'r', encoding='utf-8') as f:
                    info = json.load(f)
                
                # 嘗試從訓練數據的 annotations.json 載入類別資訊
                train_image_dir = info.get('train_image_dir', 'model/train/images')
                # 推斷 annotations.json 的路徑
                train_annotation_file = Path(train_image_dir).parent / "annotations.json"
                
            except Exception as e:
                logger.warning(f"無法讀取 training_info.json: {e}")
        
        # 如果找不到，嘗試常見的路徑
        if train_annotation_file is None or not train_annotation_file.exists():
            common_paths = [
                Path("model/train/annotations.json"),
                Path("Preprocessing2/train/annotations.json"),
                Path("Preprocessing/train/annotations.json"),
                Path("c2/annotations.json"),
            ]
            for path in common_paths:
                if path.exists():
                    train_annotation_file = path
                    break
        
        # 從 annotations.json 載入類別資訊
        if train_annotation_file and train_annotation_file.exists():
            try:
                with open(train_annotation_file, 'r', encoding='utf-8') as f:
                    annotations_data = json.load(f)
                
                # 從 categories 中載入類別名稱
                for cat in annotations_data.get('categories', []):
                    cat_id = cat.get('id')
                    cat_name = cat.get('name')
                    if cat_id is not None and cat_name:
                        category_names[cat_id] = cat_name
                        logger.info(f"載入類別: {cat_id} -> {cat_name}")
                
            except Exception as e:
                logger.warning(f"無法從 {train_annotation_file} 載入類別資訊: {e}")
        
        # 如果沒有載入到類別資訊，使用預設名稱
        if len(category_names) == 1:  # 只有 background
            logger.warning("無法載入類別名稱，使用預設名稱 class_1, class_2, class_3")
            for i in range(1, self.num_classes):
                category_names[i] = f"class_{i}"
        
        return category_names
    
    @staticmethod
    def _binary_segment_petri_dish(img: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        
        # 轉換為灰階
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 二值化：黑色背景 < BINARY_PETRI_DISH_THRESHOLD，培養皿區域 >= BINARY_PETRI_DISH_THRESHOLD
        _, binary = cv2.threshold(gray, BINARY_PETRI_DISH_THRESHOLD, 255, cv2.THRESH_BINARY)
        
        # 形態學操作：去除小噪點，填充空洞
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 尋找最大連通區域（培養皿）
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        if num_labels < 2:
            # 如果找不到培養皿，返回原圖
            logger.warning("無法找到培養皿區域，返回原圖")
            return img, (0, 0)
        
        # 找出面積最大的連通區域（排除背景，索引 0）
        max_area_idx = 1
        max_area = stats[1, cv2.CC_STAT_AREA]
        for i in range(2, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > max_area:
                max_area = area
                max_area_idx = i
        
        # 獲取培養皿的邊界框
        x = stats[max_area_idx, cv2.CC_STAT_LEFT]
        y = stats[max_area_idx, cv2.CC_STAT_TOP]
        w = stats[max_area_idx, cv2.CC_STAT_WIDTH]
        h = stats[max_area_idx, cv2.CC_STAT_HEIGHT]
        
        # 添加一些邊距（避免裁剪過緊）
        margin = 10
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(img.shape[1] - x, w + 2 * margin)
        h = min(img.shape[0] - y, h + 2 * margin)
        
        # 裁剪培養皿區域
        petri_dish_region = img[y:y+h, x:x+w]
        
        return petri_dish_region, (x, y)
    
    @staticmethod
    def _binary_segment_objects(img: np.ndarray) -> np.ndarray:
        
        # 轉換為灰階
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 二值化分割物件
        if BINARY_USE_OTSU:
            # 使用 Otsu 自動閾值
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            # 使用固定閾值
            _, binary = cv2.threshold(gray, BINARY_OBJECTS_THRESHOLD, 255, cv2.THRESH_BINARY)
        
        # 形態學操作：去除小噪點
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        return binary
    
    @staticmethod
    def _detect_petri_dish_circle(img: np.ndarray) -> Optional[Tuple[int, int, int]]:
        
        # 轉換為灰階
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 方法1: 嘗試使用 HoughCircles 檢測圓形
        # 使用高斯模糊減少噪點
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # HoughCircles 參數
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=int(min(img.shape[:2]) * 0.5),  # 最小圓心距離
            param1=50,   # Canny 邊緣檢測的高閾值
            param2=30,   # 累積器閾值
            minRadius=int(min(img.shape[:2]) * 0.2),  # 最小半徑
            maxRadius=int(min(img.shape[:2]) * 0.45)  # 最大半徑
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            # 選擇半徑最大的圓（通常是培養皿）
            best_circle = circles[0][np.argmax(circles[0][:, 2])]
            center_x, center_y, radius = int(best_circle[0]), int(best_circle[1]), int(best_circle[2])
            logger.debug(f"使用 HoughCircles 檢測到圓形: center=({center_x}, {center_y}), radius={radius}")
            return (center_x, center_y, radius)
        
        # 方法2: 如果 HoughCircles 失敗，使用 Otsu + 輪廓檢測
        logger.debug("HoughCircles 未檢測到圓形，嘗試使用 Otsu + 輪廓檢測")
        
        # Otsu 二值化
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 形態學操作：去除小噪點，填充空洞
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 尋找輪廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            logger.warning("無法找到培養皿輪廓")
            return None
        
        # 找出面積最大的輪廓（通常是培養皿）
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 使用最小外接圓來近似培養皿
        (center_x, center_y), radius = cv2.minEnclosingCircle(largest_contour)
        center_x, center_y, radius = int(center_x), int(center_y), int(radius)
        
        logger.debug(f"使用輪廓檢測到圓形: center=({center_x}, {center_y}), radius={radius}")
        return (center_x, center_y, radius)
    
    @staticmethod
    def _get_rfid_bboxes_from_annotations(
        annotations: List[Dict],
        category_names: Optional[Dict[int, str]] = None
    ) -> List[np.ndarray]:
        
        rfid_bboxes = []
        
        # 找出 RFID 的 category_id（如果提供了 category_names）
        rfid_category_id = None
        if category_names is not None:
            for cat_id, cat_name in category_names.items():
                if cat_name == 'RFID':
                    rfid_category_id = cat_id
                    break
        
        # 轉換所有 RFID 標註的 bbox
        for ann in annotations:
            # 檢查是否為 RFID 標註
            is_rfid = False
            
            # 方法1: 通過 category_id 檢查（JSON/COCO 格式）
            if rfid_category_id is not None and ann.get('category_id') == rfid_category_id:
                is_rfid = True
            # 方法2: 直接檢查 category_name（XML 格式或包含 category_name 的格式）
            elif ann.get('category_name', '').lower() == 'rfid':
                is_rfid = True
            
            if not is_rfid:
                continue
            
            # 轉換 COCO 格式 [x, y, w, h] 到 [x_min, y_min, x_max, y_max]
            bbox_coco = ann['bbox']
            x_min = bbox_coco[0]
            y_min = bbox_coco[1]
            x_max = x_min + bbox_coco[2]
            y_max = y_min + bbox_coco[3]
            rfid_bboxes.append(np.array([x_min, y_min, x_max, y_max], dtype=np.float32))
        
        return rfid_bboxes
    
    @staticmethod
    def _apply_rfid_mask(
        image: np.ndarray,
        rfid_bboxes: List[np.ndarray],
        mode: str = "noise",
        noise_intensity: float = 0.3
    ) -> np.ndarray:
        
        if len(rfid_bboxes) == 0:
            return image
        
        result = image.copy()
        h, w = image.shape[:2]
        
        for bbox in rfid_bboxes:
            x_min, y_min, x_max, y_max = bbox.astype(int)
            
            # 確保座標在圖像範圍內
            x_min = max(0, min(w, x_min))
            y_min = max(0, min(h, y_min))
            x_max = max(0, min(w, x_max))
            y_max = max(0, min(h, y_max))
            
            if x_max <= x_min or y_max <= y_min:
                continue
            
            # 提取 RFID 區域
            rfid_region = result[y_min:y_max, x_min:x_max]
            
            if mode == "mean":
                # 使用該區域的平均灰階值填充
                mean_value = np.mean(rfid_region).astype(np.uint8)
                result[y_min:y_max, x_min:x_max] = mean_value
            elif mode == "noise":
                # 使用隨機噪點填充
                # 生成與原區域相同尺寸的隨機噪點
                noise = np.random.randint(0, 256, size=rfid_region.shape, dtype=np.uint8)
                # 混合原圖和噪點
                result[y_min:y_max, x_min:x_max] = (
                    rfid_region.astype(np.float32) * (1 - noise_intensity) +
                    noise.astype(np.float32) * noise_intensity
                ).astype(np.uint8)
            else:
                logger.warning(f"未知的 RFID 遮罩模式: {mode}，跳過處理")
        
        return result
    
    @staticmethod
    def _new_preprocess_with_circle_detection(
        img: np.ndarray,
        annotations: Optional[List[Dict]] = None,
        category_names: Optional[Dict[int, str]] = None
    ) -> np.ndarray:
        
        original_shape = img.shape[:2]  # 保存原始尺寸 (H, W)
        
        # 1. 轉換為灰階
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. 檢測培養皿圓形區域
        circle_info = FasterRCNNDetector._detect_petri_dish_circle(img)
        
        if circle_info is None:
            logger.warning("無法檢測到培養皿圓形區域，返回原圖")
            # 即使沒有圓形區域，仍然可以應用 RFID 遮罩（如果啟用）
            if RFID_MASK_ENABLED and annotations is not None and category_names is not None:
                rfid_bboxes = FasterRCNNDetector._get_rfid_bboxes_from_annotations(
                    annotations, category_names
                )
                if len(rfid_bboxes) > 0:
                    gray = FasterRCNNDetector._apply_rfid_mask(
                        gray, rfid_bboxes, RFID_MASK_MODE, RFID_NOISE_INTENSITY
                    )
            return gray
        
        center_x, center_y, radius = circle_info
        
        # 3. 建立遮罩：圓形區域為白色(255)，背景為黑色(0)
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(mask, (center_x, center_y), radius, 255, -1)  # 填充圓形
        
        # 4. 將圓形區域以外的背景填滿純黑色
        masked_gray = gray.copy()
        masked_gray[mask == 0] = 0  # 圓形區域外設為黑色
        
        # 5. 對圓形區域內部進行 CLAHE（自適應直方圖均衡化）
        # 創建 CLAHE 對象
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # 對整個圖像應用 CLAHE（但只保留圓形區域內的部分）
        clahe_result = clahe.apply(masked_gray)
        
        # 只保留圓形區域內的 CLAHE 結果，圓形區域外保持黑色
        masked_gray[mask > 0] = clahe_result[mask > 0]
        
        # 6. 應用 RFID 遮罩（數據增強策略，如果啟用）
        if RFID_MASK_ENABLED and annotations is not None and category_names is not None:
            rfid_bboxes = FasterRCNNDetector._get_rfid_bboxes_from_annotations(
                annotations, category_names
            )
            if len(rfid_bboxes) > 0:
                masked_gray = FasterRCNNDetector._apply_rfid_mask(
                    masked_gray, rfid_bboxes, RFID_MASK_MODE, RFID_NOISE_INTENSITY
                )
                logger.debug(f"已應用 RFID 遮罩: {len(rfid_bboxes)} 個 RFID 區域")
        
        # 確保影像尺寸不變
        assert masked_gray.shape == original_shape, \
            f"影像尺寸改變: 原始={original_shape}, 處理後={masked_gray.shape}"
        
        return masked_gray
    
    @staticmethod
    def _preprocess_image(
        img: np.ndarray,
        annotations: Optional[List[Dict]] = None,
        category_names: Optional[Dict[int, str]] = None
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        
        # 使用新的預處理方法（圓形檢測 + CLAHE）
        processed_gray = FasterRCNNDetector._new_preprocess_with_circle_detection(
            img, annotations=annotations, category_names=category_names
        )
        
        # 轉換為 RGB 格式（PIL Image 需要）
        # 將單通道灰階圖像複製為 3 通道 RGB
        processed_rgb = cv2.cvtColor(processed_gray, cv2.COLOR_GRAY2RGB)
        
        # 因為保持影像尺寸不變，所以 crop_offset 為 (0, 0)
        return processed_rgb, (0, 0)
    
    @staticmethod
    def _check_box_inside_rfid(box: torch.Tensor, rfid_boxes: torch.Tensor) -> bool:
        
        if len(rfid_boxes) == 0:
            return False
        
        # 計算框的中心點
        box_center_x = (box[0] + box[2]) / 2
        box_center_y = (box[1] + box[3]) / 2
        
        # 檢查中心點是否在任何 RFID 框內
        center_inside = (
            (box_center_x >= rfid_boxes[:, 0]) & 
            (box_center_x <= rfid_boxes[:, 2]) &
            (box_center_y >= rfid_boxes[:, 1]) & 
            (box_center_y <= rfid_boxes[:, 3])
        )
        return center_inside.any().item()
    
    @staticmethod
    def _check_box_inside_colony(box: torch.Tensor, colony_boxes: torch.Tensor) -> bool:
        
        if len(colony_boxes) == 0:
            return False
        
        # 計算框的中心點
        box_center_x = (box[0] + box[2]) / 2
        box_center_y = (box[1] + box[3]) / 2
        
        # 檢查中心點是否在任何 colony 框內
        center_inside = (
            (box_center_x >= colony_boxes[:, 0]) & 
            (box_center_x <= colony_boxes[:, 2]) &
            (box_center_y >= colony_boxes[:, 1]) & 
            (box_center_y <= colony_boxes[:, 3])
        )
        return center_inside.any().item()
    
    @staticmethod
    def _calculate_box_area(box: torch.Tensor) -> float:
        """計算邊界框的面積"""
        width = box[2] - box[0]
        height = box[3] - box[1]
        return width * height
    
    @staticmethod
    def _calculate_overlap_ratio(box1: torch.Tensor, box2: torch.Tensor) -> float:
        """計算兩個框的重疊面積相對於較小框的面積比例"""
        # 計算交集區域
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # 交集區域
        inter_x_min = torch.max(x1_min, x2_min)
        inter_y_min = torch.max(y1_min, y2_min)
        inter_x_max = torch.min(x1_max, x2_max)
        inter_y_max = torch.min(y1_max, y2_max)
        
        # 如果沒有交集，返回0
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        
        # 交集面積
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # 計算兩個框的面積
        area1 = FasterRCNNDetector._calculate_box_area(box1)
        area2 = FasterRCNNDetector._calculate_box_area(box2)
        
        # 重疊面積相對於較小框的比例
        min_area = min(area1, area2)
        if min_area == 0:
            return 0.0
        
        overlap_ratio = inter_area / min_area
        return overlap_ratio
    
    def _analyze_box_color_features(
        self,
        box: torch.Tensor,
        original_image: np.ndarray
    ) -> Dict[str, float]:
        
        x1, y1, x2, y2 = box.cpu().numpy().astype(int)
        h, w = original_image.shape[:2]
        
        # 確保座標在圖像範圍內
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))
        
        if x2 <= x1 or y2 <= y1:
            return {'center_darkness': 0.5, 'edge_darkness': 0.5, 'solidity_ratio': 1.0}
        
        # 提取邊界框區域
        box_region = original_image[y1:y2, x1:x2]
        
        # 轉換為灰階（如果原本是 RGB）
        if len(box_region.shape) == 3:
            gray_region = cv2.cvtColor(box_region, cv2.COLOR_RGB2GRAY)
        else:
            gray_region = box_region
        
        # 計算中心區域（取中間 50% 的區域）
        center_x1 = int((x2 - x1) * 0.25)
        center_y1 = int((y2 - y1) * 0.25)
        center_x2 = int((x2 - x1) * 0.75)
        center_y2 = int((y2 - y1) * 0.75)
        
        center_region = gray_region[center_y1:center_y2, center_x1:center_x2]
        
        # 計算邊緣區域（外圍 25% 的區域）
        edge_mask = np.ones_like(gray_region, dtype=bool)
        edge_mask[center_y1:center_y2, center_x1:center_x2] = False
        edge_region = gray_region[edge_mask]
        
        # 計算平均暗度（0-255 轉換為 0-1，值越小越暗）
        center_mean = center_region.mean() if center_region.size > 0 else 128
        edge_mean = edge_region.mean() if edge_region.size > 0 else 128
        
        # 轉換為暗度（0-1，1 表示最暗）
        center_darkness = 1.0 - (center_mean / 255.0)
        edge_darkness = 1.0 - (edge_mean / 255.0)
        
        # 計算實心度比例（>1 表示中心比邊緣暗，即實心）
        if edge_darkness > 0.01:  # 避免除零
            solidity_ratio = center_darkness / edge_darkness
        else:
            solidity_ratio = 1.0
        
        return {
            'center_darkness': center_darkness,
            'edge_darkness': edge_darkness,
            'solidity_ratio': solidity_ratio
        }
    
    def _is_text_region(
        self,
        box: torch.Tensor,
        original_image: np.ndarray
    ) -> bool:
        """
        檢測邊界框區域是否為文字區域
        文字特徵：高對比度、邊緣密集、矩形形狀明顯
        """
        x1, y1, x2, y2 = box.cpu().numpy().astype(int)
        h, w = original_image.shape[:2]
        
        # 確保座標在圖像範圍內
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))
        
        if x2 <= x1 or y2 <= y1:
            return False
        
        # 提取邊界框區域
        box_region = original_image[y1:y2, x1:x2]
        
        # 轉換為灰階
        if len(box_region.shape) == 3:
            gray_region = cv2.cvtColor(box_region, cv2.COLOR_RGB2GRAY)
        else:
            gray_region = box_region
        
        # 計算區域的寬高比（文字通常是橫向的）
        box_width = x2 - x1
        box_height = y2 - y1
        aspect_ratio = box_width / max(box_height, 1)
        
        # 計算對比度（標準差）
        contrast = np.std(gray_region)
        
        # 使用 Canny 邊緣檢測計算邊緣密度
        edges = cv2.Canny(gray_region, 50, 150)
        edge_density = np.sum(edges > 0) / max(box_width * box_height, 1)
        
        # 計算水平投影（文字通常有明顯的水平線條）
        horizontal_projection = np.sum(gray_region < 128, axis=1)  # 暗像素數量
        horizontal_variance = np.var(horizontal_projection)
        
        # 文字判斷條件：
        # 1. 寬高比 > 1.5（橫向文字）
        # 2. 對比度 > 30（高對比度）
        # 3. 邊緣密度 > 0.1（邊緣密集）
        # 4. 水平投影變異數 > 100（有明顯的水平線條特徵）
        is_text = (
            aspect_ratio > 1.5 and
            contrast > 30 and
            edge_density > 0.1 and
            horizontal_variance > 100
        )
        
        return is_text
    
    def detect(
        self,
        image: torch.Tensor,
        threshold: Optional[float] = None,
        category_thresholds: Optional[Dict[str, float]] = None,
        original_image: Optional[np.ndarray] = None
    ) -> Dict:
        
        with torch.no_grad():
            # RTX 5090 優化：使用混合精度推理
            if torch.cuda.is_available():
                with autocast(device_type='cuda'):
                    predictions = self.model([image.to(self.device, non_blocking=True)])
            else:
                predictions = self.model([image.to(self.device)])
        
        prediction = predictions[0]
        
        # 確定使用的閾值設定
        if threshold is not None:
            # 如果提供了全域閾值，使用全域閾值（所有類別使用相同閾值）
            use_category_thresholds = False
            global_threshold = threshold
        else:
            # 使用類別特定閾值
            use_category_thresholds = True
            thresholds_dict = category_thresholds if category_thresholds is not None else CATEGORY_THRESHOLDS
        
        # 過濾低置信度檢測（使用類別特定閾值或全域閾值）
        scores = prediction['scores']
        labels = prediction['labels']
        
        # 調試：記錄模型原始輸出
        logger.info("=" * 60)
        logger.info("模型原始輸出統計（過濾前）:")
        unique_labels_raw = torch.unique(labels)
        for label_id in unique_labels_raw:
            label_mask = labels == label_id
            if label_mask.sum() > 0:
                label_scores = scores[label_mask]
                category_name = self.category_names.get(int(label_id), f"class_{label_id}")
                logger.info(f"  {category_name} (id={label_id}): {label_mask.sum()} 個偵測")
                logger.info(f"    信心值範圍: {label_scores.min().item():.4f} ~ {label_scores.max().item():.4f}")
                logger.info(f"    平均信心值: {label_scores.mean().item():.4f}")
        logger.info("=" * 60)
        
        if use_category_thresholds:
            # 按類別分別應用閾值
            keep_mask = torch.zeros(len(scores), dtype=torch.bool, device=scores.device)
            
            for label_id, category_name in self.category_names.items():
                if label_id == 0:  # 跳過背景類別
                    continue
                
                # 獲取該類別的閾值（如果未設定，記錄警告並跳過）
                if category_name not in thresholds_dict:
                    logger.warning(f"類別 '{category_name}' 未在 CATEGORY_THRESHOLDS 中設定，將跳過該類別的偵測")
                    continue
                
                category_threshold = thresholds_dict[category_name]
                
                # 找出該類別的所有偵測
                label_mask = labels == label_id
                if label_mask.sum() > 0:
                    # 應用該類別的閾值
                    label_scores = scores[label_mask]
                    label_keep = label_scores >= category_threshold
                    keep_mask[label_mask] = label_keep
                    
                    # 調試：記錄過濾結果
                    passed_count = label_keep.sum().item()
                    total_count = label_mask.sum().item()
                    if passed_count < total_count:
                        logger.debug(f"  {category_name}: {total_count} 個偵測，{passed_count} 個通過閾值 {category_threshold}")
                    elif passed_count > 0:
                        logger.debug(f"  {category_name}: {passed_count} 個偵測全部通過閾值 {category_threshold}")
        else:
            # 使用全域閾值
            keep_mask = scores >= global_threshold
        
        boxes = prediction['boxes'][keep_mask]
        labels = labels[keep_mask]
        scores = scores[keep_mask]
        
        # 轉換為 numpy 以便處理
        boxes_np = boxes.cpu().numpy()
        labels_np = labels.cpu().numpy()
        scores_np = scores.cpu().numpy()
        
        return {
            'boxes': boxes_np,
            'labels': labels_np,
            'scores': scores_np
        }
    def detect_from_path(
        self,
        image_path: str,
        threshold: Optional[float] = None,
        category_thresholds: Optional[Dict[str, float]] = None
    ) -> Tuple[np.ndarray, Dict]:
        
        # 使用 OpenCV 讀取圖像（BGR 格式）
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise ValueError(f"無法讀取圖像: {image_path}")
        
        # 保存原始圖像（用於返回和繪製）
        original_image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # 執行第一次二值化分割（培養皿和背景）
        petri_dish_region, crop_offset_first = self._binary_segment_petri_dish(img_bgr)
        
        # 執行第二次二值化分割（物件分割）
        self._binary_segment_objects(petri_dish_region)
        
        # 預處理圖像（圓形檢測 + CLAHE，與 model.py 保持一致）
        processed_rgb, crop_offset = self._preprocess_image(img_bgr)
        x_offset, y_offset = crop_offset
        
        # 轉換為 PIL Image
        image = Image.fromarray(processed_rgb)
        
        # 轉換為張量
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        image_tensor = transform(image)
        
        # 進行偵測
        detections = self.detect(
            image_tensor,
            threshold=threshold,
            category_thresholds=category_thresholds,
            original_image=original_image
        )
        # 調整偵測結果的座標：加上裁剪偏移量（通常為 (0, 0)，因為保持影像尺寸不變）
        if len(detections['boxes']) > 0:
            detections['boxes'][:, 0] += x_offset  # x_min
            detections['boxes'][:, 1] += y_offset  # y_min
            detections['boxes'][:, 2] += x_offset  # x_max
            detections['boxes'][:, 3] += y_offset  # y_max
        return original_image, detections
    
    def draw_detections(
        self,
        image: np.ndarray,
        detections: Dict,
        show_scores: Optional[bool] = None,
        line_width: Optional[int] = None
    ) -> np.ndarray:
        
        # 使用配置類的預設值
        config = DetectionConfig
        if show_scores is None:
            show_scores = config.DRAW_SHOW_SCORES
        if line_width is None:
            line_width = config.DRAW_LINE_WIDTH
        
        # 轉換為 PIL Image 以便繪製
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)
        
        # 嘗試載入字體
        font_size = config.DRAW_FONT_SIZE
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                font = ImageFont.load_default()
        
        boxes = detections['boxes']
        labels = detections['labels']
        scores = detections['scores']
        
        # 使用配置類中的顏色設定
        category_colors = config.DRAW_CATEGORY_COLORS
        default_colors = config.DRAW_DEFAULT_COLORS
        
        for i in range(len(boxes)):
            box = boxes[i]
            label = labels[i]
            score = scores[i]
            
            # 獲取類別名稱
            category_name = self.category_names.get(int(label), f"class_{label}")
            # 根據類別名稱選擇顏色
            if category_name in category_colors:
                color = category_colors[category_name]
            else:
                # 如果類別名稱不在映射中，使用預設顏色
                color = default_colors[int(label) % len(default_colors)]

            # 繪製邊界框
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
            
            # 繪製標籤
            label_text = category_name
            if show_scores:
                label_text += f" {score:.2f}"
            
            # 計算文字位置（在邊界框上方）
            bbox = draw.textbbox((0, 0), label_text, font=font)
            text_height = bbox[3] - bbox[1]
            
            # 繪製文字（使用與邊框相同的顏色，無背景框）
            draw.text((x1, y1 - text_height - 2), label_text, fill=color, font=font)
        
        return np.array(pil_image)
    
    def process_directory(
        self,
        input_dir: str = "final/input",
        output_dir: str = "final/output",
        threshold: Optional[float] = None,
        category_thresholds: Optional[Dict[str, float]] = None,
        save_images: bool = True
    ) -> Dict:
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if not input_path.exists():
            raise FileNotFoundError(f"輸入目錄不存在: {input_dir}")
        
        # 收集所有圖片檔案
        image_files = []
        for ext in IMAGE_EXTENSIONS:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        image_files = sorted(image_files)
        
        if len(image_files) == 0:
            logger.warning(f"在 {input_dir} 中未找到圖片檔案")
            return {
                "total_images": 0,
                "processed_images": 0,
                "total_detections": 0
            }
        
        logger.info(f"找到 {len(image_files)} 張圖片，開始處理...")
        
        # 處理結果
        results = []
        total_detections = 0
        
        for idx, image_file in enumerate(image_files, 1):
            logger.info(f"[{idx}/{len(image_files)}] 處理: {image_file.name}")
            try:
                # 進行偵測
                image, detections = self.detect_from_path(
                    str(image_file),
                    threshold=threshold,
                    category_thresholds=category_thresholds
                )
                
                num_detections = len(detections['boxes'])
                total_detections += num_detections
                
                # 保存標記後的圖片
                if save_images:
                    marked_image = self.draw_detections(image, detections)
                    output_image_path = output_path / image_file.name
                    Image.fromarray(marked_image).save(output_image_path)
                
                # 記錄結果
                result = {
                    "image_file": image_file.name,
                    "num_detections": num_detections,
                    "detections": []
                }
                
                for i in range(num_detections):
                    box = detections['boxes'][i]
                    label = int(detections['labels'][i])
                    score = float(detections['scores'][i])
                    category_name = self.category_names.get(label, f"class_{label}")
                    
                    result["detections"].append({
                        "category_id": label,
                        "category_name": category_name,
                        "bbox": box.tolist(),  # [x1, y1, x2, y2]
                        "score": score
                    })
                
                results.append(result)
                logger.info(f"  ✓ 偵測到 {num_detections} 個物體")
                
            except Exception as e:
                logger.error(f"  ✗ 處理 {image_file.name} 時發生錯誤: {e}")
                results.append({
                    "image_file": image_file.name,
                    "error": str(e)
                })
        
        # 保存摘要
        summary = {
            "timestamp": datetime.now().isoformat(),
            "model_path": str(self.model_path),
            "num_classes": self.num_classes,
            "threshold": threshold,
            "total_images": len(image_files),
            "processed_images": len([r for r in results if "error" not in r]),
            "total_detections": total_detections,
            "results": results
        }
        summary_file = output_path / "detection_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info("=" * 60)
        logger.info("處理完成")
        logger.info("=" * 60)
        logger.info(f"總圖片數: {len(image_files)}")
        logger.info(f"成功處理: {summary['processed_images']}")
        logger.info(f"總偵測數: {total_detections}")
        logger.info(f"摘要檔案: {summary_file}")
        
        return summary

def main():
    
    logger.info("=" * 60)
    logger.info("Faster R-CNN 物體偵測與標記")
    logger.info("=" * 60)
    
    try:
        # 創建偵測器（自動尋找最新模型）
        detector = FasterRCNNDetector(
            model_path=None,  # 自動尋找最新模型
            num_classes=4  # 背景 + 3個類別（RFID、colony、point）
        )
        # 批次處理目錄
        # 【信心值閾值調整】
        # - 可選：提供自訂類別閾值
        summary = detector.process_directory(
            input_dir="final/input",
            output_dir="final/output",
            threshold=None,              # None = 使用 CATEGORY_THRESHOLDS
            category_thresholds=None     # None = 使用預設的 CATEGORY_THRESHOLDS
        )
        
        logger.info(f"\n✓ 處理完成！結果已保存到 final/output/")
        
    except FileNotFoundError as e:
        logger.error(f"檔案或目錄不存在: {e}")
    except Exception as e:
        logger.error(f"處理過程發生錯誤: {e}", exc_info=True)

if __name__ == "__main__":
    main()