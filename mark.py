"""
Faster R-CNN 物體偵測與標記模組
使用訓練好的模型對圖片進行物體偵測，並在圖片上標記偵測結果
優化以兼容 GPU 5090 環境
"""

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
from torchvision.ops import nms, box_iou

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
    """偵測器配置類，統一管理所有設定參數"""
    
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
        'colony': 0.395,   # 進一步降低以偵測更多 colony
        'point': 0.01    # 降低從 0.05 到 0.03，偵測更多 point
    }
    # NMS（非極大值抑制）閾值配置(越高越寬鬆)
    DEFAULT_NMS_THRESHOLD = 0.6  # 同類別 NMS 閾值
    DEFAULT_CROSS_CATEGORY_NMS_THRESHOLD = 0.8  # 跨類別 NMS 閾值
    
    # NMS 過濾開關(True = 啟用，False = 禁用)
    DEFAULT_ENABLE_SAME_CATEGORY_NMS = False  # 同類別過濾
    DEFAULT_ENABLE_CROSS_CATEGORY_NMS = True  # 跨類別過濾
    
    # 模型推理 NMS 閾值(越高越寬鬆)
    MODEL_INFERENCE_NMS_THRESHOLD = 0.3  # 模型內部 NMS 閾值
    
    # 預處理配置(與 model.py 保持一致)
    PREPROCESS_DENOISE_STRENGTH = 10  # 降噪強度
    PREPROCESS_SHARPEN_RADIUS = 5    # 銳化半徑
    PREPROCESS_CONTRAST_ALPHA = 1.0   # 對比度增強係數
    
    # 二值化配置(與 model.py 保持一致)
    BINARY_PETRI_DISH_THRESHOLD = 30  # 培養皿與背景分割的閾值(黑色背景 < 30)
    BINARY_OBJECTS_THRESHOLD = 127    # 物件分割的閾值(Otsu 自動閾值或固定閾值)
    BINARY_USE_OTSU = True            # 是否使用 Otsu 自動閾值進行物件分割
    
    # RFID 遮罩配置(數據增強策略，推理時通常不需要，但保留以保持與 model.py 一致)
    RFID_MASK_ENABLED = False          # 是否啟用 RFID 遮罩處理（與 model.py 保持一致，已關閉）
    RFID_MASK_MODE = "noise"           # 填充模式: "noise" (隨機噪點) 或 "mean" (平均灰階值)
    RFID_NOISE_INTENSITY = 0.3         # 隨機噪點強度 (0.0-1.0)，僅在 mode="noise" 時使用
    
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
DEFAULT_NMS_THRESHOLD = DetectionConfig.DEFAULT_NMS_THRESHOLD
DEFAULT_CROSS_CATEGORY_NMS_THRESHOLD = DetectionConfig.DEFAULT_CROSS_CATEGORY_NMS_THRESHOLD
DEFAULT_ENABLE_SAME_CATEGORY_NMS = DetectionConfig.DEFAULT_ENABLE_SAME_CATEGORY_NMS
DEFAULT_ENABLE_CROSS_CATEGORY_NMS = DetectionConfig.DEFAULT_ENABLE_CROSS_CATEGORY_NMS
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
    """Faster R-CNN 物體偵測器（優化支援 RTX 5090）"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        num_classes: int = DEFAULT_NUM_CLASSES,
        device: Optional[str] = None
    ):
        """
        初始化偵測器
        
        Args:
            model_path: 模型路徑（如果為 None，自動尋找最新模型）
            num_classes: 類別數量（包括背景）
            device: 設備（'cuda' 或 'cpu'，None 時自動選擇）
        """
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
        """
        自動尋找最新的訓練模型
        
        Returns:
            最新模型目錄路徑，如果找不到則返回 None
        """
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
        """
        載入訓練好的模型
        
        Returns:
            載入的模型
        """
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
        
        # 設置 Inference 階段的 box_nms_thresh（從 0.5 調降到 0.3）
        # 降低 NMS 閾值可以避免大菌落被錯誤過濾，允許更多候選框通過
        if hasattr(model.roi_heads, 'nms_thresh'):
            model.roi_heads.nms_thresh = 0.3
            logger.info(f"已設置模型推理 NMS 閾值: {model.roi_heads.nms_thresh}")
        elif hasattr(model.roi_heads, 'box_nms_thresh'):
            model.roi_heads.box_nms_thresh = 0.3
            logger.info(f"已設置模型推理 box_nms_thresh: {model.roi_heads.box_nms_thresh}")
        else:
            logger.warning("無法找到模型 roi_heads 的 nms_thresh 或 box_nms_thresh 屬性")
        
        return model
    
    def _load_category_names(self) -> Dict[int, str]:
        """
        從 training_info.json 和 annotations.json 載入類別名稱
        
        Returns:
            category_id -> category_name 的映射
        """
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
        """
        第一步二值化：分割培養皿和黑色背景（四個角落的黑色背景）
        
        Args:
            img: 原始圖像（BGR 格式）
        
        Returns:
            (petri_dish_region, crop_offset) 元組
            - petri_dish_region: 裁剪後的培養皿區域（BGR 格式）
            - crop_offset: (x_offset, y_offset) 裁剪偏移量，用於調整標註座標
        """
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
        """
        第二步二值化：在培養皿區域內分割物件（RFID、colony、point）
        
        Args:
            img: 培養皿區域圖像（BGR 格式）
            
        Returns:
            二值化後的圖像（灰階，255 = 物件，0 = 背景）
        """
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
        """
        使用 HoughCircles 或 Otsu + 輪廓檢測來定位培養皿的圓形區域
        
        Args:
            img: 原始圖像（BGR 格式）
            
        Returns:
            (center_x, center_y, radius) 元組，如果找不到則返回 None
        """
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
        """
        從標註中獲取 RFID 的邊界框座標（支援 JSON 和 XML 格式）
        
        Args:
            annotations: 標註列表（COCO 格式或包含 'category_name' 的格式）
            category_names: category_id -> category_name 的映射（可選，用於 JSON 格式）
        
        Returns:
            RFID 邊界框列表，每個邊界框為 [x_min, y_min, x_max, y_max] 格式
        """
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
        """
        將 RFID 區域填充為隨機噪點或平均灰階值
        
        Args:
            image: 圖像陣列（灰階）
            rfid_bboxes: RFID 邊界框列表，每個為 [x_min, y_min, x_max, y_max]
            mode: 填充模式，"noise" (隨機噪點) 或 "mean" (平均灰階值)
            noise_intensity: 隨機噪點強度 (0.0-1.0)，僅在 mode="noise" 時使用
            
        Returns:
            處理後的圖像陣列
        """
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
        """
        新的預處理方法：使用圓形檢測 + CLAHE + RFID 遮罩（與 model.py 保持一致）
        
        處理步驟：
        1. 將影像轉為灰階
        2. 使用 HoughCircles 或 Otsu + 輪廓檢測來定位培養皿的圓形區域
        3. 建立遮罩，將圓形區域以外的背景填滿純黑色
        4. 對圓形區域內部進行 CLAHE（自適應直方圖均衡化）
        5. 如果啟用 RFID 遮罩，將 RFID 區域填充為隨機噪點或平均灰階值
        6. 保持影像尺寸不變
        
        Args:
            img: 原始圖像（BGR 格式，從 cv2.imread 讀取）
            annotations: COCO 格式的標註列表（可選）
            category_names: category_id -> category_name 的映射（可選）
            
        Returns:
            預處理後的圖像陣列（灰階，尺寸與原圖相同）
        """
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
        """
        使用圓形檢測 + CLAHE 進行圖像預處理（與 model.py 保持一致）
        
        處理步驟：
        1. 將影像轉為灰階
        2. 使用 HoughCircles 或 Otsu + 輪廓檢測來定位培養皿的圓形區域
        3. 建立遮罩，將圓形區域以外的背景填滿純黑色
        4. 對圓形區域內部進行 CLAHE（自適應直方圖均衡化）
        5. 如果啟用 RFID 遮罩，將 RFID 區域填充為隨機噪點或平均灰階值
        6. 保持影像尺寸不變
        7. 轉換為 RGB 格式（PIL Image 需要）
        
        Args:
            img: 原始圖像（BGR 格式，從 cv2.imread 讀取）
            annotations: COCO 格式的標註列表（可選）
            category_names: category_id -> category_name 的映射（可選）
        
        Returns:
            (預處理後的圖像（RGB 格式）, crop_offset) 元組
            - crop_offset: (0, 0) 因為保持影像尺寸不變，不需要調整座標
        """
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
        """
        檢查框是否在 RFID 框內部（檢查中心點是否在內部）
        
        Args:
            box: 要檢查的框 [x_min, y_min, x_max, y_max]
            rfid_boxes: RFID 框列表 [N, 4]
            
        Returns:
            如果框的中心點在任何 RFID 框內部返回 True，否則返回 False
        """
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
        """
        檢查框是否在 colony 框內部（檢查中心點是否在內部）
        
        Args:
            box: 要檢查的框 [x_min, y_min, x_max, y_max]
            colony_boxes: colony 框列表 [N, 4]
            
        Returns:
            如果框的中心點在任何 colony 框內部返回 True，否則返回 False
        """
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
    
    def _analyze_box_color_features(
        self,
        box: torch.Tensor,
        original_image: np.ndarray
    ) -> Dict[str, float]:
        """
        分析邊界框內的顏色特徵，用於區分實心 colony 和空心 point
        
        Args:
            box: 邊界框 [x1, y1, x2, y2]
            original_image: 原始圖像 (H, W, 3) RGB 格式
            
        Returns:
            特徵字典，包含：
            - center_darkness: 中心區域的平均暗度（0-1，越高越暗）
            - edge_darkness: 邊緣區域的平均暗度（0-1，越高越暗）
            - solidity_ratio: 實心度比例（中心暗度/邊緣暗度，>1 表示實心）
        """
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
    
    def _apply_nms_filtering(
        self,
        boxes: torch.Tensor,
        labels: torch.Tensor,
        scores: torch.Tensor,
        same_category_nms_threshold: float = DEFAULT_NMS_THRESHOLD,
        cross_category_nms_threshold: float = DEFAULT_CROSS_CATEGORY_NMS_THRESHOLD,
        enable_same_category_nms: bool = DEFAULT_ENABLE_SAME_CATEGORY_NMS,
        enable_cross_category_nms: bool = DEFAULT_ENABLE_CROSS_CATEGORY_NMS,
        image_shape: Optional[Tuple[int, int]] = None,
        original_image: Optional[np.ndarray] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        統一的 NMS 過濾方法，整合三種過濾機制：
        1. 重疊框過濾（同類別內）：使用標準 NMS 過濾同一類別內重疊的框
        2. 同類別過濾：按類別分別執行 NMS，避免不同類別的框互相抑制
        3. 跨類別過濾：過濾不同類別之間重疊的框
        
        Args:
            boxes: 邊界框張量 [N, 4]
            labels: 類別標籤張量 [N]
            scores: 信心值張量 [N]
            same_category_nms_threshold: 同類別 NMS IoU 閾值（預設 0.1）
                                        - 用於過濾同一類別內重疊的框
            cross_category_nms_threshold: 跨類別 NMS IoU 閾值（預設 0.3）
                                         - 用於過濾不同類別之間重疊的框
                                         - 設為 0 或負數可禁用跨類別過濾
            enable_same_category_nms: 是否啟用同類別過濾（預設 False）
                                     - True: 按類別分別執行 NMS
                                     - False: 不執行同類別過濾（已關閉）
            enable_cross_category_nms: 是否啟用跨類別過濾（預設 True）
                                      - True: 過濾不同類別之間重疊的框
                                      - False: 不過濾跨類別重疊
        
        Returns:
            (過濾後的 boxes, labels, scores)
        """
        if len(boxes) == 0:
            return boxes, labels, scores
        
        # ========================================================================
        # 步驟 1: 同類別過濾（重疊框過濾 + 按類別分別處理）
        # ========================================================================
        if enable_same_category_nms and same_category_nms_threshold > 0:
            # 按類別分別執行 NMS，避免不同類別的框互相抑制
            unique_labels = torch.unique(labels)
            keep_mask = torch.zeros(len(boxes), dtype=torch.bool, device=boxes.device)
            
            for label in unique_labels:
                # 找出該類別的所有框
                label_mask = labels == label
                if label_mask.sum() > 0:
                    # 對該類別執行 NMS（重疊框過濾）
                    label_boxes = boxes[label_mask]
                    label_scores = scores[label_mask]
                    label_keep_indices = nms(label_boxes, label_scores, same_category_nms_threshold)
                    
                    # 將保留的框標記為 True
                    label_indices = torch.where(label_mask)[0]
                    keep_mask[label_indices[label_keep_indices]] = True
            
            # 只保留通過同類別 NMS 的框
            boxes = boxes[keep_mask]
            labels = labels[keep_mask]
            scores = scores[keep_mask]
        
        # ========================================================================
        # 步驟 2: 跨類別過濾（不同類別之間重疊框過濾）
        # ========================================================================
        if enable_cross_category_nms and cross_category_nms_threshold > 0 and len(boxes) > 0:
            # 查找類別 ID（用於優先級排序）
            rfid_label_id = None
            colony_label_id = None
            point_label_id = None
            for label_id, category_name in self.category_names.items():
                if category_name == 'RFID':
                    rfid_label_id = label_id
                elif category_name == 'colony':
                    colony_label_id = label_id
                elif category_name == 'point':
                    point_label_id = label_id
            
            # NMS 優先級設定：RFID > colony > point
            # 優先級順序：RFID（最高）> colony（中等）> point（最低）
            # 優先級數值：RFID=0, colony=1, point=2（數字越小優先級越高）
            # 高優先級類別可以抑制低優先級類別的重疊框
            # 在相同優先級內，按信心值降序排序
            category_priorities = torch.full((len(labels),), 999, dtype=torch.long, device=labels.device)
            if rfid_label_id is not None:
                category_priorities[labels == rfid_label_id] = 0  # RFID：最高優先級
            if colony_label_id is not None:
                category_priorities[labels == colony_label_id] = 1  # colony：中等優先級
            if point_label_id is not None:
                category_priorities[labels == point_label_id] = 2  # point：最低優先級
            
            # 排序：先按類別優先級（升序），再按信心值（降序）
            # 使用優先級*1000 - 信心值來實現：優先級優先，相同優先級內信心值高的在前
            sort_keys = category_priorities.float() * 1000 - scores
            sorted_indices = torch.argsort(sort_keys)
            boxes_sorted = boxes[sorted_indices]
            labels_sorted = labels[sorted_indices]
            scores_sorted = scores[sorted_indices]
            
            keep = []
            suppressed = torch.zeros(len(boxes), dtype=torch.bool, device=boxes.device)
            
            # 特殊處理：先處理 RFID 與 colony 的重疊情況
            # 如果 RFID 框與 colony 框重疊，且 RFID 信心度低於 0.7，則優先保留 colony
            if rfid_label_id is not None and colony_label_id is not None:
                rfid_mask = labels_sorted == rfid_label_id
                colony_mask = labels_sorted == colony_label_id
                
                rfid_indices = torch.where(rfid_mask)[0]
                colony_indices = torch.where(colony_mask)[0]
                
                if len(rfid_indices) > 0 and len(colony_indices) > 0:
                    rfid_boxes = boxes_sorted[rfid_indices]
                    colony_boxes = boxes_sorted[colony_indices]
                    rfid_scores = scores_sorted[rfid_indices]
                    
                    # 計算所有 RFID 與 colony 之間的 IoU
                    rfid_colony_ious = box_iou(rfid_boxes, colony_boxes)  # [num_rfid, num_colony]
                    
                    # 找出重疊的 RFID-colony 對（IoU > threshold）
                    high_iou_mask = rfid_colony_ious > cross_category_nms_threshold
                    
                    # 對於每個 RFID，檢查是否應該被抑制
                    for rfid_idx_in_sorted, rfid_idx in enumerate(rfid_indices):
                        rfid_score = scores_sorted[rfid_idx].item()
                        
                        # 檢查該 RFID 是否與任何 colony 重疊
                        overlapping_colonies = high_iou_mask[rfid_idx_in_sorted]
                        
                        if overlapping_colonies.any():
                            # 如果 RFID 信心度低於 0.7，則抑制該 RFID，優先保留重疊的 colony
                            if rfid_score < 0.86:
                                suppressed[rfid_idx] = True
                                logger.info(f"抑制誤判的 RFID（信心度: {rfid_score:.4f}），優先保留重疊的 colony")
                                # 不抑制與其重疊的 colony（讓 colony 保留）
                                continue
                            # 否則，正常的 RFID 會抑制重疊的 colony（在後續邏輯中處理）
            
            # 標準跨類別 NMS 處理
            # 追蹤 point 被過濾的情況
            point_suppressed_by_rfid = 0
            point_suppressed_by_colony = 0
            
            for i in range(len(boxes_sorted)):
                if suppressed[i]:
                    continue
                
                # 獲取當前框的類別（需要在檢查之前獲取）
                current_label = labels_sorted[i].item()
                current_box = boxes_sorted[i:i+1]  # [1, 4]
                
                # 找出與當前框重疊的框
                # 計算 IoU
                ious = box_iou(current_box, boxes_sorted)  # [1, N]
                ious = ious.squeeze(0)  # [N]
                
                # 找出重疊且類別不同的框
                different_label_mask = labels_sorted != current_label
                same_label_mask = labels_sorted == current_label  # 增加這行
                high_iou_mask = ious > cross_category_nms_threshold
                
                # 處理 point 和 colony 重疊時的情況（使用顏色特徵判斷實心/空心）
                should_suppress_current = False
                if point_label_id is not None and colony_label_id is not None and original_image is not None:
                    if current_label == point_label_id:
                        # 如果當前是 point，檢查與 colony 的重疊情況
                        colony_mask = labels_sorted == colony_label_id
                        overlapping_colonies_mask = colony_mask & high_iou_mask & ~suppressed
                        
                        if overlapping_colonies_mask.any():
                            # 找出重疊的 colony 索引
                            overlapping_colony_indices = torch.where(overlapping_colonies_mask)[0]
                            current_point_box = boxes_sorted[i]
                            
                            # 分析 point 的顏色特徵
                            point_features = self._analyze_box_color_features(current_point_box, original_image)
                            point_solidity = point_features['solidity_ratio']
                            
                            # 檢查每個重疊的 colony
                            for colony_idx in overlapping_colony_indices:
                                colony_idx_int = colony_idx.item()
                                colony_box = boxes_sorted[colony_idx_int]
                                
                                # 分析 colony 的顏色特徵
                                colony_features = self._analyze_box_color_features(colony_box, original_image)
                                colony_solidity = colony_features['solidity_ratio']
                                
                                # 顏色特徵判斷：
                                # - 實心（solidity_ratio > 1.0）：中心比邊緣暗，是 colony
                                # - 空心（solidity_ratio < 1.0）：中心比邊緣亮，是 point
                                if point_solidity > 1.0 and colony_solidity < 1.0:
                                    # point 是實心，colony 是空心 → point 應該是 colony，抑制 colony
                                    suppressed[colony_idx_int] = True
                                    if colony_idx_int in keep:
                                        keep.remove(colony_idx_int)
                                    logger.debug(f"顏色判斷：point 實心度 {point_solidity:.3f} > colony 實心度 {colony_solidity:.3f}，point 是實心 colony，抑制誤判的 colony")
                                elif point_solidity < 1.0 and colony_solidity > 1.0:
                                    # point 是空心，colony 是實心 → 抑制 point，保留 colony
                                    should_suppress_current = True
                                    logger.debug(f"顏色判斷：point 實心度 {point_solidity:.3f} < colony 實心度 {colony_solidity:.3f}，point 是空心，抑制 point，保留 colony")
                                    break
                                else:
                                    # 兩者都是實心或都是空心，使用預設邏輯（colony 優先級更高）
                                    should_suppress_current = True
                                    logger.debug(f"顏色判斷：point 實心度 {point_solidity:.3f}，colony 實心度 {colony_solidity:.3f}，使用預設邏輯（colony 優先級更高）")
                                    break
                    elif current_label == colony_label_id:
                        # 如果當前是 colony，檢查與 point 的重疊情況（使用顏色特徵判斷）
                        point_mask = labels_sorted == point_label_id
                        overlapping_points_mask = point_mask & high_iou_mask & ~suppressed
                        
                        if overlapping_points_mask.any():
                            # 找出重疊的 point 索引
                            overlapping_point_indices = torch.where(overlapping_points_mask)[0]
                            current_colony_box = boxes_sorted[i]
                            
                            # 分析 colony 的顏色特徵
                            colony_features = self._analyze_box_color_features(current_colony_box, original_image)
                            colony_solidity = colony_features['solidity_ratio']
                            
                            # 檢查每個重疊的 point
                            for point_idx in overlapping_point_indices:
                                point_idx_int = point_idx.item()
                                point_box = boxes_sorted[point_idx_int]
                                
                                # 分析 point 的顏色特徵
                                point_features = self._analyze_box_color_features(point_box, original_image)
                                point_solidity = point_features['solidity_ratio']
                                
                                # 顏色特徵判斷
                                if point_solidity > 1.0 and colony_solidity < 1.0:
                                    # point 是實心，colony 是空心 → point 應該是 colony，抑制當前 colony
                                    should_suppress_current = True
                                    logger.debug(f"顏色判斷：point 實心度 {point_solidity:.3f} > colony 實心度 {colony_solidity:.3f}，point 是實心 colony，抑制誤判的 colony")
                                    break
                                elif point_solidity < 1.0 and colony_solidity > 1.0:
                                    # point 是空心，colony 是實心 → 保留 colony（point 會被後續處理抑制）
                                    logger.debug(f"顏色判斷：point 實心度 {point_solidity:.3f} < colony 實心度 {colony_solidity:.3f}，point 是空心，保留 colony")
                                    break
                                else:
                                    # 兩者都是實心或都是空心，使用預設邏輯（colony 優先級更高）
                                    logger.debug(f"顏色判斷：point 實心度 {point_solidity:.3f}，colony 實心度 {colony_solidity:.3f}，使用預設邏輯（colony 優先級更高）")
                                    break
                
                # 如果應該抑制當前框，跳過後續處理
                if should_suppress_current:
                    suppressed[i] = True
                    continue
                
                # 保留當前框
                keep.append(i)
                
                # 核心邏輯調整：
                if current_label == point_label_id:
                    # 如果當前是 point，只抑制其他 point（IoU > 0.9 才抑制，避免過度過濾）
                    suppress_mask = same_label_mask & (ious > 0.9) & ~suppressed 
                elif current_label == colony_label_id:
                    # 如果當前是 colony，處理同類別 colony 和跨類別（但不包括 point，因為 point 已在前面處理過）
                    # 排除 point，因為 colony 和 point 的重疊已在第 927-971 行處理
                    point_mask = labels_sorted == point_label_id if point_label_id is not None else torch.zeros_like(labels_sorted, dtype=torch.bool)
                    suppress_mask = ((different_label_mask & ~point_mask) | same_label_mask) & high_iou_mask & ~suppressed
                else:
                    # 如果是 RFID 或其他類別，處理同類別與跨類別，但確保 point 不會被抑制
                    suppress_mask = (different_label_mask | same_label_mask) & high_iou_mask & ~suppressed
                    # 確保 point 不會被 RFID 或其他類別抑制
                    if point_label_id is not None:
                        point_mask = labels_sorted == point_label_id
                        suppress_mask = suppress_mask & ~point_mask
                
                # 追蹤 point 被過濾的情況（用於日誌記錄）
                if point_label_id is not None:
                    point_mask = labels_sorted == point_label_id
                    point_suppressed = suppress_mask & point_mask
                    if point_suppressed.any():
                        if current_label == rfid_label_id:
                            point_suppressed_by_rfid += point_suppressed.sum().item()
                        elif current_label == colony_label_id:
                            point_suppressed_by_colony += point_suppressed.sum().item()
                
                suppressed[suppress_mask] = True
            
            # 記錄 point 被過濾的情況（point 不會被非 point 類別抑制，此日誌主要用於調試）
            if point_label_id is not None and (point_suppressed_by_rfid > 0 or point_suppressed_by_colony > 0):
                logger.debug(f"ℹ️  point 抑制統計（僅同類別 point 之間）：原本會被 RFID 抑制 {point_suppressed_by_rfid} 個，被 colony 抑制 {point_suppressed_by_colony} 個（實際已保護 point 不被非 point 類別抑制）")
            
            # 使用保留的索引來獲取原始框
            if len(keep) > 0:
                keep_tensor = torch.tensor(keep, dtype=torch.long, device=boxes.device)
                boxes = boxes_sorted[keep_tensor]
                labels = labels_sorted[keep_tensor]
                scores = scores_sorted[keep_tensor]
            else:
                # 如果跨類別 NMS 過濾掉所有框，保留空結果
                boxes = boxes[[]]
                labels = labels[[]]
                scores = scores[[]]
        
        return boxes, labels, scores
    
    def detect(
        self,
        image: torch.Tensor,
        threshold: Optional[float] = None,
        category_thresholds: Optional[Dict[str, float]] = None,
        same_category_nms_threshold: float = DEFAULT_NMS_THRESHOLD,
        cross_category_nms_threshold: Optional[float] = None,
        enable_same_category_nms: bool = DEFAULT_ENABLE_SAME_CATEGORY_NMS,
        enable_cross_category_nms: bool = DEFAULT_ENABLE_CROSS_CATEGORY_NMS,
        original_image: Optional[np.ndarray] = None
    ) -> Dict:
        """
        對單張圖像進行偵測
        
        Args:
            image: 圖像張量 [C, H, W]，值範圍 [0, 1]
            threshold: 全域置信度閾值（可選，預設使用 CATEGORY_THRESHOLDS）
                       - 如果提供，所有類別使用相同閾值
                       - 如果為 None，使用 CATEGORY_THRESHOLDS 中的類別特定閾值
            category_thresholds: 類別特定閾值字典（可選）
                                 - 格式：{'RFID': 0.5, 'colony': 0.3, 'point': 0.3}
                                 - 如果提供，會覆蓋預設的 CATEGORY_THRESHOLDS
            nms_threshold: 同類別 NMS IoU 閾值（預設 0.1）
                          - 用於過濾同一類別內重疊的框
            cross_category_nms_threshold: 跨類別 NMS IoU 閾值（可選，預設 0.3）
                                         - 用於過濾不同類別之間重疊的框
                                         - 如果為 None，使用預設值 DEFAULT_CROSS_CATEGORY_NMS_THRESHOLD
                                         - 如果設為 0，則不執行跨類別過濾
            
        Returns:
            偵測結果字典，包含 'boxes', 'labels', 'scores'
        """
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
        
        # 確定跨類別 NMS 閾值
        if cross_category_nms_threshold is None:
            cross_category_nms_threshold = DEFAULT_CROSS_CATEGORY_NMS_THRESHOLD
        
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
        
        # 調試：記錄閾值過濾後的結果
        logger.info("閾值過濾後統計:")
        unique_labels_filtered = torch.unique(labels)
        point_count_before_nms = 0
        point_label_id = None
        for label_id, category_name in self.category_names.items():
            if category_name == 'point':
                point_label_id = label_id
                break
        
        for label_id in unique_labels_filtered:
            label_mask = labels == label_id
            if label_mask.sum() > 0:
                category_name = self.category_names.get(int(label_id), f"class_{label_id}")
                logger.info(f"  {category_name}: {label_mask.sum()} 個偵測")
                if label_id == point_label_id:
                    point_count_before_nms = label_mask.sum().item()
        
        # 獲取圖像尺寸（用於形狀檢查）
        image_shape = None
        if len(boxes) > 0:
            # 從邊界框推斷圖像尺寸（取最大值）
            max_x = boxes[:, 2].max().item()
            max_y = boxes[:, 3].max().item()
            # 使用一個合理的估計（實際圖像尺寸可能更大，但用於形狀檢查足夠）
            image_shape = (int(max_y * 1.1), int(max_x * 1.1))
        
        # 記錄 NMS 過濾前的 colony 數量（在閾值過濾後）
        colony_label_id = None
        colony_count_before_nms = 0
        for label_id, category_name in self.category_names.items():
            if category_name == 'colony':
                colony_label_id = label_id
                break
        if colony_label_id is not None:
            colony_mask_before = labels == colony_label_id
            colony_count_before_nms = colony_mask_before.sum().item()
        
        # 應用統一的 NMS 過濾（整合：重疊框過濾、同類別過濾、跨類別過濾）
        boxes, labels, scores = self._apply_nms_filtering(
            boxes=boxes,
            labels=labels,
            scores=scores,
            same_category_nms_threshold=same_category_nms_threshold,
            cross_category_nms_threshold=cross_category_nms_threshold,
            enable_same_category_nms=enable_same_category_nms,
            enable_cross_category_nms=enable_cross_category_nms,
            image_shape=image_shape,
            original_image=original_image  # 傳入原始圖像用於顏色特徵判斷
        )
        
        # 調試：記錄 NMS 過濾後的結果
        logger.info("NMS 過濾後統計:")
        unique_labels_nms = torch.unique(labels)
        colony_count_after_nms = 0
        point_count_after_nms = 0
        
        for label_id in unique_labels_nms:
            label_mask = labels == label_id
            if label_mask.sum() > 0:
                category_name = self.category_names.get(int(label_id), f"class_{label_id}")
                logger.info(f"  {category_name}: {label_mask.sum()} 個偵測")
                if label_id == colony_label_id:
                    colony_count_after_nms = label_mask.sum().item()
                elif label_id == point_label_id:
                    point_count_after_nms = label_mask.sum().item()
        
        # 如果 colony 被 NMS 過濾掉了，記錄警告
        if colony_label_id is not None and colony_count_before_nms > colony_count_after_nms:
            filtered_count = colony_count_before_nms - colony_count_after_nms
            logger.warning(f"⚠️  NMS 過濾掉了 {filtered_count} 個 colony（過濾前: {colony_count_before_nms}, 過濾後: {colony_count_after_nms}）")
            logger.warning(f"   建議：降低 NMS 閾值或關閉跨類別 NMS 來保留更多 colony")
        
        # 如果 point 被 NMS 過濾掉了，記錄警告
        if point_label_id is not None and point_count_before_nms > point_count_after_nms:
            filtered_count = point_count_before_nms - point_count_after_nms
            logger.warning(f"⚠️  NMS 過濾掉了 {filtered_count} 個 point（過濾前: {point_count_before_nms}, 過濾後: {point_count_after_nms}）")
            logger.warning(f"   可能原因：被 RFID 或 colony 抑制（point 優先級最低）")
        elif point_label_id is not None and point_count_before_nms == 0:
            logger.warning(f"⚠️  沒有偵測到任何 point（可能被閾值過濾掉，閾值: {CATEGORY_THRESHOLDS.get('point', 'N/A')}）")
        
        # RFID 特殊處理：每張圖只保留一個 RFID，且保留信心度最高的
        rfid_label_id = None
        for label_id, category_name in self.category_names.items():
            if category_name == 'RFID':
                rfid_label_id = label_id
                break
        if rfid_label_id is not None:
            rfid_mask = labels == rfid_label_id
            rfid_count = rfid_mask.sum().item()
            
            if rfid_count > 1:
                # 有多個 RFID，只保留信心度最高的
                logger.info(f"發現 {rfid_count} 個 RFID 偵測，只保留信心度最高的")
                
                # 找出所有 RFID 的索引
                rfid_indices = torch.where(rfid_mask)[0]
                
                # 找出信心度最高的 RFID 索引
                rfid_scores = scores[rfid_mask]
                best_rfid_idx_in_mask = torch.argmax(rfid_scores)
                best_rfid_idx = rfid_indices[best_rfid_idx_in_mask]
                best_rfid_score = rfid_scores[best_rfid_idx_in_mask].item()
                
                # 建立保留遮罩：保留所有非 RFID 的偵測 + 信心度最高的 RFID
                keep_mask = ~rfid_mask  # 先保留所有非 RFID
                keep_mask[best_rfid_idx] = True  # 再保留最好的 RFID
                
                # 過濾結果
                boxes = boxes[keep_mask]
                labels = labels[keep_mask]
                scores = scores[keep_mask]
                
                logger.info(f"已過濾：保留 1 個 RFID（信心度: {best_rfid_score:.4f}）")
            elif rfid_count == 1:
                # 只有一個 RFID，記錄信心度
                rfid_score = scores[rfid_mask].item()
                logger.debug(f"發現 1 個 RFID 偵測（信心度: {rfid_score:.4f}）")
        
        # Point 特殊處理：每張圖最多只保留一個 point，且保留信心度大於 0.5 且信心度最高的
        # 同時檢查與 colony 的重疊，避免誤判的 point 覆蓋正確的 colony
        point_label_id = None
        colony_label_id = None
        for label_id, category_name in self.category_names.items():
            if category_name == 'point':
                point_label_id = label_id
            elif category_name == 'colony':
                colony_label_id = label_id
        
        if point_label_id is not None:
            point_mask = labels == point_label_id
            point_count = point_mask.sum().item()
            
            # 輔助函數：檢查 point 是否應該被抑制（與 colony 重疊且判斷為誤判）
            def should_suppress_point(point_idx: int, point_box: torch.Tensor, point_score: float) -> bool:
                """檢查 point 是否應該被抑制（與 colony 重疊且判斷為誤判）"""
                if colony_label_id is None or original_image is None:
                    return False
                
                colony_mask = labels == colony_label_id
                if not colony_mask.any():
                    return False
                
                # 找出所有 colony
                colony_boxes = boxes[colony_mask]
                colony_scores = scores[colony_mask]
                
                # 重疊判斷：僅使用 IoU 值
                # 計算 point 與所有 colony 的 IoU
                point_box_expanded = point_box.unsqueeze(0)  # [1, 4]
                ious = box_iou(point_box_expanded, colony_boxes).squeeze(0)  # [num_colonies]
                
                # 找出重疊的 colony（IoU > 0.0005）
                overlapping_mask = ious > 0.0005
                
                if not overlapping_mask.any():
                    return False
                
                overlapping_colony_indices = torch.where(overlapping_mask)[0]
                overlapping_colony_scores = colony_scores[overlapping_mask]
                
                # 分析 point 的顏色特徵
                point_features = self._analyze_box_color_features(point_box, original_image)
                point_solidity = point_features['solidity_ratio']
                        
                # 檢查每個重疊的 colony
                # 記錄所有重疊的 colony 資訊，用於最後的信心度比較
                overlapping_colonies_info = []
                
                for colony_idx_in_mask in overlapping_colony_indices:
                    colony_idx = torch.where(colony_mask)[0][colony_idx_in_mask]
                    colony_box = boxes[colony_idx]
                    colony_score = colony_scores[colony_idx_in_mask].item()
                    
                    # 分析 colony 的顏色特徵
                    colony_features = self._analyze_box_color_features(colony_box, original_image)
                    colony_solidity = colony_features['solidity_ratio']
                    
                    # 記錄重疊的 colony 資訊
                    overlapping_colonies_info.append({
                        'idx': colony_idx,
                        'score': colony_score,
                        'solidity': colony_solidity
                    })
                    
                    # 判斷邏輯（調整閾值以保護 point）：
                    # 1. 如果 point 是明顯實心（solidity_ratio > 1.15），應該是 colony，抑制 point
                    #    提高閾值從 1.0 到 1.15，避免誤判輕微實心的 point
                    if point_solidity > 1.15:
                        logger.info(f"抑制誤判的 point：point 實心度 {point_solidity:.3f} > 1.15（應為 colony），與 colony（信心度: {colony_score:.4f}，實心度: {colony_solidity:.3f}）重疊")
                        return True
                    
                    # 2. 如果 colony 是明顯實心（> 1.15）但 point 是明顯空心（< 0.85），且 point 信心度未明顯高於 colony，抑制 point
                    #    這表示 colony 更可能是正確的，point 可能是誤判
                    #    調整：colony 實心度閾值提高到 1.15，point 空心度閾值降低到 0.85，讓判斷更嚴格
                    if colony_solidity > 1.15 and point_solidity < 0.85 and point_score < colony_score * 1.5:
                        logger.info(f"抑制 point：colony 是明顯實心（{colony_solidity:.3f} > 1.15）但 point 是明顯空心（{point_solidity:.3f} < 0.85），且 point 信心度 {point_score:.4f} 未明顯高於 colony 信心度 {colony_score:.4f}（需要 > {colony_score * 1.5:.4f}）")
                        return True
                    
                    # 3. 如果兩者實心度非常接近（差異 < 0.15），且 point 信心度未明顯高於 colony，抑制 point
                    #    這表示兩者特徵非常相似，但 colony 優先級更高
                    #    降低閾值從 0.3 到 0.15，讓判斷更嚴格，避免誤判
                    if abs(point_solidity - colony_solidity) < 0.15 and point_score < colony_score * 1.5:
                        logger.info(f"抑制 point：兩者實心度非常接近（point: {point_solidity:.3f}, colony: {colony_solidity:.3f}，差異 < 0.15），且 point 信心度 {point_score:.4f} 未明顯高於 colony 信心度 {colony_score:.4f}（需要 > {colony_score * 1.5:.4f}）")
                        return True
                
                # 4. 如果經過前面的判斷後，仍然有重疊度高的 colony，比較信心度，保留信心度較高者
                #    這是最後的判斷邏輯，用於處理前面無法明確判斷的情況
                if len(overlapping_colonies_info) > 0:
                    # 找出信心度最高的 colony
                    best_colony = max(overlapping_colonies_info, key=lambda x: x['score'])
                    best_colony_score = best_colony['score']
                    
                    # 比較 point 和 colony 的信心度
                    if point_score > best_colony_score:
                        # point 信心度較高，保留 point
                        logger.info(f"保留 point：point 信心度 {point_score:.4f} > colony 信心度 {best_colony_score:.4f}，保留 point")
                        return False
                    else:
                        # colony 信心度較高或相等，抑制 point（colony 優先級更高）
                        logger.info(f"抑制 point：colony 信心度 {best_colony_score:.4f} >= point 信心度 {point_score:.4f}，保留 colony")
                        return True
                
                return False
            
            if point_count > 1:
                # 有多個 point，只保留信心度大於 0.5 且信心度最高的
                logger.info(f"發現 {point_count} 個 point 偵測，只保留信心度大於 0.5 且信心度最高的")
                
                # 找出所有 point 的索引
                point_indices = torch.where(point_mask)[0]
                point_scores = scores[point_mask]
                
                # 過濾出信心度大於 0.5 的 point
                high_confidence_mask = point_scores > 0.5
                high_confidence_indices = point_indices[high_confidence_mask]
                high_confidence_scores = point_scores[high_confidence_mask]
                
                if len(high_confidence_indices) > 0:
                    # 找出信心度最高的 point 索引
                    best_point_idx_in_filtered = torch.argmax(high_confidence_scores)
                    best_point_idx = high_confidence_indices[best_point_idx_in_filtered]
                    best_point_score = high_confidence_scores[best_point_idx_in_filtered].item()
                    best_point_box = boxes[best_point_idx]
                    
                    # 檢查是否應該抑制這個 point（與 colony 重疊且判斷為誤判）
                    if should_suppress_point(best_point_idx, best_point_box, best_point_score):
                        # 抑制 point，移除所有 point
                        logger.warning(f"抑制誤判的 point（信心度: {best_point_score:.4f}），保留重疊的 colony")
                        keep_mask = ~point_mask  # 移除所有 point
                    else:
                        # 建立保留遮罩：保留所有非 point 的偵測 + 信心度最高的 point
                        keep_mask = ~point_mask  # 先保留所有非 point
                        keep_mask[best_point_idx] = True  # 再保留最好的 point
                        logger.info(f"已過濾：保留 1 個 point（信心度: {best_point_score:.4f}，從 {len(high_confidence_indices)} 個信心度 > 0.5 的 point 中選擇）")
                    
                    # 過濾結果
                    boxes = boxes[keep_mask]
                    labels = labels[keep_mask]
                    scores = scores[keep_mask]
                else:
                    # 沒有信心度大於 0.5 的 point，移除所有 point
                    logger.warning(f"發現 {point_count} 個 point，但沒有信心度 > 0.5 的 point，移除所有 point")
                    keep_mask = ~point_mask  # 移除所有 point
                    
                    # 過濾結果
                    boxes = boxes[keep_mask]
                    labels = labels[keep_mask]
                    scores = scores[keep_mask]
            elif point_count == 1:
                # 只有一個 point，檢查信心度是否大於 0.5
                point_idx = torch.where(point_mask)[0][0]
                point_score = scores[point_mask].item()
                point_box = boxes[point_idx]
                
                if point_score <= 0.5:
                    # 信心度不足，移除這個 point
                    logger.warning(f"發現 1 個 point 偵測，但信心度 ({point_score:.4f}) <= 0.5，移除該 point")
                    keep_mask = ~point_mask  # 移除這個 point
                    
                    # 過濾結果
                    boxes = boxes[keep_mask]
                    labels = labels[keep_mask]
                    scores = scores[keep_mask]
                else:
                    # 檢查是否應該抑制這個 point（與 colony 重疊且判斷為誤判）
                    if should_suppress_point(point_idx, point_box, point_score):
                        # 抑制 point
                        logger.warning(f"抑制誤判的 point（信心度: {point_score:.4f}），保留重疊的 colony")
                        keep_mask = ~point_mask  # 移除這個 point
                        
                        # 過濾結果
                        boxes = boxes[keep_mask]
                        labels = labels[keep_mask]
                        scores = scores[keep_mask]
                    else:
                        logger.debug(f"發現 1 個 point 偵測（信心度: {point_score:.4f}）")
        
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
        category_thresholds: Optional[Dict[str, float]] = None,
        same_category_nms_threshold: float = DEFAULT_NMS_THRESHOLD,
        cross_category_nms_threshold: Optional[float] = None,
        enable_same_category_nms: bool = DEFAULT_ENABLE_SAME_CATEGORY_NMS,
        enable_cross_category_nms: bool = DEFAULT_ENABLE_CROSS_CATEGORY_NMS
    ) -> Tuple[np.ndarray, Dict]:
        """
        Args:
            image_path: 圖像檔案路徑
            threshold: 全域置信度閾值（可選）
            category_thresholds: 類別特定閾值字典（可選）
            same_category_nms_threshold: 同類別 NMS 閾值
            cross_category_nms_threshold: 跨類別 NMS 閾值
            enable_same_category_nms: 是否啟用同類別過濾
            enable_cross_category_nms: 是否啟用跨類別過濾
        Returns:
            (原始圖像陣列, 偵測結果字典) 元組
            - 偵測結果的座標已在原始圖像座標系（因為預處理保持影像尺寸不變）
        """
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
        
        # 進行偵測（傳入原始圖像用於顏色特徵判斷）
        detections = self.detect(
            image_tensor,
            threshold=threshold,
            category_thresholds=category_thresholds,
            same_category_nms_threshold=same_category_nms_threshold,
            cross_category_nms_threshold=cross_category_nms_threshold,
            enable_same_category_nms=enable_same_category_nms,
            enable_cross_category_nms=enable_cross_category_nms,
            original_image=original_image  # 傳入原始圖像用於顏色特徵判斷
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
        """
        在圖像上繪製偵測結果
        Args:
            image: 原始圖像陣列
            detections: 偵測結果字典
            show_scores: 是否顯示置信度分數
            line_width: 邊界框線寬  
        Returns:
            標記後的圖像陣列
        """
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
        same_category_nms_threshold: float = DEFAULT_NMS_THRESHOLD,
        cross_category_nms_threshold: Optional[float] = None,
        enable_same_category_nms: bool = DEFAULT_ENABLE_SAME_CATEGORY_NMS,
        enable_cross_category_nms: bool = DEFAULT_ENABLE_CROSS_CATEGORY_NMS,
        save_images: bool = True
    ) -> Dict:
        """
        批次處理目錄中的所有圖片
        Args:
            input_dir: 輸入圖片目錄
            output_dir: 輸出目錄
            threshold: 全域置信度閾值（可選，預設使用 CATEGORY_THRESHOLDS）
            category_thresholds: 類別特定閾值字典（可選）
                                 - 格式：{'RFID': 0.5, 'colony': 0.3, 'point': 0.3}
            nms_threshold: 同類別 NMS IoU 閾值（預設 0.1）
            cross_category_nms_threshold: 跨類別 NMS IoU 閾值（可選，預設 0.3）
            save_images: 是否保存標記後的圖片
        """
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
                    category_thresholds=category_thresholds,
                    same_category_nms_threshold=same_category_nms_threshold,
                    cross_category_nms_threshold=cross_category_nms_threshold,
                    enable_same_category_nms=enable_same_category_nms,
                    enable_cross_category_nms=enable_cross_category_nms
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
            "same_category_nms_threshold": same_category_nms_threshold,
            "cross_category_nms_threshold": cross_category_nms_threshold,
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
    """主函數"""
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
            category_thresholds=None,    # None = 使用預設的 CATEGORY_THRESHOLDS
            same_category_nms_threshold=0.1,      # 同類別 NMS 閾值（提高以保留更多 colony，避免過度過濾）(NMS參數越高，越寬鬆)
            cross_category_nms_threshold=0.3,     # 跨類別 NMS 閾值（提高以保留更多 colony，避免與 RFID/point 重疊時被過濾）
            enable_same_category_nms=False,       # 同類別過濾已關閉（使用預設值）
            enable_cross_category_nms=True        # 啟用跨類別過濾
        )
        
        logger.info(f"\n✓ 處理完成！結果已保存到 final/output/")
        
    except FileNotFoundError as e:
        logger.error(f"檔案或目錄不存在: {e}")
    except Exception as e:
        logger.error(f"處理過程發生錯誤: {e}", exc_info=True)

if __name__ == "__main__":
    main()