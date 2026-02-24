

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
from torchvision.ops import box_iou, nms

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
    CATEGORY_THRESHOLDS = {
        'RFID': 0.5,
        'colony': 0.365,
        'point': 0.22
    }
    
    # 預處理配置（與 model.py 保持一致）
    PREPROCESS_DENOISE_STRENGTH = 10  # 降噪強度
    PREPROCESS_SHARPEN_RADIUS = 5    # 銳化半徑
    PREPROCESS_CONTRAST_ALPHA = 1.0   # 對比度增強係數
    
    # 同類別 NMS 配置(數值高， 越不積極)
    SAME_CLASS_NMS_ENABLED = True
    SAME_CLASS_NMS_IOU_THRESHOLD = 0.2
    
    
    # 跨類別 NMS 配置
    CROSS_CLASS_NMS_ENABLED = True
    CROSS_CLASS_NMS_IOU_THRESHOLD = 0.3
    
    # Colony 幾何判斷配置
    COLONY_GEOMETRIC_FILTER_ENABLED = False  # 暫時關閉幾何過濾
    COLONY_MIN_CIRCULARITY = 0.2        # 最小圓形度（允許不規則形狀，如左上角的不規則 colony，降低以保留更多）
    COLONY_MIN_ASPECT_RATIO = 0.3       # 最小寬高比（允許橢圓形）
    COLONY_MAX_ASPECT_RATIO = 3.0       # 最大寬高比（過濾過長的形狀）
    COLONY_MIN_COMPACTNESS = 0.2        # 最小緊湊度（降低以保留大 colony，即使檢測框較大）
    
    # 繪圖配置
    DRAW_SHOW_SCORES = True
    DRAW_LINE_WIDTH = 2
    DRAW_FONT_SIZE = 16
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
                    if cat_id is not None and cat_name is not None:
                        category_names[cat_id] = cat_name
                
            except Exception as e:
                logger.warning(f"無法從 {train_annotation_file} 載入類別資訊: {e}")
        
        # 如果沒有載入到類別資訊，使用預設名稱
        if len(category_names) == 1:  # 只有 background
            logger.warning("無法載入類別名稱，使用預設名稱 class_1, class_2, class_3")
            for i in range(1, self.num_classes):
                category_names[i] = f"class_{i}"
        
        return category_names
    
    @staticmethod
    def _analyze_colony_geometry(
        box: torch.Tensor,
        original_image: np.ndarray
    ) -> Dict[str, float]:
        """
        分析 colony 檢測框的幾何特徵
        
        返回：
        - circularity: 圓形度 (0-1, 1 表示完美圓形)
        - aspect_ratio: 寬高比
        - compactness: 緊湊度 (實際面積 / 邊界框面積)
        """
        x1, y1, x2, y2 = box.cpu().numpy().astype(int)
        h, w = original_image.shape[:2]
        
        # 確保座標在圖像範圍內
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))
        
        if x2 <= x1 or y2 <= y1:
            return {
                'circularity': 0.0,
                'aspect_ratio': 1.0,
                'compactness': 0.0,
                'valid': False
            }
        
        # 提取邊界框區域
        box_region = original_image[y1:y2, x1:x2]
        
        # 轉換為灰階（如果原本是 RGB）
        if len(box_region.shape) == 3:
            gray_region = cv2.cvtColor(box_region, cv2.COLOR_RGB2GRAY)
        else:
            gray_region = box_region
        
        # 計算寬高比
        box_width = x2 - x1
        box_height = y2 - y1
        aspect_ratio = box_width / max(box_height, 1)
        if aspect_ratio < 1.0:
            aspect_ratio = 1.0 / aspect_ratio  # 統一為 >= 1.0
        
        # 使用 Otsu 二值化來分割 colony 區域
        _, binary = cv2.threshold(gray_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 反轉二值化（colony 通常是暗色）
        binary = cv2.bitwise_not(binary)
        
        # 形態學操作：去除小噪點
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # 尋找輪廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # 如果找不到輪廓，使用邊界框的簡單特徵
            box_area = box_width * box_height
            return {
                'circularity': 0.5,  # 預設值
                'aspect_ratio': aspect_ratio,
                'compactness': 1.0,  # 假設完全填充
                'valid': True
            }
        
        # 找出最大的輪廓（通常是主要的 colony）
        largest_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest_contour)
        contour_perimeter = cv2.arcLength(largest_contour, True)
        
        # 計算圓形度：4π × 面積 / 周長²
        if contour_perimeter > 0:
            circularity = (4 * np.pi * contour_area) / (contour_perimeter ** 2)
        else:
            circularity = 0.0
        
        # 計算緊湊度：實際面積 / 邊界框面積
        box_area = box_width * box_height
        if box_area > 0:
            compactness = contour_area / box_area
        else:
            compactness = 0.0
        
        return {
            'circularity': float(circularity),
            'aspect_ratio': float(aspect_ratio),
            'compactness': float(compactness),
            'valid': True
        }
    
    @staticmethod
    def _is_point_inside_rfid(
        point_box: torch.Tensor,
        rfid_box: torch.Tensor
    ) -> bool:
        """
        檢查 point 是否在 RFID 內部
        
        參數:
            point_box: point 的邊界框 [x1, y1, x2, y2]
            rfid_box: RFID 的邊界框 [x1, y1, x2, y2]
        
        返回:
            True 如果 point 的中心點在 RFID 內部
        """
        # 計算 point 的中心點
        point_center_x = (point_box[0] + point_box[2]) / 2
        point_center_y = (point_box[1] + point_box[3]) / 2
        
        # 檢查中心點是否在 RFID 邊界框內
        return (rfid_box[0] <= point_center_x <= rfid_box[2] and
                rfid_box[1] <= point_center_y <= rfid_box[3])
    
    @staticmethod
    def _is_point_inside_colony(
        point_box: torch.Tensor,
        colony_box: torch.Tensor
    ) -> bool:
        """
        檢查 point 是否在 colony 內部
        
        參數:
            point_box: point 的邊界框 [x1, y1, x2, y2]
            colony_box: colony 的邊界框 [x1, y1, x2, y2]
        
        返回:
            True 如果 point 的中心點在 colony 內部
        """
        # 計算 point 的中心點
        point_center_x = (point_box[0] + point_box[2]) / 2
        point_center_y = (point_box[1] + point_box[3]) / 2
        
        # 檢查中心點是否在 colony 邊界框內
        return (colony_box[0] <= point_center_x <= colony_box[2] and
                colony_box[1] <= point_center_y <= colony_box[3])
    
    @staticmethod
    def _preprocess_image(
        img: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        預處理流程（與 model.py 保持一致）：
        1. 轉灰階（第一步）
        2. 銳利化
        3. 降低雜訊
        4. 對比度增強（CLAHE）
        
        注意：保持影像尺寸不變，不進行裁剪以避免標註座標偏移
        """
        original_shape = img.shape[:2]  # 保存原始尺寸 (H, W)
        
        # 1. 轉換為灰階（第一步，確保無論輸入格式如何都轉為灰階）
        if len(img.shape) == 3:
            # 如果是彩色圖像（BGR 或 RGB），轉換為灰階
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif len(img.shape) == 2:
            # 如果已經是灰階圖像，直接使用
            gray = img.copy()
        else:
            raise ValueError(f"不支援的圖像格式: shape={img.shape}")
        
        # 2. 銳利化
        # 使用 Unsharp Masking 方法：銳化圖 = 原圖 + (原圖 - 模糊圖) * 強度
        # 計算高斯模糊的核大小（必須是奇數）
        kernel_size = int(DetectionConfig.PREPROCESS_SHARPEN_RADIUS * 2 + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), DetectionConfig.PREPROCESS_SHARPEN_RADIUS)
        # Unsharp Masking：原圖 + (原圖 - 模糊圖)
        sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
        # 確保值在有效範圍內
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        # 3. 降低雜訊
        # 使用 Non-local Means Denoising（對灰階圖像）
        denoised = cv2.fastNlMeansDenoising(
            sharpened,
            h=DetectionConfig.PREPROCESS_DENOISE_STRENGTH,  # 降噪強度參數
            templateWindowSize=7,
            searchWindowSize=21
        )
        
        # 4. 對比度增強
        # 使用 CLAHE（自適應直方圖均衡化）進行對比度增強
        clahe = cv2.createCLAHE(
            clipLimit=DetectionConfig.PREPROCESS_CONTRAST_ALPHA * 2.0,  # 對比度增強係數
            tileGridSize=(8, 8)
        )
        contrast_enhanced = clahe.apply(denoised)
        
        # 確保影像尺寸不變
        assert contrast_enhanced.shape == original_shape, \
            f"影像尺寸改變: 原始={original_shape}, 處理後={contrast_enhanced.shape}"
        
        # 轉換為 RGB 格式（PIL Image 需要）
        processed_rgb = cv2.cvtColor(contrast_enhanced, cv2.COLOR_GRAY2RGB)
        
        # 保持影像尺寸不變，所以 crop_offset 為 (0, 0)
        return processed_rgb, (0, 0)
    
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
                # 對於 colony 和 RFID，顯示更多統計信息
                if category_name in ['colony', 'RFID']:
                    threshold = thresholds_dict.get(category_name, 0.3) if use_category_thresholds else global_threshold
                    above_threshold = (label_scores >= threshold).sum().item()
                    below_threshold = (label_scores < threshold).sum().item()
                    if below_threshold > 0:
                        below_scores = label_scores[label_scores < threshold]
                        logger.info(f"    高於閾值 {threshold:.3f}: {above_threshold} 個")
                        logger.info(f"    低於閾值 {threshold:.3f}: {below_threshold} 個（範圍: {below_scores.min().item():.4f} ~ {below_scores.max().item():.4f}）")
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
                    
                    # 記錄過濾結果（改為 info 級別以便調試）
                    passed_count = label_keep.sum().item()
                    total_count = label_mask.sum().item()
                    if passed_count < total_count:
                        filtered_count = total_count - passed_count
                        filtered_scores = label_scores[~label_keep]
                        if len(filtered_scores) > 0:
                            min_filtered = filtered_scores.min().item()
                            max_filtered = filtered_scores.max().item()
                        else:
                            logger.info(f"  {category_name}: {total_count} 個偵測，{passed_count} 個通過閾值 {category_threshold}")
                    elif passed_count > 0:
                        logger.info(f"  {category_name}: {passed_count} 個偵測全部通過閾值 {category_threshold}")
        else:
            # 使用全域閾值
            keep_mask = scores >= global_threshold
        
        boxes = prediction['boxes'][keep_mask]
        labels = labels[keep_mask]
        scores = scores[keep_mask]
        
        # 找出 RFID 的 category_id
        rfid_category_id = None
        for cat_id, cat_name in self.category_names.items():
            if cat_name == 'RFID':
                rfid_category_id = cat_id
                break
        
        # RFID 處理：每張圖只保留信心度最高的 RFID
        if rfid_category_id is not None:
            rfid_mask = labels == rfid_category_id
            rfid_count = rfid_mask.sum().item()
            
            if rfid_count > 1:
                # 找出所有 RFID 檢測的索引
                rfid_indices = torch.where(rfid_mask)[0]
                rfid_scores = scores[rfid_indices]
                
                # 找出信心度最高的 RFID 索引
                best_rfid_local_idx = rfid_scores.argmax()
                best_rfid_idx = rfid_indices[best_rfid_local_idx]
                best_rfid_score = rfid_scores[best_rfid_local_idx].item()
                
                # 創建新的 mask，只保留最好的 RFID 和其他所有非 RFID 檢測
                new_keep_mask = torch.zeros(len(boxes), dtype=torch.bool, device=boxes.device)
                # 保留所有非 RFID 檢測
                new_keep_mask[~rfid_mask] = True
                # 只保留最好的 RFID
                new_keep_mask[best_rfid_idx] = True
                
                # 應用新的 mask
                boxes = boxes[new_keep_mask]
                labels = labels[new_keep_mask]
                scores = scores[new_keep_mask]
                
                logger.info(f"檢測到 {rfid_count} 個 RFID，保留信心度最高的 (score: {best_rfid_score:.4f})")
        
        # RFID 內 Point 過濾：RFID 內的 point 必須信心度 > 0.4 才保留
        if rfid_category_id is not None:
            # 找出 point 的 category_id
            point_category_id = None
            for cat_id, cat_name in self.category_names.items():
                if cat_name == 'point':
                    point_category_id = cat_id
                    break
            
            if point_category_id is not None:
                rfid_mask = labels == rfid_category_id
                point_mask = labels == point_category_id
                rfid_count = rfid_mask.sum().item()
                point_count = point_mask.sum().item()
                
                if rfid_count > 0 and point_count > 0:
                    # 找出所有 RFID 和 point 的索引
                    rfid_indices = torch.where(rfid_mask)[0]
                    point_indices = torch.where(point_mask)[0]
                    rfid_boxes = boxes[rfid_indices]
                    point_boxes = boxes[point_indices]
                    point_scores = scores[point_indices]
                    
                    # 檢查每個 point 是否在 RFID 內部
                    points_to_remove = []
                    rfid_internal_point_threshold = 0.48
                    
                    for i, point_idx in enumerate(point_indices):
                        point_box = boxes[point_idx]
                        point_score = scores[point_idx].item()
                        
                        # 檢查是否在任何 RFID 內部
                        is_inside_rfid = False
                        for rfid_box in rfid_boxes:
                            if FasterRCNNDetector._is_point_inside_rfid(point_box, rfid_box):
                                is_inside_rfid = True
                                break
                        
                        # 如果在 RFID 內部且信心度 <= 0.4，標記為移除
                        if is_inside_rfid and point_score <= rfid_internal_point_threshold:
                            points_to_remove.append(point_idx)
                            logger.info(
                                f"移除 RFID 內的 Point (score: {point_score:.4f} <= {rfid_internal_point_threshold}, "
                                f"bbox: [{point_box[0]:.1f}, {point_box[1]:.1f}, {point_box[2]:.1f}, {point_box[3]:.1f}])"
                            )
                    
                    # 如果有需要移除的 point，創建新的 mask
                    if len(points_to_remove) > 0:
                        points_to_remove_set = set(points_to_remove)
                        new_keep_mask = torch.ones(len(boxes), dtype=torch.bool, device=boxes.device)
                        for point_idx in points_to_remove_set:
                            new_keep_mask[point_idx] = False
                        
                        # 應用新的 mask
                        boxes = boxes[new_keep_mask]
                        labels = labels[new_keep_mask]
                        scores = scores[new_keep_mask]
                        
                        logger.info(
                            f"RFID 內 Point 過濾完成: 移除 {len(points_to_remove)} 個信心度 <= {rfid_internal_point_threshold} 的 point"
                        )
        
        # Colony 幾何判斷和過濾
        if DetectionConfig.COLONY_GEOMETRIC_FILTER_ENABLED and original_image is not None:
            # 找出 colony 的 category_id
            colony_category_id = None
            for cat_id, cat_name in self.category_names.items():
                if cat_name == 'colony':
                    colony_category_id = cat_id
                    break
            
            if colony_category_id is not None:
                colony_mask = labels == colony_category_id
                colony_count = colony_mask.sum().item()
                
                if colony_count > 0:
                    colony_indices = torch.where(colony_mask)[0]
                    colony_boxes = boxes[colony_mask]
                    colony_scores = scores[colony_mask]
                    
                    valid_colony_mask = torch.ones(len(colony_indices), dtype=torch.bool, device=boxes.device)
                    
                    logger.info(f"開始 Colony 幾何判斷: {colony_count} 個檢測")
                    
                    for i, colony_box in enumerate(colony_boxes):
                        geometry = FasterRCNNDetector._analyze_colony_geometry(
                            colony_box, original_image
                        )
                        
                        if not geometry['valid']:
                            valid_colony_mask[i] = False
                            logger.info(
                                f"  過濾 Colony #{i+1} (無效的幾何特徵, "
                                f"score: {colony_scores[i].item():.4f})"
                            )
                            continue
                        
                        # 檢查幾何特徵是否符合 colony 的特徵
                        circularity = geometry['circularity']
                        aspect_ratio = geometry['aspect_ratio']
                        compactness = geometry['compactness']
                        
                        # 計算檢測框面積（用於判斷是否為大 colony）
                        box_width = colony_box[2] - colony_box[0]
                        box_height = colony_box[3] - colony_box[1]
                        box_area = box_width * box_height
                        
                        # 計算圖像總面積（用於判斷相對大小）
                        img_height, img_width = original_image.shape[:2]
                        img_total_area = img_height * img_width
                        area_ratio = box_area / img_total_area if img_total_area > 0 else 0
                        
                        # 對大 colony（面積比例 > 5%）使用更寬鬆的緊湊度標準
                        is_large_colony = area_ratio > 0.05
                        effective_min_compactness = DetectionConfig.COLONY_MIN_COMPACTNESS
                        if is_large_colony:
                            # 大 colony 的緊湊度標準降低一半
                            effective_min_compactness = DetectionConfig.COLONY_MIN_COMPACTNESS * 0.5
                        
                        # 判斷是否符合條件
                        is_valid = True
                        reasons = []
                        
                        if circularity < DetectionConfig.COLONY_MIN_CIRCULARITY:
                            is_valid = False
                            reasons.append(f"圓形度過低 ({circularity:.3f} < {DetectionConfig.COLONY_MIN_CIRCULARITY})")
                        
                        if aspect_ratio < DetectionConfig.COLONY_MIN_ASPECT_RATIO:
                            is_valid = False
                            reasons.append(f"寬高比過小 ({aspect_ratio:.3f} < {DetectionConfig.COLONY_MIN_ASPECT_RATIO})")
                        
                        if aspect_ratio > DetectionConfig.COLONY_MAX_ASPECT_RATIO:
                            is_valid = False
                            reasons.append(f"寬高比過大 ({aspect_ratio:.3f} > {DetectionConfig.COLONY_MAX_ASPECT_RATIO})")
                        
                        if compactness < effective_min_compactness:
                            is_valid = False
                            reasons.append(
                                f"緊湊度過低 ({compactness:.3f} < {effective_min_compactness:.3f}"
                                f"{' (大 colony 寬鬆標準)' if is_large_colony else ''})"
                            )
                        
                        if not is_valid:
                            valid_colony_mask[i] = False
                            logger.info(
                                f"  過濾 Colony #{i+1} (score: {colony_scores[i].item():.4f}, "
                                f"circularity: {circularity:.3f}, aspect_ratio: {aspect_ratio:.3f}, "
                                f"compactness: {compactness:.3f}) - {', '.join(reasons)}"
                            )
                        else:
                            logger.debug(
                                f"  保留 Colony #{i+1} (score: {colony_scores[i].item():.4f}, "
                                f"circularity: {circularity:.3f}, aspect_ratio: {aspect_ratio:.3f}, "
                                f"compactness: {compactness:.3f})"
                            )
                    
                    # 應用幾何過濾
                    if not valid_colony_mask.all():
                        valid_colony_indices = colony_indices[valid_colony_mask]
                        valid_count = len(valid_colony_indices)
                        removed_count = colony_count - valid_count
                        
                        # 創建新的 mask，只保留有效的 colony 和其他所有非 colony 檢測
                        new_keep_mask = torch.zeros(len(boxes), dtype=torch.bool, device=boxes.device)
                        # 保留所有非 colony 檢測
                        new_keep_mask[~colony_mask] = True
                        # 只保留有效的 colony
                        new_keep_mask[valid_colony_indices] = True
                        
                        # 應用新的 mask
                        boxes = boxes[new_keep_mask]
                        labels = labels[new_keep_mask]
                        scores = scores[new_keep_mask]
                        
                        logger.info(
                            f"Colony 幾何過濾完成: 原始 {colony_count} 個檢測，"
                            f"移除 {removed_count} 個不符合幾何特徵的檢測，"
                            f"保留 {valid_count} 個"
                        )
                    else:
                        logger.info(f"Colony 幾何判斷: 所有 {colony_count} 個檢測都符合幾何特徵")
        
        # Colony 內 Point 過濾：Colony 內的 point 必須信心度 > 0.4 才保留
        # 找出 colony 的 category_id
        colony_category_id = None
        for cat_id, cat_name in self.category_names.items():
            if cat_name == 'colony':
                colony_category_id = cat_id
                break
        
        if colony_category_id is not None:
            # 找出 point 的 category_id
            point_category_id = None
            for cat_id, cat_name in self.category_names.items():
                if cat_name == 'point':
                    point_category_id = cat_id
                    break
            
            if point_category_id is not None:
                colony_mask = labels == colony_category_id
                point_mask = labels == point_category_id
                colony_count = colony_mask.sum().item()
                point_count = point_mask.sum().item()
                
                if colony_count > 0 and point_count > 0:
                    # 找出所有 colony 和 point 的索引
                    colony_indices = torch.where(colony_mask)[0]
                    point_indices = torch.where(point_mask)[0]
                    colony_boxes = boxes[colony_indices]
                    point_boxes = boxes[point_indices]
                    point_scores = scores[point_indices]
                    
                    # 檢查每個 point 是否在 colony 內部
                    points_to_remove = []
                    colony_internal_point_threshold = 0.4
                    
                    for i, point_idx in enumerate(point_indices):
                        point_box = boxes[point_idx]
                        point_score = scores[point_idx].item()
                        
                        # 檢查是否在任何 colony 內部
                        is_inside_colony = False
                        for colony_box in colony_boxes:
                            if FasterRCNNDetector._is_point_inside_colony(point_box, colony_box):
                                is_inside_colony = True
                                break
                        
                        # 如果在 colony 內部且信心度 <= 0.4，標記為移除
                        if is_inside_colony and point_score <= colony_internal_point_threshold:
                            points_to_remove.append(point_idx)
                            logger.info(
                                f"移除 Colony 內的 Point (score: {point_score:.4f} <= {colony_internal_point_threshold}, "
                                f"bbox: [{point_box[0]:.1f}, {point_box[1]:.1f}, {point_box[2]:.1f}, {point_box[3]:.1f}])"
                            )
                    
                    # 如果有需要移除的 point，創建新的 mask
                    if len(points_to_remove) > 0:
                        points_to_remove_set = set(points_to_remove)
                        new_keep_mask = torch.ones(len(boxes), dtype=torch.bool, device=boxes.device)
                        for point_idx in points_to_remove_set:
                            new_keep_mask[point_idx] = False
                        
                        # 應用新的 mask
                        boxes = boxes[new_keep_mask]
                        labels = labels[new_keep_mask]
                        scores = scores[new_keep_mask]
                        
                        logger.info(
                            f"Colony 內 Point 過濾完成: 移除 {len(points_to_remove)} 個信心度 <= {colony_internal_point_threshold} 的 point"
                        )
        
        # 同類別 NMS：對每個類別分別進行 NMS 去重
        if DetectionConfig.SAME_CLASS_NMS_ENABLED:
            # 獲取所有非背景類別
            unique_labels = torch.unique(labels)
            unique_labels = unique_labels[unique_labels != 0]  # 排除背景類別
            
            if len(unique_labels) > 0:
                total_before_same_class = len(boxes)
                logger.info(f"開始同類別 NMS 處理: {total_before_same_class} 個檢測")
                
                # 收集所有要保留的索引
                all_keep_indices = []
                
                for label_id in unique_labels:
                    label_mask = labels == label_id
                    label_count = label_mask.sum().item()
                    
                    if label_count > 1:
                        # 找出該類別的所有檢測索引
                        label_indices = torch.where(label_mask)[0]
                        label_boxes = boxes[label_mask]
                        label_scores = scores[label_mask]
                        
                        category_name = self.category_names.get(int(label_id), f"class_{label_id}")
                        logger.debug(f"處理 {category_name} (id={label_id}): {label_count} 個檢測")
                        
                        # 使用 torchvision 的標準 NMS 函數進行去重
                        # 所有類別使用相同的同類別 NMS 閾值
                        nms_threshold = DetectionConfig.SAME_CLASS_NMS_IOU_THRESHOLD
                        
                        keep_indices_local = nms(
                            label_boxes,
                            label_scores,
                            nms_threshold
                        )
                        
                        # 轉換為原始 boxes 中的索引
                        keep_label_indices = label_indices[keep_indices_local]
                        removed_count = label_count - len(keep_indices_local)
                        
                        # 計算實際的重疊情況（用於調試）
                        if removed_count == 0 and label_count > 1:
                            # 如果沒有移除任何檢測，計算最大 IoU 來幫助調試
                            iou_matrix = box_iou(label_boxes, label_boxes)
                            # 排除對角線（自己與自己的 IoU = 1.0）
                            iou_matrix = iou_matrix.fill_diagonal_(0)
                            max_iou = iou_matrix.max().item()
                            logger.info(
                                f"  {category_name}: {label_count} 個檢測，"
                                f"最大 IoU = {max_iou:.3f}，閾值 = {nms_threshold:.3f}，"
                                f"未移除任何檢測（因為最大 IoU < 閾值）"
                            )
                        elif removed_count > 0:
                            logger.info(
                                f"  {category_name}: 移除 {removed_count} 個重疊檢測 "
                                f"(IoU > {nms_threshold:.3f})，保留 {len(keep_indices_local)} 個"
                            )
                        
                        # 添加到保留列表
                        all_keep_indices.append(keep_label_indices)
                    else:
                        # 只有一個檢測，直接保留
                        label_indices = torch.where(label_mask)[0]
                        all_keep_indices.append(label_indices)
                
                # 合併所有保留的索引
                if len(all_keep_indices) > 0:
                    keep_indices = torch.cat(all_keep_indices)
                    keep_indices = torch.sort(keep_indices)[0]  # 排序以保持順序
                    
                    # 應用 NMS 結果
                    boxes = boxes[keep_indices]
                    labels = labels[keep_indices]
                    scores = scores[keep_indices]
                    
                    total_after_same_class = len(boxes)
                    removed_total = total_before_same_class - total_after_same_class
                    
                    if removed_total > 0:
                        logger.info(
                            f"同類別 NMS 完成: 原始 {total_before_same_class} 個檢測，"
                            f"移除 {removed_total} 個重疊檢測，"
                            f"保留 {total_after_same_class} 個"
                        )
                    else:
                        logger.info(f"同類別 NMS: 無重疊檢測，保留所有 {total_before_same_class} 個檢測")
        
        # 跨類別 NMS：對所有檢測框進行 NMS，不管類別
        if DetectionConfig.CROSS_CLASS_NMS_ENABLED and len(boxes) > 1:
            total_before = len(boxes)
            logger.info(f"開始跨類別 NMS 處理: {total_before} 個檢測（所有類別）")
            
            # 使用 torchvision 的標準 NMS 函數進行跨類別去重
            # 對所有檢測框進行 NMS，不管它們屬於哪個類別
            keep_indices = nms(
                boxes,
                scores,
                DetectionConfig.CROSS_CLASS_NMS_IOU_THRESHOLD
            )
            
            removed_count = total_before - len(keep_indices)
            
            if removed_count > 0:
                # 應用 NMS 結果
                boxes = boxes[keep_indices]
                labels = labels[keep_indices]
                scores = scores[keep_indices]
                
                logger.info(
                    f"跨類別 NMS 完成: 原始 {total_before} 個檢測，"
                    f"移除 {removed_count} 個重疊檢測 (IoU > {DetectionConfig.CROSS_CLASS_NMS_IOU_THRESHOLD:.3f})，"
                    f"保留 {len(keep_indices)} 個"
                )
            else:
                # 計算實際的重疊情況（用於調試）
                if total_before > 1:
                    iou_matrix = box_iou(boxes, boxes)
                    # 排除對角線（自己與自己的 IoU = 1.0）
                    iou_matrix = iou_matrix.fill_diagonal_(0)
                    max_iou = iou_matrix.max().item()
                    logger.info(
                        f"跨類別 NMS: {total_before} 個檢測，"
                        f"最大 IoU = {max_iou:.3f}，閾值 = {DetectionConfig.CROSS_CLASS_NMS_IOU_THRESHOLD:.3f}，"
                        f"未移除任何檢測（因為最大 IoU < 閾值）"
                    )
                else:
                    logger.info(f"跨類別 NMS: 只有 {total_before} 個檢測，無需去重")
        
        # Point 處理：每張圖最多只保留 2 個信心度最高的 point，且信心度必須大於 0.26
        point_category_id = None
        for cat_id, cat_name in self.category_names.items():
            if cat_name == 'point':
                point_category_id = cat_id
                break
        
        if point_category_id is not None:
            point_mask = labels == point_category_id
            point_count = point_mask.sum().item()
            
            if point_count > 0:
                # 找出所有 point 檢測的索引
                point_indices = torch.where(point_mask)[0]
                point_scores = scores[point_indices]
                
                # 先過濾掉信心度 <= 0.26 的 point
                point_min_threshold = 0.26
                valid_point_mask = point_scores > point_min_threshold
                valid_point_indices = point_indices[valid_point_mask]
                valid_point_scores = point_scores[valid_point_mask]
                valid_point_count = valid_point_mask.sum().item()
                
                # 記錄被過濾掉的 point
                filtered_by_threshold = point_count - valid_point_count
                if filtered_by_threshold > 0:
                    filtered_scores = point_scores[~valid_point_mask]
                    logger.info(
                        f"Point 信心度過濾: 原始 {point_count} 個檢測，"
                        f"移除 {filtered_by_threshold} 個信心度 <= {point_min_threshold} 的檢測 "
                        f"(範圍: {filtered_scores.min().item():.4f} ~ {filtered_scores.max().item():.4f})"
                    )
                
                # 從符合信心度要求的 point 中，選擇最多 2 個信心度最高的
                if valid_point_count > 2:
                    # 使用 topk 來獲取前 2 個最高分數的索引
                    top2_point_local_indices = valid_point_scores.topk(2).indices
                    top2_point_indices = valid_point_indices[top2_point_local_indices]
                    top2_point_scores = valid_point_scores[top2_point_local_indices].tolist()
                    
                    # 創建新的 mask，只保留最好的 2 個 point 和其他所有非 point 檢測
                    new_keep_mask = torch.zeros(len(boxes), dtype=torch.bool, device=boxes.device)
                    # 保留所有非 point 檢測
                    new_keep_mask[~point_mask] = True
                    # 只保留最好的 2 個 point
                    new_keep_mask[top2_point_indices] = True
                    
                    # 應用新的 mask
                    boxes = boxes[new_keep_mask]
                    labels = labels[new_keep_mask]
                    scores = scores[new_keep_mask]
                    
                    logger.info(
                        f"Point 數量限制: {valid_point_count} 個符合信心度要求的檢測，"
                        f"保留信心度最高的 2 個 (scores: {', '.join([f'{s:.4f}' for s in top2_point_scores])})"
                    )
                elif valid_point_count > 0:
                    # 符合信心度要求的 point 數量 <= 2，全部保留
                    # 但需要過濾掉不符合信心度要求的 point
                    new_keep_mask = torch.zeros(len(boxes), dtype=torch.bool, device=boxes.device)
                    # 保留所有非 point 檢測
                    new_keep_mask[~point_mask] = True
                    # 只保留符合信心度要求的 point
                    new_keep_mask[valid_point_indices] = True
                    
                    # 應用新的 mask
                    boxes = boxes[new_keep_mask]
                    labels = labels[new_keep_mask]
                    scores = scores[new_keep_mask]
                    
                    logger.info(
                        f"Point 過濾完成: {point_count} 個原始檢測，"
                        f"{filtered_by_threshold} 個因信心度 <= {point_min_threshold} 被過濾，"
                        f"保留 {valid_point_count} 個符合要求的檢測 "
                        f"(scores: {', '.join([f'{s:.4f}' for s in valid_point_scores.tolist()])})"
                    )
                else:
                    # 所有 point 都不符合信心度要求，全部過濾掉
                    new_keep_mask = ~point_mask
                    
                    # 應用新的 mask
                    boxes = boxes[new_keep_mask]
                    labels = labels[new_keep_mask]
                    scores = scores[new_keep_mask]
                    
                    logger.info(
                        f"Point 過濾完成: {point_count} 個原始檢測，"
                        f"全部因信心度 <= {point_min_threshold} 被過濾"
                    )
        
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
        
        # 預處理圖像（CLAHE 增強）
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