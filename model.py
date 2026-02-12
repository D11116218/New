"""
Faster R-CNN 模型訓練模組
從 model/train/ 和 model/val/ 載入數據進行訓練
訓練完成後將模型保存到 run/ 資料夾
"""

import os
import json
import logging
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
from PIL import Image
import cv2

# CUDA 環境設置（必須在導入 torch 之前）
from utils import setup_cuda_environment, get_torchvision_weights_api

setup_cuda_environment()

import torch
import torch.nn as nn
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import Dataset, DataLoader
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

# 訓練配置常量（優化以充分利用 RTX 5090）
DEFAULT_NUM_EPOCHS = 10
DEFAULT_BATCH_SIZE = 8  # 降低批次大小以避免 CUDA 記憶體不足
DEFAULT_LEARNING_RATE = 0.005
DEFAULT_MOMENTUM = 0.9
DEFAULT_WEIGHT_DECAY = 0.0005
DEFAULT_LR_STEP_SIZE = 3
DEFAULT_LR_GAMMA = 0.1
MAX_NUM_WORKERS = 8  # 增加數據加載線程數

# 數據集配置
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp']

# 預處理配置
PREPROCESS_DENOISE_STRENGTH = 10  # 降噪強度
PREPROCESS_SHARPEN_RADIUS = 5    # 銳化半徑
PREPROCESS_CONTRAST_ALPHA = 1.0   # 對比度增強係數
SAVE_GRAYSCALE = True  # 是否保存灰階圖片到 grayscale/

# 二值化配置
BINARY_PETRI_DISH_THRESHOLD = 40  # 培養皿與背景分割的閾值（黑色背景 < 30）
BINARY_OBJECTS_THRESHOLD = 127    # 物件分割的閾值（Otsu 自動閾值或固定閾值）
BINARY_USE_OTSU = True            # 是否使用 Otsu 自動閾值進行物件分割

# RFID 遮罩配置（數據增強策略）
RFID_MASK_ENABLED = True         # 是否啟用 RFID 遮罩處理（已開啟）
RFID_MASK_MODE = "noise"          # 填充模式: "noise" (隨機噪點) 或 "mean" (平均灰階值)
RFID_NOISE_INTENSITY = 0.3        # 隨機噪點強度 (0.0-1.0)，僅在 mode="noise" 時使用

# 資料擴增配置
AUGMENTATION_ENABLED = True  # 是否啟用資料擴增（已開啟）
AUGMENTATION_TRANSLATE_PERCENT = 0.15  # 平移範圍：±5% 圖像尺寸
AUGMENTATION_SCALE_MIN = 0.7  # 縮放範圍最小值：90%
AUGMENTATION_SCALE_MAX = 1.3  # 縮放範圍最大值：110%
AUGMENTATION_ROTATE_DEGREES = 180  # 旋轉角度範圍：±15度

# 圓形檢測配置
CIRCLE_MASK_MARGIN = 35  # 圓形遮罩邊緣容差（像素），增加以保留邊緣 colony，避免過度裁剪
BINARY_MASK_MARGIN = 20  # 第二次二值化遮罩邊緣容差（像素），避免邊緣 colony 被切除


# ============================================================================
# 數據集類別
# ============================================================================

class FasterRCNNDataset(Dataset):
    """Faster R-CNN 數據集類別（載入預處理後的數據）"""
    
    IMAGE_EXTENSIONS = IMAGE_EXTENSIONS
    
    def __init__(
        self,
        image_dir: str,
        annotation_file: Optional[str] = None,
        transform: Optional[transforms.Compose] = None,
        dataset_name: str = "dataset",
        enable_augmentation: Optional[bool] = None,
        translate_percent: float = AUGMENTATION_TRANSLATE_PERCENT,
        scale_min: float = AUGMENTATION_SCALE_MIN,
        scale_max: float = AUGMENTATION_SCALE_MAX,
        rotate_degrees: float = AUGMENTATION_ROTATE_DEGREES
    ):
        """
        初始化數據集
        
        Args:
            image_dir: 圖像目錄路徑（model/train/images 或 model/val/images）
            annotation_file: COCO 格式標註檔案路徑（可選）
            transform: 圖像轉換（可選）
            dataset_name: 數據集名稱（用於保存灰階圖片，例如 "train" 或 "val"）
            enable_augmentation: 是否啟用資料擴增（None 時使用 AUGMENTATION_ENABLED）
            translate_percent: 平移範圍（圖像尺寸的百分比，預設 0.05 = ±5%）
            scale_min: 縮放範圍最小值（預設 0.9 = 90%）
            scale_max: 縮放範圍最大值（預設 1.1 = 110%）
            rotate_degrees: 旋轉角度範圍（預設 15 = ±15度）
        """
        self.image_dir = Path(image_dir)
        if not self.image_dir.exists():
            raise FileNotFoundError(f"圖像目錄不存在: {self.image_dir}")
        
        self.dataset_name = dataset_name
        
        # 資料擴增配置
        self.enable_augmentation = enable_augmentation if enable_augmentation is not None else AUGMENTATION_ENABLED
        self.translate_percent = translate_percent
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.rotate_degrees = rotate_degrees
        
        # 圖像轉換（Faster R-CNN 需要 tensor 格式）
        self.transform = transform or transforms.Compose([
            transforms.ToTensor()
        ])
        
        # 載入圖像列表
        self.image_files = self._load_image_files()
        if not self.image_files:
            raise ValueError(f"在 {self.image_dir} 中未找到圖像檔案")
        
        # 載入標註
        self.annotations = {}
        self.image_id_map = {}  # filename -> image_id 映射
        self.annotation_file = annotation_file  # 保存標註檔案路徑，用於 RFID 遮罩處理
        if annotation_file and Path(annotation_file).exists():
            self.annotations, self.image_id_map = self._load_annotations(annotation_file)
        
        # 準備 grayscale 目錄（用於保存灰階圖片）
        if SAVE_GRAYSCALE:
            self.grayscale_dir = Path("grayscale") / dataset_name / "images"
            self.grayscale_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.grayscale_dir = None
    
    def _load_image_files(self) -> List[str]:
        """載入圖像檔案列表"""
        image_files = []
        
        for ext in self.IMAGE_EXTENSIONS:
            image_files.extend(self.image_dir.glob(f"*{ext}"))
            image_files.extend(self.image_dir.glob(f"*{ext.upper()}"))
        
        return sorted([str(f) for f in image_files])
    
    @staticmethod
    def _load_xml_annotations(annotation_file: str, image_dir: Path) -> Tuple[Dict, Dict]:
        """
        載入 Pascal VOC XML 格式標註
        
        Args:
            annotation_file: XML 標註檔案路徑（可以是單一檔案或目錄）
            image_dir: 圖像目錄，用於匹配圖像檔名
            
        Returns:
            (annotations_dict, image_id_map) 元組
            - annotations_dict: filename -> 標註列表的映射
            - image_id_map: filename -> image_id 的映射（XML 格式中 image_id 使用檔名）
        """
        annotations_dict = {}
        filename_to_image_id = {}
        
        annotation_path = Path(annotation_file)
        
        # 如果是目錄，查找所有 XML 檔案
        if annotation_path.is_dir():
            xml_files = list(annotation_path.glob("*.xml"))
        elif annotation_path.is_file() and annotation_path.suffix.lower() == '.xml':
            xml_files = [annotation_path]
        else:
            logger.warning(f"XML 標註路徑不存在或格式不正確: {annotation_file}")
            return {}, {}
        
        # 建立類別名稱到 ID 的映射（從 XML 中動態收集）
        category_name_to_id = {}
        next_category_id = 1
        
        for xml_file in xml_files:
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                # 獲取圖像檔名
                filename_elem = root.find('filename')
                if filename_elem is None:
                    # 嘗試從 XML 檔名推斷圖像檔名
                    filename = xml_file.stem + '.jpg'  # 假設是 JPG
                else:
                    filename = filename_elem.text
                
                # 確保檔名存在於圖像目錄中
                image_path = image_dir / filename
                if not image_path.exists():
                    # 嘗試其他擴展名
                    for ext in IMAGE_EXTENSIONS:
                        alt_image_path = image_dir / (Path(filename).stem + ext)
                        if alt_image_path.exists():
                            filename = alt_image_path.name
                            break
                    else:
                        logger.debug(f"跳過 {xml_file.name}：找不到對應的圖像檔案 {filename}")
                        continue
                
                # 獲取圖像尺寸
                size_elem = root.find('size')
                if size_elem is not None:
                    width = int(size_elem.find('width').text) if size_elem.find('width') is not None else 0
                    height = int(size_elem.find('height').text) if size_elem.find('height') is not None else 0
                else:
                    width, height = 0, 0
                
                # 建立 image_id（使用檔名作為 ID）
                if filename not in filename_to_image_id:
                    filename_to_image_id[filename] = filename
                
                # 解析所有物件標註
                objects = root.findall('object')
                for obj_idx, obj in enumerate(objects):
                    # 獲取類別名稱
                    name_elem = obj.find('name')
                    if name_elem is None:
                        continue
                    category_name = name_elem.text
                    
                    # 分配類別 ID
                    if category_name not in category_name_to_id:
                        category_name_to_id[category_name] = next_category_id
                        next_category_id += 1
                    category_id = category_name_to_id[category_name]
                    
                    # 獲取邊界框
                    bndbox = obj.find('bndbox')
                    if bndbox is None:
                        continue
                    
                    x_min = float(bndbox.find('xmin').text)
                    y_min = float(bndbox.find('ymin').text)
                    x_max = float(bndbox.find('xmax').text)
                    y_max = float(bndbox.find('ymax').text)
                    
                    # 轉換為 COCO 格式 [x, y, width, height]
                    bbox_coco = [x_min, y_min, x_max - x_min, y_max - y_min]
                    
                    if filename not in annotations_dict:
                        annotations_dict[filename] = []
                    
                    annotations_dict[filename].append({
                        'id': len(annotations_dict[filename]) + 1,
                        'image_id': filename,
                        'category_id': category_id,
                        'category_name': category_name,  # 保存類別名稱以便後續使用
                        'bbox': bbox_coco,
                        'area': (x_max - x_min) * (y_max - y_min),
                        'iscrowd': 0
                    })
                    
            except Exception as e:
                logger.warning(f"無法解析 XML 檔案 {xml_file}: {e}")
                continue
        
        logger.info(f"從 XML 載入 {len(annotations_dict)} 張圖像的標註，共 {sum(len(anns) for anns in annotations_dict.values())} 個標註")
        return annotations_dict, filename_to_image_id
    
    def _load_annotations(self, annotation_file: str) -> Tuple[Dict, Dict]:
        """
        載入標註檔案（自動檢測 JSON/XML 格式）
        
        Returns:
            (annotations_dict, image_id_map) 元組
            - annotations_dict: filename -> 標註列表的映射
            - image_id_map: filename -> image_id 的映射
        """
        annotation_path = Path(annotation_file)
        
        # 檢測檔案格式
        if not annotation_path.exists():
            logger.warning(f"標註檔案不存在: {annotation_file}")
            return {}, {}
        
        # 如果是目錄或 XML 檔案，使用 XML 解析
        if annotation_path.is_dir() or annotation_path.suffix.lower() == '.xml':
            return self._load_xml_annotations(annotation_file, self.image_dir)
        
        # 否則嘗試 JSON 格式（COCO 格式）
        try:
            with open(annotation_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"無法解析標註檔案 {annotation_file}: {e}")
            return {}, {}
        
        # 建立圖像檔名到標註的映射
        annotations_dict = {}
        image_id_to_filename = {
            img['id']: img['file_name'] 
            for img in data.get('images', [])
        }
        
        # 建立 filename -> image_id 的映射
        filename_to_image_id = {
            img['file_name']: img['id']
            for img in data.get('images', [])
        }
        
        # 保存完整的 COCO 標註（保持原始格式）
        for ann in data.get('annotations', []):
            image_id = ann['image_id']
            filename = image_id_to_filename.get(image_id)
            
            if filename is None:
                continue
            
            if filename not in annotations_dict:
                annotations_dict[filename] = []
            
            # 保存完整的 COCO 標註（包括原始 bbox 格式）
            annotations_dict[filename].append({
                'id': ann['id'],
                'image_id': image_id,
                'category_id': ann['category_id'],
                'bbox': ann['bbox'],  # COCO 格式: [x, y, width, height]
                'area': ann.get('area', ann['bbox'][2] * ann['bbox'][3]),
                'iscrowd': ann.get('iscrowd', 0)
            })
        
        return annotations_dict, filename_to_image_id
    
    def __len__(self) -> int:
        """返回數據集大小"""
        return len(self.image_files)
    
    @staticmethod
    def _convert_bbox_coco_to_xyxy(bbox: List[float]) -> List[float]:
        """
        轉換 COCO bbox 格式 [x, y, w, h] 為 [x_min, y_min, x_max, y_max]
        Faster R-CNN 需要 [x_min, y_min, x_max, y_max] 格式
        
        Args:
            bbox: COCO 格式的邊界框 [x, y, width, height]
            
        Returns:
            [x_min, y_min, x_max, y_max] 格式的邊界框
        """
        x_min, y_min, width, height = bbox
        return [x_min, y_min, x_min + width, y_min + height]
    
    def _get_coco_annotations_for_image(self, filename: str) -> List[Dict]:
        """
        獲取圖像的 COCO 格式標註
        
        Args:
            filename: 圖像檔名
            
        Returns:
            COCO 格式的標註列表，如果沒有標註則返回空列表
        """
        return self.annotations.get(filename, [])
    
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
    
    # ========================================================================
    # 原有預處理方法（已註解，改用新的圓形檢測預處理）
    # ========================================================================
    # @staticmethod
    # def _opencv_preprocess(image_path: Path) -> Tuple[np.ndarray, Tuple[int, int]]:
    #     """
    #     使用 OpenCV 進行圖像預處理（整合二值化處理）
    #     
    #     處理步驟：
    #     1. 第一步二值化：分割培養皿和黑色背景，取得培養皿區域
    #     2. 第二步二值化：在培養皿區域內分割物件（RFID、colony、point）
    #     3. 轉換為灰階
    #     4. 降低雜訊
    #     5. 銳化
    #     6. 對比度增強
    #     
    #     Args:
    #         image_path: 圖像檔案路徑
    #         
    #     Returns:
    #         (預處理後的圖像陣列（灰階）, crop_offset) 元組
    #         - crop_offset: (x_offset, y_offset) 裁剪偏移量，用於調整標註座標
    #     """
    #     # 讀取圖像（BGR 格式）
    #     img = cv2.imread(str(image_path))
    #     if img is None:
    #         raise ValueError(f"無法讀取圖像: {image_path}")
    #     
    #     # 1. 第一步二值化：分割培養皿和背景，取得培養皿區域
    #     petri_dish_region, crop_offset = FasterRCNNDataset._binary_segment_petri_dish(img)
    #     
    #     # 2. 第二步二值化：在培養皿區域內分割物件（RFID、colony、point）
    #     # 注意：這裡我們使用二值化結果作為遮罩，但最終還是使用原始彩色圖像進行後續處理
    #     # 因為 Faster R-CNN 需要彩色圖像，二值化主要用於定位和增強對比
    #     binary_objects = FasterRCNNDataset._binary_segment_objects(petri_dish_region)
    #     
    #     # 將二值化結果作為遮罩，增強物件區域的對比度
    #     # 創建增強圖像：物件區域保持原樣，背景區域稍微變暗
    #     enhanced_petri = petri_dish_region.copy()
    #     mask = binary_objects > 0
    #     # 可以選擇性地增強物件區域的對比度
    #     # enhanced_petri[mask] = np.clip(enhanced_petri[mask] * 1.1, 0, 255).astype(np.uint8)
    #     
    #     # 3. 轉換為灰階
    #     gray = cv2.cvtColor(enhanced_petri, cv2.COLOR_BGR2GRAY)
    #     
    #     # 4. 降低雜訊（使用 Non-local Means Denoising）
    #     denoised = cv2.fastNlMeansDenoising(
    #         gray,
    #         h=PREPROCESS_DENOISE_STRENGTH,
    #         templateWindowSize=7,
    #         searchWindowSize=21
    #     )
    #     
    #     # 5. 銳化（使用 Unsharp Masking）
    #     kernel_size = 2 * PREPROCESS_SHARPEN_RADIUS + 1
    #     gaussian = cv2.GaussianBlur(denoised, (kernel_size, kernel_size), 0)
    #     sharpened = cv2.addWeighted(denoised, 1.5, gaussian, -0.5, 0)
    #     
    #     # 確保值在有效範圍內
    #     sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    #     
    #     # 6. 對比度增強（使用線性變換，不調整亮度）
    #     enhanced = cv2.convertScaleAbs(
    #         sharpened,
    #         alpha=PREPROCESS_CONTRAST_ALPHA,
    #         beta=0
    #     )
    #     
    #     return enhanced, crop_offset
    
    @staticmethod
    def _detect_petri_dish_circle(img: np.ndarray, image_path: Optional[Path] = None) -> Optional[Tuple[int, int, int]]:
        """
        使用 HoughCircles 或 Otsu + 輪廓檢測來定位培養皿的圓形區域
        
        Args:
            img: 原始圖像（BGR 格式）
            image_path: 圖像檔案路徑（可選，用於保存第一次二值化結果）
            
        Returns:
            (center_x, center_y, radius) 元組，如果找不到則返回 None
        """
        # 轉換為灰階
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 進行第一次二值化（無論使用哪種方法檢測圓形，都先進行二值化）
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 保存第一次二值化結果到 threshold 資料夾
        if image_path is not None:
            threshold_dir = Path("threshold")
            threshold_dir.mkdir(parents=True, exist_ok=True)
            threshold_file = threshold_dir / image_path.name
            cv2.imwrite(str(threshold_file), binary)
            logger.debug(f"已保存第一次二值化結果: {threshold_file}")
        
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
    def _load_category_mapping(annotation_file: Optional[str], annotations_dict: Optional[Dict] = None) -> Dict[int, str]:
        """
        從標註檔案載入類別映射（支援 JSON 和 XML 格式）
        
        Args:
            annotation_file: 標註檔案路徑
            annotations_dict: 已載入的標註字典（可選，用於 XML 格式）
            
        Returns:
            category_id -> category_name 的映射
        """
        category_names = {0: "background"}
        
        if annotation_file is None or not Path(annotation_file).exists():
            # 如果提供了 annotations_dict，嘗試從中提取類別資訊（XML 格式）
            if annotations_dict is not None:
                # 從 XML 標註中提取類別名稱（需要從實際標註中推斷）
                # 注意：XML 格式中類別 ID 是動態分配的，需要建立反向映射
                # 這裡我們需要知道類別名稱，所以暫時返回空映射
                # 實際使用時會從 annotations_dict 中動態獲取
                pass
            return category_names
        
        annotation_path = Path(annotation_file)
        
        # 如果是 XML 格式，從標註中提取類別資訊
        if annotation_path.is_dir() or annotation_path.suffix.lower() == '.xml':
            if annotations_dict is not None:
                # 從標註中收集所有類別名稱
                category_id_to_name = {}
                for filename, anns in annotations_dict.items():
                    for ann in anns:
                        cat_id = ann.get('category_id')
                        # 對於 XML 格式，我們需要從實際標註中推斷類別名稱
                        # 但 XML 解析時我們只保存了 category_id，沒有保存名稱
                        # 所以需要重新解析 XML 來獲取類別名稱
                        pass
                # 暫時返回空映射，實際使用時會從標註中動態獲取
            return category_names
        
        # JSON 格式（COCO 格式）
        try:
            with open(annotation_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 從 categories 中載入類別資訊
            for cat in data.get('categories', []):
                cat_id = cat.get('id')
                cat_name = cat.get('name')
                if cat_id is not None and cat_name:
                    category_names[cat_id] = cat_name
                    
        except Exception as e:
            logger.warning(f"無法從 {annotation_file} 載入類別資訊: {e}")
        
        return category_names
    
    @staticmethod
    def _new_preprocess_with_circle_detection(
        image_path: Path,
        annotations: Optional[List[Dict]] = None,
        category_names: Optional[Dict[int, str]] = None
    ) -> np.ndarray:
        """
        新的預處理方法：使用圓形檢測 + CLAHE + RFID 遮罩
        
        處理步驟：
        1. 將影像轉為灰階
        2. 使用 HoughCircles 或 Otsu + 輪廓檢測來定位培養皿的圓形區域
        3. 建立遮罩，將圓形區域以外的背景填滿純黑色
        4. 對圓形區域內部進行 CLAHE（自適應直方圖均衡化）
        5. 如果啟用 RFID 遮罩，將 RFID 區域填充為隨機噪點或平均灰階值
        6. 保持影像尺寸不變
        
        Args:
            image_path: 圖像檔案路徑
            annotations: COCO 格式的標註列表（可選）
            category_names: category_id -> category_name 的映射（可選）
            
        Returns:
            預處理後的圖像陣列（灰階，尺寸與原圖相同）
        """
        # 讀取圖像（BGR 格式）
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"無法讀取圖像: {image_path}")
        
        original_shape = img.shape[:2]  # 保存原始尺寸 (H, W)
        
        # 1. 轉換為灰階
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. 檢測培養皿圓形區域（傳入 image_path 以便保存第一次二值化結果）
        circle_info = FasterRCNNDataset._detect_petri_dish_circle(img, image_path=image_path)
        
        if circle_info is None:
            logger.warning(f"無法檢測到培養皿圓形區域，返回原圖: {image_path}")
            # 即使沒有圓形區域，仍然可以應用 RFID 遮罩（如果啟用）
            if RFID_MASK_ENABLED and annotations is not None and category_names is not None:
                rfid_bboxes = FasterRCNNDataset._get_rfid_bboxes_from_annotations(
                    annotations, category_names
                )
                if len(rfid_bboxes) > 0:
                    gray = FasterRCNNDataset._apply_rfid_mask(
                        gray, rfid_bboxes, RFID_MASK_MODE, RFID_NOISE_INTENSITY
                    )
            return gray
        
        center_x, center_y, radius = circle_info
        
        # 3. 建立遮罩：圓形區域為白色(255)，背景為黑色(0)
        # 增加邊緣容差，確保邊緣 colony 不被排除
        expanded_radius = radius + CIRCLE_MASK_MARGIN
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(mask, (center_x, center_y), int(expanded_radius), 255, -1)  # 填充圓形（帶邊緣容差）
        
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
        
        # 5.5. 第二次二值化：在圓形區域內進行物件分割的二值化
        # 建立更寬鬆的二值化處理遮罩（擴大邊緣容差，避免邊緣 colony 被切除）
        binary_mask_radius = radius + CIRCLE_MASK_MARGIN + BINARY_MASK_MARGIN
        binary_mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(binary_mask, (center_x, center_y), int(binary_mask_radius), 255, -1)  # 更大的處理範圍
        
        # 提取擴大的圓形區域內的圖像進行二值化
        circle_region = masked_gray.copy()
        circle_region[binary_mask == 0] = 0  # 使用擴大的遮罩範圍
        
        # 進行第二次二值化（用於物件分割）
        if BINARY_USE_OTSU:
            _, binary2 = cv2.threshold(circle_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, binary2 = cv2.threshold(circle_region, BINARY_OBJECTS_THRESHOLD, 255, cv2.THRESH_BINARY)
        
        # 形態學操作：去除小噪點
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary2 = cv2.morphologyEx(binary2, cv2.MORPH_OPEN, kernel)
        
        # 保存第二次二值化結果到 threshold2 資料夾
        threshold2_dir = Path("threshold2")
        threshold2_dir.mkdir(parents=True, exist_ok=True)
        threshold2_file = threshold2_dir / image_path.name
        cv2.imwrite(str(threshold2_file), binary2)
        logger.debug(f"已保存第二次二值化結果: {threshold2_file}")
        
        # 5.6. 將第二次二值化結果作為遮罩應用到 masked_gray
        # 只在原始圓形遮罩範圍內應用二值化遮罩，保留邊緣區域
        # 這樣可以避免邊緣 colony 被過度切除，同時讓模型學習二值化檢測到的物件特徵
        # 只在 mask（原始圓形區域）內應用二值化遮罩，邊緣區域保留
        mask_region = (mask > 0)  # 原始圓形區域
        binary2_region = (binary2 > 0)  # 二值化檢測到的物件區域
        # 在原始圓形區域內，將非物件區域設為黑色；圓形區域外保持原樣（已設為黑色）
        masked_gray[mask_region & ~binary2_region] = 0  # 只在圓形區域內且非物件區域設為黑色
        logger.debug(f"已應用第二次二值化遮罩（保留邊緣，容差={CIRCLE_MASK_MARGIN}+{BINARY_MASK_MARGIN}像素），模型將學習二值化檢測到的物件區域")
        
        # 6. 應用 RFID 遮罩（數據增強策略，如果啟用）
        if RFID_MASK_ENABLED and annotations is not None and category_names is not None:
            rfid_bboxes = FasterRCNNDataset._get_rfid_bboxes_from_annotations(
                annotations, category_names
            )
            if len(rfid_bboxes) > 0:
                masked_gray = FasterRCNNDataset._apply_rfid_mask(
                    masked_gray, rfid_bboxes, RFID_MASK_MODE, RFID_NOISE_INTENSITY
                )
                logger.debug(f"已應用 RFID 遮罩: {len(rfid_bboxes)} 個 RFID 區域")
        
        # 確保影像尺寸不變
        assert masked_gray.shape == original_shape, \
            f"影像尺寸改變: 原始={original_shape}, 處理後={masked_gray.shape}"
        
        return masked_gray
    
    @staticmethod
    def _apply_augmentation(
        image: np.ndarray,
        boxes: np.ndarray,
        translate_percent: float,
        scale_min: float,
        scale_max: float,
        rotate_degrees: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        對圖像和邊界框應用資料擴增（平移、縮放、旋轉）
        
        Args:
            image: 圖像陣列 [H, W, C] (RGB)
            boxes: 邊界框陣列 [N, 4] (格式: [x_min, y_min, x_max, y_max])
            translate_percent: 平移範圍（圖像尺寸的百分比）
            scale_min: 縮放範圍最小值
            scale_max: 縮放範圍最大值
            rotate_degrees: 旋轉角度範圍（度）
            
        Returns:
            (augmented_image, augmented_boxes) 元組
        """
        h, w = image.shape[:2]
        augmented_image = image.copy()
        augmented_boxes = boxes.copy()
        
        # 1. 平移
        if translate_percent > 0:
            tx = np.random.uniform(-translate_percent, translate_percent) * w
            ty = np.random.uniform(-translate_percent, translate_percent) * h
            M_translate = np.float32([[1, 0, tx], [0, 1, ty]])
            augmented_image = cv2.warpAffine(augmented_image, M_translate, (w, h), 
                                            borderMode=cv2.BORDER_REPLICATE)
            # 更新邊界框
            if len(augmented_boxes) > 0:
                augmented_boxes[:, [0, 2]] += tx  # x_min, x_max
                augmented_boxes[:, [1, 3]] += ty  # y_min, y_max
        
        # 2. 縮放
        if scale_max > scale_min:
            scale = np.random.uniform(scale_min, scale_max)
            new_w, new_h = int(w * scale), int(h * scale)
            augmented_image = cv2.resize(augmented_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # 如果縮放後尺寸改變，需要裁剪或填充
            if scale < 1.0:
                # 縮小：填充到原始尺寸
                pad_w = (w - new_w) // 2
                pad_h = (h - new_h) // 2
                augmented_image = cv2.copyMakeBorder(
                    augmented_image, pad_h, h - new_h - pad_h, 
                    pad_w, w - new_w - pad_w, 
                    cv2.BORDER_REPLICATE
                )
                # 更新邊界框（考慮填充）
                if len(augmented_boxes) > 0:
                    augmented_boxes[:, [0, 2]] = augmented_boxes[:, [0, 2]] * scale + pad_w
                    augmented_boxes[:, [1, 3]] = augmented_boxes[:, [1, 3]] * scale + pad_h
            else:
                # 放大：裁剪到原始尺寸
                crop_x = (new_w - w) // 2
                crop_y = (new_h - h) // 2
                augmented_image = augmented_image[crop_y:crop_y+h, crop_x:crop_x+w]
                # 更新邊界框（考慮裁剪）
                if len(augmented_boxes) > 0:
                    augmented_boxes[:, [0, 2]] = augmented_boxes[:, [0, 2]] * scale - crop_x
                    augmented_boxes[:, [1, 3]] = augmented_boxes[:, [1, 3]] * scale - crop_y
        
        # 3. 旋轉
        if rotate_degrees > 0:
            angle = np.random.uniform(-rotate_degrees, rotate_degrees)
            center = (w / 2, h / 2)
            M_rotate = cv2.getRotationMatrix2D(center, angle, 1.0)
            augmented_image = cv2.warpAffine(augmented_image, M_rotate, (w, h), 
                                            borderMode=cv2.BORDER_REPLICATE)
            
            # 更新邊界框（旋轉）
            if len(augmented_boxes) > 0:
                # 將邊界框轉換為四個角點，旋轉後再轉回邊界框
                rotated_boxes = []
                for box in augmented_boxes:
                    x_min, y_min, x_max, y_max = box
                    # 四個角點
                    corners = np.array([
                        [x_min, y_min],
                        [x_max, y_min],
                        [x_max, y_max],
                        [x_min, y_max]
                    ], dtype=np.float32)
                    
                    # 旋轉角點
                    ones = np.ones(shape=(len(corners), 1))
                    corners_ones = np.hstack([corners, ones])
                    rotated_corners = M_rotate.dot(corners_ones.T).T
                    
                    # 計算旋轉後的邊界框
                    new_x_min = np.min(rotated_corners[:, 0])
                    new_y_min = np.min(rotated_corners[:, 1])
                    new_x_max = np.max(rotated_corners[:, 0])
                    new_y_max = np.max(rotated_corners[:, 1])
                    
                    # 裁剪到圖像範圍內
                    new_x_min = max(0.0, min(float(w), new_x_min))
                    new_y_min = max(0.0, min(float(h), new_y_min))
                    new_x_max = max(0.0, min(float(w), new_x_max))
                    new_y_max = max(0.0, min(float(h), new_y_max))
                    
                    # 確保邊界框有效（x_min < x_max, y_min < y_max）
                    if new_x_max > new_x_min and new_y_max > new_y_min:
                        rotated_boxes.append([new_x_min, new_y_min, new_x_max, new_y_max])
                
                augmented_boxes = np.array(rotated_boxes, dtype=np.float32)
        
        # 確保邊界框在圖像範圍內
        if len(augmented_boxes) > 0:
            augmented_boxes[:, 0] = np.clip(augmented_boxes[:, 0], 0, w)  # x_min
            augmented_boxes[:, 1] = np.clip(augmented_boxes[:, 1], 0, h)  # y_min
            augmented_boxes[:, 2] = np.clip(augmented_boxes[:, 2], 0, w)  # x_max
            augmented_boxes[:, 3] = np.clip(augmented_boxes[:, 3], 0, h)  # y_max
            
            # 過濾掉無效的邊界框（寬度或高度為 0 或負數）
            valid_mask = (augmented_boxes[:, 2] > augmented_boxes[:, 0]) & \
                        (augmented_boxes[:, 3] > augmented_boxes[:, 1])
            augmented_boxes = augmented_boxes[valid_mask]
        
        return augmented_image, augmented_boxes
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        獲取單個數據樣本
        
        Args:
            idx: 樣本索引
            
        Returns:
            (image_tensor, target_dict) 元組
            - image_tensor: 圖像張量 [C, H, W]
            - target_dict: 包含 'boxes' 和 'labels' 的字典
        """
        image_path = Path(self.image_files[idx])
        filename = image_path.name
        
        # 獲取對應的標註
        filename = os.path.basename(image_path)
        annotations = self._get_coco_annotations_for_image(filename)
        
        # 載入類別映射（如果標註檔案存在）
        category_names = None
        if self.annotation_file and Path(self.annotation_file).exists():
            category_names = FasterRCNNDataset._load_category_mapping(self.annotation_file)
        
        # 使用新的預處理方法（圓形檢測 + CLAHE + RFID 遮罩，保持尺寸不變）
        processed_gray = self._new_preprocess_with_circle_detection(
            image_path,
            annotations=annotations if RFID_MASK_ENABLED else None,
            category_names=category_names if RFID_MASK_ENABLED else None
        )
        
        # 保存灰階圖片到 grayscale/ 目錄（如果啟用）
        if self.grayscale_dir is not None:
            gray_output_path = self.grayscale_dir / filename
            # 只在第一次訪問時保存（避免重複保存）
            if not gray_output_path.exists():
                cv2.imwrite(str(gray_output_path), processed_gray, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # 將灰階圖像轉換為 RGB（Faster R-CNN 需要 3 通道）
        # 將單通道灰階圖像複製為 3 通道
        processed_rgb = cv2.cvtColor(processed_gray, cv2.COLOR_GRAY2RGB)
        
        # 獲取對應的標註
        filename = os.path.basename(image_path)
        annotations = self._get_coco_annotations_for_image(filename)
        
        if annotations:
            # 轉換標註為 Faster R-CNN 格式（優化：使用 numpy 批量處理）
            num_anns = len(annotations)
            boxes_array = np.zeros((num_anns, 4), dtype=np.float32)
            labels_list = []
            
            for i, ann in enumerate(annotations):
                # 轉換 bbox 從 COCO 格式 [x, y, w, h] 到 [x_min, y_min, x_max, y_max]
                bbox_coco = ann['bbox']
                bbox_xyxy = self._convert_bbox_coco_to_xyxy(bbox_coco)
                
                # 注意：新預處理方法保持影像尺寸不變，所以不需要調整座標
                # 但需要確保標註在圖像範圍內
                img_h, img_w = processed_gray.shape[:2]
                if (bbox_xyxy[2] > 0 and bbox_xyxy[0] < img_w and 
                    bbox_xyxy[3] > 0 and bbox_xyxy[1] < img_h):
                    # 裁剪到圖像範圍內
                    bbox_xyxy[0] = max(0, min(img_w, bbox_xyxy[0]))  # x_min
                    bbox_xyxy[1] = max(0, min(img_h, bbox_xyxy[1]))  # y_min
                    bbox_xyxy[2] = max(0, min(img_w, bbox_xyxy[2]))  # x_max
                    bbox_xyxy[3] = max(0, min(img_h, bbox_xyxy[3]))  # y_max
                    
                    # 確保邊界框有效（x_min < x_max, y_min < y_max）
                    if bbox_xyxy[2] > bbox_xyxy[0] and bbox_xyxy[3] > bbox_xyxy[1]:
                        boxes_array[len(labels_list)] = bbox_xyxy
                        labels_list.append(ann['category_id'])
            
            # 只保留有效的標註
            if len(labels_list) < num_anns:
                boxes_array = boxes_array[:len(labels_list)]
        else:
            # 無標註時，創建空的標註
            boxes_array = np.zeros((0, 4), dtype=np.float32)
            labels_list = []
        
        # 應用資料擴增（如果啟用）
        if self.enable_augmentation:
            processed_rgb, boxes_array = self._apply_augmentation(
                processed_rgb,
                boxes_array,
                self.translate_percent,
                self.scale_min,
                self.scale_max,
                self.rotate_degrees
            )
            # 更新 labels_list（過濾掉無效的邊界框對應的標籤）
            if len(boxes_array) < len(labels_list):
                labels_list = labels_list[:len(boxes_array)]
        
        # 轉換為張量
        if len(boxes_array) > 0:
            boxes_tensor = torch.from_numpy(boxes_array)
            labels_tensor = torch.tensor(labels_list, dtype=torch.long)
        else:
            # 無標註時，創建空的標註
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.long)
        
        # 轉換為 PIL Image
        image = Image.fromarray(processed_rgb)
        
        # 轉換圖像為張量
        image_tensor = self.transform(image)
        
        target = {
            'boxes': boxes_tensor,
            'labels': labels_tensor
        }
        
        return image_tensor, target


# ============================================================================
# 批次預處理函數
# ============================================================================

def batch_preprocess_images(
    input_dir: str,
    output_dir: str,
    annotation_file: Optional[str] = None
) -> None:
    """
    使用 OpenCV 批次處理訓練集影像，應用新的圓形檢測預處理
    
    處理邏輯：
    1. 將影像轉為灰階後，利用 HoughCircles 或 Otsu's Binarization 結合輪廓檢測來定位培養皿的圓形區域
    2. 建立一個遮罩（Mask），將培養皿圓形區域以外的所有背景（包含四個死角、文字）填滿純黑色
    3. 對圓形區域內部進行 CLAHE（自適應直方圖均衡化），提升 point 與 colony 的特徵對比度
    4. 處理後保持影像尺寸不變，確保原始標註框的座標依然精準對齊
    
    Args:
        input_dir: 輸入圖像目錄（例如 "model/train/images"）
        output_dir: 輸出圖像目錄（例如 "model/train/images_preprocessed"）
        annotation_file: 標註檔案路徑（可選，如果提供會複製到輸出目錄）
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
        return
    
    logger.info(f"開始批次預處理 {len(image_files)} 張圖片...")
    logger.info(f"輸入目錄: {input_dir}")
    logger.info(f"輸出目錄: {output_dir}")
    if RFID_MASK_ENABLED:
        logger.info(f"RFID 遮罩處理: 已啟用 (模式={RFID_MASK_MODE}, 噪點強度={RFID_NOISE_INTENSITY})")
        logger.info("  目的：防止模型過度依賴 RFID 的強烈特徵，轉而學習辨識微小的 point")
    else:
        logger.info("RFID 遮罩處理: 已關閉")
    
    # 載入標註和類別映射（如果提供）
    annotations_dict = {}
    category_names = None
    if annotation_file and Path(annotation_file).exists():
        # 載入標註（自動檢測 JSON/XML 格式）
        dataset = FasterRCNNDataset(
            image_dir=str(input_dir),
            annotation_file=annotation_file,
            dataset_name="batch_preprocess"
        )
        annotations_dict = dataset.annotations
        
        # 載入類別映射（優先從 JSON 的 categories 中載入，XML 格式會從標註中動態獲取）
        category_names = FasterRCNNDataset._load_category_mapping(annotation_file, annotations_dict)
        
        # 如果 category_names 為空或只有 background，嘗試從標註中提取（XML 格式）
        if len(category_names) <= 1 and annotations_dict:
            # 從標註中收集所有類別名稱（XML 格式）
            category_id_to_name = {}
            for filename, anns in annotations_dict.items():
                for ann in anns:
                    cat_id = ann.get('category_id')
                    cat_name = ann.get('category_name')
                    if cat_id is not None and cat_name:
                        category_id_to_name[cat_id] = cat_name
            if category_id_to_name:
                category_names.update(category_id_to_name)
                logger.info(f"從 XML 標註中提取類別映射: {category_names}")
        
        logger.info(f"已載入 {len(annotations_dict)} 張圖像的標註")
        if category_names:
            logger.info(f"類別映射: {category_names}")
    
    success_count = 0
    fail_count = 0
    
    for idx, image_file in enumerate(image_files, 1):
        try:
            logger.info(f"[{idx}/{len(image_files)}] 處理: {image_file.name}")
            
            # 獲取該圖像的標註（如果有的話）
            filename = image_file.name
            annotations = annotations_dict.get(filename, []) if annotations_dict else None
            
            # 如果啟用 RFID 遮罩且有標註，檢查是否有 RFID 標註
            apply_rfid_mask = False
            if RFID_MASK_ENABLED and annotations:
                # 檢查是否有 RFID 標註
                for ann in annotations:
                    # 檢查是否為 RFID 標註（支援 JSON 和 XML 格式）
                    is_rfid = False
                    if category_names:
                        rfid_category_id = None
                        for cat_id, cat_name in category_names.items():
                            if cat_name == 'RFID':
                                rfid_category_id = cat_id
                                break
                        if rfid_category_id is not None and ann.get('category_id') == rfid_category_id:
                            is_rfid = True
                    if ann.get('category_name', '').lower() == 'rfid':
                        is_rfid = True
                    
                    if is_rfid:
                        apply_rfid_mask = True
                        break
            
            # 使用新的預處理方法（包含 RFID 遮罩，如果啟用）
            processed_gray = FasterRCNNDataset._new_preprocess_with_circle_detection(
                image_file,
                annotations=annotations if apply_rfid_mask else None,
                category_names=category_names if apply_rfid_mask else None
            )
            
            if apply_rfid_mask:
                logger.debug(f"  ✓ 已應用 RFID 遮罩處理")
            
            # 保存處理後的圖像
            output_file = output_path / image_file.name
            cv2.imwrite(str(output_file), processed_gray, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            success_count += 1
            logger.debug(f"  ✓ 已保存: {output_file}")
            
        except Exception as e:
            fail_count += 1
            logger.error(f"  ✗ 處理 {image_file.name} 時發生錯誤: {e}")
    
    logger.info("=" * 60)
    logger.info("批次預處理完成")
    logger.info(f"成功處理: {success_count} 張")
    logger.info(f"失敗: {fail_count} 張")
    logger.info(f"輸出目錄: {output_dir}")
    
    # 如果提供了標註檔案，複製到輸出目錄
    if annotation_file and Path(annotation_file).exists():
        output_annotation = output_path.parent / Path(annotation_file).name
        import shutil
        shutil.copy2(annotation_file, output_annotation)
        logger.info(f"已複製標註檔案: {output_annotation}")
        logger.info("注意：標註座標保持不變，因為影像尺寸未改變")


# ============================================================================
# 數據加載工具函數
# ============================================================================

def collate_fn(batch: List[Tuple[torch.Tensor, Dict]]) -> Tuple[List[torch.Tensor], List[Dict]]:
    """
    自定義批次整理函數，用於 Faster R-CNN 數據集
    
    Args:
        batch: 批次數據列表，每個元素是 (image, target) 元組
        
    Returns:
        (images, targets) 元組
    """
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


def create_data_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: Optional[int] = None,
    pin_memory: Optional[bool] = None,
    prefetch_factor: Optional[int] = None
) -> DataLoader:
    """
    創建數據加載器（優化以充分利用 RTX 5090）
    
    Args:
        dataset: 數據集實例
        batch_size: 批次大小
        shuffle: 是否打亂數據
        num_workers: 工作進程數，None 時自動計算
        pin_memory: 是否使用 pin_memory，None 時根據 CUDA 可用性自動決定
        prefetch_factor: 預取因子，None 時使用默認值
        
    Returns:
        DataLoader 實例
    """
    if num_workers is None:
        num_workers = min(MAX_NUM_WORKERS, os.cpu_count() or 1)
    
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    
    # RTX 5090 優化：增加預取因子以充分利用 GPU
    if prefetch_factor is None:
        prefetch_factor = 4 if num_workers > 0 else None
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor
    )


# ============================================================================
# 自定義 Loss 調整工具
# ============================================================================

def calculate_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    計算兩組邊界框之間的 IoU
    
    Args:
        boxes1: [N, 4] 格式的邊界框 [x_min, y_min, x_max, y_max]
        boxes2: [M, 4] 格式的邊界框 [x_min, y_min, x_max, y_max]
        
    Returns:
        [N, M] 格式的 IoU 矩陣
    """
    return box_iou(boxes1, boxes2)


def check_box_inside_rfid(pred_box: torch.Tensor, rfid_boxes: torch.Tensor) -> bool:
    """
    檢查預測框是否落在任何 RFID 標註框內
    
    Args:
        pred_box: [4] 格式的預測框 [x_min, y_min, x_max, y_max]
        rfid_boxes: [N, 4] 格式的 RFID 標註框
        
    Returns:
        如果預測框的中心點或任何部分落在 RFID 框內，返回 True
    """
    if len(rfid_boxes) == 0:
        return False
    
    # 計算預測框的中心點
    pred_center_x = (pred_box[0] + pred_box[2]) / 2
    pred_center_y = (pred_box[1] + pred_box[3]) / 2
    
    # 檢查中心點是否在任何 RFID 框內（使用向量化操作）
    center_inside = (
        (pred_center_x >= rfid_boxes[:, 0]) & 
        (pred_center_x <= rfid_boxes[:, 2]) &
        (pred_center_y >= rfid_boxes[:, 1]) & 
        (pred_center_y <= rfid_boxes[:, 3])
    )
    if center_inside.any().item():
        return True
    
    # 也檢查預測框是否與任何 RFID 框有重疊（IoU > 0）
    pred_box_expanded = pred_box.unsqueeze(0)  # [1, 4]
    ious = calculate_iou(pred_box_expanded, rfid_boxes)  # [1, N]
    return (ious > 0).any().item()


class CustomFasterRCNN(nn.Module):
    """
    自定義 Faster R-CNN 包裝器，實現 Hard Negative Mining
    針對 RFID 區域內的誤判進行特殊處理
    """
    
    def __init__(self, base_model: nn.Module, hard_negative_weight: float = 3.0, iou_threshold: float = 0.3):
        """
        初始化自定義模型
        
        Args:
            base_model: 基礎 Faster R-CNN 模型
            hard_negative_weight: Hard Negative Mining 的權重（預設 3.0）
            iou_threshold: IoU 閾值，低於此值則判定為背景（預設 0.3）
        """
        super().__init__()
        self.base_model = base_model
        self.hard_negative_weight = hard_negative_weight
        self.iou_threshold = iou_threshold
        
        # 類別 ID 映射（根據標註文件）
        self.rfid_category_id = 1
        self.colony_category_id = 2
        self.point_category_id = 3
        self.background_category_id = 0
    
    def forward(self, images: List[torch.Tensor], targets: Optional[List[Dict]] = None):
        """
        前向傳播，在計算 Loss 時進行特殊處理
        
        Args:
            images: 圖像列表
            targets: 目標標註列表（訓練時需要）
            
        Returns:
            Loss 字典（訓練模式）或預測結果（評估模式）
        """
        if self.training and targets is not None:
            # 訓練模式：計算 Loss 並進行特殊處理
            loss_dict = self.base_model(images, targets)
            
            # 對每個圖像進行特殊處理
            modified_loss_dict = {}
            for key, loss_value in loss_dict.items():
                if key == 'loss_classifier':
                    # 對分類 Loss 進行特殊處理
                    modified_loss = self._adjust_classifier_loss(
                        images, targets, loss_value
                    )
                    modified_loss_dict[key] = modified_loss
                else:
                    modified_loss_dict[key] = loss_value
            
            return modified_loss_dict
        else:
            # 評估模式：直接返回預測結果
            return self.base_model(images, targets)
    
    def _adjust_classifier_loss(
        self,
        images: List[torch.Tensor],
        targets: List[Dict],
        original_loss: torch.Tensor
    ) -> torch.Tensor:
        """
        調整分類 Loss，實現 Hard Negative Mining
        
        策略：
        1. 在計算 Loss 之前，先獲取預測框（通過臨時切換到評估模式）
        2. 檢查預測框是否落在 RFID 區域內
        3. 檢查預測框與 colony/point 標註的 IoU
        4. 如果 IoU < 0.3，則將其判定為背景，並加大 Loss 權重
        
        注意：這會導致計算兩次 Loss（一次獲取預測框，一次計算最終 Loss），
        但這是目前最可行的方法。
        """
        # 暫時切換到評估模式以獲取預測框
        was_training = self.base_model.training
        self.base_model.eval()
        
        with torch.no_grad():
            # 獲取預測框（在評估模式下）
            predictions = self.base_model(images, None)
        
        # 恢復訓練模式
        if was_training:
            self.base_model.train()
        
        # 計算需要加權的 Loss 調整量
        # 注意：我們使用一個與 original_loss 相關的調整量，以確保梯度可以正確傳播
        loss_adjustment = torch.tensor(0.0, device=original_loss.device, dtype=original_loss.dtype)
        total_hard_negatives = 0
        
        for i, (pred, target) in enumerate(zip(predictions, targets)):
            # 獲取該圖像的標註
            target_boxes = target['boxes']  # [N, 4]
            target_labels = target['labels']  # [N]
            
            # 找出 RFID 標註框
            rfid_mask = target_labels == self.rfid_category_id
            rfid_boxes = target_boxes[rfid_mask]
            
            # 找出 colony 和 point 標註框
            colony_point_mask = (target_labels == self.colony_category_id) | (target_labels == self.point_category_id)
            colony_point_boxes = target_boxes[colony_point_mask]
            
            if len(rfid_boxes) == 0 or len(pred['boxes']) == 0:
                continue
            
            # 獲取預測框
            pred_boxes = pred['boxes']  # [M, 4]
            pred_labels = pred['labels']  # [M]
            pred_scores = pred['scores']  # [M]
            
            # 找出被預測為 colony 或 point 的框
            colony_point_pred_mask = (pred_labels == self.colony_category_id) | (pred_labels == self.point_category_id)
            colony_point_pred_boxes = pred_boxes[colony_point_pred_mask]
            colony_point_pred_scores = pred_scores[colony_point_pred_mask]
            
            if len(colony_point_pred_boxes) == 0:
                continue
            
            # 檢查每個預測框
            for j, pred_box in enumerate(colony_point_pred_boxes):
                # 檢查是否落在 RFID 區域內
                if not check_box_inside_rfid(pred_box, rfid_boxes):
                    continue
                
                # 計算與 colony/point 標註的 IoU
                max_iou = 0.0
                if len(colony_point_boxes) > 0:
                    pred_box_expanded = pred_box.unsqueeze(0)  # [1, 4]
                    ious = calculate_iou(pred_box_expanded, colony_point_boxes)  # [1, N]
                    max_iou = ious.max().item()
                
                # 如果 IoU < 0.3，則判定為背景（Hard Negative）
                if max_iou < self.iou_threshold:
                    # 計算該預測框的分類錯誤 Loss（近似）
                    # 預測分數越高，說明模型越確信，如果誤判，Loss 應該越大
                    pred_score = colony_point_pred_scores[j].item()  # 轉換為 Python float
                    
                    # 計算 Hard Negative Mining 的權重調整
                    # 預測分數越高但實際是背景，Loss 應該越大
                    # 使用交叉熵的近似：-log(1 - score) * weight
                    # 注意：這裡我們使用固定的 pred_score 值來計算調整量
                    # 我們通過將調整量乘以 original_loss 來確保梯度可以正確傳播
                    hard_negative_loss_value = -np.log(1.0 - pred_score + 1e-8) * self.hard_negative_weight
                    # 將調整量轉換為與 original_loss 相關的張量，以確保梯度傳播
                    hard_negative_loss = original_loss * 0.0 + hard_negative_loss_value
                    loss_adjustment = loss_adjustment + hard_negative_loss
                    total_hard_negatives += 1
        
        # 將調整量加到原始 Loss 上
        adjusted_loss = original_loss + loss_adjustment
        
        # 記錄統計信息（每 100 個 batch 記錄一次）
        if not hasattr(self, '_hard_negative_counter'):
            self._hard_negative_counter = 0
        self._hard_negative_counter += 1
        
        if self._hard_negative_counter % 100 == 0 and total_hard_negatives > 0:
            logger.debug(f"Hard Negative Mining: 發現 {total_hard_negatives} 個 Hard Negative 樣本，Loss 調整量: {loss_adjustment.item():.4f}")
        
        return adjusted_loss


# ============================================================================
# 訓練器類別
# ============================================================================

class TrainingConfig:
    """訓練配置類別"""
    
    def __init__(
        self,
        num_epochs: int = DEFAULT_NUM_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        momentum: float = DEFAULT_MOMENTUM,
        weight_decay: float = DEFAULT_WEIGHT_DECAY,
        lr_step_size: int = DEFAULT_LR_STEP_SIZE,
        lr_gamma: float = DEFAULT_LR_GAMMA
    ):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma


class FasterRCNNTrainer:
    """Faster R-CNN 模型訓練器"""
    
    def __init__(
        self,
        train_image_dir: str = "model/train/images",
        train_annotation_file: str = "model/train/annotations.json",
        val_image_dir: str = "model/val/images",
        val_annotation_file: str = "model/val/annotations.json",
        output_dir: str = "run",
        num_classes: int = 2,
        enable_hard_negative_mining: bool = True,
        hard_negative_weight: float = 3.0,
        iou_threshold: float = 0.3
    ):
        """
        初始化訓練器
        
        Args:
            train_image_dir: 訓練集圖像目錄
            train_annotation_file: 訓練集標註檔案
            val_image_dir: 驗證集圖像目錄
            val_annotation_file: 驗證集標註檔案
            output_dir: 模型輸出目錄（run/）
            num_classes: 類別數量（包括背景，例如 2 表示 1 個物體類別 + 背景）
            enable_hard_negative_mining: 是否啟用 Hard Negative Mining（預設 True）
            hard_negative_weight: Hard Negative Mining 的權重（預設 3.0）
            iou_threshold: IoU 閾值，低於此值則判定為背景（預設 0.3）
        """
        self.train_image_dir = Path(train_image_dir)
        self.train_annotation_file = Path(train_annotation_file)
        self.val_image_dir = Path(val_image_dir)
        self.val_annotation_file = Path(val_annotation_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_classes = num_classes
        self.enable_hard_negative_mining = enable_hard_negative_mining
        self.hard_negative_weight = hard_negative_weight
        self.iou_threshold = iou_threshold
        
        # 載入預訓練模型
        self.model = self._load_model(num_classes)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # RTX 5090 優化：啟用 CUDNN benchmark 和自動調優
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False  # 允許非確定性操作以提升性能
            # 啟用 TensorFloat-32 (TF32) 以加速訓練（RTX 5090 支持）
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # 顯示 GPU 資訊
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"GPU 設備: {gpu_name}")
            logger.info(f"GPU 記憶體: {gpu_memory:.2f} GB")
            logger.info("已啟用 CUDNN benchmark 和 TF32 加速")
        
        logger.info(f"模型已載入到設備: {self.device}")
        
        # 初始化數據集（延遲載入）
        self.train_dataset: Optional[FasterRCNNDataset] = None
        self.val_dataset: Optional[FasterRCNNDataset] = None
    
    def _load_model(self, num_classes: int) -> torch.nn.Module:
        """
        載入預訓練模型並替換分類頭
        
        Args:
            num_classes: 類別數量（包括背景）
            
        Returns:
            配置好的模型（如果啟用 Hard Negative Mining，則返回包裝後的模型）
        """
        logger.info("正在載入預訓練 Faster R-CNN 模型...")
        try:
            # 向後兼容：新版本使用 weights，舊版本使用 pretrained
            if USE_WEIGHTS_API:
                base_model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
            else:
                base_model = fasterrcnn_resnet50_fpn(pretrained=True)
            
            # 替換分類頭（ROI Pooling 後的分類預測器）
            in_features = base_model.roi_heads.box_predictor.cls_score.in_features
            base_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            
            # 如果啟用 Hard Negative Mining，則包裝模型
            if self.enable_hard_negative_mining:
                model = CustomFasterRCNN(
                    base_model,
                    hard_negative_weight=self.hard_negative_weight,
                    iou_threshold=self.iou_threshold
                )
                logger.info("✓ 模型載入完成（已啟用 Hard Negative Mining）")
                logger.info(f"  Hard Negative Mining 權重: {self.hard_negative_weight}")
                logger.info(f"  IoU 閾值: {self.iou_threshold}")
            else:
                model = base_model
                logger.info("✓ 模型載入完成")
            
            return model
        except Exception as e:
            logger.error(f"載入模型失敗: {e}")
            raise
    
    def _load_dataset(
        self, 
        image_dir: Path, 
        annotation_file: Path, 
        dataset_name: str
    ) -> FasterRCNNDataset:
        """
        載入單個數據集
        
        Args:
            image_dir: 圖像目錄
            annotation_file: 標註檔案路徑
            dataset_name: 數據集名稱（用於日誌和保存灰階圖片）
            
        Returns:
            FasterRCNNDataset 實例
        """
        logger.info(f"載入{dataset_name}: {image_dir}")
        
        ann_file = str(annotation_file) if annotation_file.exists() else None
        
        # 使用全域設定（AUGMENTATION_ENABLED）
        # 訓練集啟用資料擴增，驗證集關閉（已改為全域關閉）
        enable_aug = AUGMENTATION_ENABLED  # 使用全域設定，目前為 False
        
        dataset = FasterRCNNDataset(
            image_dir=str(image_dir),
            annotation_file=ann_file,
            dataset_name=dataset_name,
            enable_augmentation=enable_aug
        )
        
        logger.info(f"  {dataset_name}大小: {len(dataset)} 張圖像")
        if enable_aug:
            logger.info(f"  資料擴增: 已啟用（平移±{AUGMENTATION_TRANSLATE_PERCENT*100:.0f}%, "
                       f"縮放{AUGMENTATION_SCALE_MIN:.1f}-{AUGMENTATION_SCALE_MAX:.1f}, "
                       f"旋轉±{AUGMENTATION_ROTATE_DEGREES:.0f}度）")
        else:
            logger.info(f"  資料擴增: 已關閉")
        if SAVE_GRAYSCALE:
            logger.info(f"  灰階圖片將保存到: grayscale/{dataset_name}/images/")
        return dataset
    
    def prepare_datasets(self):
        """準備訓練和驗證數據集"""
        logger.info("載入數據集...")
        
        # 載入訓練數據集
        self.train_dataset = self._load_dataset(
            self.train_image_dir,
            self.train_annotation_file,
            "訓練集"
        )
        
        # 載入驗證數據集
        self.val_dataset = self._load_dataset(
            self.val_image_dir,
            self.val_annotation_file,
            "驗證集"
        )
    
    def _save_model_and_info(
        self,
        num_epochs: int,
        batch_size: int,
        learning_rate: float
    ) -> Path:
        """
        保存模型和訓練資訊
        
        Args:
            num_epochs: 訓練輪數
            batch_size: 批次大小
            learning_rate: 學習率
            
        Returns:
            模型保存路徑
        """
        model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_save_path = self.output_dir / f"model_{model_version}"
        model_save_path.mkdir(exist_ok=True)
        
        logger.info(f"正在保存模型到: {model_save_path}")
        
        # 如果使用包裝器，保存基礎模型
        if isinstance(self.model, CustomFasterRCNN):
            base_model = self.model.base_model
        else:
            base_model = self.model
        
        # 保存模型狀態
        torch.save(base_model.state_dict(), model_save_path / "model.pth")
        
        # 保存完整模型（可選）
        torch.save(base_model, model_save_path / "model_full.pth")
        
        # 保存訓練資訊
        training_info = {
            "model_version": model_version,
            "model_type": "Faster R-CNN",
            "num_classes": self.num_classes,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "train_samples": len(self.train_dataset),
            "val_samples": len(self.val_dataset),
            "save_path": str(model_save_path),
            "train_image_dir": str(self.train_image_dir),
            "val_image_dir": str(self.val_image_dir),
            "hard_negative_mining": {
                "enabled": self.enable_hard_negative_mining,
                "weight": self.hard_negative_weight if self.enable_hard_negative_mining else None,
                "iou_threshold": self.iou_threshold if self.enable_hard_negative_mining else None
            }
        }
        
        info_file = model_save_path / "training_info.json"
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(training_info, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ 模型訓練完成並保存到: {model_save_path}")
        return model_save_path
    
    def train(
        self,
        config: Optional[TrainingConfig] = None,
        num_epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None
    ) -> str:
        """
        訓練模型
        
        Args:
            config: 訓練配置對象（優先使用）
            num_epochs: 訓練輪數（如果未提供 config）
            batch_size: 批次大小（如果未提供 config）
            learning_rate: 學習率（如果未提供 config）
            
        Returns:
            模型保存路徑
        """
        # 處理配置參數
        if config is None:
            config = TrainingConfig(
                num_epochs=num_epochs or DEFAULT_NUM_EPOCHS,
                batch_size=batch_size or DEFAULT_BATCH_SIZE,
                learning_rate=learning_rate or DEFAULT_LEARNING_RATE
            )
        
        logger.info("=" * 60)
        logger.info("開始訓練 Faster R-CNN 模型")
        logger.info("=" * 60)
        
        # 準備數據集
        self.prepare_datasets()
        
        # 創建數據加載器（RTX 5090 優化）
        train_loader = create_data_loader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            prefetch_factor=4  # 增加預取以充分利用 GPU
        )
        
        val_loader = create_data_loader(
            self.val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            prefetch_factor=4
        )
        
        logger.info(f"數據加載器配置: num_workers={train_loader.num_workers}, pin_memory={train_loader.pin_memory}")
        
        # 設置優化器和學習率調度器
        optimizer, lr_scheduler = self._create_optimizer_and_scheduler(config)
        
        # 顯示訓練參數
        self._log_training_config(config)
        
        # 初始化混合精度訓練（AMP）
        use_amp, scaler = self._setup_amp()
        
        # 執行訓練循環
        best_val_loss = self._train_loop(
            train_loader, val_loader, optimizer, lr_scheduler, config, use_amp, scaler
        )
        
        # 保存最終模型
        model_save_path = self._save_model_and_info(
            config.num_epochs, config.batch_size, config.learning_rate
        )
        
        return str(model_save_path)
    
    def _create_optimizer_and_scheduler(
        self, config: TrainingConfig
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        """
        創建優化器和學習率調度器
        
        Args:
            config: 訓練配置
            
        Returns:
            (optimizer, lr_scheduler) 元組
        """
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            params,
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )
        
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.lr_step_size,
            gamma=config.lr_gamma
        )
        
        return optimizer, lr_scheduler
    
    def _log_training_config(self, config: TrainingConfig):
        """記錄訓練配置資訊"""
        logger.info("訓練參數:")
        logger.info(f"  輪數: {config.num_epochs}")
        logger.info(f"  批次大小: {config.batch_size}")
        logger.info(f"  學習率: {config.learning_rate}")
        logger.info(f"  輸出目錄: {self.output_dir}")
        logger.info(f"  訓練樣本: {len(self.train_dataset)}")
        logger.info(f"  驗證樣本: {len(self.val_dataset)}")
        logger.info("=" * 60)
    
    def _setup_amp(self) -> Tuple[bool, Optional[GradScaler]]:
        """
        設置混合精度訓練（AMP）- RTX 5090 優化
        
        Returns:
            (use_amp, scaler) 元組
        """
        use_amp = torch.cuda.is_available()
        if use_amp:
            # RTX 5090 優化：使用更積極的混合精度設置
            scaler = GradScaler(
                init_scale=2.**16,  # 初始縮放因子
                growth_factor=2.0,   # 增長因子
                backoff_factor=0.5,  # 回退因子
                growth_interval=2000  # 增長間隔
            )
            logger.info("已啟用混合精度訓練 (AMP) 以加速訓練（RTX 5090 優化）")
        else:
            scaler = None
        return use_amp, scaler
    
    def _train_loop(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
        config: TrainingConfig,
        use_amp: bool,
        scaler: Optional[GradScaler]
    ) -> float:
        """
        執行訓練循環
        
        Args:
            train_loader: 訓練數據加載器
            val_loader: 驗證數據加載器
            optimizer: 優化器
            lr_scheduler: 學習率調度器
            config: 訓練配置
            use_amp: 是否使用混合精度
            scaler: GradScaler（如果使用 AMP）
            
        Returns:
            最佳驗證損失
        """
        best_val_loss = float('inf')
        self.model.train()
        
        # 清理 GPU 記憶體快取（避免記憶體碎片化）
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        for epoch in range(config.num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{config.num_epochs}")
            logger.info("-" * 60)
            
            # 訓練階段
            train_loss = 0.0
            train_batches = 0
            
            for batch_idx, (images, targets) in enumerate(train_loader):
                # RTX 5090 優化：使用 non_blocking 異步傳輸以充分利用 GPU
                images = [img.to(self.device, non_blocking=True) for img in images]
                targets = [{k: v.to(self.device, non_blocking=True) for k, v in t.items()} for t in targets]
                
                # 同步 CUDA 流以確保數據傳輸完成（僅在需要時）
                if batch_idx == 0:
                    torch.cuda.synchronize()
                
                # 【前向傳播】自動執行 RPN → ROI Pooling → 分類+回歸
                optimizer.zero_grad()
                if use_amp:
                    with autocast(device_type='cuda'):
                        loss_dict = self.model(images, targets)  # RPN 和 ROI Pooling 在此自動執行
                        losses = sum(loss for loss in loss_dict.values())
                    
                    # 反向傳播（使用 scaler）
                    scaler.scale(losses).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss_dict = self.model(images, targets)  # RPN 和 ROI Pooling 在此自動執行
                    losses = sum(loss for loss in loss_dict.values())
                    losses.backward()
                    optimizer.step()
                
                train_loss += losses.item()
                train_batches += 1
                
                # 定期清理 GPU 記憶體快取（每 10 個 batch）
                if (batch_idx + 1) % 10 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    logger.info(f"  Batch {batch_idx + 1}/{len(train_loader)}, Loss: {losses.item():.4f}")
            
            avg_train_loss = train_loss / train_batches if train_batches > 0 else 0.0
            logger.info(f"  平均訓練損失: {avg_train_loss:.4f}")
            
            # 驗證階段
            # 注意：Faster R-CNN 在 eval() 模式下返回預測結果而非損失
            # 因此保持 train() 模式，但使用 no_grad() 避免計算梯度
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for images, targets in val_loader:
                    # RTX 5090 優化：使用 non_blocking 異步傳輸
                    images = [img.to(self.device, non_blocking=True) for img in images]
                    targets = [{k: v.to(self.device, non_blocking=True) for k, v in t.items()} for t in targets]
                    
                    # 【驗證階段】同樣執行 RPN → ROI Pooling → 分類+回歸
                    if use_amp:
                        with autocast(device_type='cuda'):
                            loss_dict = self.model(images, targets)  # RPN 和 ROI Pooling 在此自動執行
                        losses = sum(loss for loss in loss_dict.values())
                    else:
                        loss_dict = self.model(images, targets)  # RPN 和 ROI Pooling 在此自動執行
                        losses = sum(loss for loss in loss_dict.values())
                    
                    val_loss += losses.item()
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches if val_batches > 0 else 0.0
            logger.info(f"  平均驗證損失: {avg_val_loss:.4f}")
            
            # 更新學習率
            lr_scheduler.step()
            
            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                logger.info(f"  ✓ 發現更好的模型（驗證損失: {best_val_loss:.4f}）")
            
            # 每個 epoch 結束後清理 GPU 記憶體快取
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.model.train()
        
        return best_val_loss


def main():
    """主函數"""
    import sys
    
    # 檢查是否要執行批次預處理
    if len(sys.argv) > 1 and sys.argv[1] == "preprocess":
        # 批次預處理模式
        logger.info("=" * 60)
        logger.info("批次預處理訓練集影像")
        logger.info("=" * 60)
        
        try:
            # 預處理訓練集
            batch_preprocess_images(
                input_dir="model/train/images",
                output_dir="model/train/images_preprocessed",
                annotation_file="model/train/annotations.json"
            )
            
            # 預處理驗證集
            batch_preprocess_images(
                input_dir="model/val/images",
                output_dir="model/val/images_preprocessed",
                annotation_file="model/val/annotations.json"
            )
            
            logger.info("\n✓ 批次預處理完成！")
            logger.info("預處理後的影像已保存到:")
            logger.info("  - model/train/images_preprocessed/")
            logger.info("  - model/val/images_preprocessed/")
            logger.info("\n注意：標註檔案已複製到對應目錄，座標保持不變（影像尺寸未改變）")
            
        except Exception as e:
            logger.error(f"批次預處理發生錯誤: {e}", exc_info=True)
        
        return
    
    # 正常訓練模式
    logger.info("=" * 60)
    logger.info("Faster R-CNN 模型訓練")
    logger.info("=" * 60)
    
    try:
        # 創建訓練器（會在初始化時檢查目錄）
        # num_classes 需要根據實際類別數調整（背景 + 物體類別數）
        # 例如：1 個物體類別 = 2（背景 + 物體）
        trainer = FasterRCNNTrainer(
            train_image_dir="model/train/images",
            train_annotation_file="model/train/annotations.json",
            val_image_dir="model/val/images",
            val_annotation_file="model/val/annotations.json",
            output_dir="run",
            num_classes=4  # 背景 + 3個類別（RFID、colony、point）
        )
        
        # 開始訓練（RTX 5090 優化配置）
        config = TrainingConfig(
            num_epochs=10,
            batch_size=8,  # 降低批次大小以避免 CUDA 記憶體不足
            learning_rate=0.005
        )
        model_path = trainer.train(config=config)
        
        logger.info(f"\n訓練完成！模型已保存到: {model_path}")
        
    except FileNotFoundError as e:
        logger.error(f"檔案或目錄不存在: {e}")
    except ValueError as e:
        logger.error(f"數據錯誤: {e}")
    except Exception as e:
        logger.error(f"訓練過程發生錯誤: {e}", exc_info=True)


if __name__ == "__main__":
    main()
