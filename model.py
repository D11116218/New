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

# 資料擴增配置
AUGMENTATION_ENABLED = True  # 是否啟用資料擴增（已開啟）
AUGMENTATION_TRANSLATE_PERCENT = 0.05  # 平移範圍：±5% 圖像尺寸
AUGMENTATION_SCALE_MIN = 0.7  # 縮放範圍最小值：90%
AUGMENTATION_SCALE_MAX = 1.1  # 縮放範圍最大值：110%
AUGMENTATION_ROTATE_DEGREES = 180  # 旋轉角度範圍：±15度

# ============================================================================
# 數據集類別
# ============================================================================

class FasterRCNNDataset(Dataset):
    
    
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
        self.annotation_file = annotation_file
        if annotation_file and Path(annotation_file).exists():
            self.annotations, self.image_id_map = self._load_annotations(annotation_file)
        
        # 準備 grayscale 目錄（用於保存灰階圖片）
        if SAVE_GRAYSCALE:
            self.grayscale_dir = Path("grayscale") / dataset_name / "images"
            self.grayscale_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.grayscale_dir = None
    
    def _load_image_files(self) -> List[str]:
        image_files = []
        for ext in self.IMAGE_EXTENSIONS:
            image_files.extend(self.image_dir.glob(f"*{ext}"))
            image_files.extend(self.image_dir.glob(f"*{ext.upper()}"))
        return sorted([str(f) for f in image_files])
    
    @staticmethod
    def _load_xml_annotations(annotation_file: str, image_dir: Path) -> Tuple[Dict, Dict]:
        
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
        
        return len(self.image_files)
    
    @staticmethod
    def _convert_bbox_coco_to_xyxy(bbox: List[float]) -> List[float]:
        
        x_min, y_min, width, height = bbox
        return [x_min, y_min, x_min + width, y_min + height]
    
    def _get_coco_annotations_for_image(self, filename: str) -> List[Dict]:
        return self.annotations.get(filename, [])
    
    
    @staticmethod
    def _load_category_mapping(annotation_file: Optional[str], annotations_dict: Optional[Dict] = None) -> Dict[int, str]:
        category_names = {0: "background"}
        if annotation_file is None or not Path(annotation_file).exists():
            return category_names
        
        annotation_path = Path(annotation_file)
        if annotation_path.is_dir() or annotation_path.suffix.lower() == '.xml':
            return category_names
        
        try:
            with open(annotation_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for cat in data.get('categories', []):
                cat_id = cat.get('id')
                cat_name = cat.get('name')
                if cat_id is not None and cat_name:
                    category_names[cat_id] = cat_name
        except Exception as e:
            logger.warning(f"無法從 {annotation_file} 載入類別資訊: {e}")
        return category_names
    
    @staticmethod
    def _preprocess_image(
        image_path: Path
    ) -> np.ndarray:
        """
        預處理流程（按順序執行）：
        1. 轉灰階（opencv預設）
        2. 銳利化
        3. 降低雜訊
        4. 對比度增強
        """
        # 讀取圖像（BGR 格式）
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"無法讀取圖像: {image_path}")
        
        original_shape = img.shape[:2]  # 保存原始尺寸 (H, W)
        
        # 1. 轉換為灰階（opencv預設方法）
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. 銳利化
        # 使用 Unsharp Masking 方法：銳化圖 = 原圖 + (原圖 - 模糊圖) * 強度
        # 計算高斯模糊的核大小（必須是奇數）
        kernel_size = int(PREPROCESS_SHARPEN_RADIUS * 2 + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), PREPROCESS_SHARPEN_RADIUS)
        # Unsharp Masking：原圖 + (原圖 - 模糊圖)
        sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
        # 確保值在有效範圍內
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        # 3. 降低雜訊
        # 使用 Non-local Means Denoising（對灰階圖像）
        denoised = cv2.fastNlMeansDenoising(
            sharpened,
            h=PREPROCESS_DENOISE_STRENGTH,  # 降噪強度參數
            templateWindowSize=7,
            searchWindowSize=21
        )
        
        # 4. 對比度增強
        # 使用 CLAHE（自適應直方圖均衡化）進行對比度增強
        clahe = cv2.createCLAHE(
            clipLimit=PREPROCESS_CONTRAST_ALPHA * 2.0,  # 對比度增強係數
            tileGridSize=(8, 8)
        )
        contrast_enhanced = clahe.apply(denoised)
        
        # 確保影像尺寸不變
        assert contrast_enhanced.shape == original_shape, \
            f"影像尺寸改變: 原始={original_shape}, 處理後={contrast_enhanced.shape}"
        
        return contrast_enhanced
    
    @staticmethod
    def _apply_augmentation(
        image: np.ndarray,
        boxes: np.ndarray,
        translate_percent: float,
        scale_min: float,
        scale_max: float,
        rotate_degrees: float
    ) -> Tuple[np.ndarray, np.ndarray]:
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
        
        if len(augmented_boxes) > 0:
            augmented_boxes[:, [0, 2]] = np.clip(augmented_boxes[:, [0, 2]], 0, w)
            augmented_boxes[:, [1, 3]] = np.clip(augmented_boxes[:, [1, 3]], 0, h)
            valid_mask = (augmented_boxes[:, 2] > augmented_boxes[:, 0]) & (augmented_boxes[:, 3] > augmented_boxes[:, 1])
            augmented_boxes = augmented_boxes[valid_mask]
        return augmented_image, augmented_boxes
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        image_path = Path(self.image_files[idx])
        filename = image_path.name
        
        processed_gray = self._preprocess_image(image_path)
        
        if self.grayscale_dir and not (self.grayscale_dir / filename).exists():
            cv2.imwrite(str(self.grayscale_dir / filename), processed_gray, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        processed_rgb = cv2.cvtColor(processed_gray, cv2.COLOR_GRAY2RGB)
        annotations = self._get_coco_annotations_for_image(filename)
        
        img_h, img_w = processed_gray.shape[:2]
        boxes_list, labels_list = [], []
        
        for ann in annotations:
            bbox_xyxy = self._convert_bbox_coco_to_xyxy(ann['bbox'])
            bbox_xyxy[0] = max(0, min(img_w, bbox_xyxy[0]))
            bbox_xyxy[1] = max(0, min(img_h, bbox_xyxy[1]))
            bbox_xyxy[2] = max(0, min(img_w, bbox_xyxy[2]))
            bbox_xyxy[3] = max(0, min(img_h, bbox_xyxy[3]))
            
            if bbox_xyxy[2] > bbox_xyxy[0] and bbox_xyxy[3] > bbox_xyxy[1]:
                boxes_list.append(bbox_xyxy)
                labels_list.append(ann['category_id'])
        
        boxes_array = np.array(boxes_list, dtype=np.float32) if boxes_list else np.zeros((0, 4), dtype=np.float32)
        
        if self.enable_augmentation:
            processed_rgb, boxes_array = self._apply_augmentation(
                processed_rgb, boxes_array, self.translate_percent,
                self.scale_min, self.scale_max, self.rotate_degrees
            )
            labels_list = labels_list[:len(boxes_array)]
        
        boxes_tensor = torch.from_numpy(boxes_array) if len(boxes_array) > 0 else torch.zeros((0, 4), dtype=torch.float32)
        labels_tensor = torch.tensor(labels_list, dtype=torch.long) if labels_list else torch.zeros((0,), dtype=torch.long)
        
        image_tensor = self.transform(Image.fromarray(processed_rgb))
        return image_tensor, {'boxes': boxes_tensor, 'labels': labels_tensor}

# ============================================================================
# 批次預處理函數
# ============================================================================

def batch_preprocess_images(input_dir: str, output_dir: str, annotation_file: Optional[str] = None) -> None:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        raise FileNotFoundError(f"輸入目錄不存在: {input_dir}")
    
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(input_path.glob(f"*{ext}"))
        image_files.extend(input_path.glob(f"*{ext.upper()}"))
    image_files = sorted(image_files)
    
    if not image_files:
        logger.warning(f"在 {input_dir} 中未找到圖片檔案")
        return
    
    logger.info(f"開始批次預處理 {len(image_files)} 張圖片...")
    
    success_count = fail_count = 0
    for idx, image_file in enumerate(image_files, 1):
        try:
            logger.info(f"[{idx}/{len(image_files)}] 處理: {image_file.name}")
            processed_gray = FasterRCNNDataset._preprocess_image(image_file)
            cv2.imwrite(str(output_path / image_file.name), processed_gray, [cv2.IMWRITE_JPEG_QUALITY, 95])
            success_count += 1
        except Exception as e:
            fail_count += 1
            logger.error(f"  ✗ 處理 {image_file.name} 時發生錯誤: {e}")
    
    logger.info(f"批次預處理完成: 成功 {success_count} 張, 失敗 {fail_count} 張")
    
    if annotation_file and Path(annotation_file).exists():
        import shutil
        shutil.copy2(annotation_file, output_path.parent / Path(annotation_file).name)
        logger.info("已複製標註檔案（座標保持不變）")

# ============================================================================
# 數據加載工具函數
# ============================================================================

def collate_fn(batch: List[Tuple[torch.Tensor, Dict]]) -> Tuple[List[torch.Tensor], List[Dict]]:
    return [item[0] for item in batch], [item[1] for item in batch]

def create_data_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: Optional[int] = None,
    pin_memory: Optional[bool] = None,
    prefetch_factor: Optional[int] = None
) -> DataLoader:
    num_workers = num_workers or min(MAX_NUM_WORKERS, os.cpu_count() or 1)
    pin_memory = pin_memory if pin_memory is not None else torch.cuda.is_available()
    prefetch_factor = prefetch_factor if prefetch_factor is not None else (4 if num_workers > 0 else None)
    
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=num_workers > 0, prefetch_factor=prefetch_factor
    )

# ============================================================================
# 訓練器類別
# ============================================================================

class TrainingConfig:
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
    def __init__(
        self,
        train_image_dir: str = "model/train/images",
        train_annotation_file: str = "model/train/annotations.json",
        val_image_dir: str = "model/val/images",
        val_annotation_file: str = "model/val/annotations.json",
        output_dir: str = "run",
        num_classes: int = 2
    ):
        
        self.train_image_dir = Path(train_image_dir)
        self.train_annotation_file = Path(train_annotation_file)
        self.val_image_dir = Path(val_image_dir)
        self.val_annotation_file = Path(val_annotation_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_classes = num_classes
        
        # 載入預訓練模型
        self.model = self._load_model(num_classes)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"GPU: {gpu_name}, 記憶體: {gpu_memory:.2f} GB, 已啟用 TF32")
        
        logger.info(f"模型已載入到設備: {self.device}")
        
        # 初始化數據集（延遲載入）
        self.train_dataset: Optional[FasterRCNNDataset] = None
        self.val_dataset: Optional[FasterRCNNDataset] = None
    
    def _load_model(self, num_classes: int) -> torch.nn.Module:
        logger.info("正在載入預訓練 Faster R-CNN 模型...")
        try:
            if USE_WEIGHTS_API:
                base_model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
            else:
                base_model = fasterrcnn_resnet50_fpn(pretrained=True)
            in_features = base_model.roi_heads.box_predictor.cls_score.in_features
            base_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            logger.info("✓ 模型載入完成")
            return base_model
        except Exception as e:
            logger.error(f"載入模型失敗: {e}")
            raise
    
    def _load_dataset(self, image_dir: Path, annotation_file: Path, dataset_name: str) -> FasterRCNNDataset:
        logger.info(f"載入{dataset_name}: {image_dir}")
        ann_file = str(annotation_file) if annotation_file.exists() else None
        dataset = FasterRCNNDataset(
            image_dir=str(image_dir), annotation_file=ann_file,
            dataset_name=dataset_name, enable_augmentation=AUGMENTATION_ENABLED
        )
        logger.info(f"  {dataset_name}大小: {len(dataset)} 張圖像")
        if AUGMENTATION_ENABLED:
            logger.info(f"  資料擴增: 已啟用")
        return dataset
    
    def prepare_datasets(self):
        logger.info("載入數據集...")
        self.train_dataset = self._load_dataset(self.train_image_dir, self.train_annotation_file, "訓練集")
        self.val_dataset = self._load_dataset(self.val_image_dir, self.val_annotation_file, "驗證集")
    
    def _save_model_and_info(self, num_epochs: int, batch_size: int, learning_rate: float) -> Path:
        model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_save_path = self.output_dir / f"model_{model_version}"
        model_save_path.mkdir(exist_ok=True)
        
        logger.info(f"正在保存模型到: {model_save_path}")
        torch.save(self.model.state_dict(), model_save_path / "model.pth")
        torch.save(self.model, model_save_path / "model_full.pth")
        
        training_info = {
            "model_version": model_version, "model_type": "Faster R-CNN",
            "num_classes": self.num_classes, "num_epochs": num_epochs,
            "batch_size": batch_size, "learning_rate": learning_rate,
            "train_samples": len(self.train_dataset), "val_samples": len(self.val_dataset),
            "save_path": str(model_save_path),
            "train_image_dir": str(self.train_image_dir), "val_image_dir": str(self.val_image_dir)
        }
        
        with open(model_save_path / "training_info.json", 'w', encoding='utf-8') as f:
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
        if config is None:
            config = TrainingConfig(
                num_epochs=num_epochs or DEFAULT_NUM_EPOCHS,
                batch_size=batch_size or DEFAULT_BATCH_SIZE,
                learning_rate=learning_rate or DEFAULT_LEARNING_RATE
            )
        
        logger.info("=" * 60)
        logger.info("開始訓練 Faster R-CNN 模型")
        logger.info("=" * 60)
        
        self.prepare_datasets()
        train_loader = create_data_loader(self.train_dataset, config.batch_size, shuffle=True, prefetch_factor=4)
        val_loader = create_data_loader(self.val_dataset, config.batch_size, shuffle=False, prefetch_factor=4)
        
        optimizer, lr_scheduler = self._create_optimizer_and_scheduler(config)
        self._log_training_config(config)
        use_amp, scaler = self._setup_amp()
        
        self._train_loop(train_loader, val_loader, optimizer, lr_scheduler, config, use_amp, scaler)
        return str(self._save_model_and_info(config.num_epochs, config.batch_size, config.learning_rate))
    
    def _create_optimizer_and_scheduler(self, config: TrainingConfig) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_step_size, gamma=config.lr_gamma)
        return optimizer, lr_scheduler
    
    def _log_training_config(self, config: TrainingConfig):
        logger.info(f"訓練參數: epochs={config.num_epochs}, batch_size={config.batch_size}, lr={config.learning_rate}")
        logger.info(f"訓練樣本: {len(self.train_dataset)}, 驗證樣本: {len(self.val_dataset)}")
        logger.info("=" * 60)
    
    def _setup_amp(self) -> Tuple[bool, Optional[GradScaler]]:
        use_amp = torch.cuda.is_available()
        if use_amp:
            scaler = GradScaler(init_scale=2.**16, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000)
            logger.info("已啟用混合精度訓練 (AMP)")
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
                images = [img.to(self.device, non_blocking=True) for img in images]
                targets = [{k: v.to(self.device, non_blocking=True) for k, v in t.items()} for t in targets]
                
                optimizer.zero_grad()
                if use_amp:
                    with autocast(device_type='cuda'):
                        losses = sum(self.model(images, targets).values())
                    scaler.scale(losses).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    losses = sum(self.model(images, targets).values())
                    losses.backward()
                    optimizer.step()
                
                train_loss += losses.item()
                train_batches += 1
                
                if (batch_idx + 1) % 10 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    logger.info(f"  Batch {batch_idx + 1}/{len(train_loader)}, Loss: {losses.item():.4f}")
            
            avg_train_loss = train_loss / train_batches if train_batches > 0 else 0.0
            logger.info(f"  平均訓練損失: {avg_train_loss:.4f}")
            
            val_loss = val_batches = 0.0
            with torch.no_grad():
                for images, targets in val_loader:
                    images = [img.to(self.device, non_blocking=True) for img in images]
                    targets = [{k: v.to(self.device, non_blocking=True) for k, v in t.items()} for t in targets]
                    
                    if use_amp:
                        with autocast(device_type='cuda'):
                            losses = sum(self.model(images, targets).values())
                    else:
                        losses = sum(self.model(images, targets).values())
                    val_loss += losses.item()
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches if val_batches > 0 else 0.0
            logger.info(f"  平均驗證損失: {avg_val_loss:.4f}")
            
            lr_scheduler.step()
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                logger.info(f"  ✓ 發現更好的模型（驗證損失: {best_val_loss:.4f}）")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.model.train()
        
        return best_val_loss

def main():
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "preprocess":
        logger.info("=" * 60)
        logger.info("批次預處理訓練集影像")
        logger.info("=" * 60)
        try:
            batch_preprocess_images("model/train/images", "model/train/images_preprocessed", "model/train/annotations.json")
            batch_preprocess_images("model/val/images", "model/val/images_preprocessed", "model/val/annotations.json")
            logger.info("\n✓ 批次預處理完成！")
        except Exception as e:
            logger.error(f"批次預處理發生錯誤: {e}", exc_info=True)
        return
    
    logger.info("=" * 60)
    logger.info("Faster R-CNN 模型訓練")
    logger.info("=" * 60)
    
    try:
        trainer = FasterRCNNTrainer(
            train_image_dir="model/train/images",
            train_annotation_file="model/train/annotations.json",
            val_image_dir="model/val/images",
            val_annotation_file="model/val/annotations.json",
            output_dir="run",
            num_classes=4
        )
        config = TrainingConfig(num_epochs=10, batch_size=8, learning_rate=0.005)
        model_path = trainer.train(config=config)
        logger.info(f"\n訓練完成！模型已保存到: {model_path}")
    except Exception as e:
        logger.error(f"訓練過程發生錯誤: {e}", exc_info=True)

if __name__ == "__main__":
    main()
