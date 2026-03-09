
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
from utils import setup_cuda_environment, get_torchvision_weights_api, preprocess_image

setup_cuda_environment()

import torch
import torch.nn as nn
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

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
DEFAULT_BATCH_SIZE = 2  
DEFAULT_LEARNING_RATE = 0.005
DEFAULT_MOMENTUM = 0.9
DEFAULT_WEIGHT_DECAY = 0.0005
DEFAULT_LR_STEP_SIZE = 3
DEFAULT_LR_GAMMA = 0.1
MAX_NUM_WORKERS = 8  

# 數據集配置
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp']

# 預處理配置
PREPROCESS_DENOISE_STRENGTH = 10  
PREPROCESS_SHARPEN_RADIUS = 5    
PREPROCESS_CONTRAST_ALPHA = 1.0   

# 資料擴增配置
AUGMENTATION_ENABLED = True  
AUGMENTATION_TRANSLATE_PERCENT = 0.05  
AUGMENTATION_SCALE_MIN = 0.7  
AUGMENTATION_SCALE_MAX = 1.1  
AUGMENTATION_ROTATE_DEGREES = 180  
AUGMENTATION_FLIP_HORIZONTAL_PROB = 0.5  
AUGMENTATION_FLIP_VERTICAL_PROB = 0.5  

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
        rotate_degrees: float = AUGMENTATION_ROTATE_DEGREES,
        flip_horizontal_prob: float = AUGMENTATION_FLIP_HORIZONTAL_PROB,
        flip_vertical_prob: float = AUGMENTATION_FLIP_VERTICAL_PROB,
        negative_sample_dir: Optional[str] = "c2"  
    ):
        
        self.image_dir = Path(image_dir)
        if not self.image_dir.exists():
            logger.warning(f"圖像目錄不存在: {self.image_dir}")
        
        self.dataset_name = dataset_name
        self.enable_augmentation = enable_augmentation if enable_augmentation is not None else AUGMENTATION_ENABLED
        self.translate_percent = translate_percent
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.rotate_degrees = rotate_degrees
        self.flip_horizontal_prob = flip_horizontal_prob
        self.flip_vertical_prob = flip_vertical_prob
        
        self.transform = transform or transforms.Compose([
            transforms.ToTensor()
        ])
        
        self.image_files = self._load_image_files()
        if not self.image_files:
            raise ValueError(f"在 {self.image_dir} 中未找到圖像檔案")
        
        self.negative_sample_dir = Path(negative_sample_dir) if negative_sample_dir else None
        negative_image_files = []
        negative_annotations = {}
        if self.negative_sample_dir and self.negative_sample_dir.exists():
            negative_image_files = self._load_negative_sample_images()
            if negative_image_files:
                negative_annotations = self._load_negative_sample_annotations()
                self.image_files.extend(negative_image_files)
        
        self.negative_image_set = set(negative_image_files) if negative_image_files else set()
        
        self.annotations = {}
        self.image_id_map = {}  
        self.annotation_file = annotation_file
        if annotation_file and Path(annotation_file).exists():
            self.annotations, self.image_id_map = self._load_annotations(annotation_file)
        
        if negative_annotations:
            for filename, anns in negative_annotations.items():
                if filename not in self.annotations:
                    self.annotations[filename] = []
                for ann in anns:
                    ann['category_id'] = 0  
                    self.annotations[filename].append(ann)
        
        self.grayscale_dir = None
    
    def _load_image_files(self) -> List[str]:
        image_files = []
        for ext in self.IMAGE_EXTENSIONS:
            image_files.extend(self.image_dir.glob(f"*{ext}"))
            image_files.extend(self.image_dir.glob(f"*{ext.upper()}"))
        return sorted([str(f) for f in image_files])
    
    def _load_negative_sample_images(self) -> List[str]:
        if not self.negative_sample_dir or not self.negative_sample_dir.exists():
            return []
        image_files = []
        neg_img_dir = self.negative_sample_dir / "images"
        if not neg_img_dir.exists():
            neg_img_dir = self.negative_sample_dir
        for ext in self.IMAGE_EXTENSIONS:
            image_files.extend(neg_img_dir.glob(f"*{ext}"))
            image_files.extend(neg_img_dir.glob(f"*{ext.upper()}"))
        return sorted([str(f) for f in image_files])
    
    def _load_negative_sample_annotations(self) -> Dict[str, List[Dict]]:
        if not self.negative_sample_dir or not self.negative_sample_dir.exists():
            return {}
        annotations_dict = {}
        coco_file = self.negative_sample_dir / "annotations.json"
        if coco_file.exists():
            try:
                with open(coco_file, 'r', encoding='utf-8') as f:
                    coco_data = json.load(f)
                if 'images' in coco_data and 'annotations' in coco_data and 'categories' in coco_data:
                    category_id_to_name = {cat['id']: cat['name'] for cat in coco_data.get('categories', [])}
                    image_id_to_filename = {img['id']: img['file_name'] for img in coco_data.get('images', [])}
                    for ann in coco_data.get('annotations', []):
                        image_id = ann['image_id']
                        filename = image_id_to_filename.get(image_id)
                        if filename is None: continue
                        if filename not in annotations_dict: annotations_dict[filename] = []
                        annotations_dict[filename].append({
                            'id': ann['id'], 'image_id': image_id,
                            'category_id': ann['category_id'], 'category_name': category_id_to_name.get(ann['category_id'], ''),
                            'bbox': ann['bbox'], 'area': ann.get('area', ann['bbox'][2] * ann['bbox'][3]),
                            'iscrowd': ann.get('iscrowd', 0)
                        })
                    return annotations_dict
            except Exception as e:
                logger.warning(f"無法載入負樣本 COCO 標註 {coco_file}: {e}")
        return {}

    @staticmethod
    def _load_xml_annotations(annotation_file: str, image_dir: Path) -> Tuple[Dict, Dict]:
        annotations_dict = {}
        filename_to_image_id = {}
        path = Path(annotation_file)
        xml_files = list(path.glob("*.xml")) if path.is_dir() else [path]
        next_cat_id = 1
        cat_map = {}
        for x in xml_files:
            try:
                root = ET.parse(x).getroot()
                fname = root.findtext('filename') or (x.stem + '.jpg')
                if fname not in filename_to_image_id: filename_to_image_id[fname] = fname
                for obj in root.findall('object'):
                    name = obj.findtext('name')
                    if name not in cat_map:
                        cat_map[name] = next_cat_id
                        next_cat_id += 1
                    b = obj.find('bndbox')
                    bbox = [float(b.findtext('xmin')), float(b.findtext('ymin')), 
                            float(b.findtext('xmax')) - float(b.findtext('xmin')), 
                            float(b.findtext('ymax')) - float(b.findtext('ymin'))]
                    if fname not in annotations_dict: annotations_dict[fname] = []
                    annotations_dict[fname].append({
                        'category_id': cat_map[name], 'category_name': name,
                        'bbox': bbox, 'area': bbox[2] * bbox[3], 'iscrowd': 0
                    })
            except: continue
        return annotations_dict, filename_to_image_id

    def _load_annotations(self, annotation_file: str) -> Tuple[Dict, Dict]:
        p = Path(annotation_file)
        if p.is_dir() or p.suffix.lower() == '.xml':
            return self._load_xml_annotations(annotation_file, self.image_dir)
        try:
            with open(annotation_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            img_id_to_fname = {img['id']: img['file_name'] for img in data.get('images', [])}
            fname_to_id = {img['file_name']: img['id'] for img in data.get('images', [])}
            anns_dict = {}
            for ann in data.get('annotations', []):
                fname = img_id_to_fname.get(ann['image_id'])
                if not fname: continue
                if fname not in anns_dict: anns_dict[fname] = []
                anns_dict[fname].append({
                    'id': ann['id'], 'image_id': ann['image_id'],
                    'category_id': ann['category_id'], 'bbox': ann['bbox'],
                    'area': ann.get('area', ann['bbox'][2] * ann['bbox'][3]),
                    'iscrowd': ann.get('iscrowd', 0)
                })
            return anns_dict, fname_to_id
        except: return {}, {}

    def __len__(self) -> int: return len(self.image_files)

    @staticmethod
    def _convert_bbox_coco_to_xyxy(bbox: List[float]) -> List[float]:
        x, y, w, h = bbox
        return [x, y, x + w, y + h]

    def _get_coco_annotations_for_image(self, filename: str) -> List[Dict]:
        return self.annotations.get(filename, [])

    @staticmethod
    def _load_category_mapping(annotation_file: Optional[str], annotations_dict: Optional[Dict] = None) -> Dict[int, str]:
        mapping = {0: "background"}
        if annotation_file and Path(annotation_file).exists() and Path(annotation_file).suffix == '.json':
            try:
                with open(annotation_file, 'r', encoding='utf-8') as f:
                    for cat in json.load(f).get('categories', []):
                        mapping[cat['id']] = cat['name']
            except: pass
        return mapping

    def _preprocess_image(self, image_path: Path) -> np.ndarray:
        img = cv2.imread(str(image_path))
        if img is None: raise ValueError(f"無法讀取圖像: {image_path}")
        return preprocess_image(img, PREPROCESS_DENOISE_STRENGTH, PREPROCESS_SHARPEN_RADIUS, PREPROCESS_CONTRAST_ALPHA)

    @staticmethod
    def _apply_augmentation(image: np.ndarray, boxes: np.ndarray, 
                          translate_percent: float, scale_min: float, scale_max: float, 
                          rotate_degrees: float, flip_horizontal_prob: float = 0.5, 
                          flip_vertical_prob: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        h, w = image.shape[:2]
        if flip_horizontal_prob > np.random.random():
            image = cv2.flip(image, 1)
            if len(boxes) > 0: boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        if flip_vertical_prob > np.random.random():
            image = cv2.flip(image, 0)
            if len(boxes) > 0: boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
        return image, boxes

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        path = Path(self.image_files[idx])
        fname = path.name
        processed_gray = self._preprocess_image(path)
        processed_rgb = cv2.cvtColor(processed_gray, cv2.COLOR_GRAY2RGB)
        anns = self._get_coco_annotations_for_image(fname)
        boxes_array = []
        labels_list = []
        for ann in anns:
            b = ann['bbox']
            boxes_array.append([b[0], b[1], b[0]+b[2], b[1]+b[3]])
            labels_list.append(ann['category_id'])
        boxes_array = np.array(boxes_array, dtype=np.float32) if boxes_array else np.zeros((0, 4), dtype=np.float32)
        if self.enable_augmentation or (str(path) in self.negative_image_set):
            processed_rgb, boxes_array = self._apply_augmentation(processed_rgb, boxes_array, 
                                                               self.translate_percent, self.scale_min, 
                                                               self.scale_max, self.rotate_degrees)
        target = {
            'boxes': torch.as_tensor(boxes_array, dtype=torch.float32),
            'labels': torch.as_tensor(labels_list, dtype=torch.int64)
        }
        return self.transform(Image.fromarray(processed_rgb)), target

# ============================================================================
# 批次預處理函數
# ============================================================================

def batch_preprocess_images(
    input_dir: str,
    output_dir: str,
    annotation_file: Optional[str] = None
) -> None:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    if not input_path.exists(): raise FileNotFoundError(f"輸入目錄不存在: {input_dir}")
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(input_path.glob(f"*{ext}"))
        image_files.extend(input_path.glob(f"*{ext.upper()}"))
    image_files = sorted(image_files)
    if not image_files: return
    
    for idx, image_file in enumerate(image_files, 1):
        try:
            processed_gray = FasterRCNNDataset._preprocess_image(None, image_file)
            cv2.imwrite(str(output_path / image_file.name), processed_gray, [cv2.IMWRITE_JPEG_QUALITY, 95])
        except: continue

# ============================================================================
# 數據加載工具函數
# ============================================================================

def collate_fn(batch):
    return tuple(zip(*batch))

def create_data_loader(dataset, batch_size, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

# ============================================================================
# 訓練器類別
# ============================================================================

class TrainingConfig:
    def __init__(self, num_epochs=DEFAULT_NUM_EPOCHS, batch_size=DEFAULT_BATCH_SIZE, learning_rate=DEFAULT_LEARNING_RATE):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = DEFAULT_MOMENTUM
        self.weight_decay = DEFAULT_WEIGHT_DECAY
        self.lr_step_size = DEFAULT_LR_STEP_SIZE
        self.lr_gamma = DEFAULT_LR_GAMMA

class FasterRCNNTrainer:
    def __init__(self, train_image_dir, train_annotation_file, val_image_dir, val_annotation_file, output_dir="run", num_classes=4):
        self.train_image_dir = Path(train_image_dir)
        self.train_annotation_file = Path(train_annotation_file)
        self.val_image_dir = Path(val_image_dir)
        self.val_annotation_file = Path(val_annotation_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(num_classes).to(self.device)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def _load_model(self, num_classes):
        kw = {"weights": FasterRCNN_ResNet50_FPN_Weights.DEFAULT} if USE_WEIGHTS_API else {"pretrained": True}
        model = fasterrcnn_resnet50_fpn(**kw)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model

    def train(self, config=None):
        if config is None: config = TrainingConfig()
        train_ds = FasterRCNNDataset(self.train_image_dir, str(self.train_annotation_file), enable_augmentation=True)
        val_ds = FasterRCNNDataset(self.val_image_dir, str(self.val_annotation_file))
        train_loader = create_data_loader(train_ds, config.batch_size, shuffle=True)
        val_loader = create_data_loader(val_ds, config.batch_size, shuffle=False)
        
        optimizer = torch.optim.SGD([p for p in self.model.parameters() if p.requires_grad], 
                                  lr=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_step_size, gamma=config.lr_gamma)
        scaler = GradScaler(enabled=torch.cuda.is_available())

        for epoch in range(config.num_epochs):
            self.model.train()
            for images, targets in train_loader:
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                optimizer.zero_grad()
                with autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                    losses = sum(loss for loss in self.model(images, targets).values())
                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()
            lr_scheduler.step()
            logger.info(f"Epoch {epoch+1} 完成")
        
        ver = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = self.output_dir / f"model_{ver}"
        save_path.mkdir(exist_ok=True)
        torch.save(self.model.state_dict(), save_path / "model.pth")
        return str(save_path)

def main():
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "preprocess":
        batch_preprocess_images("model/train/images", "model/train/images_preprocessed", "model/train/annotations.json")
        batch_preprocess_images("model/val/images", "model/val/images_preprocessed", "model/val/annotations.json")
        return
    try:
        trainer = FasterRCNNTrainer("model/train/images", "model/train/annotations.json", 
                                  "model/val/images", "model/val/annotations.json")
        trainer.train()
    except Exception as e: logger.error(f"錯誤: {e}")

if __name__ == "__main__":
    main()
