import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

# CUDA 環境設置與共同工具
from utils import setup_cuda_environment, get_torchvision_weights_api, preprocess_image

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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DetectionConfig:
    DEFAULT_NUM_CLASSES = 4
    IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp']
    CATEGORY_THRESHOLDS = {'RFID': 0.5, 'colony': 0.4, 'point': 0.48}
    
    # 影像預處理參數
    PREPROCESS_DENOISE_STRENGTH = 10
    PREPROCESS_SHARPEN_RADIUS = 5
    PREPROCESS_CONTRAST_ALPHA = 1.0
    
    # NMS 參數
    SAME_CLASS_NMS_ENABLED = True
    SAME_CLASS_NMS_IOU_THRESHOLD = 0.2
    CROSS_CLASS_NMS_ENABLED = True
    CROSS_CLASS_NMS_IOU_THRESHOLD = 0.3
    
    # Colony 幾何過濾
    COLONY_GEOMETRIC_FILTER_ENABLED = False
    COLONY_MIN_CIRCULARITY = 0.2
    COLONY_MIN_ASPECT_RATIO = 0.3
    COLONY_MAX_ASPECT_RATIO = 3.0
    COLONY_MIN_COMPACTNESS = 0.2
    
    # 繪圖
    DRAW_SHOW_SCORES = True
    DRAW_LINE_WIDTH = 2
    DRAW_FONT_SIZE = 16
    DRAW_CATEGORY_COLORS = {
        'RFID': (170, 255, 255),
        'colony': (255, 255, 170),
        'point': (255, 170, 255),
    }
    DRAW_DEFAULT_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

# 向後兼容
DEFAULT_NUM_CLASSES = DetectionConfig.DEFAULT_NUM_CLASSES
CATEGORY_THRESHOLDS = DetectionConfig.CATEGORY_THRESHOLDS
IMAGE_EXTENSIONS = DetectionConfig.IMAGE_EXTENSIONS

class FasterRCNNDetector:
    def __init__(self, model_path: Optional[str] = None, num_classes: int = DEFAULT_NUM_CLASSES, device: Optional[str] = None):
        self.num_classes = num_classes
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # RTX 5090 優化
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        self.model_path = Path(model_path if model_path else self._find_latest_model())
        if not self.model_path: raise FileNotFoundError("找不到模型檔案")
        
        self.model = self._load_model()
        self.model.to(self.device).eval()
        self.category_names = self._load_category_names()
        logger.info(f"✓ 模型載入完成: {self.model_path}")
    
    def _find_latest_model(self) -> Optional[str]:
        run_dir = Path("run")
        if not run_dir.exists(): return None
        model_dirs = sorted([d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("model_")], key=lambda x: x.name, reverse=True)
        for d in model_dirs:
            if (d / "model.pth").exists() or (d / "model_full.pth").exists(): return str(d)
        return None
    
    def _load_model(self) -> torch.nn.Module:
        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT if USE_WEIGHTS_API else None, pretrained=not USE_WEIGHTS_API)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        
        w_path = self.model_path / "model_full.pth"
        if not w_path.exists(): w_path = self.model_path / "model.pth"
        
        if "full" in w_path.name:
            return torch.load(w_path, map_location=self.device, weights_only=False)
        model.load_state_dict(torch.load(w_path, map_location=self.device, weights_only=False))
        return model

    def _load_category_names(self) -> Dict[int, str]:
        names = {0: "background"}
        # 尋找 annotations.json
        for p in [self.model_path / "training_info.json", Path("model/train/annotations.json"), Path("c2/annotations.json")]:
            if p.name == "training_info.json" and p.exists():
                with open(p, 'r') as f:
                    train_dir = json.load(f).get('train_image_dir', '')
                    p = Path(train_dir).parent / "annotations.json"
            if p.exists():
                with open(p, 'r') as f:
                    data = json.load(f)
                    for cat in data.get('categories', []):
                        names[cat['id']] = cat['name']
                break
        return names if len(names) > 1 else {0: "background", 1: "class_1", 2: "class_2", 3: "class_3"}

    @staticmethod
    def _preprocess_image(img: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """調用共通影像預處理。"""
        processed = preprocess_image(img, DetectionConfig.PREPROCESS_DENOISE_STRENGTH, 
                                    DetectionConfig.PREPROCESS_SHARPEN_RADIUS, 
                                    DetectionConfig.PREPROCESS_CONTRAST_ALPHA)
        return cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB), (0, 0)

    def _filter_by_thresholds(self, scores, labels, thresholds):
        """依類別閾值篩選。"""
        keep = torch.zeros_like(scores, dtype=torch.bool)
        for lid, name in self.category_names.items():
            if lid == 0 or name not in thresholds: continue
            mask = labels == lid
            keep[mask] = scores[mask] >= thresholds[name]
        return keep

    def _limit_points(self, boxes, labels, scores):
        """限制 Point 數量並過濾低信心度者。"""
        pid = next((i for i, n in self.category_names.items() if n == 'point'), None)
        if pid is None: return boxes, labels, scores
        
        p_mask = labels == pid
        if not p_mask.any(): return boxes, labels, scores
        
        p_indices = torch.where(p_mask)[0]
        p_scores = scores[p_indices]
        
        # 信心度過濾 (> 0.26) 且最多保留 2 個
        valid = p_scores > 0.26
        valid_indices = p_indices[valid]
        valid_scores = p_scores[valid]
        
        if len(valid_indices) > 2:
            top2 = valid_scores.topk(2).indices
            keep_indices = valid_indices[top2]
        else:
            keep_indices = valid_indices
            
        final_mask = torch.ones_like(labels, dtype=torch.bool)
        final_mask[p_mask] = False
        final_mask[keep_indices] = True
        return boxes[final_mask], labels[final_mask], scores[final_mask]

    def detect(self, image: torch.Tensor, threshold: Optional[float] = None, 
               category_thresholds: Optional[Dict[str, float]] = None, 
               original_image: Optional[np.ndarray] = None) -> Dict:
        """執行偵測流程。"""
        with torch.no_grad():
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                prediction = self.model([image.to(self.device)])[0]
        
        scores, labels, boxes = prediction['scores'], prediction['labels'], prediction['boxes']
        
        # 1. 閾值篩選
        thresh = category_thresholds if category_thresholds else DetectionConfig.CATEGORY_THRESHOLDS
        keep = self._filter_by_thresholds(scores, labels, thresh)
        boxes, labels, scores = boxes[keep], labels[keep], scores[keep]
        
        # 2. RFID 保留最高信心度者
        rid = next((i for i, n in self.category_names.items() if n == 'RFID'), None)
        if rid is not None and (labels == rid).any():
            r_mask = labels == rid
            best_idx = scores[r_mask].argmax()
            keep_r = torch.where(r_mask)[0][best_idx]
            final_r = torch.ones_like(labels, dtype=torch.bool)
            final_r[r_mask] = False
            final_r[keep_r] = True
            boxes, labels, scores = boxes[final_r], labels[final_r], scores[final_r]

        # 3. NMS (同類別與跨類別)
        if len(boxes) > 1:
            indices = nms(boxes, scores, DetectionConfig.CROSS_CLASS_NMS_IOU_THRESHOLD)
            boxes, labels, scores = boxes[indices], labels[indices], scores[indices]

        # 4. Point 限制
        boxes, labels, scores = self._limit_points(boxes, labels, scores)
        
        return {'boxes': boxes.cpu().numpy(), 'labels': labels.cpu().numpy(), 'scores': scores.cpu().numpy()}

    def detect_from_path(self, path: str) -> Tuple[np.ndarray, Dict]:
        img_bgr = cv2.imread(path)
        if img_bgr is None: raise ValueError(f"無法讀取: {path}")
        original = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        processed, (ox, oy) = self._preprocess_image(img_bgr)
        
        image_tensor = transforms.ToTensor()(Image.fromarray(processed))
        detections = self.detect(image_tensor, original_image=original)
        
        if len(detections['boxes']) > 0:
            detections['boxes'][:, [0, 2]] += ox
            detections['boxes'][:, [1, 3]] += oy
        return original, detections

    def draw_detections(self, image: np.ndarray, detections: Dict) -> np.ndarray:
        pil_img = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_img)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 
                                    DetectionConfig.DRAW_FONT_SIZE)
        except:
            font = ImageFont.load_default()
            
        for i in range(len(detections['boxes'])):
            box, label, score = detections['boxes'][i], detections['labels'][i], detections['scores'][i]
            name = self.category_names.get(int(label), f"class_{label}")
            if name in ['RFID', 'point']: continue
            
            color = DetectionConfig.DRAW_CATEGORY_COLORS.get(name, (255, 0, 0))
            draw.rectangle(box.tolist(), outline=color, width=DetectionConfig.DRAW_LINE_WIDTH)
            draw.text((box[0], box[1]-18), f"{name} {score:.2f}", fill=color, font=font)
        return np.array(pil_img)

    def process_directory(self, input_dir: str = "final/input", output_dir: str = "final/output"):
        p_in, p_out = Path(input_dir), Path(output_dir)
        p_out.mkdir(parents=True, exist_ok=True)
        
        files = sorted([f for f in p_in.glob("*") if f.suffix.lower() in DetectionConfig.IMAGE_EXTENSIONS])
        results = []
        for f in files:
            try:
                img, dets = self.detect_from_path(str(f))
                marked = self.draw_detections(img, dets)
                Image.fromarray(marked).save(p_out / f.name)
                results.append({"file": f.name, "count": len(dets['boxes'])})
            except Exception as e:
                logger.error(f"處理 {f.name} 失敗: {e}")
        
        with open(p_out / "summary.json", 'w') as jf:
            json.dump(results, jf, indent=2)
        return results

def main():
    try:
        detector = FasterRCNNDetector()
        detector.process_directory()
        logger.info("✓ 批次處理完成")
    except Exception as e:
        logger.error(f"執行失敗: {e}")

if __name__ == "__main__":
    main()