import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from PIL import Image

class AnnotationConverter:
    """將 LabelMe 標註格式轉換為 COCO 格式。"""
    
    def __init__(self, input_dir: str = "c1", output_dir: str = "c2", category_name: str = "object"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.category_name = category_name
        
        # 確保輸出目錄存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_images_dir = self.output_dir / "images"
        self.output_images_dir.mkdir(parents=True, exist_ok=True)
        
        self.image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']
    
    def _get_image_files(self) -> List[Path]:
        """收集所有支援的圖片檔案。"""
        files = []
        for ext in self.image_extensions:
            files.extend(self.input_dir.glob(f"*{ext}"))
        return sorted(files)

    def _get_json_path(self, img_path: Path) -> Optional[Path]:
        """尋找圖片對應的 JSON 標註檔。"""
        for ext in ['.json', '.JSON']:
            json_file = self.input_dir / f"{img_path.stem}{ext}"
            if json_file.exists():
                return json_file
        return None

    def convert_labelme_bbox(self, shape: Dict, img_w: int, img_h: int) -> Optional[List[float]]:
        """將 LabelMe 的 shape 轉換為 COCO [x, y, w, h] 格式。"""
        points = shape.get('points', [])
        if len(points) < 2: return None
        
        xs = [max(0, min(img_w, p[0])) for p in points]
        ys = [max(0, min(img_h, p[1])) for p in points]
        
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        
        return [x_min, y_min, x_max - x_min, y_max - y_min]
    
    def convert_labelme_file(self, json_path: Path, img_path: Path) -> Optional[Dict]:
        """處理單一 LabelMe JSON 檔案。"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 優先使用實體圖片尺寸
            if img_path.exists():
                with Image.open(img_path) as img:
                    w, h = img.size
            else:
                w, h = data.get('imageWidth', 0), data.get('imageHeight', 0)
            
            if w == 0 or h == 0: return None
            
            annotations = []
            for shape in data.get('shapes', []):
                bbox = self.convert_labelme_bbox(shape, w, h)
                if bbox:
                    annotations.append({
                        'bbox': bbox,
                        'label': shape.get('label', self.category_name)
                    })
            
            return {'image_path': img_path, 'width': w, 'height': h, 'annotations': annotations}
        except Exception as e:
            print(f"  錯誤: 處理 {json_path.name} 失敗: {e}")
            return None

    def _collect_categories(self, image_files: List[Path]) -> Set[str]:
        """掃描所有標註檔以收集所有類別。"""
        categories = set()
        for img_file in image_files:
            json_file = self._get_json_path(img_file)
            if json_file:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    for shape in data.get('shapes', []):
                        label = shape.get('label', '').strip()
                        if label: categories.add(label)
                except: continue
        return categories or {self.category_name}

    def convert_all(self) -> Dict:
        """執行整體轉換流程。"""
        print(f"開始轉換: {self.input_dir} -> {self.output_dir}")
        
        if not self.input_dir.exists():
            return {"success": False, "converted_images": 0}
        
        image_files = self._get_image_files()
        if not image_files:
            return {"success": False, "converted_images": 0}
        
        # 建立類別映射
        sorted_cats = sorted(self._collect_categories(image_files))
        cat_to_id = {name: i + 1 for i, name in enumerate(sorted_cats)}
        coco_categories = [{"id": i + 1, "name": name, "supercategory": "none"} for i, name in enumerate(sorted_cats)]
        
        coco_images, coco_annotations = [], []
        img_id, ann_id = 1, 1
        
        for img_file in image_files:
            json_file = self._get_json_path(img_file)
            data = self.convert_labelme_file(json_file, img_file) if json_file else None
            
            if not data or not data['annotations']:
                continue
            
            # 複製圖片並記錄資訊
            shutil.copy2(img_file, self.output_images_dir / img_file.name)
            coco_images.append({
                "id": img_id, "file_name": img_file.name,
                "width": data['width'], "height": data['height']
            })
            
            for ann in data['annotations']:
                bbox = ann['bbox']
                coco_annotations.append({
                    "id": ann_id, "image_id": img_id,
                    "category_id": cat_to_id.get(ann['label'], 1),
                    "bbox": bbox, "area": bbox[2] * bbox[3], "iscrowd": 0
                })
                ann_id += 1
            img_id += 1

        # 儲存結果
        out_json = self.output_dir / "annotations.json"
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump({"images": coco_images, "annotations": coco_annotations, "categories": coco_categories}, f, indent=2, ensure_ascii=False)
        
        print(f"轉換完成: 成功轉換 {img_id-1} 張圖片")
        return {"success": True, "converted_images": img_id-1, "output_json": str(out_json)}

def main():
    converter = AnnotationConverter(input_dir="c1", output_dir="nsamples")
    result = converter.convert_all()
    
    if result["success"]:
        c1_dir = Path("c1")
        if c1_dir.exists():
            print("清理 c1/ 目錄...")
            for item in c1_dir.iterdir():
                if item.is_file(): item.unlink()
                elif item.is_dir(): shutil.rmtree(item)

if __name__ == "__main__":
    main()
