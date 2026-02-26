

import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from PIL import Image

class AnnotationConverter:
    
    
    def __init__(self, input_dir: str = "c1", output_dir: str = "nsamples", category_name: str = "object"):
        
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.category_name = category_name
        
        # 創建輸出目錄
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_images_dir = self.output_dir / "images"
        self.output_images_dir.mkdir(parents=True, exist_ok=True)
        
        # 支援的圖片格式
        self.image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']
    
    def detect_format(self, json_data: Dict) -> str:
        
        # 檢查是否為 LabelMe 格式
        if 'shapes' in json_data and 'imagePath' in json_data:
            return 'labelme'
        
        # 檢查是否為 COCO 格式
        if 'images' in json_data and 'annotations' in json_data and 'categories' in json_data:
            return 'coco'
        
        # 檢查是否為單一 JSON 包含所有標註
        if 'annotations' in json_data or 'objects' in json_data:
            return 'single_json'
        
        return 'unknown'
    
    def convert_labelme_bbox(self, shape: Dict, image_width: int, image_height: int) -> Optional[List[float]]:
        
        shape_type = shape.get('shape_type', '').lower()
        points = shape.get('points', [])
        
        if shape_type == 'rectangle':
            if len(points) == 2:
                # LabelMe 矩形格式：兩個點 [x1, y1], [x2, y2]
                x1, y1 = points[0]
                x2, y2 = points[1]
                
                # 確保座標順序正確
                x_min = min(x1, x2)
                x_max = max(x1, x2)
                y_min = min(y1, y2)
                y_max = max(y1, y2)
                
                # 轉換為 COCO 格式 [x_min, y_min, width, height]
                width = x_max - x_min
                height = y_max - y_min
                
                # 確保座標在圖片範圍內
                x_min = max(0, min(x_min, image_width))
                y_min = max(0, min(y_min, image_height))
                width = max(0, min(width, image_width - x_min))
                height = max(0, min(height, image_height - y_min))
                
                return [x_min, y_min, width, height]
            elif len(points) >= 3:
                # 矩形格式但有多個點（通常是 4 個角點）：計算最小外接矩形
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                
                x_min = max(0, min(xs))
                y_min = max(0, min(ys))
                x_max = min(image_width, max(xs))
                y_max = min(image_height, max(ys))
                
                width = x_max - x_min
                height = y_max - y_min
                
                return [x_min, y_min, width, height]
        
        elif shape_type == 'polygon' and len(points) >= 3:
            # 多邊形：計算最小外接矩形
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            
            x_min = max(0, min(xs))
            y_min = max(0, min(ys))
            x_max = min(image_width, max(xs))
            y_max = min(image_height, max(ys))
            
            width = x_max - x_min
            height = y_max - y_min
            
            return [x_min, y_min, width, height]
        
        return None
    
    def convert_labelme_file(self, json_path: Path, image_path: Path) -> Optional[Dict]:
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                labelme_data = json.load(f)
            
            # 讀取圖片尺寸
            if image_path.exists():
                image = Image.open(image_path)
                image_width = image.width
                image_height = image.height
            else:
                # 從 JSON 中讀取尺寸
                image_width = labelme_data.get('imageWidth', labelme_data.get('image_width', 0))
                image_height = labelme_data.get('imageHeight', labelme_data.get('image_height', 0))
            
            if image_width == 0 or image_height == 0:
                print(f"  警告: 無法取得圖片尺寸: {json_path.name}")
                return None
            
            # 轉換標註
            annotations = []
            shapes = labelme_data.get('shapes', [])
            
            for idx, shape in enumerate(shapes):
                bbox = self.convert_labelme_bbox(shape, image_width, image_height)
                if bbox is not None:
                    annotations.append({
                        'bbox': bbox,
                        'label': shape.get('label', self.category_name)
                    })
            
            return {
                'image_path': image_path,
                'image_width': image_width,
                'image_height': image_height,
                'annotations': annotations
            }
            
        except Exception as e:
            print(f"  錯誤: 轉換 {json_path.name} 時發生錯誤: {str(e)}")
            return None
    
    def convert_single_json(self, json_path: Path) -> Optional[Dict]:
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 這裡需要根據實際格式調整
            # 假設格式為: {"images": [...], "annotations": [...]} 或類似結構
            # 如果已經是 COCO 格式，直接返回
            if self.detect_format(data) == 'coco':
                return data
            
            # 否則需要根據實際格式進行轉換
            # 這裡提供一個基本框架，可能需要根據您的實際格式調整
            print(f"  警告: 不支援的 JSON 格式: {json_path.name}")
            return None
            
        except Exception as e:
            print(f"  錯誤: 讀取 {json_path.name} 時發生錯誤: {str(e)}")
            return None
    
    def convert_all(self) -> Dict:
        
        print("=" * 60)
        print("開始轉換標註格式")
        print("=" * 60)
        print(f"輸入目錄: {self.input_dir}")
        print(f"輸出目錄: {self.output_dir}")
        print()
        
        if not self.input_dir.exists():
            print(f"錯誤: 輸入目錄不存在: {self.input_dir}")
            return {
                "success": False,
                "total_images": 0,
                "converted_images": 0,
                "total_annotations": 0
            }
        
        # 收集所有圖片和對應的 JSON 檔案
        image_files = []
        for ext in self.image_extensions:
            image_files.extend(self.input_dir.glob(f"*{ext}"))
        
        image_files = sorted(image_files)
        
        if len(image_files) == 0:
            print(f"警告: 在 {self.input_dir} 中未找到圖片檔案")
            return {
                "success": False,
                "total_images": 0,
                "converted_images": 0,
                "total_annotations": 0
            }
        
        print(f"找到 {len(image_files)} 張圖片")
        print()
        
        # 第一步：掃描所有 JSON 檔案，收集所有不同的類別名稱
        print("掃描所有標註檔案，收集類別資訊...")
        all_categories = set()
        for img_file in image_files:
            json_file = None
            for ext in ['.json', '.JSON']:
                potential_json = self.input_dir / f"{img_file.stem}{ext}"
                if potential_json.exists():
                    json_file = potential_json
                    break
            
            if json_file is not None:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        labelme_data = json.load(f)
                    
                    shapes = labelme_data.get('shapes', [])
                    for shape in shapes:
                        label = shape.get('label', '').strip()
                        if label:  # 只收集非空的類別名稱
                            all_categories.add(label)
                except Exception as e:
                    # 如果讀取失敗，繼續處理下一個檔案
                    continue
        
        # 如果沒有找到任何類別，使用預設類別
        if len(all_categories) == 0:
            all_categories = {self.category_name}
            print(f"  警告: 未找到任何類別，使用預設類別: {self.category_name}")
        else:
            print(f"  找到 {len(all_categories)} 個類別: {sorted(all_categories)}")
        
        # 建立類別映射：類別名稱 -> category_id
        sorted_categories = sorted(all_categories)  # 排序以確保一致性
        category_name_to_id = {name: idx + 1 for idx, name in enumerate(sorted_categories)}
        
        # 建立 COCO categories
        coco_categories = []
        for idx, category_name in enumerate(sorted_categories):
            coco_categories.append({
                "id": idx + 1,
                "name": category_name,
                "supercategory": "none"
            })
        
        print()
        
        # 準備 COCO 格式數據
        coco_images = []
        coco_annotations = []
        
        image_id = 1
        annotation_id = 1
        converted_count = 0
        total_annotations = 0
        
        # 處理每張圖片
        for img_file in image_files:
            # 尋找對應的 JSON 檔案
            json_file = None
            for ext in ['.json', '.JSON']:
                potential_json = self.input_dir / f"{img_file.stem}{ext}"
                if potential_json.exists():
                    json_file = potential_json
                    break
            
            # 讀取圖片
            try:
                image = Image.open(img_file).convert("RGB")
                image_width = image.width
                image_height = image.height
            except Exception as e:
                print(f"  [{image_id}/{len(image_files)}] {img_file.name}: 無法讀取圖片: {str(e)}")
                continue
            
            # 轉換標註
            converted_data = self.convert_labelme_file(json_file, img_file)
            
            if converted_data is None or len(converted_data['annotations']) == 0:
                print(f"  [{image_id}/{len(image_files)}] {img_file.name}: 無有效標註，跳過")
                continue
            
            # 複製圖片到輸出目錄
            dest_image_path = self.output_images_dir / img_file.name
            shutil.copy2(img_file, dest_image_path)
            
            # 添加圖像資訊
            coco_images.append({
                "id": image_id,
                "file_name": img_file.name,
                "width": image_width,
                "height": image_height
            })

            # 添加標註
            for ann in converted_data['annotations']:
                bbox = ann['bbox']
                area = bbox[2] * bbox[3]  # width * height
                label = ann['label']
                
                # 根據類別名稱獲取對應的 category_id
                category_id = category_name_to_id.get(label, 1)  # 如果找不到，預設為 1
                
                coco_annotations.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": bbox,  # [x_min, y_min, width, height]
                    "area": area,
                    "iscrowd": 0
                })
                annotation_id += 1
                total_annotations += 1
            converted_count += 1
            image_id += 1
        
        # 保存 COCO 格式 JSON
        coco_data = {
            "images": coco_images,
            "annotations": coco_annotations,
            "categories": coco_categories
        }
        
        output_json_path = self.output_dir / "annotations.json"
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, indent=2, ensure_ascii=False)
        
        # 輸出統計
        print()
        print("=" * 60)
        print("轉換完成")
        print("=" * 60)
        print(f"總圖片數: {len(image_files)}")
        print(f"成功轉換: {converted_count} 張")
        print()
        
        return {
            "success": True,
            "total_images": len(image_files),
            "converted_images": converted_count,
            "total_annotations": total_annotations,
            "output_json": str(output_json_path),
            "output_images_dir": str(self.output_images_dir)
        }

def main():


    
    # 創建轉換器（可以自訂類別名稱）
    converter = AnnotationConverter(
        input_dir="c1",
        output_dir="nsamples",
        category_name="object"  # 如果您的類別名稱不同，請修改這裡
    )
    
    # 執行轉換
    result = converter.convert_all()
    
    if result["success"]:
        print("✓ 轉換完成！")
        
        # 刪除 c1/ 目錄中的所有資料
        c1_dir = Path("c1")
        if c1_dir.exists():
            print(f"\n正在刪除 c1/ 目錄中的資料...")
            deleted_count = 0
            try:
                # 刪除目錄中的所有檔案和子目錄
                for item in c1_dir.iterdir():
                    if item.is_file():
                        item.unlink()
                        deleted_count += 1
                    elif item.is_dir():
                        shutil.rmtree(item)
                        deleted_count += 1
                print(f"  ✓ 已刪除 {deleted_count} 個項目")
            except Exception as e:
                print(f"  ✗ 刪除時發生錯誤: {e}")

    else:
        print("✗ 轉換失敗，請檢查錯誤訊息")

if __name__ == "__main__":
    main()
