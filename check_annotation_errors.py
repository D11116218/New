"""
標註錯誤檢查工具
檢查是否有 colony 和 point 標註錯誤的情況
根據 bbox 大小來判斷可能的標註錯誤
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple


def analyze_bbox_sizes(annotation_file: str, category_mapping: Dict[int, str]) -> Dict[str, Dict]:
    """
    分析每個類別的 bbox 大小分佈
    
    Args:
        annotation_file: COCO 格式標註檔案路徑
        category_mapping: category_id -> category_name 的映射
    
    Returns:
        每個類別的統計資訊
    """
    annotation_path = Path(annotation_file)
    if not annotation_path.exists():
        return {}
    
    try:
        with open(annotation_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"錯誤: 無法讀取 {annotation_file}: {e}")
        return {}
    
    # 收集每個類別的 bbox 大小
    category_stats = defaultdict(lambda: {
        'areas': [],
        'widths': [],
        'heights': [],
        'count': 0
    })
    
    for ann in data.get('annotations', []):
        cat_id = ann.get('category_id')
        if cat_id not in category_mapping:
            continue
        
        cat_name = category_mapping[cat_id]
        bbox = ann.get('bbox', [])
        
        if len(bbox) >= 4:
            width = bbox[2]
            height = bbox[3]
            area = width * height
            
            category_stats[cat_name]['areas'].append(area)
            category_stats[cat_name]['widths'].append(width)
            category_stats[cat_name]['heights'].append(height)
            category_stats[cat_name]['count'] += 1
    
    # 計算統計資訊
    result = {}
    for cat_name, stats in category_stats.items():
        if stats['count'] > 0:
            areas = stats['areas']
            widths = stats['widths']
            heights = stats['heights']
            
            result[cat_name] = {
                'count': stats['count'],
                'area': {
                    'min': min(areas),
                    'max': max(areas),
                    'mean': sum(areas) / len(areas),
                    'median': sorted(areas)[len(areas) // 2]
                },
                'width': {
                    'min': min(widths),
                    'max': max(widths),
                    'mean': sum(widths) / len(widths),
                    'median': sorted(widths)[len(widths) // 2]
                },
                'height': {
                    'min': min(heights),
                    'max': max(heights),
                    'mean': sum(heights) / len(heights),
                    'median': sorted(heights)[len(heights) // 2]
                }
            }
    
    return result


def find_potential_errors(
    annotation_file: str,
    category_mapping: Dict[int, str],
    colony_stats: Dict,
    point_stats: Dict
) -> List[Dict]:
    """
    找出可能的標註錯誤
    
    假設：
    - colony 通常比 point 大
    - 如果一個標記為 colony 的 bbox 很小，可能是誤標為 colony 的 point
    - 如果一個標記為 point 的 bbox 很大，可能是誤標為 point 的 colony
    """
    annotation_path = Path(annotation_file)
    if not annotation_path.exists():
        return []
    
    try:
        with open(annotation_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"錯誤: 無法讀取 {annotation_file}: {e}")
        return []
    
    # 找出 colony 和 point 的 category_id
    colony_id = None
    point_id = None
    for cat_id, cat_name in category_mapping.items():
        if cat_name == 'colony':
            colony_id = cat_id
        elif cat_name == 'point':
            point_id = cat_id
    
    if colony_id is None or point_id is None:
        return []
    
    # 計算閾值（使用中位數作為參考）
    colony_area_median = colony_stats.get('area', {}).get('median', 0)
    point_area_median = point_stats.get('area', {}).get('median', 0)
    
    # 如果 colony 的中位數面積小於 point 的中位數面積，這可能表示有問題
    if colony_area_median > 0 and point_area_median > 0:
        # 使用較小的中位數作為判斷標準
        threshold = min(colony_area_median, point_area_median) * 0.5
    
    potential_errors = []
    
    # 建立 image_id -> filename 映射
    image_id_to_filename = {
        img['id']: img['file_name']
        for img in data.get('images', [])
    }
    
    for ann in data.get('annotations', []):
        cat_id = ann.get('category_id')
        image_id = ann.get('image_id')
        filename = image_id_to_filename.get(image_id, f"image_{image_id}")
        
        bbox = ann.get('bbox', [])
        if len(bbox) < 4:
            continue
        
        width = bbox[2]
        height = bbox[3]
        area = width * height
        
        # 檢查可能的錯誤
        if cat_id == colony_id:
            # colony 標記但面積很小，可能是誤標的 point
            if area < point_area_median * 1.5 and point_area_median > 0:
                potential_errors.append({
                    'type': 'colony_marked_as_point',
                    'filename': filename,
                    'image_id': image_id,
                    'annotation_id': ann.get('id'),
                    'category_id': cat_id,
                    'current_label': 'colony',
                    'suggested_label': 'point',
                    'area': area,
                    'width': width,
                    'height': height,
                    'reason': f'colony 標記但面積 {area:.2f} 小於 point 中位數 {point_area_median:.2f}'
                })
        elif cat_id == point_id:
            # point 標記但面積很大，可能是誤標的 colony
            if area > colony_area_median * 0.5 and colony_area_median > 0:
                potential_errors.append({
                    'type': 'point_marked_as_colony',
                    'filename': filename,
                    'image_id': image_id,
                    'annotation_id': ann.get('id'),
                    'category_id': cat_id,
                    'current_label': 'point',
                    'suggested_label': 'colony',
                    'area': area,
                    'width': width,
                    'height': height,
                    'reason': f'point 標記但面積 {area:.2f} 大於 colony 中位數 {colony_area_median:.2f}'
                })
    
    return potential_errors


def main():
    """主函數"""
    print("=" * 60)
    print("標註錯誤檢查工具")
    print("=" * 60)
    print()
    
    # 載入類別定義
    annotation_file = "model/train/annotations.json"
    annotation_path = Path(annotation_file)
    
    if not annotation_path.exists():
        print(f"錯誤: 檔案不存在: {annotation_file}")
        return
    
    try:
        with open(annotation_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"錯誤: 無法讀取 {annotation_file}: {e}")
        return
    
    category_mapping = {}
    for cat in data.get('categories', []):
        category_mapping[cat['id']] = cat['name']
    
    print("類別定義:")
    for cat_id in sorted(category_mapping.keys()):
        print(f"  id: {cat_id} -> '{category_mapping[cat_id]}'")
    print()
    
    # 分析 bbox 大小分佈
    print("步驟 1: 分析 bbox 大小分佈")
    print("-" * 60)
    stats = analyze_bbox_sizes(annotation_file, category_mapping)
    
    for cat_name, cat_stats in sorted(stats.items()):
        print(f"\n{cat_name} 統計 ({cat_stats['count']} 個標註):")
        print(f"  面積: 最小={cat_stats['area']['min']:.2f}, "
              f"最大={cat_stats['area']['max']:.2f}, "
              f"平均={cat_stats['area']['mean']:.2f}, "
              f"中位數={cat_stats['area']['median']:.2f}")
        print(f"  寬度: 最小={cat_stats['width']['min']:.2f}, "
              f"最大={cat_stats['width']['max']:.2f}, "
              f"平均={cat_stats['width']['mean']:.2f}, "
              f"中位數={cat_stats['width']['median']:.2f}")
        print(f"  高度: 最小={cat_stats['height']['min']:.2f}, "
              f"最大={cat_stats['height']['max']:.2f}, "
              f"平均={cat_stats['height']['mean']:.2f}, "
              f"中位數={cat_stats['height']['median']:.2f}")
    
    # 檢查可能的標註錯誤
    if 'colony' in stats and 'point' in stats:
        print("\n步驟 2: 檢查可能的標註錯誤")
        print("-" * 60)
        
        colony_stats = stats['colony']
        point_stats = stats['point']
        
        # 比較 colony 和 point 的大小
        print("\n類別大小比較:")
        colony_median = colony_stats['area']['median']
        point_median = point_stats['area']['median']
        print(f"  colony 面積中位數: {colony_median:.2f}")
        print(f"  point 面積中位數: {point_median:.2f}")
        
        if colony_median < point_median:
            print(f"  ⚠ 警告: colony 的中位數面積 ({colony_median:.2f}) 小於 point ({point_median:.2f})")
            print(f"     這可能表示有 colony 被誤標為 point，或 point 被誤標為 colony")
        else:
            print(f"  ✓ colony 通常比 point 大（符合預期）")
        
        # 找出可能的錯誤
        potential_errors = find_potential_errors(
            annotation_file,
            category_mapping,
            colony_stats,
            point_stats
        )
        
        if potential_errors:
            print(f"\n發現 {len(potential_errors)} 個可能的標註錯誤:")
            
            # 按類型分組
            errors_by_type = defaultdict(list)
            for error in potential_errors:
                errors_by_type[error['type']].append(error)
            
            for error_type, errors in errors_by_type.items():
                print(f"\n{error_type} ({len(errors)} 個):")
                for error in errors[:10]:  # 只顯示前 10 個
                    print(f"  - {error['filename']}: {error['reason']}")
                if len(errors) > 10:
                    print(f"  ... 還有 {len(errors) - 10} 個")
        else:
            print("\n✓ 未發現明顯的標註錯誤")
    else:
        print("\n⚠ 無法檢查標註錯誤：缺少 colony 或 point 類別")
    
    print("\n" + "=" * 60)
    print("檢查完成")
    print("=" * 60)
    print("\n注意: 此工具僅根據 bbox 大小進行推測，實際標註是否錯誤需要人工確認")


if __name__ == "__main__":
    main()
