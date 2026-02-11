"""
類別定義檢查工具
檢查原始標註資料和訓練資料中的類別定義是否一致
"""

import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, Set, List


def collect_categories_from_source(source_dir: str) -> Dict[str, int]:
    """
    從原始標註檔案（c1/）收集所有類別名稱及其出現次數
    
    Returns:
        類別名稱 -> 出現次數的字典
    """
    source_path = Path(source_dir)
    if not source_path.exists():
        print(f"錯誤: 目錄不存在: {source_dir}")
        return {}
    
    category_counter = Counter()
    category_variations = defaultdict(set)  # 記錄類別名稱的變體
    
    json_files = list(source_path.glob("*.json")) + list(source_path.glob("*.JSON"))
    
    print(f"掃描 {len(json_files)} 個原始標註檔案...")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            shapes = data.get('shapes', [])
            for shape in shapes:
                label = shape.get('label', '').strip()
                if label:
                    category_counter[label] += 1
                    # 記錄原始大小寫變體
                    category_variations[label.lower()].add(label)
        except Exception as e:
            print(f"  警告: 無法讀取 {json_file.name}: {e}")
    
    print(f"\n原始標註中的類別統計:")
    print(f"  總類別數（含大小寫變體）: {len(category_counter)}")
    print(f"  唯一類別數（不區分大小寫）: {len(category_variations)}")
    print()
    
    # 顯示所有類別（含大小寫變體）
    print("類別名稱（含大小寫變體）:")
    for label, count in sorted(category_counter.items()):
        print(f"  '{label}': {count} 次")
    
    # 檢查大小寫變體
    print("\n大小寫變體檢查:")
    has_variations = False
    for normalized, variations in category_variations.items():
        if len(variations) > 1:
            has_variations = True
            print(f"  警告: '{normalized}' 有多種大小寫寫法: {variations}")
    
    if not has_variations:
        print("  ✓ 沒有發現大小寫變體問題")
    
    return dict(category_counter)


def check_coco_categories(annotation_file: str) -> Dict[int, str]:
    """
    檢查 COCO 格式標註檔案中的類別定義
    
    Returns:
        category_id -> category_name 的字典
    """
    annotation_path = Path(annotation_file)
    if not annotation_path.exists():
        print(f"錯誤: 檔案不存在: {annotation_file}")
        return {}
    
    try:
        with open(annotation_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"錯誤: 無法讀取 {annotation_file}: {e}")
        return {}
    
    categories = {}
    for cat in data.get('categories', []):
        cat_id = cat.get('id')
        cat_name = cat.get('name', '')
        if cat_id is not None:
            categories[cat_id] = cat_name
    
    print(f"\n{annotation_file} 中的類別定義:")
    for cat_id in sorted(categories.keys()):
        print(f"  id: {cat_id} -> '{categories[cat_id]}'")
    
    return categories


def check_category_mapping(
    source_categories: Dict[str, int],
    coco_categories: Dict[int, str],
    expected_order: List[str] = None
) -> bool:
    """
    檢查類別映射是否正確
    
    Args:
        source_categories: 原始標註中的類別統計
        coco_categories: COCO 格式中的類別定義
        expected_order: 預期的類別順序（用於驗證 ID 分配）
    
    Returns:
        是否一致
    """
    print("\n" + "=" * 60)
    print("類別映射檢查")
    print("=" * 60)
    
    # 標準化原始類別名稱（轉為小寫）
    source_normalized = {name.lower(): name for name in source_categories.keys()}
    
    # 標準化 COCO 類別名稱
    coco_normalized = {cat_id: name.lower() for cat_id, name in coco_categories.items()}
    
    # 檢查類別數量
    if len(source_normalized) != len(coco_categories):
        print(f"⚠ 警告: 類別數量不一致")
        print(f"  原始標註: {len(source_normalized)} 個類別")
        print(f"  COCO 格式: {len(coco_categories)} 個類別")
    
    # 檢查每個類別是否存在
    print("\n類別對應檢查:")
    all_match = True
    
    # 按照 COCO 中的 ID 順序檢查
    for cat_id in sorted(coco_categories.keys()):
        coco_name = coco_categories[cat_id]
        coco_normalized_name = coco_name.lower()
        
        if coco_normalized_name in source_normalized:
            source_name = source_normalized[coco_normalized_name]
            count = source_categories[source_name]
            print(f"  ✓ id:{cat_id} '{coco_name}' <-> 原始標註 '{source_name}' ({count} 次)")
        else:
            print(f"  ✗ id:{cat_id} '{coco_name}' 在原始標註中找不到對應")
            all_match = False
    
    # 檢查原始標註中是否有未使用的類別
    print("\n未使用的類別檢查:")
    unused = []
    for source_normalized_name, source_name in source_normalized.items():
        found = False
        for coco_normalized_name in coco_normalized.values():
            if source_normalized_name == coco_normalized_name:
                found = True
                break
        if not found:
            unused.append(source_name)
            print(f"  ⚠ 原始標註中的 '{source_name}' 在 COCO 格式中未使用")
    
    # 檢查 ID 分配順序
    if expected_order:
        print("\nID 分配順序檢查:")
        coco_order = [coco_categories[i] for i in sorted(coco_categories.keys())]
        expected_normalized = [name.lower() for name in expected_order]
        coco_order_normalized = [name.lower() for name in coco_order]
        
        if expected_normalized == coco_order_normalized:
            print(f"  ✓ ID 分配順序正確: {coco_order}")
        else:
            print(f"  ✗ ID 分配順序不一致")
            print(f"    預期: {expected_order}")
            print(f"    實際: {coco_order}")
            all_match = False
    
    return all_match and len(unused) == 0


def analyze_annotation_distribution(annotation_file: str, coco_categories: Dict[int, str]):
    """
    分析標註中每個類別的使用情況
    """
    annotation_path = Path(annotation_file)
    if not annotation_path.exists():
        return
    
    try:
        with open(annotation_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"錯誤: 無法讀取 {annotation_file}: {e}")
        return
    
    category_counts = Counter()
    for ann in data.get('annotations', []):
        cat_id = ann.get('category_id')
        if cat_id in coco_categories:
            category_counts[cat_id] += 1
    
    print(f"\n{annotation_file} 中的類別使用統計:")
    for cat_id in sorted(category_counts.keys()):
        cat_name = coco_categories[cat_id]
        count = category_counts[cat_id]
        print(f"  id:{cat_id} '{cat_name}': {count} 個標註")


def main():
    """主函數"""
    print("=" * 60)
    print("類別定義檢查工具")
    print("=" * 60)
    print()
    
    # 1. 收集原始標註中的類別
    print("步驟 1: 檢查原始標註檔案 (c1/)")
    print("-" * 60)
    source_categories = collect_categories_from_source("c1")
    
    if not source_categories:
        print("錯誤: 無法收集原始標註類別")
        return
    
    # 2. 檢查轉換後的 COCO 格式 (c2/)
    print("\n步驟 2: 檢查轉換後的 COCO 格式 (c2/annotations.json)")
    print("-" * 60)
    c2_categories = check_coco_categories("c2/annotations.json")
    
    # 3. 檢查訓練資料 (model/train/)
    print("\n步驟 3: 檢查訓練資料 (model/train/annotations.json)")
    print("-" * 60)
    train_categories = check_coco_categories("model/train/annotations.json")
    
    # 4. 檢查驗證資料 (model/val/)
    print("\n步驟 4: 檢查驗證資料 (model/val/annotations.json)")
    print("-" * 60)
    val_categories = check_coco_categories("model/val/annotations.json")
    
    # 5. 比對類別映射
    print("\n步驟 5: 比對類別映射")
    print("-" * 60)
    
    # 預期的類別順序（按照字母順序，RFID 因為大寫 R 會排在最前面）
    expected_order = ["RFID", "colony", "point"]
    
    print("\n5.1 原始標註 vs c2/annotations.json")
    c2_match = check_category_mapping(source_categories, c2_categories, expected_order)
    
    print("\n5.2 原始標註 vs model/train/annotations.json")
    train_match = check_category_mapping(source_categories, train_categories, expected_order)
    
    print("\n5.3 原始標註 vs model/val/annotations.json")
    val_match = check_category_mapping(source_categories, val_categories, expected_order)
    
    # 6. 檢查訓練資料和驗證資料是否一致
    print("\n步驟 6: 檢查訓練資料和驗證資料的一致性")
    print("-" * 60)
    if train_categories == val_categories:
        print("  ✓ 訓練資料和驗證資料的類別定義一致")
    else:
        print("  ✗ 訓練資料和驗證資料的類別定義不一致！")
        print(f"    訓練: {train_categories}")
        print(f"    驗證: {val_categories}")
    
    # 7. 分析標註分佈
    print("\n步驟 7: 分析標註分佈")
    print("-" * 60)
    if train_categories:
        analyze_annotation_distribution("model/train/annotations.json", train_categories)
    if val_categories:
        analyze_annotation_distribution("model/val/annotations.json", val_categories)
    
    # 8. 總結
    print("\n" + "=" * 60)
    print("檢查總結")
    print("=" * 60)
    
    all_consistent = c2_match and train_match and val_match and (train_categories == val_categories)
    
    if all_consistent:
        print("✓ 所有類別定義一致")
        print("\n類別 ID 分配:")
        for cat_id in sorted(train_categories.keys()):
            print(f"  id: {cat_id} -> '{train_categories[cat_id]}'")
    else:
        print("✗ 發現類別定義不一致的問題，請檢查上述警告訊息")
        print("\n建議:")
        print("  1. 檢查原始標註檔案中的類別名稱是否一致（大小寫、拼寫）")
        print("  2. 確認 tococo.py 轉換時類別 ID 分配是否正確")
        print("  3. 確認訓練資料和驗證資料使用相同的類別定義")


if __name__ == "__main__":
    main()
