#!/usr/bin/env python3
"""
比對 model/train/images 和 model/val/images 資料夾中相同名稱的檔案
"""

import os
from pathlib import Path

def compare_image_folders():
    # 設定路徑
    base_dir = Path(__file__).parent
    train_images_dir = base_dir / "model" / "train" / "images"
    val_images_dir = base_dir / "model" / "val" / "images"
    
    # 檢查資料夾是否存在
    if not train_images_dir.exists():
        print(f"錯誤: {train_images_dir} 不存在")
        return
    
    if not val_images_dir.exists():
        print(f"錯誤: {val_images_dir} 不存在")
        return
    
    # 取得 train/images 中的所有檔案名稱
    train_files = set()
    for file in train_images_dir.iterdir():
        if file.is_file():
            train_files.add(file.name)
    
    # 取得 val/images 中的所有檔案名稱
    val_files = set()
    for file in val_images_dir.iterdir():
        if file.is_file():
            val_files.add(file.name)
    
    # 找出相同名稱的檔案
    common_files = train_files & val_files
    
    # 輸出結果
    print(f"Train 資料夾中的檔案數量: {len(train_files)}")
    print(f"Val 資料夾中的檔案數量: {len(val_files)}")
    print(f"\n相同名稱的檔案數量: {len(common_files)}")
    
    if common_files:
        print("\n相同名稱的檔案列表:")
        print("-" * 60)
        for filename in sorted(common_files):
            print(filename)
    else:
        print("\n沒有找到相同名稱的檔案")

if __name__ == "__main__":
    compare_image_folders()
