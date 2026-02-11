#!/bin/bash
# CUDA Toolkit 交互式安裝腳本

echo "=========================================="
echo "CUDA Toolkit 安裝腳本"
echo "=========================================="
echo ""
echo "此腳本將安裝 CUDA Toolkit 以解決 libcufile.so.0 缺失問題"
echo "需要 sudo 權限，請準備好您的密碼"
echo ""
read -p "按 Enter 繼續，或 Ctrl+C 取消..."

cd /home/dssignal/coding-flow

echo ""
echo "步驟 1: 安裝 CUDA Repository Keyring..."
if [ ! -f "cuda-keyring_1.1-1_all.deb" ]; then
    echo "下載 CUDA keyring..."
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
    echo "✓ 下載完成"
fi

echo "安裝 keyring..."
sudo dpkg -i cuda-keyring_1.1-1_all.deb || {
    echo "警告: keyring 安裝可能失敗，繼續嘗試..."
}

echo ""
echo "步驟 2: 更新 apt 套件列表..."
sudo apt-get update

echo ""
echo "步驟 3: 檢查可用的 CUDA Toolkit 版本..."
AVAILABLE_VERSIONS=$(apt-cache search cuda-toolkit 2>/dev/null | grep -E "^cuda-toolkit-[0-9]" | head -5)
if [ -z "$AVAILABLE_VERSIONS" ]; then
    echo "警告: 未找到 CUDA Toolkit，可能需要手動添加 repository"
else
    echo "$AVAILABLE_VERSIONS"
fi

echo ""
echo "步驟 4: 安裝 CUDA Toolkit..."
# 嘗試安裝不同版本
if apt-cache show cuda-toolkit-12-8 &>/dev/null 2>&1; then
    echo "安裝 cuda-toolkit-12-8..."
    sudo apt-get install -y cuda-toolkit-12-8
elif apt-cache show cuda-toolkit-12-6 &>/dev/null 2>&1; then
    echo "安裝 cuda-toolkit-12-6..."
    sudo apt-get install -y cuda-toolkit-12-6
elif apt-cache show cuda-toolkit-12-7 &>/dev/null 2>&1; then
    echo "安裝 cuda-toolkit-12-7..."
    sudo apt-get install -y cuda-toolkit-12-7
else
    echo "錯誤: 找不到可用的 CUDA Toolkit 12.x 版本"
    echo "請檢查 CUDA repository 是否正確設置"
    exit 1
fi

echo ""
echo "步驟 5: 更新系統庫緩存..."
sudo ldconfig

echo ""
echo "步驟 6: 驗證安裝..."
echo "檢查 libcufile.so.0..."
if ldconfig -p | grep -q libcufile.so.0; then
    echo "✓ libcufile.so.0 已找到"
    ldconfig -p | grep libcufile
else
    echo "警告: libcufile.so.0 未在系統庫中找到"
    echo "嘗試查找文件..."
    FOUND_LIB=$(find /usr/local/cuda* /usr/lib -name "libcufile.so*" 2>/dev/null | head -1)
    if [ -n "$FOUND_LIB" ]; then
        echo "找到: $FOUND_LIB"
        echo "請設置 LD_LIBRARY_PATH:"
        CUDA_DIR=$(dirname $(dirname "$FOUND_LIB"))
        echo "  export LD_LIBRARY_PATH=$CUDA_DIR/lib64:\$LD_LIBRARY_PATH"
    else
        echo "未找到 libcufile.so"
    fi
fi

echo ""
echo "檢查 nvcc..."
if command -v nvcc &> /dev/null; then
    echo "✓ nvcc 已安裝"
    nvcc --version | head -3
else
    echo "警告: nvcc 未找到，可能需要設置 PATH"
    echo "請執行: export PATH=/usr/local/cuda/bin:\$PATH"
fi

echo ""
echo "=========================================="
echo "安裝完成！"
echo "=========================================="
echo ""
echo "現在可以測試 PyTorch:"
echo "  cd /home/dssignal/coding-flow"
echo "  poetry run python mark.py"
echo ""
