# CUDA Toolkit 安裝指南

## 問題
PyTorch 無法載入，錯誤訊息：`libcufile.so.0: cannot open shared object file`

## 解決方案：安裝 CUDA Toolkit

### 方法 1：使用交互式安裝腳本（推薦）

```bash
cd /home/dssignal/coding-flow
bash install_cuda_interactive.sh
```

腳本會引導您完成整個安裝過程，包括：
1. 安裝 CUDA repository keyring
2. 更新 apt 套件列表
3. 安裝 CUDA Toolkit 12.8/12.6
4. 更新系統庫緩存
5. 驗證安裝

### 方法 2：手動執行安裝命令

```bash
cd /home/dssignal/coding-flow

# 1. 安裝 keyring（如果尚未下載）
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb

# 2. 更新 apt
sudo apt-get update

# 3. 安裝 CUDA Toolkit（優先 12.8，不可用則 12.6）
sudo apt-get install -y cuda-toolkit-12-8 || sudo apt-get install -y cuda-toolkit-12-6

# 4. 更新庫緩存
sudo ldconfig

# 5. 驗證安裝
ldconfig -p | grep libcufile
```

### 方法 3：如果 apt 安裝失敗，使用 NVIDIA 官方安裝器

```bash
# 下載 CUDA 12.8 安裝器
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_560.27.03_linux.run

# 執行安裝（需要 sudo）
sudo sh cuda_12.8.0_560.27.03_linux.run
```

## 安裝後驗證

```bash
# 檢查 libcufile
ldconfig -p | grep libcufile

# 檢查 nvcc
nvcc --version

# 測試 PyTorch
cd /home/dssignal/coding-flow
poetry run python mark.py
```

## 如果仍然無法找到 libcufile

如果安裝後仍然找不到 `libcufile.so.0`，請設置環境變量：

```bash
# 臨時設置（當前終端）
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# 永久設置（添加到 ~/.bashrc）
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

## 注意事項

- 安裝過程可能需要 5-10 分鐘
- 需要約 3-5 GB 磁碟空間
- 安裝後建議重啟終端或執行 `source ~/.bashrc`
