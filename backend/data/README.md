# 数据集说明

本项目包含以下深度学习数据集，用于CNN训练演示：

## 已下载的数据集

### ✅ MNIST (63 MB)
- **描述**: 手写数字数据集，60,000训练 + 10,000测试
- **类别**: 10类（0-9）
- **图像**: 28×28 灰度图
- **状态**: 完整

### ✅ Fashion-MNIST (83 MB)
- **描述**: 时尚商品数据集，60,000训练 + 10,000测试
- **类别**: 10类（T恤、裤子、套衫等）
- **图像**: 28×28 灰度图
- **状态**: 完整

### ✅ CIFAR-10 (178 MB)
- **描述**: 彩色图像分类数据集，50,000训练 + 10,000测试
- **类别**: 10类（飞机、汽车、鸟、猫等）
- **图像**: 32×32 RGB彩色图
- **状态**: 完整

### ⚠️ CIFAR-100 (部分)
- **描述**: 细粒度彩色图像分类，50,000训练 + 10,000测试
- **类别**: 100类
- **图像**: 32×32 RGB彩色图
- **状态**: 测试集完整，训练集文件过大（148MB）需单独下载

### ❌ KMNIST (缺失)
- **描述**: 日本假名数据集，60,000训练 + 10,000测试
- **类别**: 10类
- **图像**: 28×28 灰度图
- **状态**: 下载失败（日本服务器连接超时）

## 如何手动下载大文件

由于GitHub单文件大小限制（100MB），以下文件需要单独下载：

### CIFAR-100训练集（148MB）
```bash
# 在backend/data目录运行：
wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
tar -xzf cifar-100-python.tar.gz
```

### CIFAR压缩包（可选，用于备份）
- cifar-10-python.tar.gz (163MB)
- cifar-100-python.tar.gz (161MB)

## 如何手动下载KMNIST

如果需要KMNIST数据集，请从以下链接手动下载：

1. 训练集图像: https://github.com/rois-codh/kmnist/releases/download/v1.0.0/kmnist-train-imgs-idx3-ubyte.gz
2. 训练集标签: https://github.com/rois-codh/kmnist/releases/download/v1.0.0/kmnist-train-labels-idx1-ubyte.gz
3. 测试集图像: https://github.com/rois-codh/kmnist/releases/download/v1.0.0/kmnist-t10k-imgs-idx3-ubyte.gz
4. 测试集标签: https://github.com/rois-codh/kmnist/releases/download/v1.0.0/kmnist-t10k-labels-idx1-ubyte.gz

下载后，将文件解压并放置到 `backend/data/KMNIST/raw/` 目录下。

## 数据集使用

项目启动时会自动加载已下载的数据集：
- **预加载**: MNIST, Fashion-MNIST, KMNIST（如可用）
- **按需加载**: CIFAR-10, CIFAR-100

## 离线部署

对于无网络环境的服务器部署，所有数据集文件已包含在此目录中（除KMNIST外），可以直接使用。

## 总空间占用

- MNIST: 63 MB
- Fashion-MNIST: 83 MB
- CIFAR-10: 178 MB
- CIFAR-100: 178 MB
- **总计**: ~502 MB（不含KMNIST）
