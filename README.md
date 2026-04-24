# CNN Training View

🧠 交互式卷积神经网络教学可视化平台

一个通过3D可视化帮助深度学习初学者理解CNN工作原理的项目。

## 功能特性

### 🔍 卷积层可视化
观察卷积核如何在输入特征图上滑动，理解特征提取过程。
- 可调节输入尺寸、卷积核尺寸、步长
- 动态动画展示卷积运算

### 📊 池化层动画
Max Pooling降采样的动态演示。
- 高亮显示池化窗口内最大值位置
- 对比输入输出尺寸变化

### ⚡ 训练过程追踪
实时观察模型训练过程中的指标变化。
- 损失曲线和准确率曲线
- 网络层激活状态3D可视化
- 可调节学习率、批量大小等超参数

## 技术栈

**前端:**
- React 18 + TypeScript
- Three.js + React-Three-Fiber (3D可视化)
- Zustand (状态管理)
- Vite (构建工具)

**后端:**
- Python FastAPI
- PyTorch (神经网络)
- torchvision (数据集)

## 快速开始

### 前端

```bash
# 安装依赖
npm install

# 启动开发服务器 (http://localhost:3000)
npm run dev
```

### 后端

```bash
# 安装Python依赖
cd backend
pip install -r requirements.txt

# 启动API服务器 (http://localhost:8000)
python main.py
```

## 项目结构

```
CNN_Training_View/
├── src/
│   ├── components/       # React组件
│   ├── pages/            # 页面组件
│   │   ├── HomePage.tsx       # 首页
│   │   ├── ConvolutionPage.tsx # 卷积层页面
│   │   ├── PoolingPage.tsx     # 池化层页面
│   │   └── TrainingPage.tsx    # 训练过程页面
│   ├── three/            # Three.js 3D场景
│   │   ├── ConvolutionScene.tsx
│   │   ├── PoolingScene.tsx
│   │   └── TrainingScene.tsx
│   └── ...
├── backend/
│   ├── main.py           # FastAPI应用
│   └── requirements.txt
└── ...
```

## License

MIT
