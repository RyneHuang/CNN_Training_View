# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CNN Training View 是一个交互式3D可视化教学平台，帮助深度学习初学者理解卷积神经网络(CNN)的工作原理。

### Tech Stack

- **Frontend**: React 18 + TypeScript + Vite
- **3D Visualization**: Three.js + React-Three-Fiber + Drei
- **State Management**: Zustand
- **Backend**: Python FastAPI + PyTorch
- **Data**: MNIST / CIFAR-10 via torchvision

## Commands

```bash
# Install frontend dependencies
npm install

# Run frontend development server (port 3000)
npm run dev

# Run backend API server (port 8000)
npm run backend

# Build for production
npm run build

# Run both frontend and backend (concurrently)
npm run dev & npm run backend
```

## Architecture

```
CNN_Training_View/
├── src/
│   ├── components/     # React components (Layout, etc.)
│   ├── pages/           # Page components (Home, Convolution, Pooling, Training)
│   ├── three/           # Three.js 3D scene components
│   │   ├── ConvolutionScene.tsx   # 卷积动画3D场景
│   │   ├── PoolingScene.tsx       # 池化动画3D场景
│   │   └── TrainingScene.tsx      # 训练过程3D场景
│   ├── hooks/           # Custom React hooks
│   ├── store/           # Zustand state stores
│   ├── utils/           # Utility functions
│   └── types/           # TypeScript type definitions
├── backend/
│   └── main.py          # FastAPI server with PyTorch CNN model
└── public/              # Static assets
```

## Key Features

1. **Convolution Visualization**: 观察卷积核在输入特征图上滑动，计算局部加权和的过程
2. **Pooling Visualization**: Max Pooling降采样的动态演示，高亮显示最大值位置
3. **Training Monitoring**: 实时显示损失曲线、准确率变化和网络层激活状态

## API Endpoints

- `POST /api/train` - 执行一个训练epoch，返回loss/accuracy/layer activations
- `GET /api/model/summary` - 获取模型结构信息

## Environment Variables

```env
# Backend runs on port 8000
# Frontend dev server proxies /api to backend
```
