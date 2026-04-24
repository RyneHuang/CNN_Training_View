import { Link } from 'react-router-dom'
import { Canvas } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import ConvolutionScene from '../three/ConvolutionScene'
import './HomePage.css'

export default function HomePage() {
  return (
    <div className="home-page">
      <div className="hero">
        <div className="hero-content">
          <h1>🧠 CNN Training View</h1>
          <p className="subtitle">卷积神经网络教学可视化平台</p>
          <p className="description">
            通过交互式3D可视化，深入理解CNN的工作原理。
            观察卷积核如何在图像上滑动、池化层如何降采样、
            以及训练过程中权重如何更新。
          </p>
          <div className="cta-buttons">
            <Link to="/convolution" className="btn btn-primary">
              开始学习卷积层 →
            </Link>
            <Link to="/training" className="btn btn-secondary">
              查看训练过程
            </Link>
          </div>
        </div>
        <div className="hero-visual">
          <Canvas camera={{ position: [0, 0, 8], fov: 50 }}>
            <OrbitControls enableZoom={false} autoRotate autoRotateSpeed={0.5} />
            <ConvolutionScene inputSize={6} kernelSize={3} stride={1} isAnimating={true} />
          </Canvas>
        </div>
      </div>

      <div className="features">
        <div className="feature-card">
          <div className="feature-icon">🔍</div>
          <h3>卷积层可视化</h3>
          <p>观察卷积核在输入特征图上滑动的每一步，理解特征如何被提取</p>
        </div>
        <div className="feature-card">
          <div className="feature-icon">📊</div>
          <h3>池化层动画</h3>
          <p>Max Pooling和Average Pooling的动态演示，直观理解降采样原理</p>
        </div>
        <div className="feature-card">
          <div className="feature-icon">⚡</div>
          <h3>训练过程追踪</h3>
          <p>实时观察损失曲线、准确率变化和梯度流动</p>
        </div>
      </div>

      <div className="tech-stack">
        <h2>技术栈</h2>
        <div className="tech-badges">
          <span className="badge">React</span>
          <span className="badge">Three.js</span>
          <span className="badge">React-Three-Fiber</span>
          <span className="badge">PyTorch</span>
          <span className="badge">FastAPI</span>
        </div>
      </div>
    </div>
  )
}
