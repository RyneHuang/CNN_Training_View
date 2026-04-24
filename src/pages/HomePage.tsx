import { Link } from 'react-router-dom'
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
            通过直观的数字动画，深入理解CNN的工作原理。
            观察卷积核如何计算、池化层如何降采样、
            以及训练过程中各项指标的变化。
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
          <ConvolutionScene
            inputSize={5}
            kernelSize={3}
            stride={1}
            isAnimating={true}
          />
        </div>
      </div>

      <div className="features">
        <div className="feature-card">
          <div className="feature-icon">🔍</div>
          <h3>卷积层可视化</h3>
          <p>用数字展示卷积核滑动计算的每一步，理解特征如何被提取</p>
        </div>
        <div className="feature-card">
          <div className="feature-icon">📊</div>
          <h3>池化层动画</h3>
          <p>Max Pooling的动态演示，直观理解降采样原理</p>
        </div>
        <div className="feature-card">
          <div className="feature-icon">⚡</div>
          <h3>训练过程追踪</h3>
          <p>实时观察损失曲线、准确率变化和层激活状态</p>
        </div>
      </div>

      <div className="tech-stack">
        <h2>技术栈</h2>
        <div className="tech-badges">
          <span className="badge">React</span>
          <span className="badge">Three.js</span>
          <span className="badge">PyTorch</span>
          <span className="badge">FastAPI</span>
        </div>
      </div>
    </div>
  )
}
