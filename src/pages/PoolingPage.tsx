import { useState } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import PoolingScene from '../three/PoolingScene'
import './PoolingPage.css'

export default function PoolingPage() {
  const [inputSize, setInputSize] = useState(8)
  const [poolSize, setPoolSize] = useState(2)
  const [isAnimating, setIsAnimating] = useState(true)

  const outputSize = inputSize / poolSize

  return (
    <div className="pooling-page">
      <div className="page-header">
        <h1>📊 池化层可视化</h1>
        <p>池化层通过降采样减少特征图尺寸，同时保留重要特征信息</p>
      </div>

      <div className="content-grid">
        <div className="visualization">
          <Canvas camera={{ position: [0, 0, 10], fov: 50 }}>
            <OrbitControls />
            <ambientLight intensity={0.5} />
            <pointLight position={[10, 10, 10]} intensity={1} />
            <PoolingScene
              inputSize={inputSize}
              poolSize={poolSize}
              isAnimating={isAnimating}
            />
          </Canvas>
          <div className="legend">
            <div className="legend-item">
              <div className="legend-color input"></div>
              <span>输入</span>
            </div>
            <div className="legend-item">
              <div className="legend-color window"></div>
              <span>池化窗口</span>
            </div>
            <div className="legend-item">
              <div className="legend-color highlight"></div>
              <span>最大值位置</span>
            </div>
            <div className="legend-item">
              <div className="legend-color output"></div>
              <span>输出</span>
            </div>
          </div>
        </div>

        <div className="controls-panel">
          <div className="control-group">
            <label>
              <span>输入尺寸</span>
              <span className="value">{inputSize}×{inputSize}</span>
            </label>
            <input
              type="range"
              min="4"
              max="8"
              step="2"
              value={inputSize}
              onChange={(e) => setInputSize(Number(e.target.value))}
            />
          </div>

          <div className="control-group">
            <label>
              <span>池化窗口尺寸</span>
              <span className="value">{poolSize}×{poolSize}</span>
            </label>
            <input
              type="range"
              min="2"
              max="4"
              step="2"
              value={poolSize}
              onChange={(e) => setPoolSize(Number(e.target.value))}
            />
          </div>

          <div className="control-group">
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={isAnimating}
                onChange={(e) => setIsAnimating(e.target.checked)}
              />
              <span>播放动画</span>
            </label>
          </div>

          <div className="info-card">
            <h3>Max Pooling 原理</h3>
            <code>
              输出 = max(池化窗口内的所有值)
            </code>
            <p className="result">
              当前输出: <strong>{outputSize}×{outputSize}</strong>
            </p>
          </div>

          <div className="info-card">
            <h3>为什么需要池化？</h3>
            <ul>
              <li>减少特征图尺寸，降低计算量</li>
              <li>提供平移不变性</li>
              <li>提取主要特征，过滤噪声</li>
              <li>减少过拟合风险</li>
            </ul>
          </div>

          <div className="info-card">
            <h3>池化类型</h3>
            <div className="pool-types">
              <div className="pool-type">
                <strong>Max Pooling</strong>
                <span>取最大值，保留显著特征</span>
              </div>
              <div className="pool-type">
                <strong>Average Pooling</strong>
                <span>取平均值，保留背景信息</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
