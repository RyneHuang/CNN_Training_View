import { useState } from 'react'
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
          <PoolingScene
            inputSize={inputSize}
            poolSize={poolSize}
            isAnimating={isAnimating}
          />
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
            <h3>输出尺寸计算</h3>
            <code>
              输出尺寸 = 输入尺寸 / 池化窗口尺寸
            </code>
            <p className="result">
              当前: {inputSize} / {poolSize} = <strong>{outputSize}</strong>
            </p>
          </div>

          <div className="info-card">
            <h3>Max Pooling 原理</h3>
            <p>在每个池化窗口中，选择所有元素的最大值作为输出。</p>
            <div className="formula">
              <span>输出 = max(窗口内所有值)</span>
            </div>
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
            <h3>图例说明</h3>
            <div className="legend-items">
              <div className="legend-item">
                <div className="legend-color blue"></div>
                <span>输入特征图</span>
              </div>
              <div className="legend-item">
                <div className="legend-color yellow"></div>
                <span>当前窗口内的最大值</span>
              </div>
              <div className="legend-item">
                <div className="legend-color purple"></div>
                <span>输出</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
