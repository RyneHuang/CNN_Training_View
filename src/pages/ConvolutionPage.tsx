import { useState } from 'react'
import ConvolutionScene from '../three/ConvolutionScene'
import './ConvolutionPage.css'

export default function ConvolutionPage() {
  const [inputSize, setInputSize] = useState(5)
  const [kernelSize, setKernelSize] = useState(3)
  const [stride, setStride] = useState(1)
  const [isAnimating, setIsAnimating] = useState(true)

  const outputSize = Math.floor((inputSize - kernelSize) / stride) + 1

  return (
    <div className="convolution-page">
      <div className="page-header">
        <h1>🔍 卷积层工作原理</h1>
        <p>卷积核在输入特征图上滑动，计算局部区域的加权和，生成输出特征图</p>
      </div>

      <div className="content-grid">
        <div className="visualization">
          <ConvolutionScene
            inputSize={inputSize}
            kernelSize={kernelSize}
            stride={stride}
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
              step="1"
              value={inputSize}
              onChange={(e) => setInputSize(Number(e.target.value))}
            />
          </div>

          <div className="control-group">
            <label>
              <span>卷积核尺寸</span>
              <span className="value">{kernelSize}×{kernelSize}</span>
            </label>
            <input
              type="range"
              min="2"
              max="4"
              step="1"
              value={kernelSize}
              onChange={(e) => setKernelSize(Number(e.target.value))}
            />
          </div>

          <div className="control-group">
            <label>
              <span>步长 (Stride)</span>
              <span className="value">{stride}</span>
            </label>
            <input
              type="range"
              min="1"
              max="2"
              step="1"
              value={stride}
              onChange={(e) => setStride(Number(e.target.value))}
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
            <h3>输出尺寸计算公式</h3>
            <code>
              输出尺寸 = ⌊(输入尺寸 - 卷积核尺寸) / 步长⌋ + 1
            </code>
            <p className="result">
              当前: ⌊({inputSize} - {kernelSize}) / {stride}⌋ + 1 = <strong>{outputSize}</strong>
            </p>
          </div>

          <div className="info-card">
            <h3>卷积运算过程</h3>
            <ol>
              <li>卷积核在输入特征图上按步长滑动</li>
              <li>对应位置元素相乘，然后求和</li>
              <li>将结果填入输出特征图对应位置</li>
              <li>重复直到遍历完整张图</li>
            </ol>
          </div>

          <div className="info-card">
            <h3>图例说明</h3>
            <div className="legend-items">
              <div className="legend-item">
                <div className="legend-color blue"></div>
                <span>输入特征图 (数字越大颜色越深)</span>
              </div>
              <div className="legend-item">
                <div className="legend-color green"></div>
                <span>卷积核正值</span>
              </div>
              <div className="legend-item">
                <div className="legend-color red"></div>
                <span>卷积核负值</span>
              </div>
              <div className="legend-item">
                <div className="legend-color purple"></div>
                <span>输出特征图</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
