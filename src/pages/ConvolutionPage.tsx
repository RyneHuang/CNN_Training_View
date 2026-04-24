import { useState } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import ConvolutionScene from '../three/ConvolutionScene'
import './ConvolutionPage.css'

export default function ConvolutionPage() {
  const [inputSize, setInputSize] = useState(8)
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
          <Canvas camera={{ position: [0, 0, 10], fov: 50 }}>
            <OrbitControls />
            <ambientLight intensity={0.5} />
            <pointLight position={[10, 10, 10]} intensity={1} />
            <ConvolutionScene
              inputSize={inputSize}
              kernelSize={kernelSize}
              stride={stride}
              isAnimating={isAnimating}
            />
          </Canvas>
          <div className="legend">
            <div className="legend-item">
              <div className="legend-color input"></div>
              <span>输入特征图</span>
            </div>
            <div className="legend-item">
              <div className="legend-color kernel"></div>
              <span>卷积核 (滑动窗口)</span>
            </div>
            <div className="legend-item">
              <div className="legend-color output"></div>
              <span>输出特征图</span>
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
              max="12"
              step="2"
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
              max="5"
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
              当前输出: <strong>{outputSize}×{outputSize}</strong>
            </p>
          </div>

          <div className="info-card">
            <h3>卷积运算示例</h3>
            <p>
              卷积核在每个位置与输入的局部区域进行逐元素乘法，
              然后将结果相加得到一个输出值。这个过程就像用放大镜
              在图像上逐处观察，提取特征。
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
