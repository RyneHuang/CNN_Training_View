import { useState, useEffect, useCallback, useRef } from 'react'
import './ConvolutionScene.css'

interface ConvolutionSceneProps {
  inputSize?: number
  kernelSize?: number
  stride?: number
  isAnimating?: boolean
}

export default function ConvolutionScene({
  inputSize = 5,
  kernelSize = 3,
  stride = 1,
  isAnimating = false
}: ConvolutionSceneProps) {
  const [inputData, setInputData] = useState<number[][]>([])
  const [kernel, setKernel] = useState<number[][]>([])
  const [outputData, setOutputData] = useState<number[][]>([])
  const [currentPos, setCurrentPos] = useState<{ row: number; col: number } | null>(null)
  const [currentWindow, setCurrentWindow] = useState<number[][]>([])
  const [currentSum, setCurrentSum] = useState<number | null>(null)
  const [isComplete, setIsComplete] = useState(false)
  const stepCountRef = useRef(0)

  const outputSize = Math.floor((inputSize - kernelSize) / stride) + 1
  const totalSteps = outputSize * outputSize

  useEffect(() => {
    const newInput: number[][] = []
    for (let i = 0; i < inputSize; i++) {
      const row: number[] = []
      for (let j = 0; j < inputSize; j++) {
        const val = Math.round(Math.sin(i * 0.8) * Math.cos(j * 0.8) * 4 + 5)
        row.push(val)
      }
      newInput.push(row)
    }
    setInputData(newInput)

    const newKernel: number[][] = []
    for (let i = 0; i < kernelSize; i++) {
      const row: number[] = []
      for (let j = 0; j < kernelSize; j++) {
        row.push(Math.random() > 0.5 ? 1 : -1)
      }
      newKernel.push(row)
    }
    setKernel(newKernel)

    setOutputData([])
    setCurrentPos(null)
    setCurrentWindow([])
    setCurrentSum(null)
    setIsComplete(false)
    stepCountRef.current = 0
  }, [inputSize, kernelSize])

  const computeNextStep = useCallback(() => {
    if (inputData.length === 0 || outputData.length === totalSteps) return

    const row = Math.floor(stepCountRef.current / outputSize)
    const col = stepCountRef.current % outputSize

    setCurrentPos({ row, col })

    const window: number[][] = []
    for (let ki = 0; ki < kernelSize; ki++) {
      const wRow: number[] = []
      for (let kj = 0; kj < kernelSize; kj++) {
        wRow.push(inputData[row + ki]?.[col + kj] || 0)
      }
      window.push(wRow)
    }
    setCurrentWindow(window)

    let sum = 0
    for (let ki = 0; ki < kernelSize; ki++) {
      for (let kj = 0; kj < kernelSize; kj++) {
        sum += window[ki][kj] * kernel[ki][kj]
      }
    }
    setCurrentSum(sum)

    setOutputData(prev => {
      const newOutput = [...prev]
      if (!newOutput[row]) newOutput[row] = []
      newOutput[row][col] = sum
      return newOutput
    })

    stepCountRef.current++

    if (stepCountRef.current >= totalSteps) {
      setIsComplete(true)
    }
  }, [inputData, kernel, kernelSize, outputSize, totalSteps])

  const reset = () => {
    setOutputData([])
    setCurrentPos(null)
    setCurrentWindow([])
    setCurrentSum(null)
    setIsComplete(false)
    stepCountRef.current = 0
  }

  const isInWindow = (r: number, c: number) => {
    if (!currentPos) return false
    return (
      r >= currentPos.row &&
      r < currentPos.row + kernelSize &&
      c >= currentPos.col &&
      c < currentPos.col + kernelSize
    )
  }

  const getCellColor = (value: number, min: number, max: number) => {
    const ratio = (value - min) / (max || 1)
    const r = Math.round(34 + ratio * 200)
    const g = Math.round(197 - ratio * 100)
    const b = Math.round(246 - ratio * 150)
    return `rgb(${r}, ${g}, ${b})`
  }

  const minInput = inputData.flat?.().length ? Math.min(...inputData.flat()) : 0
  const maxInput = inputData.flat?.().length ? Math.max(...inputData.flat()) : 10

  const currentStep = stepCountRef.current
  const remainingSteps = totalSteps - currentStep

  return (
    <div className="convolution-scene">
      <div className="convolution-info">
        <div className="info-badge">
          卷积运算 · 第 {currentStep} / {totalSteps} 步
        </div>
        <div className="info-badge kernel-badge">
          卷积核: {kernel.map(row => row.join(',')).join(' | ')}
        </div>
      </div>

      <div className="grids-container">
        <div className="grid-section">
          <h4>输入特征图 ({inputSize}×{inputSize})</h4>
          <div className="grid input-grid" style={{ gridTemplateColumns: `repeat(${inputSize}, 36px)` }}>
            {inputData.map((row, i) =>
              row.map((val, j) => (
                <div
                  key={`${i}-${j}`}
                  className={`cell ${isInWindow(i, j) ? 'highlighted' : ''}`}
                  style={{ backgroundColor: getCellColor(val, minInput, maxInput) }}
                >
                  {val}
                </div>
              ))
            )}
          </div>
        </div>

        <div className="kernel-section">
          <h4>卷积核 ({kernelSize}×{kernelSize})</h4>
          <div className="grid kernel-grid" style={{ gridTemplateColumns: `repeat(${kernelSize}, 36px)` }}>
            {kernel.map((row, i) =>
              row.map((val, j) => (
                <div key={`${i}-${j}`} className={`cell kernel-cell ${val > 0 ? 'positive' : 'negative'}`}>
                  {val}
                </div>
              ))
            )}
          </div>
        </div>

        <div className="calculation-section">
          <h4>当前计算</h4>
          {currentWindow.length > 0 ? (
            <div className="calculation-box">
              <div className="window-display">
                <div className="window-grid">
                  {currentWindow.map((row, i) =>
                    <div key={i} className="window-row">
                      {row.map((val, j) => (
                        <span key={j} className="window-val">{val}</span>
                      ))}
                    </div>
                  )}
                </div>
                <span className="multiply">×</span>
                <div className="window-grid kernel-display">
                  {kernel.map((row, i) =>
                    <div key={i} className="window-row">
                      {row.map((val, j) => (
                        <span key={j} className={`window-val ${val > 0 ? 'positive' : 'negative'}`}>{val}</span>
                      ))}
                    </div>
                  )}
                </div>
                <span className="equals">=</span>
                <span className="result">{currentSum}</span>
              </div>
            </div>
          ) : (
            <div className="calculation-placeholder">
              {isComplete ? '计算完成！' : '点击"下一步"开始'}
            </div>
          )}
        </div>

        <div className="grid-section">
          <h4>输出特征图 ({outputSize}×{outputSize})</h4>
          <div className="grid output-grid" style={{ gridTemplateColumns: `repeat(${outputSize}, 36px)` }}>
            {Array(outputSize).fill(null).map((_, i) =>
              Array(outputSize).fill(null).map((_, j) => (
                <div
                  key={`out-${i}-${j}`}
                  className={`cell output-cell ${outputData[i]?.[j] !== undefined ? 'filled' : ''} ${currentPos?.row === i && currentPos?.col === j ? 'current' : ''}`}
                >
                  {outputData[i]?.[j] !== undefined ? outputData[i][j] : '?'}
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      <div className="controls">
        {!isComplete ? (
          <button className="step-btn" onClick={computeNextStep} disabled={inputData.length === 0}>
            ▶ 下一步 ({remainingSteps} 步剩余)
          </button>
        ) : (
          <button className="step-btn reset" onClick={reset}>
            ↻ 重置
          </button>
        )}
      </div>
    </div>
  )
}
