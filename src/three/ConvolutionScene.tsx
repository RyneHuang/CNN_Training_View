import { useState, useEffect, useCallback } from 'react'
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
  isAnimating = true
}: ConvolutionSceneProps) {
  const [inputData, setInputData] = useState<number[][]>([])
  const [kernel, setKernel] = useState<number[][]>([])
  const [outputData, setOutputData] = useState<number[][]>([])
  const [currentPos, setCurrentPos] = useState<{ row: number; col: number } | null>(null)
  const [currentWindow, setCurrentWindow] = useState<number[][]>([])
  const [currentSum, setCurrentSum] = useState<number | null>(null)

  const outputSize = Math.floor((inputSize - kernelSize) / stride) + 1

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
  }, [inputSize, kernelSize])

  const computeConvolution = useCallback(() => {
    if (!isAnimating || inputData.length === 0) return

    let row = 0
    let col = 0

    const step = () => {
      if (row >= outputSize) {
        setCurrentPos(null)
        setCurrentWindow([])
        setCurrentSum(null)
        return
      }

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

      setTimeout(() => {
        setOutputData(prev => {
          const newOutput = [...prev]
          if (!newOutput[row]) newOutput[row] = []
          newOutput[row][col] = sum
          return newOutput
        })

        col += stride
        if (col >= outputSize) {
          col = 0
          row += stride
        }

        setTimeout(step, 300)
      }, 400)
    }

    setTimeout(step, 300)
  }, [inputData, kernel, isAnimating, kernelSize, stride, outputSize])

  useEffect(() => {
    if (isAnimating && inputData.length > 0) {
      setOutputData([])
      setTimeout(computeConvolution, 300)
    } else {
      setCurrentPos(null)
      setCurrentWindow([])
      setCurrentSum(null)
    }
  }, [isAnimating])

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
    const ratio = (value - min) / (max - min || 1)
    const r = Math.round(34 + ratio * 200)
    const g = Math.round(197 - ratio * 100)
    const b = Math.round(246 - ratio * 150)
    return `rgb(${r}, ${g}, ${b})`
  }

  const getSum = () => {
    let s = 0
    for (let i = 0; i < inputSize; i++) {
      for (let j = 0; j < inputSize; j++) {
        s += inputData[i][j]
      }
    }
    return s
  }

  const minInput = inputData.flat?.().length ? Math.min(...inputData.flat()) : 0
  const maxInput = inputData.flat?.().length ? Math.max(...inputData.flat()) : 10

  return (
    <div className="convolution-scene">
      <div className="convolution-info">
        <div className="info-badge">卷积运算</div>
        <div className="info-badge kernel-badge">
          卷积核: {kernel.map(row => row.join(',')).join(' | ')}
        </div>
      </div>

      <div className="grids-container">
        <div className="grid-section">
          <h4>输入特征图 ({inputSize}×{inputSize})</h4>
          <div className="grid input-grid">
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
          <div className="grid kernel-grid">
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
            <div className="calculation-placeholder">等待计算...</div>
          )}
        </div>

        <div className="grid-section">
          <h4>输出特征图 ({outputSize}×{outputSize})</h4>
          <div className="grid output-grid">
            {Array(outputSize).fill(null).map((_, i) =>
              Array(outputSize).fill(null).map((_, j) => (
                <div key={`out-${i}-${j}`} className={`cell output-cell ${outputData[i]?.[j] !== undefined ? 'filled' : ''}`}>
                  {outputData[i]?.[j] !== undefined ? outputData[i][j] : '?'}
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      {!isAnimating && (
        <button className="replay-btn" onClick={() => {
          setOutputData([])
          setTimeout(computeConvolution, 100)
        }}>
          ▶ 重播动画
        </button>
      )}
    </div>
  )
}
