import { useState, useEffect, useCallback, useRef } from 'react'
import './PoolingScene.css'

interface PoolingSceneProps {
  inputSize?: number
  poolSize?: number
  isAnimating?: boolean
}

export default function PoolingScene({
  inputSize = 8,
  poolSize = 2,
  isAnimating = true
}: PoolingSceneProps) {
  const [inputData, setInputData] = useState<number[][]>([])
  const [outputData, setOutputData] = useState<number[][]>([])
  const [currentPos, setCurrentPos] = useState<{ row: number; col: number } | null>(null)
  const [currentWindow, setCurrentWindow] = useState<number[][]>([])
  const [currentMax, setCurrentMax] = useState<{ value: number; pos: { r: number; c: number } } | null>(null)
  const [isComplete, setIsComplete] = useState(false)
  const stepCountRef = useRef(0)

  const outputSize = Math.floor(inputSize / poolSize)
  const totalSteps = outputSize * outputSize

  useEffect(() => {
    const newInput: number[][] = []
    for (let i = 0; i < inputSize; i++) {
      const row: number[] = []
      for (let j = 0; j < inputSize; j++) {
        row.push(Math.floor(Math.random() * 9) + 1)
      }
      newInput.push(row)
    }
    setInputData(newInput)
    setOutputData([])
    setCurrentPos(null)
    setCurrentWindow([])
    setCurrentMax(null)
    setIsComplete(false)
    stepCountRef.current = 0
  }, [inputSize, poolSize])

  const computeNextStep = useCallback(() => {
    if (inputData.length === 0 || stepCountRef.current >= totalSteps) return

    const row = Math.floor(stepCountRef.current / outputSize)
    const col = stepCountRef.current % outputSize

    setCurrentPos({ row, col })

    const window: number[][] = []
    let maxVal = -Infinity
    let maxR = row * poolSize
    let maxC = col * poolSize

    for (let pi = 0; pi < poolSize; pi++) {
      const wRow: number[] = []
      for (let pj = 0; pj < poolSize; pj++) {
        const val = inputData[row * poolSize + pi]?.[col * poolSize + pj] || 0
        wRow.push(val)
        if (val > maxVal) {
          maxVal = val
          maxR = row * poolSize + pi
          maxC = col * poolSize + pj
        }
      }
      window.push(wRow)
    }
    setCurrentWindow(window)
    setCurrentMax({ value: maxVal, pos: { r: maxR, c: maxC } })

    setOutputData(prev => {
      const newOutput = [...prev]
      if (!newOutput[row]) newOutput[row] = []
      newOutput[row][col] = maxVal
      return newOutput
    })

    stepCountRef.current++

    if (stepCountRef.current >= totalSteps) {
      setIsComplete(true)
    }
  }, [inputData, poolSize, outputSize, totalSteps])

  const reset = () => {
    setOutputData([])
    setCurrentPos(null)
    setCurrentWindow([])
    setCurrentMax(null)
    setIsComplete(false)
    stepCountRef.current = 0
  }

  // Auto-play animation
  useEffect(() => {
    if (!isAnimating || isComplete || inputData.length === 0) return
    const timer = setInterval(() => {
      computeNextStep()
    }, 600)
    return () => clearInterval(timer)
  }, [isAnimating, isComplete, inputData.length, computeNextStep])

  const isInWindow = (r: number, c: number) => {
    if (!currentPos) return false
    const startR = currentPos.row * poolSize
    const startC = currentPos.col * poolSize
    return (
      r >= startR &&
      r < startR + poolSize &&
      c >= startC &&
      c < startC + poolSize
    )
  }

  const isMax = (r: number, c: number) => {
    return currentMax?.pos.r === r && currentMax?.pos.c === c
  }

  const getCellColor = (value: number) => {
    const ratio = value / 9
    const r = Math.round(34 + ratio * 100)
    const g = Math.round(197 + ratio * 50)
    const b = Math.round(246 - ratio * 100)
    return `rgb(${r}, ${g}, ${b})`
  }

  const currentStep = stepCountRef.current
  const remainingSteps = totalSteps - currentStep

  return (
    <div className="pooling-scene">
      <div className="pooling-info">
        <div className="info-badge">
          Max Pooling · 第 {currentStep} / {totalSteps} 步
        </div>
        <div className="info-badge">窗口: {poolSize}×{poolSize}</div>
      </div>

      <div className="grids-container">
        <div className="grid-section">
          <h4>输入 ({inputSize}×{inputSize})</h4>
          <div className="grid input-grid" style={{ gridTemplateColumns: `repeat(${inputSize}, 32px)` }}>
            {inputData.map((row, i) =>
              row.map((val, j) => (
                <div
                  key={`${i}-${j}`}
                  className={`cell ${isInWindow(i, j) ? 'highlighted' : ''} ${isMax(i, j) ? 'max' : ''}`}
                  style={{ backgroundColor: isMax(i, j) ? '#fbbf24' : getCellColor(val) }}
                >
                  {val}
                </div>
              ))
            )}
          </div>
        </div>

        <div className="calculation-section">
          <h4>当前窗口</h4>
          {currentWindow.length > 0 ? (
            <div className="calculation-box">
              <div className="window-display">
                {currentWindow.map((row, i) => (
                  <div key={i} className="window-row">
                    {row.map((val, j) => (
                      <span
                        key={j}
                        className={`window-val ${isMax(currentPos!.row * poolSize + i, currentPos!.col * poolSize + j) ? 'max' : ''}`}
                      >
                        {val}
                      </span>
                    ))}
                  </div>
                ))}
              </div>
              <div className="max-result">
                <span className="max-label">Max:</span>
                <span className="max-value">{currentMax?.value}</span>
              </div>
            </div>
          ) : (
            <div className="calculation-placeholder">
              {isComplete ? '计算完成！' : '点击"下一步"开始'}
            </div>
          )}
        </div>

        <div className="grid-section">
          <h4>输出 ({outputSize}×{outputSize})</h4>
          <div className="grid output-grid" style={{ gridTemplateColumns: `repeat(${outputSize}, 40px)` }}>
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
