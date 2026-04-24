import { useState, useEffect, useCallback } from 'react'
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

  const outputSize = inputSize / poolSize

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
  }, [inputSize, poolSize])

  const computePooling = useCallback(() => {
    if (!isAnimating || inputData.length === 0) return

    let row = 0
    let col = 0

    const step = () => {
      if (row >= outputSize) {
        setCurrentPos(null)
        setCurrentWindow([])
        setCurrentMax(null)
        return
      }

      setCurrentPos({ row, col })

      const window: number[][] = []
      let maxVal = -Infinity
      let maxR = row
      let maxC = col

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

      setTimeout(() => {
        setOutputData(prev => {
          const newOutput = [...prev]
          if (!newOutput[row]) newOutput[row] = []
          newOutput[row][col] = maxVal
          return newOutput
        })

        col++
        if (col >= outputSize) {
          col = 0
          row++
        }

        setTimeout(step, 500)
      }, 600)
    }

    setTimeout(step, 300)
  }, [inputData, isAnimating, poolSize, outputSize])

  useEffect(() => {
    if (isAnimating && inputData.length > 0) {
      setOutputData([])
      setTimeout(computePooling, 300)
    } else {
      setCurrentPos(null)
      setCurrentWindow([])
      setCurrentMax(null)
    }
  }, [isAnimating])

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

  const getCellColor = (value: number, isHighlight: boolean = false) => {
    if (isHighlight) return '#fbbf24'
    const ratio = value / 9
    const r = Math.round(34 + ratio * 100)
    const g = Math.round(197 + ratio * 50)
    const b = Math.round(246 - ratio * 100)
    return `rgb(${r}, ${g}, ${b})`
  }

  return (
    <div className="pooling-scene">
      <div className="pooling-info">
        <div className="info-badge">Max Pooling</div>
        <div className="info-badge">窗口: {poolSize}×{poolSize}</div>
      </div>

      <div className="grids-container">
        <div className="grid-section">
          <h4>输入 ({inputSize}×{inputSize})</h4>
          <div className="grid input-grid" style={{ gridTemplateColumns: `repeat(${inputSize}, 36px)` }}>
            {inputData.map((row, i) =>
              row.map((val, j) => (
                <div
                  key={`${i}-${j}`}
                  className={`cell ${isInWindow(i, j) ? 'highlighted' : ''} ${isMax(i, j) ? 'max' : ''}`}
                  style={{ backgroundColor: getCellColor(val, isMax(i, j)) }}
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
            <div className="calculation-placeholder">等待计算...</div>
          )}
        </div>

        <div className="grid-section">
          <h4>输出 ({outputSize}×{outputSize})</h4>
          <div className="grid output-grid" style={{ gridTemplateColumns: `repeat(${outputSize}, 48px)` }}>
            {Array(outputSize).fill(null).map((_, i) =>
              Array(outputSize).fill(null).map((_, j) => (
                <div
                  key={`out-${i}-${j}`}
                  className={`cell output-cell ${outputData[i]?.[j] !== undefined ? 'filled' : ''}`}
                >
                  {outputData[i]?.[j] !== undefined ? outputData[i][j] : '?'}
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      <div className="pooling-explanation">
        <p><strong>Max Pooling 原理：</strong>在每个 {poolSize}×{poolSize} 的窗口中，取最大值作为输出，实现降采样。</p>
        <p>输出尺寸 = 输入尺寸 / 池化窗口尺寸 = {inputSize} / {poolSize} = {outputSize}</p>
      </div>

      {!isAnimating && (
        <button className="replay-btn" onClick={() => {
          setOutputData([])
          setTimeout(computePooling, 100)
        }}>
          ▶ 重播动画
        </button>
      )}
    </div>
  )
}
