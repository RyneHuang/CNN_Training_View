import { useState, useEffect, useCallback } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import TrainingScene from '../three/TrainingScene'
import axios from 'axios'
import './TrainingPage.css'

interface TrainingState {
  epoch: number
  loss: number
  accuracy: number
  phase: 'idle' | 'training' | 'completed'
  layerActivations: number[]
  lossHistory: number[]
  accuracyHistory: number[]
}

export default function TrainingPage() {
  const [trainingState, setTrainingState] = useState<TrainingState>({
    epoch: 0,
    loss: 0,
    accuracy: 0,
    phase: 'idle',
    layerActivations: [1, 0.8, 0.6, 0.4, 0.2],
    lossHistory: [],
    accuracyHistory: []
  })

  const [config, setConfig] = useState({
    epochs: 10,
    batchSize: 32,
    learningRate: 0.001,
    dataset: 'mnist'
  })

  const startTraining = useCallback(async () => {
    setTrainingState(prev => ({ ...prev, phase: 'training' }))

    for (let epoch = 1; epoch <= config.epochs; epoch++) {
      try {
        const response = await axios.post('/api/train', {
          epoch,
          batch_size: config.batchSize,
          learning_rate: config.learningRate,
          dataset: config.dataset
        })

        const { loss, accuracy, layer_activations } = response.data

        setTrainingState(prev => ({
          ...prev,
          epoch,
          loss,
          accuracy,
          layerActivations: layer_activations || prev.layerActivations.map(v => Math.max(0.1, v + (Math.random() - 0.5) * 0.2)),
          lossHistory: [...prev.lossHistory, loss],
          accuracyHistory: [...prev.accuracyHistory, accuracy]
        }))

        await new Promise(resolve => setTimeout(resolve, 500))
      } catch (error) {
        const mockLoss = Math.max(0.1, 2.5 - epoch * 0.2 + (Math.random() - 0.5) * 0.1)
        const mockAccuracy = Math.min(0.99, 0.3 + epoch * 0.06 + (Math.random() - 0.5) * 0.02)

        setTrainingState(prev => ({
          ...prev,
          epoch,
          loss: mockLoss,
          accuracy: mockAccuracy,
          layerActivations: prev.layerActivations.map(v => Math.min(1, v + 0.1)),
          lossHistory: [...prev.lossHistory, mockLoss],
          accuracyHistory: [...prev.accuracyHistory, mockAccuracy]
        }))

        await new Promise(resolve => setTimeout(resolve, 500))
      }
    }

    setTrainingState(prev => ({ ...prev, phase: 'completed' }))
  }, [config])

  const resetTraining = () => {
    setTrainingState({
      epoch: 0,
      loss: 0,
      accuracy: 0,
      phase: 'idle',
      layerActivations: [1, 0.8, 0.6, 0.4, 0.2],
      lossHistory: [],
      accuracyHistory: []
    })
  }

  return (
    <div className="training-page">
      <div className="page-header">
        <h1>⚡ 训练过程可视化</h1>
        <p>实时观察模型训练过程中的损失变化、准确率提升和梯度流动</p>
      </div>

      <div className="content-grid">
        <div className="visualization">
          <Canvas camera={{ position: [0, 0, 8], fov: 50 }}>
            <OrbitControls />
            <ambientLight intensity={0.5} />
            <TrainingScene
              epoch={trainingState.epoch}
              loss={trainingState.loss}
              accuracy={trainingState.accuracy}
              layerActivations={trainingState.layerActivations}
            />
          </Canvas>
        </div>

        <div className="controls-panel">
          <div className="metrics">
            <div className="metric">
              <span className="metric-label">Epoch</span>
              <span className="metric-value">{trainingState.epoch}/{config.epochs}</span>
            </div>
            <div className="metric">
              <span className="metric-label">Loss</span>
              <span className="metric-value loss">{trainingState.loss.toFixed(4)}</span>
            </div>
            <div className="metric">
              <span className="metric-label">Accuracy</span>
              <span className="metric-value accuracy">{(trainingState.accuracy * 100).toFixed(1)}%</span>
            </div>
          </div>

          <div className="charts">
            <div className="chart">
              <h4>损失曲线</h4>
              <div className="chart-area">
                {trainingState.lossHistory.length > 0 ? (
                  <svg viewBox="0 0 200 60" className="loss-chart">
                    <polyline
                      fill="none"
                      stroke="#ef4444"
                      strokeWidth="2"
                      points={trainingState.lossHistory.map((loss, i) => {
                        const x = (i / (config.epochs - 1)) * 200
                        const y = 60 - (loss / Math.max(...trainingState.lossHistory, 1)) * 50
                        return `${x},${y}`
                      }).join(' ')}
                    />
                  </svg>
                ) : (
                  <div className="chart-placeholder">等待开始训练...</div>
                )}
              </div>
            </div>
            <div className="chart">
              <h4>准确率曲线</h4>
              <div className="chart-area">
                {trainingState.accuracyHistory.length > 0 ? (
                  <svg viewBox="0 0 200 60" className="accuracy-chart">
                    <polyline
                      fill="none"
                      stroke="#22c55e"
                      strokeWidth="2"
                      points={trainingState.accuracyHistory.map((acc, i) => {
                        const x = (i / (config.epochs - 1)) * 200
                        const y = 60 - acc * 60
                        return `${x},${y}`
                      }).join(' ')}
                    />
                  </svg>
                ) : (
                  <div className="chart-placeholder">等待开始训练...</div>
                )}
              </div>
            </div>
          </div>

          <div className="control-group">
            <label>
              <span>数据集</span>
              <select
                value={config.dataset}
                onChange={(e) => setConfig(c => ({ ...c, dataset: e.target.value }))}
                disabled={trainingState.phase === 'training'}
              >
                <option value="mnist">MNIST (手写数字)</option>
                <option value="cifar10">CIFAR-10 (物体识别)</option>
              </select>
            </label>
          </div>

          <div className="control-group">
            <label>
              <span>训练轮次</span>
              <span className="value">{config.epochs}</span>
            </label>
            <input
              type="range"
              min="5"
              max="50"
              step="5"
              value={config.epochs}
              onChange={(e) => setConfig(c => ({ ...c, epochs: Number(e.target.value) }))}
              disabled={trainingState.phase === 'training'}
            />
          </div>

          <div className="control-group">
            <label>
              <span>批量大小</span>
              <span className="value">{config.batchSize}</span>
            </label>
            <input
              type="range"
              min="8"
              max="128"
              step="8"
              value={config.batchSize}
              onChange={(e) => setConfig(c => ({ ...c, batchSize: Number(e.target.value) }))}
              disabled={trainingState.phase === 'training'}
            />
          </div>

          <div className="control-group">
            <label>
              <span>学习率</span>
              <span className="value">{config.learningRate}</span>
            </label>
            <input
              type="range"
              min="0.0001"
              max="0.01"
              step="0.0001"
              value={config.learningRate}
              onChange={(e) => setConfig(c => ({ ...c, learningRate: Number(e.target.value) }))}
              disabled={trainingState.phase === 'training'}
            />
          </div>

          <div className="button-group">
            {trainingState.phase === 'idle' && (
              <button className="btn btn-primary" onClick={startTraining}>
                开始训练
              </button>
            )}
            {trainingState.phase === 'training' && (
              <button className="btn btn-secondary" disabled>
                训练中...
              </button>
            )}
            {trainingState.phase === 'completed' && (
              <>
                <button className="btn btn-primary" onClick={startTraining}>
                  重新训练
                </button>
                <button className="btn btn-secondary" onClick={resetTraining}>
                  重置
                </button>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
