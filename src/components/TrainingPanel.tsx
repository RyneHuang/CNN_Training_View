import { useRef } from 'react'
import { useCNNStore, CNNLayer } from '../store/cnnStore'
import { getLayerInChannels } from '../utils/layerUtils'
import { Play, Square, RotateCcw, Loader2 } from 'lucide-react'

function safeNumber(value: string, fallback: number): number {
  const n = Number(value)
  return isNaN(n) || n <= 0 ? fallback : n
}

export function TrainingPanel() {
  const {
    trainingStatus,
    currentEpoch,
    lossHistory,
    cnnConfig,
    datasetInfo,
    setTrainingStatus,
    setCurrentEpoch,
    addTrainingHistory,
    clearTrainingHistory,
    setHasTrainedModel,
    setEpochs,
    setBatchSize,
    setLearningRate,
    setOptimizer
  } = useCNNStore()

  const abortRef = useRef<AbortController | null>(null)

  const latestEntry = lossHistory[lossHistory.length - 1]

  const startTraining = async () => {
    console.log('[TrainingPanel] Starting training...')

    const currentConfig = useCNNStore.getState().cnnConfig
    const tenantId = useCNNStore.getState().tenantId

    // Delete existing model and create new one with current config
    try {
      const deleteRes = await fetch(`/api/model/${tenantId}`, { method: 'DELETE' })
      console.log('[TrainingPanel] Delete model response:', deleteRes.status)
    } catch (e) {
      console.error('[TrainingPanel] Failed to delete model:', e)
    }

    try {
      const layers = currentConfig.layers.map((l: CNNLayer, i: number) => ({
        type: l.type,
        name: l.name,
        in_channels: l.type === 'conv' ? getLayerInChannels(currentConfig.layers, i, datasetInfo) : undefined,
        out_channels: l.outChannels,
        kernel_size: l.kernelSize,
        padding: l.padding,
        pool_size: l.poolSize,
        stride: l.stride,
        units: l.units,
        activation: l.activation
      }))

      const createRes = await fetch('/api/model/create', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          tenant_id: tenantId,
          config: {
            layers: layers,
            optimizer: currentConfig.optimizer,
            learning_rate: currentConfig.learningRate
          },
          dataset: useCNNStore.getState().datasetInfo?.name || 'mnist'
        })
      })

      if (!createRes.ok) {
        throw new Error(`Failed to create model: ${createRes.status}`)
      }
      console.log('[TrainingPanel] Model recreated with new config')
    } catch (e) {
      console.error('[TrainingPanel] Failed to recreate model:', e)
      alert('创建模型失败: ' + (e instanceof Error ? e.message : String(e)))
      setTrainingStatus('idle')
      return
    }

    // Create AbortController for this training session
    const controller = new AbortController()
    abortRef.current = controller

    setTrainingStatus('training')
    clearTrainingHistory()

    try {
      // Train epoch by epoch for real-time UI updates
      for (let epoch = 1; epoch <= cnnConfig.epochs; epoch++) {
        // Check if user stopped training
        if (useCNNStore.getState().trainingStatus !== 'training') break

        setCurrentEpoch(epoch)

        const response = await fetch('/api/train', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            tenant_id: tenantId,
            batch_size: cnnConfig.batchSize,
            learning_rate: cnnConfig.learningRate,
            dataset: useCNNStore.getState().datasetInfo?.name || 'mnist',
            epochs: 1
          }),
          signal: controller.signal
        })

        if (response.status === 409) {
          console.log('[TrainingPanel] Training already in progress')
          return
        }

        if (!response.ok) {
          throw new Error(`Training failed (${response.status})`)
        }

        const result = await response.json()

        addTrainingHistory({
          epoch: result.epoch,
          loss: result.loss,
          accuracy: result.accuracy
        })
        setHasTrainedModel(true)
      }

      // Only set completed if still in training state (user may have stopped)
      if (useCNNStore.getState().trainingStatus === 'training') {
        setTrainingStatus('completed')
      }
    } catch (error) {
      if ((error as Error).name === 'AbortError') {
        console.log('[TrainingPanel] Training aborted by user')
        return
      }
      console.error('[TrainingPanel] Training error:', error)
      if (useCNNStore.getState().trainingStatus === 'training') {
        alert('训练失败: ' + (error instanceof Error ? error.message : String(error)))
        setTrainingStatus('error')
      }
    } finally {
      abortRef.current = null
    }
  }

  const stopTraining = () => {
    if (abortRef.current) {
      abortRef.current.abort()
      abortRef.current = null
    }
    setTrainingStatus('idle')
  }

  const resetTraining = () => {
    clearTrainingHistory()
    setTrainingStatus('idle')
  }

  const progress = cnnConfig.epochs > 0 ? (currentEpoch / cnnConfig.epochs) * 100 : 0

  return (
    <div className="panel-card">
      <div className="panel-header">
        <h3>训练控制</h3>
        <p>控制神经网络的训练过程</p>
      </div>
      <div className="panel-content">
        <div className="metrics-row">
          <div className="metric-box">
            <div className="metric-label">Epoch</div>
            <div className="metric-value">{currentEpoch}/{cnnConfig.epochs}</div>
          </div>
          <div className="metric-box">
            <div className="metric-label">Loss</div>
            <div className="metric-value loss">{latestEntry?.loss.toFixed(4) || '-.----'}</div>
          </div>
          <div className="metric-box">
            <div className="metric-label">Accuracy</div>
            <div className="metric-value accuracy">{latestEntry ? `${(latestEntry.accuracy * 100).toFixed(1)}%` : '--%'}</div>
          </div>
        </div>

        <div className="status-badge" style={{ marginBottom: '1rem' }}>
          {trainingStatus === 'idle' && <span className="status-idle">空闲</span>}
          {trainingStatus === 'training' && (
            <span className="status-training">
              <Loader2 size={12} style={{ marginRight: 4, animation: 'spin 1s linear infinite' }} />
              训练中
            </span>
          )}
          {trainingStatus === 'completed' && <span className="status-completed">已完成</span>}
        </div>

        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.5rem', lineHeight: 1.4 }}>
          当前网络: {cnnConfig.layers.map(l => {
            if (l.type === 'conv') return `${l.name}(${l.outChannels}ch)`
            if (l.type === 'pool') return `${l.name}(${l.poolSize}x${l.poolSize})`
            if (l.type === 'fc') return `${l.name}(${l.units})`
            return l.name
          }).join(' → ')}
        </div>

        {trainingStatus === 'training' && (
          <div className="progress-bar">
            <div className="progress-fill" style={{ width: `${progress}%` }} />
          </div>
        )}

        <div className="section-title">训练参数</div>
        <div className="config-row">
          <span className="config-label">优化器</span>
          <select
            className="config-input"
            value={cnnConfig.optimizer}
            onChange={(e) => setOptimizer(e.target.value as 'adam' | 'adamw' | 'sgd')}
            disabled={trainingStatus === 'training'}
          >
            <option value="adam">Adam</option>
            <option value="adamw">AdamW</option>
            <option value="sgd">SGD</option>
          </select>
        </div>
        <div className="config-row">
          <span className="config-label">Epochs</span>
          <input
            type="number"
            className="config-input"
            value={cnnConfig.epochs}
            onChange={(e) => setEpochs(safeNumber(e.target.value, cnnConfig.epochs))}
            disabled={trainingStatus === 'training'}
          />
        </div>
        <div className="config-row">
          <span className="config-label">Batch Size</span>
          <input
            type="number"
            className="config-input"
            value={cnnConfig.batchSize}
            onChange={(e) => setBatchSize(safeNumber(e.target.value, cnnConfig.batchSize))}
            disabled={trainingStatus === 'training'}
          />
        </div>
        <div className="config-row">
          <span className="config-label">Learning Rate</span>
          <input
            type="number"
            step="0.0001"
            className="config-input"
            value={cnnConfig.learningRate}
            onChange={(e) => setLearningRate(safeNumber(e.target.value, cnnConfig.learningRate))}
            disabled={trainingStatus === 'training'}
          />
        </div>

        <div className="btn-row">
          {(trainingStatus === 'idle' || trainingStatus === 'error') && (
            <button className="btn btn-primary" onClick={startTraining}>
              <Play size={14} style={{ marginRight: 4 }} />
              开始训练
            </button>
          )}
          {trainingStatus === 'training' && (
            <button className="btn btn-secondary" onClick={stopTraining}>
              <Square size={14} style={{ marginRight: 4 }} />
              停止
            </button>
          )}
          {trainingStatus === 'completed' && (
            <>
              <button className="btn btn-primary" onClick={startTraining}>
                重新训练
              </button>
              <button className="btn btn-secondary" onClick={resetTraining}>
                <RotateCcw size={14} style={{ marginRight: 4 }} />
                重置
              </button>
            </>
          )}
          {trainingStatus === 'error' && (
            <button className="btn btn-secondary" onClick={resetTraining}>
              <RotateCcw size={14} style={{ marginRight: 4 }} />
              重置
            </button>
          )}
        </div>
      </div>
    </div>
  )
}