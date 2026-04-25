import { useCNNStore } from '../store/cnnStore'
import { Play, Square, RotateCcw, Loader2 } from 'lucide-react'

// Auto-derive input channels for each layer
function getLayerInChannels(layers: any[], index: number, datasetInfo: any): number {
  if (index === 0) {
    // First layer: derive from dataset
    if (datasetInfo?.name === 'mnist') return 1
    if (datasetInfo?.name === 'cifar10') return 3
    return 3
  }
  // Subsequent layers: from previous layer's output
  const prevLayer = layers[index - 1]
  if (prevLayer.type === 'conv') {
    return prevLayer.outChannels || 32
  }
  if (prevLayer.type === 'pool') {
    // Find previous conv layer
    for (let i = index - 2; i >= 0; i--) {
      if (layers[i].type === 'conv') {
        return layers[i].outChannels || 32
      }
    }
  }
  return 1
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
    setEpochs,
    setBatchSize,
    setLearningRate,
    setOptimizer
  } = useCNNStore()

  const latestEntry = lossHistory[lossHistory.length - 1]

  const startTraining = async () => {
    console.log('[TrainingPanel] Starting training...')
    setTrainingStatus('training')
    clearTrainingHistory()
    console.log('[TrainingPanel] Training status set to training, clearing history')

    // Delete existing model and create new one with current config
    try {
      const tenantId = useCNNStore.getState().tenantId
      const currentConfig = useCNNStore.getState().cnnConfig

      // Delete old model
      await fetch(`/api/model/${tenantId}`, { method: 'DELETE' })

      // Create new model with current network config
      const layers = currentConfig.layers.map((l, i) => ({
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

      await fetch('/api/model/create', {
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
      console.log('[TrainingPanel] Model recreated with new config')
    } catch (e) {
      console.error('[TrainingPanel] Failed to recreate model:', e)
    }

    for (let epoch = 1; epoch <= cnnConfig.epochs; epoch++) {
      const currentStatus = useCNNStore.getState().trainingStatus
      if (currentStatus !== 'training') break

      setCurrentEpoch(epoch)

      try {
        console.log('[TrainingPanel] Fetching /api/train for epoch', epoch)
        const response = await fetch('/api/train', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            tenant_id: useCNNStore.getState().tenantId,
            batch_size: cnnConfig.batchSize,
            learning_rate: cnnConfig.learningRate,
            dataset: useCNNStore.getState().datasetInfo?.name || 'mnist',
            epochs: 1
          })
        })
        console.log('[TrainingPanel] Response status:', response.status)

        if (!response.ok) {
          throw new Error('Training failed')
        }

        const result = await response.json()
        console.log('[TrainingPanel] Result:', result)

        addTrainingHistory({
          epoch: result.epoch,
          loss: result.loss,
          accuracy: result.accuracy
        })
      } catch (error) {
        console.error('[TrainingPanel] Training error:', error)
        alert('训练失败: ' + (error instanceof Error ? error.message : String(error)))
        setTrainingStatus('error')
        return
      }
    }

    console.log('[TrainingPanel] Training loop completed, setting status to completed')
    setTrainingStatus('completed')
  }

  const stopTraining = () => {
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
            onChange={(e) => setEpochs(Number(e.target.value))}
            disabled={trainingStatus === 'training'}
          />
        </div>
        <div className="config-row">
          <span className="config-label">Batch Size</span>
          <input
            type="number"
            className="config-input"
            value={cnnConfig.batchSize}
            onChange={(e) => setBatchSize(Number(e.target.value))}
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
            onChange={(e) => setLearningRate(Number(e.target.value))}
            disabled={trainingStatus === 'training'}
          />
        </div>

        <div className="btn-row">
          {trainingStatus === 'idle' && (
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
        </div>
      </div>
    </div>
  )
}