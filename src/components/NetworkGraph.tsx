import { useCNNStore, CNNLayer } from '../store/cnnStore'
import { ChevronRight } from 'lucide-react'

const layerIcons: Record<string, string> = {
  conv: '⚡',
  pool: '📉',
  fc: '🔗',
  flatten: '⬜'
}

// Auto-derive input channels for each layer
function getLayerInChannels(layers: CNNLayer[], index: number, datasetName?: string): number {
  if (index === 0) {
    if (datasetName === 'mnist') return 1
    if (datasetName === 'cifar10') return 3
    return 3
  }
  const prevLayer = layers[index - 1]
  if (prevLayer.type === 'conv') {
    return prevLayer.outChannels || 32
  }
  if (prevLayer.type === 'pool') {
    for (let i = index - 2; i >= 0; i--) {
      if (layers[i].type === 'conv') {
        return layers[i].outChannels || 32
      }
    }
  }
  return 1
}

export function NetworkGraph() {
  const { cnnConfig, datasetInfo } = useCNNStore()

  const getLayerParams = (layer: CNNLayer, index: number): string => {
    switch (layer.type) {
      case 'conv':
        const inCh = getLayerInChannels(cnnConfig.layers, index, datasetInfo?.name)
        return `${inCh}→${layer.outChannels}ch, ${layer.kernelSize}×${layer.kernelSize}`
      case 'pool':
        return `${layer.poolSize}×${layer.poolSize}, stride ${layer.stride}`
      case 'fc':
        return `${layer.units} 神经元`
      case 'flatten':
        return '3D → 1D'
      default:
        return ''
    }
  }

  return (
    <div className="panel-card">
      <div className="panel-header">
        <h3>网络结构</h3>
        <p>当前配置的卷积神经网络层级</p>
      </div>
      <div className="panel-content">
        <div className="network-layers">
          {cnnConfig.layers.map((layer, index) => (
            <div key={index} className="layer-card">
              <div className="layer-icon">{layerIcons[layer.type] || '?'}</div>
              <div className="layer-info">
                <div className="layer-name">{layer.name}</div>
                <div className="layer-detail">{getLayerParams(layer, index)}</div>
              </div>
              {index < cnnConfig.layers.length - 1 && (
                <ChevronRight size={16} style={{ color: 'var(--text-muted)' }} />
              )}
            </div>
          ))}
        </div>
        <div className="info-grid" style={{ marginTop: '1rem' }}>
          <div className="info-item">
            <span className="info-label">总层数:</span>
            <span className="info-value">{cnnConfig.layers.length}</span>
          </div>
          <div className="info-item">
            <span className="info-label">学习率:</span>
            <span className="info-value">{cnnConfig.learningRate}</span>
          </div>
        </div>
      </div>
    </div>
  )
}