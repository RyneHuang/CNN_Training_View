import { useState, useEffect } from 'react'
import { useCNNStore } from '../store/cnnStore'
import { CNNLayer, DatasetInfo } from '../types'
import { Info, X, BookOpen } from 'lucide-react'

// Convert backend snake_case layer to frontend camelCase
function mapBackendLayer(l: Record<string, any>): CNNLayer {
  return {
    type: l.type,
    name: l.name,
    outChannels: l.out_channels,
    kernelSize: l.kernel_size,
    padding: l.padding,
    poolSize: l.pool_size,
    stride: l.stride,
    units: l.units,
    activation: l.activation,
  }
}

interface DatasetOption {
  name: string
  label: string
  description: string
  difficulty: string
  samples: number
  imageSize: number
  classes: number
  channels: number
}

const DATASETS: DatasetOption[] = [
  { name: 'mnist', label: 'MNIST', description: '手写数字识别', difficulty: '入门级', samples: 60000, imageSize: 28, classes: 10, channels: 1 },
  { name: 'fashion_mnist', label: 'Fashion-MNIST', description: '时尚商品分类', difficulty: '初级', samples: 60000, imageSize: 28, classes: 10, channels: 1 },
  { name: 'kmnist', label: 'KMNIST', description: '日文草书识别', difficulty: '初级-中级', samples: 60000, imageSize: 28, classes: 10, channels: 1 },
  { name: 'cifar10', label: 'CIFAR-10', description: '物体识别', difficulty: '中级', samples: 50000, imageSize: 32, classes: 10, channels: 3 },
  { name: 'cifar100', label: 'CIFAR-100', description: '细粒度物体识别', difficulty: '高级', samples: 50000, imageSize: 32, classes: 100, channels: 3 },
]

const DIFFICULTY_COLORS: Record<string, string> = {
  '入门级': '#4ade80',
  '初级': '#60a5fa',
  '初级-中级': '#818cf8',
  '中级': '#f59e0b',
  '高级': '#ef4444',
}

interface DatasetDetail {
  name: string
  full_name: string
  description: string
  difficulty: string
  paper: string
  classes: string[]
  channels: string
  recommended_model: string
  image_size: number
  input_channels: number
  num_classes: number
  train_samples: number
  test_samples: number
}

export function DatasetSelector() {
  const { datasetInfo, setDatasetInfo, setCNNConfig, clearTrainingHistory, setTrainingStatus } = useCNNStore()
  const [detailDataset, setDetailDataset] = useState<DatasetDetail | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    // Set default dataset on mount
    if (!datasetInfo) {
      const defaultDs = DATASETS[0]
      setDatasetInfo({
        name: defaultDs.name,
        description: defaultDs.description,
        type: 'classification',
        samples: defaultDs.samples,
        imageSize: defaultDs.imageSize,
        classes: defaultDs.classes
      })

      // Apply default network for MNIST
      fetch(`/api/datasets/${defaultDs.name}/default-network`)
        .then(res => res.ok ? res.json() : null)
        .then(config => {
          if (config) {
            setCNNConfig({
              layers: config.layers,
              optimizer: config.optimizer,
              learningRate: config.learning_rate,
              batchSize: config.batch_size,
              epochs: config.epochs
            })
          }
        })
        .catch(() => {})
    }
  }, [])

  const fetchDatasetInfo = async (name: string) => {
    setLoading(true)
    try {
      const res = await fetch(`/api/datasets/${name}/info`)
      if (!res.ok) throw new Error('Failed to fetch')
      const data = await res.json()
      setDetailDataset(data)
    } catch {
      alert('获取数据集信息失败')
    } finally {
      setLoading(false)
    }
  }

  const handleSelect = async (ds: DatasetOption) => {
    setDatasetInfo({
      name: ds.name,
      description: ds.description,
      type: 'classification',
      samples: ds.samples,
      imageSize: ds.imageSize,
      classes: ds.classes
    } as DatasetInfo)

    // Reset training state
    clearTrainingHistory()
    setTrainingStatus('idle')

    // Fetch and apply default network for this dataset
    try {
      const res = await fetch(`/api/datasets/${ds.name}/default-network`)
      if (res.ok) {
        const config = await res.json()
        setCNNConfig({
          layers: config.layers.map(mapBackendLayer),
          optimizer: config.optimizer,
          learningRate: config.learning_rate,
          batchSize: config.batch_size,
          epochs: config.epochs
        })
      }
    } catch {
      // Silently ignore - keep current network config
    }
  }

  return (
    <div className="panel-card">
      <div className="panel-header">
        <h3>数据集选择</h3>
        <p>选择训练使用的数据集</p>
      </div>
      <div className="panel-content">
        <div className="config-row">
          <span className="config-label">当前数据集</span>
          <span className="config-value">{DATASETS.find(d => d.name === datasetInfo?.name)?.label || datasetInfo?.name?.toUpperCase() || '未选择'}</span>
        </div>

        <div className="section-title">可用数据集</div>
        {DATASETS.map(ds => (
          <div
            key={ds.name}
            className={`layer-item ${datasetInfo?.name === ds.name ? 'selected' : ''}`}
            style={{ cursor: 'pointer', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}
          >
            <div style={{ flex: 1, display: 'flex', alignItems: 'center', gap: '0.5rem' }} onClick={() => handleSelect(ds)}>
              <span className="layer-type" style={{ fontSize: '0.8rem' }}>{ds.label}</span>
              <span className="layer-params" style={{ fontSize: '0.75rem' }}>{ds.description}</span>
              <span
                style={{
                  fontSize: '0.65rem',
                  padding: '1px 6px',
                  borderRadius: '10px',
                  background: DIFFICULTY_COLORS[ds.difficulty] || '#888',
                  color: '#fff',
                  whiteSpace: 'nowrap'
                }}
              >
                {ds.difficulty}
              </span>
            </div>
            <button
              className="btn-icon"
              onClick={(e) => { e.stopPropagation(); fetchDatasetInfo(ds.name) }}
              title="查看数据集详情"
              style={{
                background: 'none',
                border: 'none',
                color: 'var(--text-muted)',
                cursor: 'pointer',
                padding: '4px',
                display: 'flex',
                alignItems: 'center'
              }}
            >
              <Info size={14} />
            </button>
          </div>
        ))}

        {datasetInfo && (
          <div className="info-grid" style={{ marginTop: '1rem' }}>
            <div className="info-item">
              <span className="info-label">图像尺寸</span>
              <span className="info-value">{datasetInfo.imageSize}×{datasetInfo.imageSize}</span>
            </div>
            <div className="info-item">
              <span className="info-label">类别数</span>
              <span className="info-value">{datasetInfo.classes}</span>
            </div>
            <div className="info-item">
              <span className="info-label">通道</span>
              <span className="info-value">{DATASETS.find(d => d.name === datasetInfo.name)?.channels === 3 ? 'RGB' : '灰度'}</span>
            </div>
          </div>
        )}
      </div>

      {/* Dataset info modal */}
      {detailDataset && (
        <div className="modal-overlay" onClick={() => setDetailDataset(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <BookOpen size={18} />
                <h3>{detailDataset.name}</h3>
              </div>
              <button
                className="modal-close"
                onClick={() => setDetailDataset(null)}
                style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer' }}
              >
                <X size={18} />
              </button>
            </div>
            <div className="modal-body">
              <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
                {detailDataset.full_name}
              </p>

              <div style={{ fontSize: '0.85rem', lineHeight: 1.6, marginBottom: '1rem' }}>
                {detailDataset.description}
              </div>

              <div className="info-grid">
                <div className="info-item">
                  <span className="info-label">难度等级</span>
                  <span className="info-value">
                    <span style={{
                      color: DIFFICULTY_COLORS[detailDataset.difficulty] || '#fff',
                      fontWeight: 600
                    }}>
                      {detailDataset.difficulty}
                    </span>
                  </span>
                </div>
                <div className="info-item">
                  <span className="info-label">图像尺寸</span>
                  <span className="info-value">{detailDataset.image_size}×{detailDataset.image_size}</span>
                </div>
                <div className="info-item">
                  <span className="info-label">通道数</span>
                  <span className="info-value">{detailDataset.channels}</span>
                </div>
                <div className="info-item">
                  <span className="info-label">类别数</span>
                  <span className="info-value">{detailDataset.num_classes}</span>
                </div>
                <div className="info-item">
                  <span className="info-label">训练集</span>
                  <span className="info-value">{detailDataset.train_samples?.toLocaleString()}</span>
                </div>
                <div className="info-item">
                  <span className="info-label">测试集</span>
                  <span className="info-value">{detailDataset.test_samples?.toLocaleString()}</span>
                </div>
              </div>

              <div style={{ marginTop: '1rem' }}>
                <div className="section-title">类别列表</div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
                  {detailDataset.classes.map((cls, i) => (
                    <span
                      key={i}
                      style={{
                        fontSize: '0.7rem',
                        padding: '2px 8px',
                        borderRadius: '10px',
                        background: 'var(--bg-secondary)',
                        color: 'var(--text-primary)'
                      }}
                    >
                      {cls}
                    </span>
                  ))}
                </div>
              </div>

              <div style={{ marginTop: '1rem' }}>
                <div className="section-title">推荐模型</div>
                <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                  {detailDataset.recommended_model}
                </p>
              </div>

              {detailDataset.paper && (
                <div style={{ marginTop: '0.75rem', fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                  参考: {detailDataset.paper}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {loading && (
        <div className="modal-overlay" onClick={() => setLoading(false)}>
          <div className="modal-content" style={{ textAlign: 'center', padding: '2rem' }}>
            加载中...
          </div>
        </div>
      )}
    </div>
  )
}
