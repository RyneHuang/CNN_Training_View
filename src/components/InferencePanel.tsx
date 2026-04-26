import { useState, useRef, useCallback, useEffect } from 'react'
import { useCNNStore } from '../store/cnnStore'
import { Image as ImageIcon } from 'lucide-react'

interface SampleImage {
  data: number[]
  label: number
}

export function InferencePanel() {
  const { hasTrainedModel, datasetInfo, tenantId } = useCNNStore()
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [prediction, setPrediction] = useState<{ class: number, confidence: number, probs: number[] } | null>(null)
  const [loading, setLoading] = useState(false)
  const [samples, setSamples] = useState<SampleImage[]>([])
  const [selectedSample, setSelectedSample] = useState<number | null>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Fetch validation samples
  useEffect(() => {
    if (hasTrainedModel && datasetInfo) {
      fetch(`/api/datasets/${datasetInfo.name}/samples?count=20`)
        .then(res => res.json())
        .then(data => {
          if (data.samples) {
            setSamples(data.samples.map((s: number[], i: number) => ({
              data: s,
              label: data.labels[i]
            })))
          }
        })
        .catch(err => console.error('Failed to load samples:', err))
    }
  }, [hasTrainedModel, datasetInfo])

  const renderSampleToCanvas = (sampleData: number[], size: number = 28, channels: number = 1) => {
    const canvas = document.createElement('canvas')
    canvas.width = size
    canvas.height = size
    const ctx = canvas.getContext('2d')
    if (!ctx) return ''

    const imageData = ctx.createImageData(size, size)
    if (channels === 1) {
      // Grayscale: data is [1, H, W] flattened
      for (let i = 0; i < size * size; i++) {
        const val = Math.max(0, Math.min(255, sampleData[i]))
        imageData.data[i * 4] = val
        imageData.data[i * 4 + 1] = val
        imageData.data[i * 4 + 2] = val
        imageData.data[i * 4 + 3] = 255
      }
    } else {
      // RGB: data is [C, H, W] flattened (CHW format)
      const pixelCount = size * size
      for (let i = 0; i < pixelCount; i++) {
        imageData.data[i * 4] = Math.max(0, Math.min(255, sampleData[i]))                    // R
        imageData.data[i * 4 + 1] = Math.max(0, Math.min(255, sampleData[i + pixelCount]))   // G
        imageData.data[i * 4 + 2] = Math.max(0, Math.min(255, sampleData[i + pixelCount * 2])) // B
        imageData.data[i * 4 + 3] = 255
      }
    }
    ctx.putImageData(imageData, 0, 0)
    return canvas.toDataURL()
  }

  const getDatasetChannels = () => {
    const name = datasetInfo?.name || 'mnist'
    return ['cifar10', 'cifar100'].includes(name) ? 3 : 1
  }

  const handleSampleClick = (sample: SampleImage, index: number) => {
    setSelectedSample(index)
    const imageSize = datasetInfo?.imageSize || 28
    const channels = getDatasetChannels()
    const imageUrl = renderSampleToCanvas(sample.data, imageSize, channels)
    setPreviewUrl(imageUrl)
    setPrediction(null)
  }

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    const reader = new FileReader()
    reader.onload = (event) => {
      setPreviewUrl(event.target?.result as string)
      setPrediction(null)
    }
    reader.readAsDataURL(file)
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    const file = e.dataTransfer.files?.[0]
    if (!file || !file.type.startsWith('image/')) return

    const reader = new FileReader()
    reader.onload = (event) => {
      setPreviewUrl(event.target?.result as string)
      setPrediction(null)
    }
    reader.readAsDataURL(file)
  }, [])

  const handlePredict = async () => {
    if (!previewUrl && selectedSample === null) return

    setLoading(true)

    let imageData: number[] = []

    // If a sample is selected, use its data directly
    if (selectedSample !== null && samples[selectedSample]) {
      imageData = samples[selectedSample].data
      // Normalize to [0, 1] (backend handles normalization)
      imageData = imageData.map(v => v / 255.0)
    } else if (previewUrl) {
      // Load image from upload and convert to grayscale
      const img = new Image()
      await new Promise<void>((resolve) => {
        img.onload = () => resolve()
        img.src = previewUrl
      })

      const canvas = canvasRef.current
      if (!canvas) return

      const ctx = canvas.getContext('2d')
      if (!ctx) return

      // Resize to dataset size
      const size = datasetInfo?.imageSize || 28
      canvas.width = size
      canvas.height = size
      ctx.fillRect(0, 0, size, size)
      ctx.drawImage(img, 0, 0, size, size)

      // Get pixel data
      const imgData = ctx.getImageData(0, 0, size, size)
      const grayscale: number[] = []

      for (let i = 0; i < imgData.data.length; i += 4) {
        // Convert to grayscale and normalize to [0, 1]
        const gray = (imgData.data[i] * 0.299 +
          imgData.data[i + 1] * 0.587 +
          imgData.data[i + 2] * 0.114) / 255.0
        grayscale.push(gray) // No inversion needed
      }
      imageData = grayscale
    }

    // Call actual API
    try {
      const response = await fetch('/api/inference', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          tenant_id: tenantId,
          image_data: imageData
        })
      })

      if (!response.ok) {
        throw new Error('Inference failed')
      }

      const result = await response.json()

      setPrediction({
        class: result.predicted_class,
        confidence: result.confidence,
        probs: result.probabilities
      })
    } catch (error) {
      console.error('Inference error:', error)
      alert('推理失败，请重试')
    } finally {
      setLoading(false)
    }
  }

  if (!hasTrainedModel) {
    return (
      <div className="panel-card">
        <div className="panel-header">
          <h3>图片推理</h3>
          <p>训练完成后可上传图片进行预测</p>
        </div>
        <div className="panel-content">
          <div style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-muted)' }}>
            <ImageIcon size={48} style={{ marginBottom: '0.5rem', opacity: 0.5 }} />
            <p>请先完成模型训练</p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="panel-card">
      <div className="panel-header">
        <h3>图片推理</h3>
        <p>上传图片进行 MNIST 分类预测</p>
      </div>
      <div className="panel-content">
        <input
          type="file"
          accept="image/*"
          ref={fileInputRef}
          onChange={handleFileSelect}
          style={{ display: 'none' }}
        />

        <div
          className="upload-zone"
          onClick={() => fileInputRef.current?.click()}
          onDrop={handleDrop}
          onDragOver={(e) => e.preventDefault()}
        >
          {previewUrl ? (
            <img src={previewUrl} alt="Preview" style={{ maxWidth: '100%', maxHeight: '120px', borderRadius: '8px' }} />
          ) : (
            <>
              <div className="upload-icon">📷</div>
              <div className="upload-text">点击或拖拽上传图片</div>
            </>
          )}
        </div>

        <canvas ref={canvasRef} style={{ display: 'none' }} width={28} height={28} />

        {samples.length > 0 && (
          <div className="sample-gallery">
            <div className="section-title">验证集样本 (点击选择)</div>
            <div className="sample-grid">
              {samples.map((sample, index) => (
                <div
                  key={index}
                  className={`sample-item ${selectedSample === index ? 'selected' : ''}`}
                  onClick={() => handleSampleClick(sample, index)}
                  title={`Label: ${sample.label}`}
                >
                  <img
                    src={renderSampleToCanvas(sample.data, datasetInfo?.imageSize || 28, getDatasetChannels())}
                    alt={`Sample ${index}`}
                  />
                  {selectedSample === index && (
                    <div className="sample-label">真实: {sample.label}</div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        <button
          className="btn btn-primary"
          style={{ width: '100%', marginTop: '0.75rem' }}
          onClick={handlePredict}
          disabled={(!previewUrl && selectedSample === null) || loading}
        >
          {loading ? '预测中...' : '🔮 开始预测'}
        </button>

        {prediction && (
          <div className="prediction-result">
            <div className="prediction-class">预测: {prediction.class}</div>
            <div className="prediction-confidence">置信度: {(prediction.confidence * 100).toFixed(1)}%</div>
            <div className="probability-bars">
              {prediction.probs.map((prob, i) => (
                <div key={i} className="prob-bar-item">
                  <span className="prob-label">{i}</span>
                  <div className="prob-bar">
                    <div className="prob-bar-fill" style={{ width: `${prob * 100}%` }} />
                  </div>
                  <span className="prob-value">{(prob * 100).toFixed(0)}%</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {datasetInfo && (
          <div className="info-grid" style={{ marginTop: '0.75rem' }}>
            <div className="info-item">
              <span className="info-label">数据集</span>
              <span className="info-value">{datasetInfo.name.toUpperCase()}</span>
            </div>
            <div className="info-item">
              <span className="info-label">类别</span>
              <span className="info-value">{datasetInfo.classes} 类</span>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}