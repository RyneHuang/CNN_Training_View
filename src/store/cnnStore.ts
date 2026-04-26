import { create } from 'zustand'
import { CNNLayer, CNNConfig, TrainingHistoryEntry, DatasetInfo } from '../types'

// Generate unique tenant ID for each user session
const generateTenantId = (): string => {
  const stored = localStorage.getItem('cnn_tenant_id')
  if (stored) return stored
  const newId = `user-${Date.now()}-${Math.random().toString(36).substring(2, 8)}`
  localStorage.setItem('cnn_tenant_id', newId)
  return newId
}

export type { CNNLayer, CNNConfig, TrainingHistoryEntry, DatasetInfo }

interface CNNStore {
  // Network config
  cnnConfig: CNNConfig
  setCNNConfig: (config: CNNConfig) => void
  addLayer: (layer: CNNLayer) => void
  removeLayer: (index: number) => void
  updateLayer: (index: number, layer: CNNLayer) => void

  // Dataset
  datasetInfo: DatasetInfo | null
  setDatasetInfo: (info: DatasetInfo | null) => void

  // Training state
  trainingStatus: 'idle' | 'training' | 'completed' | 'error'
  hasTrainedModel: boolean
  currentEpoch: number
  lossHistory: TrainingHistoryEntry[]
  modelId: string | null
  tenantId: string

  // Actions
  setTrainingStatus: (status: 'idle' | 'training' | 'completed' | 'error') => void
  setCurrentEpoch: (epoch: number) => void
  addTrainingHistory: (entry: TrainingHistoryEntry) => void
  clearTrainingHistory: () => void
  setHasTrainedModel: (value: boolean) => void
  setModelId: (id: string | null) => void
  setTenantId: (id: string) => void
  setEpochs: (epochs: number) => void
  setBatchSize: (batchSize: number) => void
  setLearningRate: (learningRate: number) => void
  setOptimizer: (optimizer: 'adam' | 'adamw' | 'sgd') => void

  // Inference
  predict: (imageData: number[]) => Promise<{ prediction: number[], predictedClass: number, confidence: number }>
}

const defaultConfig: CNNConfig = {
  layers: [
    { type: 'conv', name: 'Conv1', outChannels: 64, kernelSize: 3, padding: 1, activation: 'relu' },
    { type: 'pool', name: 'Pool1', poolSize: 2, stride: 2 },
    { type: 'conv', name: 'Conv2', outChannels: 128, kernelSize: 3, padding: 1, activation: 'relu' },
    { type: 'pool', name: 'Pool2', poolSize: 2, stride: 2 },
    { type: 'flatten', name: 'Flat' },
    { type: 'fc', name: 'FC1', units: 512, activation: 'relu' },
    { type: 'fc', name: 'FC2', units: 10 },
  ],
  optimizer: 'adam',
  learningRate: 0.003,
  batchSize: 256,
  epochs: 50
}

export const useCNNStore = create<CNNStore>((set, get) => ({
  // Network config
  cnnConfig: defaultConfig,
  setCNNConfig: (config) => set({ cnnConfig: config }),
  addLayer: (layer) => set((state) => ({
    cnnConfig: { ...state.cnnConfig, layers: [...state.cnnConfig.layers, layer] }
  })),
  removeLayer: (index) => set((state) => ({
    cnnConfig: {
      ...state.cnnConfig,
      layers: state.cnnConfig.layers.filter((_: CNNLayer, i: number) => i !== index)
    }
  })),
  updateLayer: (index, layer) => set((state) => ({
    cnnConfig: {
      ...state.cnnConfig,
      layers: state.cnnConfig.layers.map((l: CNNLayer, i: number) => i === index ? layer : l)
    }
  })),

  // Dataset
  datasetInfo: { name: 'mnist', description: '手写数字识别', type: 'classification', samples: 60000, imageSize: 28, classes: 10 },
  setDatasetInfo: (info) => set({ datasetInfo: info }),

  // Training state
  trainingStatus: 'idle',
  hasTrainedModel: false,
  currentEpoch: 0,
  lossHistory: [],
  modelId: null,
  tenantId: generateTenantId(),

  setTrainingStatus: (status) => set({ trainingStatus: status }),
  setCurrentEpoch: (epoch) => set({ currentEpoch: epoch }),
  addTrainingHistory: (entry) => set((state) => ({
    lossHistory: [...state.lossHistory, entry]
  })),
  clearTrainingHistory: () => set({ lossHistory: [], currentEpoch: 0, hasTrainedModel: false }),
  setHasTrainedModel: (value) => set({ hasTrainedModel: value }),
  setModelId: (id) => set({ modelId: id }),
  setTenantId: (id) => set({ tenantId: id }),
  setEpochs: (epochs) => set((state) => ({ cnnConfig: { ...state.cnnConfig, epochs } })),
  setBatchSize: (batchSize) => set((state) => ({ cnnConfig: { ...state.cnnConfig, batchSize } })),
  setLearningRate: (learningRate) => set((state) => ({ cnnConfig: { ...state.cnnConfig, learningRate } })),
  setOptimizer: (optimizer) => set((state) => ({ cnnConfig: { ...state.cnnConfig, optimizer } })),

  // Inference
  predict: async (imageData: number[]) => {
    const { tenantId } = get()

    const response = await fetch('/api/inference', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        tenant_id: tenantId,
        image_data: imageData
      })
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => null)
      const message = errorData?.detail || `推理失败 (${response.status})`
      throw new Error(message)
    }

    const result = await response.json()
    return {
      prediction: result.probabilities as number[],
      predictedClass: result.predicted_class as number,
      confidence: result.confidence as number
    }
  }
}))