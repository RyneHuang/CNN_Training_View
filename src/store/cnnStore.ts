import { create } from 'zustand'

export interface CNNLayer {
  type: 'conv' | 'pool' | 'fc' | 'flatten'
  name: string
  inChannels?: number
  outChannels?: number
  kernelSize?: number
  padding?: number
  poolSize?: number
  stride?: number
  units?: number
  activation?: 'relu' | 'softmax' | 'sigmoid'
}

export interface CNNConfig {
  layers: CNNLayer[]
  optimizer: 'adam' | 'adamw' | 'sgd'
  learningRate: number
  batchSize: number
  epochs: number
}

export interface TrainingHistoryEntry {
  epoch: number
  loss: number
  accuracy: number
  valLoss?: number
  valAccuracy?: number
}

export interface DatasetInfo {
  name: string
  description: string
  type: 'classification'
  samples: number
  imageSize?: number
  classes?: number
}

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
  currentEpoch: number
  lossHistory: TrainingHistoryEntry[]
  modelId: string | null
  tenantId: string

  // Actions
  setTrainingStatus: (status: 'idle' | 'training' | 'completed' | 'error') => void
  setCurrentEpoch: (epoch: number) => void
  addTrainingHistory: (entry: TrainingHistoryEntry) => void
  clearTrainingHistory: () => void
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
    // 更深的网络结构，适合 MNIST
    { type: 'conv', name: 'Conv1', outChannels: 64, kernelSize: 3, padding: 1, activation: 'relu' },
    { type: 'pool', name: 'Pool1', poolSize: 2, stride: 2 },
    { type: 'conv', name: 'Conv2', outChannels: 128, kernelSize: 3, padding: 1, activation: 'relu' },
    { type: 'pool', name: 'Pool2', poolSize: 2, stride: 2 },
    { type: 'flatten', name: 'Flat' },
    { type: 'fc', name: 'FC1', units: 512, activation: 'relu' },
    { type: 'fc', name: 'FC2', units: 10 },  // 输出层不需要 softmax, CrossEntropyLoss 会处理
  ],
  optimizer: 'adam',
  learningRate: 0.003,  // 配合 CosineAnnealing 学习率调度器
  batchSize: 256,  // 更大的 batch size 加快训练
  epochs: 50
}

export const useCNNStore = create<CNNStore>((set) => ({
  // Network config
  cnnConfig: defaultConfig,
  setCNNConfig: (config) => set({ cnnConfig: config }),
  addLayer: (layer) => set((state) => ({
    cnnConfig: { ...state.cnnConfig, layers: [...state.cnnConfig.layers, layer] }
  })),
  removeLayer: (index) => set((state) => ({
    cnnConfig: {
      ...state.cnnConfig,
      layers: state.cnnConfig.layers.filter((_, i) => i !== index)
    }
  })),
  updateLayer: (index, layer) => set((state) => ({
    cnnConfig: {
      ...state.cnnConfig,
      layers: state.cnnConfig.layers.map((l, i) => i === index ? layer : l)
    }
  })),

  // Dataset
  datasetInfo: { name: 'mnist', description: '手写数字识别', type: 'classification', samples: 60000, imageSize: 28, classes: 10 },
  setDatasetInfo: (info) => set({ datasetInfo: info }),

  // Training state
  trainingStatus: 'idle',
  currentEpoch: 0,
  lossHistory: [],
  modelId: null,
  tenantId: 'default-tenant',

  setTrainingStatus: (status) => set({ trainingStatus: status }),
  setCurrentEpoch: (epoch) => set({ currentEpoch: epoch }),
  addTrainingHistory: (entry) => set((state) => ({
    lossHistory: [...state.lossHistory, entry]
  })),
  clearTrainingHistory: () => set({ lossHistory: [], currentEpoch: 0 }),
  setModelId: (id) => set({ modelId: id }),
  setTenantId: (id) => set({ tenantId: id }),
  setEpochs: (epochs) => set((state) => ({ cnnConfig: { ...state.cnnConfig, epochs } })),
  setBatchSize: (batchSize) => set((state) => ({ cnnConfig: { ...state.cnnConfig, batchSize } })),
  setLearningRate: (learningRate) => set((state) => ({ cnnConfig: { ...state.cnnConfig, learningRate } })),
  setOptimizer: (optimizer) => set((state) => ({ cnnConfig: { ...state.cnnConfig, optimizer } })),

  // Inference
  predict: async (_imageData) => {
    // TODO: Call backend API
    // For now, return mock prediction
    const mockPrediction = new Array(10).fill(0).map(() => Math.random())
    const maxIdx = mockPrediction.indexOf(Math.max(...mockPrediction))
    mockPrediction[maxIdx] = 1
    return {
      prediction: mockPrediction,
      predictedClass: maxIdx,
      confidence: 0.95
    }
  }
}))