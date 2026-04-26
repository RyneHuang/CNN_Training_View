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