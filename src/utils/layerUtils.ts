import { CNNLayer, DatasetInfo } from '../types'

/**
 * Calculate input channels for a layer based on its position and dataset.
 */
export function getLayerInChannels(
  layers: CNNLayer[],
  index: number,
  datasetInfoOrName: DatasetInfo | string | null
): number {
  let datasetName: string | undefined
  if (typeof datasetInfoOrName === 'string') {
    datasetName = datasetInfoOrName
  } else if (datasetInfoOrName) {
    datasetName = datasetInfoOrName.name
  }

  if (index === 0) {
    // Grayscale datasets: mnist, fashion_mnist, kmnist
    if (['mnist', 'fashion_mnist', 'kmnist'].includes(datasetName || '')) return 1
    // RGB datasets: cifar10, cifar100
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

/**
 * Format layer output shape for display.
 */
export function formatLayerShape(
  layer: CNNLayer,
  inputChannels: number,
  inputSize: number
): string {
  const k = layer.kernelSize || 3
  const p = layer.padding || 0

  switch (layer.type) {
    case 'conv': {
      const outChannels = layer.outChannels || 32
      const outSize = Math.floor((inputSize + 2 * p - k) / 1) + 1
      return `${outSize}×${outSize}×${outChannels}`
    }
    case 'pool': {
      const ps = layer.poolSize || 2
      const outSize = Math.floor(inputSize / ps)
      return `${outSize}×${outSize}×${inputChannels}`
    }
    case 'flatten': {
      const total = inputChannels * inputSize * inputSize
      return `${total}`
    }
    case 'fc': {
      const units = layer.units || 128
      return `${units}`
    }
    default:
      return ''
  }
}