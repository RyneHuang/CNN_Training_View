import { describe, it, expect } from 'vitest'
import { getLayerInChannels } from '../utils/layerUtils'
import { CNNLayer } from '../types'

describe('layerUtils', () => {
  describe('getLayerInChannels', () => {
    const mockDatasetInfo = {
      name: 'mnist',
      description: '手写数字识别',
      type: 'classification' as const,
      samples: 60000,
      imageSize: 28,
      classes: 10
    }

    it('should return 1 for first layer in MNIST', () => {
      const layers: CNNLayer[] = []
      expect(getLayerInChannels(layers, 0, mockDatasetInfo)).toBe(1)
    })

    it('should return 3 for first layer in CIFAR10', () => {
      const layers: CNNLayer[] = []
      const cifarInfo = { ...mockDatasetInfo, name: 'cifar10' }
      expect(getLayerInChannels(layers, 0, cifarInfo)).toBe(3)
    })

    it('should return outChannels from previous conv layer', () => {
      const layers: CNNLayer[] = [
        { type: 'conv', name: 'Conv1', outChannels: 64 }
      ]
      expect(getLayerInChannels(layers, 1, mockDatasetInfo)).toBe(64)
    })

    it('should find previous conv layer for pool layer', () => {
      const layers: CNNLayer[] = [
        { type: 'conv', name: 'Conv1', outChannels: 32 },
        { type: 'pool', name: 'Pool1', poolSize: 2 }
      ]
      expect(getLayerInChannels(layers, 1, mockDatasetInfo)).toBe(32)
    })
  })
})