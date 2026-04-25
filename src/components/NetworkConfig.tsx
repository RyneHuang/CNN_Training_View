import { useState, useEffect, useMemo } from 'react'
import { useCNNStore, CNNLayer } from '../store/cnnStore'
import { Plus, Trash2, ChevronDown, ChevronUp, ArrowRight, GripVertical, AlertCircle } from 'lucide-react'
import {
  DndContext,
  closestCenter,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
  DragEndEvent,
} from '@dnd-kit/core'
import {
  arrayMove,
  SortableContext,
  sortableKeyboardCoordinates,
  useSortable,
  verticalListSortingStrategy,
} from '@dnd-kit/sortable'
import { CSS } from '@dnd-kit/utilities'

const layerTypes = [
  { type: 'conv', label: '卷积层' },
  { type: 'pool', label: '池化层' },
  { type: 'flatten', label: '扁平化' },
  { type: 'fc', label: '全连接层' }
]

const activations = ['relu', 'softmax', 'sigmoid']

interface LayerShape {
  channels: number
  height: number
  width: number
  outputStr: string
}

interface ValidationError {
  message: string
  affectedLayers: number[]
}

// Sortable layer item component
function SortableLayerItem({
  layer,
  index,
  shape,
  isExpanded,
  onToggle,
  onDelete,
  onUpdate,
  getInChannels
}: {
  layer: CNNLayer
  index: number
  shape: LayerShape | null
  isExpanded: boolean
  onToggle: () => void
  onDelete: () => void
  onUpdate: (field: keyof CNNLayer, value: any) => void
  getInChannels: (index: number) => number
}) {
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging
  } = useSortable({ id: `layer-${index}` })

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.5 : 1,
  }

  const getLayerTitle = (l: CNNLayer): string => {
    switch (l.type) {
      case 'conv': return `卷积 ${l.outChannels}通道`
      case 'pool': return `池化 ${l.poolSize}×${l.poolSize}`
      case 'fc': return `全连接 ${l.units}神经元`
      case 'flatten': return '扁平化'
      default: return l.name
    }
  }

  return (
    <div ref={setNodeRef} style={style} className="config-layer-item">
      <div className="config-layer-header">
        <span {...attributes} {...listeners} className="drag-handle">
          <GripVertical size={14} />
        </span>
        <span className="config-layer-icon">
          {layer.type === 'conv' && '⚡'}
          {layer.type === 'pool' && '📉'}
          {layer.type === 'fc' && '🔗'}
          {layer.type === 'flatten' && '⬜'}
        </span>
        <span className="config-layer-name" onClick={onToggle}>{getLayerTitle(layer)}</span>
        {isExpanded ? <ChevronUp size={14} onClick={onToggle} /> : <ChevronDown size={14} onClick={onToggle} />}
      </div>

      {!isExpanded && shape && (
        <div className="layer-shape-info">
          <ArrowRight size={12} />
          <span className="shape-output">{shape.outputStr}</span>
        </div>
      )}

      {isExpanded && (
        <div className="config-layer-params">
          {layer.type === 'conv' && (
            <>
              <div className="param-row param-row-readonly">
                <label>输入通道</label>
                <span className="readonly-value">{getInChannels(index)}</span>
                <span className="param-hint">（自动推导）</span>
              </div>
              <div className="param-row">
                <label>输出通道</label>
                <input
                  type="number"
                  value={layer.outChannels || 32}
                  onChange={(e) => onUpdate('outChannels', Number(e.target.value))}
                />
              </div>
              <div className="param-row">
                <label>卷积核尺寸</label>
                <input
                  type="number"
                  value={layer.kernelSize || 3}
                  onChange={(e) => onUpdate('kernelSize', Number(e.target.value))}
                />
              </div>
              <div className="param-row">
                <label>填充</label>
                <input
                  type="number"
                  value={layer.padding || 0}
                  onChange={(e) => onUpdate('padding', Number(e.target.value))}
                />
              </div>
              <div className="param-row">
                <label>激活函数</label>
                <select
                  value={layer.activation || 'relu'}
                  onChange={(e) => onUpdate('activation', e.target.value)}
                >
                  {activations.map(a => <option key={a} value={a}>{a}</option>)}
                </select>
              </div>
            </>
          )}

          {layer.type === 'pool' && (
            <>
              <div className="param-row">
                <label>窗口尺寸</label>
                <input
                  type="number"
                  value={layer.poolSize || 2}
                  onChange={(e) => onUpdate('poolSize', Number(e.target.value))}
                />
              </div>
              <div className="param-row">
                <label>步长</label>
                <input
                  type="number"
                  value={layer.stride || 2}
                  onChange={(e) => onUpdate('stride', Number(e.target.value))}
                />
              </div>
            </>
          )}

          {layer.type === 'fc' && (
            <>
              <div className="param-row">
                <label>神经元数量</label>
                <input
                  type="number"
                  value={layer.units || 128}
                  onChange={(e) => onUpdate('units', Number(e.target.value))}
                />
              </div>
              <div className="param-row">
                <label>激活函数</label>
                <select
                  value={layer.activation || 'relu'}
                  onChange={(e) => onUpdate('activation', e.target.value)}
                >
                  {activations.filter(a => a !== 'sigmoid').map(a => (
                    <option key={a} value={a}>{a}</option>
                  ))}
                </select>
              </div>
            </>
          )}

          <button className="btn btn-danger btn-sm" onClick={onDelete}>
            <Trash2 size={12} /> 删除层
          </button>
        </div>
      )}
    </div>
  )
}

export function NetworkConfig() {
  const { cnnConfig, addLayer, removeLayer, updateLayer, setCNNConfig, datasetInfo } = useCNNStore()
  const [editingIndex, setEditingIndex] = useState<number | null>(null)
  const [showAddMenu, setShowAddMenu] = useState(false)
  const [validationErrors, setValidationErrors] = useState<ValidationError[]>([])

  const sensors = useSensors(
    useSensor(PointerSensor),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    })
  )

  // Validate network structure
  const validateStructure = (layers: CNNLayer[]): ValidationError[] => {
    const errors: ValidationError[] = []

    if (!layers || layers.length === 0) return errors

    // Check first layer
    const firstLayer = layers[0]
    if (firstLayer && firstLayer.type === 'fc') {
      errors.push({
        message: '第一层不能是全连接层，必须以卷积层或池化层开始',
        affectedLayers: [0]
      })
    }

    // Check if there's a flatten when there's FC
    const hasFC = layers.some(l => l.type === 'fc')
    const hasFlatten = layers.some(l => l.type === 'flatten')
    if (hasFC && !hasFlatten) {
      errors.push({
        message: '包含全连接层时必须先扁平化',
        affectedLayers: layers.map((_, i) => i)
      })
    }

    // After flatten, only fc allowed
    for (let i = 0; i < layers.length; i++) {
      const current = layers[i]
      if (current.type === 'flatten') {
        const afterFlatten = layers.slice(i + 1)
        const invalidAfterFlatten = afterFlatten.filter(l => l.type !== 'fc')
        if (invalidAfterFlatten.length > 0) {
          errors.push({
            message: '扁平化层之后只能跟全连接层',
            affectedLayers: [i]
          })
        }
        break // Only need to check once
      }
    }

    // Check out_channels validity
    for (let i = 0; i < layers.length; i++) {
      const layer = layers[i]
      if (layer.type === 'conv' && (!layer.outChannels || layer.outChannels < 1)) {
        errors.push({
          message: `卷积层 ${layer.name} 的输出通道数必须 >= 1`,
          affectedLayers: [i]
        })
      }
    }

    // Check if flatten has valid input (must have conv before it)
    const flattenIndex = layers.findIndex(l => l.type === 'flatten')
    if (flattenIndex > 0) {
      const hasConvBeforeFlatten = layers.slice(0, flattenIndex).some(l => l.type === 'conv')
      if (!hasConvBeforeFlatten) {
        errors.push({
          message: '扁平化层之前必须有卷积层',
          affectedLayers: [flattenIndex]
        })
      }
    }

    return errors
  }

  // Get input channels for each layer (auto-derived)
  const getInChannels = (index: number): number => {
    if (index === 0) {
      // First layer: derive from dataset
      if (datasetInfo?.name === 'mnist') return 1
      if (datasetInfo?.name === 'cifar10') return 3
      return 3 // default
    }
    // Subsequent layers: from previous layer's output
    const prevLayer = cnnConfig.layers[index - 1]
    if (prevLayer.type === 'conv') {
      return prevLayer.outChannels || 32
    }
    if (prevLayer.type === 'pool') {
      // Find previous conv layer
      for (let i = index - 2; i >= 0; i--) {
        if (cnnConfig.layers[i].type === 'conv') {
          return cnnConfig.layers[i].outChannels || 32
        }
      }
    }
    return 1
  }

  // Calculate shapes after each layer
  const layerShapes = useMemo(() => {
    const shapes: LayerShape[] = []
    const imageSize = datasetInfo?.imageSize || 28
    let channels = datasetInfo?.classes || 10
    let height = imageSize
    let width = imageSize

    if (datasetInfo?.name === 'mnist') {
      channels = 1
    } else if (datasetInfo?.name === 'cifar10') {
      channels = 3
    }

    for (let i = 0; i < cnnConfig.layers.length; i++) {
      const layer = cnnConfig.layers[i]

      if (layer.type === 'conv') {
        const k = layer.kernelSize || 3
        const p = layer.padding || 0
        const outChannels = layer.outChannels || 32
        const outSize = Math.floor((height + 2 * p - k) / 1) + 1
        channels = outChannels
        height = outSize
        width = outSize
        shapes.push({ channels, height, width, outputStr: `${height}×${width}×${channels}` })
      } else if (layer.type === 'pool') {
        const ps = layer.poolSize || 2
        const outSize = Math.floor(height / ps)
        height = outSize
        width = outSize
        shapes.push({ channels, height, width, outputStr: `${height}×${width}×${channels}` })
      } else if (layer.type === 'flatten') {
        const total = channels * height * width
        shapes.push({ channels: 1, height: 1, width: total, outputStr: `${total}` })
      } else if (layer.type === 'fc') {
        const units = layer.units || 128
        shapes.push({ channels: 1, height: 1, width: units, outputStr: `${units}` })
      }
    }

    return shapes
  }, [cnnConfig.layers, datasetInfo])

  // Validate whenever layers change
  useEffect(() => {
    try {
      const errors = validateStructure(cnnConfig.layers)
      setValidationErrors(errors)
    } catch (e) {
      console.error('Validation error:', e)
      setValidationErrors([])
    }
  }, [cnnConfig.layers])

  const handleDragEnd = (event: DragEndEvent) => {
    try {
      const { active, over } = event
      if (!over || active.id === over.id) return

      const oldIndex = parseInt(active.id.toString().replace('layer-', ''))
      const newIndex = parseInt(over.id.toString().replace('layer-', ''))

      if (isNaN(oldIndex) || isNaN(newIndex)) return
      if (oldIndex < 0 || newIndex < 0) return
      if (oldIndex >= cnnConfig.layers.length || newIndex >= cnnConfig.layers.length) return

      const newLayers = arrayMove(cnnConfig.layers, oldIndex, newIndex)
      setCNNConfig({ ...cnnConfig, layers: newLayers })
    } catch (e) {
      console.error('Drag error:', e)
    }
  }

  const handleAddLayer = (type: string) => {
    const newLayer: CNNLayer = {
      type: type as CNNLayer['type'],
      name: `${type.toUpperCase()}${cnnConfig.layers.length + 1}`,
      inChannels: type === 'conv' ? 1 : undefined,
      outChannels: type === 'conv' ? 32 : undefined,
      kernelSize: type === 'conv' ? 3 : undefined,
      padding: type === 'conv' ? 1 : undefined,
      poolSize: type === 'pool' ? 2 : undefined,
      stride: type === 'pool' ? 2 : undefined,
      units: type === 'fc' ? 128 : undefined,
      activation: type === 'conv' || type === 'fc' ? 'relu' : undefined
    }
    addLayer(newLayer)
    setShowAddMenu(false)
  }

  const handleUpdateLayer = (index: number, field: keyof CNNLayer, value: any) => {
    const currentLayers = useCNNStore.getState().cnnConfig.layers
    const layer = currentLayers[index]
    if (!layer) return
    updateLayer(index, { ...layer, [field]: value })
  }

  const canDrag = validationErrors.length === 0

  return (
    <div className="panel-card">
      <div className="panel-header">
        <h3>网络配置</h3>
        <p>调整神经网络的层结构和参数</p>
      </div>
      <div className="panel-content">
        {validationErrors.length > 0 && (
          <div className="validation-errors">
            <AlertCircle size={14} />
            <div className="error-messages">
              {validationErrors.map((err, i) => (
                <p key={i}>{err.message}</p>
              ))}
            </div>
          </div>
        )}

        <div className="layer-list">
          <DndContext
            sensors={sensors}
            collisionDetection={closestCenter}
            onDragEnd={handleDragEnd}
          >
            <SortableContext
              items={cnnConfig.layers.map((_, i) => `layer-${i}`)}
              strategy={verticalListSortingStrategy}
            >
              {cnnConfig.layers.map((layer, index) => (
                <SortableLayerItem
                  key={`layer-${index}`}
                  layer={layer}
                  index={index}
                  shape={layerShapes[index] || null}
                  isExpanded={editingIndex === index}
                  onToggle={() => setEditingIndex(editingIndex === index ? null : index)}
                  onDelete={() => removeLayer(index)}
                  onUpdate={(field, value) => handleUpdateLayer(index, field, value)}
                  getInChannels={getInChannels}
                />
              ))}
            </SortableContext>
          </DndContext>
        </div>

        <div className="add-layer-section">
          <button
            className="btn btn-secondary"
            onClick={() => setShowAddMenu(!showAddMenu)}
          >
            <Plus size={14} /> 添加层
          </button>

          {showAddMenu && (
            <div className="add-layer-menu">
              {layerTypes.map(({ type, label }) => (
                <button
                  key={type}
                  className="add-layer-option"
                  onClick={() => handleAddLayer(type)}
                >
                  {label}
                </button>
              ))}
            </div>
          )}
        </div>

        <div className="config-summary">
          <span>共 {cnnConfig.layers.length} 层</span>
          {!canDrag && <span className="warning-text">（请先修复结构问题）</span>}
        </div>

        <div className="structure-rules">
          <p><strong>结构规则：</strong></p>
          <ul>
            <li>第一层必须是卷积层或池化层</li>
            <li>全连接层前必须先扁平化</li>
            <li>扁平化后只能跟全连接层</li>
          </ul>
        </div>
      </div>
    </div>
  )
}
