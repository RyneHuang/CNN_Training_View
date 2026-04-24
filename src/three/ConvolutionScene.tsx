import { useRef, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'

interface ConvolutionSceneProps {
  inputSize?: number
  kernelSize?: number
  stride?: number
  isAnimating?: boolean
}

export default function ConvolutionScene({
  inputSize = 8,
  kernelSize = 3,
  stride = 1,
  isAnimating = true
}: ConvolutionSceneProps) {
  const groupRef = useRef<THREE.Group>(null)
  const filterRef = useRef<THREE.Mesh>(null)
  const progressRef = useRef(0)

  const { inputData, outputSize } = useMemo(() => {
    const data: number[][] = []
    for (let i = 0; i < inputSize; i++) {
      const row: number[] = []
      for (let j = 0; j < inputSize; j++) {
        const dist = Math.sqrt((i - inputSize/2)**2 + (j - inputSize/2)**2)
        row.push(Math.max(0, 1 - dist / (inputSize/2)))
      }
      data.push(row)
    }
    const outSize = Math.floor((inputSize - kernelSize) / stride) + 1
    return { inputData: data, outputSize: outSize }
  }, [inputSize, kernelSize, stride])

  const cubes = useMemo(() => {
    const meshes: { position: [number, number, number]; value: number }[] = []
    const offset = (inputSize - 1) / 2
    for (let i = 0; i < inputSize; i++) {
      for (let j = 0; j < inputSize; j++) {
        meshes.push({
          position: [j - offset, inputSize - i - offset, 0],
          value: inputData[i][j]
        })
      }
    }
    return meshes
  }, [inputData, inputSize])

  const outputCubes = useMemo(() => {
    const meshes: { position: [number, number, number]; value: number }[] = []
    const outOffset = (outputSize - 1) / 2
    for (let i = 0; i < outputSize; i++) {
      for (let j = 0; j < outputSize; j++) {
        meshes.push({
          position: [j - outOffset, outputSize - i - outOffset, -2 - (i * outputSize + j) * 0.01],
          value: 0.5
        })
      }
    }
    return meshes
  }, [outputSize])

  useFrame((_, delta) => {
    if (isAnimating && filterRef.current && groupRef.current) {
      progressRef.current += delta * 0.5
      const maxSteps = outputSize * outputSize
      const currentStep = Math.floor(progressRef.current % maxSteps)

      const outRow = Math.floor(currentStep / outputSize)
      const outCol = currentStep % outputSize

      const startI = outRow * stride
      const startJ = outCol * stride

      filterRef.current.position.x = startJ - (inputSize - 1) / 2
      filterRef.current.position.y = (inputSize - 1) / 2 - startI

      filterRef.current.visible = true

      let sum = 0
      for (let ki = 0; ki < kernelSize; ki++) {
        for (let kj = 0; kj < kernelSize; kj++) {
          const ii = startI + ki
          const jj = startJ + kj
          if (ii < inputSize && jj < inputSize) {
            sum += inputData[ii][jj]
          }
        }
      }

      const avgSum = sum / (kernelSize * kernelSize)
      outputCubes[outRow * outputSize + outCol].value = avgSum
    }
  })

  const getColor = (value: number) => {
    const r = value
    const g = value * 0.5
    const b = 1 - value * 0.3
    return new THREE.Color(r, g, b)
  }

  return (
    <group ref={groupRef}>
      {cubes.map((cube, idx) => (
        <mesh key={idx} position={[cube.position[0], cube.position[1], cube.position[2]]}>
          <boxGeometry args={[0.9, 0.9, 0.9]} />
          <meshStandardMaterial
            color={getColor(cube.value)}
            transparent
            opacity={0.8}
          />
        </mesh>
      ))}

      <mesh ref={filterRef} position={[0, 0, 1]}>
        <boxGeometry args={[kernelSize * 0.95, kernelSize * 0.95, 0.3]} />
        <meshStandardMaterial
          color="#fbbf24"
          transparent
          opacity={0.6}
          wireframe
        />
      </mesh>

      <mesh position={[0, 0, -2]}>
        <boxGeometry args={[outputSize * 0.9, outputSize * 0.9, 0.1]} />
        <meshStandardMaterial color="#6366f1" opacity={0.3} transparent />
      </mesh>

      {outputCubes.map((cube, idx) => (
        <mesh key={idx} position={[cube.position[0], cube.position[1], cube.position[2]]}>
          <boxGeometry args={[0.85, 0.85, 0.5]} />
          <meshStandardMaterial
            color={getColor(cube.value)}
            transparent
            opacity={0.9}
          />
        </mesh>
      ))}

      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} intensity={1} />
    </group>
  )
}
