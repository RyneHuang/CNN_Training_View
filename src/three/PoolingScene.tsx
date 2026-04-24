import { useRef, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'

interface PoolingSceneProps {
  inputSize?: number
  poolSize?: number
  isAnimating?: boolean
}

export default function PoolingScene({
  inputSize = 8,
  poolSize = 2,
  isAnimating = true
}: PoolingSceneProps) {
  const groupRef = useRef<THREE.Group>(null)
  const windowRef = useRef<THREE.Mesh>(null)
  const highlightRef = useRef<THREE.Mesh>(null)
  const progressRef = useRef(0)

  const inputData = useMemo(() => {
    const data: number[][] = []
    for (let i = 0; i < inputSize; i++) {
      const row: number[] = []
      for (let j = 0; j < inputSize; j++) {
        row.push(Math.random())
      }
      data.push(row)
    }
    return data
  }, [inputSize])

  const outputSize = inputSize / poolSize

  const inputCubes = useMemo(() => {
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
    const meshes: { position: [number, number, number]; value: number; isHighlight: boolean }[] = []
    const outOffset = (outputSize - 1) / 2
    for (let i = 0; i < outputSize; i++) {
      for (let j = 0; j < outputSize; j++) {
        meshes.push({
          position: [j - outOffset, outputSize - i - outOffset, -2],
          value: 0,
          isHighlight: false
        })
      }
    }
    return meshes
  }, [outputSize])

  useFrame((_, delta) => {
    if (isAnimating && windowRef.current && highlightRef.current && groupRef.current) {
      progressRef.current += delta * 0.8
      const maxSteps = outputSize * outputSize
      const currentStep = Math.floor(progressRef.current % maxSteps)

      const outRow = Math.floor(currentStep / outputSize)
      const outCol = currentStep % outputSize

      const startI = outRow * poolSize
      const startJ = outCol * poolSize

      windowRef.current.position.x = startJ + (poolSize - 1) / 2 - (inputSize - 1) / 2
      windowRef.current.position.y = (inputSize - 1) / 2 - (startI + (poolSize - 1) / 2)

      let maxVal = 0
      let maxI = startI, maxJ = startJ
      for (let pi = 0; pi < poolSize; pi++) {
        for (let pj = 0; pj < poolSize; pj++) {
          const val = inputData[startI + pi]?.[startJ + pj] || 0
          if (val > maxVal) {
            maxVal = val
            maxI = startI + pi
            maxJ = startJ + pj
          }
        }
      }

      highlightRef.current.position.x = maxJ - (inputSize - 1) / 2
      highlightRef.current.position.y = (inputSize - 1) / 2 - maxI

      outputCubes[currentStep].value = maxVal
      outputCubes[currentStep].isHighlight = true

      setTimeout(() => {
        if (outputCubes[currentStep]) {
          outputCubes[currentStep].isHighlight = false
        }
      }, 300)
    }
  })

  const getColor = (value: number, isHighlight = false) => {
    if (isHighlight) return new THREE.Color(1, 0.8, 0)
    return new THREE.Color(value * 0.3, value * 0.7, value)
  }

  return (
    <group ref={groupRef}>
      {inputCubes.map((cube, idx) => (
        <mesh key={idx} position={[cube.position[0], cube.position[1], cube.position[2]]}>
          <boxGeometry args={[0.9, 0.9, 0.9]} />
          <meshStandardMaterial
            color={getColor(cube.value)}
            transparent
            opacity={0.8}
          />
        </mesh>
      ))}

      <mesh ref={windowRef} position={[0, 0, 1]}>
        <boxGeometry args={[poolSize * 0.95, poolSize * 0.95, 0.2]} />
        <meshStandardMaterial
          color="#22d3ee"
          transparent
          opacity={0.3}
          wireframe
        />
      </mesh>

      <mesh ref={highlightRef} position={[0, 0, 0.5]}>
        <boxGeometry args={[0.85, 0.85, 1]} />
        <meshStandardMaterial
          color="#fbbf24"
          transparent
          opacity={0.4}
        />
      </mesh>

      {outputCubes.map((cube, idx) => (
        <mesh key={idx} position={[cube.position[0], cube.position[1], cube.position[2]]}>
          <boxGeometry args={[0.85, 0.85, 0.5]} />
          <meshStandardMaterial
            color={getColor(cube.value, cube.isHighlight)}
            transparent
            opacity={0.9}
            emissive={cube.isHighlight ? "#fbbf24" : "#000000"}
            emissiveIntensity={cube.isHighlight ? 0.5 : 0}
          />
        </mesh>
      ))}

      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} intensity={1} />
    </group>
  )
}
