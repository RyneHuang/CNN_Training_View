import { useRef, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { Line } from '@react-three/drei'

interface TrainingSceneProps {
  layerActivations?: number[]
}

export default function TrainingScene({
  layerActivations = [1, 0.8, 0.6, 0.4, 0.2]
}: TrainingSceneProps) {
  const groupRef = useRef<THREE.Group>(null)
  const timeRef = useRef(0)

  const layers = useMemo(() => {
    return layerActivations.map((activation, idx) => ({
      id: idx,
      y: (layerActivations.length - 1) / 2 - idx,
      activation,
      width: activation * 2 + 0.5
    }))
  }, [layerActivations])

  const connections = useMemo(() => {
    const lines: { start: [number, number, 0]; end: [number, number, 0]; opacity: number }[] = []
    for (let i = 0; i < layers.length - 1; i++) {
      for (let j = 0; j < 3; j++) {
        const startX = (j - 1) * 0.8
        const endX = (j - 1) * 0.8
        lines.push({
          start: [startX, layers[i].y, 0],
          end: [endX, layers[i + 1].y, 0],
          opacity: Math.min(layers[i].activation, layers[i + 1].activation)
        })
      }
    }
    return lines
  }, [layers])

  const lossCurvePoints = useMemo(() => {
    const points: THREE.Vector3[] = []
    for (let i = 0; i < 50; i++) {
      const x = (i - 25) * 0.2
      const y = Math.exp(-i * 0.05) * 2 + Math.sin(i * 0.3) * 0.1
      points.push(new THREE.Vector3(x, y - 3, 0))
    }
    return points
  }, [])

  useFrame((_, delta) => {
    timeRef.current += delta
    if (groupRef.current) {
      groupRef.current.rotation.y = Math.sin(timeRef.current * 0.3) * 0.1
    }
  })

  return (
    <group ref={groupRef}>
      {layers.map((layer) => (
        <mesh key={layer.id} position={[0, layer.y, 0]}>
          <cylinderGeometry args={[layer.width, layer.width, 0.3, 32]} />
          <meshStandardMaterial
            color={new THREE.Color(
              0.2 + layer.activation * 0.3,
              0.5 + layer.activation * 0.3,
              0.8
            )}
            emissive={new THREE.Color(0.1, 0.2, 0.5)}
            emissiveIntensity={0.3}
          />
          <mesh position={[1.5, layer.y, 0]}>
            <sphereGeometry args={[0.1, 16, 16]} />
            <meshStandardMaterial
              color={new THREE.Color(layer.activation, layer.activation * 0.5, 0.2)}
              emissive={new THREE.Color(layer.activation, layer.activation * 0.3, 0)}
              emissiveIntensity={0.5}
            />
          </mesh>
        </mesh>
      ))}

      {connections.map((conn, idx) => (
        <Line
          key={idx}
          points={[conn.start, conn.end]}
          color={new THREE.Color(conn.opacity * 0.5, conn.opacity * 0.8, 1)}
          lineWidth={conn.opacity * 2}
          transparent
          opacity={conn.opacity * 0.6}
        />
      ))}

      <mesh position={[0, -3, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[8, 3]} />
        <meshStandardMaterial
          color="#1e293b"
          transparent
          opacity={0.8}
        />
      </mesh>

      <Line
        points={lossCurvePoints}
        color="#ef4444"
        lineWidth={3}
      />

      <ambientLight intensity={0.4} />
      <pointLight position={[5, 5, 5]} intensity={0.8} color="#60a5fa" />
      <pointLight position={[-5, -5, 5]} intensity={0.5} color="#a78bfa" />
    </group>
  )
}
