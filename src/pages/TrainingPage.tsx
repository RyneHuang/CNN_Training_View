import { DatasetSelector } from '../components/DatasetSelector'
import { TrainingPanel } from '../components/TrainingPanel'
import { NetworkConfig } from '../components/NetworkConfig'
import { LossChart } from '../components/LossChart'
import { InferencePanel } from '../components/InferencePanel'
import './TrainingPage.css'

export default function TrainingPage() {

  return (
    <div className="training-page">
      <div className="training-header">
        <h1>🧠 CNN 训练与推理</h1>
        <p>配置卷积神经网络、训练模型并上传图片进行推理</p>
      </div>

      <div className="training-grid">
        {/* Left Column */}
        <div className="left-column">
          <DatasetSelector />
          <NetworkConfig />
        </div>

        {/* Middle Column */}
        <div className="middle-column">
          <TrainingPanel />
          <LossChart />
        </div>

        {/* Right Column */}
        <div className="right-column">
          <InferencePanel />
        </div>
      </div>
    </div>
  )
}