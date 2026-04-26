import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { useCNNStore } from '../store/cnnStore'
import { TrainingHistoryEntry } from '../types'

export function LossChart() {
  const { lossHistory } = useCNNStore()

  if (lossHistory.length === 0) {
    return (
      <div className="panel-card">
        <div className="panel-header">
          <h3>训练曲线</h3>
          <p>开始训练后显示损失和准确率曲线</p>
        </div>
        <div className="panel-content">
          <div className="empty-state" style={{ height: 200, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)' }}>
            等待开始训练...
          </div>
        </div>
      </div>
    )
  }

  const chartData = lossHistory.map((entry: TrainingHistoryEntry) => ({
    epoch: entry.epoch,
    loss: entry.loss,
    accuracy: entry.accuracy * 100
  }))

  return (
    <div className="panel-card">
      <div className="panel-header">
        <h3>训练曲线</h3>
        <p>实时显示训练过程中的损失和准确率</p>
      </div>
      <div className="panel-content">
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--surface-light)" />
            <XAxis
              dataKey="epoch"
              stroke="var(--text-muted)"
              fontSize={12}
              tickLine={false}
            />
            <YAxis
              yAxisId="left"
              stroke="var(--text-muted)"
              fontSize={12}
              tickLine={false}
              label={{ value: 'Loss', angle: -90, position: 'insideLeft', fill: 'var(--text-muted)', fontSize: 12 }}
            />
            <YAxis
              yAxisId="right"
              orientation="right"
              stroke="var(--text-muted)"
              fontSize={12}
              tickLine={false}
              label={{ value: 'Acc %', angle: 90, position: 'insideRight', fill: 'var(--text-muted)', fontSize: 12 }}
            />
            <Tooltip
              contentStyle={{
                background: 'var(--surface)',
                border: '1px solid var(--surface-light)',
                borderRadius: 8,
                fontSize: 12
              }}
            />
            <Legend />
            <Line yAxisId="left" type="monotone" dataKey="loss" stroke="#ef4444" name="损失" dot={false} strokeWidth={2} />
            <Line yAxisId="right" type="monotone" dataKey="accuracy" stroke="#22c55e" name="准确率%" dot={false} strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}