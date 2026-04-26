import { Component, ReactNode } from 'react'

interface Props {
  children: ReactNode
}

interface State {
  hasError: boolean
  error: Error | null
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false, error: null }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error('ErrorBoundary caught:', error, info.componentStack)
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          height: '100vh',
          padding: '2rem',
          textAlign: 'center',
          color: 'var(--text-primary, #333)'
        }}>
          <h2 style={{ marginBottom: '1rem' }}>页面出现错误</h2>
          <p style={{ color: 'var(--text-muted, #999)', marginBottom: '1.5rem' }}>
            {this.state.error?.message || '未知错误'}
          </p>
          <button
            onClick={() => {
              this.setState({ hasError: false, error: null })
              window.location.reload()
            }}
            style={{
              padding: '0.5rem 1.5rem',
              borderRadius: 8,
              border: 'none',
              background: 'var(--primary, #6366f1)',
              color: '#fff',
              cursor: 'pointer',
              fontSize: 14
            }}
          >
            刷新页面
          </button>
        </div>
      )
    }
    return this.props.children
  }
}