import { Suspense } from 'react'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import { ErrorBoundary } from './components/ErrorBoundary'
import HomePage from './pages/HomePage'
import ConvolutionPage from './pages/ConvolutionPage'
import PoolingPage from './pages/PoolingPage'
import TrainingPage from './pages/TrainingPage'
import './App.css'

function App() {
  return (
    <ErrorBoundary>
      <BrowserRouter>
        <Layout>
          <Suspense fallback={<Loading />}>
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route path="/convolution" element={<ConvolutionPage />} />
              <Route path="/pooling" element={<PoolingPage />} />
              <Route path="/training" element={<TrainingPage />} />
            </Routes>
          </Suspense>
        </Layout>
      </BrowserRouter>
    </ErrorBoundary>
  )
}

function Loading() {
  return (
    <div className="loading-container">
      <div className="loading-spinner" />
      <p>加载中...</p>
    </div>
  )
}

export default App