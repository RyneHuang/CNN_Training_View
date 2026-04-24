import { Link, useLocation } from 'react-router-dom'
import './Layout.css'

interface LayoutProps {
  children: React.ReactNode
}

const navItems = [
  { path: '/', label: '首页', icon: '🏠' },
  { path: '/convolution', label: '卷积层', icon: '🔍' },
  { path: '/pooling', label: '池化层', icon: '📊' },
  { path: '/training', label: '训练过程', icon: '⚡' },
]

export default function Layout({ children }: LayoutProps) {
  const location = useLocation()

  return (
    <div className="layout">
      <nav className="sidebar">
        <div className="logo">
          <span className="logo-icon">🧠</span>
          <span className="logo-text">CNN Training View</span>
        </div>
        <ul className="nav-links">
          {navItems.map((item) => (
            <li key={item.path}>
              <Link
                to={item.path}
                className={`nav-link ${location.pathname === item.path ? 'active' : ''}`}
              >
                <span className="nav-icon">{item.icon}</span>
                <span className="nav-label">{item.label}</span>
              </Link>
            </li>
          ))}
        </ul>
        <div className="sidebar-footer">
          <p className="version">v1.0.0</p>
        </div>
      </nav>
      <main className="main-content">
        {children}
      </main>
    </div>
  )
}
