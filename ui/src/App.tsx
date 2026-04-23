import { useState, useEffect } from 'react'
import { saveKey, hasKey, clearKey, fetchStatus } from './api'
import type { StatusData } from './api'
import Dashboard from './pages/Dashboard'
import Models from './pages/Models'
import Nginx from './pages/Nginx'
import Logs from './pages/Logs'
import Settings from './pages/Settings'
import Stats from './pages/Stats'
import {
  LayoutDashboard, HardDrive, Globe, ScrollText, Settings as SettingsIcon, Lock, BarChart2, LogOut
} from 'lucide-react'

type Page = 'dashboard' | 'models' | 'nginx' | 'logs' | 'settings' | 'stats'

const VALID_PAGES: Page[] = ['dashboard', 'models', 'nginx', 'logs', 'settings', 'stats']

function pageFromHash(): Page {
  const hash = window.location.hash.replace(/^#\/?/, '')
  return VALID_PAGES.includes(hash as Page) ? (hash as Page) : 'dashboard'
}

const PAGES: { id: Page; label: string; Icon: React.ElementType }[] = [
  { id: 'dashboard', label: 'Dashboard',  Icon: LayoutDashboard },
  { id: 'models',    label: 'Models',     Icon: HardDrive },
  { id: 'stats',     label: 'Stats',      Icon: BarChart2 },
  { id: 'nginx',     label: 'Nginx',      Icon: Globe },
  { id: 'logs',      label: 'Logs',       Icon: ScrollText },
  { id: 'settings',  label: 'Settings',   Icon: SettingsIcon },
]

const PAGE_TITLE: Record<Page, string> = {
  dashboard: 'Dashboard',
  models:    'Models',
  stats:     'Stats',
  nginx:     'Nginx',
  logs:      'Logs',
  settings:  'Settings',
}

// ── Lock screen ──────────────────────────────────────────────────────────────
function LockScreen({ onUnlock }: { onUnlock: () => void }) {
  const [key, setKey] = useState('')
  const [err, setErr] = useState('')
  const [busy, setBusy] = useState(false)

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!key.trim()) { setErr('Enter your API key.'); return }
    setBusy(true)
    setErr('')
    saveKey(key.trim())
    try {
      await fetchStatus()   // will fail with 401 if wrong key
      onUnlock()
    } catch {
      setErr('Invalid API key. Check the key with: bikai token show')
      import('./api').then(({ clearKey }) => clearKey())
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="lock-screen">
      <div className="lock-card">
        <div style={{ fontSize: 28, marginBottom: 8 }}>
          <Lock size={32} color="var(--blue)" />
        </div>
        <h1>Bik AI</h1>
        <p>Enter your API key to access the control panel</p>
        <form onSubmit={handleSubmit}>
          <div className="form-group mb-16">
            <input
              className="form-input mono"
              type="password"
              placeholder="Paste API key…"
              value={key}
              onChange={e => setKey(e.target.value)}
              autoFocus
            />
          </div>
          {err && (
            <div className="alert alert-error mb-12" style={{ textAlign: 'left' }}>
              {err}
            </div>
          )}
          <button className="btn btn-primary" style={{ width: '100%' }} disabled={busy}>
            {busy ? <><span className="spin">⟳</span> Verifying…</> : 'Unlock'}
          </button>
        </form>
        <p className="text-muted mt-12" style={{ fontSize: 11 }}>
          Find your key: <code>bikai token show</code>
        </p>
      </div>
    </div>
  )
}

// ── Main app ─────────────────────────────────────────────────────────────────
export default function App() {
  const [page, setPage] = useState<Page>(pageFromHash())
  const [unlocked, setUnlocked] = useState(hasKey())
  const [status, setStatus] = useState<StatusData | null>(null)

  function handleLogout() {
    clearKey()
    setUnlocked(false)
    setStatus(null)
  }

  // Keep URL hash in sync
  function navigate(p: Page) {
    setPage(p)
    window.location.hash = p
  }

  // Sync page if user uses browser back/forward
  useEffect(() => {
    const handler = () => setPage(pageFromHash())
    window.addEventListener('hashchange', handler)
    return () => window.removeEventListener('hashchange', handler)
  }, [])

  useEffect(() => {
    if (!unlocked) return
    fetchStatus().then(setStatus).catch(() => {})
    const id = setInterval(() => fetchStatus().then(setStatus).catch(() => {}), 5000)
    return () => clearInterval(id)
  }, [unlocked])

  if (!unlocked) {
    return <LockScreen onUnlock={() => setUnlocked(true)} />
  }

  const isRunning = status?.running ?? false

  return (
    <div className="shell">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-name">BIK AI</div>
          <div className="brand-sub">Local LLM Server</div>
        </div>
        <nav className="nav">
          <div className="nav-section">Control</div>
          {PAGES.map(({ id, label, Icon }) => (
            <button
              key={id}
              className={`nav-item ${page === id ? 'active' : ''}`}
              onClick={() => navigate(id)}
            >
              <Icon />
              <span>{label}</span>
            </button>
          ))}
        </nav>
        <div className="sidebar-footer">
          <a href="https://bikiran.com" target="_blank" rel="noreferrer">bikiran.com</a>
        </div>
      </aside>

      {/* Main */}
      <div className="main">
        <header className="topbar">
          <span className="topbar-title">{PAGE_TITLE[page]}</span>
          <span className={`status-pill ${isRunning ? 'running' : 'stopped'}`}>
            <span className={`dot ${isRunning ? 'pulse' : ''}`} />
            {isRunning ? 'Server Running' : 'Server Stopped'}
          </span>
          {status?.model_name && status.model_name !== '—' && (
            <span className="text-muted" style={{ fontSize: 12 }}>
              {status.model_name}
            </span>
          )}
          <button
            onClick={handleLogout}
            className="btn btn-ghost"
            style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 6, fontSize: 13 }}
            title="Log out"
          >
            <LogOut size={14} />
            Logout
          </button>
        </header>

        <main className="content">
          {page === 'dashboard' && <Dashboard />}
          {page === 'models'    && <Models />}
          {page === 'stats'     && <Stats />}
          {page === 'nginx'     && <Nginx />}
          {page === 'logs'      && <Logs />}
          {page === 'settings'  && <Settings />}
        </main>
      </div>
    </div>
  )
}
