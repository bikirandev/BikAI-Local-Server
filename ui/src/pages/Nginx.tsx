import { useState, useEffect } from 'react'
import { fetchNginxConfig, applyNginxConfig, fetchNginxStatus } from '../api'
import type { NginxConfig, NginxPayload } from '../api'
import { Save, RefreshCw, CheckCircle, XCircle, Globe } from 'lucide-react'

interface Alert { type: 'success' | 'error' | 'info' | 'warn'; msg: string }

const DEFAULT_FORM: NginxPayload = {
  domain: '',
  ai_port: 8000,
  ctrl_port: 8001,
  listen_port: 80,
  ssl: false,
  cors_origin: '*',
  read_timeout: 300,
  client_max_body_size: '10m',
  gzip: true,
  worker_processes: 'auto',
  worker_connections: 1024,
}

export default function Nginx() {
  const [cfg, setCfg] = useState<NginxConfig | null>(null)
  const [form, setForm] = useState<NginxPayload>(DEFAULT_FORM)
  const [busy, setBusy] = useState(false)
  const [statusBusy, setStatusBusy] = useState(false)
  const [nginxStatus, setNginxStatus] = useState<{ config_valid: boolean; config_message: string; service_status: string } | null>(null)
  const [alert, setAlert] = useState<Alert | null>(null)

  function showAlert(type: Alert['type'], msg: string) {
    setAlert({ type, msg })
    setTimeout(() => setAlert(null), 8000)
  }

  async function load() {
    try {
      const c = await fetchNginxConfig()
      setCfg(c)
      setForm(f => ({
        ...f,
        domain: c.domain || f.domain,
        worker_processes: c.worker_processes || f.worker_processes,
        worker_connections: c.worker_connections || f.worker_connections,
      }))
    } catch {
      // ignore
    }
  }

  async function loadStatus() {
    setStatusBusy(true)
    try {
      const s = await fetchNginxStatus()
      setNginxStatus(s)
    } catch {
      // ignore
    } finally {
      setStatusBusy(false)
    }
  }

  useEffect(() => { load(); loadStatus() }, [])

  function patch<K extends keyof NginxPayload>(key: K, val: NginxPayload[K]) {
    setForm(f => ({ ...f, [key]: val }))
  }

  async function handleApply() {
    setBusy(true)
    setAlert(null)
    try {
      const r = await applyNginxConfig(form)
      showAlert('success', `nginx configured. URL: ${r.url}${r.ssl ? ' (HTTPS)' : ''}`)
      await load()
      await loadStatus()
    } catch (e: unknown) {
      const msg = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail ?? String(e)
      showAlert('error', `Failed: ${msg}`)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div>
      {alert && (
        <div className={`alert alert-${alert.type === 'error' ? 'error' : alert.type === 'success' ? 'success' : alert.type === 'warn' ? 'warn' : 'info'}`}>
          {alert.msg}
        </div>
      )}

      {/* Status card */}
      <div className="card mb-16">
        <div className="card-header">
          <span className="card-title" style={{ margin: 0 }}>Nginx Status</span>
          <button className="btn btn-ghost btn-sm" onClick={loadStatus} disabled={statusBusy}>
            <RefreshCw size={13} className={statusBusy ? 'spin' : ''} />
          </button>
        </div>
        <div className="flex gap-12 flex-wrap mb-12">
          {[
            ['Installed', cfg?.installed ? 'Yes' : 'No', cfg?.installed],
            ['Site enabled', cfg?.enabled ? 'Yes' : 'No', cfg?.enabled],
            ['Service active', cfg?.active ? 'Yes' : 'No', cfg?.active],
            ['Domain', cfg?.domain || '(none)', true],
          ].map(([label, val, ok]) => (
            <div key={String(label)} className="stat-box" style={{ minWidth: 140 }}>
              <div className="stat-label">{label}</div>
              <div className="flex items-center gap-8 mt-8">
                {ok
                  ? <CheckCircle size={14} color="var(--green)" />
                  : <XCircle size={14} color="var(--red)" />
                }
                <span style={{ fontSize: 13, fontWeight: 600 }}>{String(val)}</span>
              </div>
            </div>
          ))}
        </div>

        {nginxStatus && (
          <div>
            <div className={`badge ${nginxStatus.config_valid ? 'active' : 'inactive'} mb-8`}>
              {nginxStatus.config_valid ? '✓ Config valid' : '✗ Config error'}
            </div>
            {nginxStatus.config_message && (
              <pre style={{ fontSize: 11, color: 'var(--muted)', whiteSpace: 'pre-wrap', marginTop: 4 }}>
                {nginxStatus.config_message}
              </pre>
            )}
          </div>
        )}
      </div>

      {/* Config form */}
      <div className="card">
        <div className="card-title">Configure Reverse Proxy</div>

        <div className="form-grid">
          <div className="form-group">
            <label className="form-label">Domain / IP</label>
            <input
              className="form-input"
              placeholder="e.g. api.example.com  (leave blank to auto-detect IP)"
              value={form.domain}
              onChange={e => patch('domain', e.target.value)}
            />
            <span className="form-hint">Leave blank to auto-detect public IP</span>
          </div>
          <div className="form-group">
            <label className="form-label">AI Server Port</label>
            <input
              className="form-input"
              type="number"
              value={form.ai_port}
              onChange={e => patch('ai_port', Number(e.target.value))}
            />
          </div>
          <div className="form-group">
            <label className="form-label">Controller Port</label>
            <input
              className="form-input"
              type="number"
              value={form.ctrl_port}
              onChange={e => patch('ctrl_port', Number(e.target.value))}
            />
          </div>
          <div className="form-group">
            <label className="form-label">Listen Port</label>
            <input
              className="form-input"
              type="number"
              value={form.listen_port}
              onChange={e => patch('listen_port', Number(e.target.value))}
            />
          </div>
        </div>

        <div className="form-grid">
          <div className="form-group">
            <label className="form-label">CORS Origin</label>
            <input
              className="form-input"
              value={form.cors_origin}
              onChange={e => patch('cors_origin', e.target.value)}
            />
            <span className="form-hint">Use * for all origins or a specific domain</span>
          </div>
          <div className="form-group">
            <label className="form-label">Read Timeout (seconds)</label>
            <input
              className="form-input"
              type="number" min={30}
              value={form.read_timeout}
              onChange={e => patch('read_timeout', Number(e.target.value))}
            />
            <span className="form-hint">SSE streaming needs ≥ 300s</span>
          </div>
          <div className="form-group">
            <label className="form-label">Max Request Body Size</label>
            <input
              className="form-input"
              placeholder="e.g. 10m, 50m"
              value={form.client_max_body_size}
              onChange={e => patch('client_max_body_size', e.target.value)}
            />
          </div>
        </div>

        <div className="divider" />
        <p className="card-title">Performance &amp; Tuning</p>

        <div className="form-grid">
          <div className="form-group">
            <label className="form-label">Worker Processes</label>
            <input
              className="form-input"
              placeholder="auto"
              value={form.worker_processes}
              onChange={e => patch('worker_processes', e.target.value)}
            />
            <span className="form-hint">"auto" = one per CPU core</span>
          </div>
          <div className="form-group">
            <label className="form-label">Worker Connections</label>
            <input
              className="form-input"
              type="number"
              value={form.worker_connections}
              onChange={e => patch('worker_connections', Number(e.target.value))}
            />
          </div>
        </div>

        <div className="toggle-row mb-16" style={{ maxWidth: 320 }}>
          <div>
            <div className="toggle-label">Gzip Compression</div>
            <div className="toggle-hint">Compress JSON / JS responses</div>
          </div>
          <label className="toggle">
            <input type="checkbox" checked={form.gzip} onChange={e => patch('gzip', e.target.checked)} />
            <span className="toggle-slider" />
          </label>
        </div>

        <div className="toggle-row mb-16" style={{ maxWidth: 320 }}>
          <div>
            <div className="toggle-label">Enable SSL (Let's Encrypt)</div>
            <div className="toggle-hint">Requires a domain name and port 80 open</div>
          </div>
          <label className="toggle">
            <input type="checkbox" checked={form.ssl} onChange={e => patch('ssl', e.target.checked)} />
            <span className="toggle-slider" />
          </label>
        </div>

        {form.ssl && (
          <div className="alert alert-warn mb-16">
            <Globe size={15} />
            SSL requires DNS pointing to this server and port 80 reachable publicly.
            Certbot will be installed automatically if not present.
          </div>
        )}

        <button className="btn btn-primary" onClick={handleApply} disabled={busy}>
          {busy
            ? <><span className="spin">⟳</span> Applying…</>
            : <><Save size={14} /> Apply &amp; Reload Nginx</>
          }
        </button>
      </div>

      {/* Current config preview */}
      {cfg?.config_text && (
        <div className="card">
          <div className="card-title">Current Config (/etc/nginx/sites-available/bikai)</div>
          <pre className="code-block" style={{ maxHeight: 320, overflowY: 'auto' }}>
            {cfg.config_text}
          </pre>
        </div>
      )}
    </div>
  )
}
