import { useState, useEffect, useCallback } from 'react'
import { fetchToken, rotateToken } from '../api'
import { saveKey, clearKey, hasKey } from '../api'
import { Eye, EyeOff, RefreshCw, Copy, Check, LogOut } from 'lucide-react'

interface Alert { type: 'success' | 'error' | 'info'; msg: string }

export default function Settings() {
  const [serverKey, setServerKey] = useState<string | null>(null)
  const [localKey, setLocalKey] = useState(localStorage.getItem('bikai_api_key') ?? '')
  const [showKey, setShowKey] = useState(false)
  const [busy, setBusy] = useState(false)
  const [copied, setCopied] = useState(false)
  const [alert, setAlert] = useState<Alert | null>(null)

  const showAlert = useCallback((type: Alert['type'], msg: string) => {
    setAlert({ type, msg })
    setTimeout(() => setAlert(null), 7000)
  }, [])

  const loadKey = useCallback(async () => {
    try {
      const r = await fetchToken()
      setServerKey(r.key)
    } catch (e: unknown) {
      const msg = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail ?? String(e)
      showAlert('error', `Could not fetch key: ${msg}`)
    }
  }, [showAlert])

  useEffect(() => { if (hasKey()) loadKey() }, [loadKey])

  async function handleRotate() {
    if (!confirm('This will invalidate the current API key. All connected clients will need the new key. Continue?')) return
    setBusy(true)
    try {
      const r = await rotateToken()
      setServerKey(r.key)
      // Update local storage too
      saveKey(r.key)
      setLocalKey(r.key)
      showAlert('success', 'New API key generated and saved.')
    } catch (e: unknown) {
      const msg = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail ?? String(e)
      showAlert('error', `Rotation failed: ${msg}`)
    } finally {
      setBusy(false)
    }
  }

  function handleSaveLocal() {
    if (!localKey.trim()) {
      showAlert('error', 'Key cannot be empty.')
      return
    }
    saveKey(localKey.trim())
    showAlert('success', 'API key saved to browser storage.')
  }

  function handleForget() {
    clearKey()
    setLocalKey('')
    setServerKey(null)
    showAlert('info', 'Key cleared from browser. Re-enter to use protected endpoints.')
  }

  function copy(text: string) {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    })
  }

  return (
    <div>
      {alert && (
        <div className={`alert alert-${alert.type === 'error' ? 'error' : alert.type === 'success' ? 'success' : 'info'}`}>
          {alert.msg}
        </div>
      )}

      {/* Browser key */}
      <div className="card">
        <div className="card-title">Browser API Key</div>
        <p className="text-muted mb-12" style={{ fontSize: 13 }}>
          Enter your API key here. It is stored in your browser's localStorage and
          sent automatically with all requests to protected endpoints.
        </p>
        <div className="form-group mb-16">
          <label className="form-label">API Key</label>
          <div className="flex gap-8">
            <input
              className="form-input mono"
              type={showKey ? 'text' : 'password'}
              placeholder="Paste your API key…"
              value={localKey}
              onChange={e => setLocalKey(e.target.value)}
              style={{ flex: 1 }}
            />
            <button className="btn btn-ghost btn-sm" onClick={() => setShowKey(v => !v)}>
              {showKey ? <EyeOff size={14} /> : <Eye size={14} />}
            </button>
            {localKey && (
              <button className="btn btn-ghost btn-sm" onClick={() => copy(localKey)}>
                {copied ? <Check size={14} color="var(--green)" /> : <Copy size={14} />}
              </button>
            )}
          </div>
        </div>
        <div className="btn-group">
          <button className="btn btn-primary" onClick={handleSaveLocal}>Save Key</button>
          <button className="btn btn-ghost" onClick={handleForget} disabled={!hasKey()}>
            <LogOut size={14} /> Forget Key
          </button>
          {hasKey() && (
            <button className="btn btn-ghost" onClick={loadKey}>
              <RefreshCw size={14} /> Show Server Key
            </button>
          )}
        </div>
      </div>

      {/* Server key */}
      {serverKey !== null && (
        <div className="card">
          <div className="card-title">Server API Key (from .env)</div>
          <p className="text-muted mb-12" style={{ fontSize: 13 }}>
            This is the key stored in <code>.env</code> on the server. All AI inference
            requests require this as the <code>X-API-Key</code> header.
          </p>
          <div className="flex gap-8 mb-16">
            <div className="token-box" style={{ flex: 1 }}>
              {showKey ? serverKey : '•'.repeat(Math.min(serverKey.length, 40))}
            </div>
            <button className="btn btn-ghost btn-sm" onClick={() => setShowKey(v => !v)}>
              {showKey ? <EyeOff size={14} /> : <Eye size={14} />}
            </button>
            <button className="btn btn-ghost btn-sm" onClick={() => copy(serverKey)}>
              {copied ? <Check size={14} color="var(--green)" /> : <Copy size={14} />}
            </button>
          </div>
          <button className="btn btn-warning" onClick={handleRotate} disabled={busy}>
            {busy
              ? <><span className="spin">⟳</span> Rotating…</>
              : <><RefreshCw size={14} /> Rotate Key</>
            }
          </button>
          <p className="form-hint mt-8">
            Rotating generates a new random key and saves it to <code>.env</code>.
            The server uses it immediately — no restart needed.
          </p>
        </div>
      )}

      {/* Usage example */}
      <div className="card">
        <div className="card-title">Example Usage</div>
        <pre className="code-block">{`# Chat completion
curl http://localhost:8000/v1/chat/completions \\
  -H "X-API-Key: YOUR_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "gemma",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'

# Python (openai SDK)
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="YOUR_KEY")
resp = client.chat.completions.create(
    model="gemma",
    messages=[{"role": "user", "content": "Hello!"}]
)`}
        </pre>
      </div>
    </div>
  )
}
