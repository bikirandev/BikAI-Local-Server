import { useState, useEffect, useRef, useCallback } from 'react'
import { fetchModels, downloadModel, fetchDownloadStatus, deleteModel } from '../api'
import type { ModelInfo } from '../api'
import { Download, HardDrive, CheckCircle, Circle, RefreshCw, Copy, Star, Trash2 } from 'lucide-react'

const RECOMMENDED_MODEL = {
  name: 'Gemma 3 4B (Q4_K_M)',
  description: 'Google\'s Gemma 3 4B — best balance of speed and quality for CPU inference.',
  size: '~2.3 GB RAM',
  gdrive_id: '1kO_KTjQ-GcaarzLxqXnUyJkEmbM6UC3d',
}

interface Alert { type: 'success' | 'error' | 'info'; msg: string }
type DlType = 'huggingface' | 'gdrive' | 'url'

export default function Models() {
  const [models, setModels] = useState<ModelInfo[]>([])
  const [loading, setLoading] = useState(true)
  const [dlType, setDlType] = useState<DlType>('gdrive')
  const [busy, setBusy] = useState(false)
  const [alert, setAlert] = useState<Alert | null>(null)
  const [dlActive, setDlActive] = useState(false)
  const [dlLines, setDlLines] = useState<string[]>([])
  const dlLogRef = useRef<HTMLDivElement>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const [copied, setCopied] = useState(false)
  const [deleting, setDeleting] = useState<string | null>(null)

  async function handleDelete(name: string) {
    if (!confirm(`Delete ${name}? This cannot be undone.`)) return
    setDeleting(name)
    try {
      await deleteModel(name)
      showAlert('success', `${name} deleted.`)
      await load()
    } catch (e: unknown) {
      const msg = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail ?? String(e)
      showAlert('error', `Delete failed: ${msg}`)
    } finally {
      setDeleting(null)
    }
  }

  // HuggingFace fields
  const [hfRepo, setHfRepo] = useState('')
  const [hfFile, setHfFile] = useState('')
  // GDrive
  const [gdId, setGdId] = useState('')
  // URL
  const [dlUrl, setDlUrl] = useState('')
  const [setDefault, setSetDefault] = useState(true)

  const showAlert = useCallback((type: Alert['type'], msg: string) => {
    setAlert({ type, msg })
    setTimeout(() => setAlert(null), 8000)
  }, [])

  const load = useCallback(async () => {
    try {
      const r = await fetchModels()
      setModels(r.models)
    } catch {
      // ignore
    } finally {
      setLoading(false)
    }
  }, [])

  const stopPolling = () => {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null }
  }

  const startPolling = () => {
    stopPolling()
    pollRef.current = setInterval(async () => {
      try {
        const s = await fetchDownloadStatus()
        setDlLines(s.lines)
        if (!s.active) {
          setDlActive(false)
          stopPolling()
          void load()   // refresh model list when done
        }
        // auto-scroll log
        if (dlLogRef.current) {
          dlLogRef.current.scrollTop = dlLogRef.current.scrollHeight
        }
      } catch { /* ignore */ }
    }, 1500)
  }

  useEffect(() => { load() }, [load])

  // On mount, check if a download is already running (e.g. page refresh mid-download)
  useEffect(() => {
    fetchDownloadStatus().then(s => {
      if (s.active) {
        setDlActive(true)
        setDlLines(s.lines)
        startPolling()
      }
    }).catch(() => {})
    return () => stopPolling()
  // startPolling and stopPolling are stable (defined inline, deps don't change)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  async function handleDownload() {
    setBusy(true)
    setAlert(null)
    try {
      const payload =
        dlType === 'huggingface'
          ? { type: 'huggingface' as const, repo: hfRepo, file: hfFile, set_default: setDefault }
          : dlType === 'gdrive'
          ? { type: 'gdrive' as const, id: gdId, set_default: setDefault }
          : { type: 'url' as const, url: dlUrl, set_default: setDefault }
      const r = await downloadModel(payload)
      setDlActive(true)
      setDlLines([r.message ?? 'Download started…'])
      startPolling()
      showAlert('info', 'Download started — progress shown below.')
    } catch (e: unknown) {
      const msg = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail ?? String(e)
      showAlert('error', `Download failed: ${msg}`)
    } finally {
      setBusy(false)
    }
  }

  const totalGB = models.reduce((s, m) => s + parseFloat(m.size), 0)
  const hasGemma = models.some(m => m.name.toLowerCase().includes('gemma3-4b') || m.name.toLowerCase().includes('gemma-3-4b') || m.name.toLowerCase().includes('gemma3_4b'))

  return (
    <div>
      {alert && (
        <div className={`alert alert-${alert.type === 'error' ? 'error' : alert.type === 'success' ? 'success' : 'info'}`}>
          {alert.msg}
        </div>
      )}

      {/* Model list */}
      <div className="card">
        <div className="card-header">
          <span className="card-title" style={{ margin: 0 }}>Downloaded Models</span>
          <div className="flex items-center gap-8">
            <span className="text-muted" style={{ fontSize: 12 }}>
              {models.length} model{models.length !== 1 ? 's' : ''} · {totalGB.toFixed(2)} GB total
            </span>
            <button className="btn btn-ghost btn-sm" onClick={load}>
              <RefreshCw size={13} />
            </button>
          </div>
        </div>

        {loading ? (
          <p className="text-muted" style={{ fontSize: 13 }}>Loading…</p>
        ) : models.length === 0 ? (
          <div style={{ padding: '24px 0', textAlign: 'center' }}>
            <HardDrive size={28} color="var(--muted)" />
            <p className="text-muted" style={{ marginTop: 8, fontSize: 13 }}>
              No models found in <code>./models/</code>.
              Download one below.
            </p>
          </div>
        ) : (
          <div className="table-wrap">
            <table className="tbl">
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Size</th>
                  <th>Status</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {models.map(m => (
                  <tr key={m.path}>
                    <td>
                      <div className="flex items-center gap-8">
                        {m.active
                          ? <CheckCircle size={14} color="var(--green)" />
                          : <Circle size={14} color="var(--muted)" />
                        }
                        <span className="mono" style={{ fontSize: 13 }}>{m.name}</span>
                      </div>
                    </td>
                    <td>{m.size}</td>
                    <td>
                      <span className={`badge ${m.active ? 'active' : 'inactive'}`}>
                        {m.active ? 'Active' : 'Available'}
                      </span>
                    </td>
                    <td style={{ textAlign: 'right' }}>
                      <button
                        className="btn btn-ghost btn-sm"
                        style={{ color: 'var(--red)', padding: '3px 8px' }}
                        disabled={deleting === m.name}
                        onClick={() => handleDelete(m.name)}
                        title={m.active ? 'Stop server before deleting active model' : 'Delete model'}
                      >
                        {deleting === m.name
                          ? <span className="spin">⟳</span>
                          : <Trash2 size={13} />
                        }
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Recommended model — hidden once gemma3-4b is downloaded */}
      {!hasGemma && (
        <div className="card" style={{ borderColor: 'var(--blue)', borderWidth: 1, borderStyle: 'solid' }}>
          <div className="card-header">
            <span className="card-title" style={{ margin: 0, display: 'flex', alignItems: 'center', gap: 8 }}>
              <Star size={15} color="var(--blue)" fill="var(--blue)" /> Recommended Model
            </span>
            <span className="badge active">Free download</span>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            <div style={{ fontSize: 15, fontWeight: 600 }}>{RECOMMENDED_MODEL.name}</div>
            <div className="text-muted" style={{ fontSize: 13 }}>{RECOMMENDED_MODEL.description}</div>
            <div style={{ fontSize: 12, color: 'var(--muted)' }}>Requires: {RECOMMENDED_MODEL.size}</div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 4 }}>
              <span className="text-muted" style={{ fontSize: 12 }}>Google Drive ID:</span>
              <code style={{
                background: 'var(--bg)', padding: '3px 8px', borderRadius: 4,
                fontSize: 12, fontFamily: 'monospace', letterSpacing: 0.3
              }}>
                {RECOMMENDED_MODEL.gdrive_id}
              </code>
              <button
                className="btn btn-ghost btn-sm"
                style={{ padding: '3px 8px' }}
                onClick={() => {
                  navigator.clipboard.writeText(RECOMMENDED_MODEL.gdrive_id)
                  setCopied(true)
                  setTimeout(() => setCopied(false), 2000)
                }}
              >
                <Copy size={12} /> {copied ? 'Copied!' : 'Copy'}
              </button>
            </div>
            <button
              className="btn btn-primary"
              style={{ marginTop: 8, alignSelf: 'flex-start' }}
              disabled={dlActive || busy}
              onClick={async () => {
                setDlType('gdrive')
                setGdId(RECOMMENDED_MODEL.gdrive_id)
                setBusy(true)
                setAlert(null)
                try {
                  const r = await downloadModel({ type: 'gdrive', id: RECOMMENDED_MODEL.gdrive_id, set_default: true })
                  setDlActive(true)
                  setDlLines([r.message ?? 'Download started…'])
                  startPolling()
                  showAlert('info', 'Download started — progress shown below.')
                } catch (e: unknown) {
                  const msg = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail ?? String(e)
                  showAlert('error', `Download failed: ${msg}`)
                } finally {
                  setBusy(false)
                }
              }}
            >
              {dlActive ? <><span className="spin">⟳</span> Downloading…</> : <><Download size={13} /> Download Gemma 3 4B</>}
            </button>
          </div>
        </div>
      )}
      <div className="card">
        <div className="card-title">Download a Model</div>

        {/* Type tabs */}
        <div className="flex gap-8 mb-16">
          {(['huggingface', 'gdrive', 'url'] as DlType[]).map(t => (
            <button
              key={t}
              className={`btn btn-sm ${dlType === t ? 'btn-secondary' : 'btn-ghost'}`}
              onClick={() => setDlType(t)}
            >
              {t === 'huggingface' ? '🤗 HuggingFace' : t === 'gdrive' ? '📁 Google Drive' : '🔗 Direct URL'}
            </button>
          ))}
        </div>

        {dlType === 'huggingface' && (
          <div className="form-grid">
            <div className="form-group">
              <label className="form-label">Repository</label>
              <input
                className="form-input"
                placeholder="e.g. bartowski/gemma-3-4b-it-GGUF"
                value={hfRepo}
                onChange={e => setHfRepo(e.target.value)}
              />
            </div>
            <div className="form-group">
              <label className="form-label">Filename</label>
              <input
                className="form-input"
                placeholder="e.g. gemma-3-4b-it-Q4_K_M.gguf"
                value={hfFile}
                onChange={e => setHfFile(e.target.value)}
              />
            </div>
          </div>
        )}

        {dlType === 'gdrive' && (
          <div className="form-grid">
            <div className="form-group" style={{ gridColumn: '1 / -1' }}>
              <label className="form-label">Google Drive File ID or Share URL</label>
              <input
                className="form-input"
                placeholder="e.g. 1aBcDeFgHiJkLmNoPqRsTuV  or  https://drive.google.com/file/d/…"
                value={gdId}
                onChange={e => setGdId(e.target.value)}
              />
            </div>
          </div>
        )}

        {dlType === 'url' && (
          <div className="form-grid">
            <div className="form-group" style={{ gridColumn: '1 / -1' }}>
              <label className="form-label">Direct Download URL</label>
              <input
                className="form-input"
                placeholder="https://your-storage.com/model.gguf"
                value={dlUrl}
                onChange={e => setDlUrl(e.target.value)}
              />
            </div>
          </div>
        )}

        <div className="toggle-row mb-16" style={{ maxWidth: 300 }}>
          <div>
            <div className="toggle-label">Set as default model</div>
            <div className="toggle-hint">Saves to MODEL_PATH in .env</div>
          </div>
          <label className="toggle">
            <input
              type="checkbox"
              checked={setDefault}
              onChange={e => setSetDefault(e.target.checked)}
            />
            <span className="toggle-slider" />
          </label>
        </div>

        <button
          className="btn btn-primary"
          onClick={handleDownload}
          disabled={busy || dlActive ||
            (dlType === 'huggingface' && (!hfRepo.trim() || !hfFile.trim())) ||
            (dlType === 'gdrive' && !gdId.trim()) ||
            (dlType === 'url' && !dlUrl.trim())
          }
        >
          {busy
            ? <><span className="spin">⟳</span> Starting download…</>
            : <><Download size={14} /> Download Model</>
          }
        </button>
        <p className="form-hint mt-8">
          Download runs in the background. Progress appears below.
        </p>
      </div>

      {/* Download progress panel */}
      {(dlActive || dlLines.length > 0) && (
        <div className="card">
          <div className="card-header">
            <span className="card-title" style={{ margin: 0 }}>
              {dlActive
                ? <><span className="spin">⟳</span> Downloading…</>
                : '✓ Download complete'}
            </span>
            {!dlActive && (
              <button className="btn btn-ghost btn-sm" onClick={() => setDlLines([])}>
                Dismiss
              </button>
            )}
          </div>
          <div
            ref={dlLogRef}
            style={{
              background: 'var(--bg)',
              borderRadius: 6,
              padding: '10px 14px',
              fontFamily: 'monospace',
              fontSize: 12,
              lineHeight: 1.6,
              maxHeight: 200,
              overflowY: 'auto',
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-all',
            }}
          >
            {dlLines.length === 0
              ? <span className="text-muted">Waiting for output…</span>
              : dlLines.map((l, i) => <div key={i}>{l || '\u00a0'}</div>)
            }
          </div>
        </div>
      )}
    </div>
  )
}
