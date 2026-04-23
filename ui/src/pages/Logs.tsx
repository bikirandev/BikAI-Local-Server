import { useState, useEffect, useRef, useCallback } from 'react'
import { fetchLogs } from '../api'
import { RefreshCw, Download } from 'lucide-react'

export default function Logs() {
  const [lines, setLines] = useState<string[]>([])
  const [numLines, setNumLines] = useState(200)
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)

  const load = useCallback(async (scroll = false) => {
    setLoading(true)
    try {
      const r = await fetchLogs(numLines)
      setLines(r.lines)
      if (scroll) setTimeout(() => bottomRef.current?.scrollIntoView({ behavior: 'smooth' }), 50)
    } catch {
      // ignore
    } finally {
      setLoading(false)
    }
  }, [numLines])

  useEffect(() => { load(true) }, [load])

  useEffect(() => {
    if (!autoRefresh) return
    const id = setInterval(() => load(false), 3000)
    return () => clearInterval(id)
  }, [autoRefresh, load])

  function colorLine(line: string): string {
    const l = line.toLowerCase()
    if (l.includes('error') || l.includes('traceback') || l.includes('exception')) return 'log-line-err'
    if (l.includes('warn') || l.includes('warning')) return 'log-line-warn'
    if (l.includes('[+]') || l.includes('ready') || l.includes('started')) return 'log-line-ok'
    return ''
  }

  function downloadLogs() {
    const blob = new Blob([lines.join('\n')], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'bikai-server.log'
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div>
      <div className="card">
        <div className="card-header">
          <span className="card-title" style={{ margin: 0 }}>Server Logs</span>
          <div className="flex items-center gap-12">
            <div className="flex items-center gap-8">
              <span className="form-label" style={{ margin: 0 }}>Lines:</span>
              <select
                className="form-select"
                style={{ width: 90, padding: '4px 8px' }}
                value={numLines}
                onChange={e => setNumLines(Number(e.target.value))}
              >
                {[50, 100, 200, 500, 1000].map(n => (
                  <option key={n} value={n}>{n}</option>
                ))}
              </select>
            </div>
            <div className="toggle-row" style={{ padding: 0, gap: 8 }}>
              <span className="form-label" style={{ margin: 0 }}>Auto-refresh</span>
              <label className="toggle">
                <input
                  type="checkbox"
                  checked={autoRefresh}
                  onChange={e => setAutoRefresh(e.target.checked)}
                />
                <span className="toggle-slider" />
              </label>
            </div>
            <button className="btn btn-ghost btn-sm" onClick={() => load(true)} disabled={loading}>
              <RefreshCw size={13} className={loading ? 'spin' : ''} />
            </button>
            <button className="btn btn-ghost btn-sm" onClick={downloadLogs} disabled={lines.length === 0}>
              <Download size={13} /> Save
            </button>
          </div>
        </div>

        <div className="log-wrap">
          {lines.length === 0
            ? <span className="text-muted">No log entries yet. Start the server to see output here.</span>
            : lines.map((line, i) => (
              <div key={i} className={colorLine(line)}>{line || ' '}</div>
            ))
          }
          <div ref={bottomRef} />
        </div>
        <p className="form-hint mt-8">
          Logs are written when the server is started from this controller.
          {autoRefresh && ' Auto-refreshing every 3s.'}
        </p>
      </div>
    </div>
  )
}
