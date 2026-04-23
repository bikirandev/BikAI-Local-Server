import { useState, useEffect, useRef } from 'react'
import { Cpu, MemoryStick, HardDrive, Activity } from 'lucide-react'

interface Metrics {
  cpu_pct: number
  ram_total_mb: number
  ram_used_mb: number
  ram_free_mb: number
  ram_pct: number
  ai_mem_mb: number
  disk_total_gb: number
  disk_used_gb: number
  disk_free_gb: number
  disk_pct: number
}

function Bar({ pct, color }: { pct: number; color: string }) {
  return (
    <div style={{ background: 'var(--border)', borderRadius: 4, height: 8, overflow: 'hidden' }}>
      <div style={{
        width: `${Math.min(pct, 100)}%`,
        height: '100%',
        background: color,
        borderRadius: 4,
        transition: 'width 0.4s ease',
      }} />
    </div>
  )
}

function StatCard({
  icon, label, value, sub, pct, color,
}: {
  icon: React.ReactNode
  label: string
  value: string
  sub: string
  pct: number
  color: string
}) {
  return (
    <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      <div className="flex items-center gap-8">
        <span style={{ color }}>{icon}</span>
        <span className="text-muted" style={{ fontSize: 13 }}>{label}</span>
        <span style={{ marginLeft: 'auto', fontWeight: 700, fontSize: 20 }}>{value}</span>
      </div>
      <Bar pct={pct} color={color} />
      <div className="text-muted" style={{ fontSize: 12 }}>{sub}</div>
    </div>
  )
}

export default function Stats() {
  const [metrics, setMetrics] = useState<Metrics | null>(null)
  const [connected, setConnected] = useState(false)
  const [history, setHistory] = useState<{ t: number; cpu: number; ram: number }[]>([])
  const esRef = useRef<EventSource | null>(null)

  useEffect(() => {
    const es = new EventSource('/api/controller/metrics')
    esRef.current = es

    es.onopen = () => setConnected(true)
    es.onerror = () => setConnected(false)
    es.onmessage = (e) => {
      try {
        const d: Metrics = JSON.parse(e.data)
        setMetrics(d)
        setHistory(h => {
          const next = [...h, { t: Date.now(), cpu: d.cpu_pct, ram: d.ram_pct }]
          return next.slice(-60)  // keep last 60 samples (~60s)
        })
      } catch { /* ignore */ }
    }

    return () => { es.close(); esRef.current = null }
  }, [])

  function fmtMB(mb: number) {
    return mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${mb} MB`
  }

  if (!metrics) {
    return (
      <div style={{ padding: 24, color: 'var(--muted)' }}>
        <span className="spin">⟳</span> Connecting to metrics stream…
      </div>
    )
  }

  const cpuColor  = metrics.cpu_pct > 80 ? 'var(--red)' : metrics.cpu_pct > 50 ? 'var(--yellow)' : 'var(--green)'
  const ramColor  = metrics.ram_pct  > 85 ? 'var(--red)' : metrics.ram_pct  > 60 ? 'var(--yellow)' : 'var(--blue)'
  const diskColor = metrics.disk_pct > 90 ? 'var(--red)' : metrics.disk_pct > 70 ? 'var(--yellow)' : 'var(--muted)'

  // Mini sparkline using SVG
  const sparkW = 200, sparkH = 36
  function sparkPath(values: number[], color: string) {
    if (values.length < 2) return null
    const max = 100
    const pts = values.map((v, i) => {
      const x = (i / (values.length - 1)) * sparkW
      const y = sparkH - (v / max) * sparkH
      return `${x},${y}`
    })
    return (
      <polyline
        points={pts.join(' ')}
        fill="none"
        stroke={color}
        strokeWidth={1.5}
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    )
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* Connection status */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <span style={{
          width: 8, height: 8, borderRadius: '50%',
          background: connected ? 'var(--green)' : 'var(--red)',
          display: 'inline-block',
        }} />
        <span className="text-muted" style={{ fontSize: 12 }}>
          {connected ? 'Live — updating every second' : 'Disconnected'}
        </span>
      </div>

      {/* Stat cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 12 }}>
        <StatCard
          icon={<Cpu size={16} />}
          label="CPU Usage"
          value={`${metrics.cpu_pct}%`}
          sub={`${os_cpu_count()} logical cores`}
          pct={metrics.cpu_pct}
          color={cpuColor}
        />
        <StatCard
          icon={<MemoryStick size={16} />}
          label="System RAM"
          value={`${metrics.ram_pct}%`}
          sub={`${fmtMB(metrics.ram_used_mb)} used of ${fmtMB(metrics.ram_total_mb)} · ${fmtMB(metrics.ram_free_mb)} free`}
          pct={metrics.ram_pct}
          color={ramColor}
        />
        <StatCard
          icon={<HardDrive size={16} />}
          label="Disk"
          value={`${metrics.disk_pct}%`}
          sub={`${metrics.disk_used_gb} GB used of ${metrics.disk_total_gb} GB · ${metrics.disk_free_gb} GB free`}
          pct={metrics.disk_pct}
          color={diskColor}
        />
        {metrics.ai_mem_mb > 0 && (
          <StatCard
            icon={<Activity size={16} />}
            label="AI Server RAM"
            value={fmtMB(metrics.ai_mem_mb)}
            sub="Memory used by the inference process"
            pct={Math.round(100 * metrics.ai_mem_mb / metrics.ram_total_mb)}
            color="var(--purple, #a78bfa)"
          />
        )}
      </div>

      {/* Sparkline chart */}
      {history.length > 1 && (
        <div className="card">
          <div className="card-title" style={{ marginBottom: 12 }}>Last 60s</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            {/* CPU */}
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 4 }}>
                <span className="text-muted">CPU</span>
                <span style={{ color: cpuColor, fontWeight: 600 }}>{metrics.cpu_pct}%</span>
              </div>
              <svg width="100%" height={sparkH} viewBox={`0 0 ${sparkW} ${sparkH}`} preserveAspectRatio="none">
                {sparkPath(history.map(h => h.cpu), cpuColor)}
              </svg>
            </div>
            {/* RAM */}
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 4 }}>
                <span className="text-muted">RAM</span>
                <span style={{ color: ramColor, fontWeight: 600 }}>{metrics.ram_pct}%</span>
              </div>
              <svg width="100%" height={sparkH} viewBox={`0 0 ${sparkW} ${sparkH}`} preserveAspectRatio="none">
                {sparkPath(history.map(h => h.ram), ramColor)}
              </svg>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

// placeholder — server tells us cores via metrics but we can just show a dash if unknown
function os_cpu_count() { return '—' }
