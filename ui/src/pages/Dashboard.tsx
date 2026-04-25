import { useState, useEffect, useCallback } from "react";
import {
  fetchStatus,
  fetchModels,
  startServer,
  stopServer,
  restartServer,
} from "../api";
import type { StatusData, ModelInfo, StartPayload } from "../api";
import {
  Play,
  Square,
  RotateCcw,
  Cpu,
  Database,
  Zap,
  Clock,
  Activity,
} from "lucide-react";

interface Alert {
  type: "success" | "error" | "info";
  msg: string;
}

export default function Dashboard() {
  const [status, setStatus] = useState<StatusData | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState<string | null>(null);
  const [alert, setAlert] = useState<Alert | null>(null);

  // Server reconfigure form
  const [form, setForm] = useState<StartPayload>({
    model: "",
    parallel: 4,
    port: 8000,
    ctx: 4096,
    threads: 4,
  });
  const [formDirty, setFormDirty] = useState(false);
  const [rawFields, setRawFields] = useState({
    parallel: "4",
    port: "8000",
    ctx: "4096",
    threads: "4",
  });

  const showAlert = useCallback((type: Alert["type"], msg: string) => {
    setAlert({ type, msg });
    setTimeout(() => setAlert(null), 6000);
  }, []);

  const load = useCallback(async () => {
    try {
      const [s, m] = await Promise.all([fetchStatus(), fetchModels()]);
      setStatus(s);
      setModels(m.models);
      if (!formDirty) {
        setForm({
          model: s.model_path || m.active_path || (m.models[0]?.path ?? ""),
          parallel: s.parallel,
          port: s.port,
          ctx: s.ctx,
          threads: s.threads,
        });
        setRawFields({
          parallel: String(s.parallel),
          port: String(s.port),
          ctx: String(s.ctx),
          threads: String(s.threads),
        });
      }
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      showAlert("error", `Failed to load status: ${msg}`);
    } finally {
      setLoading(false);
    }
  }, [formDirty, showAlert]);

  useEffect(() => {
    load();
  }, [load]);
  useEffect(() => {
    const id = setInterval(load, 5000);
    return () => clearInterval(id);
  }, [load]);

  function patch(k: keyof StartPayload, v: string | number) {
    setForm((f) => ({ ...f, [k]: v }));
    setFormDirty(true);
  }

  async function doStart() {
    setBusy("start");
    setAlert(null);
    try {
      await startServer({
        ...form,
        parallel: parseInt(rawFields.parallel) || 1,
        port: parseInt(rawFields.port) || 8000,
        ctx: parseInt(rawFields.ctx) || 4096,
        threads: parseInt(rawFields.threads) || 1,
      });
      setFormDirty(false);
      showAlert("success", "Server started successfully.");
      setTimeout(load, 2000);
    } catch (e: unknown) {
      const msg =
        (e as { response?: { data?: { detail?: string } } })?.response?.data
          ?.detail ?? String(e);
      showAlert("error", `Start failed: ${msg}`);
    } finally {
      setBusy(null);
    }
  }

  async function doStop() {
    setBusy("stop");
    setAlert(null);
    try {
      await stopServer();
      showAlert("success", "Server stopped.");
      setTimeout(load, 1500);
    } catch (e: unknown) {
      const msg =
        (e as { response?: { data?: { detail?: string } } })?.response?.data
          ?.detail ?? String(e);
      showAlert("error", `Stop failed: ${msg}`);
    } finally {
      setBusy(null);
    }
  }

  async function doRestart() {
    setBusy("restart");
    setAlert(null);
    try {
      await restartServer();
      showAlert("success", "Server restarted. Loading model…");
      setTimeout(load, 3000);
    } catch (e: unknown) {
      const msg =
        (e as { response?: { data?: { detail?: string } } })?.response?.data
          ?.detail ?? String(e);
      showAlert("error", `Restart failed: ${msg}`);
    } finally {
      setBusy(null);
    }
  }

  const isRunning = status?.running ?? false;
  const isBusy = busy !== null;

  if (loading) {
    return (
      <div style={{ padding: 24, color: "var(--muted)" }}>
        <span className="spin">⟳</span> Loading…
      </div>
    );
  }

  return (
    <div>
      {alert && (
        <div
          className={`alert alert-${alert.type === "error" ? "error" : alert.type === "success" ? "success" : "info"}`}
        >
          <Activity size={15} />
          {alert.msg}
        </div>
      )}

      {/* Stat row */}
      <div className="stat-grid mb-16">
        <div className="stat-box">
          <div className="stat-label">Status</div>
          <div className="stat-value" style={{ fontSize: 14, marginTop: 4 }}>
            <span
              className={`status-pill ${isRunning ? "running" : "stopped"}`}
            >
              <span className={`dot ${isRunning ? "pulse" : ""}`} />
              {isRunning ? "Running" : "Stopped"}
            </span>
          </div>
          {status?.pid && <div className="stat-sub">PID {status.pid}</div>}
        </div>
        <div className="stat-box">
          <div className="stat-label">Uptime</div>
          <div className="stat-value blue" style={{ fontSize: 16 }}>
            {status?.uptime ?? "—"}
          </div>
          <div className="stat-sub">since last start</div>
        </div>
        <div className="stat-box">
          <div className="stat-label">Model</div>
          <div
            className="stat-value"
            style={{ fontSize: 13, fontWeight: 600, wordBreak: "break-all" }}
          >
            {status?.model_name ?? "—"}
          </div>
          <div className="stat-sub">{status?.model_size ?? ""}</div>
        </div>
        <div className="stat-box">
          <div className="stat-label">Parallel Slots</div>
          <div className="stat-value yellow">{status?.parallel ?? "—"}</div>
          <div className="stat-sub">concurrent requests</div>
        </div>
        <div className="stat-box">
          <div className="stat-label">Context Length</div>
          <div className="stat-value purple">
            {status?.ctx ? status.ctx.toLocaleString() : "—"}
          </div>
          <div className="stat-sub">tokens per slot</div>
        </div>
        <div className="stat-box">
          <div className="stat-label">Memory Usage</div>
          <div className="stat-value orange">
            {status?.mem_mb ? `${status.mem_mb.toLocaleString()} MB` : "—"}
          </div>
          <div className="stat-sub">controller process</div>
        </div>
      </div>

      {/* Configure + Control */}
      <div className="card">
        <div className="card-title">Server Configuration</div>

        {/* Model select */}
        <div className="form-grid">
          <div className="form-group" style={{ gridColumn: "1 / -1" }}>
            <label className="form-label">Model</label>
            <select
              className="form-select"
              value={form.model}
              onChange={(e) => patch("model", e.target.value)}
            >
              {models.length === 0 && (
                <option value="">— No models found —</option>
              )}
              {models.map((m) => (
                <option key={m.path} value={m.path}>
                  {m.name} ({m.size}){m.active ? "  ← active" : ""}
                </option>
              ))}
            </select>
            <span className="form-hint">
              Select a downloaded model. Download more in the{" "}
              <strong>Models</strong> tab.
            </span>
          </div>
        </div>

        <div className="form-grid">
          <div className="form-group">
            <label className="form-label">
              <span className="flex items-center gap-8">
                <Zap size={12} /> Parallel Slots
              </span>
            </label>
            <input
              className="form-input"
              type="number"
              min={1}
              max={32}
              value={rawFields.parallel}
              onChange={(e) => {
                setRawFields((f) => ({ ...f, parallel: e.target.value }));
                setFormDirty(true);
              }}
            />
            <span className="form-hint">
              Concurrent requests — more = more RAM
            </span>
          </div>
          <div className="form-group">
            <label className="form-label">
              <span className="flex items-center gap-8">
                <Activity size={12} /> Port
              </span>
            </label>
            <input
              className="form-input"
              type="number"
              min={1024}
              max={65535}
              value={rawFields.port}
              onChange={(e) => {
                setRawFields((f) => ({ ...f, port: e.target.value }));
                setFormDirty(true);
              }}
            />
            <span className="form-hint">Inference API port</span>
          </div>
          <div className="form-group">
            <label className="form-label">
              <span className="flex items-center gap-8">
                <Database size={12} /> Context Length
              </span>
            </label>
            <input
              className="form-input"
              type="number"
              min={512}
              max={8192}
              step={512}
              value={rawFields.ctx}
              onChange={(e) => {
                setRawFields((f) => ({ ...f, ctx: e.target.value }));
                setFormDirty(true);
              }}
            />
            <span className="form-hint">
              Tokens per slot. Max 8192 for Gemma.
            </span>
          </div>
          <div className="form-group">
            <label className="form-label">
              <span className="flex items-center gap-8">
                <Cpu size={12} /> CPU Threads
              </span>
            </label>
            <input
              className="form-input"
              type="number"
              min={1}
              max={64}
              value={rawFields.threads}
              onChange={(e) => {
                setRawFields((f) => ({ ...f, threads: e.target.value }));
                setFormDirty(true);
              }}
            />
            <span className="form-hint">Inference CPU threads</span>
          </div>
        </div>

        <div className="divider" />

        <div className="btn-group">
          <button
            className="btn btn-primary"
            onClick={doStart}
            disabled={isBusy || !form.model}
          >
            {busy === "start" ? (
              <>
                <span className="spin">⟳</span> Starting…
              </>
            ) : (
              <>
                <Play size={14} />{" "}
                {isRunning ? "Save & Restart" : "Start Server"}
              </>
            )}
          </button>
          <button
            className="btn btn-danger"
            onClick={doStop}
            disabled={isBusy || !isRunning}
          >
            {busy === "stop" ? (
              <>
                <span className="spin">⟳</span> Stopping…
              </>
            ) : (
              <>
                <Square size={14} /> Stop Server
              </>
            )}
          </button>
          <button
            className="btn btn-secondary"
            onClick={doRestart}
            disabled={isBusy || !isRunning}
          >
            {busy === "restart" ? (
              <>
                <span className="spin">⟳</span> Restarting…
              </>
            ) : (
              <>
                <RotateCcw size={14} /> Restart
              </>
            )}
          </button>
          {formDirty && (
            <span className="badge warn" style={{ marginLeft: 4 }}>
              <Clock size={10} style={{ marginRight: 4 }} />
              Unsaved changes
            </span>
          )}
        </div>
      </div>

      {/* Quick info */}
      {isRunning && status && (
        <div className="card">
          <div className="card-title">Endpoints</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {[
              ["GET", "/health", "Health check (no auth)"],
              ["GET", `/v1/models`, "List models (X-API-Key required)"],
              [
                "POST",
                `/v1/chat/completions`,
                "Chat completions — OpenAI-compatible, streaming",
              ],
              ["POST", `/generate`, "Simple text generation"],
              ["GET", `/docs`, "Swagger / interactive API docs"],
            ].map(([method, path, desc]) => (
              <div
                key={path}
                style={{ display: "flex", gap: 10, alignItems: "flex-start" }}
              >
                <span
                  className={`badge ${method === "GET" ? "info" : "active"}`}
                  style={{ marginTop: 1, flexShrink: 0 }}
                >
                  {method}
                </span>
                <div>
                  <span className="mono" style={{ fontSize: 13 }}>
                    http://localhost:{status.port}
                    {path}
                  </span>
                  <div
                    className="text-muted"
                    style={{ fontSize: 11, marginTop: 2 }}
                  >
                    {desc}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
