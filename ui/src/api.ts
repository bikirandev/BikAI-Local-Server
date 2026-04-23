import axios from 'axios'

// Base URL — in dev the proxy forwards /api → controller:8001
// In prod (served by controller) it's same-origin.
const BASE = ''

function getKey(): string {
  return localStorage.getItem('bikai_api_key') || ''
}

const http = axios.create({ baseURL: BASE })

http.interceptors.request.use((cfg) => {
  const key = getKey()
  if (key) cfg.headers['X-API-Key'] = key
  return cfg
})

export function saveKey(key: string) {
  localStorage.setItem('bikai_api_key', key)
}

export function clearKey() {
  localStorage.removeItem('bikai_api_key')
}

export function hasKey(): boolean {
  return !!getKey()
}

// ── Status ─────────────────────────────────────────────────────────────────

export interface NginxInfo {
  installed: boolean
  enabled: boolean
  active: boolean
  domain: string
}

export interface StatusData {
  running: boolean
  pid: number | null
  uptime: string
  model_name: string
  model_path: string
  model_size: string
  parallel: number
  ctx: number
  threads: number
  rate_limit: string
  port: number
  controller_port: number
  domain: string
  mem_mb: number
  nginx: NginxInfo
}

export async function fetchStatus(): Promise<StatusData> {
  const r = await http.get('/api/controller/status')
  return r.data
}

// ── Models ─────────────────────────────────────────────────────────────────

export interface ModelInfo {
  name: string
  path: string
  size: string
  active: boolean
}

export async function fetchModels(): Promise<{ models: ModelInfo[]; active_path: string }> {
  const r = await http.get('/api/controller/models')
  return r.data
}

// ── Start / Stop / Restart ──────────────────────────────────────────────────

export interface StartPayload {
  model: string
  parallel: number
  port: number
  ctx: number
  threads: number
}

export async function startServer(payload: StartPayload) {
  const r = await http.post('/api/controller/start', payload)
  return r.data
}

export async function stopServer() {
  const r = await http.post('/api/controller/stop')
  return r.data
}

export async function restartServer() {
  const r = await http.post('/api/controller/restart')
  return r.data
}

// ── Logs ───────────────────────────────────────────────────────────────────

export async function fetchLogs(lines = 200): Promise<{ lines: string[] }> {
  const r = await http.get(`/api/controller/logs?lines=${lines}`)
  return r.data
}

// ── Nginx ──────────────────────────────────────────────────────────────────

export interface NginxConfig {
  installed: boolean
  enabled: boolean
  active: boolean
  config_text: string
  domain: string
  worker_processes: string
  worker_connections: number
}

export interface NginxPayload {
  domain: string
  ai_port: number
  ctrl_port: number
  listen_port: number
  ssl: boolean
  cors_origin: string
  read_timeout: number
  client_max_body_size: string
  gzip: boolean
  worker_processes: string
  worker_connections: number
}

export async function fetchNginxConfig(): Promise<NginxConfig> {
  const r = await http.get('/api/controller/nginx')
  return r.data
}

export async function applyNginxConfig(payload: NginxPayload) {
  const r = await http.post('/api/controller/nginx', payload)
  return r.data
}

export async function fetchNginxStatus() {
  const r = await http.get('/api/controller/nginx/status')
  return r.data
}

// ── Token ──────────────────────────────────────────────────────────────────

export async function fetchToken(): Promise<{ key: string }> {
  const r = await http.get('/api/controller/token')
  return r.data
}

export async function rotateToken(): Promise<{ ok: boolean; key: string }> {
  const r = await http.post('/api/controller/token/new')
  return r.data
}

// ── Download ───────────────────────────────────────────────────────────────

export interface DownloadPayload {
  type: 'gdrive' | 'huggingface' | 'url'
  id?: string
  repo?: string
  file?: string
  url?: string
  set_default?: boolean
}

export async function downloadModel(payload: DownloadPayload) {
  const r = await http.post('/api/controller/download', payload)
  return r.data
}

export interface DownloadStatus {
  active: boolean
  lines: string[]
}

export async function fetchDownloadStatus(): Promise<DownloadStatus> {
  const r = await http.get('/api/controller/download/status')
  return r.data
}

export async function deleteModel(name: string) {
  const r = await http.delete(`/api/controller/models/${encodeURIComponent(name)}`)
  return r.data
}
