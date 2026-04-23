# ui/ — React Control Panel

## Overview

The control panel is a React + TypeScript + Vite SPA served at `/controller/ui`.  
It is built to `ui/dist/` and served as static files by `controller.py`.

**It is gitignored** — `ui/dist/` is never committed. The controller auto-builds it on startup if missing.

---

## Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| React | 19 | UI framework |
| TypeScript | 6 | Type safety |
| Vite | 8 | Build tool + dev server |
| axios | 1.x | HTTP client |
| lucide-react | 1.x | Icons |
| eslint-plugin-react-hooks | 7 | Linting |

---

## Project Structure

```
ui/
├── src/
│   ├── App.tsx          # Root: lock screen, sidebar, hash routing
│   ├── api.ts           # All HTTP calls to /api/controller/*
│   ├── index.css        # All styles (CSS variables, components)
│   ├── main.tsx         # Entry point
│   └── pages/
│       ├── Dashboard.tsx    # Server control + config form
│       ├── Models.tsx       # Model list, download, delete
│       ├── Stats.tsx        # Real-time system metrics
│       ├── Nginx.tsx        # nginx configuration
│       ├── Logs.tsx         # Server log viewer
│       └── Settings.tsx     # API key management
├── dist/               # Built output (gitignored, auto-generated)
├── package.json
├── vite.config.ts
├── tsconfig.app.json
└── eslint.config.js
```

---

## Vite Config

```typescript
// vite.config.ts
export default defineConfig({
  plugins: [react()],
  base: '/controller/',          // All assets served under /controller/
  server: {
    proxy: {
      '/api': 'http://localhost:8001',  // Dev proxy
    },
  },
})
```

The `base: '/controller/'` setting means all built asset paths are prefixed with `/controller/`.  
In production (served by the controller), same-origin requests to `/api/*` work without a proxy.

---

## App.tsx — Routing & Auth

### Lock Screen
- Shown when `localStorage` has no `bikai_api_key`
- User enters API key → `saveKey(key)` → `fetchStatus()` to validate
- On 401: `clearKey()` and show error
- On success: sets `unlocked = true`

### Hash Routing
```typescript
type Page = 'dashboard' | 'models' | 'nginx' | 'logs' | 'settings' | 'stats'

function pageFromHash(): Page  // reads window.location.hash
function navigate(p: Page)     // sets hash + React state
```

Hash-based routing is used because the SPA is served at `/controller/ui` (a sub-path). React Router's history mode would break on refresh. Hash routing (`#dashboard`) works at any path.

### Topbar
- Shows current page title
- Server running/stopped pill (polls every 5s)
- Active model name
- **Logout button** — calls `clearKey()`, returns to lock screen

---

## api.ts — API Layer

All HTTP calls go through a single axios instance with the API key header injected:

```typescript
const http = axios.create({ baseURL: '' })

http.interceptors.request.use((cfg) => {
  const key = localStorage.getItem('bikai_api_key') || ''
  if (key) cfg.headers['X-API-Key'] = key
  return cfg
})
```

### Key Functions

```typescript
// Auth
saveKey(key: string)            // localStorage.setItem
clearKey()                      // localStorage.removeItem
hasKey(): boolean               // !!getKey()

// Status
fetchStatus(): Promise<StatusData>
fetchModels(): Promise<{ models: ModelInfo[]; active_path: string }>

// Control
startServer(payload: StartPayload)
stopServer()
restartServer()

// Nginx
fetchNginxConfig(): Promise<NginxConfig>
applyNginxConfig(payload: NginxPayload)
fetchNginxStatus()

// Token
fetchToken(): Promise<{ key: string }>
rotateToken(): Promise<{ ok: boolean; key: string }>

// Downloads
downloadModel(payload: DownloadPayload)
fetchDownloadStatus(): Promise<{ active: boolean; lines: string[] }>

// Models
deleteModel(name: string)

// Logs & Metrics
fetchLogs(lines?: number): Promise<{ lines: string[] }>
// Metrics: use EventSource directly in Stats.tsx (SSE)
```

---

## Pages

### Dashboard.tsx
- **Load**: `Promise.all([fetchStatus(), fetchModels()])` on mount + every 5s
- **Form**: pre-fills with current config from status; model selector from model list
- **Actions**: Start, Stop, Restart buttons with spinners
- **State guard**: `formDirty` flag prevents form from resetting when user is editing

### Models.tsx
- **Model table**: lists all `.gguf` files with size, active status, delete button
- **Delete**: confirms via `window.confirm()`, calls `deleteModel()`, refreshes list
  - Button disabled + tooltip if model is active and server running
- **Download form**: 3 tabs — HuggingFace, Google Drive, Direct URL
- **Download progress**: polls `/api/controller/download/status` every 1.5s via `setInterval`
- **Recommended model card**: shows Gemma 3 4B card with Google Drive ID copy button; hidden once gemma3-4b is already downloaded

### Stats.tsx
- Connects to `/api/controller/metrics` via native `EventSource` API (SSE)
- Displays: CPU bar, RAM bar, Disk bar, AI process memory card
- Color coding: green < 60%, yellow < 85%, red ≥ 85%
- 60-second sparkline SVG chart for CPU and RAM history
- Auto-reconnects on error; shows connection status indicator

### Nginx.tsx
- Loads current config on mount
- Form fields: domain, ports, SSL toggle, CORS, timeout, gzip, worker settings
- Submit calls `applyNginxConfig()` which writes and reloads nginx
- Separate "Check nginx status" button

### Logs.tsx
- Auto-refreshes every 3s when `autoRefresh` is enabled
- Line count selector (50, 100, 200, 500)
- Color-coded lines: red=error, yellow=warning, green=ready/started
- Download button saves log as `.txt` file

### Settings.tsx
- Shows masked API key (toggle reveal)
- Copy to clipboard button
- Rotate key button (with confirmation dialog)
- Logout button (actually in the App.tsx topbar, not this page)

---

## ESLint Config

```javascript
// eslint.config.js — two rules disabled:
'react-hooks/react-compiler': 'off'     // Babel plugin not installed
'react-hooks/set-state-in-effect': 'off'  // Standard data-fetch pattern
```

The React Compiler ESLint rule (`react-compiler`) is disabled because the Babel plugin is not configured in `vite.config.ts`.

---

## Dev Workflow

```bash
cd ui
npm install
npm run dev    # Dev server at http://localhost:5173 (proxies /api → localhost:8001)
npm run build  # Build to ui/dist/
npm run lint   # ESLint check
```

---

## Adding a New Page

1. Create `ui/src/pages/MyPage.tsx`
2. In `App.tsx`:
   - Add `'mypage'` to `Page` type and `VALID_PAGES` array
   - Add `{ id: 'mypage', label: 'My Page', Icon: SomeIcon }` to `PAGES`
   - Add `'mypage': 'My Page'` to `PAGE_TITLE`
   - Add `import MyPage from './pages/MyPage'`
   - Add `{page === 'mypage' && <MyPage />}` in the content area
3. If it needs new API calls, add them to `api.ts` first
4. Rebuild: `npm run build`

---

## Styling

All styles are in `ui/src/index.css` using CSS custom properties:

```css
--bg: #0f1117         /* main background */
--surface: #181b24    /* card/sidebar background */
--border: #2a2d3a     /* borders */
--text: #e2e8f0       /* primary text */
--muted: #6b7280      /* secondary text */
--blue: #3b82f6
--green: #22c55e
--red: #ef4444
--yellow: #f59e0b
```

Use existing utility classes: `.card`, `.btn`, `.btn-primary`, `.btn-ghost`, `.badge`, `.form-input`, `.mono`, `.flex`, `.items-center`, `.gap-8`, etc.
