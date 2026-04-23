# BikAI Local Server — Documentation Index

> **For AI assistants**: Start here. Read `architecture.md` first for the big picture, then the specific file you need to modify.

## Files in This Folder

| File | What it covers |
|---|---|
| [architecture.md](architecture.md) | **Start here** — two-process design, file structure, data flow, key decisions |
| [controller.md](controller.md) | `controller.py` — all API endpoints, metrics SSE, nginx management, download system |
| [server.md](server.md) | `server.py` — LlamaPool, inference endpoints, streaming, OpenAI compatibility |
| [cli.md](cli.md) | `cli.py` — all `bikai` commands, download logic, daemon management |
| [ui.md](ui.md) | `ui/` — React pages, routing, api.ts, adding new pages |
| [setup.md](setup.md) | `setup.sh` — installer steps, systemd, manual install, .env variables |

## Quick Context for AI Assistants

### The 3 things that matter most:

1. **Two processes**: `controller.py` (port 8001, always running) spawns/kills `server.py` (port 8000, AI inference). Never merge them.

2. **`start_new_session=True`**: Every `subprocess.Popen` that creates a daemon MUST have this. Without it, processes suspend when the terminal closes (SIGHUP).

3. **`asyncio.to_thread()`**: Every `subprocess.run()` inside an `async` FastAPI handler MUST be wrapped with this. Without it, the uvicorn event loop blocks and ALL requests hang.

### Where things live:

| Task | File |
|---|---|
| Add a new API endpoint | `controller.py` |
| Add a new inference feature | `server.py` |
| Add a new CLI command | `cli.py` |
| Add a new UI page | `ui/src/pages/` + update `ui/src/App.tsx` |
| Add a new API call from UI | `ui/src/api.ts` |
| Change install behaviour | `setup.sh` |

### Common patterns:

**New controller endpoint (protected):**
```python
@app.post("/api/controller/my-endpoint", dependencies=[Depends(require_api_key)])
async def api_my_endpoint(req: MyRequest):
    # use asyncio.to_thread() for any subprocess.run() calls
    return {"ok": True}
```

**New UI page:**
1. Create `ui/src/pages/MyPage.tsx`
2. Add to `App.tsx`: `Page` type, `VALID_PAGES`, `PAGES` array, `PAGE_TITLE`, import, render
3. Add API calls to `api.ts` if needed
4. Run `npm run build` in `ui/`

**New CLI command:**
```python
@cli.command()
@click.option("--flag", is_flag=True)
def my_command(flag):
    """Description for --help."""
    _header("My Command")
    _ok("Done")
```
