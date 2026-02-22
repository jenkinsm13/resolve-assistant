# resolve-assistant

AI-powered video editing assistant. Uses Google Gemini to analyze footage, plan professional edits, and build timelines in DaVinci Resolve.

## Architecture

- **Package**: `resolve_assistant/` — all modules use relative imports (`from .config import mcp`)
- **Entry point**: `resolve_assistant/__init__.py` → `main()` → `mcp.run()`
- **Tool registration**: `ingest.py` and `build.py` are imported in `__init__.py`, registering 4 `@mcp.tool` functions
- **Gemini is required** — `config.py` raises `RuntimeError` if `GEMINI_API_KEY` is not set

## The 4 MCP Tools

| Tool | Module | Purpose |
|------|--------|---------|
| `ingest_footage(folder_path)` | `ingest.py` | Analyze all clips with Gemini (background) |
| `ingest_status(folder_path)` | `ingest.py` | Poll ingest progress |
| `build_timeline(folder_path, instruction)` | `build.py` | Plan edit + build timeline (background) |
| `build_status(folder_path)` | `build.py` | Poll build progress |

## Pipeline

```
Raw Footage → ffprobe → transcode proxy → Gemini analysis → sidecar JSON
                                                                ↓
Instruction → upload proxies → Gemini edit plan → AppendToTimeline → Resolve timeline
                                    ↓
                            FCP7 XML + Director's Notes + Voiceover + Music Brief
```

## Key Patterns

### Sidecar JSON architecture
- Each analyzed clip gets a `.json` sidecar file next to the source (e.g., `clip.mp4.json`)
- Re-running ingest skips files with existing sidecars
- Delete a sidecar to force re-analysis
- Schema defined in `schemas.py` (Pydantic models: `VideoSidecar`, `AudioSidecar`, `Segment`)

### Background workers
- `ingest_worker.py` and `build_worker.py` run in background threads
- Tools return immediately with a status message
- Use `_status()` tools to poll for completion

### Transcoding (`transcode.py`)
- Apple Silicon: `hevc_videotoolbox` (hardware)
- Other platforms: `libx265` (software)
- Proxy files saved as `.gemini.mp4` next to source
- Resolution capped at 1280px longest edge
- Files already in H.264/H.265, under 2GB, ≤1280px are uploaded as-is

### Resolve connection (`resolve.py`)
- `get_resolve()` returns Resolve scripting object (or `None`)
- `_boilerplate()` returns `(resolve, project, media_pool, timeline)`
- Cross-platform: macOS, Windows, Linux module paths handled automatically

### Prompts
- `prompts.py` — shared prompt utilities
- `prompts_analysis.py` — Gemini prompts for footage analysis (ingest)
- `prompts_edit.py` — Gemini prompts for edit planning (build)
- `prompts_music.py` — Gemini prompts for music analysis

### Build outputs (`outputs.py`)
- Director's notes (`.md`)
- Voiceover script (`.txt` + `.json`)
- Music production brief (`.md` + `.json`)
- Edit plan (`.edl.json`)

### Error handling & retry
- `errors.py` — `ResolveError` hierarchy + `@safe_resolve_call` decorator
- `retry.py` — Gemini API retry with exponential backoff

## Constants (`config.py`)
- `VIDEO_EXTS`: .mp4, .mov, .mxf, .avi, .webm, .mkv, .r3d, .braw
- `AUDIO_EXTS`: .mp3, .wav, .aac, .flac, .ogg, .m4a
- `GEMINI_MAX_BYTES`: 2 GB
- `GEMINI_MAX_LONG_EDGE`: 1280px
- `MODEL`: gemini-3-flash-preview

## Build & Publish

```bash
# Bump version in pyproject.toml, then:
uv build && uv publish --token $PYPI_TOKEN
git add -A && git commit -m "..." && git push origin main
```

## Related

- **[resolve-mcp](https://github.com/jenkinsm13/resolve-mcp)** — 215+ MCP tools for direct Resolve control (separate MCP server)
- Install both: `pip install resolve-mcp[assistant]`
