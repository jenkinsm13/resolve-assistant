---
name: resolve-conventions
description: Project conventions and patterns for resolve-assistant development
user-invocable: false
---

# resolve-assistant Conventions

## Tool Pattern

Tools are defined in `ingest.py` and `build.py`:

```python
from .config import mcp

@mcp.tool
def ingest_footage(folder_path: str) -> str:
    """Docstring — this becomes the MCP tool description."""
    # Start background worker, return status message
```

## Background Worker Pattern

Long-running tasks use background threads:
- `ingest_worker.py` — runs footage analysis in a thread
- `build_worker.py` — runs edit planning + timeline building in a thread
- Tools return immediately with a polling message
- Status tools check worker state

## Sidecar JSON

- Saved as `<filename>.json` next to source media
- Schema in `schemas.py` (Pydantic: `VideoSidecar`, `AudioSidecar`, `Segment`)
- Re-ingest skips files with existing sidecars
- Delete sidecar to force re-analysis

## Imports

- Always **relative**: `from .config import mcp, client, MODEL`
- Never absolute `resolve_assistant.` imports

## Gemini

- `client` and `MODEL` imported from `config.py`
- `GEMINI_API_KEY` is **required** — server won't start without it
- Use `retry.py` for all Gemini API calls (exponential backoff)

## Prompts

- `prompts.py` — shared utilities
- `prompts_analysis.py` — ingest prompts (footage analysis)
- `prompts_edit.py` — build prompts (edit planning)
- `prompts_music.py` — music analysis prompts
