# Code Reviewer

Review resolve-assistant code for consistency and correctness.

## Checklist

1. **Import consistency**: All modules use relative imports (`from .config import ...`)
2. **Gemini calls**: All Gemini API calls go through `retry.py` for exponential backoff
3. **Sidecar schema**: Sidecar JSON creation/loading uses Pydantic models from `schemas.py`
4. **Background workers**: `ingest_worker.py` and `build_worker.py` handle exceptions and update status
5. **Docstrings**: All 4 `@mcp.tool` functions have docstrings
6. **Config usage**: `client` and `MODEL` imported from config, not created locally
7. **Proxy handling**: Transcoded files use `.gemini.mp4` suffix and respect `GEMINI_MAX_LONG_EDGE`

## How to Review

Read each module and verify against the checklist. Report violations grouped by module.
