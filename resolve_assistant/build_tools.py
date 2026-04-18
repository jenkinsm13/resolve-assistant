"""
Build timeline MCP tools: build_timeline and build_status.
"""

import json
from pathlib import Path
from typing import Optional

from .config import log, mcp
from .media import load_sidecars
from .build_worker import (
    _build_worker,
    _active_build_workers,
    _read_build_progress,
)


@mcp.tool
def build_timeline(folder_path: str, instruction: str) -> str:
    """
    Read all sidecar JSONs in *folder_path*, upload the actual video/audio
    proxy files to Gemini so it can watch the footage, then send the editing
    instruction.  Gemini plans the EDL and the result is built directly into
    the active DaVinci Resolve project via AppendToTimeline.

    If a cached .edl.json exists from a previous build, the timeline is
    re-rendered from that cached plan without re-querying Gemini.  Delete
    the .edl.json to force a fresh Gemini query.
    """
    root = Path(folder_path).resolve()
    if not root.is_dir():
        return f"Error: '{folder_path}' is not a valid directory."

    sidecars = load_sidecars(root)
    if not sidecars:
        return "No sidecar JSONs found. Run ingest_footage first."

    key = str(root)
    if key in _active_build_workers and _active_build_workers[key].is_alive():
        progress = _read_build_progress(root)
        if progress:
            return f"Build already running: {progress.get('status', '?')} — {progress.get('detail', '?')}"
        return "Build already running."

    cached_plan: Optional[dict] = None
    edl_files = sorted(f for f in root.glob("*.edl.json") if not f.name.startswith("."))
    if edl_files:
        try:
            cached_plan = json.loads(edl_files[0].read_text(encoding="utf-8"))
            if not cached_plan.get("cuts"):
                cached_plan = None
            else:
                log.info("Using cached edit plan: %s", edl_files[0].name)
        except (json.JSONDecodeError, OSError):
            cached_plan = None

    if cached_plan:
        # Run synchronously — Resolve's AppendToTimeline requires the main thread.
        _build_worker(root, sidecars, instruction, cached_plan=cached_plan)
        progress = _read_build_progress(root)
        detail = progress.get("detail", "") if progress else "done"
        return f"Re-rendering from cached edit plan ({len(cached_plan.get('cuts', []))} cuts). {detail}"
    return (
        f"Timeline build started ({len(sidecars)} sidecars, uploading proxies to Gemini). "
        f"Use build_status('{folder_path}') to monitor."
    )


@mcp.tool
def build_status(folder_path: str) -> str:
    """
    Check progress of a running or completed timeline build.
    Returns current step (uploading/editing/building/complete/error) and XML path when done.
    """
    root = Path(folder_path).resolve()
    if not root.is_dir():
        return f"Error: '{folder_path}' is not a valid directory."

    progress = _read_build_progress(root)
    if progress is None:
        return "No build in progress. Run build_timeline first."

    status = progress.get("status", "unknown")
    detail = progress.get("detail", "")
    error = progress.get("error")
    xml_path = progress.get("xml_path")

    if status == "complete":
        msg = f"Build complete: {detail}"
        if xml_path:
            msg += f"\nXML: {xml_path}"
        return msg

    if status == "error":
        msg = f"Build failed: {detail}"
        if error:
            msg += f"\n{error}"
        return msg

    return f"Build {status}: {detail}"
