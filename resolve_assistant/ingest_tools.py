"""
Ingest MCP tools: ingest_footage and ingest_status.
"""

import shutil
import threading
from pathlib import Path
from typing import Optional

from .config import mcp, OLLAMA_BASE_URL
from .media import (
    list_all_videos,
    list_all_audio,
    list_pending_videos,
    list_pending_audio,
)
from .ingest_worker import (
    _ingest_worker,
    _active_workers,
    _write_progress,
    _read_progress,
)


@mcp.tool
def ingest_footage(
    folder_path: str,
    instruction: Optional[str] = None,
    backend: str = "ollama",
    fps: float = 10.0,
) -> str:
    """
    Scan a folder for video and audio files and analyze them.
    Launches a background worker that processes ALL pending files.
    Returns immediately — use ingest_status() to monitor progress.
    Files with existing .json sidecars are skipped automatically.

    If *instruction* is provided, a timeline build is automatically triggered
    once all sidecars are written — no manual follow-up needed.

    *backend*: "ollama" (default, local Gemma 4 e4b) or "gemini" (cloud API).
    *fps*: Frame sampling rate for Ollama backend (default 10.0).
           Frames are composited into a contact sheet grid for temporal analysis.
           Ignored for Gemini backend.
    """
    root = Path(folder_path).resolve()
    if not root.is_dir():
        return f"Error: '{folder_path}' is not a valid directory."

    all_videos = list_all_videos(root)
    all_audio = list_all_audio(root)
    if not all_videos and not all_audio:
        return f"No media files found in {root}"

    pending_v = list_pending_videos(root)
    pending_a = list_pending_audio(root)
    pending = pending_v + pending_a
    total = len(all_videos) + len(all_audio)

    if not pending:
        return f"All {total} file(s) already have sidecars. Nothing to do."

    key = str(root)
    if key in _active_workers and _active_workers[key].is_alive():
        progress = _read_progress(root)
        if progress:
            return (
                f"Ingestion already running: {progress.get('current_step', '?')} "
                f"{progress.get('current_file', '?')} "
                f"({progress.get('completed', '?')}/{progress.get('total', '?')})"
            )
        return "Ingestion already running."

    if backend == "ollama":
        # Verify Ollama is running
        try:
            import urllib.request

            urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        except Exception:
            return f"Error: Ollama not reachable at {OLLAMA_BASE_URL}. Is it running?"
    else:
        if not Path("/usr/bin/avconvert").exists():
            return "Error: avconvert not found. Requires macOS 13+ with Xcode command-line tools."

    if not shutil.which("ffprobe"):
        return "Error: ffprobe not found on PATH. Install: brew install ffmpeg"

    already_done = total - len(pending)
    _write_progress(
        root,
        {
            "status": "starting",
            "current_file": pending[0].name,
            "current_step": "queued",
            "completed": already_done,
            "total": total,
            "errors": [],
        },
    )

    thread = threading.Thread(
        target=_ingest_worker,
        args=(root, instruction, backend, fps),
        daemon=True,
    )
    thread.start()
    _active_workers[key] = thread

    engine = "Ollama/Gemma4" if backend == "ollama" else "Gemini"
    parts = [f"Ingestion started for {len(pending)} file(s) via {engine}"]
    if pending_v and backend != "ollama":
        parts.append(f"({len(pending_v)} video via avconvert)")
    elif pending_v:
        parts.append(f"({len(pending_v)} video at {fps} fps)")
    if pending_a:
        parts.append(f"({len(pending_a)} audio)")
    if already_done:
        parts.append(f"{already_done} already done.")
    parts.append(f"Use ingest_status('{folder_path}') to monitor.")
    return " ".join(parts)


@mcp.tool
def ingest_status(folder_path: str) -> str:
    """
    Check progress of a running or completed ingestion job.
    Returns current file, step (transcoding/uploading/analyzing),
    completion count, and any errors.
    """
    root = Path(folder_path).resolve()
    if not root.is_dir():
        return f"Error: '{folder_path}' is not a valid directory."

    progress = _read_progress(root)
    if progress is None:
        pending = list_pending_videos(root) + list_pending_audio(root)
        total = len(list_all_videos(root)) + len(list_all_audio(root))
        if not pending:
            return f"All {total} file(s) have sidecars. No ingestion needed."
        return (
            f"{len(pending)} of {total} file(s) pending. Run ingest_footage to start."
        )

    status = progress.get("status", "unknown")
    completed = progress.get("completed", 0)
    total = progress.get("total", 0)
    current = progress.get("current_file")
    step = progress.get("current_step")
    errors = progress.get("errors", [])

    if status == "complete":
        msg = f"Ingestion complete: {completed}/{total} files analyzed."
        if errors:
            msg += f"\nErrors ({len(errors)}):\n  " + "\n  ".join(errors)
        return msg

    if status == "running":
        msg = f"Ingestion running: {completed}/{total} done. Now {step} {current}."
        if errors:
            msg += f"\n{len(errors)} error(s) so far."
        return msg

    return f"Ingestion status: {status}. {completed}/{total} done."


@mcp.tool
def ingest_drill_down(
    folder_path: str,
    clip_name: str,
    start_sec: float,
    end_sec: float,
    fps: float = 4.0,
) -> str:
    """
    Re-analyze a specific time range of a clip at higher frame density.

    Use after a coarse ingest pass to get precise segment boundaries or
    investigate a specific moment. Merges high-density segments into the
    existing sidecar, replacing any segments that overlap the drill range.

    Only works with the Ollama backend (local analysis).

    *clip_name*: filename of the clip (e.g. "A001_02031700_C006.mov")
    *start_sec*: start of the range to re-analyze
    *end_sec*: end of the range to re-analyze
    *fps*: frame sampling rate (default 4.0 = 4 frames/sec for detailed analysis)
    """
    import json as _json
    import urllib.request
    from .ollama_analyzer import analyze_video

    root = Path(folder_path).resolve()
    if not root.is_dir():
        return f"Error: '{folder_path}' is not a valid directory."

    # Find the clip
    clip_path = root / clip_name
    if not clip_path.exists():
        matches = [f for f in root.iterdir() if f.name.lower() == clip_name.lower()]
        if matches:
            clip_path = matches[0]
        else:
            return f"Error: clip '{clip_name}' not found in {root}"

    sidecar_path = clip_path.with_suffix(clip_path.suffix + ".json")

    # Verify Ollama is running
    try:
        urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
    except Exception:
        return f"Error: Ollama not reachable at {OLLAMA_BASE_URL}. Is it running?"

    try:
        result = analyze_video(
            clip_path,
            fps=fps,
            start_sec=start_sec,
            end_sec=end_sec,
        )

        new_segments = result.get("segments", [])

        if sidecar_path.exists():
            existing = _json.loads(sidecar_path.read_text(encoding="utf-8"))
            old_segments = existing.get("segments", [])

            # Keep segments fully outside the drill range
            kept = [
                s
                for s in old_segments
                if s["end_sec"] <= start_sec or s["start_sec"] >= end_sec
            ]
            kept.extend(new_segments)
            kept.sort(key=lambda s: s["start_sec"])

            existing["segments"] = kept
            sidecar_path.write_text(_json.dumps(existing, indent=2), encoding="utf-8")
            return (
                f"Drill-down complete: {clip_name} [{start_sec:.1f}s-{end_sec:.1f}s] "
                f"at {fps} fps. {len(new_segments)} new segments merged "
                f"into existing sidecar ({len(kept)} total segments)."
            )
        else:
            sidecar_path.write_text(_json.dumps(result, indent=2), encoding="utf-8")
            return (
                f"Drill-down complete: {clip_name} [{start_sec:.1f}s-{end_sec:.1f}s] "
                f"at {fps} fps. {len(new_segments)} segments written."
            )

    except Exception as exc:
        return f"Drill-down failed for {clip_name}: {exc}"
