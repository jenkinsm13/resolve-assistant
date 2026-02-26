"""
Ingest background worker: Gemini upload + analysis loop, progress tracking.

Transcoding runs in parallel (up to MAX_TRANSCODE_WORKERS concurrent) via
macOS native avconvert, which uses dedicated VideoToolbox hardware encoders.
Upload + Gemini analysis remains sequential to respect API rate limits.
"""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from google.genai import types

from .config import MODEL, AUDIO_EXTS, client, log
from .retry import retry_gemini
from .schemas import VideoSidecar, AudioSidecar
from .ffprobe import ffprobe_fps, ffprobe_duration
from .transcode import prepare_for_gemini
from .media import (
    list_all_videos, list_all_audio,
    list_pending_videos, list_pending_audio,
)
from .prompts import ANALYSIS_PROMPT, AUDIO_ANALYSIS_PROMPT

_PROGRESS_FILENAME = ".ingest_progress.json"

# M1 Max has 2 dedicated HEVC encode engines — run 2 transcodes in parallel.
MAX_TRANSCODE_WORKERS = 2

# Active worker threads keyed by resolved folder path.
_active_workers: dict[str, threading.Thread] = {}


def _write_progress(root: Path, data: dict) -> None:
    (root / _PROGRESS_FILENAME).write_text(json.dumps(data, indent=2))


def _read_progress(root: Path) -> Optional[dict]:
    pf = root / _PROGRESS_FILENAME
    if not pf.exists():
        return None
    try:
        return json.loads(pf.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _transcode_one(video_path: Path) -> tuple[Path, Path | None, str | None]:
    """Transcode a single video. Returns (original, proxy_path, error_or_None)."""
    try:
        proxy = prepare_for_gemini(video_path)
        return video_path, proxy, None
    except Exception as exc:
        return video_path, None, str(exc)


def _batch_transcode(
    videos: list[Path], root: Path, already_done: int, total: int, errors: list[str],
) -> dict[str, Path]:
    """Transcode all pending videos in parallel. Returns {filename: proxy_path}."""
    proxies: dict[str, Path] = {}

    with ThreadPoolExecutor(max_workers=MAX_TRANSCODE_WORKERS) as pool:
        futures = {pool.submit(_transcode_one, v): v for v in videos}
        done_count = 0

        for future in as_completed(futures):
            original, proxy, err = future.result()
            done_count += 1

            if err:
                errors.append(f"{original.name}: {err}")
                log.error("Transcode failed: %s — %s", original.name, err)
            elif proxy:
                proxies[original.name] = proxy

            _write_progress(root, {
                "status": "running",
                "current_file": f"transcoding ({done_count}/{len(videos)})",
                "current_step": "transcoding",
                "completed": already_done,
                "total": total,
                "errors": errors,
            })

    return proxies


def _ingest_worker(root: Path, build_instruction: Optional[str] = None) -> None:
    """Background thread: transcode in parallel, then upload+analyze sequentially.

    If *build_instruction* is provided, a timeline build is automatically
    started once all sidecars are written.
    """
    pending_videos = list_pending_videos(root)
    pending_audio = list_pending_audio(root)
    total = len(list_all_videos(root)) + len(list_all_audio(root))
    already_done = total - len(pending_videos) - len(pending_audio)
    errors: list[str] = []

    # --- Phase 1: Parallel transcode (videos only) ---
    _write_progress(root, {
        "status": "running",
        "current_file": f"transcoding 0/{len(pending_videos)}",
        "current_step": "transcoding",
        "completed": already_done,
        "total": total,
        "errors": errors,
    })

    proxies = _batch_transcode(pending_videos, root, already_done, total, errors)

    # --- Phase 2: Sequential upload + analyze ---
    pending = pending_videos + pending_audio

    for i, media_path in enumerate(pending):
        sidecar_path = media_path.with_suffix(media_path.suffix + ".json")
        is_audio = media_path.suffix.lower() in AUDIO_EXTS

        _write_progress(root, {
            "status": "running", "current_file": media_path.name,
            "current_step": "uploading", "completed": already_done + i,
            "total": total, "errors": errors,
        })

        try:
            if is_audio:
                upload_path = media_path
            else:
                upload_path = proxies.get(media_path.name, media_path)

            file_ref = retry_gemini(client.files.upload, file=str(upload_path))
            while file_ref.state.name == "PROCESSING":
                time.sleep(2)
                file_ref = client.files.get(name=file_ref.name)

            if file_ref.state.name != "ACTIVE":
                errors.append(f"{media_path.name}: upload state={file_ref.state.name}")
                continue

            _write_progress(root, {
                "status": "running", "current_file": media_path.name,
                "current_step": "analyzing", "completed": already_done + i,
                "total": total, "errors": errors,
            })

            if is_audio:
                response = retry_gemini(
                    client.models.generate_content, model=MODEL,
                    contents=[file_ref, AUDIO_ANALYSIS_PROMPT],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=AudioSidecar,
                    ),
                )
            else:
                response = retry_gemini(
                    client.models.generate_content, model=MODEL,
                    contents=[file_ref, ANALYSIS_PROMPT],
                    config=types.GenerateContentConfig(
                        media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
                        response_mime_type="application/json",
                        response_schema=VideoSidecar,
                    ),
                )

            sidecar_data = json.loads(response.text)
            sidecar_data["file_path"] = str(media_path)
            sidecar_data["filename"] = media_path.name
            sidecar_data["analysis_model"] = MODEL

            if not is_audio:
                probe_fps = ffprobe_fps(media_path)
                if probe_fps:
                    sidecar_data["fps"] = round(probe_fps, 3)
                probe_dur = ffprobe_duration(media_path)
                if probe_dur:
                    sidecar_data["duration"] = round(probe_dur, 3)

            sidecar_path.write_text(json.dumps(sidecar_data, indent=2), encoding="utf-8")

        except Exception as exc:
            errors.append(f"{media_path.name}: {exc}")

    done_count = total - len(list_pending_videos(root)) - len(list_pending_audio(root))
    _write_progress(root, {
        "status": "complete", "current_file": None, "current_step": None,
        "completed": done_count, "total": total, "errors": errors,
    })

    if build_instruction:
        try:
            from .build import _build_worker  # lazy import avoids circular dep
            _build_worker(root, build_instruction)
        except Exception as exc:
            log.error("Auto-build after ingest failed: %s", exc)
