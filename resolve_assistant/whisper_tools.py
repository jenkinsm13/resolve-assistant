"""
Whisper MCP tools: transcribe_audio_file.

Thin wrapper around `resolve_assistant.whisper.transcribe_audio()` that accepts
video OR audio input and writes a `<stem>.transcript.json` sidecar next to the
source file. Exposes whisper-cpp directly so it can be called without routing
through ingest_footage's auto-fallback audio path.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path

from .config import mcp
from .whisper import transcribe_audio

log = logging.getLogger(__name__)

_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".m4v", ".avi", ".webm"}


def _extract_audio(video_path: Path) -> Path:
    """Extract audio to a 16kHz mono WAV so whisper-cpp can consume it directly."""
    out = Path(tempfile.mktemp(suffix=".wav", prefix="whisper_src_"))
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(out),
        ],
        check=True,
    )
    return out


@mcp.tool
def transcribe_audio_file(
    media_path: str,
    model: str = "medium.en",
    write_sidecar: bool = True,
) -> str:
    """
    Transcribe speech in a video or audio file using whisper.cpp.

    Produces word-level timestamps (verbatim — not paraphrased like gemma4).
    Accepts video (.mp4/.mov/etc) or audio files; video audio is extracted first.

    *model*: whisper model name — "medium.en" (default), "large-v3", "small.en",
    "base.en", "tiny.en". Must exist at ~/.local/share/whisper-models/ggml-<model>.bin
    or be overridable via WHISPER_MODEL_PATH env var.

    *write_sidecar*: when True (default), writes `<stem>.transcript.json` next to
    the source file with the full transcript and word timestamps.

    Returns a JSON string with: transcript (str), word_count (int), duration_sec
    (float — last word end), engine, model, sidecar_path (if written).
    """
    path = Path(media_path).expanduser()
    if not path.exists():
        return json.dumps({"error": f"File not found: {path}"})

    model_path = Path(
        os.environ.get("WHISPER_MODEL_PATH", "")
        or Path.home() / ".local/share/whisper-models" / f"ggml-{model}.bin"
    )
    if not model_path.exists():
        return json.dumps(
            {
                "error": f"Whisper model not found at {model_path}. "
                f"Download from https://huggingface.co/ggerganov/whisper.cpp."
            }
        )

    # Video → extract audio first; audio → pass through (whisper.py handles re-encoding)
    tmp_wav: Path | None = None
    try:
        if path.suffix.lower() in _VIDEO_EXTS:
            tmp_wav = _extract_audio(path)
            audio_input = tmp_wav
        else:
            audio_input = path

        result = transcribe_audio(audio_input, model_path=model_path)
    finally:
        if tmp_wav is not None:
            tmp_wav.unlink(missing_ok=True)

    duration = result["words"][-1]["end_sec"] if result["words"] else 0.0

    summary = {
        "transcript": result["transcript"],
        "word_count": len(result["words"]),
        "duration_sec": duration,
        "engine": result["engine"],
        "model": result["model"],
    }

    if write_sidecar:
        sidecar = path.with_name(path.stem + ".transcript.json")
        sidecar.write_text(
            json.dumps(
                {
                    "source_file": str(path),
                    "transcript": result["transcript"],
                    "words": result["words"],
                    "engine": result["engine"],
                    "model": result["model"],
                },
                indent=2,
            )
        )
        summary["sidecar_path"] = str(sidecar)

    return json.dumps(summary, indent=2)
