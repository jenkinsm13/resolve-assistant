"""
Video transcoding for Gemini upload — downsample to HEVC at ≤1280px.
Uses macOS native avconvert (AVFoundation/VideoToolbox hardware encoders).
NEVER uses ffmpeg for transcoding.
"""

import subprocess
from pathlib import Path

from .config import GEMINI_MAX_BYTES, GEMINI_MAX_LONG_EDGE, SAFE_CODECS, log
from .ffprobe import ffprobe_codec, ffprobe_resolution

_AVCONVERT = "/usr/bin/avconvert"
_PRESET = "Preset1280x720"


def _needs_transcode(video_path: Path) -> bool:
    """Decide whether a video needs transcoding before Gemini upload."""
    if video_path.stat().st_size > GEMINI_MAX_BYTES:
        return True
    codec = ffprobe_codec(video_path)
    if codec is None:
        return True
    if codec not in SAFE_CODECS:
        return True
    w, h = ffprobe_resolution(video_path)
    if w is not None and h is not None:
        if max(w, h) > GEMINI_MAX_LONG_EDGE:
            return True
    return False


def prepare_for_gemini(video_path: Path) -> Path:
    """Return a Gemini-safe file path.

    If the source is already H.264/H.265 at ≤1280px and under 2 GB, return as-is.
    Otherwise, transcode via macOS avconvert and cache as {name}.gemini.mp4.
    """
    if not _needs_transcode(video_path):
        return video_path

    cache_path = video_path.with_suffix(".gemini.mp4")
    if cache_path.exists():
        return cache_path

    log.info("Transcoding %s → %s (avconvert %s)", video_path.name, cache_path.name, _PRESET)

    cmd = [
        _AVCONVERT,
        "--preset", _PRESET,
        "--source", str(video_path),
        "--output", str(cache_path),
        "--replace",
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=3600, check=True)
    except FileNotFoundError:
        raise RuntimeError(
            "avconvert not found at /usr/bin/avconvert. "
            "Requires macOS 13+ with Xcode command-line tools."
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"avconvert failed for {video_path.name}: {exc.stderr[:500]}"
        )

    if cache_path.stat().st_size > GEMINI_MAX_BYTES:
        log.warning(
            "%s is still %.1f GB after transcode — Gemini may reject it.",
            cache_path.name,
            cache_path.stat().st_size / (1024 ** 3),
        )

    return cache_path
