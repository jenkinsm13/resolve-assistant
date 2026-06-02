"""
resolve_assistant — AI-powered video editing assistant.
Uses Gemini to analyze footage, plan edits, and build timelines in DaVinci Resolve.
"""

# PATH augmentation must happen at import time, before any module reaches
# for ffprobe/ffmpeg. GUI-launched macOS apps (Dock / Finder / Xcode) get
# a sanitized PATH that omits /opt/homebrew/bin, which breaks every bare
# `subprocess.run(["ffprobe", ...])` call. Prepend the standard Homebrew
# prefixes so all downstream invocations resolve correctly.
import os as _os

_brew_dirs = ["/opt/homebrew/bin", "/usr/local/bin"]
_current_path = _os.environ.get("PATH", "")
_existing = set(_current_path.split(":"))
_prepend = [d for d in _brew_dirs if _os.path.isdir(d) and d not in _existing]
if _prepend:
    _os.environ["PATH"] = ":".join(
        _prepend + ([_current_path] if _current_path else [])
    )

from .config import mcp  # noqa: F401 — re-export for entry points


def main():
    """Entry point for `resolve-assistant` console script."""
    mcp.run()


# Importing these modules registers their @mcp.tool decorators.
from . import ingest  # noqa: F401  — ingest_footage, ingest_status
from . import build  # noqa: F401  — build_timeline, build_status
from . import key_moments  # noqa: F401  — build_key_moments_timeline
from . import whisper_tools  # noqa: F401  — transcribe_audio_file
