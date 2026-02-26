"""
Global configuration, constants, and shared singletons (Gemini client, MCP server).
"""

import logging
import os

from dotenv import load_dotenv
from fastmcp import FastMCP
from google import genai

load_dotenv()

log = logging.getLogger("resolve-assistant")

# ---------------------------------------------------------------------------
# Gemini (required)
# ---------------------------------------------------------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set. Copy .env.example to .env and fill it in.")

MODEL = "gemini-3-flash-preview"  # free preview pricing
client = genai.Client(api_key=GEMINI_API_KEY)

# ---------------------------------------------------------------------------
# MCP server instance — tools register via @mcp.tool in other modules
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "resolve-assistant",
    instructions=(
        "AI-powered video editing assistant using Gemini + DaVinci Resolve.\n\n"
        "Analyzes footage with Google Gemini, plans professional edits, and builds "
        "timelines directly in DaVinci Resolve via AppendToTimeline.\n\n"
        "PREREQUISITES:\n"
        "- GEMINI_API_KEY must be set in .env\n"
        "- DaVinci Resolve must be running with scripting enabled "
        "(Preferences → System → General → External scripting using = Network)\n"
        "- macOS 13+ required (uses native avconvert for hardware transcoding)\n"
        "- ffprobe required for metadata probing (brew install ffmpeg)\n\n"
        "WORKFLOW:\n"
        "1. ingest_footage(folder) — analyze all clips with Gemini\n"
        "2. ingest_status(folder) — poll until complete\n"
        "3. build_timeline(folder, instruction) — Gemini plans an edit, built as FCP7 XML\n"
        "4. build_status(folder) — poll until complete\n"
        "5. build_key_moments_timeline(folder) — auto-generated timeline with every "
        "key moment from every clip (no Gemini needed, parses sidecar JSONs). "
        "Always run this after ingest completes to give the editor a visual index."
    ),
)

# ---------------------------------------------------------------------------
# Media constants
# ---------------------------------------------------------------------------

VIDEO_EXTS = {".mp4", ".mov", ".mxf", ".avi", ".webm", ".mkv", ".r3d", ".braw"}
AUDIO_EXTS = {".mp3", ".wav", ".aac", ".flac", ".ogg", ".m4a"}
GEMINI_MAX_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB Files API ceiling
SAFE_CODECS = {"h264", "avc", "avc1", "hevc", "h265", "hev1"}
GEMINI_MAX_LONG_EDGE = 1280
