"""
Global configuration, constants, and shared singletons (Gemini client, MCP server).
"""

import logging
import os

from dotenv import load_dotenv
from fastmcp import FastMCP

load_dotenv()

log = logging.getLogger("resolve-assistant")

# ---------------------------------------------------------------------------
# Gemini (optional — set GEMINI_API_KEY for cloud analysis)
# ---------------------------------------------------------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = "gemini-3-flash-preview"  # free preview pricing
client = None

if GEMINI_API_KEY:
    try:
        from google import genai

        client = genai.Client(api_key=GEMINI_API_KEY)
    except ImportError:
        log.warning("google-genai not installed. Gemini backend unavailable.")
else:
    log.warning(
        "GEMINI_API_KEY not set. Gemini backend unavailable — use backend='ollama' for local analysis."
    )

# ---------------------------------------------------------------------------
# MCP server instance — tools register via @mcp.tool in other modules
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "resolve-assistant",
    instructions=(
        "AI-powered video editing assistant for DaVinci Resolve.\n\n"
        "Analyzes footage locally with Ollama/Gemma 4 (default) or via Gemini cloud API, "
        "plans professional edits, and builds timelines in DaVinci Resolve.\n\n"
        "PREREQUISITES:\n"
        "- Ollama running locally with gemma4:e4b model (default backend)\n"
        "- OR GEMINI_API_KEY in .env for cloud analysis\n"
        "- DaVinci Resolve running with scripting enabled "
        "(Preferences → System → General → External scripting using = Network)\n"
        "- ffmpeg/ffprobe required (brew install ffmpeg)\n\n"
        "WORKFLOW:\n"
        "1. ingest_footage(folder) — analyze all clips (default: Ollama local, or backend='gemini')\n"
        "   Ollama extracts frames into a contact sheet grid for temporal analysis.\n"
        "2. ingest_status(folder) — poll until complete\n"
        "3. ingest_drill_down(folder, clip, start, end) — re-analyze a time range at higher fps (Ollama only)\n"
        "4. build_timeline(folder, instruction) — Gemini plans an edit, built as FCP7 XML\n"
        "5. build_status(folder) — poll until complete\n"
        "6. build_key_moments_timeline(folder) — auto-generated timeline with every "
        "key moment from every clip (no API needed, parses sidecar JSONs). "
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

# ---------------------------------------------------------------------------
# Ollama — local multimodal analysis (gemma4:e4b default)
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# Default MUST be a model with BOTH vision AND audio capabilities. The video
# analyzer attaches the clip's extracted audio track alongside the frame
# images on every request (see ollama_analyzer.analyze_video). nemotron3:33b
# has `vision` but NOT `audio` (verify with `ollama show <model>`), so it
# returns HTTP 400 "Failed to load image or audio file" on every clip that
# carries audio — i.e. all of them. gemma4:e4b advertises vision+audio and is
# the documented/known-good default. Do not switch this to an audio-incapable
# model without first gating the audio payload on model capability.
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e4b")
OLLAMA_FRAME_RATE = 2.0  # frames per second to extract for analysis
OLLAMA_FRAME_MAX_EDGE = 896  # max long edge for extracted frames (pixels)
OLLAMA_AUDIO_SAMPLE_RATE = 16000  # 16kHz mono for speech recognition
OLLAMA_TIMEOUT = 600  # seconds per API call
# nemotron3 audio (if ever re-enabled) requires the OpenAI /v1/chat/completions
# endpoint with input_audio content parts. Gemma4 uses the native /api/chat
# images field with audio prepended to the images array.
OLLAMA_USE_OPENAI_ENDPOINT = OLLAMA_MODEL.startswith("nemotron")
# Send the clip's audio track alongside the frames. Default ON: combined
# audio+vision is the wanted behavior (gemma4:e4b has both — verified via
# `ollama show`). Vision-only is faster (no audio extraction, smaller payload),
# so set OLLAMA_SEND_AUDIO=0 to disable. NOTE: an audio-incapable model (e.g.
# nemotron3, vision-only in Ollama) returns HTTP 400 on clips with audio, so if
# you switch OLLAMA_MODEL to such a model you must also set OLLAMA_SEND_AUDIO=0.
OLLAMA_SEND_AUDIO = os.getenv("OLLAMA_SEND_AUDIO", "1") == "1"
