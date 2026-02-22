# resolve-assistant

**AI-powered video editing assistant that uses Google Gemini to analyze footage, plan professional edits, and build timelines in DaVinci Resolve — automatically.** Give it a folder of clips and an editing instruction. It watches every frame, plans the edit, and builds the timeline.

[![PyPI](https://img.shields.io/pypi/v/resolve-assistant)](https://pypi.org/project/resolve-assistant/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## What is resolve-assistant?

`resolve-assistant` is an [MCP server](https://modelcontextprotocol.io) that turns raw footage into edited timelines using AI. It combines Google Gemini's multimodal video understanding with DaVinci Resolve's professional editing engine.

**The workflow is simple:**

1. **Point it at a folder of clips** → Gemini watches every frame, identifies A-roll vs B-roll, transcribes speech, rates quality, tags camera movements, and analyzes music structure
2. **Give it an editing instruction** → *"Cut a 60-second highlight reel with high energy"* or *"Build a narrative interview piece, lead with the strongest soundbite"*
3. **It builds the timeline** → Gemini plans every cut, selects the best footage, writes a voiceover script, generates director's notes, and constructs the timeline directly in DaVinci Resolve

No manual logging. No spreadsheets. No bin diving. The AI watches your footage and edits it.

### Why resolve-assistant?

- **AI watches your footage** — Gemini processes every video and audio file with frame-level analysis, not just thumbnails or metadata
- **Professional editorial decisions** — A-roll/B-roll classification, camera movement detection, speech transcription, take quality scoring, filler word detection, music structure analysis
- **Complete edit output** — timeline built in Resolve + FCP7 XML backup + director's notes + voiceover script + music production brief
- **Works with any MCP client** — Claude Desktop, Claude Code, Cursor, or any MCP-compatible AI assistant
- **Sidecar architecture** — analyze once, edit forever. Sidecar JSONs persist next to your media so re-edits are instant
- **One-line install** — `uvx resolve-assistant` or `pip install resolve-assistant`

---

## Quick Start

### Prerequisites

1. **Google Gemini API key** — free at [aistudio.google.com](https://aistudio.google.com/)
2. **DaVinci Resolve** (Free or Studio) running with scripting enabled (`Preferences → System → General → External scripting using = Network`)
3. **Python 3.11+**
4. **ffmpeg** — `brew install ffmpeg` (macOS) or `apt install ffmpeg` (Linux)

### Install

```bash
# Using uvx (recommended — runs in isolated environment)
uvx resolve-assistant

# Using pip
pip install resolve-assistant

# Using pipx
pipx install resolve-assistant
```

### Configure Claude Desktop

Add to your Claude Desktop MCP config (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "resolve-assistant": {
      "command": "uvx",
      "args": ["resolve-assistant"],
      "env": {
        "GEMINI_API_KEY": "your-gemini-api-key-here"
      }
    }
  }
}
```

### Configure Claude Code

Add to your project's `.mcp.json`:

```json
{
  "mcpServers": {
    "resolve-assistant": {
      "command": "uvx",
      "args": ["resolve-assistant"],
      "env": {
        "GEMINI_API_KEY": "your-gemini-api-key-here"
      }
    }
  }
}
```

---

## How It Works

### Step 1: Ingest — AI Footage Analysis

```
> "Analyze all clips in /Volumes/Media/Project/Footage"
```

The ingest pipeline processes every media file:

**For video clips:**
- Transcodes to a Gemini-safe proxy (HEVC/AAC, ≤1280px) using hardware acceleration (VideoToolbox on Apple Silicon)
- Uploads to Gemini Files API
- Gemini watches the footage at high resolution and produces structured analysis:
  - **Segments** — timestamped A-roll (speech/interview) and B-roll (visual/cutaway) segments
  - **Speech transcription** — verbatim transcripts for all spoken content
  - **Take quality** — good take vs bad take classification for A-roll
  - **Filler words** — detection of um, uh, like, you know, etc.
  - **Camera movement** — Pan Left/Right, Tilt Up/Down, Dolly, Truck, Gimbal, Drone, Static, etc.
  - **Quality score** — 1-10 rating for focus, stability, exposure, and content value
  - **Tags** — notable objects, locations, actions, and visual elements
- Probes fps and duration with ffprobe (authoritative, never trusts Gemini's guess)
- Saves analysis as a sidecar JSON (e.g., `clip001.mov.json`) next to the source file

**For audio files (music, sound effects):**
- Uploads directly to Gemini
- Analysis includes: BPM, musical key, genre, and timestamped sections (intro, verse, chorus, drop, bridge, outro)
- Each section gets energy level (1-10), mood, and instrument tags

**Sidecar architecture:** analysis results are saved as JSON files alongside your media. Re-running ingest skips already-analyzed files. Delete a sidecar to force re-analysis.

### Step 2: Build — AI Edit Planning + Timeline Construction

```
> "Build a 60-second highlight reel with high energy, cut to the beat"
```

The build pipeline:

1. **Loads all sidecar JSONs** from the footage folder
2. **Uploads proxy files to Gemini** so it can watch the actual footage (not just read metadata)
3. **Gemini plans the edit** — selects cuts, assigns tracks, sets in/out points, designs camera moves, plans speed ramps, and writes supplementary outputs
4. **Builds the timeline in Resolve** using `AppendToTimeline` — clips appear on the timeline with correct source timecodes
5. **Renders FCP7 XML backup** with proper timecode offsets (probed via ffprobe)
6. **Saves supplementary outputs:**
   - **Edit plan** (`.edl.json`) — complete cut list with all decisions
   - **Director's notes** (`.md`) — editorial reasoning for every cut: what was chosen, what was rejected, and why
   - **Voiceover script** (`.txt` + `.json`) — original narration copy timed to the edit, ready for recording or TTS
   - **Music production brief** (`.md` + `.json`) — if no music was provided, a detailed brief for producing an original score: BPM, key, arrangement, instrumentation with ADSR envelopes, lyrics, mix notes, and processing chains

### What the AI Editor Does

The AI editor receives the actual footage (not just metadata) and makes professional editorial decisions:

- **Clip selection** — picks the strongest material from all available footage
- **A-roll on V1** — speech, interviews, and primary narrative on track 1
- **B-roll on V2** — cutaways, visuals, and supporting footage on track 2
- **Music-driven pacing** — if audio is provided, cuts land on beats and energy matches section dynamics
- **Speed ramps** — high-frame-rate clips (≥90fps) automatically get speed ramp control points for dramatic slow-motion
- **Camera moves** — zoom, pan, tilt, and dynamic zoom ease applied per-clip where they add story value
- **Voiceover scripting** — original narration written to complement the visuals, timed to fill gaps between A-roll speech
- **Director's notes** — transparent reasoning for every editorial decision

---

## MCP Tools

resolve-assistant exposes exactly 4 tools via MCP:

| Tool | Description |
|------|-------------|
| `ingest_footage(folder_path)` | Analyze all video and audio files in a folder with Gemini. Runs in the background — returns immediately. |
| `ingest_status(folder_path)` | Check progress of a running or completed ingest job. Returns current file, step, and completion count. |
| `build_timeline(folder_path, instruction)` | Plan an edit with Gemini and build the timeline in DaVinci Resolve. Uses cached sidecar analysis. |
| `build_status(folder_path)` | Check progress of a running or completed build job. Returns status and output paths when done. |

### Workflow in Practice

```
You: "Analyze the footage in /Volumes/Media/OffRoad"

Claude: I'll start the ingest process.
→ ingest_footage("/Volumes/Media/OffRoad")
← "Ingestion started for 12 file(s) (10 video using hevc_videotoolbox) (2 audio).
   Use ingest_status('/Volumes/Media/OffRoad') to monitor."

You: "How's it going?"

Claude: Let me check.
→ ingest_status("/Volumes/Media/OffRoad")
← "Ingestion running: 7/12 done. Now analyzing GoPro_0042.mp4."

You: "Build a 30-second social reel — fast cuts, high energy, focus on the jumps"

Claude: I'll start the timeline build.
→ build_timeline("/Volumes/Media/OffRoad", "30-second social reel, fast cuts, high energy, focus on the jumps")
← "Timeline build started (12 sidecars, uploading proxies to Gemini).
   Use build_status('/Volumes/Media/OffRoad') to monitor."

You: "Is it done?"

Claude: Let me check.
→ build_status("/Volumes/Media/OffRoad")
← "Build complete: Timeline 'Off-Road Glory Reel' — 18 cuts.
   TC offsets: 10 probed. Timeline created with 18/18 video clips."
```

---

## Sidecar JSON Format

Each analyzed clip gets a sidecar JSON file. Here's what a video sidecar looks like:

```json
{
  "filename": "GoPro_0042.mp4",
  "file_path": "/Volumes/Media/OffRoad/GoPro_0042.mp4",
  "media_type": "video",
  "analysis_model": "gemini-3-flash-preview",
  "fps": 59.94,
  "duration": 47.214,
  "segments": [
    {
      "start_sec": 0.0,
      "end_sec": 8.5,
      "type": "b-roll",
      "description": "Wide establishing shot of desert trail with dust cloud from approaching vehicle",
      "camera_movement": "Static",
      "quality_score": 8,
      "tags": ["desert", "trail", "dust", "wide-shot", "establishing"]
    },
    {
      "start_sec": 8.5,
      "end_sec": 15.2,
      "type": "a-roll",
      "description": "Driver talking to camera: 'This is the gnarliest trail in Moab, we're about to send it'",
      "camera_movement": "Handheld",
      "quality_score": 7,
      "is_good_take": true,
      "filler_words": [],
      "tags": ["interview", "driver", "excitement"]
    }
  ]
}
```

---

## Architecture

```
resolve-assistant/
├── resolve_assistant/
│   ├── __init__.py          # Package init, registers MCP tools
│   ├── __main__.py          # python -m resolve_assistant entry point
│   ├── config.py            # FastMCP server + Gemini client
│   ├── resolve.py           # DaVinci Resolve scripting API connection
│   ├── ingest.py            # Ingest pipeline orchestration
│   ├── ingest_tools.py      # ingest_footage + ingest_status MCP tools
│   ├── ingest_worker.py     # Background ingest worker thread
│   ├── build.py             # Build pipeline orchestration
│   ├── build_tools.py       # build_timeline + build_status MCP tools
│   ├── build_worker.py      # Background build worker thread
│   ├── timeline.py          # OTIO timeline construction + FCP7 XML rendering
│   ├── transcode.py         # Video transcoding for Gemini (VideoToolbox/libx265)
│   ├── media.py             # File discovery + sidecar loading
│   ├── schemas.py           # Pydantic models for sidecar JSON
│   ├── prompts.py           # Gemini prompt templates
│   ├── outputs.py           # Director's notes, voiceover, music brief writers
│   ├── ffprobe.py           # Video metadata extraction
│   ├── resolve_build.py     # AppendToTimeline builder
│   ├── resolve_transforms.py # Clip transform + speed ramp helpers
│   ├── errors.py            # Error handling
│   └── retry.py             # Gemini retry logic with exponential backoff
├── pyproject.toml
├── .env.example
└── README.md
```

### Processing Pipeline

```
Raw Footage → ffprobe metadata → transcode proxy → Gemini analysis → sidecar JSON
                                                                        ↓
Editing instruction → upload proxies → Gemini edit plan → AppendToTimeline → Resolve timeline
                                            ↓
                                    FCP7 XML + Director's Notes + Voiceover Script + Music Brief
```

---

## Transcoding

resolve-assistant automatically transcodes video for Gemini upload:

- **Apple Silicon** — uses `hevc_videotoolbox` (hardware HEVC encoder, very fast)
- **Other platforms** — falls back to `libx265` (software encoder)
- **Skip criteria** — files already in H.264/H.265, under 2GB, and ≤1280px are uploaded as-is
- **Cache** — transcoded proxies are saved as `.gemini.mp4` next to the source and reused on re-ingest
- **Resolution cap** — 1280px on the longest edge to optimize Gemini token usage while maintaining visual quality

---

## Troubleshooting

### "GEMINI_API_KEY not set"
- Add your Gemini API key to the `env` section of your MCP config
- Get a free key at [aistudio.google.com](https://aistudio.google.com/)

### "DaVinci Resolve is not running"
- Make sure Resolve is open before starting a build
- Enable scripting: `Preferences → System → General → External scripting using = Network`
- Note: ingest (analysis) works without Resolve — only build (timeline construction) needs it

### "ffmpeg not found"
- Install ffmpeg: `brew install ffmpeg` (macOS) or `apt install ffmpeg` (Linux)
- Required for video transcoding before Gemini upload

### Ingest is slow
- First run transcodes all videos — this is I/O and CPU intensive
- Subsequent runs skip already-analyzed files (sidecar JSONs exist)
- Hardware encoding (Apple Silicon) is significantly faster than software encoding

### Build creates timeline but clips are missing
- Ensure source files are imported into Resolve's media pool
- The builder auto-imports missing files, but the paths must be valid

---

## Related Projects

- **[resolve-mcp](https://github.com/jenkinsm13/resolve-mcp)** — 215+ MCP tools for direct DaVinci Resolve control (project management, editing, color grading, rendering, and more)
- **[FastMCP](https://github.com/jlowin/fastmcp)** — The MCP framework powering this server
- **[Model Context Protocol](https://modelcontextprotocol.io)** — The open protocol for AI tool use

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Contributing

Contributions welcome! Please open an issue or PR on [GitHub](https://github.com/jenkinsm13/resolve-assistant).

---

*Built for video editors and content creators who want AI to handle the tedious parts of editing — footage logging, clip selection, and rough cut assembly — so they can focus on the creative decisions that matter.*
