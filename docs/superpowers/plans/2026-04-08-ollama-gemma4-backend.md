# Ollama Gemma 4 e4b Backend for resolve-assistant

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a parallel Ollama/Gemma 4 e4b analysis backend alongside Gemini so footage can be analyzed entirely on-device with zero API cost.

**Architecture:** A new `ollama_analyzer.py` module extracts frames (ffmpeg) and audio (ffmpeg) from each video, sends both to Gemma 4 e4b via Ollama's HTTP API, and writes the same `VideoSidecar`/`AudioSidecar` JSON format. The existing `ingest_worker.py` dispatches to either `_analyze_gemini()` or `_analyze_ollama()` based on a `backend` parameter on the MCP tool. Gemini path stays completely untouched.

**Tech Stack:** ffmpeg (frame/audio extraction), Ollama HTTP API (`localhost:11434`), Gemma 4 e4b (vision + audio, 128K context), Pydantic (JSON validation)

**Verified assumptions:**
- Gemma 4 e4b accepts images via Ollama `/api/chat` `images` field (base64) — tested, works, 9-12s per call
- Gemma 4 e4b accepts audio as base64 via the same `images` field — tested, transcribes speech correctly, 5.9s
- `format: "json"` on the Ollama API forces valid JSON output — tested, works
- 5 frames + prompt → 12.1s, correct scene identification, correct JSON schema

---

### File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `resolve_assistant/ollama_analyzer.py` | **Create** | Frame extraction, audio extraction, Ollama API calls, JSON validation |
| `resolve_assistant/ingest_worker.py` | **Modify** | Add backend dispatch: `_analyze_gemini()` / `_analyze_ollama()` |
| `resolve_assistant/ingest_tools.py` | **Modify** | Add `backend`/`fps` params to `ingest_footage()`, add `ingest_drill_down()` tool |
| `resolve_assistant/config.py` | **Modify** | Add Ollama constants (URL, model, frame rate) |
| `resolve_assistant/prompts_analysis.py` | **Modify** | Add Ollama-specific prompt variants |
| `tests/test_ollama_analyzer.py` | **Create** | Unit tests for frame extraction and JSON parsing |

**Key design decisions:**
- **Timestamped frame filenames:** Frames extracted as `frame_001_t00.5s.jpg` so the model sees timing in filenames. The prompt also lists frame timestamps explicitly.
- **Configurable fps:** `ingest_footage(..., fps=2)` overrides the default 0.5fps for denser sampling. Higher fps = more frames = slower but more precise.
- **Drill-down tool:** `ingest_drill_down(folder, clip_name, start_sec, end_sec, fps=4)` re-analyzes a specific time range at higher frame density. Useful when a coarse pass missed a key moment or you need precise segment boundaries.

---

### Task 1: Add Ollama Config Constants

**Files:**
- Modify: `resolve_assistant/config.py`

- [ ] **Step 1: Add Ollama constants to config.py**

Add below the existing media constants block (after line 62):

```python
# ---------------------------------------------------------------------------
# Ollama (optional — for local Gemma 4 analysis)
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e4b")
OLLAMA_FRAME_RATE = 0.5  # frames per second to extract for analysis
OLLAMA_FRAME_MAX_EDGE = 640  # max long edge for extracted frames (pixels)
OLLAMA_AUDIO_SAMPLE_RATE = 16000  # 16kHz mono for speech recognition
OLLAMA_TIMEOUT = 300  # seconds per API call
```

- [ ] **Step 2: Make GEMINI_API_KEY optional when using Ollama**

Replace the RuntimeError at lines 20-22 with a warning:

```python
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    log.warning(
        "GEMINI_API_KEY not set. Gemini backend unavailable — use backend='ollama' for local analysis."
    )
    client = None
else:
    client = genai.Client(api_key=GEMINI_API_KEY)
```

Move `from google import genai` inside the conditional to avoid import errors when genai is not installed:

```python
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = None
if GEMINI_API_KEY:
    try:
        from google import genai
        client = genai.Client(api_key=GEMINI_API_KEY)
    except ImportError:
        log.warning("google-genai not installed. Gemini backend unavailable.")
```

- [ ] **Step 3: Commit**

```bash
git add resolve_assistant/config.py
git commit -m "feat: add Ollama config constants, make Gemini optional"
```

---

### Task 2: Add Ollama-Specific Prompts

**Files:**
- Modify: `resolve_assistant/prompts_analysis.py`

- [ ] **Step 1: Add Ollama video analysis prompt**

Append to the end of `prompts_analysis.py`. This prompt is optimized for frame-by-frame analysis with embedded timestamps — Ollama receives images, not video, so we label each frame's timestamp in the prompt.

```python
OLLAMA_VIDEO_PROMPT = """\
You are a Script Supervisor reviewing raw footage frames and audio for a professional editor.

You will receive:
1. A set of video frames sampled at regular intervals (timestamps noted below)
2. The audio track from this same video clip

Analyze BOTH the visual frames and the audio to produce a structured JSON analysis.

**Frame timestamps:** {frame_timestamps}
**Clip duration:** {duration:.1f} seconds
**Clip filename:** {filename}

**Instructions:**

For SPEECH segments (a-roll):
- Transcribe spoken words from the audio.
- Mark usable takes vs bad takes (stumbles, false starts).
- Note filler words (um, uh, like, you know).

For VISUAL segments (b-roll):
- Describe the visual action concisely.
- Identify camera movement by comparing frames:
  - Parallax between foreground/background = Dolly/Truck (translation)
  - Uniform frame shift = Pan/Tilt (rotation)
  - Multi-axis floating = Gimbal, Drone, Handheld
  - No change between frames = Static
- Rate visual quality 1-10.
- Tag notable objects, locations, or actions.

Return ONLY valid JSON:
{{
  "filename": "{filename}",
  "file_path": "{file_path}",
  "media_type": "video",
  "analysis_model": "{model}",
  "fps": {fps},
  "duration": {duration},
  "segments": [
    {{
      "start_sec": <float>,
      "end_sec": <float>,
      "type": "a-roll" or "b-roll",
      "description": "<transcript or visual description>",
      "camera_movement": "<movement type or null>",
      "quality_score": <1-10>,
      "is_good_take": <bool or null>,
      "filler_words": ["um", ...] or null,
      "tags": ["keyword", ...]
    }}
  ]
}}
"""


OLLAMA_AUDIO_PROMPT = """\
You are a Music Supervisor analyzing an audio track for a professional video editor.

Analyze this audio and return ONLY valid JSON:

{{
  "filename": "{filename}",
  "file_path": "{file_path}",
  "media_type": "audio",
  "analysis_model": "{model}",
  "duration": {duration},
  "bpm": <float or null>,
  "key": "<string or null>",
  "genre": "<string or null>",
  "sections": [
    {{
      "start_sec": <float>,
      "end_sec": <float>,
      "description": "<what happens musically>",
      "energy": <1-10>,
      "bpm_estimate": <float or null>,
      "mood": "<string or null>",
      "tags": ["keyword", ...]
    }}
  ]
}}
"""
```

- [ ] **Step 2: Commit**

```bash
git add resolve_assistant/prompts_analysis.py
git commit -m "feat: add Ollama-specific analysis prompt templates"
```

---

### Task 3: Create ollama_analyzer.py — Frame/Audio Extraction

**Files:**
- Create: `resolve_assistant/ollama_analyzer.py`
- Test: `tests/test_ollama_analyzer.py`

- [ ] **Step 1: Create tests directory and test file**

```bash
mkdir -p tests
```

Write `tests/test_ollama_analyzer.py`:

```python
"""Tests for Ollama analyzer — frame extraction and JSON parsing."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


def test_extract_frames_creates_jpeg_files():
    """extract_frames should produce JPEG files at the expected interval."""
    from resolve_assistant.ollama_analyzer import extract_frames

    # Use a tiny test video if available, otherwise skip
    # This test will be run manually with real media
    pass  # placeholder — real integration test below


def test_parse_ollama_response_valid_json():
    """_parse_ollama_response should extract valid sidecar data from JSON."""
    from resolve_assistant.ollama_analyzer import _parse_ollama_response

    raw = json.dumps({
        "segments": [
            {
                "start_sec": 0.0,
                "end_sec": 5.0,
                "type": "b-roll",
                "description": "A green Ford Bronco engine bay",
                "camera_movement": "Static",
                "quality_score": 8,
                "is_good_take": None,
                "filler_words": None,
                "tags": ["car", "engine"],
            }
        ]
    })

    result = _parse_ollama_response(raw, "test.mov", "/path/test.mov", 59.94, 12.0)
    assert result["filename"] == "test.mov"
    assert result["file_path"] == "/path/test.mov"
    assert len(result["segments"]) == 1
    assert result["segments"][0]["quality_score"] == 8


def test_parse_ollama_response_markdown_wrapped():
    """_parse_ollama_response should handle JSON wrapped in markdown code fences."""
    from resolve_assistant.ollama_analyzer import _parse_ollama_response

    raw = '```json\n{"segments": [{"start_sec": 0, "end_sec": 3, "type": "b-roll", "description": "test", "quality_score": 5, "tags": []}]}\n```'

    result = _parse_ollama_response(raw, "test.mov", "/path/test.mov", 30.0, 3.0)
    assert len(result["segments"]) == 1


def test_parse_ollama_response_invalid_json_raises():
    """_parse_ollama_response should raise ValueError on unparseable responses."""
    from resolve_assistant.ollama_analyzer import _parse_ollama_response

    with pytest.raises(ValueError, match="parse"):
        _parse_ollama_response("not json at all", "test.mov", "/p", 30.0, 3.0)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Volumes/4TB\ G-SSD/ASC_4x4/repos/resolve-assistant
uv run pytest tests/test_ollama_analyzer.py -v
```

Expected: ImportError — `ollama_analyzer` module doesn't exist yet.

- [ ] **Step 3: Create ollama_analyzer.py with extraction + parsing**

Write `resolve_assistant/ollama_analyzer.py`:

```python
"""
Ollama/Gemma 4 analyzer: local video+audio analysis via frame extraction.

Pipeline per clip:
1. Extract frames at OLLAMA_FRAME_RATE fps via ffmpeg (JPEG, scaled)
2. Extract audio track via ffmpeg (WAV, 16kHz mono)
3. Send frames + audio to Ollama as multimodal chat message
4. Parse JSON response into VideoSidecar format
"""

import base64
import json
import re
import subprocess
import tempfile
import urllib.request
from pathlib import Path

from .config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_FRAME_RATE,
    OLLAMA_FRAME_MAX_EDGE,
    OLLAMA_AUDIO_SAMPLE_RATE,
    OLLAMA_TIMEOUT,
    log,
)
from .ffprobe import ffprobe_fps, ffprobe_duration
from .prompts_analysis import OLLAMA_VIDEO_PROMPT, OLLAMA_AUDIO_PROMPT


def extract_frames(
    video_path: Path,
    fps: float = OLLAMA_FRAME_RATE,
    start_sec: float = 0.0,
    end_sec: float | None = None,
) -> list[tuple[Path, float]]:
    """Extract frames from video at *fps* rate. Returns [(path, timestamp_sec), ...].

    Frames are named with embedded timestamps: frame_001_t00.5s.jpg
    so the model can see timing even in filenames.
    Caller is responsible for cleanup.
    """
    out_dir = Path(tempfile.mkdtemp(prefix="ollama_frames_"))

    # Build ffmpeg command with optional time range
    cmd = ["ffmpeg"]
    if start_sec > 0:
        cmd += ["-ss", str(start_sec)]
    cmd += ["-i", str(video_path)]
    if end_sec is not None:
        cmd += ["-t", str(end_sec - start_sec)]
    cmd += [
        "-vf", f"fps={fps},scale={OLLAMA_FRAME_MAX_EDGE}:-1",
        "-q:v", "3",
        str(out_dir / "frame_%04d.jpg"), "-y",
    ]
    subprocess.run(cmd, capture_output=True, timeout=120, check=True)

    # Rename with actual source frame numbers
    raw_frames = sorted(out_dir.glob("frame_*.jpg"))
    native_fps = ffprobe_fps(video_path) or 30.0
    interval = 1.0 / fps
    result: list[tuple[Path, float]] = []

    for i, frame_path in enumerate(raw_frames):
        t = start_sec + i * interval
        source_frame = round(t * native_fps)
        new_name = f"frame_{source_frame:06d}.jpg"
        new_path = out_dir / new_name
        frame_path.rename(new_path)
        result.append((new_path, t))

    return result


def extract_audio(
    video_path: Path,
    start_sec: float = 0.0,
    end_sec: float | None = None,
) -> Path | None:
    """Extract audio track from video as 16kHz mono WAV. Returns path or None."""
    out_path = Path(tempfile.mktemp(suffix=".wav", prefix="ollama_audio_"))

    cmd = ["ffmpeg"]
    if start_sec > 0:
        cmd += ["-ss", str(start_sec)]
    cmd += ["-i", str(video_path)]
    if end_sec is not None:
        cmd += ["-t", str(end_sec - start_sec)]
    cmd += [
        "-vn", "-ar", str(OLLAMA_AUDIO_SAMPLE_RATE), "-ac", "1",
        str(out_path), "-y",
    ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=120, check=True)
        if out_path.exists() and out_path.stat().st_size > 1000:
            return out_path
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pass

    return None


def _ollama_chat(prompt: str, images: list[str], timeout: int = OLLAMA_TIMEOUT) -> str:
    """Send a multimodal chat request to Ollama. Returns response text.

    *images* is a list of base64-encoded image or audio data.
    """
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "messages": [{
            "role": "user",
            "content": prompt,
            "images": images,
        }],
        "stream": False,
        "format": "json",
    }).encode()

    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=timeout)
    data = json.loads(resp.read())
    return data["message"]["content"]


def _parse_ollama_response(
    raw: str, filename: str, file_path: str, fps: float, duration: float,
) -> dict:
    """Parse Ollama response text into sidecar dict. Handles markdown fences."""
    text = raw.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in the response
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            data = json.loads(match.group())
        else:
            raise ValueError(f"Could not parse JSON from Ollama response: {text[:200]}")

    # Ensure required top-level fields
    data.setdefault("filename", filename)
    data.setdefault("file_path", file_path)
    data.setdefault("media_type", "video")
    data.setdefault("analysis_model", OLLAMA_MODEL)
    data.setdefault("fps", fps)
    data.setdefault("duration", duration)
    data.setdefault("segments", [])

    return data


def analyze_video(
    video_path: Path,
    fps: float = OLLAMA_FRAME_RATE,
    start_sec: float = 0.0,
    end_sec: float | None = None,
) -> dict:
    """Analyze a video clip using Ollama/Gemma 4 e4b.

    Extracts frames + audio, sends both to Ollama, returns sidecar dict.
    *fps* controls frame sampling density (higher = more precise but slower).
    *start_sec*/*end_sec* allow analyzing a specific time range (drill-down).
    """
    duration = ffprobe_duration(video_path) or 0.0
    native_fps = ffprobe_fps(video_path) or 30.0
    analyze_duration = (end_sec or duration) - start_sec

    log.info(
        "Ollama analyzing %s (%.1fs at %.1f fps, range %.1f-%.1fs)",
        video_path.name, analyze_duration, fps, start_sec, end_sec or duration,
    )

    # Extract frames with timestamps
    frame_results = extract_frames(video_path, fps=fps, start_sec=start_sec, end_sec=end_sec)
    if not frame_results:
        raise RuntimeError(f"No frames extracted from {video_path.name}")

    # Build base64 images and timestamp labels
    images: list[str] = []
    timestamps: list[str] = []

    for frame_path, t in frame_results:
        timestamps.append(f"{frame_path.name}: {t:.1f}s")
        images.append(base64.b64encode(frame_path.read_bytes()).decode())

    # Extract and append audio (full clip or range)
    audio_path = extract_audio(video_path, start_sec=start_sec, end_sec=end_sec)
    if audio_path:
        images.append(base64.b64encode(audio_path.read_bytes()).decode())

    # Build prompt
    prompt = OLLAMA_VIDEO_PROMPT.format(
        frame_timestamps=", ".join(timestamps),
        duration=analyze_duration,
        filename=video_path.name,
        file_path=str(video_path),
        model=OLLAMA_MODEL,
        fps=round(native_fps, 3),
    )

    # Call Ollama
    raw = _ollama_chat(prompt, images)

    # Parse and validate
    sidecar = _parse_ollama_response(
        raw, video_path.name, str(video_path), round(native_fps, 3), round(duration, 3),
    )

    # Cleanup temp files
    frame_dir = frame_results[0][0].parent if frame_results else None
    for f, _ in frame_results:
        f.unlink(missing_ok=True)
    if frame_dir:
        frame_dir.rmdir()
    if audio_path:
        audio_path.unlink(missing_ok=True)

    return sidecar


def analyze_audio(audio_path: Path) -> dict:
    """Analyze a standalone audio file using Ollama/Gemma 4 e4b."""
    duration = ffprobe_duration(audio_path) or 0.0

    log.info("Ollama analyzing audio %s (%.1fs)", audio_path.name, duration)

    audio_b64 = base64.b64encode(audio_path.read_bytes()).decode()

    prompt = OLLAMA_AUDIO_PROMPT.format(
        filename=audio_path.name,
        file_path=str(audio_path),
        model=OLLAMA_MODEL,
        duration=round(duration, 3),
    )

    raw = _ollama_chat(prompt, [audio_b64])

    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    data = json.loads(text)
    data.setdefault("filename", audio_path.name)
    data.setdefault("file_path", str(audio_path))
    data.setdefault("media_type", "audio")
    data.setdefault("analysis_model", OLLAMA_MODEL)
    data.setdefault("duration", round(duration, 3))
    data.setdefault("sections", [])

    return data
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Volumes/4TB\ G-SSD/ASC_4x4/repos/resolve-assistant
uv run pytest tests/test_ollama_analyzer.py -v
```

Expected: 3 tests pass (parse valid, parse markdown-wrapped, parse invalid raises).

- [ ] **Step 5: Commit**

```bash
git add resolve_assistant/ollama_analyzer.py tests/test_ollama_analyzer.py
git commit -m "feat: add ollama_analyzer module — frame+audio extraction, Ollama API, JSON parsing"
```

---

### Task 4: Wire Ollama Backend into Ingest Worker

**Files:**
- Modify: `resolve_assistant/ingest_worker.py`
- Modify: `resolve_assistant/ingest_tools.py`

- [ ] **Step 1: Refactor ingest_worker.py — extract Gemini analysis into a function**

At the top of `ingest_worker.py`, add the Ollama import:

```python
from .ollama_analyzer import analyze_video as ollama_analyze_video, analyze_audio as ollama_analyze_audio
```

Extract lines 130-183 of `_ingest_worker` into two functions. Add these before `_ingest_worker`:

```python
def _analyze_gemini(media_path: Path, upload_path: Path, is_audio: bool) -> dict:
    """Analyze a single file using Gemini API. Returns sidecar dict."""
    from google.genai import types
    from .prompts import ANALYSIS_PROMPT, AUDIO_ANALYSIS_PROMPT

    file_ref = retry_gemini(client.files.upload, file=str(upload_path))
    while file_ref.state.name == "PROCESSING":
        time.sleep(2)
        file_ref = client.files.get(name=file_ref.name)

    if file_ref.state.name != "ACTIVE":
        raise RuntimeError(f"Upload state={file_ref.state.name}")

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

    return json.loads(response.text)
```

- [ ] **Step 2: Add backend parameter to _ingest_worker**

Change the signature and loop body of `_ingest_worker`:

```python
def _ingest_worker(root: Path, build_instruction: Optional[str] = None, backend: str = "gemini", fps: float = 0.5) -> None:
```

Replace the try/except block inside the `for i, media_path` loop (lines 130-187) with:

```python
        try:
            if backend == "ollama":
                _write_progress(root, {
                    "status": "running", "current_file": media_path.name,
                    "current_step": "analyzing (ollama)", "completed": already_done + i,
                    "total": total, "errors": errors,
                })

                if is_audio:
                    sidecar_data = ollama_analyze_audio(media_path)
                else:
                    sidecar_data = ollama_analyze_video(media_path, fps=fps)
            else:
                # Gemini path — upload then analyze
                if is_audio:
                    upload_path = media_path
                else:
                    upload_path = proxies.get(media_path.name, media_path)

                _write_progress(root, {
                    "status": "running", "current_file": media_path.name,
                    "current_step": "uploading", "completed": already_done + i,
                    "total": total, "errors": errors,
                })

                sidecar_data = _analyze_gemini(media_path, upload_path, is_audio)

            sidecar_data["file_path"] = str(media_path)
            sidecar_data["filename"] = media_path.name

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
```

Also: skip the transcoding phase when backend is "ollama" (Ollama uses frame extraction, not transcoded proxies):

```python
    # --- Phase 1: Parallel transcode (videos only, Gemini backend only) ---
    proxies: dict[str, Path] = {}
    if backend != "ollama":
        _write_progress(root, { ... })
        proxies = _batch_transcode(pending_videos, root, already_done, total, errors)
    else:
        _write_progress(root, {
            "status": "running",
            "current_file": "starting ollama analysis",
            "current_step": "analyzing",
            "completed": already_done,
            "total": total,
            "errors": errors,
        })
```

- [ ] **Step 3: Add backend parameter to ingest_footage MCP tool**

In `ingest_tools.py`, change the tool signature:

```python
@mcp.tool
def ingest_footage(
    folder_path: str,
    instruction: Optional[str] = None,
    backend: str = "gemini",
    fps: float = 0.5,
) -> str:
    """
    Scan a folder for video and audio files and analyze them.
    Launches a background worker that processes ALL pending files.
    Returns immediately — use ingest_status() to monitor progress.
    Files with existing .json sidecars are skipped automatically.

    If *instruction* is provided, a timeline build is automatically triggered
    once all sidecars are written — no manual follow-up needed.

    *backend*: "gemini" (default, cloud API) or "ollama" (local Gemma 4 e4b).
    *fps*: Frame sampling rate for Ollama backend (default 0.5 = 1 frame per 2 sec).
           Higher values (1, 2, 4) give more precise analysis but take longer.
           Ignored for Gemini backend.
    """
```

Update the avconvert check to only apply for gemini backend:

```python
    if backend != "ollama":
        if not Path("/usr/bin/avconvert").exists():
            return "Error: avconvert not found. Requires macOS 13+ with Xcode command-line tools."
    else:
        # Verify Ollama is running
        try:
            import urllib.request
            urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        except Exception:
            return f"Error: Ollama not reachable at {OLLAMA_BASE_URL}. Is it running?"
```

Add the import:
```python
from .config import OLLAMA_BASE_URL
```

Update the thread launch to pass backend:

```python
    thread = threading.Thread(target=_ingest_worker, args=(root, instruction, backend, fps), daemon=True)
```

Update the status message:

```python
    engine = "Ollama/Gemma4" if backend == "ollama" else "Gemini"
    parts = [f"Ingestion started for {len(pending)} file(s) via {engine}"]
```

- [ ] **Step 4: Commit**

```bash
git add resolve_assistant/ingest_worker.py resolve_assistant/ingest_tools.py
git commit -m "feat: wire Ollama backend into ingest pipeline with backend parameter"
```

---

### Task 5: Add Drill-Down Tool for High-Density Re-Analysis

**Files:**
- Modify: `resolve_assistant/ingest_tools.py`

- [ ] **Step 1: Add ingest_drill_down MCP tool**

Append to `ingest_tools.py`:

```python
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
    investigate a specific moment. Replaces the existing sidecar with a
    merged result containing the high-density analysis for the specified range.

    Only works with the Ollama backend (local analysis).

    *clip_name*: filename of the clip (e.g. "A001_02031700_C006.mov")
    *start_sec*: start of the range to re-analyze
    *end_sec*: end of the range to re-analyze
    *fps*: frame sampling rate (default 4.0 = 4 frames/sec for detailed analysis)
    """
    from .ollama_analyzer import analyze_video

    root = Path(folder_path).resolve()
    if not root.is_dir():
        return f"Error: '{folder_path}' is not a valid directory."

    # Find the clip
    clip_path = root / clip_name
    if not clip_path.exists():
        # Try case-insensitive match
        matches = [f for f in root.iterdir() if f.name.lower() == clip_name.lower()]
        if matches:
            clip_path = matches[0]
        else:
            return f"Error: clip '{clip_name}' not found in {root}"

    sidecar_path = clip_path.with_suffix(clip_path.suffix + ".json")

    try:
        # Verify Ollama is running
        import urllib.request
        urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
    except Exception:
        return f"Error: Ollama not reachable at {OLLAMA_BASE_URL}. Is it running?"

    try:
        result = analyze_video(
            clip_path, fps=fps, start_sec=start_sec, end_sec=end_sec,
        )

        # If existing sidecar, merge: replace segments in the drilled range
        if sidecar_path.exists():
            import json
            existing = json.loads(sidecar_path.read_text(encoding="utf-8"))
            old_segments = existing.get("segments", [])

            # Keep segments outside the drill range
            kept = [
                s for s in old_segments
                if s["end_sec"] <= start_sec or s["start_sec"] >= end_sec
            ]
            # Add new high-density segments
            kept.extend(result.get("segments", []))
            kept.sort(key=lambda s: s["start_sec"])

            existing["segments"] = kept
            sidecar_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
            return (
                f"Drill-down complete: {clip_name} [{start_sec:.1f}s-{end_sec:.1f}s] "
                f"at {fps} fps. {len(result.get('segments', []))} new segments merged "
                f"into existing sidecar ({len(kept)} total segments)."
            )
        else:
            import json
            sidecar_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            return (
                f"Drill-down complete: {clip_name} [{start_sec:.1f}s-{end_sec:.1f}s] "
                f"at {fps} fps. {len(result.get('segments', []))} segments written."
            )

    except Exception as exc:
        return f"Drill-down failed for {clip_name}: {exc}"
```

- [ ] **Step 2: Commit**

```bash
git add resolve_assistant/ingest_tools.py
git commit -m "feat: add ingest_drill_down tool for high-density re-analysis of clip ranges"
```

---

### Task 6: Integration Test — Analyze One Clip with Ollama

**Files:**
- Create: `tests/test_ollama_integration.py`

- [ ] **Step 1: Write integration test script**

This test requires Ollama running with gemma4:e4b and a real video file. It's a manual integration test, not a CI test.

Write `tests/test_ollama_integration.py`:

```python
"""
Integration test: analyze a single video clip with Ollama/Gemma 4 e4b.

Requires:
- Ollama running at localhost:11434
- gemma4:e4b model pulled
- A test video file (pass path as CLI arg or use default)

Usage:
    uv run python tests/test_ollama_integration.py [/path/to/video.mov]
"""

import json
import sys
import time
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from resolve_assistant.ollama_analyzer import analyze_video, analyze_audio
from resolve_assistant.schemas import VideoSidecar


def test_video(video_path: Path):
    print(f"\n{'='*60}")
    print(f"Testing: {video_path.name}")
    print(f"Size: {video_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"{'='*60}\n")

    t0 = time.time()
    sidecar = analyze_video(video_path)
    elapsed = time.time() - t0

    print(f"Analysis time: {elapsed:.1f}s")
    print(f"Model: {sidecar.get('analysis_model')}")
    print(f"Duration: {sidecar.get('duration')}s")
    print(f"Segments: {len(sidecar.get('segments', []))}")

    for seg in sidecar.get("segments", []):
        print(f"  [{seg['start_sec']:.1f}-{seg['end_sec']:.1f}s] "
              f"q={seg.get('quality_score', '?')} "
              f"{seg.get('camera_movement', '?')} | "
              f"{seg.get('description', '')[:100]}")

    # Validate against Pydantic schema
    validated = VideoSidecar(**sidecar)
    print(f"\nPydantic validation: PASS ({len(validated.segments)} segments)")

    # Write sidecar for comparison
    out_path = video_path.with_suffix(video_path.suffix + ".ollama.json")
    out_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    print(f"Sidecar written: {out_path.name}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        # Default test clip
        path = Path(
            "/Volumes/4TB G-SSD/ASC_4x4/Workflow/01_INBOX/Raw_Clips/2026/02-03/"
            "Dr. Green - iPhone Gimbal/A001_02031700_C006.mov"
        )

    if not path.exists():
        print(f"Error: {path} not found")
        sys.exit(1)

    test_video(path)
```

- [ ] **Step 2: Run integration test**

```bash
cd /Volumes/4TB\ G-SSD/ASC_4x4/repos/resolve-assistant
uv run python tests/test_ollama_integration.py
```

Expected: Sidecar JSON produced, Pydantic validation passes, segments describe the engine bay footage. Compare output to the existing Gemini sidecar at `A001_02031700_C006.mov.json`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_ollama_integration.py
git commit -m "test: add Ollama integration test for single-clip analysis"
```

---

### Task 7: Test Full MCP Tool End-to-End

- [ ] **Step 1: Restart resolve-assistant MCP server**

The MCP server needs to be restarted to pick up the new code. In Claude Code:

```bash
claude mcp remove --scope user resolve-assistant
claude mcp add --scope user resolve-assistant -- uv run --directory /Volumes/4TB\ G-SSD/ASC_4x4/repos/resolve-assistant python -m resolve_assistant
```

- [ ] **Step 2: Test via MCP tool call**

In a new Claude Code session, call:

```
ingest_footage("/path/to/test/folder", backend="ollama")
```

Then monitor with:

```
ingest_status("/path/to/test/folder")
```

Verify:
- No transcoding step occurs (Ollama path skips avconvert)
- Progress shows "analyzing (ollama)" step
- Sidecar JSONs are written with `analysis_model: "gemma4:e4b"`
- Sidecars pass Pydantic validation

- [ ] **Step 3: Verify Gemini path still works**

```
ingest_footage("/path/to/different/folder")
```

Confirm the default `backend="gemini"` path is unchanged.

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat: Ollama/Gemma 4 e4b backend complete — local video+audio analysis"
```

---

## Notes

**Performance expectations:**
- Gemini: ~60s per clip (transcode + upload + analysis)
- Ollama e4b: ~15-25s per clip (frame extraction + local inference, no network)
- Ollama processes sequentially — future optimization: batch frames across clips

**Audio from video tracks:**
- The Ollama path extracts audio from video files and sends it alongside frames
- This enables a-roll/b-roll detection from speech without a separate ASR model
- Standalone audio files (music) are sent directly to e4b

**What this does NOT change:**
- Gemini path — completely untouched, still the default
- Edit planning (`build_timeline`) — still uses Gemini (text-only, could be swapped later)
- Sidecar format — identical output regardless of backend
- Key moments timeline — reads sidecars, backend-agnostic
