"""
Ollama/Gemma 4 analyzer: local video+audio analysis via frame extraction.

Pipeline per clip:
1. Extract frames at configurable fps via ffmpeg (JPEG, scaled)
2. Extract audio track via ffmpeg (WAV, 16kHz mono)
3. Send frames + audio to Ollama as multimodal chat message
4. Parse JSON response into VideoSidecar format

Frame filenames use real source frame numbers (e.g. frame_000060.jpg = frame 60
in the original clip) so the model sees meaningful identifiers.
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
    """Extract frames from video at *fps* rate.

    Returns [(path, timestamp_sec), ...] sorted by time.
    Frame filenames use source frame numbers: frame_000060.jpg = frame 60.
    Caller is responsible for cleanup of the temp directory.
    """
    out_dir = Path(tempfile.mkdtemp(prefix="ollama_frames_"))

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

    # Rename sequential frames to source frame numbers
    native_fps = ffprobe_fps(video_path) or 30.0
    raw_frames = sorted(out_dir.glob("frame_*.jpg"))
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
    analyze_end = end_sec if end_sec is not None else duration
    analyze_duration = analyze_end - start_sec

    log.info(
        "Ollama analyzing %s (%.1fs at %.1f fps, range %.1f-%.1fs)",
        video_path.name, analyze_duration, fps, start_sec, analyze_end,
    )

    # Extract frames with source frame numbers
    frame_results = extract_frames(video_path, fps=fps, start_sec=start_sec, end_sec=end_sec)
    if not frame_results:
        raise RuntimeError(f"No frames extracted from {video_path.name}")

    # Build base64 images and timestamp labels
    images: list[str] = []
    timestamps: list[str] = []

    for frame_path, t in frame_results:
        timestamps.append(f"{frame_path.name}: {t:.1f}s")
        images.append(base64.b64encode(frame_path.read_bytes()).decode())

    # Extract and append audio
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
