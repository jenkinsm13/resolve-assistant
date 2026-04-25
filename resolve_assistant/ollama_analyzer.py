"""
Ollama/Gemma 4 analyzer: local video+audio analysis via frame extraction.

Pipeline per clip:
1. Extract frames at configurable fps via ffmpeg (JPEG, scaled)
2. Extract audio track via ffmpeg (WAV, 16kHz mono)
3. Build a contact sheet grid (temporal overview) + label individual frames
4. Send contact sheet + individual frames + audio to Ollama
5. Parse JSON response into VideoSidecar format

The dual-image approach gives the model both temporal context (contact sheet
for motion/segmentation) and fine detail (individual frames for badges, text,
people, vehicle details).
"""

import base64
import io
import json
import os
import re
import shutil
import subprocess
import time
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

# Debug escape hatch: when set, frame/audio temp files are retained on disk
# instead of being cleaned up. Useful when reproducing an Ollama HTTP 500 or
# a model parse failure and you need to inspect what the model actually saw.
# Without this, every cleanup path destroys the evidence.
_KEEP_FRAMES = os.getenv("RESOLVE_ASSISTANT_KEEP_FRAMES", "").lower() in (
    "1",
    "true",
    "yes",
)


def _cleanup_frame_dir(dir_path: Path | None) -> None:
    """Best-effort recursive removal of a frame temp dir. Honors KEEP_FRAMES."""
    if dir_path is None or _KEEP_FRAMES:
        return
    shutil.rmtree(dir_path, ignore_errors=True)


def _cleanup_audio_file(audio_path: Path | None) -> None:
    """Best-effort removal of an audio temp file. Honors KEEP_FRAMES."""
    if audio_path is None or _KEEP_FRAMES:
        return
    try:
        audio_path.unlink(missing_ok=True)
    except OSError:
        pass


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
    Caller is responsible for cleanup of the returned paths' parent directory
    on the success path. On failure (ffmpeg timeout, non-zero exit, no frames
    extracted), this function cleans up its own temp dir before re-raising,
    so callers never see a leaked dir from a failed call.
    """
    out_dir = Path(tempfile.mkdtemp(prefix="ollama_frames_"))

    try:
        cmd = ["ffmpeg"]
        if start_sec > 0:
            cmd += ["-ss", str(start_sec)]
        cmd += ["-i", str(video_path)]
        if end_sec is not None:
            cmd += ["-t", str(end_sec - start_sec)]
        cmd += [
            "-vf",
            f"fps={fps},scale={OLLAMA_FRAME_MAX_EDGE}:-1",
            "-q:v",
            "3",
            str(out_dir / "frame_%04d.jpg"),
            "-y",
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

        if not result:
            # ffmpeg returned 0 but produced no frames (rare — corrupt file,
            # zero-duration source). Treat as failure and self-clean.
            raise RuntimeError(f"ffmpeg produced no frames from {video_path.name}")

        return result
    except BaseException:
        # Includes TimeoutExpired, CalledProcessError, RuntimeError above, and
        # also KeyboardInterrupt — any exit other than the normal return must
        # not leak the dir. We re-raise after cleanup.
        _cleanup_frame_dir(out_dir)
        raise


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
        "-vn",
        "-ar",
        str(OLLAMA_AUDIO_SAMPLE_RATE),
        "-ac",
        "1",
        str(out_path),
        "-y",
    ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=120, check=True)
        if out_path.exists() and out_path.stat().st_size > 1000:
            return out_path
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pass

    # Failure path: ffmpeg either errored, timed out, or wrote a too-small
    # partial. Remove the file (if any) so it doesn't leak into tempdir.
    _cleanup_audio_file(out_path)
    return None


def _label_frame(img_path: Path, label: str) -> bytes:
    """Burn a label into the bottom-left of a frame image. Returns JPEG bytes."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)

    # Use a size proportional to image height
    font_size = max(16, img.height // 20)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except (OSError, IOError):
        font = ImageFont.load_default(size=font_size)

    # Text with black outline for readability
    x, y = 8, img.height - font_size - 12
    for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2), (-2, -2), (2, 2)]:
        draw.text((x + dx, y + dy), label, fill="black", font=font)
    draw.text((x, y), label, fill="white", font=font)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def build_contact_sheet(
    frame_results: list[tuple[Path, float]],
    cols: int = 5,
    thumb_width: int = 320,
) -> bytes:
    """Build a labeled contact sheet grid from extracted frames.

    Each thumbnail gets a frame number label burned in (e.g. "#1 0.0s").
    Frames are tiled left-to-right, top-to-bottom in chronological order.
    Returns JPEG bytes of the composite image.
    """
    from PIL import Image, ImageDraw, ImageFont

    if not frame_results:
        raise ValueError("No frames to build contact sheet from")

    # Load and label each thumbnail
    thumbs: list[Image.Image] = []
    for i, (frame_path, t) in enumerate(frame_results):
        img = Image.open(frame_path)
        # Scale to thumb width
        ratio = thumb_width / img.width
        thumb = img.resize((thumb_width, int(img.height * ratio)), Image.LANCZOS)

        # Burn in label
        draw = ImageDraw.Draw(thumb)
        font_size = max(14, thumb.height // 12)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
        except (OSError, IOError):
            font = ImageFont.load_default(size=font_size)

        label = f"#{i + 1} {t:.1f}s"
        x, y = 6, thumb.height - font_size - 8
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            draw.text((x + dx, y + dy), label, fill="black", font=font)
        draw.text((x, y), label, fill="white", font=font)

        thumbs.append(thumb)

    # Compute grid dimensions
    rows = (len(thumbs) + cols - 1) // cols
    thumb_h = thumbs[0].height
    pad = 4
    grid_w = cols * thumb_width + (cols + 1) * pad
    grid_h = rows * thumb_h + (rows + 1) * pad

    grid = Image.new("RGB", (grid_w, grid_h), "black")
    for i, thumb in enumerate(thumbs):
        r, c = divmod(i, cols)
        x = pad + c * (thumb_width + pad)
        y = pad + r * (thumb_h + pad)
        grid.paste(thumb, (x, y))

    buf = io.BytesIO()
    grid.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _ollama_chat(
    prompt: str,
    images: list[str],
    schema: dict | None = None,
    timeout: int = OLLAMA_TIMEOUT,
    retries: int = 3,
    backoff_ms: float = 2000.0,
) -> str:
    """Send a multimodal chat request to Ollama with exponential-backoff retry.

    Gemma 4 e4b on Metal can OOM-return 500s when the runner is near capacity.
    Retries with increasing delay give the model time to recover between attempts.

        *images* is a list of base64-encoded image or audio data.
        *schema* if provided, passed as structured output format (Pydantic JSON schema).
    """
    last_exc: Exception | None = None

    for attempt in range(retries):
        try:
            payload = json.dumps(
                {
                    "model": OLLAMA_MODEL,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                            "images": images,
                        }
                    ],
                    "stream": False,
                    "format": schema if schema else "json",
                    # llama.cpp #21825: audio encoder emits ~750 tokens; keeping num_ctx
                    # modest prevents KV-cache contention with audio embeddings that
                    # otherwise triggers the GGML alignment crash (ollama #15333).
                    "options": {"num_ctx": 8192},
                }
            ).encode()

            req = urllib.request.Request(
                f"{OLLAMA_BASE_URL}/api/chat",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            resp = urllib.request.urlopen(req, timeout=timeout)
            data = json.loads(resp.read())
            return data["message"]["content"]
        except (urllib.error.HTTPError, urllib.error.URLError, OSError) as exc:
            last_exc = exc
            is_500 = hasattr(exc, "code") and exc.code == 500
            if attempt < retries - 1:
                delay = backoff_ms * (2**attempt) / 1000.0
                log.warning(
                    "Ollama %s (attempt %d/%d), retrying in %.1fs",
                    "HTTP 500" if is_500 else type(exc).__name__,
                    attempt + 1,
                    retries,
                    delay,
                )
                time.sleep(delay)

    assert last_exc is not None
    raise last_exc


def _parse_ollama_response(
    raw: str,
    filename: str,
    file_path: str,
    fps: float,
    duration: float,
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

    # Adaptive fps: treat the caller's fps as a ceiling, not a mandate.
    # Ollama/gemma 4 chokes (HTTP 500) once the combined payload gets too
    # large — empirically ~35 extracted frames is the safe ceiling on M5 Max.
    # For long clips, scale fps down so we stay within the budget. Short
    # clips keep the requested density.
    MAX_FRAMES = 30
    effective_fps = fps
    if analyze_duration > 0 and analyze_duration * fps > MAX_FRAMES:
        effective_fps = MAX_FRAMES / analyze_duration

    log.info(
        "Ollama analyzing %s (%.1fs at %.2f fps [requested %.2f], range %.1f-%.1fs)",
        video_path.name,
        analyze_duration,
        effective_fps,
        fps,
        start_sec,
        analyze_end,
    )

    # Extract frames with source frame numbers. extract_frames self-cleans
    # on its own failure paths, so on a raise from here, no dir leaks.
    frame_results = extract_frames(
        video_path, fps=effective_fps, start_sec=start_sec, end_sec=end_sec
    )
    # frame_results is guaranteed non-empty (extract_frames raises otherwise)
    # but keep a defensive check for clarity.
    if not frame_results:
        raise RuntimeError(f"No frames extracted from {video_path.name}")

    frame_dir: Path | None = frame_results[0][0].parent
    audio_path: Path | None = None

    # Once frames exist on disk, EVERY exit path (success, Ollama HTTP 500,
    # Pydantic validation error, KeyboardInterrupt) must clean up the temp
    # dir. The previous code only cleaned on the success branch, leaking
    # ~50 frames per failed analysis. The try/finally is the fix.
    try:
        # Pick detail frames (evenly spaced subset of extracted frames).
        max_detail_frames = 8
        n = len(frame_results)
        if n <= max_detail_frames:
            detail_indices = list(range(n))
        else:
            step = (n - 1) / (max_detail_frames - 1)
            detail_indices = [round(i * step) for i in range(max_detail_frames)]

        timestamps: list[str] = []
        for i, (frame_path, t) in enumerate(frame_results):
            timestamps.append(f"#{i + 1} {t:.1f}s")

        # Build the images array in the ORDER gemma 4's model card recommends:
        # "place image and/or audio content BEFORE the text in your prompt."
        # Within the array, the model card ordering guidance puts audio first
        # among the non-text modalities — audio conditions vision interpretation
        # (voiceover, sync cues) more than the other way around.
        images: list[str] = []
        detail_labels: list[str] = []

        # 1) Audio FIRST (if safe to include). Metal's compute graph
        # (ggml-metal-device.m:608 GGML_ASSERT([rsets->data count] == 0))
        # aborts the llama runner when the combined vision + audio payload
        # exceeds ~20s of total encoder work. Empirically clips ≤17s survive
        # with both. For longer clips we must drop one — vision wins because
        # it covers the full clip duration.
        _COMBINED_SAFE_DURATION = 17.0
        audio_included = False
        if analyze_duration <= _COMBINED_SAFE_DURATION:
            audio_path = extract_audio(
                video_path, start_sec=start_sec, end_sec=analyze_end
            )
            if audio_path:
                images.append(base64.b64encode(audio_path.read_bytes()).decode())
                audio_included = True
        else:
            log.info(
                "Audio skipped — clip %.1fs exceeds combined safe duration %.1fs (Metal ceiling)",
                analyze_duration,
                _COMBINED_SAFE_DURATION,
            )

        # 2) Contact sheet (temporal/motion overview of every frame).
        sheet_bytes = build_contact_sheet(frame_results)
        images.append(base64.b64encode(sheet_bytes).decode())

        # 3) Labeled detail frames for fine inspection.
        for idx in detail_indices:
            frame_path, t = frame_results[idx]
            label = f"#{idx + 1} {t:.1f}s"
            labeled_bytes = _label_frame(frame_path, label)
            images.append(base64.b64encode(labeled_bytes).decode())
            detail_labels.append(label)

        # Build prompt
        prompt = OLLAMA_VIDEO_PROMPT.format(
            frame_timestamps=", ".join(timestamps),
            frame_count=len(frame_results),
            detail_frames=", ".join(detail_labels),
            detail_count=len(detail_labels),
            duration=analyze_duration,
            filename=video_path.name,
            file_path=str(video_path),
            model=OLLAMA_MODEL,
            fps=round(native_fps, 3),
        )

        # Call Ollama — use fast "json" mode, then validate with Pydantic.
        # Structured schema mode is 50-70% slower due to constrained decoding.
        from .schemas import VideoSidecar

        raw = _ollama_chat(prompt, images)

        # Parse and validate — force through Pydantic for format parity with Gemini
        sidecar = _parse_ollama_response(
            raw,
            video_path.name,
            str(video_path),
            round(native_fps, 3),
            round(duration, 3),
        )

        # Validate via Pydantic — coerces types and fills defaults
        validated = VideoSidecar.model_validate(sidecar)
        sidecar = validated.model_dump()

        return sidecar
    finally:
        _cleanup_frame_dir(frame_dir)
        _cleanup_audio_file(audio_path)


# Empirically determined on M5 Max + ollama 0.20.6 Metal backend:
# ggml-metal-device.m:608 `GGML_ASSERT([rsets->data count] == 0)` fires above
# ~22s per request. CPU-only (num_gpu=0) extends this to ~35s but adds latency.
# 20s chunks stay comfortably inside the Metal ceiling. Bug tracked at
# llama.cpp PR #17869 (crash reporting) — underlying fix still upstream.
# Note: for pure verbatim transcription whisper.cpp is faster and more accurate;
# this path is best used for semantic audio understanding.
_AUDIO_CHUNK_SEC = 20.0


def _chunk_audio(
    audio_path: Path, chunk_sec: float = _AUDIO_CHUNK_SEC
) -> list[tuple[Path, float]]:
    """Split audio into ≤*chunk_sec* WAVs. Returns [(path, start_offset_sec), ...].

    Fixed-length splits (not silence-aware) — deterministic and simple.
    Caller must clean up temp files other than the original input.
    """
    duration = ffprobe_duration(audio_path) or 0.0
    if duration <= chunk_sec:
        return [(audio_path, 0.0)]

    out: list[tuple[Path, float]] = []
    t = 0.0
    while t < duration:
        end = min(t + chunk_sec, duration)
        chunk = Path(tempfile.mktemp(suffix=".wav", prefix="ollama_audio_chunk_"))
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(audio_path),
                "-ss",
                f"{t:.3f}",
                "-to",
                f"{end:.3f}",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(chunk),
            ],
            check=True,
        )
        out.append((chunk, t))
        t = end
    return out


def _analyze_audio_single(src: Path, prompt: str, max_retries: int = 3) -> str:
    """Call _ollama_chat with progressive-trim retry on HTTP 500.

    The ollama #15333 crash is sensitive to exact audio token count; shortening
    the clip by 0.5s per retry shifts past the offending alignment boundary.
    """
    src_duration = ffprobe_duration(src) or 0.0
    last_exc: Exception | None = None

    for attempt in range(max_retries):
        trim = attempt * 0.5
        trimmed: Path | None = None
        if trim > 0:
            trimmed = Path(tempfile.mktemp(suffix=".wav", prefix="ollama_audio_trim_"))
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(src),
                    "-t",
                    f"{max(0.5, src_duration - trim):.3f}",
                    "-c",
                    "copy",
                    str(trimmed),
                ],
                check=True,
            )
            payload_path = trimmed
        else:
            payload_path = src

        try:
            audio_b64 = base64.b64encode(payload_path.read_bytes()).decode()
            return _ollama_chat(prompt, [audio_b64])
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code != 500 or attempt == max_retries - 1:
                raise
            log.warning(
                "Ollama audio HTTP 500 (attempt %d/%d), retrying with %.1fs trim",
                attempt + 1,
                max_retries,
                trim + 0.5,
            )
        finally:
            if trimmed is not None:
                trimmed.unlink(missing_ok=True)

    assert last_exc is not None
    raise last_exc


def analyze_audio(audio_path: Path, engine: str = "auto") -> dict:
    """Analyze a standalone audio file.

    *engine*:
      - "auto" (default): whisper.cpp for speech (fast, verbatim, word-level
        timestamps). Falls back to ollama/gemma4 music analysis if whisper
        returns empty transcript (pure music/ambience) OR if whisper is
        unavailable.
      - "whisper": force whisper.cpp.
      - "ollama": force gemma4 — expect 500s/crashes on >20s clips.
    """
    duration = ffprobe_duration(audio_path) or 0.0

    if engine in ("auto", "whisper"):
        try:
            from .whisper import transcribe_audio as _whisper

            log.info("Whisper transcribing %s (%.1fs)", audio_path.name, duration)
            w = _whisper(audio_path)
            if w["transcript"] or engine == "whisper":
                from .schemas import AudioSidecar

                return AudioSidecar.model_validate(
                    {
                        "filename": audio_path.name,
                        "file_path": str(audio_path),
                        "media_type": "audio",
                        "analysis_model": f"whisper.cpp/{w['model']}",
                        "duration": round(duration, 3),
                        "transcript": w["transcript"],
                        "words": w["words"],
                        "transcription_engine": w["engine"],
                    }
                ).model_dump()
            log.info(
                "Whisper found no speech in %s; falling through to gemma4",
                audio_path.name,
            )
        except FileNotFoundError as e:
            if engine == "whisper":
                raise
            log.warning("Whisper unavailable (%s); falling back to gemma4", e)
        except subprocess.CalledProcessError as e:
            if engine == "whisper":
                raise
            log.warning("Whisper failed (%s); falling back to gemma4", e)

    log.info("Ollama analyzing audio %s (%.1fs)", audio_path.name, duration)

    chunks = _chunk_audio(audio_path)
    all_sections: list[dict] = []
    transcript_parts: list[str] = []

    try:
        for chunk_path, offset in chunks:
            chunk_dur = ffprobe_duration(chunk_path) or 0.0
            prompt = OLLAMA_AUDIO_PROMPT.format(
                filename=chunk_path.name,
                file_path=str(chunk_path),
                model=OLLAMA_MODEL,
                duration=round(chunk_dur, 3),
            )
            raw = _analyze_audio_single(chunk_path, prompt)

            text = raw.strip()
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\s*", "", text)
                text = re.sub(r"\s*```$", "", text)
            chunk_data = json.loads(text)

            for sec in chunk_data.get("sections", []):
                if "start_sec" in sec:
                    sec["start_sec"] = round(sec["start_sec"] + offset, 3)
                if "end_sec" in sec:
                    sec["end_sec"] = round(sec["end_sec"] + offset, 3)
                all_sections.append(sec)

            if chunk_data.get("transcript"):
                transcript_parts.append(chunk_data["transcript"])
    finally:
        for chunk_path, _ in chunks:
            if chunk_path != audio_path:
                chunk_path.unlink(missing_ok=True)

    data: dict = {
        "filename": audio_path.name,
        "file_path": str(audio_path),
        "media_type": "audio",
        "analysis_model": OLLAMA_MODEL,
        "duration": round(duration, 3),
        "sections": all_sections,
    }
    if transcript_parts:
        data["transcript"] = " ".join(transcript_parts)

    from .schemas import AudioSidecar

    validated = AudioSidecar.model_validate(data)
    return validated.model_dump()
