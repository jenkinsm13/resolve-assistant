"""
whisper.cpp integration for speech transcription.

Runs whisper-cli locally (via Homebrew's whisper-cpp) to produce verbatim
word-level timestamps. This is the default path for any clip with speech —
gemma4 via ollama paraphrases and crashes on >20s chunks (ggml-metal-device.m
#608 assertion on M5 Max), so whisper is the right tool for speech audio.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path(
    os.environ.get(
        "WHISPER_MODEL_PATH",
        str(Path.home() / ".local/share/whisper-models/ggml-medium.en.bin"),
    )
)
WHISPER_BIN = os.environ.get("WHISPER_CLI", "whisper-cli")


def _ensure_16k_mono_wav(audio_path: Path) -> tuple[Path, bool]:
    """Return (wav_path, is_temp). Re-encodes if needed for whisper-cpp."""
    # whisper-cpp specifically requires 16kHz mono 16-bit PCM WAV
    if audio_path.suffix.lower() == ".wav":
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=sample_rate,channels,codec_name",
                "-of",
                "default=nw=1:nk=1",
                str(audio_path),
            ],
            capture_output=True,
            text=True,
        )
        lines = probe.stdout.strip().splitlines()
        # codec, sr, ch
        if (
            len(lines) >= 3
            and lines[0] == "pcm_s16le"
            and lines[1] == "16000"
            and lines[2] == "1"
        ):
            return audio_path, False

    out = Path(tempfile.mktemp(suffix=".wav", prefix="whisper_in_"))
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(audio_path),
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
    return out, True


def transcribe_audio(
    audio_path: Path,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> dict:
    """Transcribe speech with word-level timestamps.

    Returns a dict with keys:
      - transcript: full text
      - words: [{start_sec, end_sec, text}, ...] — word-level (-ml 1)
      - engine: "whisper.cpp"
      - model: model filename
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Whisper model not found at {model_path}. "
            f"Download from https://huggingface.co/ggerganov/whisper.cpp or set "
            f"WHISPER_MODEL_PATH."
        )

    wav, is_temp = _ensure_16k_mono_wav(audio_path)
    out_stem = Path(tempfile.mktemp(prefix="whisper_out_"))

    try:
        subprocess.run(
            [
                WHISPER_BIN,
                "-m",
                str(model_path),
                "-f",
                str(wav),
                "-ml",
                "1",  # word-level token timestamps
                "-oj",  # JSON output
                "-of",
                str(out_stem),
            ],
            check=True,
            capture_output=True,
        )
        json_path = out_stem.with_suffix(".json")
        data = json.loads(json_path.read_text())
    finally:
        if is_temp:
            wav.unlink(missing_ok=True)
        for ext in (".json",):
            out_stem.with_suffix(ext).unlink(missing_ok=True)

    segments = data.get("transcription", [])
    words = [
        {
            "start_sec": s["offsets"]["from"] / 1000.0,
            "end_sec": s["offsets"]["to"] / 1000.0,
            "text": s["text"],
        }
        for s in segments
        if s.get("text", "").strip()
    ]
    transcript = "".join(s["text"] for s in segments).strip()

    return {
        "transcript": transcript,
        "words": words,
        "engine": "whisper.cpp",
        "model": model_path.name,
    }
