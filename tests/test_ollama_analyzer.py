"""Tests for Ollama analyzer — JSON parsing and validation."""

import json

import pytest


def test_parse_ollama_response_valid_json():
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
    assert result["fps"] == 59.94
    assert result["duration"] == 12.0
    assert len(result["segments"]) == 1
    assert result["segments"][0]["quality_score"] == 8


def test_parse_ollama_response_markdown_wrapped():
    from resolve_assistant.ollama_analyzer import _parse_ollama_response

    raw = '```json\n{"segments": [{"start_sec": 0, "end_sec": 3, "type": "b-roll", "description": "test", "quality_score": 5, "tags": []}]}\n```'

    result = _parse_ollama_response(raw, "test.mov", "/path/test.mov", 30.0, 3.0)
    assert len(result["segments"]) == 1
    assert result["analysis_model"] == "gemma4:e4b"


def test_parse_ollama_response_invalid_json_raises():
    from resolve_assistant.ollama_analyzer import _parse_ollama_response

    with pytest.raises(ValueError, match="parse"):
        _parse_ollama_response("not json at all", "test.mov", "/p", 30.0, 3.0)


def test_parse_ollama_response_preserves_model_fields():
    """If the model returns its own filename/fps/etc, our defaults don't clobber them."""
    from resolve_assistant.ollama_analyzer import _parse_ollama_response

    raw = json.dumps({
        "filename": "model_chose_this.mov",
        "fps": 24.0,
        "segments": [],
    })

    result = _parse_ollama_response(raw, "override.mov", "/path", 59.94, 10.0)
    # setdefault doesn't overwrite existing keys
    assert result["filename"] == "model_chose_this.mov"
    assert result["fps"] == 24.0
    # But missing keys get defaults
    assert result["duration"] == 10.0
    assert result["media_type"] == "video"
