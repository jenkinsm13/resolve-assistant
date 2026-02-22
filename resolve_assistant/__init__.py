"""
resolve_assistant — AI-powered video editing assistant.
Uses Gemini to analyze footage, plan edits, and build timelines in DaVinci Resolve.
"""

from .config import mcp  # noqa: F401 — re-export for entry points


def main():
    """Entry point for `resolve-assistant` console script."""
    mcp.run()


# Importing these modules registers their @mcp.tool decorators.
from . import ingest              # noqa: F401  — ingest_footage, ingest_status
from . import build               # noqa: F401  — build_timeline, build_status
