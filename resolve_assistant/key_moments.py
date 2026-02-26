"""
Build a 'Key Moments' timeline: one clip per segment found in sidecar JSONs.

Every segment identified by Gemini during ingestion becomes a separate clip
on the timeline, giving the editor a visual index of every distinct moment
across all analysed footage.
"""

import json
from pathlib import Path

from .config import log, mcp
from .media import load_sidecars
from .resolve import get_resolve
from .resolve_build import build_timeline_direct
from .timeline import render_xml


def _segments_to_edit_plan(
    sidecars: list[dict],
    timeline_name: str = "Key Moments",
    clip_filter: list[str] | None = None,
) -> dict:
    """Convert sidecar segments into an edit plan (same schema as Gemini EDL).

    Args:
        sidecars: Loaded sidecar dicts from load_sidecars().
        timeline_name: Name for the resulting timeline.
        clip_filter: Optional list of filename stems to include.
                     If None, all video sidecars are included.
    """
    cuts: list[dict] = []

    for sc in sidecars:
        media_type = sc.get("media_type", "")
        if media_type == "audio" or media_type.startswith("audio/"):
            continue

        file_path = sc.get("file_path", "")
        stem = Path(file_path).stem if file_path else sc.get("filename", "")

        if clip_filter and stem not in clip_filter:
            continue

        for seg in sc.get("segments", []):
            start = float(seg.get("start_sec", 0))
            end = float(seg.get("end_sec", 0))
            if end <= start:
                continue

            cuts.append({
                "track": 1,
                "source_file": file_path,
                "start_sec": start,
                "end_sec": end,
            })

    # Sort by source file then by start time within each file
    cuts.sort(key=lambda c: (c["source_file"], c["start_sec"]))

    return {
        "timeline_name": timeline_name,
        "cuts": cuts,
    }


@mcp.tool
def build_key_moments_timeline(
    folder_path: str,
    timeline_name: str = "Key Moments",
    clip_filter: str = "",
) -> str:
    """Build a timeline with one clip per key moment found in sidecar metadata.

    Every segment identified during ingestion becomes a separate clip on the
    timeline — if a clip has 3 key moments, that's 3 clips.  No Gemini call
    is needed; this parses existing sidecar JSONs only.

    *clip_filter*: optional comma-separated list of filename stems to include
    (e.g. "DSC_4293,DSC_4294").  If empty, all video sidecars are used.
    """
    root = Path(folder_path).resolve()
    if not root.is_dir():
        return f"Error: '{folder_path}' is not a valid directory."

    sidecars = load_sidecars(root)
    if not sidecars:
        return "No sidecar JSONs found. Run ingest_footage first."

    filter_list = (
        [s.strip() for s in clip_filter.split(",") if s.strip()]
        if clip_filter else None
    )

    edit_plan = _segments_to_edit_plan(sidecars, timeline_name, filter_list)
    cuts = edit_plan.get("cuts", [])
    if not cuts:
        return "No segments found in sidecars (filter may have excluded all clips)."

    # Render XML backup
    xml_path = None
    try:
        xml_path, _ = render_xml(root, edit_plan, sidecars)
    except Exception as exc:
        log.warning("XML render failed (non-fatal): %s", exc)

    # Build directly in Resolve
    resolve_obj = get_resolve()
    if resolve_obj:
        success, msg = build_timeline_direct(edit_plan, resolve_obj)
        if not success and xml_path:
            msg += f" Backup XML: {xml_path.name}"
    else:
        xml_note = f" XML: {xml_path.name}" if xml_path else ""
        msg = f"Resolve not running — import XML manually.{xml_note}"

    # Save the EDL for reference (.key-moments prefix avoids build_timeline cache collision)
    safe_name = timeline_name.replace(":", " -").replace("/", "-").replace("\\", "-")
    edl_path = root / f".key-moments-{safe_name}.edl.json"
    edl_path.write_text(json.dumps(edit_plan, indent=2))

    n_clips = len([
        sc for sc in sidecars
        if not (sc.get("media_type", "").startswith("audio"))
    ])
    filtered = f" (filtered to {len(filter_list)} clips)" if filter_list else ""
    return (
        f"Key Moments timeline built: {len(cuts)} segments from "
        f"{n_clips} clip(s){filtered}. {msg}"
    )
