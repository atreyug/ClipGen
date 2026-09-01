"""
services/smart_layouts.py
──────────────────────────
Smart layout engine for multi-face video clips.

When more than one person is on screen the engine decides the best
visual composition and generates the FFmpeg filter graph to render it.

Supported layouts
─────────────────
SINGLE          – single-face crop (classic ClipGen behaviour)
SPLIT_VERTICAL  – two faces side-by-side  (left | right)
SPLIT_HORIZONTAL– two faces stacked       (top / bottom)
PICTURE_IN_PIC  – main face large, secondary face in corner PiP
TRIO_STACK      – three faces stacked vertically
GRID_2x2        – four faces in a 2×2 grid
AUTO            – engine picks the best layout based on face count & sizes

All layouts produce output at the target resolution (default 1080×1920 for
9:16 or 1920×1080 for 16:9).

Integration in utils/clip.py:
    from services.smart_layouts import SmartLayoutEngine, LayoutPreset

    engine = SmartLayoutEngine(frame_w, frame_h, target_aspect="9:16")
    vf_filter, layout_used = engine.build_filter(
        face_timeline, speaker_segments, clip_duration
    )
    # vf_filter replaces the crop filter in clipper.py subprocess call
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Enums & constants
# ──────────────────────────────────────────────────────────────────────────────

class LayoutPreset(str, Enum):
    SINGLE           = "single"
    SPLIT_VERTICAL   = "split_vertical"
    SPLIT_HORIZONTAL = "split_horizontal"
    PICTURE_IN_PIC   = "picture_in_pic"
    TRIO_STACK       = "trio_stack"
    GRID_2X2         = "grid_2x2"
    AUTO             = "auto"


# Standard output resolutions
RESOLUTION_9_16 = (1080, 1920)   # (width, height)
RESOLUTION_16_9 = (1920, 1080)

# PiP inset fraction of the main face area
PIP_FRACTION = 0.28

# Gap between tiles in pixels
TILE_GAP = 6


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FaceRegion:
    """A face with its source crop geometry for a particular frame snapshot."""
    face_id:   int
    src_x:     int   # left edge of desired crop in source frame
    src_y:     int   # top  edge of desired crop in source frame
    src_w:     int   # crop width  in source frame
    src_h:     int   # crop height in source frame
    box_area:  float = 0.0


def _clamp(val: int, lo: int, hi: int) -> int:
    return max(lo, min(val, hi))


def _face_to_region(
    face,           # TrackedFace
    frame_w: int,
    frame_h: int,
    tile_w:  int,
    tile_h:  int,
) -> FaceRegion:
    """
    Build a FaceRegion that crops `tile_w × tile_h` centred on the face.
    """
    cx = int(face.box.cx)
    cy = int(face.box.cy)
    x  = _clamp(cx - tile_w // 2, 0, max(0, frame_w - tile_w))
    y  = _clamp(cy - tile_h // 2, 0, max(0, frame_h - tile_h))
    return FaceRegion(
        face_id=face.face_id,
        src_x=x, src_y=y,
        src_w=tile_w, src_h=tile_h,
        box_area=face.box.area,
    )


# ──────────────────────────────────────────────────────────────────────────────
# FFmpeg filter builders
# ──────────────────────────────────────────────────────────────────────────────

def _single_filter(
    region: FaceRegion,
    out_w:  int,
    out_h:  int,
) -> str:
    """Single face, cropped and scaled to full output size."""
    return (
        f"[0:v]crop={region.src_w}:{region.src_h}:{region.src_x}:{region.src_y},"
        f"scale={out_w}:{out_h}:force_original_aspect_ratio=increase,"
        f"crop={out_w}:{out_h}[v]"
    )


def _split_vertical_filter(
    r0: FaceRegion, r1: FaceRegion,
    out_w: int, out_h: int,
) -> str:
    """
    Side-by-side: [face0 | face1]
    Each tile is (out_w/2 - gap/2) × out_h, then hstacked.
    """
    tile_w = (out_w - TILE_GAP) // 2
    tile_h = out_h
    return (
        f"[0:v]crop={r0.src_w}:{r0.src_h}:{r0.src_x}:{r0.src_y},"
        f"scale={tile_w}:{tile_h}:force_original_aspect_ratio=increase,"
        f"crop={tile_w}:{tile_h}[left];"
        f"[0:v]crop={r1.src_w}:{r1.src_h}:{r1.src_x}:{r1.src_y},"
        f"scale={tile_w}:{tile_h}:force_original_aspect_ratio=increase,"
        f"crop={tile_w}:{tile_h}[right];"
        f"[left][right]hstack=inputs=2[v]"
    )


def _split_horizontal_filter(
    r0: FaceRegion, r1: FaceRegion,
    out_w: int, out_h: int,
) -> str:
    """
    Stacked: [face0 on top / face1 on bottom]
    """
    tile_w = out_w
    tile_h = (out_h - TILE_GAP) // 2
    return (
        f"[0:v]crop={r0.src_w}:{r0.src_h}:{r0.src_x}:{r0.src_y},"
        f"scale={tile_w}:{tile_h}:force_original_aspect_ratio=increase,"
        f"crop={tile_w}:{tile_h}[top];"
        f"[0:v]crop={r1.src_w}:{r1.src_h}:{r1.src_x}:{r1.src_y},"
        f"scale={tile_w}:{tile_h}:force_original_aspect_ratio=increase,"
        f"crop={tile_w}:{tile_h}[bottom];"
        f"[top][bottom]vstack=inputs=2[v]"
    )


def _pip_filter(
    main: FaceRegion, pip: FaceRegion,
    out_w: int, out_h: int,
    pip_fraction: float = PIP_FRACTION,
) -> str:
    """
    Picture-in-picture: main face fills frame, pip face overlaid in top-right.
    """
    pip_w = int(out_w * pip_fraction)
    pip_h = int(out_h * pip_fraction)
    # PiP position: top-right with a small margin
    margin = 20
    pip_x  = out_w - pip_w - margin
    pip_y  = margin
    return (
        # Main background
        f"[0:v]crop={main.src_w}:{main.src_h}:{main.src_x}:{main.src_y},"
        f"scale={out_w}:{out_h}:force_original_aspect_ratio=increase,"
        f"crop={out_w}:{out_h}[bg];"
        # PiP foreground
        f"[0:v]crop={pip.src_w}:{pip.src_h}:{pip.src_x}:{pip.src_y},"
        f"scale={pip_w}:{pip_h}:force_original_aspect_ratio=increase,"
        f"crop={pip_w}:{pip_h},"
        # rounded-corner look: slight pad + black border
        f"pad={pip_w + 4}:{pip_h + 4}:2:2:black[pip];"
        # Overlay
        f"[bg][pip]overlay={pip_x}:{pip_y}[v]"
    )


def _trio_stack_filter(
    r0: FaceRegion, r1: FaceRegion, r2: FaceRegion,
    out_w: int, out_h: int,
) -> str:
    """Three faces stacked vertically (equal thirds)."""
    tile_h = (out_h - 2 * TILE_GAP) // 3
    tile_w = out_w
    return (
        f"[0:v]crop={r0.src_w}:{r0.src_h}:{r0.src_x}:{r0.src_y},"
        f"scale={tile_w}:{tile_h}:force_original_aspect_ratio=increase,"
        f"crop={tile_w}:{tile_h}[t0];"
        f"[0:v]crop={r1.src_w}:{r1.src_h}:{r1.src_x}:{r1.src_y},"
        f"scale={tile_w}:{tile_h}:force_original_aspect_ratio=increase,"
        f"crop={tile_w}:{tile_h}[t1];"
        f"[0:v]crop={r2.src_w}:{r2.src_h}:{r2.src_x}:{r2.src_y},"
        f"scale={tile_w}:{tile_h}:force_original_aspect_ratio=increase,"
        f"crop={tile_w}:{tile_h}[t2];"
        f"[t0][t1][t2]vstack=inputs=3[v]"
    )


def _grid_2x2_filter(
    r0: FaceRegion, r1: FaceRegion, r2: FaceRegion, r3: FaceRegion,
    out_w: int, out_h: int,
) -> str:
    """2×2 grid of four faces."""
    tile_w = (out_w - TILE_GAP) // 2
    tile_h = (out_h - TILE_GAP) // 2

    def tile(r: FaceRegion, label: str) -> str:
        return (
            f"[0:v]crop={r.src_w}:{r.src_h}:{r.src_x}:{r.src_y},"
            f"scale={tile_w}:{tile_h}:force_original_aspect_ratio=increase,"
            f"crop={tile_w}:{tile_h}[{label}]"
        )

    return (
        f"{tile(r0,'g0')};"
        f"{tile(r1,'g1')};"
        f"{tile(r2,'g2')};"
        f"{tile(r3,'g3')};"
        f"[g0][g1]hstack=inputs=2[row0];"
        f"[g2][g3]hstack=inputs=2[row1];"
        f"[row0][row1]vstack=inputs=2[v]"
    )


# ──────────────────────────────────────────────────────────────────────────────
# SmartLayoutEngine
# ──────────────────────────────────────────────────────────────────────────────

class SmartLayoutEngine:
    """
    Decides the best multi-face layout and generates the FFmpeg filter graph.

    Parameters
    ----------
    frame_w         : Source video width.
    frame_h         : Source video height.
    target_aspect   : "9:16" or "16:9".
    layout          : LayoutPreset — use AUTO to let the engine decide.
    preferred_face  : face_id that should always be the "main" face.
    """

    def __init__(
        self,
        frame_w:         int,
        frame_h:         int,
        target_aspect:   str          = "9:16",
        layout:          LayoutPreset = LayoutPreset.AUTO,
        preferred_face:  Optional[int] = None,
    ):
        self.frame_w        = frame_w
        self.frame_h        = frame_h
        self.layout         = layout
        self.preferred_face = preferred_face

        if target_aspect == "9:16":
            self.out_w, self.out_h = RESOLUTION_9_16
        else:
            self.out_w, self.out_h = RESOLUTION_16_9

    # ── public API ────────────────────────────────────────────────────────────

    def build_filter(
        self,
        face_timeline:    List,                  # List[FrameFaceData]
        speaker_segments: Optional[List] = None, # List[SpeakerSegment]
        clip_duration:    float          = 30.0,
    ) -> Tuple[str, LayoutPreset]:
        """
        Build the FFmpeg complex filter string for the chosen layout.

        Returns
        -------
        (filter_str, layout_used)
          filter_str  : FFmpeg -filter_complex value (uses stream [v] as output).
          layout_used : Which LayoutPreset was actually applied.
        """
        # Snapshot: pick a representative frame near the clip midpoint
        representative = self._pick_representative_frame(face_timeline, clip_duration / 2)
        face_count = len(representative.faces) if representative else 0

        # Determine layout
        chosen_layout = self.layout
        if chosen_layout == LayoutPreset.AUTO:
            chosen_layout = self._auto_pick(face_count, speaker_segments)

        filter_str = self._build(chosen_layout, representative, speaker_segments)
        return filter_str, chosen_layout

    def layout_for_face_count(self, n: int) -> LayoutPreset:
        """Public helper to know which layout the AUTO mode would pick for n faces."""
        return self._auto_pick(n, None)

    # ── private ───────────────────────────────────────────────────────────────

    def _auto_pick(
        self,
        face_count: int,
        speaker_segments: Optional[List],
    ) -> LayoutPreset:
        """Heuristic layout selection."""
        if face_count <= 1:
            return LayoutPreset.SINGLE
        if face_count == 2:
            # Prefer PiP if there is a dominant active speaker
            if speaker_segments and self._has_dominant_speaker(speaker_segments):
                return LayoutPreset.PICTURE_IN_PIC
            return LayoutPreset.SPLIT_VERTICAL
        if face_count == 3:
            return LayoutPreset.TRIO_STACK
        # 4+
        return LayoutPreset.GRID_2X2

    @staticmethod
    def _has_dominant_speaker(speaker_segments: List) -> bool:
        """
        Returns True if one speaker holds > 65% of speaking time.
        """
        if not speaker_segments:
            return False
        totals: Dict[str, float] = {}
        for seg in speaker_segments:
            dur = max(0.0, seg.end - seg.start)
            totals[seg.speaker_id] = totals.get(seg.speaker_id, 0.0) + dur
        total = sum(totals.values())
        if total < 1e-6:
            return False
        return max(totals.values()) / total > 0.65

    def _pick_representative_frame(self, face_timeline: List, t: float):
        """Find the timeline frame closest to time t."""
        if not face_timeline:
            return None
        return min(face_timeline, key=lambda f: abs(f.timestamp - t))

    def _make_regions(
        self,
        frame_data,
        speaker_segments: Optional[List],
        tile_w: int,
        tile_h: int,
        n: int = 2,
    ) -> List[FaceRegion]:
        """
        Build up to `n` FaceRegion objects sorted by relevance:
        active speaker first, then by face area.
        """
        if frame_data is None or not frame_data.faces:
            # No faces – fallback to centre crop
            cx = max(0, self.frame_w // 2 - tile_w // 2)
            cy = max(0, self.frame_h // 2 - tile_h // 2)
            dummy = FaceRegion(
                face_id=-1,
                src_x=cx, src_y=cy,
                src_w=tile_w, src_h=tile_h,
                box_area=0.0,
            )
            return [dummy] * n

        t = frame_data.timestamp
        active_id = None
        if speaker_segments:
            for seg in speaker_segments:
                if seg.start <= t <= seg.end and seg.face_id is not None:
                    active_id = seg.face_id
                    break

        # Sort: preferred_face first, then active speaker, then by area
        def sort_key(face):
            if self.preferred_face is not None and face.face_id == self.preferred_face:
                return (0, -face.box.area)
            if face.face_id == active_id:
                return (1, -face.box.area)
            return (2, -face.box.area)

        sorted_faces = sorted(frame_data.faces, key=sort_key)[:n]

        # Pad with copies of the last face if fewer than n detected
        while len(sorted_faces) < n:
            sorted_faces.append(sorted_faces[-1])

        regions = [
            _face_to_region(f, self.frame_w, self.frame_h, tile_w, tile_h)
            for f in sorted_faces
        ]
        return regions

    def _build(
        self,
        layout: LayoutPreset,
        frame_data,
        speaker_segments: Optional[List],
    ) -> str:
        W, H = self.out_w, self.out_h

        if layout == LayoutPreset.SINGLE:
            # Single: crop tile proportional to output, centred on dominant face
            if frame_data and frame_data.faces:
                dom = frame_data.dominant_face()
                tile_w = min(self.frame_w, int(self.frame_h * W / H))
                tile_h = self.frame_h
                r = _face_to_region(dom, self.frame_w, self.frame_h, tile_w, tile_h)
            else:
                tile_w = min(self.frame_w, int(self.frame_h * W / H))
                cx = (self.frame_w - tile_w) // 2
                r = FaceRegion(-1, cx, 0, tile_w, self.frame_h)
            return _single_filter(r, W, H)

        elif layout == LayoutPreset.SPLIT_VERTICAL:
            tile_w = min(self.frame_w // 2, int(self.frame_h * (W // 2) / H))
            tile_h = self.frame_h
            r0, r1 = self._make_regions(frame_data, speaker_segments, tile_w, tile_h, 2)
            return _split_vertical_filter(r0, r1, W, H)

        elif layout == LayoutPreset.SPLIT_HORIZONTAL:
            tile_w = self.frame_w
            tile_h = min(self.frame_h // 2, int(self.frame_w * (H // 2) / W))
            r0, r1 = self._make_regions(frame_data, speaker_segments, tile_w, tile_h, 2)
            return _split_horizontal_filter(r0, r1, W, H)

        elif layout == LayoutPreset.PICTURE_IN_PIC:
            # Main gets full-frame crop; PiP gets a smaller tile
            main_w = min(self.frame_w, int(self.frame_h * W / H))
            main_h = self.frame_h
            pip_w  = int(main_w * PIP_FRACTION)
            pip_h  = int(main_h * PIP_FRACTION)
            regions = self._make_regions(frame_data, speaker_segments, main_w, main_h, 2)
            main = regions[0]
            # PiP region: re-compute with pip tile size
            if frame_data and len(frame_data.faces) >= 2:
                pip_face = sorted(frame_data.faces, key=lambda f: -f.box.area)[1]
                pip_r = _face_to_region(pip_face, self.frame_w, self.frame_h, pip_w, pip_h)
            else:
                pip_r = FaceRegion(-1, main.src_x, main.src_y, pip_w, pip_h)
            return _pip_filter(main, pip_r, W, H)

        elif layout == LayoutPreset.TRIO_STACK:
            tile_h = (self.frame_h - 2 * TILE_GAP) // 3
            tile_w = self.frame_w
            r0, r1, r2 = self._make_regions(frame_data, speaker_segments, tile_w, tile_h, 3)
            return _trio_stack_filter(r0, r1, r2, W, H)

        elif layout == LayoutPreset.GRID_2X2:
            tile_w = self.frame_w // 2
            tile_h = self.frame_h // 2
            r0, r1, r2, r3 = self._make_regions(frame_data, speaker_segments, tile_w, tile_h, 4)
            return _grid_2x2_filter(r0, r1, r2, r3, W, H)

        else:
            raise ValueError(f"Unknown layout preset: {layout}")


# ──────────────────────────────────────────────────────────────────────────────
# Convenience function
# ──────────────────────────────────────────────────────────────────────────────

def build_smart_layout_filter(
    face_timeline:    List,
    speaker_segments: Optional[List],
    frame_w:          int,
    frame_h:          int,
    clip_duration:    float,
    target_aspect:    str          = "9:16",
    layout:           LayoutPreset = LayoutPreset.AUTO,
    preferred_face:   Optional[int] = None,
) -> Tuple[str, str]:
    """
    One-shot helper.

    Returns
    -------
    (filter_complex_str, layout_name_str)
    """
    engine = SmartLayoutEngine(
        frame_w=frame_w,
        frame_h=frame_h,
        target_aspect=target_aspect,
        layout=layout,
        preferred_face=preferred_face,
    )
    filt, layout_used = engine.build_filter(face_timeline, speaker_segments, clip_duration)
    return filt, layout_used.value


def list_layout_presets() -> List[str]:
    return [p.value for p in LayoutPreset]
