"""
services/caption_styles.py
──────────────────────────
ASS caption generator with multiple pre-built style presets.

Preserves the original public API:
    - StylePreset
    - write_ass_captions_styled()
    - slice_words_for_clip()
    - group_words_into_lines()
    - list_style_presets()
    - get_style_info()
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

# ══════════════════════════════════════════════════════════════════
# Types
# ══════════════════════════════════════════════════════════════════

Word = dict[str, Any]            # {"text": str, "start": float, "end": float}
Line = dict[str, Any]            # {"text", "start", "end", "words": list[Word]}
DialogueBuilder = Callable[[Line], str]


# ══════════════════════════════════════════════════════════════════
# ASS Colour helpers  (ASS format: &HAABBGGRR&)
# ══════════════════════════════════════════════════════════════════

def rgb(r: int, g: int, b: int, a: int = 0) -> str:
    """Convert RGB(A) values to ASS colour string."""
    return f"&H{a:02X}{b:02X}{g:02X}{r:02X}"


class Colour:
    WHITE       = rgb(255, 255, 255)
    BLACK       = rgb(0, 0, 0)
    YELLOW      = rgb(255, 238, 0)
    ORANGE      = rgb(255, 138, 26)
    NEON_GREEN  = rgb(102, 255, 0)
    ORANGE_LINE = rgb(255, 128, 0)
    GREY_LIGHT  = rgb(232, 232, 232)
    GREY_DARK   = rgb(17, 17, 17)
    BG_DARK_50  = "&H80000000"
    BG_DARK_60  = "&H99000000"
    BG_DARK_66  = "&HAA000000"
    BG_DARK_73  = "&HBB000000"
    BG_DARK_80  = "&HCC000000"
    BG_DARK_87  = "&HDD000000"
    BG_SM       = "&HBB222222"
    TRANSPARENT = "&H00000000"


# ══════════════════════════════════════════════════════════════════
# Time helpers
# ══════════════════════════════════════════════════════════════════

def _ts(seconds: float) -> str:
    """Convert seconds to ASS timestamp H:MM:SS.cs."""
    cs_total = max(0, int(round(seconds * 100)))
    h, rem = divmod(cs_total, 360_000)
    m, rem = divmod(rem, 6_000)
    s, cs  = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _cs(seconds: float) -> int:
    """Seconds → centiseconds (non-negative int)."""
    return max(0, int(round(seconds * 100)))


# ══════════════════════════════════════════════════════════════════
# Word grouping
# ══════════════════════════════════════════════════════════════════

def slice_words_for_clip(
    all_words: Iterable[Word],
    clip_start: float,
    clip_end: float,
) -> list[Word]:
    """Return words within [clip_start, clip_end] with clip-relative times."""
    clip_duration = clip_end - clip_start
    return [
        {
            "text":  w["text"],
            "start": max(0.0, w["start"] - clip_start),
            "end":   min(clip_duration, w["end"] - clip_start),
        }
        for w in all_words
        if w["end"] > clip_start and w["start"] < clip_end
    ]


def group_words_into_lines(
    words: Sequence[Word],
    max_chars: int = 20,
    max_duration: float = 2.0,
    max_words: int = 3,
) -> list[Line]:
    """Group consecutive words into subtitle lines."""
    lines: list[Line] = []
    buf: list[Word] = []

    def flush() -> None:
        if not buf:
            return
        lines.append({
            "text":  " ".join(w["text"] for w in buf),
            "start": buf[0]["start"],
            "end":   buf[-1]["end"],
            "words": buf.copy(),
        })
        buf.clear()

    for w in words:
        if buf:
            new_text_len = sum(len(x["text"]) for x in buf) + len(buf) + len(w["text"])
            duration     = w["end"] - buf[0]["start"]
            if (len(buf) >= max_words
                    or new_text_len > max_chars
                    or duration > max_duration):
                flush()
        buf.append(w)

    flush()
    return lines


# ══════════════════════════════════════════════════════════════════
# ASS style definition
# ══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ASSStyleDef:
    """Represents one entry in the ASS [V4+ Styles] section."""
    name:             str
    fontname:         str   = "Arial"
    fontsize:         int   = 14
    primary_colour:   str   = Colour.WHITE
    secondary_colour: str   = "&H000000FF"
    outline_colour:   str   = Colour.BLACK
    back_colour:      str   = Colour.BG_DARK_50
    bold:             int   = -1     # -1 = true
    italic:           int   = 0
    underline:        int   = 0
    strikeout:        int   = 0
    scale_x:          int   = 100
    scale_y:          int   = 100
    spacing:          float = 0.0
    angle:            float = 0.0
    border_style:     int   = 1      # 1=outline+shadow, 3=opaque box
    outline:          float = 2.0
    shadow:           float = 1.0
    alignment:        int   = 2      # 2 = bottom-centre
    margin_l:         int   = 10
    margin_r:         int   = 10
    margin_v:         int   = 20
    encoding:         int   = 1

    def to_ass_line(self) -> str:
        fields = [
            self.name, self.fontname, self.fontsize,
            self.primary_colour, self.secondary_colour,
            self.outline_colour, self.back_colour,
            self.bold, self.italic, self.underline, self.strikeout,
            self.scale_x, self.scale_y,
            f"{self.spacing:.1f}", f"{self.angle:.1f}",
            self.border_style, f"{self.outline:.1f}", f"{self.shadow:.1f}",
            self.alignment, self.margin_l, self.margin_r, self.margin_v,
            self.encoding,
        ]
        return "Style: " + ",".join(str(f) for f in fields)


# ══════════════════════════════════════════════════════════════════
# Preset catalogue
# ══════════════════════════════════════════════════════════════════

class StylePreset(str, Enum):
    CLASSIC      = "classic"
    NEON_GREEN   = "neon_green"
    FIRE         = "fire"
    TYPEWRITER   = "typewriter"
    MINIMAL      = "minimal"
    BOLD_LOWER   = "bold_lower"
    SHADOW_POP   = "shadow_pop"
    KARAOKE      = "karaoke"
    OUTLINE_ONLY = "outline_only"
    SOCIAL_MEDIA = "social_media"

    @classmethod
    def coerce(cls, value: "StylePreset | str") -> "StylePreset":
        if isinstance(value, cls):
            return value
        try:
            return cls(value.lower())
        except ValueError as e:
            valid = ", ".join(p.value for p in cls)
            raise ValueError(f"Unknown style '{value}'. Valid: {valid}") from e


STYLE_CATALOGUE: dict[StylePreset, ASSStyleDef] = {
    StylePreset.CLASSIC: ASSStyleDef(
        name="Classic", fontsize=16,
        back_colour=Colour.BG_DARK_50,
        outline=2.5, shadow=1.5, margin_v=25,
    ),
    StylePreset.NEON_GREEN: ASSStyleDef(
        name="NeonGreen", fontname="Impact", fontsize=18,
        primary_colour=Colour.NEON_GREEN,
        back_colour=Colour.BG_DARK_80,
        border_style=3, outline=0.0, shadow=0.0,
        margin_v=30, spacing=0.5,
    ),
    StylePreset.FIRE: ASSStyleDef(
        name="Fire", fontname="Impact", fontsize=18,
        primary_colour=Colour.ORANGE,
        secondary_colour=Colour.YELLOW,
        back_colour=Colour.BG_DARK_66,
        outline=2.0, shadow=2.0, margin_v=30,
    ),
    StylePreset.TYPEWRITER: ASSStyleDef(
        name="Typewriter", fontname="Courier New", fontsize=14,
        primary_colour=Colour.GREY_LIGHT,
        outline_colour=Colour.GREY_DARK,
        back_colour=Colour.BG_DARK_87,
        bold=0, border_style=3, outline=0.0, shadow=0.0,
        spacing=1.0,
    ),
    StylePreset.MINIMAL: ASSStyleDef(
        name="Minimal", fontname="Helvetica Neue", fontsize=13,
        back_colour=Colour.TRANSPARENT,
        bold=0, outline=1.0, shadow=0.5,
    ),
    StylePreset.BOLD_LOWER: ASSStyleDef(
        name="BoldLower", fontname="Arial Black", fontsize=22,
        back_colour=Colour.BG_DARK_73,
        border_style=3, outline=0.0, shadow=0.0,
        margin_v=10, scale_x=105,
    ),
    StylePreset.SHADOW_POP: ASSStyleDef(
        name="ShadowPop", fontname="Impact", fontsize=26,
        back_colour=Colour.TRANSPARENT,
        outline=0.5, shadow=4.0, margin_v=35,
    ),
    StylePreset.KARAOKE: ASSStyleDef(
        name="Karaoke", fontsize=17,
        secondary_colour=Colour.YELLOW,
        back_colour=Colour.BG_DARK_60,
        outline=2.0, shadow=1.0, margin_v=25,
    ),
    StylePreset.OUTLINE_ONLY: ASSStyleDef(
        name="OutlineOnly", fontsize=18,
        outline_colour=Colour.ORANGE_LINE,
        back_colour=Colour.TRANSPARENT,
        outline=3.5, shadow=0.0, margin_v=25,
    ),
    StylePreset.SOCIAL_MEDIA: ASSStyleDef(
        name="SocialMedia", fontname="Arial Rounded MT Bold", fontsize=16,
        secondary_colour=Colour.YELLOW,
        outline_colour=rgb(34, 34, 34),
        back_colour=Colour.BG_SM,
        border_style=3, outline=1.5, shadow=0.0,
        margin_v=28, spacing=0.5,
    ),
}


# ══════════════════════════════════════════════════════════════════
# ASS header builder
# ══════════════════════════════════════════════════════════════════

_STYLE_FORMAT_FIELDS = (
    "Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
    "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
    "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
    "Alignment, MarginL, MarginR, MarginV, Encoding"
)

_EVENT_FORMAT_FIELDS = (
    "Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"
)


def _build_ass_header(
    styles: Sequence[ASSStyleDef],
    play_res_x: int,
    play_res_y: int,
) -> str:
    style_block = "\n".join(s.to_ass_line() for s in styles)
    return (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "WrapStyle: 0\n"
        "ScaledBorderAndShadow: yes\n"
        f"PlayResX: {play_res_x}\n"
        f"PlayResY: {play_res_y}\n\n"
        "[V4+ Styles]\n"
        f"Format: {_STYLE_FORMAT_FIELDS}\n"
        f"{style_block}\n\n"
        "[Events]\n"
        f"Format: {_EVENT_FORMAT_FIELDS}\n"
    )


# ══════════════════════════════════════════════════════════════════
# Dialogue builders
# ══════════════════════════════════════════════════════════════════

def _dialogue(start: float, end: float, style: str, text: str, layer: int = 0) -> str:
    return f"Dialogue: {layer},{_ts(start)},{_ts(end)},{style},,0,0,0,,{text}"


def _colour_switch_tag(
    default: str,
    active: str,
    revert: str,
    t_on_cs: int,
    t_off_cs: int,
) -> str:
    """Instant colour switch: default → active at t_on, → revert at t_off."""
    return (
        f"{{\\1c{default}"
        f"\\t({t_on_cs},{t_on_cs},\\1c{active})"
        f"\\t({t_off_cs},{t_off_cs},\\1c{revert})"
        f"}}"
    )


def _make_word_highlight_builder(
    style_name: str,
    default_colour: str,
    active_colour: str,
    revert_colour: str | None = None,
    uppercase: bool = False,
) -> DialogueBuilder:
    """Factory: builder that highlights each active word with a colour switch."""
    revert = revert_colour or default_colour

    def build(line: Line) -> str:
        words = line.get("words", [])
        text  = line["text"].upper() if uppercase else line["text"]
        if not words:
            return _dialogue(line["start"], line["end"], style_name, text)

        line_start = line["start"]
        parts = []
        for w in words:
            t_on  = _cs(w["start"] - line_start)
            t_off = _cs(w["end"]   - line_start)
            token = w["text"].upper() if uppercase else w["text"]
            tag   = _colour_switch_tag(default_colour, active_colour, revert, t_on, t_off)
            parts.append(f"{tag}{token}")

        return _dialogue(line["start"], line["end"], style_name, " ".join(parts))

    return build


def _make_plain_builder(style_name: str, uppercase: bool = False) -> DialogueBuilder:
    """Factory: builder that emits plain text with no per-word effects."""
    def build(line: Line) -> str:
        text = line["text"].upper() if uppercase else line["text"]
        return _dialogue(line["start"], line["end"], style_name, text)
    return build


def _build_karaoke(line: Line) -> str:
    """True ASS karaoke using \\kf progressive-fill tag."""
    words = line.get("words", [])
    if not words:
        return _dialogue(line["start"], line["end"], "Karaoke", line["text"])

    parts = [
        rf"{{\kf{max(1, _cs(w['end'] - w['start']))}}}{w['text']}"
        for w in words
    ]
    return _dialogue(line["start"], line["end"], "Karaoke", " ".join(parts))


_DIALOGUE_BUILDERS: dict[StylePreset, DialogueBuilder] = {
    StylePreset.CLASSIC:      _make_plain_builder("Classic",      uppercase=True),
    StylePreset.NEON_GREEN:   _make_plain_builder("NeonGreen",    uppercase=True),
    StylePreset.TYPEWRITER:   _make_plain_builder("Typewriter"),
    StylePreset.MINIMAL:      _make_plain_builder("Minimal"),
    StylePreset.BOLD_LOWER:   _make_plain_builder("BoldLower",    uppercase=True),
    StylePreset.SHADOW_POP:   _make_plain_builder("ShadowPop",    uppercase=True),
    StylePreset.OUTLINE_ONLY: _make_plain_builder("OutlineOnly",  uppercase=True),
    StylePreset.KARAOKE:      _build_karaoke,
    StylePreset.FIRE: _make_word_highlight_builder(
        style_name="Fire",
        default_colour=Colour.WHITE,
        active_colour=Colour.YELLOW,
        revert_colour=Colour.ORANGE,
        uppercase=True,
    ),
    StylePreset.SOCIAL_MEDIA: _make_word_highlight_builder(
        style_name="SocialMedia",
        default_colour=Colour.WHITE,
        active_colour=Colour.YELLOW,
    ),
}


# ══════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════

def write_ass_captions_styled(
    output_path: str | os.PathLike,
    words: Sequence[Word],
    style: StylePreset | str = StylePreset.CLASSIC,
    *,
    play_res_x: int = 384,
    play_res_y: int = 288,
    max_chars: int = 20,
    max_duration: float = 2.0,
    max_words: int = 3,
) -> str:
    """
    Build and write an ASS subtitle file using the given style preset.

    Returns the absolute path of the written file.
    """
    preset    = StylePreset.coerce(style)
    style_def = STYLE_CATALOGUE[preset]
    builder   = _DIALOGUE_BUILDERS[preset]

    lines    = group_words_into_lines(words, max_chars, max_duration, max_words)
    header   = _build_ass_header([style_def], play_res_x, play_res_y)
    dialogue = "\n".join(builder(ln) for ln in lines)

    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(header + dialogue + "\n", encoding="utf-8")
    return str(out)


def list_style_presets() -> list[str]:
    """Return all available style preset names."""
    return [p.value for p in StylePreset]


def get_style_info(style: StylePreset | str) -> dict[str, Any]:
    """Return a serialisable summary of a preset's style definition."""
    preset = StylePreset.coerce(style)
    s = STYLE_CATALOGUE[preset]
    info = asdict(s)
    info["preset"] = preset.value
    info["bold"]   = s.bold == -1
    return info