
from pathlib import Path


def slice_words_for_clip(all_words, clip_start, clip_end):

    clip_words = []

    for word in all_words:
        if word["end"] <= clip_start or word["start"] >= clip_end:
            continue

        rel_start = max(0.0, word["start"] - clip_start)
        rel_end = max(rel_start, min(word["end"], clip_end) - clip_start)

        clip_words.append(
            {
                "text": word["text"],
                "start": round(rel_start, 3),
                "end": round(rel_end, 3),
            }
        )

    return clip_words


def group_words_into_lines(
    words,
    max_chars=20,
    max_line_duration=2.0,
    max_words=3,
):

    if not words:
        return []

    lines = []
    current = []

    def flush():
        if not current:
            return
        lines.append(
            {
                "text": " ".join(w["text"] for w in current),
                "start": current[0]["start"],
                "end": current[-1]["end"],
            }
        )

    for word in words:
        if not current:
            current.append(word)
            continue

        candidate_text = " ".join(w["text"] for w in current) + " " + word["text"]
        candidate_duration = word["end"] - current[0]["start"]

        if (
            len(candidate_text) > max_chars
            or candidate_duration > max_line_duration
            or len(current) >= max_words
        ):
            flush()
            current = [word]
        else:
            current.append(word)

    flush()

    return lines


def _format_ass_timestamp(seconds: float) -> str:
    seconds = max(0.0, seconds)

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    return f"{hours}:{minutes:02d}:{secs:05.2f}"


def write_ass_captions(
    lines,
    output_path,
    video_width,
    video_height,
    font_size=None,
):

    output_path = Path(output_path)

    if font_size is None:
        font_size = max(22, int(video_height * 0.036))

    margin_v = int(video_height * 0.15)
    margin_lr = int(video_width * 0.08)

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {video_width}
PlayResY: {video_height}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,Arial,{font_size},&H00FFFFFF,&H00000000,&H80000000,1,0,1,2,1,2,{margin_lr},{margin_lr},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    events = []

    for line in lines:
        start = _format_ass_timestamp(line["start"])
        end = _format_ass_timestamp(line["end"])
        text = line["text"].replace("\n", " ").strip()

        if not text:
            continue

        events.append(
            f"Dialogue: 0,{start},{end},Caption,,0,0,0,,{text}"
        )

    output_path.write_text(header + "\n".join(events), encoding="utf-8")

    return output_path


def build_captions_for_clip(
    all_words,
    clip_start,
    clip_end,
    output_path,
    video_width,
    video_height,
    font_size=None,
):
    clip_words = slice_words_for_clip(all_words, clip_start, clip_end)

    if not clip_words:
        return None

    lines = group_words_into_lines(clip_words)

    if not lines:
        return None

    return write_ass_captions(
        lines=lines,
        output_path=output_path,
        video_width=video_width,
        video_height=video_height,
        font_size=font_size,
    )