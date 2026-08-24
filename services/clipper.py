import subprocess
from pathlib import Path


def clip_video(
    input_path,
    start,
    end,
    output_path
):
    duration = end - start

    command = [
        "ffmpeg",
        "-y",
        "-ss", str(start),
        "-i", str(input_path),
        "-t", str(duration),
        "-c", "copy",
        str(output_path)
    ]

    subprocess.run(
        command,
        check=True
    )


def create_clips(
    input_path,
    clips,
    output_dir="clips"
):
    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_files = []

    for index, clip in enumerate(clips, start=1):
        output_path = output_dir / f"clip_{index}.mp4"

        clip_video(
            input_path=input_path,
            start=clip["start"],
            end=clip["end"],
            output_path=output_path
        )

        output_files.append(output_path)

    return output_files


# import shutil
# import subprocess
# from pathlib import Path


# # ---------------------------------------------------------------------------
# # Text escaping
# # ---------------------------------------------------------------------------

# def escape_ass_text(text: str) -> str:
#     """
#     Escape text for ASS subtitle format.

#     Order matters: backslashes must be escaped first, then braces.
#     Newlines are converted to ASS's forced line break (\\N).
#     """

#     text = text.replace("\\", r"\\")
#     text = text.replace("{", r"\{")
#     text = text.replace("}", r"\}")
#     text = text.replace("\r\n", "\\N")
#     text = text.replace("\n", "\\N")

#     return text


# def format_ass_timestamp(seconds: float) -> str:
#     """
#     Convert seconds into ASS timestamp format.

#     Example:
#         65.25 -> 0:01:05.25
#     """

#     total_centiseconds = int(round(seconds * 100))

#     hours = total_centiseconds // 360000
#     total_centiseconds %= 360000

#     minutes = total_centiseconds // 6000
#     total_centiseconds %= 6000

#     seconds_value = total_centiseconds // 100
#     centiseconds = total_centiseconds % 100

#     return (
#         f"{hours}:"
#         f"{minutes:02d}:"
#         f"{seconds_value:02d}."
#         f"{centiseconds:02d}"
#     )


# # ---------------------------------------------------------------------------
# # Caption chunking
# # ---------------------------------------------------------------------------

# def chunk_caption_text(text: str, max_words_per_chunk: int = 4) -> list[str]:
#     """
#     Split caption text into short chunks so captions read like real
#     short-form subtitles (a few words on screen at a time) instead of
#     one static wall of text sitting there for the whole clip.

#     If the text already contains manual line breaks, each line is
#     treated as its own chunk group and split independently, so you
#     can still force your own breaks by passing text with '\\n' in it.
#     """

#     chunks: list[str] = []

#     for line in text.splitlines() or [text]:
#         words = line.split()

#         if not words:
#             continue

#         for i in range(0, len(words), max_words_per_chunk):
#             chunk = " ".join(words[i : i + max_words_per_chunk])
#             chunks.append(chunk)

#     return chunks or [text]


# def font_size_for_chunk(chunk: str, base_size: int = 64, min_size: int = 36) -> int:
#     """
#     Scale font size down a bit for longer chunks so text doesn't
#     overflow the safe caption area at 1920x1080.
#     """

#     length = len(chunk)

#     if length <= 20:
#         return base_size
#     if length <= 35:
#         return int(base_size * 0.85)
#     if length <= 50:
#         return int(base_size * 0.7)

#     return min_size


# # ---------------------------------------------------------------------------
# # ASS file generation
# # ---------------------------------------------------------------------------

# def create_ass_file(
#     text: str,
#     duration: float,
#     output_path: str,
#     max_words_per_chunk: int = 4,
# ) -> None:
#     """
#     Create an ASS subtitle file with the caption text split into
#     short, sequentially-timed chunks spanning the clip duration,
#     instead of one block of text sitting static the whole time.
#     """

#     if duration <= 0:
#         raise ValueError(
#             f"Caption duration must be positive, got {duration}"
#         )

#     chunks = chunk_caption_text(text, max_words_per_chunk=max_words_per_chunk)
#     chunk_duration = duration / len(chunks)

#     header = """[Script Info]
# ScriptType: v4.00+
# PlayResX: 1920
# PlayResY: 1080
# WrapStyle: 0
# ScaledBorderAndShadow: yes

# [V4+ Styles]
# Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
# Style: Default,Arial,64,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,1,0,0,0,100,100,0.3,0,1,3,1,2,100,100,80,1

# [Events]
# Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
# """

#     events = []

#     for index, chunk in enumerate(chunks):
#         start_time = index * chunk_duration
#         end_time = (index + 1) * chunk_duration

#         # Nudge the very last chunk's end to exactly match the clip
#         # duration, avoiding rounding gaps/overlaps.
#         if index == len(chunks) - 1:
#             end_time = duration

#         size = font_size_for_chunk(chunk)
#         escaped_chunk = escape_ass_text(chunk)

#         # Per-line font-size override via inline override tag, so
#         # short chunks stay big and readable while long ones shrink
#         # to fit instead of overflowing off screen.
#         styled_text = f"{{\\fs{size}}}{escaped_chunk}"

#         events.append(
#             f"Dialogue: 0,{format_ass_timestamp(start_time)},"
#             f"{format_ass_timestamp(end_time)},Default,,0,0,80,,{styled_text}"
#         )

#     ass_content = header + "\n".join(events) + "\n"

#     Path(output_path).write_text(ass_content, encoding="utf-8")


# # ---------------------------------------------------------------------------
# # ffmpeg helpers
# # ---------------------------------------------------------------------------

# def require_str_or_path(value, name: str):
#     """
#     Guard against list/dict/etc. being passed where a single path or
#     string is expected. Without this, pathlib raises a generic
#     'argument should be a str or os.PathLike, not list' error that
#     doesn't say which argument or which clip caused it.
#     """

#     if isinstance(value, (list, tuple, dict, set)):
#         raise TypeError(
#             f"'{name}' must be a single str/Path, but got a "
#             f"{type(value).__name__}: {value!r}. Check where this "
#             "value is coming from (e.g. a JSON field that ended up "
#             "as an array, or a glob() result passed in directly)."
#         )


# def ensure_ffmpeg_available() -> None:
#     if shutil.which("ffmpeg") is None:
#         raise RuntimeError(
#             "ffmpeg was not found on PATH. Install it and make sure "
#             "it's accessible before running caption generation."
#         )


# def build_subtitle_filter_path(subtitle_path) -> str:
#     """
#     Format a subtitle path safely for ffmpeg's subtitles/ass filter,
#     which needs forward slashes and an escaped drive-letter colon on
#     Windows.
#     """

#     subtitle_filter_path = str(Path(subtitle_path).resolve()).replace("\\", "/")

#     if len(subtitle_filter_path) > 1 and subtitle_filter_path[1] == ":":
#         subtitle_filter_path = (
#             subtitle_filter_path[0] + r"\:" + subtitle_filter_path[2:]
#         )

#     return subtitle_filter_path


# def clip_video(
#     input_path,
#     start,
#     end,
#     text,
#     output_path,
#     subtitle_path,
#     max_words_per_chunk: int = 4,
# ) -> None:
#     """
#     Trim video and burn chunked, timed captions in a single ffmpeg
#     operation.
#     """

#     require_str_or_path(input_path, "input_path")
#     require_str_or_path(output_path, "output_path")
#     require_str_or_path(subtitle_path, "subtitle_path")
#     require_str_or_path(text, "text")

#     if end <= start:
#         raise ValueError(
#             f"Clip end ({end}) must be after start ({start})"
#         )

#     if not Path(input_path).exists():
#         raise FileNotFoundError(f"Input video not found: {input_path}")

#     duration = end - start

#     create_ass_file(
#         text=text,
#         duration=duration,
#         output_path=subtitle_path,
#         max_words_per_chunk=max_words_per_chunk,
#     )

#     subtitle_filter_path = build_subtitle_filter_path(subtitle_path)
#     subtitle_filter = f"ass='{subtitle_filter_path}'"

#     command = [
#         "ffmpeg",
#         "-y",
#         # Seek to clip start (fast, input-side seek).
#         "-ss", str(start),
#         # Original video.
#         "-i", str(input_path),
#         # Clip duration.
#         "-t", str(duration),
#         # Burn captions.
#         "-vf", subtitle_filter,
#         # Re-encode video because captions need to be rendered into
#         # the frames.
#         "-c:v", "libx264",
#         "-preset", "fast",
#         "-crf", "23",
#         # Preserve audio.
#         "-c:a", "aac",
#         "-shortest",
#         str(output_path),
#     ]

#     result = subprocess.run(
#         command,
#         capture_output=True,
#         text=True,
#     )

#     if result.returncode != 0:
#         raise RuntimeError(
#             "ffmpeg failed while generating "
#             f"{output_path} (clip {start}-{end}s).\n"
#             f"Command: {' '.join(command)}\n"
#             f"--- ffmpeg stderr (last 2000 chars) ---\n"
#             f"{result.stderr[-2000:]}"
#         )


# def create_clips(
#     input_path,
#     clips,
#     output_dir="clips",
#     max_words_per_chunk: int = 7,
#     keep_subtitles: bool = False,
# ):
#     """
#     Generate one captioned clip per entry in `clips`.

#     Each clip dict needs: "start", "end", "text".
#     Returns a list of output paths (same shape as the original
#     function) for successfully generated clips only. Any clip that
#     fails does not stop the rest of the batch; failures are logged
#     to stderr with the clip index and reason instead of raising, so
#     one bad clip doesn't take down the whole batch or change the
#     return type callers depend on.
#     """

#     ensure_ffmpeg_available()
#     require_str_or_path(input_path, "input_path")
#     require_str_or_path(output_dir, "output_dir")

#     output_dir = Path(output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)

#     output_files = []
#     errors = []

#     for index, clip in enumerate(clips, start=1):
#         output_path = output_dir / f"clip_{index}.mp4"
#         subtitle_path = output_dir / f"clip_{index}.ass"

#         try:
#             require_str_or_path(clip.get("text"), f"clips[{index-1}]['text']")
#             clip_video(
#                 input_path=input_path,
#                 start=float(clip["start"]),
#                 end=float(clip["end"]),
#                 text=clip["text"],
#                 output_path=output_path,
#                 subtitle_path=subtitle_path,
#                 max_words_per_chunk=max_words_per_chunk,
#             )
#             output_files.append(output_path)
#         except Exception as exc:  # noqa: BLE001 - we want to keep going
#             errors.append({"clip_index": index, "clip": clip, "error": str(exc)})
#         finally:
#             if not keep_subtitles:
#                 subtitle_path.unlink(missing_ok=True)

#     if errors:
#         import sys

#         print(
#             f"\n{len(errors)} of {len(clips)} clip(s) failed:",
#             file=sys.stderr,
#         )
#         for err in errors:
#             print(
#                 f"  - clip_{err['clip_index']}: {err['error'].splitlines()[0]}",
#                 file=sys.stderr,
#             )

#     return output_files



