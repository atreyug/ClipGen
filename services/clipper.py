"""
services/clipper_extended.py
──────────────────────────────
CHANGES vs previous version:
- Face timeline runs ONCE per clip (dense per-frame VIDEO mode).
- Multi-face branch now derives positions from the existing timeline via
  positions_for_face() — removes the redundant second video decode pass.
- needs_face_timeline includes speaker_detection (so MAR is available).
"""

from __future__ import annotations

import copy
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from services.multi_face_tracker import (
    MultiFaceTracker,
    positions_for_face,
    build_multi_crop_expression,
    FrameFaceData,
)
from services.speaker_detector import (
    SpeakerSegment,
    detect_speakers,
)
from services.active_speaker_reframe import (
    build_active_speaker_crop_filter,
)


@dataclass
class ClipOptions:
    caption_style: str = "classic"
    caption: bool = True
    dimensions: str = "9:16"
    multi_face: bool = False
    max_faces: int = 4
    face_sample_interval: float = 0.5
    face_smoothing: float = 0.35
    speaker_detection: bool = False
    hf_token: Optional[str] = None
    use_visual_speaker: bool = True
    active_speaker_reframe: bool = False
    reframe_transition_sec: float = 0.6
    layout: str = "auto"
    preferred_face_id: Optional[int] = None
    crf: int = 23
    preset: str = "veryfast"


def _escape_path_for_filter(path: str | Path) -> str:
    path_str = str(Path(path).resolve()).replace("\\", "/")
    path_str = path_str.replace(":", r"\\:")
    path_str = path_str.replace("'", r"\'")
    path_str = path_str.replace(",", r"\,")
    path_str = path_str.replace("[", r"\[").replace("]", r"\]")
    path_str = path_str.replace(" ", r"\ ")
    return path_str


def _get_video_info(video_path) -> Tuple[int, int, float]:
    try:
        import json
        command = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            str(video_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        data = json.loads(result.stdout)
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                w = int(stream.get("width", 1920))
                h = int(stream.get("height", 1080))
                fps_str = stream.get("r_frame_rate", "25/1")
                num, den = map(int, fps_str.split("/"))
                fps = num / den if den else 25.0
                if fps <= 0:
                    fps = 25.0
                return w, h, fps
    except Exception as exc:
        print(f"[VideoInfo] ffprobe failed ({exc}) — defaults used.")
    return 1920, 1080, 25.0


def _ass_path(output_clip_path) -> Path:
    return Path(output_clip_path).with_suffix(".ass")


def _run_ffmpeg(command: List[str], description: str = "") -> None:
    try:
        subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=600,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"FFmpeg timeout ({description})")
    except subprocess.CalledProcessError as exc:
        print(f"FFmpeg failed ({description}).")
        print("STDERR:", exc.stderr[-3000:] if exc.stderr else "(empty)")
        raise RuntimeError(f"FFmpeg failed: {description}") from exc


def clip_video_advanced(
    input_path,
    output_path,
    clip_start: float,
    clip_end: float,
    all_words: List[Dict[str, Any]],
    options: Optional[ClipOptions] = None,
):
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if options is None:
        options = ClipOptions()

    clip_duration = clip_end - clip_start
    if clip_duration <= 0:
        raise ValueError(f"Invalid clip duration: {clip_duration}")
    if clip_duration > 600:
        raise ValueError(f"Clip too long: {clip_duration:.1f}s (max 600s)")

    frame_width, frame_height, fps = _get_video_info(input_path)

    print(
        f"[Clip Advanced] clip={clip_start:.2f}-{clip_end:.2f} "
        f"layout={options.layout} style={options.caption_style}"
    )

    # ── 1. DENSE per-frame face tracking (timestamps are clip-relative) ──
    face_timeline: List[FrameFaceData] = []
    needs_face_timeline = (
        options.multi_face
        or options.active_speaker_reframe
        or options.speaker_detection
        or options.layout != "single"
    )

    if needs_face_timeline:
        tracker = MultiFaceTracker(
            sample_interval=options.face_sample_interval,
            max_faces=options.max_faces,
            smoothing_factor=options.face_smoothing,
        )
        try:
            face_timeline = tracker.process_video(
                str(input_path),
                clip_start=clip_start,
                clip_end=clip_end,
                target_aspect=options.dimensions,
            )
        except Exception as e:
            print(f"[Face Tracking] failed: {e}")
            face_timeline = []
        finally:
            tracker.close()

        if face_timeline:
            ids = {f.face_id for fd in face_timeline for f in fd.faces}
            print(
                f"[Face Tracking] frames={len(face_timeline)} "
                f"unique_ids={len(ids)} (dense, clip-relative timestamps)"
            )

    # ── 2. Speaker detection (reuses dense MAR — no second video pass) ──
    speaker_segments: List[SpeakerSegment] = []
    if options.speaker_detection or options.active_speaker_reframe:
        try:
            speaker_segments, _ = detect_speakers(
                video_path=str(input_path),
                face_timeline=face_timeline if face_timeline else None,
                all_words=all_words,
                clip_start=clip_start,
                clip_end=clip_end,
                hf_token=options.hf_token,
                use_visual=options.use_visual_speaker,
            )
            if speaker_segments:
                spk = {s.speaker_id for s in speaker_segments}
                with_face = sum(1 for s in speaker_segments if s.face_id is not None)
                print(
                    f"[Speaker Detection] segments={len(speaker_segments)} "
                    f"speakers={len(spk)} mapped_to_face={with_face}"
                )
        except Exception as exc:
            print(f"[Speaker Detection] failed ({exc})")
            speaker_segments = []

    # ── 3. Build video filter ──
    vf_filter = _build_video_filter(
        options=options,
        face_timeline=face_timeline,
        speaker_segments=speaker_segments,
        frame_width=frame_width,
        frame_height=frame_height,
        clip_duration=clip_duration,
        fps=fps,
    )

    # ── 4. Captions ──
    ass_path: Optional[Path] = None
    if options.caption and all_words:
        try:
            from services.caption_styles import (
                slice_words_for_clip, write_ass_captions_styled,
            )
            clip_words = slice_words_for_clip(all_words, clip_start, clip_end)
            if clip_words:
                ass_path = _ass_path(output_path)
                play_res_x = (
                    frame_width if options.dimensions == "16:9"
                    else int(frame_height * 9 / 16)
                )
                write_ass_captions_styled(
                    output_path=ass_path,
                    words=clip_words,
                    style=options.caption_style,
                    play_res_x=play_res_x,
                    play_res_y=frame_height,
                )
        except Exception as e:
            print(f"[Captions] failed: {e}")
            ass_path = None

    # ── 5. Render ──
    try:
        _render_clip(
            input_path=input_path,
            output_path=output_path,
            clip_start=clip_start,
            clip_end=clip_end,
            vf_filter=vf_filter,
            ass_path=ass_path,
            options=options,
        )
    finally:
        if ass_path is not None and ass_path.exists():
            try:
                ass_path.unlink()
            except OSError:
                pass

    return output_path.resolve()


def _build_video_filter(
    options: ClipOptions,
    face_timeline: List[FrameFaceData],
    speaker_segments: List[SpeakerSegment],
    frame_width: int,
    frame_height: int,
    clip_duration: float,
    fps: float,
) -> str:
    # Priority 1: active-speaker reframe (dense timeline → smooth, correct t-domain)
    if options.active_speaker_reframe and face_timeline:
        return build_active_speaker_crop_filter(
            face_timeline=face_timeline,
            speaker_segments=speaker_segments,
            frame_w=frame_width,
            frame_h=frame_height,
            clip_duration=clip_duration,
            target_aspect=options.dimensions,
            transition_sec=options.reframe_transition_sec,
            smooth=True,
            fps=fps,
        )

    # Priority 2: multi-face — positions straight from the timeline (no re-decode)
    if options.multi_face and face_timeline:
        target_w, target_h = _target_dimensions(options.dimensions, frame_width, frame_height)
        active_id = _dominant_face_id(face_timeline, speaker_segments)

        if options.preferred_face_id is not None:
            cand = positions_for_face(face_timeline, options.preferred_face_id)
            if cand:
                active_id = options.preferred_face_id
                positions = cand
            else:
                positions = positions_for_face(face_timeline, active_id)
        else:
            positions = positions_for_face(face_timeline, active_id)

        if positions:
            return build_multi_crop_expression(
                positions, frame_width, target_w, frame_height, target_h
            )
        # fall through to center crop if timeline produced nothing

    # Fallback: center crop
    if options.dimensions == "9:16":
        target_w, target_h = _target_dimensions(options.dimensions, frame_width, frame_height)
        cx = (frame_width - target_w) // 2
        cx -= cx % 2
        return f"crop={target_w}:{target_h}:{cx}:0"

    return ""


def _target_dimensions(dimensions: str, frame_width: int, frame_height: int) -> Tuple[int, int]:
    if dimensions == "9:16":
        target_w = int(frame_height * 9 / 16)
        target_w -= target_w % 2
        target_h = frame_height - (frame_height % 2)
    else:
        target_w = frame_width - (frame_width % 2)
        target_h = int(frame_width * 9 / 16)
        target_h -= target_h % 2
    return target_w, target_h


def _dominant_face_id(
    face_timeline: List[FrameFaceData],
    speaker_segments: List[SpeakerSegment],
) -> int:
    if speaker_segments:
        totals: Dict[int, float] = {}
        for seg in speaker_segments:
            if seg.face_id is not None:
                totals[seg.face_id] = totals.get(seg.face_id, 0.0) + max(0.0, seg.end - seg.start)
        if totals:
            return max(totals, key=lambda fid: totals[fid])

    area_totals: Dict[int, float] = {}
    for fd in face_timeline:
        for face in fd.faces:
            area_totals[face.face_id] = area_totals.get(face.face_id, 0.0) + face.box.area
    if area_totals:
        return max(area_totals, key=lambda fid: area_totals[fid])
    return 0


def _render_clip(
    input_path: Path,
    output_path: Path,
    clip_start: float,
    clip_end: float,
    vf_filter: str,
    ass_path: Optional[Path],
    options: ClipOptions,
) -> None:
    duration = clip_end - clip_start

    base_command = [
        "ffmpeg", "-y",
        "-ss", str(clip_start),
        "-i", str(input_path),
        "-t", str(duration),
    ]

    encode_opts = [
        "-c:v", "libx264",
        "-preset", options.preset,
        "-crf", str(options.crf),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-movflags", "+faststart",
    ]

    ass_filter = None
    if ass_path is not None:
        ass_filter = f"ass=filename={_escape_path_for_filter(ass_path)}"

    filters = [f for f in (vf_filter, ass_filter) if f]
    if filters:
        command = base_command + ["-vf", ",".join(filters)] + encode_opts + [str(output_path)]
    else:
        command = base_command + ["-c:v", "copy", "-c:a", "copy", str(output_path)]

    _run_ffmpeg(command, description=f"clip {clip_start}-{clip_end}")


def create_clips_advanced(
    input_path,
    clips_data: List[Dict[str, Any]],
    all_words: List[Dict[str, Any]],
    output_dir="clips",
    options: Optional[ClipOptions] = None,
) -> List[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_files: List[Path] = []

    for index, clip in enumerate(clips_data, start=1):
        start = float(clip["start"])
        end = float(clip["end"])
        filename = clip.get("filename") or f"clip_{index}.mp4"
        output_path = output_dir / filename

        clip_opts = copy.deepcopy(options) if options else ClipOptions()
        if "style" in clip:
            clip_opts.caption_style = clip["style"]
        if "layout" in clip:
            clip_opts.layout = clip["layout"]

        try:
            rendered = clip_video_advanced(
                input_path=input_path,
                output_path=output_path,
                clip_start=start,
                clip_end=end,
                all_words=all_words,
                options=clip_opts,
            )
            output_files.append(rendered)
        except Exception as exc:
            import traceback
            print(f"[Clip Advanced] failed {filename}: {exc}")
            traceback.print_exc()

    return output_files