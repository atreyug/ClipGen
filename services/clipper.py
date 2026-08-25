import subprocess
from pathlib import Path

import cv2

from services.facetracker import (
    FaceTracker,
    get_face_crop_positions,
    smooth_positions,
)
from services.captions import build_captions_for_clip


def build_crop_expression(positions, default_x):
    if not positions:
        return str(int(default_x))

    if len(positions) == 1:
        return str(int(round(positions[0][1])))

    expression = str(int(round(positions[-1][1])))

    for i in range(len(positions) - 1, 0, -1):
        t0, x0 = positions[i - 1]
        t1, x1 = positions[i]

        if t1 <= t0:
            continue

        x0 = int(round(x0))
        x1 = int(round(x1))

        interpolation = (
            f"{x0}+"
            f"({x1}-{x0})*"
            f"(t-{t0:.3f})/"
            f"({t1:.3f}-{t0:.3f})"
        )

        expression = (
            f"if("
            f"between(t,{t0:.3f},{t1:.3f}),"
            f"{interpolation},"
            f"{expression}"
            f")"
        )

    first_t, first_x = positions[0]
    expression = (
        f"if(lt(t,{first_t:.3f}),{int(round(first_x))},{expression})"
    )

    return expression


def _escape_for_filtergraph(expression: str) -> str:
    return expression.replace(",", r"\,")


def _escape_path_for_filter(path: str | Path) -> str:
    path_str = str(Path(path).resolve()).replace("\\", "/")
    path_str = path_str.replace(":", r"\\:")
    path_str = path_str.replace("'", r"\'")
    path_str = path_str.replace(",", r"\,")
    path_str = path_str.replace("[", r"\[").replace("]", r"\]")
    path_str = path_str.replace(" ", r"\ ")

    return path_str


def clip_video_with_crop(
    input_path,
    start,
    end,
    caption,
    output_path,
    tracker: FaceTracker,
    all_words=None,
    captions_dir="clips/captions",
):
    duration = end - start

    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    crop_width = int(frame_height * 9 / 16)
    crop_width -= crop_width % 2
    crop_height = frame_height - (frame_height % 2)

    if crop_width >= frame_width:
        raise RuntimeError("Video is too narrow for a 9:16 crop.")

    default_x = (frame_width - crop_width) // 2
    default_x -= default_x % 2

    positions = get_face_crop_positions(
        video_path=str(input_path),
        start=start,
        end=end,
        tracker=tracker,
        sample_interval=0.5,
    )

    positions = smooth_positions(positions, smoothing_factor=0.25)

    crop_x_expr = build_crop_expression(
        positions=positions,
        default_x=default_x,
    )

    print(
        f"[Face Tracking] clip={start:.2f}-{end:.2f} "
        f"positions={len(positions)}"
    )

    crop_x_expr_escaped = _escape_for_filtergraph(crop_x_expr)
    filters = [f"crop={crop_width}:{crop_height}:{crop_x_expr_escaped}:0"]

    if caption:
        if all_words:
            captions_dir = Path(captions_dir)
            captions_dir.mkdir(parents=True, exist_ok=True)

            ass_path = build_captions_for_clip(
                all_words=all_words,
                clip_start=start,
                clip_end=end,
                output_path=captions_dir / f"{Path(output_path).stem}.ass",
                video_width=crop_width,
                video_height=crop_height,
            )

            if ass_path is not None:
                escaped_ass = _escape_path_for_filter(ass_path)
                filters.append(f"ass=filename={escaped_ass}")
            else:
                print(
                    f"[Captions] no words found for clip "
                    f"{start:.2f}-{end:.2f}, skipping captions"
                )

    vf = ",".join(filters)

    command = [
        "ffmpeg",
        "-y",
        "-ss", str(start),
        "-i", str(input_path),
        "-t", str(duration),
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-movflags", "+faststart",
        str(output_path),
    ]

    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        print("FFmpeg failed.")
        print("COMMAND:")
        print(" ".join(command))
        print("STDOUT:")
        print(exc.stdout)
        print("STDERR:")
        print(exc.stderr)

        raise RuntimeError(
            f"FFmpeg failed while creating clip {start}-{end}"
        ) from exc


def create_clips_9_16(input_path, clips, caption, output_dir="clips", all_words=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_files = []
    tracker = FaceTracker()

    try:
        for index, clip in enumerate(clips, start=1):
            output_path = output_dir / f"clip_{index}.mp4"

            clip_video_with_crop(
                input_path=input_path,
                start=clip["start"],
                end=clip["end"],
                caption = caption,
                output_path=output_path,
                tracker=tracker,
                all_words=all_words,
                captions_dir=output_dir / "captions",
            )

            output_files.append(output_path)

    finally:
        tracker.close()

    return output_files





def clip_video_without_crop(
    input_path,
    start,
    end,
    output_path,
    caption: bool,
    all_words=None,
    captions_dir="clips/captions",
):
    duration = end - start
    filters = []

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if caption and all_words:
        captions_dir = Path(captions_dir)
        captions_dir.mkdir(parents=True, exist_ok=True)

        ass_path = build_captions_for_clip(
            all_words=all_words,
            clip_start=start,
            clip_end=end,
            output_path=captions_dir / f"{Path(output_path).stem}.ass",
            video_width=frame_width,
            video_height=frame_height,
        )

        if ass_path is not None:
            escaped_ass = _escape_path_for_filter(ass_path)
            filters.append(f"ass=filename={escaped_ass}")
        else:
            print(f"[Captions] No words found for clip {start:.2f}-{end:.2f}, skipping.")

    command = [
        "ffmpeg",
        "-y",
        "-ss", str(start),
        "-i", str(input_path),
        "-t", str(duration),
    ]

    if filters:
        command.extend([
            "-vf", ",".join(filters),
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-movflags", "+faststart",
        ])
    else:
        command.extend([
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-movflags", "+faststart",
        ])

    command.append(str(output_path))

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        print("FFmpeg 16:9 failed.")
        print("STDERR:", exc.stderr)
        raise RuntimeError(f"FFmpeg failed while creating clip {start}-{end}") from exc


def create_clips_16_9(
    input_path,
    clips,
    caption: bool,
    all_words=None,
    output_dir="clips",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_files = []

    for index, clip in enumerate(clips, start=1):
        output_path = output_dir / f"clip_{index}.mp4"

        clip_video_without_crop(
            input_path=input_path,
            start=clip["start"],
            end=clip["end"],
            output_path=output_path,
            caption=caption,
            all_words=all_words,
            captions_dir=output_dir / "captions",
        )

        output_files.append(output_path)

    return output_files




