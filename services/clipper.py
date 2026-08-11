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