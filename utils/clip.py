import os
import shutil
import tempfile
import uuid

from pathlib import Path
from typing import Optional
from uuid import UUID

from fastapi import HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from models import Clip, Video
from services.analyze import chatbot
from services.clipper import create_clips_9_16, create_clips_16_9
from services.cloudinary import (
    upload_clip_to_cloudinary,
    upload_video_to_cloudinary,
)
from services.transcribe import transcribe


def save_upload_to_disk(
    file: UploadFile,
) -> str:
    original_filename = (
        file.filename or "video.mp4"
    )

    suffix = (
        Path(original_filename).suffix
        or ".mp4"
    )

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    ) as temp_file:

        shutil.copyfileobj(
            file.file,
            temp_file,
            length=1024 * 1024,  
        )

        return temp_file.name


def create_video_record(
    temp_video_path: str,
    original_filename: str,
    db: Session,
    current_user: dict,
) -> Video:
    video_uuid = str(uuid.uuid4())

    upload_result = upload_video_to_cloudinary(
        temp_video_path,
        video_uuid,
    )

    video_record = Video(
        user_id=UUID(current_user["user_id"]),
        filename=original_filename,
        cloudinary_public_id=upload_result["public_id"],
        videolink=upload_result["secure_url"],
        duration=upload_result.get("duration"),
        status="processing",
    )

    db.add(video_record)
    db.commit()
    db.refresh(video_record)

    return video_record


def generate_clip_files(
    temp_video_path: str,
    specification: Optional[str],
    dimensions: str,
    caption: bool,
):
    dim_clean = str(dimensions).strip().lower()

    if dim_clean not in ("9:16", "9_16", "16:9", "16_9"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid dimension '{dimensions}'. Must be '9:16' or '16:9'.",
        )

    transcript, all_words = transcribe(temp_video_path)
    print("transcription done")

    result = chatbot(
        transcript,
        specification,
    )
    print("analysis done")

    output_dir = tempfile.mkdtemp(prefix="clipgen_")

    try:
        if dim_clean in ("9:16", "9_16"):
            output_paths = create_clips_9_16(
                temp_video_path,
                result.get("clips", []),
                caption=caption,
                output_dir=output_dir,
                all_words=all_words,
            )

        else:
            output_paths = create_clips_16_9(
                temp_video_path,
                result.get("clips", []),
                caption=caption,
                all_words=all_words,
                output_dir=output_dir,
            )
        
        print("clips created")

        return (
            output_paths,
            result.get("clips", []),
            output_dir,
        )

    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise


def save_clips(
    video_record: Video,
    output_paths: list,
    clip_data: list,
    db: Session,
) -> list:
    clips = []

    for path, clip in zip(
        output_paths,
        clip_data,
    ):
        clip_filename = Path(path).name
        clip_uuid = str(uuid.uuid4())

        clip_upload = upload_clip_to_cloudinary(
            path,
            clip_uuid,
        )

        clip_url = clip_upload["secure_url"]

        clip_record = Clip(
            video_id=video_record.id,
            filename=clip_filename,
            start_time=clip["start"],
            end_time=clip["end"],
            viral_score=clip["viral_score"],
            cliplink=clip_url,
            reason=clip["reason"],
        )

        db.add(clip_record)

        clips.append({
            "url": clip_url,
            "filename": clip_filename,
            "start": clip["start"],
            "end": clip["end"],
            "viral_score": clip.get("viral_score"),
            "reason": clip.get("reason"),
        })

    return clips


def mark_video_failed(
    video_record: Optional[Video],
    db: Session,
) -> None:
    if video_record is None:
        return

    try:
        video_record.status = "failed"
        db.commit()
    except Exception:
        db.rollback()


def process_video_pipeline(
    temp_video_path: str,
    original_filename: str,
    specification: Optional[str],
    db: Session,
    dimensions: str,
    caption: bool,
    current_user: dict,
) -> dict:

    video_record = None
    output_dir = None

    try:
        video_record = create_video_record(
            temp_video_path=temp_video_path,
            original_filename=original_filename,
            db=db,
            current_user=current_user,
        )
        print("video saved")

        (
            output_paths,
            clip_data,
            output_dir,
        ) = generate_clip_files(
            temp_video_path=temp_video_path,
            specification=specification,
            dimensions= dimensions,
            caption = caption,
        )

        print("clips generated")


        clips = save_clips(
            video_record=video_record,
            output_paths=output_paths,
            clip_data=clip_data,
            db=db,
        )
        print("clips saved")

        video_record.status = "processed"

        db.commit()

        return {
            "video_id": video_record.id,
            "video_url": video_record.videolink,
            "duration": video_record.duration,
            "status": video_record.status,
            "clips": clips,
        }

    except Exception as exc:
        mark_video_failed(
            video_record=video_record,
            db=db,
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )

    finally:
        if output_dir:
            shutil.rmtree(
                output_dir,
                ignore_errors=True,
            )