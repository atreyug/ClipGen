import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from database.connection import get_db
from models import Clip, Video
from services.analyze import chatbot
from services.clipper import create_clips
from services.transcribe import transcribe
from utils.auth_guard import get_current_user
from services.cloudinary import upload_video_to_cloudinary, upload_clip_to_cloudinary

router = APIRouter(
    prefix="/clip",
    tags=["Clipper"],
)


@router.post("/clips")
def gen_clips(
    file: UploadFile = File(...),
    specification: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    video_uuid = str(uuid.uuid4())
    original_filename = file.filename or "video.mp4"

    temp_video_path = None
    video_record = None
    clips = []

    try:
        suffix = Path(original_filename).suffix or ".mp4"

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as temp_file:

            temp_video_path = temp_file.name

            shutil.copyfileobj(
                file.file,
                temp_file,
            )

        upload_result = upload_video_to_cloudinary(
            temp_video_path,
            video_uuid,
        )

        video_url = upload_result["secure_url"]
        cloudinary_public_id = upload_result["public_id"]

        duration = upload_result.get("duration")

        video_record = Video(
            user_id=UUID(current_user["user_id"]),
            filename=original_filename,
            cloudinary_public_id=cloudinary_public_id,
            videolink=video_url,
            duration=duration,
            status="processing",
        )

        db.add(video_record)
        db.commit()
        db.refresh(video_record)

        transcript = transcribe(temp_video_path)

        result = chatbot(
            transcript,
            specification,
        )


        output_dir = tempfile.mkdtemp(
            prefix="clipgen_"
        )

        try:
            output_paths = create_clips(
                temp_video_path,
                result["clips"],
                output_dir=output_dir,
            )

            clips = []

            for index, (path, clip) in enumerate(
                zip(output_paths, result["clips"])
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
                    viral_score=clip.get("viral_score"),
                    cliplink=clip_url,
                    reason=clip.get("reason"),
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


            video_record.status = "processed"

            db.commit()

        finally:
            shutil.rmtree(
                output_dir,
                ignore_errors=True,
            )

    except Exception as exc:

        try:
            if video_record is not None:
                video_record.status = "failed"
                db.commit()
        except Exception:
            db.rollback()

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )

    finally:
        if temp_video_path and os.path.exists(temp_video_path):
            os.remove(temp_video_path)

        awaitable_file = getattr(file, "file", None)

        if awaitable_file:
            awaitable_file.close()

    return {
        "video_id": video_record.id,
        "video_url": video_record.videolink,
        "duration": video_record.duration,
        "status": video_record.status,
        "clips": clips,
    }

