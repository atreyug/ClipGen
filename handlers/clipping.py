import uuid
import shutil
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

router = APIRouter(
    prefix="/clip",
    tags=["Clipper"]
)


@router.post("/clips")
def gen_clips(
    file: UploadFile = File(...),
    specification: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    video_uuid = str(uuid.uuid4())
    original_filename = file.filename
    video_path = Path("videos") / video_uuid

    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    video_record = Video(
        id=video_uuid,
        user_id=UUID(current_user["user_id"]),
        filename=video_uuid,
        status="processing",
    )
    db.add(video_record)
    db.commit()
    db.refresh(video_record)

    try:
        transcript = transcribe(str(video_path))
        result = chatbot(transcript, specification)

        output_paths = create_clips(
            str(video_path),
            result["clips"],
            output_dir="clips",
        )

        clips = []

        for path, clip in zip(output_paths, result["clips"]):
            clip_filename = Path(path).name

            clip_record = Clip(
                video_id=video_record.id,
                filename=clip_filename,
                start_time=clip["start"],
                end_time=clip["end"],
                viral_score=clip.get("viral_score"),
                reason=clip.get("reason"),
            )
            db.add(clip_record)

            clips.append({
                "url": f"/clips/{clip_filename}",
                "start": clip["start"],
                "end": clip["end"],
                "viral_score": clip.get("viral_score"),
                "reason": clip.get("reason"),
            })

        video_record.status = "processed"
        db.commit()

    except Exception as exc:
        video_record.status = "failed"
        db.commit()
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "video_id": video_record.id,
        "clips": clips,
    }
