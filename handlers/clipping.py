import uuid
import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, UploadFile

from services.analyze import chatbot
from services.transcribe import transcribe
from services.clipper import create_clips

router = APIRouter(
    prefix="/clip",
    tags=["Clipper"]
)


@router.post("/clips")
def gen_clips(file: UploadFile = File(...), specification: Optional[str] = None):

    video_id = str(uuid.uuid4())

    video_path = Path("videos") / f"{video_id}_{file.filename}"

    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    transcript = transcribe(str(video_path))

    result = chatbot(transcript, specification)

    output_paths = create_clips(
        str(video_path),
        result["clips"],
        output_dir="clips"
    )

    clips = []

    for path, clip in zip(output_paths, result["clips"]):
        clips.append({
            "url": f"/clips/{Path(path).name}",
            "start": clip["start"],
            "end": clip["end"],
            "viral_score": clip["viral_score"],
            "reason": clip["reason"]
        })

    return {
        "clips": clips
    }
