import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, Form
from pydantic import BaseModel, HttpUrl
from sqlalchemy.orm import Session

from models import Clip, Video
from database.connection import get_db
from services.yt_downloader import (
    YouTubeDownloadError,
    download_youtube_video,
)
from utils.auth_guard import get_current_user
from utils.clip import process_video_pipeline, save_upload_to_disk


class YouTubeDownloadRequest(BaseModel):
    url: HttpUrl




router = APIRouter(
    prefix="/clip",
    tags=["Clipper"],
)


@router.post("/clips")
def gen_clips(
    dimensions: str = Form(...),
    caption: bool = Form(False),
    file: UploadFile = File(...),
    specification: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    original_filename = file.filename or "video.mp4"

    temp_video_path = None

    try:
        temp_video_path = save_upload_to_disk(file)

        return process_video_pipeline(
            temp_video_path=temp_video_path,
            original_filename=original_filename,
            specification=specification,
            db=db,
            current_user=current_user,
            caption=caption,
            dimensions=dimensions,
        )

    finally:
        if (
            temp_video_path
            and os.path.exists(temp_video_path)
        ):
            os.remove(temp_video_path)

        file.file.close()


@router.post("/yt_clipgen")
def yt_clips(
    dimensions: str = Form(...),
    caption: bool = Form(False),
    request: YouTubeDownloadRequest = ...,
    specification: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    # return "success"

    
    temp_video_path = None

    try:
        video_path = download_youtube_video(
            str(request.url)
        )
    except YouTubeDownloadError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    temp_video_path = str(video_path)
    original_filename = video_path.name

    try:
        return process_video_pipeline(
            temp_video_path=temp_video_path,
            original_filename=original_filename,
            specification=specification,
            db=db,
            current_user=current_user,
            caption = caption,
            dimensions = dimensions
        )

    finally:
        if temp_video_path and os.path.exists(temp_video_path):
            os.remove(temp_video_path)

@router.get("/getmy_vid")
def get_my_vid(db: Session = Depends(get_db),current_user: dict = Depends(get_current_user)):
    videos = db.query(Video).filter(Video.user_id == current_user["user_id"]).all()

    video_links=[]
    for video in videos:
        video_links.append(video.videolink)

    return video_links

@router.get("/getmy_clips")
def get_my_clips(
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    videos = (db.query(Video).filter(Video.user_id == current_user["user_id"]).all())

    result = []

    for video in videos:
        clips = (db.query(Clip).filter(Clip.video_id == video.id).all())

        result.append({
            "video_id": video.id,
            "video_link": video.videolink,
            "clips": [
                {
                    "clip_id": clip.id,
                    "clip_link": clip.cliplink
                }
                for clip in clips
            ]
        })

    return result[::-1]


@router.delete("/delvideo/{video_id}")
def delete_video(
    video_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    video = (
        db.query(Video)
        .filter(
            Video.id == video_id,
            Video.user_id == current_user["user_id"]
        )
        .first()
    )

    if not video:
        raise HTTPException(
            status_code=404,
            detail="Video not found",
        )

    clips = (
        db.query(Clip)
        .filter(Clip.video_id == video.id)
        .all()
    )

    for clip in clips:
        if clip.cliplink:
            clip_path = Path(clip.cliplink)

            if clip_path.exists() and clip_path.is_file():
                try:
                    clip_path.unlink()
                except OSError:
                    pass

        db.delete(clip)

    if video.videolink:
        video_path = Path(video.videolink)

        if video_path.exists() and video_path.is_file():
            try:
                video_path.unlink()
            except OSError:
                pass

    db.delete(video)
    db.commit()

    return {
        "success": True,
        "message": "Video and its clips deleted successfully",
        "video_id": video_id,
    }