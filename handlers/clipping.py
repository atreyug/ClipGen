import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
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
    file: UploadFile = File(...),
    specification: Optional[str] = None,
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
    request: YouTubeDownloadRequest,
    specification: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    
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
def get_my_clips(db: Session = Depends(get_db),current_user: dict = Depends(get_current_user)):
    videos = db.query(Video).filter(Video.user_id == current_user["user_id"]).all()

    clips=[]
    for video in videos:
        clips.append(db.query(Clip).filter(Clip.video_id == video.id).all())
    
    clips_links = []
    for i in clips:
        for clip in i:
            clips_links.append(clip.cliplink)
    
    return clips_links
        
        
