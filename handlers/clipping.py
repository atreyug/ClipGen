import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, Form, status
from pydantic import BaseModel, HttpUrl
from sqlalchemy.orm import Session, joinedload

from database.connection import get_db
from models import Clip, Video
from services.yt_downloader import (
    YouTubeDownloadError,
    download_youtube_video,
)
from utils.auth_guard import get_current_user
from utils.clip import process_video_pipeline, save_upload_to_disk
from services.cloudinary import delete_video_from_cloudinary, delete_clip_from_cloudinary



class YouTubeClipGenRequest(BaseModel):
    url: HttpUrl
    dimensions: str = "9:16"
    caption: bool = False
    specification: Optional[str] = None


router = APIRouter(
    prefix="/clip",
    tags=["Clipper"],
)


@router.post("/clips")
def gen_clips_from_upload(
    dimensions: str = Form("9:16"),
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
        if temp_video_path and os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
            except OSError:
                pass
        file.file.close()


@router.post("/yt_clipgen")
def gen_clips_from_youtube(
    request: YouTubeClipGenRequest,  
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
            specification=request.specification,
            db=db,
            current_user=current_user,
            caption = request.caption,
            dimensions = request.dimensions
        )

    finally:
        if temp_video_path and os.path.exists(temp_video_path):
            os.remove(temp_video_path)


# @router.get("/videos")
# def get_my_videos(
#     db: Session = Depends(get_db),
#     current_user: dict = Depends(get_current_user),
# ):
#     videos = (
#         db.query(Video)
#         .filter(Video.user_id == current_user["user_id"])
#         .order_by(Video.id.desc())
#         .all()
#     )

#     return [
#         {
#             "id": video.id,
#             "video_link": video.videolink,
#             "created_at": getattr(video, "created_at", None),
#         }
#         for video in videos
#     ]

@router.get("/getmy_clips") #change krna hai
def get_my_clips(db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)):
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


@router.delete("/videos/{video_id}")
def delete_video_and_clips(
    video_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    video = (
        db.query(Video)
        .filter(
            Video.id == video_id,
            Video.user_id == current_user["user_id"],
        )
        .first()
    )

    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found",
        )

    clips = db.query(Clip).filter(Clip.video_id == video.id).all()

    for clip in clips:
        clip_public_id = getattr(clip, "cloudinary_public_id", None)
        if not clip_public_id and clip.cliplink:
            clip_public_id = clip.cliplink.split("/")[-1].split(".")[0]

        if clip_public_id:
            delete_clip_from_cloudinary(clip_public_id)

        if clip.cliplink:
            clip_path = Path(clip.cliplink)
            if clip_path.exists() and clip_path.is_file():
                try:
                    clip_path.unlink()
                except OSError:
                    pass

        db.delete(clip)

    video_public_id = getattr(video, "cloudinary_public_id", None)
    if not video_public_id and video.videolink:
        video_public_id = video.videolink.split("/")[-1].split(".")[0]

    if video_public_id:
        delete_video_from_cloudinary(video_public_id)

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
        "message": "Video and all its clips were deleted from Cloudinary and database successfully.",
        "video_id": video_id,
    }