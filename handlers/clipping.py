import logging
import os
from typing import Optional
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, HttpUrl
from sqlalchemy.orm import Session, joinedload

from database.connection import get_db
from utils.auth_guard import get_current_user
from utils.clip import (
    process_video_pipeline_advanced,
    get_available_styles,
    get_available_layouts,
    save_upload_to_disk
)
from models import Clip, Video
from services.yt_downloader import (
    YouTubeDownloadError,
    download_youtube_video,
)
from services.cloudinary import delete_video_from_cloudinary, delete_clip_from_cloudinary

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/clip", tags=["Clipping"])


@router.post("/clips", summary="Generate clips from uploaded file (advanced)")
def create_clips_advanced_upload(
    file: UploadFile = File(..., description="Video file to process"),
    dimensions: str = Form("9:16", description="'9:16' or '16:9'"),
    caption: bool = Form(True, description="Burn subtitles"),
    caption_style: str = Form("classic", description="Caption style preset"),
    specification: str = Form("", description="Topic hint for LLM"),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """
    Upload a video file and generate viral clips with advanced options.
    SYNCHRONOUS endpoint - no async/await.
    """
    multi_face= True
    speaker_detection = True
    active_reframe = True
    use_visual = True
    max_faces=2
    preferred_face_id=0
    layout="single"
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    # Check file size before saving
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset
    
    if file_size > 2 * 1024 * 1024 * 1024:  # 2GB
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {file_size / 1024 / 1024:.1f}MB (max 2GB)"
        )

    video_path = None
    try:
        # save_upload_to_disk should be synchronous
        import asyncio
        video_path = save_upload_to_disk(file)
        
        result = process_video_pipeline_advanced(
            video_path=video_path,
            user_id=str(current_user["user_id"]),
            db=db,
            dimensions=dimensions,
            caption=caption,
            caption_style=caption_style,
            specification=specification,
            multi_face=multi_face,
            speaker_detection=speaker_detection,
            active_reframe=active_reframe,
            layout=layout,
            max_faces=max_faces,
            preferred_face_id=preferred_face_id,
            use_visual=use_visual,
            original_filename=file.filename,
        )
        return JSONResponse(status_code=status.HTTP_200_OK, content=result)

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Advanced clip upload error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(exc)}")
    finally:
        if video_path and os.path.exists(video_path):
            try:
                os.remove(video_path)
            except OSError:
                pass


class YTClipAdvancedRequest(BaseModel):
    url: str = Field(..., description="YouTube video URL")
    dimensions: str = Field("9:16", description="'9:16' or '16:9'")
    caption: bool = Field(True, description="Burn subtitles")
    caption_style: str = Field("classic", description="Caption style preset")
    specification: str = Field("", description="Topic hint for LLM")


@router.post("/yt_clipgen", summary="Generate clips from YouTube URL (advanced)")
def create_clips_advanced_youtube(
    body: YTClipAdvancedRequest,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    
    """Download YouTube video and generate clips with advanced options."""
    from services.yt_downloader import download_youtube_video
    from urllib.parse import urlparse

    # Validate URL scheme
    parsed = urlparse(body.url)
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(status_code=400, detail="Invalid URL scheme")

    video_path = None
    try:
        logger.info("Downloading YouTube URL: %s", body.url)
        video_path = download_youtube_video(body.url)

        result = process_video_pipeline_advanced(
            video_path=video_path,
            user_id=str(current_user["user_id"]),
            db=db,
            dimensions=body.dimensions,
            caption=body.caption,
            caption_style=body.caption_style,
            specification=body.specification,
            multi_face=True,
            speaker_detection=True,
            active_reframe=True,
            layout="single",
            max_faces=2,
            preferred_face_id=0,
            use_visual=True,
            original_filename=os.path.basename(video_path),
        )
        return JSONResponse(status_code=status.HTTP_200_OK, content=result)

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Advanced YT clip error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(exc)}")
    finally:
        if video_path and os.path.exists(video_path):
            try:
                os.remove(video_path)
            except OSError:
                pass

@router.get("/getmy_clips") #change krna hai
def get_my_clips(db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)):
    videos = (db.query(Video).filter(Video.user_id == current_user["user_id"]).all())
    videos.sort(key=lambda x: x.id, reverse=True)
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

    return result


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


@router.get("/styles", summary="List available caption style presets")
def list_caption_styles():
    """Returns all available caption styles with metadata."""
    return get_available_styles()


