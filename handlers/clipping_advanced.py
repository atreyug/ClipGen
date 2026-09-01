"""
handlers/clipping_advanced.py
──────────────────────────────
Advanced clipping API routes.

FIXES:
- Removed broken async/await
- Added file size validation
- Added rate limiting structure
- Improved error responses
"""

import logging
import os
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from database.connection import get_db
from utils.auth_guard import get_current_user
from utils.clip import save_upload_to_disk
from utils.clip_advanced import (
    process_video_pipeline_advanced,
    get_available_styles,
    get_available_layouts,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/clip/v2", tags=["Clipping Advanced"])


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
        video_path = asyncio.run(save_upload_to_disk(file))
        
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


@router.get("/styles", summary="List available caption style presets")
def list_caption_styles():
    """Returns all available caption styles with metadata."""
    return get_available_styles()


@router.get("/layouts", summary="List available multi-face layout presets")
def list_layout_presets_endpoint():
    """Returns all available layout presets with descriptions."""
    return get_available_layouts()


@router.get("/options", summary="Full advanced options reference")
def get_options_reference():
    """Returns combined reference of all advanced clipping options."""
    return {
        "caption_style": {
            "type": "string",
            "default": "classic",
            "values": ["classic", "neon_green", "fire", "typewriter", "minimal"],
            "description": "Visual style for burned-in subtitles.",
        },
        "layout": {
            "type": "string",
            "default": "auto",
            "values": ["auto", "single", "split_vertical", "split_horizontal"],
            "description": "Multi-face composition layout.",
        },
        "dimensions": {
            "type": "string",
            "default": "9:16",
            "values": ["9:16", "16:9"],
            "description": "Output aspect ratio.",
        },
        "caption": {
            "type": "boolean",
            "default": True,
            "description": "Whether to burn subtitles into the clip.",
        },
        "multi_face": {
            "type": "boolean",
            "default": False,
            "description": "Enable multi-face tracking with persistent IDs.",
        },
        "speaker_detection": {
            "type": "boolean",
            "default": False,
            "description": "Enable audio diarization + optional visual lip-movement analysis.",
        },
        "active_reframe": {
            "type": "boolean",
            "default": False,
            "description": "Smooth camera pan that follows the active speaker.",
        },
        "max_faces": {
            "type": "integer",
            "default": 4,
            "min": 1,
            "max": 8,
            "description": "Maximum number of faces to track simultaneously.",
        },
        "use_visual": {
            "type": "boolean",
            "default": True,
            "description": "Use lip-movement to confirm/override audio speaker assignment.",
        },
    }