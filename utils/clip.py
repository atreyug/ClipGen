"""
utils/clip_advanced.py
───────────────────────
Extended video processing pipeline orchestration.

FIXES:
- Made all functions synchronous (removed broken async)
- Added proper file validation
- Improved error context
- Added duration limits
"""

import os
import shutil
import tempfile
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import uuid

from pathlib import Path
from typing import Optional
from uuid import UUID

from fastapi import HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from models import Clip, Video

from services.transcribe import transcribe
from services.analyze import chatbot
from services.clipper import (
    ClipOptions,
    create_clips_advanced,
)
from services.cloudinary import (
    upload_clip_to_cloudinary,
    upload_video_to_cloudinary,
)

logger = logging.getLogger(__name__)


def _build_options(
    dimensions: str,
    caption: bool,
    caption_style: Optional[str],
    multi_face: bool,
    speaker_detection: bool,
    active_reframe: bool,
    layout: Optional[str],
    max_faces: int,
    preferred_face_id: Optional[int],
    hf_token: Optional[str],
    use_visual: bool,
) -> ClipOptions:
    """Build ClipOptions from request parameters."""
    return ClipOptions(
        dimensions=dimensions,
        caption=caption,
        caption_style=caption_style or "classic",
        multi_face=multi_face,
        max_faces=max_faces,
        speaker_detection=speaker_detection,
        active_speaker_reframe=active_reframe,
        layout=layout or "auto",
        preferred_face_id=preferred_face_id,
        hf_token=hf_token or os.environ.get("HUGGINGFACE_TOKEN"),
        use_visual_speaker=use_visual,
    )


def generate_clip_files_advanced(
    temp_video_path: str,
    specification: Optional[str],
    options: ClipOptions,
):
    """
    Run transcription → LLM analysis → advanced clip rendering.
    """
    # Validate video file
    if not os.path.exists(temp_video_path):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Video file not found: {temp_video_path}",
        )
    
    # Check file size (max 2GB)
    file_size = os.path.getsize(temp_video_path)
    if file_size > 2 * 1024 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large: {file_size / 1024 / 1024:.1f}MB (max 2GB)",
        )

    dim_clean = str(options.dimensions).strip().lower()
    if dim_clean not in ("9:16", "9_16", "16:9", "16_9"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid dimension '{options.dimensions}'. Must be '9:16' or '16:9'.",
        )

    try:
        transcript, all_words = transcribe(temp_video_path)
        logger.info("Transcription complete")
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcription failed: {str(e)}",
        )

    if not transcript:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Transcription produced no segments.",
        )

    try:
        result = chatbot(transcript, specification)
        logger.info("LLM analysis complete")
    except Exception as e:
        logger.error(f"LLM analysis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"LLM analysis failed: {str(e)}",
        )

    llm_clips = result.get("clips", [])
    if not llm_clips:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"LLM returned no clip suggestions. Response: {result}",
        )

    output_dir = tempfile.mkdtemp(prefix="clipgen_advanced_")

    try:
        clips_data = []
        for i, clip in enumerate(llm_clips):
            start = float(clip.get("start", 0))
            end = float(clip.get("end", start + 30))
            
            # Safety check: max 10 minute clips
            if end - start > 600:
                logger.warning(f"Clip {i+1} too long ({end-start:.1f}s), truncating to 600s")
                end = start + 600

            clips_data.append({
                "start": start,
                "end": end,
                "viral_score": clip.get("viral_score"),
                "reason": clip.get("reason"),
                "filename": f"clip_{i + 1}.mp4",
            })

        output_paths = create_clips_advanced(
            input_path=temp_video_path,
            clips_data=clips_data,
            all_words=all_words,
            output_dir=output_dir,
            options=options,
        )

        logger.info(f"Created {len(output_paths)} clips")
        return (output_paths, clips_data, output_dir)

    except Exception as e:
        shutil.rmtree(output_dir, ignore_errors=True)
        logger.error(f"Clip generation failed: {e}")
        raise


def process_video_pipeline_advanced(
    temp_video_path: str = None,
    video_path: str = None,
    original_filename: str = "video.mp4",
    specification: Optional[str] = None,
    db: Session = None,
    dimensions: str = "9:16",
    caption: bool = True,
    current_user: dict = None,
    user_id: str = None,
    caption_style: str = "classic",
    multi_face: bool = False,
    speaker_detection: bool = False,
    active_reframe: bool = False,
    layout: str = "auto",
    max_faces: int = 4,
    preferred_face_id: Optional[int] = None,
    use_visual: bool = True,
) -> dict:
    """
    Full end-to-end advanced pipeline.
    SYNCHRONOUS - no async/await.
    """
    temp_video_path = temp_video_path or video_path

    if not temp_video_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="temp_video_path (or video_path) is required.",
        )

    if current_user is None and user_id is not None:
        current_user = {"user_id": user_id}

    if current_user is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="current_user (or user_id) is required.",
        )

    video_record = None
    output_dir = None

    try:
        video_record = create_video_record(
            temp_video_path=temp_video_path,
            original_filename=original_filename,
            db=db,
            current_user=current_user,
        )
        logger.info(f"Video record created: {video_record.id}")

        options = _build_options(
            dimensions=dimensions,
            caption=caption,
            caption_style=caption_style,
            multi_face=multi_face,
            speaker_detection=speaker_detection,
            active_reframe=active_reframe,
            layout=layout,
            max_faces=max_faces,
            preferred_face_id=preferred_face_id,
            hf_token=None,
            use_visual=use_visual,
        )

        (output_paths, clip_data, output_dir) = generate_clip_files_advanced(
            temp_video_path=temp_video_path,
            specification=specification,
            options=options,
        )

        clips = save_clips(
            video_record=video_record,
            output_paths=output_paths,
            clip_data=clip_data,
            db=db,
        )
        logger.info(f"Saved {len(clips)} clips")

        # Commit AFTER all processing succeeds
        video_record.status = "processed"
        db.commit()

        return {
            "video_id": video_record.id,
            "video_url": video_record.videolink,
            "duration": video_record.duration,
            "status": video_record.status,
            "clips": clips,
            "options_used": {
                "caption_style": options.caption_style,
                "layout": options.layout,
                "multi_face": options.multi_face,
                "speaker_detection": options.speaker_detection,
                "active_reframe": options.active_speaker_reframe,
                "dimensions": options.dimensions,
            },
        }

    except HTTPException:
        if video_record:
            mark_video_failed(video_record=video_record, db=db)
        raise

    except Exception as exc:
        logger.error(f"Pipeline failed: {exc}", exc_info=True)
        if video_record:
            mark_video_failed(video_record=video_record, db=db)
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(exc)}",
        )

    finally:
        if output_dir and os.path.exists(output_dir):
            shutil.rmtree(output_dir, ignore_errors=True)


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


def get_available_styles() -> Dict[str, Any]:
    """Return all available caption styles."""
    # Placeholder - implement based on your caption_styles module
    return {
        "styles": [
            {"preset": "classic", "description": "Classic white text with black outline"},
            {"preset": "neon_green", "description": "Neon green glowing text"},
            {"preset": "fire", "description": "Fire effect text"},
        ]
    }


def get_available_layouts() -> Dict[str, Any]:
    """Return all available layout presets."""
    return {
        "layouts": [
            {"name": "auto", "description": "Automatic layout selection"},
            {"name": "single", "description": "Single face crop"},
            {"name": "split_vertical", "description": "Two faces side by side"},
            {"name": "split_horizontal", "description": "Two faces stacked"},
        ]
    }