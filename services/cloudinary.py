import os

import cloudinary
import cloudinary.uploader

from config.config import settings


cloudinary.config(
    cloud_name=settings.CLOUDINARY_CLOUD_NAME,
    api_key=settings.CLOUDINARY_API_KEY,
    api_secret=settings.CLOUDINARY_API_SECRET,
    secure=True,
)


def upload_video_to_cloudinary(
    file_path: str,
    public_id: str,
):
    file_size = os.path.getsize(file_path)

    upload_options = {
        "resource_type": "video",
        "public_id": public_id,
        "folder": "clipgen/videos",
    }

    if file_size > 100 * 1024 * 1024:
        return cloudinary.uploader.upload_large(
            file_path,
            chunk_size=20 * 1024 * 1024,
            **upload_options,
        )

    return cloudinary.uploader.upload(
        file_path,
        **upload_options,
    )


def upload_clip_to_cloudinary(
    file_path: str,
    public_id: str,
):
    return cloudinary.uploader.upload(
        file_path,
        resource_type="video",
        public_id=public_id,
        folder="clipgen/clips",
    )



def delete_video_from_cloudinary(public_id: str) -> dict:
    if not public_id.startswith("clipgen/videos/") and "/" not in public_id:
        public_id = f"clipgen/videos/{public_id}"

    try:
        response = cloudinary.uploader.destroy(
            public_id,
            resource_type="video",
            invalidate=True,  # Clears CDN cache immediately
        )
        return response
    except Exception as exc:
        print(f"[Cloudinary] Failed to delete video '{public_id}': {exc}")
        return {"result": "error", "message": str(exc)}


def delete_clip_from_cloudinary(public_id: str) -> dict:
    
    if not public_id.startswith("clipgen/clips/") and "/" not in public_id:
        public_id = f"clipgen/clips/{public_id}"

    try:
        response = cloudinary.uploader.destroy(
            public_id,
            resource_type="video",
            invalidate=True,  # Clears CDN cache immediately
        )
        return response
    except Exception as exc:
        print(f"[Cloudinary] Failed to delete clip '{public_id}': {exc}")
        return {"result": "error", "message": str(exc)}