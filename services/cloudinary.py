import os

import cloudinary
import cloudinary.uploader

from config.config import settings


# Configure Cloudinary once at module load using env vars
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