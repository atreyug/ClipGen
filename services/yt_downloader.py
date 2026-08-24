from pathlib import Path
from tempfile import mkdtemp
from typing import Optional

import yt_dlp


class YouTubeDownloadError(Exception):
    """Raised when a YouTube video cannot be downloaded."""


_PLAYER_CLIENT_FALLBACKS = ["android", "ios", "web"]
_FORMAT_SELECTOR = (
    "best[ext=mp4][vcodec!=none][acodec!=none]/"
    "bestvideo[ext=mp4]+bestaudio[ext=m4a]/"
    "best"
)


def download_youtube_video(youtube_url: str, output_dir: Optional[str] = None) -> Path:
    if not youtube_url:
        raise YouTubeDownloadError("YouTube URL is required.")

    if output_dir:
        download_dir = Path(output_dir)
        download_dir.mkdir(parents=True, exist_ok=True)
    else:
        download_dir = Path(mkdtemp(prefix="clipgen_"))

    output_template = str(download_dir / "%(id)s.%(ext)s")

    last_error: Optional[Exception] = None

    for player_client in _PLAYER_CLIENT_FALLBACKS:
        ydl_opts = {
            "format": _FORMAT_SELECTOR,

            "outtmpl": output_template,

            "noplaylist": True,

            "merge_output_format": "mp4",

            "quiet": True,
            "no_warnings": True,

            "overwrites": False,

            "extractor_args": {
                "youtube": {
                    "player_client": [player_client],
                }
            },

            "retries": 3,
            "fragment_retries": 3,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=True)

                if not info:
                    raise YouTubeDownloadError(
                        "Unable to retrieve YouTube video information."
                    )

                downloaded_file = Path(ydl.prepare_filename(info))

                if not downloaded_file.exists():
                    mp4_file = downloaded_file.with_suffix(".mp4")

                    if mp4_file.exists():
                        downloaded_file = mp4_file
                    else:
                        files = list(download_dir.glob(f"{info['id']}.*"))

                        if not files:
                            raise YouTubeDownloadError(
                                "Video not found!"
                            )

                        downloaded_file = files[0]

                return downloaded_file

        except yt_dlp.utils.DownloadError as exc:
            last_error = exc
            continue

        except YouTubeDownloadError:
            raise

        except Exception as exc:
            last_error = exc
            continue

    raise YouTubeDownloadError(
        f"Failed to download YouTube video after trying player clients "
        f"{_PLAYER_CLIENT_FALLBACKS}: {last_error}"
    ) from last_error