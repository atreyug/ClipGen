from pathlib import Path
from tempfile import mkdtemp
from typing import Optional
import os

import yt_dlp


class YouTubeDownloadError(Exception):
    """Raised when a YouTube video cannot be downloaded."""


_PLAYER_CLIENT_FALLBACKS = ["android", "ios", "web"]

_FORMAT_SELECTOR = (
    "best[ext=mp4][vcodec!=none][acodec!=none]/"
    "bestvideo[ext=mp4]+bestaudio[ext=m4a]/"
    "best"
)

# Recommended Cloud Run Secret Manager mount location.
# You can change this if you prefer another path.
_YOUTUBE_COOKIES_FILE = Path(
    os.getenv("YOUTUBE_COOKIES_FILE", "/app/secrets/youtube_cookies.txt")
)


def _build_ydl_options(
    output_template: str,
    player_client: str,
) -> dict:
    """
    Build yt-dlp options for one YouTube player client.
    """

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

    # Use authenticated YouTube cookies when Cloud Run has them mounted.
    #
    # This is deliberately optional so local development still works
    # without a cookie file.
    if _YOUTUBE_COOKIES_FILE.is_file():
        ydl_opts["cookiefile"] = str(_YOUTUBE_COOKIES_FILE)

    return ydl_opts


def download_youtube_video(
    youtube_url: str,
    output_dir: Optional[str] = None,
) -> Path:

    if not youtube_url:
        raise YouTubeDownloadError("YouTube URL is required.")

    if output_dir:
        download_dir = Path(output_dir)
        download_dir.mkdir(parents=True, exist_ok=True)
    else:
        download_dir = Path(mkdtemp(prefix="clipgen_"))

    output_template = str(download_dir / "%(id)s.%(ext)s")

    cookies_available = _YOUTUBE_COOKIES_FILE.is_file()

    last_error: Optional[Exception] = None

    for player_client in _PLAYER_CLIENT_FALLBACKS:

        ydl_opts = _build_ydl_options(
            output_template=output_template,
            player_client=player_client,
        )

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:

                info = ydl.extract_info(
                    youtube_url,
                    download=True,
                )

                if not info:
                    raise YouTubeDownloadError(
                        "Unable to retrieve YouTube video information."
                    )

                downloaded_file = Path(
                    ydl.prepare_filename(info)
                )

                if not downloaded_file.exists():

                    mp4_file = downloaded_file.with_suffix(".mp4")

                    if mp4_file.exists():
                        downloaded_file = mp4_file

                    else:
                        files = list(
                            download_dir.glob(
                                f"{info['id']}.*"
                            )
                        )

                        if not files:
                            raise YouTubeDownloadError(
                                "Video was not downloaded."
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

    if cookies_available:
        auth_status = (
            "YouTube cookies were provided, but YouTube still rejected "
            "the request. The cookies may be expired or invalid."
        )
    else:
        auth_status = (
            "No YouTube cookie file was found. "
            "Configure the YOUTUBE_COOKIES_FILE secret on Cloud Run."
        )

    raise YouTubeDownloadError(
        f"Failed to download YouTube video after trying player clients "
        f"{_PLAYER_CLIENT_FALLBACKS}. "
        f"{auth_status} "
        f"Last error: {last_error}"
    ) from last_error