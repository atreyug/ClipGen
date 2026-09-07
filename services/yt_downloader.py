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


# Cloud Run Secret Manager mounted file
# _YOUTUBE_COOKIES_FILE = Path(
#     os.getenv(
#         "YOUTUBE_COOKIES_FILE",
#         "/app/secrets/youtube_cookies.txt",
#     )
# )

_YOUTUBE_COOKIES_FILE = Path(
    os.getenv(
        "YOUTUBE_COOKIES_FILE",
        "youtube_cookies.txt",
    )
)


def _build_ydl_options(
    output_template: str,
    player_client: str,
) -> dict:
    """Build yt-dlp options for a YouTube player client."""

    ydl_opts = {
        "format": _FORMAT_SELECTOR,
        "outtmpl": output_template,

        "noplaylist": True,
        "merge_output_format": "mp4",

        "quiet": True,
        "no_warnings": False,

        "overwrites": False,

        "extractor_args": {
            "youtube": {
                "player_client": [player_client],
            }
        },

        "retries": 3,
        "fragment_retries": 3,

        # Avoid unnecessary cache/state issues.
        "cachedir": False,
    }

    # IMPORTANT:
    # Pass the mounted Cloud Run cookie file to yt-dlp.
    if _YOUTUBE_COOKIES_FILE.is_file():
        ydl_opts["cookiefile"] = str(_YOUTUBE_COOKIES_FILE)

    return ydl_opts


def download_youtube_video(
    youtube_url: str,
    output_dir: Optional[str] = None,
) -> Path:

    if not youtube_url:
        raise YouTubeDownloadError(
            "YouTube URL is required."
        )

    # ---------------------------------------------------------
    # Prepare output directory
    # ---------------------------------------------------------

    if output_dir:
        download_dir = Path(output_dir)
        download_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
    else:
        download_dir = Path(
            mkdtemp(prefix="clipgen_")
        )

    output_template = str(
        download_dir / "%(id)s.%(ext)s"
    )

    # ---------------------------------------------------------
    # Check cookies
    # ---------------------------------------------------------

    cookies_available = (
        _YOUTUBE_COOKIES_FILE.is_file()
    )

    if cookies_available:
        cookie_size = (
            _YOUTUBE_COOKIES_FILE.stat().st_size
        )

        print(
            f"YouTube cookies found: "
            f"{_YOUTUBE_COOKIES_FILE}"
        )

        print(
            f"YouTube cookie file size: "
            f"{cookie_size} bytes"
        )

    else:
        print(
            "WARNING: YouTube cookie file not found:"
        )
        print(
            f"  {_YOUTUBE_COOKIES_FILE}"
        )

    # ---------------------------------------------------------
    # Try YouTube player clients
    # ---------------------------------------------------------

    last_error: Optional[Exception] = None

    for player_client in _PLAYER_CLIENT_FALLBACKS:

        print(
            f"Trying YouTube player client: "
            f"{player_client}"
        )

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
                        "Unable to retrieve YouTube "
                        "video information."
                    )

                downloaded_file = Path(
                    ydl.prepare_filename(info)
                )

                # yt-dlp may merge the streams into MP4.
                if not downloaded_file.exists():

                    mp4_file = (
                        downloaded_file.with_suffix(".mp4")
                    )

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

                        # Prefer actual video files.
                        video_files = [
                            f
                            for f in files
                            if f.suffix.lower()
                            in {
                                ".mp4",
                                ".mkv",
                                ".webm",
                                ".mov",
                            }
                        ]

                        if video_files:
                            downloaded_file = (
                                video_files[0]
                            )
                        else:
                            downloaded_file = files[0]

                print(
                    f"YouTube video downloaded: "
                    f"{downloaded_file}"
                )

                return downloaded_file

        except yt_dlp.utils.DownloadError as exc:

            last_error = exc

            print(
                f"YouTube client '{player_client}' "
                f"failed:"
            )
            print(exc)

            continue

        except YouTubeDownloadError:
            raise

        except Exception as exc:

            last_error = exc

            print(
                f"Unexpected error with "
                f"'{player_client}': {exc}"
            )

            continue

    # ---------------------------------------------------------
    # All clients failed
    # ---------------------------------------------------------

    if cookies_available:

        auth_status = (
            "The YouTube cookie file was found and "
            "provided to yt-dlp, but YouTube rejected "
            "the request. The cookies may be expired, "
            "invalid, or YouTube may be rejecting the "
            "Cloud Run request."
        )

    else:

        auth_status = (
            "The YouTube cookie file was not found at "
            f"{_YOUTUBE_COOKIES_FILE}."
        )

    raise YouTubeDownloadError(
        "Failed to download YouTube video after trying "
        f"player clients {_PLAYER_CLIENT_FALLBACKS}. "
        f"{auth_status} "
        f"Last error: {last_error}"
    ) from last_error