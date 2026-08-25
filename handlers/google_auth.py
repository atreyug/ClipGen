from fastapi import APIRouter, HTTPException
from fastapi.responses import RedirectResponse

from utils.email import get_google_flow, TOKEN_FILE


router = APIRouter(
    prefix="/google",
    tags=["Google OAuth"],
)

_pending_flows: dict[str, object] = {}


@router.get("/authorize")
def google_authorize():
    flow = get_google_flow()

    authorization_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )

    _pending_flows[state] = flow

    return RedirectResponse(authorization_url)


@router.get("/callback")
def google_callback(code: str, state: str):
    flow = _pending_flows.pop(state, None)

    if flow is None:
        raise HTTPException(
            status_code=400,
            detail="OAuth flow not found or expired. Start authorization again.",
        )

    try:
        flow.fetch_token(code=code)

        credentials = flow.credentials

        TOKEN_FILE.write_text(
            credentials.to_json(),
            encoding="utf-8",
        )

        return {
            "success": True,
            "message": "Gmail API authorization successful.",
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Google OAuth token exchange failed: {str(e)}",
        )