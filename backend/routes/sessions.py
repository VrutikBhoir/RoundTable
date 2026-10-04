import os

from fastapi import APIRouter, Header, HTTPException, Request

from ..deps import authorized_session, domain_error, host_token_header, store
from ..models import CreateSessionRequest, LobbyResponse, SessionCredentials

router = APIRouter(tags=["sessions"])


def frontend_origin(request: Request) -> str:
    configured = os.getenv("PUBLIC_APP_URL", "").strip().rstrip("/")
    if configured:
        return configured
    return request.headers.get("origin") or "http://localhost:5173"


@router.post("/sessions", response_model=SessionCredentials)
def create_session(payload: CreateSessionRequest) -> SessionCredentials:
    credentials, _ = store.create_session(payload.name, payload.capacity, payload.host_name)
    return credentials


@router.get("/sessions/{session_id}/lobby", response_model=LobbyResponse)
def get_lobby(session_id: str, request: Request, host_token: str | None = Header(default=None, alias="X-Host-Token")) -> LobbyResponse:
    record = authorized_session(session_id, host_token)
    return store.lobby(record, frontend_origin(request))


@router.post("/sessions/{session_id}/lock", response_model=LobbyResponse)
async def lock_session(session_id: str, request: Request, host_token: str | None = Header(default=None, alias="X-Host-Token")) -> LobbyResponse:
    record = authorized_session(session_id, host_token)
    try:
        await store.set_locked(record, True)
    except ValueError as error:
        raise domain_error(error) from error
    return store.lobby(record, frontend_origin(request))


@router.post("/sessions/{session_id}/unlock", response_model=LobbyResponse)
async def unlock_session(session_id: str, request: Request, host_token: str | None = Header(default=None, alias="X-Host-Token")) -> LobbyResponse:
    record = authorized_session(session_id, host_token)
    try:
        await store.set_locked(record, False)
    except ValueError as error:
        raise domain_error(error) from error
    return store.lobby(record, frontend_origin(request))


@router.post("/sessions/{session_id}/start", response_model=LobbyResponse)
async def start_session(session_id: str, request: Request, host_token: str | None = Header(default=None, alias="X-Host-Token")) -> LobbyResponse:
    record = authorized_session(session_id, host_token)
    try:
        await store.start(record)
    except ValueError as error:
        raise domain_error(error) from error
    return store.lobby(record, frontend_origin(request))
