import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

try:
    from dotenv import load_dotenv

    # Load the project-local tunnel settings before module-level environment
    # reads. The frontend and backend share this file during local development.
    load_dotenv(Path(__file__).resolve().parent.parent / ".env.local")
    load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .deps import store
from .realtime import RealtimeSessionManager
from .routes import intelligence, invitations, participants, sessions


async def _warmup_intelligence() -> None:
    """Preload Silero + Whisper in the background (non-blocking server boot).

    Lazy loading remains as fallback, so a failed warmup never breaks
    startup — but without this, the first speech segment would pay the
    full model download + load cost as user-visible delay.
    """
    try:
        from .audio_intelligence import transcription
        from .audio_intelligence.vad import SileroVAD

        await asyncio.to_thread(transcription.warmup)
        vad = SileroVAD()
        await asyncio.to_thread(lambda: vad.available)
        print(f"[intelligence] warmup done (silero={vad.available})", flush=True)
    except Exception as exc:
        print(f"[intelligence] warmup failed, lazy fallback active: {exc}", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.get_event_loop().create_task(_warmup_intelligence())
    yield


app = FastAPI(title="Roundtable API", version="0.1.0", lifespan=lifespan)
configured_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", *configured_origins],
    allow_origin_regex=(
        r"https?://(?:[^/]+:5173|[a-zA-Z0-9-]+\.ngrok(?:-free)?\.(?:app|dev|io)"
        r"|[a-zA-Z0-9-]+\.inc\d+\.devtunnels\.ms)"
    ),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

realtime = RealtimeSessionManager(store)
app.state.realtime = realtime

app.include_router(sessions.router, prefix="/api", dependencies=[])
app.include_router(invitations.router, prefix="/api")
app.include_router(participants.router, prefix="/api")
app.include_router(intelligence.router, prefix="/api")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.websocket("/ws/sessions/{session_id}")
async def session_socket(
    websocket: WebSocket,
    session_id: str,
    host_token: str | None = None,
    participant_id: str | None = None,
    participant_token: str | None = None,
):
    await realtime.handle_websocket(
        websocket,
        session_id,
        host_token=host_token,
        participant_id=participant_id,
        participant_token=participant_token,
    )
