"""Audio-intelligence HTTP routes. Metadata/results only — never raw audio
over the signaling socket (segments arrive here as short POST bodies)."""

import asyncio

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from ..audio_intelligence import audio_io
from ..audio_intelligence.pipeline import pipeline
from ..audio_intelligence.voice import VOICE_ENROLLMENT_DURATION_S, voice_profiles
from ..deps import authorized_session, domain_error, store

router = APIRouter(tags=["intelligence"])

MAX_SEGMENT_BYTES = 2 * 1024 * 1024  # 3 s float32 16 kHz is ~192 KB; cap far above
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_ENROLLMENT_BYTES = 25 * 1024 * 1024


def _resolve_speaker(record, participant_id: str | None, participant_token: str | None, host_token: str | None) -> str:
    """Same credential rules as the signaling socket. Returns speaker key."""
    if host_token:
        if not store.authorize_host(record, host_token):
            raise ValueError("invalid_host_token")
        return "host"
    if participant_id and participant_token and store.authorize_participant(record, participant_id, participant_token):
        if participant_id in record.participants:
            return participant_id
    raise ValueError("invalid_participant_token")


@router.post("/sessions/{session_id}/audio-segments")
async def post_audio_segment(session_id: str, request: Request, payload: dict):
    """Ingest one ~3 s PCM window. Body: participant/host creds + t0 + pcm_b64."""
    record = store.get(session_id)
    if record is None:
        raise HTTPException(status_code=404, detail={"code": "session_not_found", "message": "Session not found."})
    try:
        speaker = _resolve_speaker(
            record, payload.get("participant_id"), payload.get("participant_token"), payload.get("host_token")
        )
        b64 = payload.get("pcm_b64", "")
        if len(b64) > MAX_SEGMENT_BYTES * 2:
            raise ValueError("segment_too_large")
        samples = audio_io.decode_base64_pcm(b64)
        if samples.size > 16000 * 30:
            raise ValueError("segment_too_large")
        non_zero = int(np.count_nonzero(samples))
        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        rms = audio_io.rms(samples)
        print(
            f"[AUDIO INPUT] participant_id={payload.get('participant_id') or 'host'} "
            f"segment_duration={samples.size / 16000:.2f}s pcm_samples={samples.size} "
            f"sample_rate=16000 rms={rms:.4f} peak_amplitude={peak:.4f} "
            f"non_zero_percent={(non_zero / max(1, samples.size)) * 100:.2f}",
            flush=True,
        )
        chunk_start = float(payload.get("t0", 0.0))
    except ValueError as error:
        raise domain_error(error) from error
    try:
        new_entries = await pipeline.ingest(session_id, speaker, samples, chunk_start)
    except Exception as error:
        raise HTTPException(status_code=500, detail={"code": "pipeline_error", "message": str(error)[:200]})
    if new_entries:
        realtime = request.app.state.realtime
        await realtime.broadcast_event(session_id, {
            "type": "transcript_event",
            "session_id": session_id,
            "entries": [e.to_dict() for e in new_entries],
        })
    return {"accepted": True, "new_entries": [e.to_dict() for e in new_entries]}


@router.get("/sessions/{session_id}/transcript")
async def get_transcript(
    session_id: str,
    participant_id: str | None = None,
    participant_token: str | None = None,
    host_token: str | None = None,
):
    record = store.get(session_id)
    if record is None:
        raise HTTPException(status_code=404, detail={"code": "session_not_found", "message": "Session not found."})
    try:
        if host_token:
            authorized_session(session_id, host_token)
        else:
            _resolve_speaker(record, participant_id, participant_token, None)
    except ValueError as error:
        raise domain_error(error) from error
    return {"session_id": session_id, "entries": pipeline.transcript(session_id)}


@router.post("/sessions/{session_id}/voice-enrollment")
async def enroll_voice(
    session_id: str,
    audio: UploadFile = File(...),
    participant_id: str | None = Form(default=None),
    participant_token: str | None = Form(default=None),
    host_token: str | None = Form(default=None),
):
    record = store.get(session_id)
    if record is None:
        raise HTTPException(status_code=404, detail={"code": "session_not_found", "message": "Session not found."})
    try:
        participant = _resolve_speaker(record, participant_id, participant_token, host_token)
        data = await audio.read()
        if len(data) > MAX_ENROLLMENT_BYTES:
            raise ValueError("enrollment_too_large")
        samples = audio_io.decode_upload(data, audio.filename or "enrollment.webm")
        speech_seconds = await asyncio.to_thread(
            voice_profiles.enroll, session_id, participant, samples
        )
        print(
            f"[voice] enrollment success session_id={session_id} participant_id={participant} "
            f"speech_seconds={speech_seconds:.2f} recording_seconds={samples.size / 16000:.2f}",
            flush=True,
        )
    except (ValueError, RuntimeError) as error:
        error_text = str(error)
        error_parts = error_text.split(":")
        code = error_parts[0]
        status = (
            422 if code == "insufficient_speech"
            else 503 if "unavailable" in str(error) or "decoder" in str(error)
            else 400
        )
        message = error_text[:200]
        if code == "insufficient_speech" and len(error_parts) >= 3:
            required = error_parts[3] if len(error_parts) >= 4 else "8.00"
            message = (
                f"Not enough speech detected. We detected {error_parts[1]} seconds "
                f"of speech out of {error_parts[2]} seconds. Please speak for a "
                f"little longer and try again. Required: {required} seconds."
            )
        raise HTTPException(status_code=status, detail={"code": code, "message": message}) from error
    return {
        "participant_id": participant,
        "status": "ready",
        "profile_ready": True,
        "speech_seconds": round(speech_seconds, 2),
        "recording_seconds": round(samples.size / 16000, 2),
        "enrollment_duration_seconds": VOICE_ENROLLMENT_DURATION_S,
    }


@router.get("/sessions/{session_id}/voice-status")
async def voice_status(
    session_id: str,
    participant_id: str | None = None,
    participant_token: str | None = None,
    host_token: str | None = None,
):
    record = store.get(session_id)
    if record is None:
        raise HTTPException(status_code=404, detail={"code": "session_not_found", "message": "Session not found."})
    try:
        participant = _resolve_speaker(record, participant_id, participant_token, host_token)
    except ValueError as error:
        raise domain_error(error) from error
    status = voice_profiles.status(session_id, participant)
    print(
        f"[voice] status lookup session_id={session_id} participant_id={participant} status={status}",
        flush=True,
    )
    return {
        "participant_id": participant,
        "status": status,
        "enrollment_duration_seconds": VOICE_ENROLLMENT_DURATION_S,
    }


@router.post("/intelligence/test")
async def test_files(files: list[UploadFile] = File(...), participant_ids: str = Form("")):
    """Dev/test only: decode recorded files and run the full pipeline.

    participant_ids: comma-separated ids aligned with files[] order.
    """
    ids = [p.strip() for p in participant_ids.split(",") if p.strip()]
    if not files:
        raise HTTPException(status_code=400, detail={"code": "no_files", "message": "No files uploaded."})
    session_id = "file-test"
    out: list[dict] = []
    for i, upload in enumerate(files):
        data = await upload.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail={"code": "file_too_large", "message": f"{upload.filename} exceeds 25 MB."})
        pid = ids[i] if i < len(ids) else f"file-{i}"
        try:
            samples = audio_io.decode_upload(data, upload.filename or "audio.wav")
        except (ValueError, RuntimeError) as error:
            out.append({"participant_id": pid, "error": str(error)[:200]})
            continue
        samples = audio_io.normalize(samples)
        try:
            new_entries = await pipeline.ingest(session_id, pid, samples, 0.0)
            out.extend(e.to_dict() for e in new_entries)
        except Exception as error:
            out.append({"participant_id": pid, "error": str(error)[:200]})
    out.sort(key=lambda e: (e.get("start", 0.0), e.get("end", 0.0)))
    return {"session_id": session_id, "entries": out}
