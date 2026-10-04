"""Replaceable in-memory repository for the local MVP.

This store intentionally loses data on restart and is not safe for multiple
backend instances. A database implementation can preserve this repository's
method-level contract while using transactions and row locks.
"""

import asyncio
import hashlib
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from hmac import compare_digest
from typing import Optional
from uuid import uuid4

from .models import (
    InvitationPreview,
    InvitationResponse,
    JoinRequestResponse,
    LobbyResponse,
    ParticipantResponse,
    ParticipantStatus,
    RequestStatus,
    SessionCredentials,
    SessionStatus,
)

INVITATION_TTL = timedelta(minutes=5)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def hash_secret(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass
class InvitationRecord:
    token: str
    expires_at: datetime
    active: bool = True
    invalidated_at: Optional[datetime] = None


@dataclass
class ParticipantRecord:
    id: str
    display_name: str
    created_at: datetime
    # MVP plaintext credential for this session only. A production database
    # implementation must encrypt this at rest and support rotation.
    access_token: Optional[str] = None


@dataclass
class JoinRequestRecord:
    id: str
    display_name: str
    status: RequestStatus
    created_at: datetime
    participant_id: Optional[str] = None


@dataclass
class SessionRecord:
    id: str
    name: str
    capacity: int
    host_token_hash: str
    host_name: str = "Host"
    status: SessionStatus = SessionStatus.WAITING
    participants: dict[str, ParticipantRecord] = field(default_factory=dict)
    requests: dict[str, JoinRequestRecord] = field(default_factory=dict)
    invitation: Optional[InvitationRecord] = None
    locked_from_started: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class InMemorySessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, SessionRecord] = {}

    def create_session(self, name: str, capacity: int, host_name: Optional[str] = None) -> tuple[SessionCredentials, SessionRecord]:
        session_id = str(uuid4())
        host_token = secrets.token_urlsafe(32)
        clean_host = (host_name or "").strip() or "Host"
        record = SessionRecord(
            id=session_id,
            name=name.strip(),
            capacity=capacity,
            host_token_hash=hash_secret(host_token),
            host_name=clean_host,
        )
        self._sessions[session_id] = record
        return (
            SessionCredentials(
                session_id=session_id,
                host_token=host_token,
                name=record.name,
                capacity=capacity,
                status=record.status,
                host_name=record.host_name,
            ),
            record,
        )

    def get(self, session_id: str) -> Optional[SessionRecord]:
        return self._sessions.get(session_id)

    def find_by_invitation(self, token: str) -> Optional[SessionRecord]:
        return next(
            (
                record
                for record in self._sessions.values()
                if record.invitation and compare_digest(record.invitation.token, token)
            ),
            None,
        )

    def invitation_response(self, record: SessionRecord, frontend_origin: str) -> InvitationResponse:
        invitation = self._invitation_response(record, frontend_origin)
        if invitation is None:
            raise ValueError("invalid_invitation")
        return invitation

    def authorize_host(self, record: SessionRecord, token: str) -> bool:
        return compare_digest(record.host_token_hash, hash_secret(token))

    def _status(self, record: SessionRecord) -> SessionStatus:
        if record.status == SessionStatus.WAITING and len(record.participants) >= record.capacity:
            return SessionStatus.FULL
        return record.status

    def joinable_status(self, record: SessionRecord) -> str:
        """Public invitation state: open | in_progress | full | locked | ended."""
        if record.status == SessionStatus.ENDED:
            return "ended"
        if record.status == SessionStatus.LOCKED:
            return "locked"
        if len(record.participants) >= record.capacity:
            return "full"
        if record.status in {SessionStatus.STARTING, SessionStatus.STARTED}:
            return "in_progress"
        return "open"

    def invitation_preview(self, record: SessionRecord, frontend_origin: str) -> InvitationPreview:
        invitation = self._invitation_response(record, frontend_origin)
        if invitation is None:
            raise ValueError("invalid_invitation")
        return InvitationPreview(
            session_id=record.id,
            session_name=record.name,
            host_name=record.host_name,
            capacity=record.capacity,
            participant_count=len(record.participants),
            session_status=self.joinable_status(record),
            invitation=invitation,
        )

    def authorize_participant(self, record: SessionRecord, participant_id: str, token: str) -> bool:
        """Validate a participant credential for future participant-scoped routes."""
        participant = record.participants.get(participant_id)
        if participant is None or not participant.access_token:
            return False
        return compare_digest(participant.access_token, token)

    def _invitation_response(self, record: SessionRecord, frontend_origin: str) -> Optional[InvitationResponse]:
        invitation = record.invitation
        if invitation is None:
            return None
        is_active = invitation.active and invitation.expires_at > utc_now()
        return InvitationResponse(
            token=invitation.token,
            invitation_url=f"{frontend_origin.rstrip('/')}/join/{invitation.token}",
            expires_at=invitation.expires_at,
            active=is_active,
        )

    def lobby(self, record: SessionRecord, frontend_origin: str, include_host_token: Optional[str] = None) -> LobbyResponse:
        return LobbyResponse(
            session_id=record.id,
            name=record.name,
            host_name=record.host_name,
            capacity=record.capacity,
            status=self._status(record),
            host_token=include_host_token,
            invitation=self._invitation_response(record, frontend_origin),
            participants=[
                ParticipantResponse(id=p.id, display_name=p.display_name, status=ParticipantStatus.APPROVED, created_at=p.created_at)
                for p in record.participants.values()
            ],
            pending_requests=[
                JoinRequestResponse(id=r.id, display_name=r.display_name, status=r.status, created_at=r.created_at)
                for r in record.requests.values()
                if r.status == RequestStatus.PENDING
            ],
        )

    async def create_invitation(self, record: SessionRecord) -> InvitationRecord:
        async with record.lock:
            return self._replace_invitation(record)

    async def regenerate_invitation(self, record: SessionRecord) -> InvitationRecord:
        async with record.lock:
            return self._replace_invitation(record)

    def _replace_invitation(self, record: SessionRecord) -> InvitationRecord:
        if record.invitation:
            record.invitation.active = False
            record.invitation.invalidated_at = utc_now()
        record.invitation = InvitationRecord(token=secrets.token_urlsafe(32), expires_at=utc_now() + INVITATION_TTL)
        return record.invitation

    async def request_join(self, record: SessionRecord, invitation_token: str, display_name: str) -> JoinRequestRecord:
        async with record.lock:
            invitation = record.invitation
            if not invitation or not compare_digest(invitation.token, invitation_token):
                raise ValueError("invalid_invitation")
            if not invitation.active:
                raise ValueError("invitation_invalidated")
            if invitation.expires_at <= utc_now():
                raise ValueError("invitation_expired")
            if record.status == SessionStatus.LOCKED:
                raise ValueError("session_locked")
            if record.status == SessionStatus.STARTING:
                raise ValueError("session_already_started")
            if record.status == SessionStatus.ENDED:
                raise ValueError("session_ended")
            if len(record.participants) >= record.capacity:
                if record.status == SessionStatus.WAITING:
                    record.status = SessionStatus.FULL
                raise ValueError("session_full")
            normalized = display_name.strip()
            if any(p.display_name.casefold() == normalized.casefold() for p in record.participants.values()) or any(
                r.display_name.casefold() == normalized.casefold() and r.status == RequestStatus.PENDING for r in record.requests.values()
            ):
                raise ValueError("duplicate_request")
            request = JoinRequestRecord(id=str(uuid4()), display_name=normalized, status=RequestStatus.PENDING, created_at=utc_now())
            record.requests[request.id] = request
            return request

    async def approve_request(self, record: SessionRecord, request_id: str) -> JoinRequestRecord:
        async with record.lock:
            request = record.requests.get(request_id)
            if not request or request.status != RequestStatus.PENDING:
                raise ValueError("invalid_request")
            if record.status == SessionStatus.WAITING and len(record.participants) >= record.capacity:
                record.status = SessionStatus.FULL
                raise ValueError("session_full")
            request.status = RequestStatus.APPROVED
            participant = ParticipantRecord(id=str(uuid4()), display_name=request.display_name, created_at=utc_now())
            participant.access_token = secrets.token_urlsafe(24)
            record.participants[participant.id] = participant
            request.participant_id = participant.id
            if len(record.participants) >= record.capacity:
                record.status = SessionStatus.FULL
            return request

    async def reject_request(self, record: SessionRecord, request_id: str) -> JoinRequestRecord:
        async with record.lock:
            request = record.requests.get(request_id)
            if not request or request.status != RequestStatus.PENDING:
                raise ValueError("invalid_request")
            request.status = RequestStatus.REJECTED
            return request

    async def remove_participant(self, record: SessionRecord, participant_id: str) -> None:
        async with record.lock:
            if participant_id not in record.participants:
                raise ValueError("participant_not_found")
            del record.participants[participant_id]
            if record.status == SessionStatus.FULL:
                record.status = SessionStatus.WAITING

    async def set_locked(self, record: SessionRecord, locked: bool) -> None:
        async with record.lock:
            if record.status in {SessionStatus.STARTING, SessionStatus.ENDED}:
                raise ValueError("session_already_started")
            if locked:
                record.locked_from_started = record.status == SessionStatus.STARTED
                record.status = SessionStatus.LOCKED
            elif record.status == SessionStatus.LOCKED:
                record.status = SessionStatus.STARTED if record.locked_from_started else (
                    SessionStatus.FULL if len(record.participants) >= record.capacity else SessionStatus.WAITING
                )
                record.locked_from_started = False
            elif len(record.participants) < record.capacity:
                record.status = SessionStatus.WAITING
            else:
                record.status = SessionStatus.FULL

    async def start(self, record: SessionRecord) -> None:
        async with record.lock:
            if record.status in {SessionStatus.STARTING, SessionStatus.STARTED}:
                raise ValueError("session_already_started")
            if record.status == SessionStatus.ENDED:
                raise ValueError("session_ended")
            record.status = SessionStatus.STARTING
            record.status = SessionStatus.STARTED
