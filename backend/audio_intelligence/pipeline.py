"""Per-session intelligence orchestration. Never mixes participant streams.

Flow per ingested chunk: VAD -> quality -> cross-phone duplicate skip ->
Whisper (winner only) -> attribution -> fusion. Blocking work (Whisper)
runs in a threadpool so the signaling loop is never blocked. Any stage may
fail: the meeting continues, that segment is just skipped.
"""

import asyncio
import time

import numpy as np

from . import audio_io, attribution, dedupe, quality, selection, transcription, voice
from .attribution import TemporalVoter
from .fusion import FusionStore
from .models import AudioSegment, FusedEntry, PipelineDecision
from .vad import SileroVAD

_DECISION_TTL_S = 30.0  # recent outcomes kept for duplicate suppression
_DUP_IOU = 0.35
_DUP_QUALITY_MARGIN = 0.05
# Whisper hallucinates on scraps ("thank you" on 0.3 s fragments), so speech
# accumulates per participant before any inference call.
MIN_SPEECH_S = 1.0  # buffer this much speech before transcribing
FLUSH_MIN_S = 0.8  # silence-closed buffers flush early at this size
MAX_BUFFER_S = 8.0  # force-emit cap so long utterances stay bounded
HARD_FLOOR_S = 0.65  # defense in depth: never transcribe tiny scraps
MIN_TRANSCRIPT_CONFIDENCE = 0.30
MIN_TRANSCRIPT_QUALITY = 0.10


class SessionIntelligence:
    def __init__(self) -> None:
        self._session_id = ""
        self.vad = SileroVAD()
        self.voter = TemporalVoter()
        self.fusion = FusionStore()
        self.decisions: list[PipelineDecision] = []
        self.session_t0 = time.time()
        self.last_voice: dict[str, float] = {}  # participant -> end time of last speech
        self.pending: dict[str, dict] = {}  # participant -> buffered speech awaiting MIN_SPEECH_S

    def _prune(self, now: float) -> None:
        self.decisions = [d for d in self.decisions if now - d.end < _DECISION_TTL_S]

    def _duplicate_of(self, seg: AudioSegment, q: float, now: float) -> PipelineDecision | None:
        self._prune(now)
        for d in self.decisions:
            if selection.iou(seg.start, seg.end, d.start, d.end) >= _DUP_IOU and q <= d.quality + _DUP_QUALITY_MARGIN:
                return d
        return None

    def process_segment(self, seg: AudioSegment) -> list[FusedEntry]:
        """Full pipeline for one accumulated speech unit. Returns new fused entries."""
        t_in = time.time()
        now = time.time()
        samples = np.asarray(seg.samples, dtype=np.float32)
        dur_s = samples.size / 16000
        print(
            f"[SEG] participant={seg.participant_id} duration={dur_s:.2f}s "
            f"sr=16000 channels=1 rms={audio_io.rms(samples):.4f} final={seg.final}",
            flush=True,
        )
        if dur_s < HARD_FLOOR_S or samples.size < 1600 or audio_io.is_silent(samples):
            return []  # too short / too quiet: Whisper would hallucinate
        q = quality.score_segment(samples)
        print(
            f"[AUDIO_SEGMENT] participant_id={seg.participant_id} "
            f"duration={dur_s:.2f}s vad_speech_duration={dur_s:.2f}s "
            f"rms={q.rms:.4f} quality={q.score:.3f}",
            flush=True,
        )
        if q.score < MIN_TRANSCRIPT_QUALITY:
            return []
        dup = self._duplicate_of(seg, q.score, now)
        if dup is not None:
            return []  # another phone's copy already transcribed
        print(f"[WHISPER] started seg={dur_s:.2f}s q={q.score:.2f}", flush=True)
        t_whisper = time.time()
        print(
            f"[WHISPER] participant_id={seg.participant_id} "
            f"audio_duration_sent={dur_s:.2f}s phase=before",
            flush=True,
        )
        result = transcription.transcribe(samples, participant_id=seg.participant_id)
        whisper_ms = (time.time() - t_whisper) * 1000
        print(f"[WHISPER] finished in {whisper_ms:.0f}ms", flush=True)
        if result is None:
            print(
                f"[WHISPER] participant_id={seg.participant_id} "
                f"audio_duration_sent={dur_s:.2f}s returned_text='' "
                "language=unknown accepted=False reason=no_usable_result",
                flush=True,
            )
            return []
        text, confidence, language = result
        text = text.strip()
        if not text or confidence < MIN_TRANSCRIPT_CONFIDENCE:
            print(
                f"[TRANSCRIPT] discarded seg={dur_s:.2f}s "
                f"reason={'empty' if not text else 'low_confidence'} "
                f"confidence={confidence:.3f} quality={q.score:.3f}",
                flush=True,
            )
            return []
        print(
            f"[WHISPER] participant_id={seg.participant_id} "
            f"audio_duration_sent={dur_s:.2f}s returned_text={text!r} "
            f"language={language} confidence={confidence:.3f} "
            "accepted=True reason=accepted",
            flush=True,
        )
        # Guard against a second phone's near-identical text racing us.
        for d in self.decisions:
            if (
                selection.iou(seg.start, seg.end, d.start, d.end) >= _DUP_IOU
                and dedupe.is_duplicate(text, d.text)
            ):
                return []
        voice_speaker, voice_conf = voice.voice_profiles.identify(self._session_id, samples)
        source_speaker, source_conf, ambiguous = self.voter.vote(
            seg.participant_id, seg.start, seg.end, q.score, confidence
        )
        if voice_speaker != "unknown" and voice_conf >= source_conf:
            speaker, conf = voice_speaker, voice_conf
        elif voice.voice_profiles.has_profiles(self._session_id) and voice_speaker == "unknown":
            speaker, conf, ambiguous = "unknown", 0.0, True
        else:
            speaker, conf = source_speaker, source_conf
        overlapping = sorted({
            d.source_participant_id
            for d in self.decisions
            if d.source_participant_id
            and d.source_participant_id != seg.participant_id
            and selection.iou(seg.start, seg.end, d.start, d.end) >= _DUP_IOU
            and not dedupe.is_duplicate(text, d.text)
        } | {seg.participant_id})
        if len(overlapping) > 1:
            speaker, conf, ambiguous = "multiple", min(confidence, conf), True
        entry = self.fusion.add(
            speaker, seg.start, seg.end, text, min(confidence, conf), seg.participant_id,
            source_quality=q.score, ambiguous=ambiguous, language=language, is_final=seg.final,
            overlap_participants=overlapping if len(overlapping) > 1 else None,
        )
        new: list[FusedEntry] = []
        if entry is not None:
            new.append(entry)
            self.decisions.append(PipelineDecision(seg.start, seg.end, q.score, speaker, text, seg.participant_id))
        for emitted in new:
            print(
                "[TRANSCRIPT_EVENT] "
                f"participant_id={emitted.source_participant_id} "
                f"segment_id={emitted.id} text={emitted.text!r} "
                f"language={emitted.language} confidence={emitted.confidence:.3f} "
                f"is_final={emitted.is_final} source=live",
                flush=True,
            )
        print(f"[TRANSCRIPT] sent final={seg.final} entries={len(new)}", flush=True)
        print(
            f"[LAT] seg={dur_s:.2f}s whisper={whisper_ms:.0f}ms "
            f"total={(time.time() - t_in) * 1000:.0f}ms q={q.score:.2f} final={seg.final}",
            flush=True,
        )
        return new

    def ingest_chunk(self, participant_id: str, samples: np.ndarray, chunk_start: float) -> list[AudioSegment]:
        """VAD split of one uploaded chunk (no Whisper yet).

        Returns raw speech regions; accumulation into transcribable units
        happens in accumulate() so Whisper never sees tiny fragments.
        """
        out: list[AudioSegment] = []
        chunk_s = len(samples) / 16000
        regions = self.vad.segment(samples)
        speech_duration = sum(end - start for start, end in regions)
        print(
            f"[VAD] participant_id={participant_id} input_duration={chunk_s:.2f}s "
            f"detected_speech_duration={speech_duration:.2f}s "
            f"speech_detected={bool(regions)} regions={len(regions)} " +
            " ".join(f"{s:.2f}-{e:.2f}" for s, e in regions),
            flush=True,
        )
        for start, end in regions:
            part = samples[int(start * 16000):int(end * 16000)]
            if part.size < 1600 or audio_io.is_silent(part):
                continue
            final = (chunk_s - end) >= 0.25
            out.append(AudioSegment(participant_id, chunk_start + start, chunk_start + end, part, final=final))
        if out:
            self.last_voice[participant_id] = max(s.end for s in out)
        return out

    def accumulate(self, participant_id: str, segments: list[AudioSegment]) -> list[AudioSegment]:
        """Combine VAD regions into units worth transcribing.

        Whisper hallucinates on scraps ("thank you" on 0.3 s fragments), so
        speech accumulates per participant until MIN_SPEECH_S is buffered;
        a silence-closed buffer flushes early at FLUSH_MIN_S. Caps bound
        latency for long utterances. Returns transcribable units only.
        """
        buf = self.pending.get(participant_id)
        if buf is None:
            buf = {"samples": [], "start": 0.0, "end": 0.0, "final": False}
            self.pending[participant_id] = buf
        for seg in sorted(segments, key=lambda s: s.start):
            if not buf["samples"]:
                buf["start"] = seg.start
            buf["samples"].append(np.asarray(seg.samples, dtype=np.float32))
            buf["end"] = seg.end
            buf["final"] = seg.final
        buffered = sum(len(p) for p in buf["samples"]) / 16000
        trailing_silence = segments and segments[-1].final
        if buffered >= MIN_SPEECH_S or (trailing_silence and buffered >= FLUSH_MIN_S) or buffered >= MAX_BUFFER_S:
            combined = np.concatenate(buf["samples"]) if len(buf["samples"]) > 1 else buf["samples"][0]
            unit = AudioSegment(participant_id, buf["start"], buf["end"], combined, final=bool(trailing_silence))
            del self.pending[participant_id]
            return [unit]
        return []

    def flush_pending(self, participant_id: str, final: bool = True) -> list[AudioSegment]:
        """Emit buffered speech meeting FLUSH_MIN_S (e.g. speech that ran to
        chunk edges, then silence). Smaller scraps stay buffered for the
        next speech — Whisper never sees them."""
        buf = self.pending.get(participant_id)
        if not buf:
            return []
        total = sum(len(p) for p in buf["samples"]) / 16000
        if total < FLUSH_MIN_S:
            return []
        combined = np.concatenate(buf["samples"]) if len(buf["samples"]) > 1 else buf["samples"][0]
        unit = AudioSegment(participant_id, buf["start"], buf["end"], combined, final=final)
        del self.pending[participant_id]
        return [unit]

    def finalize_if_quiet(self, participant_id: str, chunk_end: float, quiet_s: float = 1.0) -> list[FusedEntry]:
        """A silent window after recent speech finalizes that speaker's open
        entries, so the UI stops showing them as partial. Returns flipped entries."""
        last = self.last_voice.get(participant_id, 0.0)
        if not last or chunk_end - last < quiet_s:
            return []
        flipped: list[FusedEntry] = []
        for entry in self.fusion.entries:
            if (not entry.is_final and not entry.ambiguous
                    and entry.source_participant_id == participant_id
                    and entry.end <= chunk_end):
                entry.is_final = True
                flipped.append(entry)
        return flipped

    def remove_participant(self, participant_id: str) -> None:
        """Drop one participant's pending state; history entries stay."""
        self.decisions = [d for d in self.decisions if d.speaker_id != participant_id]
        self.pending.pop(participant_id, None)
        if self.voter.last_speaker == participant_id:
            self.voter.reset()


class IntelligencePipeline:
    """All sessions. Thin async shell; heavy work runs off the event loop."""

    def __init__(self) -> None:
        self.sessions: dict[str, SessionIntelligence] = {}

    def for_session(self, session_id: str) -> SessionIntelligence:
        state = self.sessions.get(session_id)
        if state is None:
            state = SessionIntelligence()
            state._session_id = session_id
            self.sessions[session_id] = state
        return state

    async def ingest(
        self, session_id: str, participant_id: str, samples: np.ndarray, chunk_start: float
    ) -> list[FusedEntry]:
        state = self.for_session(session_id)
        segments = state.ingest_chunk(participant_id, samples, chunk_start)
        # Accumulate speech until a transcribable unit (MIN_SPEECH_S) or a
        # silence-closed flush; tiny fragments never reach Whisper.
        units = state.accumulate(participant_id, segments)
        out: list[FusedEntry] = []
        seen: set[str] = set()

        def emit(entry: FusedEntry) -> None:
            if entry.id not in seen:
                seen.add(entry.id)
                out.append(entry)

        for seg in units:
            # process_segment returns the affected entry per segment; a merge
            # may return the SAME entry twice — emit each fused entry once.
            for entry in await asyncio.to_thread(state.process_segment, seg):
                emit(entry)
        if not segments:
            # Silent window: transcribe any worthwhile buffered speech first,
            # then finalize this speaker's still-open entries.
            chunk_end = chunk_start + len(samples) / 16000
            for seg in state.flush_pending(participant_id, final=True):
                for entry in await asyncio.to_thread(state.process_segment, seg):
                    emit(entry)
            for entry in state.finalize_if_quiet(participant_id, chunk_end):
                emit(entry)
        return out

    def transcript(self, session_id: str) -> list[dict]:
        state = self.sessions.get(session_id)
        return state.fusion.to_dicts() if state else []

    def remove_participant(self, session_id: str, participant_id: str) -> None:
        state = self.sessions.get(session_id)
        if state:
            state.remove_participant(participant_id)
            voice.voice_profiles.remove_participant(session_id, participant_id)

    def end_session(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)
        voice.voice_profiles.clear_session(session_id)


pipeline = IntelligencePipeline()
