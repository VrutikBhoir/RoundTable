"""Shared types for the audio-intelligence pipeline.

The rest of the application only needs these shapes — never VAD/Whisper
internals. All audio is mono float32 at 16 kHz unless stated otherwise.
"""

from dataclasses import dataclass

SAMPLE_RATE = 16000


@dataclass
class AudioSegment:
    participant_id: str
    start: float  # seconds, session-relative
    end: float
    samples: object  # numpy float32 array, set at runtime
    source: str = "live"  # "live" | "file"
    # True when trailing silence closed the utterance (partial otherwise).
    final: bool = True


@dataclass
class QualityScore:
    rms: float
    snr_db: float
    clipping_ratio: float
    score: float  # 0..1, higher is clearer


@dataclass
class TranscriptSegment:
    participant_id: str
    start: float
    end: float
    text: str
    confidence: float
    quality: float
    language: str = "unknown"


@dataclass
class FusedEntry:
    id: str
    start: float
    end: float
    speaker_id: str  # participant id, "unknown", or "multiple"
    text: str
    confidence: float
    source_participant_id: str
    source_quality: float = 0.0
    ambiguous: bool = False
    language: str = "unknown"
    overlap_participants: list[str] | None = None
    # False = utterance may continue (UI updates text in place).
    is_final: bool = True

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "start": self.start,
            "end": self.end,
            "speaker_id": self.speaker_id,
            "text": self.text,
            "confidence": round(self.confidence, 3),
            "source_participant_id": self.source_participant_id,
            "quality": round(self.source_quality, 3),
            "ambiguous": self.ambiguous,
            "language": self.language,
            "overlap_participants": self.overlap_participants or [],
            "isFinal": self.is_final,
        }


@dataclass
class PipelineDecision:
    """Recent outcome used for duplicate suppression across phones."""

    start: float
    end: float
    quality: float
    speaker_id: str
    text: str
    source_participant_id: str = ""
