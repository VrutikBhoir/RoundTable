"""Session-scoped speaker enrollment and verification.

The embedding model is optional at runtime because it downloads model weights.
When it is unavailable, enrollment fails explicitly and the pipeline leaves
the speaker unknown; it never substitutes a device id for a voice identity.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from . import audio_io
from .vad import SileroVAD

VOICE_ENROLLMENT_DURATION_S = 15.0
MIN_ENROLLMENT_SPEECH_S = 8.0
SIMILARITY_THRESHOLD = 0.55


@dataclass
class VoiceProfile:
    participant_id: str
    embedding: np.ndarray
    speech_seconds: float


class VoiceProfileStore:
    def __init__(self) -> None:
        self._profiles: dict[str, dict[str, VoiceProfile]] = {}
        self._model = None
        self._model_error: str | None = None
        self._lock = threading.Lock()
        self._vad = SileroVAD()

    @property
    def model_available(self) -> bool:
        self._load_model()
        return self._model is not None

    @property
    def model_error(self) -> str | None:
        self._load_model()
        return self._model_error

    def _load_model(self) -> None:
        if self._model is not None or self._model_error is not None:
            return
        try:
            from speechbrain.inference.speaker import EncoderClassifier
            from speechbrain.utils.fetching import LocalStrategy

            self._model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=".cache/speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": "cpu"},
                # Windows without Developer Mode cannot create the symlinks
                # used by SpeechBrain's default cache strategy.
                local_strategy=LocalStrategy.COPY,
            )
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            self._model_error = str(exc)[:300]

    def enroll(self, session_id: str, participant_id: str, samples: np.ndarray) -> float:
        samples = np.asarray(samples, dtype=np.float32)
        speech_seconds = sum(end - start for start, end in self._vad.segment(samples))
        if speech_seconds < MIN_ENROLLMENT_SPEECH_S:
            total_seconds = samples.size / 16000
            raise ValueError(
                f"insufficient_speech:{speech_seconds:.2f}:{total_seconds:.2f}:{MIN_ENROLLMENT_SPEECH_S:.2f}"
            )
        embedding = self._embed(samples)
        with self._lock:
            self._profiles.setdefault(session_id, {})[participant_id] = VoiceProfile(
                participant_id, embedding, speech_seconds
            )
        return speech_seconds

    def identify(self, session_id: str, samples: np.ndarray) -> tuple[str, float]:
        profiles = self._profiles.get(session_id, {})
        if not profiles:
            return "unknown", 0.0
        try:
            probe = self._embed(samples)
        except (RuntimeError, ValueError):
            return "unknown", 0.0
        best_id, best_score = "unknown", 0.0
        for participant_id, profile in profiles.items():
            score = float(np.dot(probe, profile.embedding))
            if score > best_score:
                best_id, best_score = participant_id, score
        return (best_id, best_score) if best_score >= SIMILARITY_THRESHOLD else ("unknown", best_score)

    def status(self, session_id: str, participant_id: str) -> str:
        if participant_id in self._profiles.get(session_id, {}):
            return "ready"
        if not self.model_available:
            return "unavailable"
        return "not_enrolled"

    def has_profiles(self, session_id: str) -> bool:
        return bool(self._profiles.get(session_id))

    def remove_participant(self, session_id: str, participant_id: str) -> None:
        self._profiles.get(session_id, {}).pop(participant_id, None)

    def clear_session(self, session_id: str) -> None:
        self._profiles.pop(session_id, None)

    def _embed(self, samples: np.ndarray) -> np.ndarray:
        self._load_model()
        if self._model is None:
            raise RuntimeError(f"voice embedding model unavailable: {self._model_error or 'unknown error'}")
        import torch

        waveform = torch.from_numpy(audio_io.normalize(samples)).unsqueeze(0)
        with torch.no_grad():
            vector = self._model.encode_batch(waveform).squeeze().cpu().numpy().astype(np.float32)
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-8:
            raise ValueError("voice embedding was empty")
        return vector / norm


voice_profiles = VoiceProfileStore()
