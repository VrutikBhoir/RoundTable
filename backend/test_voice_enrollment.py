import numpy as np
import pytest

from backend.audio_intelligence.voice import VoiceProfileStore


class StubVAD:
    def __init__(self, speech_seconds: float):
        self.speech_seconds = speech_seconds

    def segment(self, samples: np.ndarray) -> list[tuple[float, float]]:
        return [(0.0, self.speech_seconds)]


def make_store(speech_seconds: float) -> VoiceProfileStore:
    store = VoiceProfileStore()
    store._vad = StubVAD(speech_seconds)
    store._embed = lambda samples: np.array([1.0, 0.0], dtype=np.float32)
    return store


@pytest.mark.parametrize("speech_seconds", [12.0, 8.0])
def test_fifteen_second_enrollment_accepts_at_least_eight_seconds(speech_seconds: float):
    store = make_store(speech_seconds)
    detected = store.enroll("session", "participant", np.zeros(15 * 16000, dtype=np.float32))
    assert detected == speech_seconds


@pytest.mark.parametrize("speech_seconds", [7.0, 0.0])
def test_fifteen_second_enrollment_rejects_less_than_eight_seconds(speech_seconds: float):
    store = make_store(speech_seconds)
    with pytest.raises(ValueError, match=rf"^insufficient_speech:{speech_seconds:.2f}:15.00:8.00$"):
        store.enroll("session", "participant", np.zeros(15 * 16000, dtype=np.float32))
