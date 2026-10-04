from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from backend.audio_intelligence import transcription


def _run_with_segments(*segments):
    model = SimpleNamespace(
        transcribe=lambda _samples, **_kwargs: (
            iter(segments),
            SimpleNamespace(language="en"),
        )
    )
    with patch.object(transcription, "get_model", return_value=model):
        return transcription.transcribe(np.zeros(16000, dtype=np.float32))


def test_discards_empty_and_hallucination_shaped_segments():
    result = _run_with_segments(
        SimpleNamespace(text="   ", no_speech_prob=0.0, compression_ratio=0.0, avg_logprob=0.0),
        SimpleNamespace(text="repeated words", no_speech_prob=0.9, compression_ratio=1.0, avg_logprob=0.0),
        SimpleNamespace(text="compressed words", no_speech_prob=0.0, compression_ratio=3.0, avg_logprob=0.0),
        SimpleNamespace(text="unlikely words", no_speech_prob=0.0, compression_ratio=1.0, avg_logprob=-1.2),
    )

    assert result is None


def test_keeps_a_segment_with_acceptable_quality_metadata():
    result = _run_with_segments(
        SimpleNamespace(
            text="Hello, this is a real Roundtable test.",
            no_speech_prob=0.05,
            compression_ratio=1.1,
            avg_logprob=-0.2,
        )
    )

    assert result is not None
    assert result[0] == "Hello, this is a real Roundtable test."
    assert result[2] == "en"
    assert result[1] > 0.3
