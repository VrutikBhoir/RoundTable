"""faster-whisper transcription with multilingual language detection.

Transcription only — no translation is requested.
"""

import math
import os
import threading

import numpy as np

MODEL_NAME = os.getenv("WHISPER_MODEL", "small")
_DEVICE = os.getenv("WHISPER_DEVICE", "auto")
_COMPUTE = os.getenv("WHISPER_COMPUTE_TYPE", "auto")
_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "auto").strip().lower() or "auto"

_model = None
_model_key: tuple | None = None
_lock = threading.Lock()
_load_error: str | None = None
# At most 2 concurrent Whisper inferences: transcribing is the expensive
# step, and unbounded parallel jobs would thrash GPU/CPU. transcribe()
# always runs inside a worker thread, so a blocking acquire is safe.
_infer_slots = threading.Semaphore(2)
_MAX_NO_SPEECH_PROB = 0.6
_MAX_COMPRESSION_RATIO = 2.4
_MIN_AVG_LOGPROB = -1.0


def resolve_device() -> str:
    if _DEVICE != "auto":
        return _DEVICE
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def resolve_compute_type(device: str) -> str:
    if _COMPUTE != "auto":
        return _COMPUTE
    return "float16" if device == "cuda" else "int8"


def resolve_language() -> str:
    """Configured language, or ``auto`` for faster-whisper detection."""
    return _LANGUAGE


def get_model():
    """Cached model. Raises RuntimeError with a clear message on failure."""
    global _model, _model_key, _load_error
    key = (MODEL_NAME, resolve_device(), resolve_compute_type(resolve_device()))
    with _lock:
        if _model is not None and _model_key == key:
            return _model
        try:
            from faster_whisper import WhisperModel

            device = resolve_device()
            _model = WhisperModel(MODEL_NAME, device=device, compute_type=resolve_compute_type(device))
            _model_key = key
            _load_error = None
            return _model
        except Exception as exc:
            _model = None
            _load_error = str(exc)[:300]
            raise RuntimeError(f"whisper model '{MODEL_NAME}' failed to load: {_load_error}") from exc


def warmup() -> bool:
    """Preload the model (e.g. at server startup). Never raises; False on failure."""
    try:
        get_model()
        print(f"[whisper] model '{MODEL_NAME}' ready (device={resolve_device()}, language={resolve_language()})", flush=True)
        return True
    except Exception as exc:
        print(f"[whisper] warmup failed, will retry lazily: {exc}", flush=True)
        return False


def transcribe(samples: np.ndarray, participant_id: str | None = None) -> tuple[str, float, str] | None:
    """Transcribe 16 kHz mono float32 speech without translating it."""
    try:
        model = get_model()
        kwargs: dict = {
            "beam_size": 3,
            "temperature": 0.0,
            "vad_filter": False,  # we already ran our own VAD
            "condition_on_previous_text": False,  # segments are independent; avoids cross-speaker repetition
        }
        if _LANGUAGE != "auto":
            kwargs["language"] = _LANGUAGE
        segments, _info = None, None
        with _infer_slots:
            segments, _info = model.transcribe(samples.astype(np.float32), **kwargs)
        texts: list[str] = []
        logprobs: list[float] = []
        for seg in segments:
            t = (seg.text or "").strip()
            if not t:
                continue
            no_speech_prob = getattr(seg, "no_speech_prob", 0.0) or 0.0
            compression_ratio = getattr(seg, "compression_ratio", 0.0) or 0.0
            avg_logprob = getattr(seg, "avg_logprob", None)
            if (
                no_speech_prob > _MAX_NO_SPEECH_PROB
                or compression_ratio > _MAX_COMPRESSION_RATIO
                or (avg_logprob is not None and avg_logprob < _MIN_AVG_LOGPROB)
            ):
                print(
                    "[WHISPER_SEGMENT] "
                    f"participant_id={participant_id or 'unknown'} "
                    f"text={t!r} language=pending "
                    f"no_speech_prob={no_speech_prob:.3f} "
                    f"compression_ratio={compression_ratio:.3f} "
                    f"avg_logprob={avg_logprob} accepted=False "
                    "reason=quality_metadata",
                    flush=True,
                )
                continue
            print(
                "[WHISPER_SEGMENT] "
                f"participant_id={participant_id or 'unknown'} "
                f"text={t!r} language=pending "
                f"no_speech_prob={no_speech_prob:.3f} "
                f"compression_ratio={compression_ratio:.3f} "
                f"avg_logprob={avg_logprob} accepted=True reason=accepted",
                flush=True,
            )
            texts.append(t)
            logprobs.append(avg_logprob if avg_logprob is not None else -1.0)
        text = " ".join(texts).strip()
        if not text:
            return None
        confidence = math.exp(sum(logprobs) / len(logprobs)) if logprobs else 0.0
        detected = getattr(_info, "language", None) or (_LANGUAGE if _LANGUAGE != "auto" else "unknown")
        return text, max(0.0, min(1.0, confidence)), str(detected)
    except Exception:
        return None
