"""Audio decoding / normalization. Everything downstream is mono float32 16 kHz."""

import base64
import shutil
import subprocess
import io
import wave

import numpy as np

from .models import SAMPLE_RATE

_SILENCE_RMS = 0.004


def decode_base64_pcm(b64: str) -> np.ndarray:
    """Frontend posts Float32 16 kHz mono PCM as base64."""
    try:
        raw = base64.b64decode(b64, validate=True)
    except Exception as exc:
        raise ValueError("invalid base64 audio") from exc
    if len(raw) % 4 != 0 or len(raw) == 0:
        raise ValueError("invalid PCM payload")
    samples = np.frombuffer(raw, dtype=np.float32).astype(np.float32)
    if not np.all(np.isfinite(samples)):
        raise ValueError("non-finite audio samples")
    return samples


def _ffmpeg_path() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise RuntimeError("ffmpeg is required to decode recorded files but was not found on PATH")
    return path


def decode_upload(data: bytes, filename: str) -> np.ndarray:
    """Decode uploaded audio to mono float32 16 kHz.

    PyAV handles browser WebM/Opus uploads directly. ffmpeg remains the
    fallback for formats that PyAV cannot decode.
    """
    if filename.lower().endswith(".wav") and not shutil.which("ffmpeg"):
        try:
            with wave.open(io.BytesIO(data), "rb") as wav:
                channels = wav.getnchannels()
                width = wav.getsampwidth()
                rate = wav.getframerate()
                frames = wav.readframes(wav.getnframes())
            if width != 2 or channels < 1 or rate <= 0:
                raise ValueError("unsupported WAV format without ffmpeg")
            raw = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
            if channels > 1:
                raw = raw.reshape(-1, channels).mean(axis=1)
            if rate != SAMPLE_RATE:
                target_size = max(1, round(raw.size * SAMPLE_RATE / rate))
                raw = np.interp(
                    np.linspace(0, raw.size - 1, target_size),
                    np.arange(raw.size),
                    raw,
                )
            return raw.astype(np.float32)
        except (EOFError, OSError, ValueError) as exc:
            raise ValueError("could not decode WAV audio") from exc
    try:
        import av

        decoded: list[np.ndarray] = []
        resampler = av.AudioResampler(format="flt", layout="mono", rate=SAMPLE_RATE)
        with av.open(io.BytesIO(data)) as container:
            for frame in container.decode(audio=0):
                converted = resampler.resample(frame)
                frames = converted if isinstance(converted, list) else [converted]
                decoded.extend(
                    np.asarray(item.to_ndarray(), dtype=np.float32).reshape(-1)
                    for item in frames
                )
        if decoded:
            samples = np.concatenate(decoded).astype(np.float32)
            if samples.size and np.all(np.isfinite(samples)):
                return samples
    except (ImportError, OSError, RuntimeError, ValueError):
        pass
    try:
        proc = subprocess.run(
            [_ffmpeg_path(), "-hide_banner", "-loglevel", "error",
             "-i", "pipe:0", "-ac", "1", "-ar", str(SAMPLE_RATE),
             "-f", "f32le", "pipe:1"],
            input=data,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError("audio decode timed out") from exc
    except RuntimeError as exc:
        raise RuntimeError(
            "no browser audio decoder is available; install PyAV or ffmpeg"
        ) from exc
    if proc.returncode != 0 or not proc.stdout:
        raise ValueError("could not decode audio file (unsupported or corrupt?)")
    samples = np.frombuffer(proc.stdout, dtype=np.float32).astype(np.float32)
    if samples.size == 0 or not np.all(np.isfinite(samples)):
        raise ValueError("decoded audio is empty or invalid")
    return samples


def normalize(samples: np.ndarray, peak: float = 0.9) -> np.ndarray:
    """Peak-normalize; leaves silence alone so VAD still sees quiet."""
    max_abs = float(np.max(np.abs(samples))) if samples.size else 0.0
    if max_abs < 1e-6:
        return samples
    return (samples * min(1.0, peak / max_abs)).astype(np.float32)


def rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))


def is_silent(samples: np.ndarray, threshold: float = _SILENCE_RMS) -> bool:
    return rms(samples) < threshold
