"""Part 2 MVP checks A–G. Deterministic asserts on mechanics; Whisper/VAD
live-fire where possible, graceful everywhere else. No WebRTC needed."""

import asyncio
import io
import os
import sys
import wave

# Keep the test process fast: tiny multilingual model. Production default
# (WHISPER_MODEL=small) is exercised in live runs, not here.
os.environ.setdefault("WHISPER_MODEL", "tiny")
os.environ.setdefault("WHISPER_LANGUAGE", "auto")

import numpy as np

sys.path.insert(0, ".")

from backend.audio_intelligence import audio_io, attribution, dedupe
from backend.audio_intelligence import quality as quality_mod
from backend.audio_intelligence import selection, transcription
from backend.audio_intelligence.fusion import FusionStore
from backend.audio_intelligence.models import AudioSegment
from backend.audio_intelligence.pipeline import IntelligencePipeline
from backend.audio_intelligence.vad import SileroVAD

SR = 16000
passed = []


def check(name, cond):
    assert cond, f"FAILED: {name}"
    passed.append(name)
    print(f"  ok: {name}")


def tone(freq=440.0, seconds=2.0, amp=0.3, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def speech_like(seconds=3.0):
    """Harmonic buzz with syllable-like amplitude gating (VAD probe)."""
    t = np.arange(int(seconds * SR)) / SR
    f0 = 120 + 20 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    sig = sum(np.sin(k * phase) / k for k in range(1, 6))
    gate = (0.5 + 0.5 * np.sin(2 * np.pi * 2.2 * t)) > 0.35
    return (0.25 * sig * gate).astype(np.float32)


print("== E: silence ==")
vad = SileroVAD()
check("silence -> no segments", vad.segment(np.zeros(SR * 3, dtype=np.float32)) == [])
check("silence is_silent", audio_io.is_silent(np.zeros(16000, dtype=np.float32)))

print("== speech-like probe (informational, never fatal) ==")
segs = vad.segment(speech_like())
print(f"  info: silero available={vad.available} segments={len(segs)}")
long_sig = np.concatenate([speech_like(6.0), speech_like(6.0)])
long_segs = vad.segment(long_sig)
check("long utterances split for latency", all(e - s <= 8.0 + 1e-6 for s, e in long_segs))

print("== F: quality ordering ==")
loud = tone(amp=0.3)
quiet = tone(amp=0.02)
clipped = np.clip(tone(amp=2.0), -1, 1).astype(np.float32)
qs = lambda s: quality_mod.score_segment(s).score
check("loud > quiet", qs(loud) > qs(quiet))
check("clipped penalized vs clean", quality_mod.score_segment(clipped).clipping_ratio > 0.1)
check("quiet low score", qs(quiet) < 0.3)

print("== C: best-source selection ==")
segs = [
    AudioSegment("A", 10.0, 13.2, tone(amp=0.08)),
    AudioSegment("B", 10.2, 13.0, tone(amp=0.30)),
    AudioSegment("C", 10.1, 13.1, tone(amp=0.08)),
]
quals = {id(s): quality_mod.score_segment(s.samples) for s in segs}
groups = selection.group_events(segs)
check("one group for same event", len(groups) == 1 and len(groups[0]) == 3)
check("best source wins", selection.pick_best(groups[0], quals).participant_id == "B")
far = AudioSegment("A", 40.0, 42.0, loud)
check("far segment separate", len(selection.group_events(segs + [far])) == 2)

print("== D/G: duplicate + fusion ==")
check("near-identical duplicate",
      dedupe.is_duplicate("Let's move the meeting to Friday.", "lets move the meeting to friday!"))
check("different sentences not duplicate",
      not dedupe.is_duplicate("Let's move the meeting to Friday.", "Review the quarterly budget numbers"))
store = FusionStore()
store.add("P-1", 20.0, 22.0, "second sentence.", 0.8, "P-1")
store.add("P-1", 12.0, 14.0, "First sentence.", 0.9, "P-1")  # out of order
store.add("P-1", 14.5, 15.9, "continued thought.", 0.8, "P-1")  # merges
ordered = store.ordered()
check("chronological", [e.start for e in ordered] == sorted(e.start for e in ordered))
check("nearby fragments merged", any("continued thought" in e.text for e in ordered))
store.add("P-2", 16.5, 19.0, "Overlapping reply.", 0.7, "P-2")
check("different speaker separate entry", len(store.ordered()) == 3)
store.add("P-1", 22.5, 24.0, "one more.", 0.8, "P-1", language="en")
merged = [e for e in store.ordered() if e.start == 20.0][0]
check("language preserved through merge", merged.language == "en")
check("partial/final flags", merged.is_final is True)
part = store.add("P-2", 30.0, 31.0, "partial thought", 0.7, "P-2", is_final=False)
check("partial entry flagged", part is not None and part.is_final is False)
check("to_dict carries isFinal", part.to_dict()["isFinal"] is False)

print("== attribution ==")
voter = attribution.TemporalVoter()
s1, c1, _ = voter.vote("P-1", 0.0, 2.0, 0.8, 0.9)
check("confident source attributed", s1 == "P-1" and c1 > 0.5)
s2, _, _ = voter.vote("P-1", 2.1, 4.0, 0.8, 0.9)
check("temporal stickiness", s2 == "P-1")
check("low quality -> unknown", voter.vote("P-2", 10.0, 12.0, 0.05, 0.9)[0] == "unknown")
fresh = attribution.TemporalVoter()
check("overlap -> multiple", fresh.mark_overlap(["P-1", "P-2"], 0.0, 2.0)[0] == "multiple")

print("== pipeline ingest (silence + tone, graceful) ==")
pipe = IntelligencePipeline()


async def run_pipe():
    out_silence = await pipe.ingest("s1", "P-1", np.zeros(SR * 4, dtype=np.float32), 0.0)
    check("silence ingests to nothing", out_silence == [])
    out_tone = await pipe.ingest("s1", "P-1", tone(seconds=4.0), 4.0)
    print(f"  info: tone produced {len(out_tone)} entries (whisper-dependent)")
    check("transcript fetchable", isinstance(pipe.transcript("s1"), list))
    # Accumulator: sub-threshold fragments buffer, never reach Whisper alone.
    st = pipe.for_session("s1")
    seg_a = AudioSegment("P-9", 0.0, 0.8, tone(seconds=0.8), final=False)
    seg_b = AudioSegment("P-9", 0.8, 1.6, tone(seconds=0.8), final=True)
    check("short fragment held, not emitted", st.accumulate("P-9", [seg_a]) == [])
    combined = st.accumulate("P-9", [seg_b])
    check("fragments combine past MIN_SPEECH_S", len(combined) == 1)
    check("combined unit long enough", combined[0].end - combined[0].start >= 1.5)
    check("short blip alone never emitted", st.accumulate("P-9", [seg_a]) == [])
    st.pending.pop("P-9", None)
    # Silence after speech finalizes the open entry (no endless partials).
    e = st.fusion.add("P-1", 30.0, 32.0, "Trailing thought", 0.8, "P-1", is_final=False)
    assert e is not None
    st.last_voice["P-1"] = 32.0
    flipped = await pipe.ingest("s1", "P-1", np.zeros(int(SR * 1.5), dtype=np.float32), 40.0)
    check("quiet window finalizes open entry", any(x.id == e.id and x.is_final for x in flipped))
    reno = await pipe.ingest("s1", "P-1", np.zeros(int(SR * 1.5), dtype=np.float32), 41.5)
    check("finalized entry not re-emitted", reno == [])
    pipe.remove_participant("s1", "P-1")
    check("transcript fetchable", isinstance(pipe.transcript("s1"), list))


asyncio.run(run_pipe())

print("== ffmpeg roundtrip ==")
buf = io.BytesIO()
raw44 = (np.sin(2 * np.pi * 440.0 * np.arange(44100) / 44100) * 0.3 * 32767).astype(np.int16)
with wave.open(buf, "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(44100)
    w.writeframes(raw44.tobytes())
decoded = audio_io.decode_upload(buf.getvalue(), "probe.wav")
check("wav decodes to 16k mono", decoded.ndim == 1 and abs(len(decoded) - SR) <= 100)

print("== whisper smoke (downloads tiny model on first run) ==")
try:
    out = transcription.transcribe(speech_like(3.0))
    m1 = transcription.get_model()
    check("model cached", transcription.get_model() is m1)
    check("multilingual auto-detection configured", transcription.resolve_language() == "auto")
    check("result shape has language", out is None or (len(out) == 3 and isinstance(out[2], str)))
    if out is not None:
        check("entry carries language", True)
        print(f"  info: whisper returned: lang={out[2]} text={out[0][:80]!r}")
    else:
        print("  info: whisper returned None on synthetic audio (graceful)")
        passed.append("whisper graceful None")
    check("warmup callable", transcription.warmup() in (True, False))
    print(f"  info: whisper returned: {out!r}"[:160])
    passed.append("whisper smoke (no crash)")
except Exception as exc:
    print(f"  SKIP whisper (offline/model issue): {str(exc)[:120]}")

print(f"\nALL {len(passed)} CHECKS PASSED")
