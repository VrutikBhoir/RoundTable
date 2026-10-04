"""Chronological fused transcript store, one per session."""

import uuid

from .models import FusedEntry

MERGE_GAP_S = 1.5  # nearby same-speaker fragments join one utterance


class FusionStore:
    def __init__(self) -> None:
        self.entries: list[FusedEntry] = []

    def add(
        self,
        speaker_id: str,
        start: float,
        end: float,
        text: str,
        confidence: float,
        source_participant_id: str,
        ambiguous: bool = False,
        language: str = "unknown",
        is_final: bool = True,
        source_quality: float = 0.0,
        overlap_participants: list[str] | None = None,
    ) -> FusedEntry | None:
        text = text.strip()
        if not text:
            return None
        # Merge into the temporally nearest preceding same-speaker entry
        # (insertion order is not guaranteed, so don't use entries[-1]).
        best = None
        best_gap = None
        for candidate in self.entries:
            if candidate.ambiguous or ambiguous:
                continue
            if candidate.speaker_id != speaker_id or speaker_id in ("unknown", "multiple"):
                continue
            gap = start - candidate.end
            if 0 <= gap <= MERGE_GAP_S and (best_gap is None or gap < best_gap):
                best, best_gap = candidate, gap
        if best is not None:
            best.text = f"{best.text} {text}".strip()
            best.end = max(best.end, end)
            best.confidence = round((best.confidence + confidence) / 2, 3)
            if best.language in ("unknown", "") and language not in ("unknown", ""):
                best.language = language
            # Finality follows the latest fragment: an utterance stays
            # partial until a fragment ends in silence.
            best.is_final = is_final
            return best
        entry = FusedEntry(
            id=f"t-{uuid.uuid4().hex[:8]}",
            start=start,
            end=end,
            speaker_id=speaker_id,
            text=text,
            confidence=round(confidence, 3),
            source_participant_id=source_participant_id,
            source_quality=source_quality,
            ambiguous=ambiguous,
            language=language,
            overlap_participants=overlap_participants,
            is_final=is_final,
        )
        self.entries.append(entry)
        self.entries.sort(key=lambda e: (e.start, e.end))
        return entry

    def ordered(self) -> list[FusedEntry]:
        return sorted(self.entries, key=lambda e: (e.start, e.end))

    def to_dicts(self) -> list[dict]:
        return [e.to_dict() for e in self.ordered()]

    def clear(self) -> None:
        self.entries.clear()
