from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev

from symusic import Score


@dataclass(frozen=True)
class CandidateMetrics:
    score: float
    rejected: bool
    reason: str
    notes: int
    duration_ticks: int
    pitch_range: int
    unique_pitches: int
    density: float
    repeated_pitch_ratio: float
    repeated_ngram_ratio: float
    silence_ratio: float
    bars: int


class MusicQualityScorer:
    def __init__(
        self,
        min_notes: int = 24,
        min_duration_ticks: int = 256,
        max_repeated_pitch_ratio: float = 0.42,
        max_repeated_ngram_ratio: float = 0.34,
    ):
        self.min_notes = min_notes
        self.min_duration_ticks = min_duration_ticks
        self.max_repeated_pitch_ratio = max_repeated_pitch_ratio
        self.max_repeated_ngram_ratio = max_repeated_ngram_ratio

    def evaluate(self, tokens: list[int], score: Score) -> CandidateMetrics:
        notes = [
            note
            for track in score.tracks
            for note in track.notes
            if int(note.duration) > 0
        ]
        note_count = len(notes)
        bars = tokens.count(4)

        if not notes:
            return CandidateMetrics(0.0, True, "no notes", 0, 0, 0, 0, 0.0, 1.0, 1.0, 1.0, bars)

        start_tick = min(int(note.time) for note in notes)
        end_tick = max(int(note.time + note.duration) for note in notes)
        duration_ticks = max(1, end_tick - start_tick)
        pitches = [int(note.pitch) for note in notes]
        pitch_range = max(pitches) - min(pitches)
        unique_pitches = len(set(pitches))
        density = note_count / max(1.0, duration_ticks / max(1, int(score.ticks_per_quarter)))
        repeated_pitch_ratio = _max_run_ratio(pitches)
        repeated_ngram_ratio = _repeated_ngram_ratio(tokens, 5)
        silence_ratio = _silence_ratio(notes, duration_ticks, start_tick)
        velocity_values = [int(note.velocity) for note in notes]
        velocity_spread = pstdev(velocity_values) if len(velocity_values) > 1 else 0.0

        reason = ""
        rejected = False
        if note_count < self.min_notes:
            rejected, reason = True, "too few notes"
        elif duration_ticks < self.min_duration_ticks:
            rejected, reason = True, "too short"
        elif unique_pitches < 7:
            rejected, reason = True, "pitch set too narrow"
        elif pitch_range < 10:
            rejected, reason = True, "pitch range too narrow"
        elif repeated_pitch_ratio > self.max_repeated_pitch_ratio:
            rejected, reason = True, "stuck pitch pattern"
        elif repeated_ngram_ratio > self.max_repeated_ngram_ratio:
            rejected, reason = True, "token loop"
        elif density < 0.35:
            rejected, reason = True, "too sparse"
        elif density > 14.0:
            rejected, reason = True, "too dense"
        elif silence_ratio > 0.62:
            rejected, reason = True, "too much silence"

        quality = 0.0
        quality += _target_score(note_count, 120, 520) * 20.0
        quality += _target_score(density, 1.8, 7.5) * 18.0
        quality += _target_score(pitch_range, 18, 52) * 14.0
        quality += _target_score(unique_pitches, 12, 44) * 10.0
        quality += _target_score(bars, 2, 24) * 8.0
        quality += min(1.0, velocity_spread / 18.0) * 6.0
        quality -= repeated_pitch_ratio * 16.0
        quality -= repeated_ngram_ratio * 20.0
        quality -= silence_ratio * 10.0

        if rejected:
            quality -= 100.0

        return CandidateMetrics(
            score=round(quality, 3),
            rejected=rejected,
            reason=reason or "accepted",
            notes=note_count,
            duration_ticks=duration_ticks,
            pitch_range=pitch_range,
            unique_pitches=unique_pitches,
            density=round(density, 3),
            repeated_pitch_ratio=round(repeated_pitch_ratio, 3),
            repeated_ngram_ratio=round(repeated_ngram_ratio, 3),
            silence_ratio=round(silence_ratio, 3),
            bars=bars,
        )


def _target_score(value: float, low: float, high: float) -> float:
    if value < low:
        return max(0.0, value / low)
    if value > high:
        return max(0.0, 1.0 - ((value - high) / high))
    midpoint = (low + high) / 2.0
    half_width = (high - low) / 2.0
    return 0.75 + 0.25 * (1.0 - abs(value - midpoint) / half_width)


def _max_run_ratio(values: list[int]) -> float:
    if not values:
        return 1.0
    longest = 1
    current = 1
    for left, right in zip(values, values[1:]):
        if left == right:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    return longest / len(values)


def _repeated_ngram_ratio(tokens: list[int], n: int) -> float:
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    return 1.0 - (len(set(ngrams)) / len(ngrams))


def _silence_ratio(notes: list, duration_ticks: int, start_tick: int) -> float:
    if duration_ticks <= 0:
        return 1.0
    intervals = sorted(
        (max(0, int(note.time) - start_tick), max(0, int(note.time + note.duration) - start_tick))
        for note in notes
    )
    covered = 0
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            covered += max(0, current_end - current_start)
            current_start, current_end = start, end
    covered += max(0, current_end - current_start)
    return max(0.0, min(1.0, 1.0 - (covered / duration_ticks)))
