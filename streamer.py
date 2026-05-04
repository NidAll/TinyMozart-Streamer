from __future__ import annotations

import math
import queue
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field, replace
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame
import sounddevice as sd
import soundfile as sf
from symusic import Score

from music_quality import CandidateMetrics, MusicQualityScorer
from tinymozart_model import BLOCK_SIZE, GenerationSettings, TinyMozartGenerator


SAMPLE_RATE = 44_100
STREAMER_VERSION = "quality-modes-v7"


@dataclass
class StreamStatus:
    running: bool = False
    loading: bool = False
    device: str = "unknown"
    chunks_generated: int = 0
    chunks_played: int = 0
    last_message: str = "Idle"
    error: str | None = None
    started_at: float | None = None
    best_score: float = 0.0
    rejected_candidates: int = 0
    last_metrics: str = ""
    quality_mode: str = "Balanced"


@dataclass
class StreamConfig:
    settings: GenerationSettings = field(default_factory=GenerationSettings)
    context_tokens: int = BLOCK_SIZE
    queue_size: int = 2
    quality_mode: str = "Balanced"


class TinyMozartStreamer:
    def __init__(self, config: StreamConfig | None = None):
        self.config = config or StreamConfig()
        self.status = StreamStatus()
        self._generator: TinyMozartGenerator | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._piece_queue: queue.Queue[tuple[Path, Score]] = queue.Queue(
            self.config.queue_size
        )
        self._lock = threading.Lock()
        self._temp_dir: TemporaryDirectory[str] | None = None
        self._scorer = MusicQualityScorer()

    def start(self) -> None:
        if self.is_running:
            return
        self._stop.clear()
        self._clear_queue()
        with self._lock:
            self.status = StreamStatus(running=True, loading=True, last_message="Loading model...")
        self._thread = threading.Thread(target=self._run, name="tinymozart-stream", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._clear_queue()
        with self._lock:
            self.status.running = False
            self.status.loading = False
            self.status.last_message = "Stopping..."

    @property
    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive() and not self._stop.is_set())

    def snapshot(self) -> StreamStatus:
        with self._lock:
            return StreamStatus(**self.status.__dict__)

    def _run(self) -> None:
        context = [0]
        try:
            if self._generator is None:
                self._generator = TinyMozartGenerator()
            self._temp_dir = TemporaryDirectory(prefix="tinymozart_")
            with self._lock:
                self.status.loading = False
                self.status.device = str(self._generator.device)
                self.status.started_at = time.time()
                self.status.last_message = "Generating candidates..."

            producer = threading.Thread(
                target=self._produce_pieces,
                args=(context,),
                name="tinymozart-producer",
                daemon=True,
            )
            producer.start()

            while not self._stop.is_set():
                try:
                    midi_path, score = self._piece_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                with self._lock:
                    self.status.last_message = "Playing"
                played = self._play_midi_file(midi_path)
                if not played:
                    fallback_audio = synthesize_score(score, min_start_tick=0)
                    self._play_interruptible(fallback_audio)
                with self._lock:
                    self.status.chunks_played += 1

            sd.stop()
            try:
                pygame.mixer.music.stop()
                pygame.mixer.quit()
            except Exception:
                pass
            producer.join(timeout=2.0)
        except Exception as exc:
            with self._lock:
                self.status.error = f"{type(exc).__name__}: {exc}"
                self.status.last_message = "Error"
        finally:
            self._stop.set()
            self._clear_queue()
            if self._temp_dir is not None:
                self._temp_dir.cleanup()
                self._temp_dir = None
            with self._lock:
                self.status.running = False
                self.status.loading = False
                if not self.status.error:
                    self.status.last_message = "Stopped"

    def _produce_pieces(self, context: list[int]) -> None:
        piece_index = 0
        while not self._stop.is_set():
            if self._piece_queue.full():
                time.sleep(0.05)
                continue

            if self.config.settings.candidate_count <= 1:
                candidate = self._generate_single_passage(piece_index)
            else:
                candidate = self._select_best_candidate(piece_index)
            if candidate is None:
                continue
            midi_path, score, metrics = candidate
            piece_index += 1
            if midi_path.exists():
                self._piece_queue.put((midi_path, score))
                with self._lock:
                    self.status.chunks_generated += 1
                    self.status.best_score = metrics.score
                    self.status.last_metrics = _format_metrics(metrics)
                    self.status.quality_mode = self.config.quality_mode
                    self.status.last_message = "Buffered"

    def _generate_single_passage(self, piece_index: int) -> tuple[Path, Score, CandidateMetrics] | None:
        with self._lock:
            self.status.last_message = "Generating fast passage"
            self.status.quality_mode = self.config.quality_mode
        settings = replace(
            self.config.settings,
            seed=int(time.time() * 1000) % 2_147_483_647,
        )
        new_tokens = self._generator.generate([0], settings)  # type: ignore[union-attr]
        tokens = [token for token in new_tokens if token != 0]
        if not tokens:
            return None
        midi_path, score = self._tokens_to_piece(piece_index, 0, tokens)
        metrics = self._scorer.evaluate(tokens, score)
        with self._lock:
            self.status.rejected_candidates = 1 if metrics.rejected else 0
            self.status.best_score = metrics.score
            self.status.last_metrics = _format_metrics(metrics)
            self.status.last_message = "Scored fast passage"
        return midi_path, score, metrics

    def _select_best_candidate(self, piece_index: int) -> tuple[Path, Score, CandidateMetrics] | None:
        best: tuple[Path, Score, CandidateMetrics] | None = None
        rejected = 0
        settings = self.config.settings
        base_seed = int(time.time() * 1000) % 2_147_483_647

        for candidate_index in range(settings.candidate_count):
            if self._stop.is_set():
                return None
            with self._lock:
                self.status.last_message = (
                    f"Generating candidate {candidate_index + 1}/{settings.candidate_count}"
                )

            offset = candidate_index - ((settings.candidate_count - 1) / 2.0)
            candidate_settings = replace(
                settings,
                temperature=max(0.62, settings.temperature + offset * settings.temperature_jitter),
                top_p=max(0.82, min(0.97, settings.top_p + offset * 0.015)),
                top_k=max(12, min(64, settings.top_k + int(offset * 4))),
                seed=base_seed + candidate_index,
            )
            new_tokens = self._generator.generate([0], candidate_settings)  # type: ignore[union-attr]
            tokens = [token for token in new_tokens if token != 0]
            if not tokens:
                rejected += 1
                continue

            midi_path, score = self._tokens_to_piece(piece_index, candidate_index, tokens)
            metrics = self._scorer.evaluate(tokens, score)
            if metrics.rejected:
                rejected += 1
            if best is None or metrics.score > best[2].score:
                best = (midi_path, score, metrics)

            with self._lock:
                self.status.rejected_candidates = rejected
                self.status.best_score = best[2].score if best else 0.0
                self.status.last_metrics = _format_metrics(metrics)
                self.status.last_message = f"Scored candidate {candidate_index + 1}"

        if best is None:
            return None
        with self._lock:
            self.status.rejected_candidates = rejected
        return best

    def _tokens_to_piece(self, piece_index: int, candidate_index: int, tokens: list[int]) -> tuple[Path, Score]:
        tokenizer = self._generator.tokenizer  # type: ignore[union-attr]
        midi_dir = Path(self._temp_dir.name if self._temp_dir else ".")
        midi_path = midi_dir / f"tinymozart_piece_{piece_index:04d}_{candidate_index:02d}.mid"
        score = tokenizer.decode([tokens])
        score.dump_midi(midi_path)
        return midi_path, score

    def _clear_queue(self) -> None:
        while True:
            try:
                self._piece_queue.get_nowait()
            except queue.Empty:
                break

    def _play_midi_file(self, midi_path: Path) -> bool:
        rendered_audio = self._render_with_soundfont(midi_path)
        if rendered_audio is not None:
            self._play_interruptible(rendered_audio)
            return True

        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=1024)
            pygame.mixer.music.load(str(midi_path))
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy() and not self._stop.is_set():
                time.sleep(0.05)
            pygame.mixer.music.stop()
            return True
        except Exception as exc:
            with self._lock:
                self.status.last_message = f"MIDI playback failed, using fallback: {exc}"
            try:
                pygame.mixer.music.stop()
            except Exception:
                pass
            return False

    def _render_with_soundfont(self, midi_path: Path) -> np.ndarray | None:
        fluidsynth = os.environ.get("FLUIDSYNTH_EXE") or shutil.which("fluidsynth")
        soundfont = _find_soundfont()
        if not fluidsynth or not soundfont:
            return None

        wav_path = midi_path.with_suffix(".wav")
        command = [
            fluidsynth,
            "-ni",
            str(soundfont),
            str(midi_path),
            "-F",
            str(wav_path),
            "-r",
            str(SAMPLE_RATE),
        ]
        try:
            with self._lock:
                self.status.last_message = "Rendering piano"
            subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
            audio, sample_rate = sf.read(wav_path, dtype="float32", always_2d=True)
            if sample_rate != SAMPLE_RATE:
                return None
            return audio
        except Exception as exc:
            with self._lock:
                self.status.last_message = f"SoundFont render failed, using MIDI: {exc}"
            return None

    def _play_interruptible(self, audio: np.ndarray) -> None:
        block_size = 2048
        cursor = 0
        channels = 1 if audio.ndim == 1 else audio.shape[1]
        with sd.OutputStream(samplerate=SAMPLE_RATE, channels=channels, dtype="float32") as stream:
            while cursor < audio.size and not self._stop.is_set():
                if audio.ndim == 1:
                    block = audio[cursor : cursor + block_size].reshape(-1, 1)
                    cursor += block_size
                else:
                    frame_cursor = cursor // channels
                    block = audio[frame_cursor : frame_cursor + block_size]
                    cursor += block_size * channels
                stream.write(block)


def synthesize_score(score: Score, min_start_tick: int = 0) -> np.ndarray:
    ticks_per_quarter = max(1, int(score.ticks_per_quarter))
    tempo_bpm = 121.0
    if score.tempos:
        tempo_bpm = float(score.tempos[0].qpm)
    seconds_per_tick = 60.0 / tempo_bpm / ticks_per_quarter

    notes = []
    for track in score.tracks:
        for note in track.notes:
            if int(note.time) >= min_start_tick:
                notes.append(note)

    if not notes:
        return np.zeros(0, dtype=np.float32)

    start_tick = min(int(note.time) for note in notes)
    end_tick = max(int(note.time + note.duration) for note in notes)
    duration = max(0.5, (end_tick - start_tick) * seconds_per_tick + 0.8)
    audio = np.zeros((int(duration * SAMPLE_RATE), 2), dtype=np.float32)

    for note in notes:
        start = max(0, int((int(note.time) - start_tick) * seconds_per_tick * SAMPLE_RATE))
        length = max(1, int(max(int(note.duration), 24) * seconds_per_tick * SAMPLE_RATE))
        end = min(audio.shape[0], start + length)
        if end <= start:
            continue
        pan = 0.5 + (int(note.pitch) - 60) / 72.0
        audio[start:end] += _piano_note(int(note.pitch), int(note.velocity), end - start, pan)

    audio = _add_room(audio)
    audio = np.tanh(audio * 1.35)
    max_abs = float(np.max(np.abs(audio))) if audio.size else 0.0
    if max_abs > 0:
        audio = 0.92 * audio / max(1.0, max_abs)
    return audio.astype(np.float32)


def _piano_note(pitch: int, velocity: int, samples: int, pan: float) -> np.ndarray:
    frequency = 440.0 * (2.0 ** ((pitch - 69) / 12.0))
    t = np.arange(samples, dtype=np.float32) / SAMPLE_RATE
    velocity_norm = max(0.08, min(1.0, velocity / 127.0))
    low_note = max(0.0, (72 - pitch) / 48.0)
    high_note = max(0.0, (pitch - 60) / 48.0)

    attack = np.minimum(1.0, t / 0.006)
    fast_decay = np.exp(-(2.4 + high_note * 2.0) * t)
    slow_decay = np.exp(-(0.75 + high_note * 1.5) * t)
    body_env = attack * (0.55 * fast_decay + 0.45 * slow_decay)
    hammer_env = np.exp(-95.0 * t)

    wave = np.zeros(samples, dtype=np.float32)
    partials = (
        (1.00, 1.000, 0.00),
        (0.62, 2.003, 0.21),
        (0.34, 3.009, 0.37),
        (0.20, 4.018, 0.53),
        (0.12, 5.032, 0.77),
        (0.08, 6.055, 0.91),
    )
    for gain, ratio, phase in partials:
        damp = math.exp(-0.23 * ratio * (1.0 + high_note))
        wave += (gain * damp) * np.sin(2 * math.pi * frequency * ratio * t + phase)

    hammer = 0.22 * np.sin(2 * math.pi * frequency * 9.7 * t) * hammer_env
    noise = 0.018 * velocity_norm * np.random.default_rng(pitch + samples).standard_normal(samples)
    note = velocity_norm * (0.28 + low_note * 0.1) * (body_env * wave + hammer + noise * hammer_env)

    left_gain = math.cos(max(0.0, min(1.0, pan)) * math.pi / 2.0)
    right_gain = math.sin(max(0.0, min(1.0, pan)) * math.pi / 2.0)
    stereo = np.column_stack((note * left_gain, note * right_gain))
    return stereo.astype(np.float32)


def _add_room(audio: np.ndarray) -> np.ndarray:
    if audio.size == 0:
        return audio
    delays = (
        (0.031, 0.18, 0.14),
        (0.047, 0.12, 0.16),
        (0.073, 0.09, 0.08),
        (0.113, 0.06, 0.07),
    )
    wet = np.zeros_like(audio)
    for delay_s, left_gain, right_gain in delays:
        delay = int(delay_s * SAMPLE_RATE)
        if delay >= audio.shape[0]:
            continue
        wet[delay:, 0] += audio[:-delay, 0] * left_gain
        wet[delay:, 1] += audio[:-delay, 1] * right_gain
    return audio + wet


def _format_metrics(metrics: CandidateMetrics) -> str:
    return (
        f"score {metrics.score:.1f}, notes {metrics.notes}, range {metrics.pitch_range}, "
        f"density {metrics.density}, repeat {metrics.repeated_ngram_ratio}"
    )


def _find_soundfont() -> Path | None:
    candidates = [
        os.environ.get("TINYMOZART_SF2"),
        os.environ.get("SF2_PATH"),
        "FluidR3_GM.sf2",
        "GeneralUser_GS.sf2",
        "soundfonts/FluidR3_GM.sf2",
        "soundfonts/GeneralUser_GS.sf2",
        "C:/soundfonts/FluidR3_GM.sf2",
        "C:/soundfonts/GeneralUser_GS.sf2",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists() and path.suffix.lower() in {".sf2", ".sf3"}:
            return path
    return None
