"""
services/speaker_detector.py
─────────────────────────────
Speaker detection — now consumes the DENSE per-frame MAR data produced by
multi_face_tracker (30fps lip analysis) instead of running its own video pass.

Fallback path (standalone VisualActivityDetector) kept for when no face
timeline is available.
"""

from __future__ import annotations

import os
import logging
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    from pyannote.audio import Pipeline as PyannotePipeline
    _PYANNOTE_AVAILABLE = True
except ImportError:
    _PYANNOTE_AVAILABLE = False

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SpeakerSegment:
    speaker_id: str
    start: float              # CLIP-RELATIVE seconds
    end: float                # CLIP-RELATIVE seconds
    face_id: Optional[int] = None
    confidence: float = 1.0


@dataclass
class WordWithSpeaker:
    text: str
    start: float
    end: float
    speaker_id: str
    face_id: Optional[int] = None


# ──────────────────────────────────────────────────────────────────────────────
# Audio extraction
# ──────────────────────────────────────────────────────────────────────────────

def _extract_audio_wav(video_path: str, out_wav: Optional[str] = None) -> str:
    if out_wav is None:
        fd, out_wav = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-af", "highpass=f=80,lowpass=f=8000",
        out_wav,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")
        return out_wav
    except subprocess.TimeoutExpired:
        raise RuntimeError("Audio extraction timed out")


# ──────────────────────────────────────────────────────────────────────────────
# Audio diarization
# ──────────────────────────────────────────────────────────────────────────────

class AudioSpeakerDiarizer:
    def __init__(
        self,
        hf_token: Optional[str] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ):
        self.hf_token = hf_token or os.environ.get("HUGGINGFACE_TOKEN", "")
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self._pipeline = None

        if _PYANNOTE_AVAILABLE and self.hf_token:
            try:
                self._pipeline = PyannotePipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=self.hf_token,
                )
                if _TORCH_AVAILABLE and torch.cuda.is_available():
                    self._pipeline.to(torch.device("cuda"))
                    logger.info("pyannote on CUDA")
                else:
                    logger.info("pyannote on CPU")
            except Exception as exc:
                logger.warning("pyannote load failed: %s — energy VAD fallback.", exc)
                self._pipeline = None

    def diarize(self, audio_path: str, *, is_video: bool = False) -> List[SpeakerSegment]:
        wav_path = None
        cleanup = False
        try:
            if is_video:
                wav_path = _extract_audio_wav(audio_path)
                cleanup = True
                target = wav_path
            else:
                target = audio_path

            if self._pipeline is not None:
                return self._run_pyannote(target)
            return self._run_energy_vad(target)
        finally:
            if cleanup and wav_path and os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except OSError:
                    pass

    def annotate_words(
        self, words: List[Dict[str, Any]], segments: List[SpeakerSegment]
    ) -> List[WordWithSpeaker]:
        result = []
        for w in words:
            mid = (w["start"] + w["end"]) / 2.0
            spk = self._find_speaker(mid, segments)
            face_id = None
            for seg in segments:
                if seg.start <= mid <= seg.end and seg.speaker_id == spk:
                    face_id = seg.face_id
                    break
            result.append(WordWithSpeaker(
                text=w["text"], start=w["start"], end=w["end"],
                speaker_id=spk, face_id=face_id,
            ))
        return result

    def _run_pyannote(self, audio_path: str) -> List[SpeakerSegment]:
        kwargs: Dict[str, Any] = {}
        if self.min_speakers is not None:
            kwargs["min_speakers"] = self.min_speakers
        if self.max_speakers is not None:
            kwargs["max_speakers"] = self.max_speakers
        try:
            dia = self._pipeline(audio_path, **kwargs)
            segs = [
                SpeakerSegment(speaker_id=spk, start=turn.start, end=turn.end)
                for turn, _, spk in dia.itertracks(yield_label=True)
            ]
            return self._merge_short(segs)
        except Exception as e:
            logger.error("pyannote failed: %s", e)
            return []

    @staticmethod
    def _merge_short(segs: List[SpeakerSegment], max_gap: float = 0.5) -> List[SpeakerSegment]:
        if not segs:
            return segs
        segs = sorted(segs, key=lambda s: s.start)
        merged = [segs[0]]
        for s in segs[1:]:
            last = merged[-1]
            if s.speaker_id == last.speaker_id and (s.start - last.end) < max_gap:
                last.end = max(last.end, s.end)
            else:
                merged.append(s)
        return merged

    def _run_energy_vad(self, audio_path: str) -> List[SpeakerSegment]:
        try:
            import wave
            with wave.open(audio_path, "rb") as wf:
                framerate = wf.getframerate()
                raw = wf.readframes(wf.getnframes())
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        except Exception as exc:
            logger.warning("VAD read failed: %s", exc)
            return []

        frame_len = int(framerate * 0.025)
        hop_len = int(framerate * 0.010)

        energies = []
        t, i = 0.0, 0
        while i + frame_len <= len(samples):
            frame = samples[i:i + frame_len]
            rms = float(np.sqrt(np.mean(frame ** 2)))
            energies.append((t, rms))
            i += hop_len
            t += 0.010

        if not energies:
            return []

        rms_vals = [e[1] for e in energies]
        noise_floor = np.percentile(rms_vals, 20)
        threshold = max(0.02, noise_floor * 3.0)

        segs: List[SpeakerSegment] = []
        in_seg, seg_start = False, 0.0
        for t, rms in energies:
            if rms > threshold and not in_seg:
                in_seg, seg_start = True, t
            elif rms <= threshold and in_seg:
                in_seg = False
                if t - seg_start > 0.2:
                    segs.append(SpeakerSegment("SPEAKER_00", seg_start, t))
        if in_seg and energies:
            segs.append(SpeakerSegment("SPEAKER_00", seg_start, energies[-1][0]))

        return self._merge_short(segs, max_gap=0.3)

    @staticmethod
    def _find_speaker(t: float, segments: List[SpeakerSegment]) -> str:
        for seg in segments:
            if seg.start <= t <= seg.end:
                return seg.speaker_id
        return "UNKNOWN"


# ──────────────────────────────────────────────────────────────────────────────
# MAR → speaking-segment analysis (shared by dense-timeline and fallback paths)
# ──────────────────────────────────────────────────────────────────────────────

def _rolling_variance(arr: np.ndarray, window: int) -> np.ndarray:
    out = np.zeros_like(arr)
    half = window // 2
    for i in range(len(arr)):
        lo = max(0, i - half)
        hi = min(len(arr), i + half + 1)
        out[i] = float(np.var(arr[lo:hi]))
    return out


def _robust_normalize(arr: np.ndarray) -> np.ndarray:
    p10, p90 = np.percentile(arr, [10, 90])
    rng = max(1e-6, p90 - p10)
    return np.clip((arr - p10) / rng, 0, 1)


def _sustain_filter(binary: np.ndarray, min_run: int) -> np.ndarray:
    result = binary.copy()
    # remove short ON runs
    i = 0
    while i < len(result):
        if result[i]:
            j = i
            while j < len(result) and result[j]:
                j += 1
            if j - i < min_run:
                result[i:j] = False
            i = j
        else:
            i += 1
    # fill short OFF gaps
    i = 0
    while i < len(result):
        if not result[i]:
            j = i
            while j < len(result) and not result[j]:
                j += 1
            if 0 < i and j < len(result) and (j - i) < max(1, min_run // 2):
                result[i:j] = True
            i = j
        else:
            i += 1
    return result


def mar_series_to_segments(
    times: np.ndarray,
    mars: np.ndarray,
    sample_interval: float,
    mar_threshold: float = 0.3,
    min_seg_duration: float = 0.25,
) -> List[Tuple[float, float]]:
    """Convert a (time, MAR) series into speaking intervals."""
    if len(times) < 5:
        return []

    window = max(3, int(0.4 / max(0.01, sample_interval)))
    mar_var = _rolling_variance(mars, window)
    score = _robust_normalize(mar_var)

    noise = np.percentile(score, 25)
    threshold = max(mar_threshold, noise + 0.15)

    raw = score > threshold
    min_frames = max(2, int(min_seg_duration / max(0.01, sample_interval)))
    speaking = _sustain_filter(raw, min_frames)

    segs: List[Tuple[float, float]] = []
    in_seg, start = False, 0.0
    for i, s in enumerate(speaking):
        if s and not in_seg:
            in_seg, start = True, times[i]
        elif not s and in_seg:
            in_seg = False
            if times[i] - start >= min_seg_duration:
                segs.append((float(start), float(times[i])))
    if in_seg:
        if times[-1] - start >= min_seg_duration:
            segs.append((float(start), float(times[-1])))

    # merge nearby
    merged: List[Tuple[float, float]] = []
    for s in segs:
        if merged and s[0] - merged[-1][1] < 0.3:
            merged[-1] = (merged[-1][0], s[1])
        else:
            merged.append(s)
    return merged


def _visual_from_dense_timeline(face_timeline: List) -> Dict[int, List[Tuple[float, float]]]:
    """
    Build per-face_id speaking intervals from DENSE timeline MAR data.
    Returns {face_id: [(start, end), ...]} in clip-relative time.
    """
    series: Dict[int, Tuple[List[float], List[float]]] = {}
    for fd in face_timeline:
        for face in fd.faces:
            if face.is_coasted:
                continue  # stale MAR while coasting — skip
            t, m = series.get(face.face_id, ([], []))
            t.append(fd.timestamp)
            m.append(face.mar)
            series[face.face_id] = (t, m)

    out: Dict[int, List[Tuple[float, float]]] = {}
    for fid, (ts, ms) in series.items():
        if len(ts) < 10:
            continue
        t_arr = np.array(ts)
        m_arr = np.array(ms)
        # estimate effective sample interval from median dt
        if len(t_arr) > 1:
            dt = float(np.median(np.diff(t_arr)))
        else:
            dt = 1 / 30.0
        segs = mar_series_to_segments(t_arr, m_arr, sample_interval=max(dt, 1 / 60.0))
        if segs:
            out[fid] = segs
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Unified SpeakerDetector
# ──────────────────────────────────────────────────────────────────────────────

class SpeakerDetector:
    """
    Priority order for face assignment:
      1. Dense MAR from face_timeline (best — 30fps, same pass as tracking)
      2. Audio-only area voting (when no face timeline)
    """

    def __init__(
        self,
        hf_token: Optional[str] = None,
        use_visual: bool = True,
        sample_interval: float = 0.1,
        max_speakers: Optional[int] = None,
    ):
        self.audio_diarizer = AudioSpeakerDiarizer(
            hf_token=hf_token, max_speakers=max_speakers,
        )
        self.use_visual = use_visual

    def detect(
        self,
        video_path: str,
        face_timeline: Optional[List] = None,
        all_words: Optional[List[Dict]] = None,
        clip_start: float = 0.0,
        clip_end: Optional[float] = None,
    ) -> List[SpeakerSegment]:
        # 1. Audio diarization (absolute → clip-relative)
        audio_segs = self.audio_diarizer.diarize(video_path, is_video=True)

        if clip_end is not None:
            audio_segs = [
                s for s in audio_segs
                if s.end > clip_start and s.start < clip_end
            ]
        for seg in audio_segs:
            seg.start = max(0.0, seg.start - clip_start)
            seg.end = seg.end - clip_start
            if clip_end:
                seg.end = min(seg.end, clip_end - clip_start)

        # 2. Face assignment
        if face_timeline:
            has_dense_mar = any(
                getattr(f, "mar", 0.0) > 0.0
                for fd in face_timeline[:50]
                for f in fd.faces
            )
            if self.use_visual and has_dense_mar:
                visual = _visual_from_dense_timeline(face_timeline)
                self._assign_with_visual(audio_segs, visual, face_timeline)
                logger.info(
                    "[Speaker Detection] dense-MAR visual intervals for %d faces",
                    len(visual),
                )
            else:
                self._assign_by_area(audio_segs, face_timeline)
        # else: audio-only, face_id stays None

        return sorted(audio_segs, key=lambda s: s.start)

    def annotate_words(
        self, words: List[Dict], segments: List[SpeakerSegment]
    ) -> List[WordWithSpeaker]:
        return self.audio_diarizer.annotate_words(words, segments)

    def close(self) -> None:
        pass

    # ── assignment strategies ──

    def _assign_with_visual(
        self,
        audio_segs: List[SpeakerSegment],
        visual: Dict[int, List[Tuple[float, float]]],
        face_timeline: List,
    ) -> None:
        """Score faces per audio segment: visual speaking overlap >> area >> age."""
        for seg in audio_segs:
            seg_dur = max(1e-6, seg.end - seg.start)
            scores: Dict[int, float] = defaultdict(float)

            frames_in_seg = [fd for fd in face_timeline if seg.start <= fd.timestamp <= seg.end]
            if not frames_in_seg:
                continue

            for fd in frames_in_seg:
                for face in fd.faces:
                    s = 0.0
                    # visual speaking at this exact frame → strong signal
                    intervals = visual.get(face.face_id, [])
                    for vs, ve in intervals:
                        if vs <= fd.timestamp <= ve:
                            s += 10.0
                            break
                    # area + confidence + track maturity
                    s += (face.box.area / 10000.0) * face.box.confidence
                    s += face.box.confidence
                    s += min(1.0, face.age / 15.0)
                    scores[face.face_id] += s

            if scores:
                ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
                seg.face_id = ranked[0][0]
                if len(ranked) > 1:
                    seg.confidence = float(np.clip(ranked[0][1] / (ranked[1][1] + 1e-6) / 2.0, 0.3, 1.0))
                else:
                    seg.confidence = 1.0

        # Consistency: one speaker_id → one face_id (weighted by duration × confidence)
        votes: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        for seg in audio_segs:
            if seg.face_id is not None:
                votes[seg.speaker_id][seg.face_id] += (seg.end - seg.start) * seg.confidence

        final: Dict[str, int] = {}
        for spk, fv in votes.items():
            if fv:
                final[spk] = max(fv, key=lambda fid: fv[fid])

        for seg in audio_segs:
            if seg.speaker_id in final and (seg.confidence < 0.6 or seg.face_id is None):
                seg.face_id = final[seg.speaker_id]

    def _assign_by_area(self, audio_segs: List[SpeakerSegment], face_timeline: List) -> None:
        """Fallback: largest face wins per segment (single-speaker heuristic)."""
        votes: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        for fd in face_timeline:
            for seg in audio_segs:
                if seg.start <= fd.timestamp <= seg.end:
                    for face in fd.faces:
                        votes[seg.speaker_id][face.face_id] += face.box.area
        mapping = {
            spk: max(fv, key=lambda fid: fv[fid])
            for spk, fv in votes.items() if fv
        }
        for seg in audio_segs:
            seg.face_id = mapping.get(seg.speaker_id)


# ──────────────────────────────────────────────────────────────────────────────
# Convenience function (signature unchanged)
# ──────────────────────────────────────────────────────────────────────────────

def detect_speakers(
    video_path: str,
    face_timeline: Optional[List] = None,
    all_words: Optional[List[Dict]] = None,
    clip_start: float = 0.0,
    clip_end: Optional[float] = None,
    hf_token: Optional[str] = None,
    use_visual: bool = True,
) -> Tuple[List[SpeakerSegment], Optional[List[WordWithSpeaker]]]:
    detector = SpeakerDetector(hf_token=hf_token, use_visual=use_visual)
    try:
        segments = detector.detect(
            video_path=video_path,
            face_timeline=face_timeline,
            all_words=all_words,
            clip_start=clip_start,
            clip_end=clip_end,
        )
        annotated = detector.annotate_words(all_words, segments) if all_words else None
        return segments, annotated
    finally:
        detector.close()