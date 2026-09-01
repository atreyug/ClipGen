"""
services/active_speaker_reframe.py
────────────────────────────────────
Cinematic active-speaker reframe over the DENSE per-frame timeline.

KEY FIX: when no face / no speaker match at time t, HOLD the previous crop
position (carry forward) instead of jumping to center. Combined with
clip-relative timestamps, the FFmpeg expression now always matches.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ReframeKeyframe:
    time: float
    crop_x: float
    crop_y: float


class ActiveSpeakerReframer:
    def __init__(
        self,
        frame_w: int,
        frame_h: int,
        target_aspect: str = "9:16",
        transition_sec: float = 0.6,
        vertical_center: bool = True,
        deadzone_px: int = 50,
        use_ease: bool = True,
    ):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.transition_sec = transition_sec
        self.vertical_center = vertical_center
        self.deadzone_px = deadzone_px
        self.use_ease = use_ease

        if target_aspect == "9:16":
            self.target_w = int(frame_h * 9 / 16)
            self.target_h = frame_h
        else:
            self.target_w = frame_w
            self.target_h = int(frame_w * 9 / 16)

        self.target_w = min(self.target_w - (self.target_w % 2), frame_w)
        self.target_h = min(self.target_h - (self.target_h % 2), frame_h)

        self.max_x = max(0, frame_w - self.target_w)
        self.max_y = max(0, frame_h - self.target_h)
        self.center_x = (self.max_x // 2) - ((self.max_x // 2) % 2)
        self.center_y = (self.max_y // 2) - ((self.max_y // 2) % 2)

    def build_keyframes(
        self,
        face_timeline: List,
        speaker_segments: List,
        clip_duration: float,
    ) -> List[ReframeKeyframe]:
        raw: List[Tuple[float, float, float]] = []
        prev: Optional[Tuple[float, float]] = None

        for fd in face_timeline:
            t = fd.timestamp
            target = self._pick_target_face(fd, speaker_segments, t)

            if target is not None:
                cx = int(np.clip(target.crop_x_smoothed, 0, self.max_x))
                cy = int(self._vertical_crop(target)) if self.vertical_center else self.center_y
                prev = (cx, cy)
            elif prev is not None:
                # HOLD last position — never jump to center mid-clip
                cx, cy = prev
            else:
                cx, cy = self.center_x, self.center_y

            cx -= cx % 2
            cy -= cy % 2
            raw.append((t, float(cx), float(cy)))

        if not raw:
            return [
                ReframeKeyframe(0.0, self.center_x, self.center_y),
                ReframeKeyframe(clip_duration, self.center_x, self.center_y),
            ]

        # Hysteresis: median-buffer dead-zone (prevents oscillation)
        stabilized = self._apply_hysteresis(raw)

        # Adaptive transitions + ease points
        keyframes = self._build_adaptive(stabilized, clip_duration)

        # Boundaries
        if keyframes and keyframes[-1].time < clip_duration - 0.01:
            keyframes.append(ReframeKeyframe(
                clip_duration, keyframes[-1].crop_x, keyframes[-1].crop_y
            ))
        if keyframes and keyframes[0].time > 0.01:
            keyframes.insert(0, ReframeKeyframe(
                0.0, keyframes[0].crop_x, keyframes[0].crop_y
            ))

        # Clamp all times into [0, clip_duration]
        for kf in keyframes:
            kf.time = float(np.clip(kf.time, 0.0, clip_duration))

        return keyframes

    def _apply_hysteresis(
        self, raw: List[Tuple[float, float, float]]
    ) -> List[Tuple[float, float, float]]:
        if not raw:
            return raw
        result = [raw[0]]
        cur_x, cur_y = raw[0][1], raw[0][2]
        buf: List[Tuple[float, float]] = [(raw[0][1], raw[0][2])]
        buf_size = 7  # ~0.23s at 30fps

        for t, cx, cy in raw[1:]:
            buf.append((cx, cy))
            if len(buf) > buf_size:
                buf.pop(0)
            med_x = float(np.median([b[0] for b in buf]))
            med_y = float(np.median([b[1] for b in buf]))

            if abs(med_x - cur_x) > self.deadzone_px or abs(med_y - cur_y) > self.deadzone_px * 0.7:
                cur_x, cur_y = med_x, med_y
            result.append((t, cur_x, cur_y))
        return result

    def _build_adaptive(
        self, positions: List[Tuple[float, float, float]], clip_duration: float
    ) -> List[ReframeKeyframe]:
        keyframes: List[ReframeKeyframe] = []
        last: Optional[ReframeKeyframe] = None

        for t, cx, cy in positions:
            if last is None:
                last = ReframeKeyframe(t, cx, cy)
                keyframes.append(last)
                continue

            dx = abs(cx - last.crop_x)
            dy = abs(cy - last.crop_y)
            if dx < 4 and dy < 4:
                continue

            distance = float(np.hypot(dx, dy))
            trans = float(np.clip(
                self.transition_sec * (distance / 200.0), 0.3, self.transition_sec * 1.5
            ))

            t_start = max(last.time, t - trans)
            if t_start > last.time + 0.01:
                keyframes.append(ReframeKeyframe(t_start, last.crop_x, last.crop_y))
                if self.use_ease:
                    keyframes.extend(self._ease_points(
                        t_start, t, last.crop_x, cx, last.crop_y, cy
                    ))

            last = ReframeKeyframe(t, cx, cy)
            keyframes.append(last)

        return keyframes

    @staticmethod
    def _ease_points(t0, t1, x0, x1, y0, y1, n=3) -> List[ReframeKeyframe]:
        pts = []
        dt = t1 - t0
        for i in range(1, n):
            p = i / n
            eased = p * p * (3 - 2 * p)  # smoothstep
            t = t0 + p * dt
            x = int(x0 + eased * (x1 - x0))
            y = int(y0 + eased * (y1 - y0))
            x -= x % 2
            y -= y % 2
            pts.append(ReframeKeyframe(t, float(x), float(y)))
        return pts

    def _pick_target_face(self, fd, speaker_segments: List, t: float):
        if not fd.faces:
            return None

        best_seg = None
        best_conf = 0.0
        for seg in speaker_segments:
            if seg.start <= t <= seg.end and seg.face_id is not None:
                if seg.confidence > best_conf:
                    best_conf = seg.confidence
                    best_seg = seg

        if best_seg is not None:
            for face in fd.faces:
                if face.face_id == best_seg.face_id:
                    return face

        return fd.dominant_face()

    def _vertical_crop(self, face) -> float:
        # face in upper-third framing
        return float(np.clip(face.box.cy - self.target_h * 0.4, 0, self.max_y))

    def smooth_trajectory(
        self, keyframes: List[ReframeKeyframe], fps: float = 25.0
    ) -> List[ReframeKeyframe]:
        if len(keyframes) < 2:
            return keyframes

        t0, t1 = keyframes[0].time, keyframes[-1].time
        times = np.arange(t0, t1 + 1 / fps, 1 / fps)
        cx = np.interp(times, [k.time for k in keyframes], [k.crop_x for k in keyframes])
        cy = np.interp(times, [k.time for k in keyframes], [k.crop_y for k in keyframes])

        sigma = max(1, int(fps * self.transition_sec / 4))
        cx = np.clip(self._gauss(cx, sigma), 0, self.max_x)
        cy = np.clip(self._gauss(cy, sigma), 0, self.max_y)

        step = max(1, int(fps / 10))
        sampled = [
            ReframeKeyframe(float(times[i]), float(cx[i]), float(cy[i]))
            for i in range(0, len(times), step)
        ]
        if sampled and sampled[-1].time < t1 - 0.05:
            sampled.append(ReframeKeyframe(t1, float(cx[-1]), float(cy[-1])))

        simplified = []
        for kf in sampled:
            if not simplified or kf.time in (t0, t1):
                simplified.append(kf)
            elif abs(kf.crop_x - simplified[-1].crop_x) >= 3.0 or abs(kf.crop_y - simplified[-1].crop_y) >= 3.0:
                simplified.append(kf)

        if len(simplified) > 400:
            idx = np.linspace(0, len(simplified) - 1, 400, dtype=int)
            simplified = [simplified[i] for i in idx]

        return simplified

    def _keyframes_to_ffmpeg_filter(self, keyframes: List[ReframeKeyframe]) -> str:
        W, H = self.target_w, self.target_h

        if not keyframes:
            return f"crop={W}:{H}:{self.center_x}:{self.center_y}"

        if len(keyframes) == 1:
            x = int(np.clip(keyframes[0].crop_x, 0, self.max_x)); x -= x % 2
            y = int(np.clip(keyframes[0].crop_y, 0, self.max_y)); y -= y % 2
            return f"crop={W}:{H}:{x}:{y}"

        lx = int(np.clip(keyframes[-1].crop_x, 0, self.max_x)); lx -= lx % 2
        ly = int(np.clip(keyframes[-1].crop_y, 0, self.max_y)); ly -= ly % 2
        x_expr, y_expr = f"{lx}", f"{ly}"

        for i in range(len(keyframes) - 2, -1, -1):
            k0, k1 = keyframes[i], keyframes[i + 1]
            dt = k1.time - k0.time
            if dt < 1e-4:
                continue

            x0 = int(np.clip(k0.crop_x, 0, self.max_x)); x0 -= x0 % 2
            x1 = int(np.clip(k1.crop_x, 0, self.max_x)); x1 -= x1 % 2
            y0 = int(np.clip(k0.crop_y, 0, self.max_y)); y0 -= y0 % 2
            y1 = int(np.clip(k1.crop_y, 0, self.max_y)); y1 -= y1 % 2

            sx = (x1 - x0) / dt
            sy = (y1 - y0) / dt

            x_lerp = f"max(0\\,min({x0}+{sx:.4f}*(t-{k0.time:.4f})\\,{self.max_x}))"
            y_lerp = f"max(0\\,min({y0}+{sy:.4f}*(t-{k0.time:.4f})\\,{self.max_y}))"

            x_expr = f"if(between(t\\,{k0.time:.4f}\\,{k1.time:.4f})\\,{x_lerp}\\,{x_expr})"
            y_expr = f"if(between(t\\,{k0.time:.4f}\\,{k1.time:.4f})\\,{y_lerp}\\,{y_expr})"

        x_even = f"2*trunc(({x_expr})/2)"
        y_even = f"2*trunc(({y_expr})/2)"
        return f"crop={W}:{H}:{x_even}:{y_even}"

    @staticmethod
    def _gauss(arr: np.ndarray, sigma: int) -> np.ndarray:
        if sigma < 1:
            return arr
        half = sigma * 3
        k = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2)
        k /= k.sum()
        return np.convolve(arr, k, mode="same")


def build_active_speaker_crop_filter(
    face_timeline: List,
    speaker_segments: List,
    frame_w: int,
    frame_h: int,
    clip_duration: float,
    target_aspect: str = "9:16",
    transition_sec: float = 0.6,
    smooth: bool = True,
    fps: float = 25.0,
) -> str:
    reframer = ActiveSpeakerReframer(
        frame_w=frame_w, frame_h=frame_h,
        target_aspect=target_aspect, transition_sec=transition_sec,
        vertical_center=True, use_ease=True,
    )
    kf = reframer.build_keyframes(face_timeline, speaker_segments, clip_duration)
    if smooth and len(kf) > 2:
        kf = reframer.smooth_trajectory(kf, fps=fps)
    return reframer._keyframes_to_ffmpeg_filter(kf)