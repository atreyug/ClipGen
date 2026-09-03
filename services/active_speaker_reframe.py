from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

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
        vertical_center: bool = True,
        deadzone_px: int = 40,
    ):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.vertical_center = vertical_center
        self.deadzone_px = deadzone_px
        self._last_target_id = None
        self._target_history: List[Optional[int]] = []
        self._visual_activity_by_frame: Dict[int, Dict[int, float]] = {}

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
        max_keyframes: int = 60,   # safety cap regardless of deadzone tuning
    ) -> List[ReframeKeyframe]:
        self._visual_activity_by_frame = self._build_visual_activity(face_timeline)

        keyframes: List[ReframeKeyframe] = []
        prev_cx: Optional[float] = None
        prev_cy: Optional[float] = None

        for fd in face_timeline:
            t = fd.timestamp
            target = self._pick_target_face(fd, speaker_segments, t)

            if target is not None:
                cx = int(np.clip(target.crop_x_smoothed, 0, self.max_x))
                cy = int(self._vertical_crop(target)) if self.vertical_center else self.center_y
            elif fd.faces:
                fallback = fd.dominant_face()
                if fallback:
                    cx = int(np.clip(fallback.crop_x_smoothed, 0, self.max_x))
                    cy = int(self._vertical_crop(fallback)) if self.vertical_center else self.center_y
                elif prev_cx is not None:
                    cx, cy = int(prev_cx), int(prev_cy)
                else:
                    cx, cy = self.center_x, self.center_y
            elif prev_cx is not None:
                cx, cy = int(prev_cx), int(prev_cy)
            else:
                cx, cy = self.center_x, self.center_y

            cx -= cx % 2
            cy -= cy % 2

            # Only emit a keyframe once movement clears the deadzone —
            # this is what actually keeps keyframe count (and therefore
            # nested-if depth in the ffmpeg expression) bounded.
            if (
                prev_cx is None
                or abs(cx - prev_cx) > self.deadzone_px
                or abs(cy - prev_cy) > self.deadzone_px
            ):
                keyframes.append(ReframeKeyframe(t, float(cx), float(cy)))
                prev_cx, prev_cy = float(cx), float(cy)

        if not keyframes:
            return [
                ReframeKeyframe(0.0, self.center_x, self.center_y),
                ReframeKeyframe(clip_duration, self.center_x, self.center_y),
            ]

        if keyframes[0].time > 0.0:
            keyframes.insert(0, ReframeKeyframe(0.0, keyframes[0].crop_x, keyframes[0].crop_y))
        if keyframes[-1].time < clip_duration:
            keyframes.append(ReframeKeyframe(clip_duration, keyframes[-1].crop_x, keyframes[-1].crop_y))

        for kf in keyframes:
            kf.time = float(np.clip(kf.time, 0.0, clip_duration))

        # Hard backstop: even with a sensible deadzone, pathological input
        # (rapid multi-face swapping) could still produce too many keyframes
        # for ffmpeg's eval parser to nest. Evenly downsample rather than crash.
        if len(keyframes) > max_keyframes:
            logger.warning(
                "Reframe keyframes (%d) exceed max_keyframes (%d); downsampling.",
                len(keyframes), max_keyframes,
            )
            first, last = keyframes[0], keyframes[-1]
            step = (len(keyframes) - 1) / (max_keyframes - 1)
            indices = sorted({round(i * step) for i in range(max_keyframes)})
            keyframes = [keyframes[i] for i in indices]
            keyframes[0], keyframes[-1] = first, last

        return keyframes
    
    @staticmethod
    def _build_visual_activity(
        face_timeline: List,
        window_sec: float = 0.25,
    ) -> Dict[int, Dict[int, float]]:
        history: Dict[int, Deque[Tuple[float, float]]] = {}
        result:  Dict[int, Dict[int, float]] = {}

        for fd in face_timeline:
            per_face: Dict[int, float] = {}
            for face in fd.faces:
                if getattr(face, "is_coasted", False):
                    per_face[face.face_id] = 0.0
                    continue

                samples = history.setdefault(face.face_id, deque())
                mar = max(0.0, float(getattr(face, "mar", 0.0)))
                samples.append((fd.timestamp, mar))

                cutoff = fd.timestamp - window_sec
                while samples and samples[0][0] < cutoff:
                    samples.popleft()

                values = np.asarray([v for _, v in samples], dtype=float)
                if len(values) >= 2:
                    low     = float(np.percentile(values, 20))
                    high    = float(np.percentile(values, 90))
                    motion  = float(np.clip((high - low) / 0.12, 0.0, 1.0))
                    opening = float(np.clip((high - 0.06) / 0.20, 0.0, 1.0))
                    activity = 0.65 * motion + 0.35 * opening
                else:
                    activity = float(np.clip((mar - 0.08) / 0.20, 0.0, 1.0))

                per_face[face.face_id] = activity

            result[id(fd)] = per_face

        return result

    def _pick_target_face(self, fd, speaker_segments: List, t: float):
        # Find active audio speaker
        active_speaker_seg = None
        for seg in speaker_segments:
            if seg.start <= t <= seg.end and seg.face_id is not None:
                if active_speaker_seg is None or seg.confidence > active_speaker_seg.confidence:
                    active_speaker_seg = seg
            elif seg.start <= t <= seg.end and seg.face_id is None:
                if active_speaker_seg is None:
                    active_speaker_seg = seg

        # Silence: hold current face
        if active_speaker_seg is None and self._last_target_id is not None:
            for face in fd.faces:
                if (face.face_id == self._last_target_id
                        and not getattr(face, "is_coasted", False)):
                    return face

        activity_by_face       = self._visual_activity_by_frame.get(id(fd), {})
        max_visual_activity    = max(activity_by_face.values(), default=0.0)
        strong_visual_speaker  = max_visual_activity > 0.35

        scores: Dict[int, float] = {}
        for face in fd.faces:
            score         = 0.0
            visual_act    = activity_by_face.get(face.face_id, 0.0)

            # Audio diarization
            if active_speaker_seg and face.face_id == active_speaker_seg.face_id:
                reliable = getattr(active_speaker_seg, "identity_reliable", True)
                if reliable and strong_visual_speaker and visual_act < 0.12:
                    score += 2.0 * active_speaker_seg.confidence
                elif reliable:
                    score += 12.0 * active_speaker_seg.confidence

            if active_speaker_seg and active_speaker_seg.face_id is None:
                score += 4.0 * visual_act * active_speaker_seg.confidence

            # Visual lip activity
            if not getattr(face, "is_profile", False):
                score += 8.0 * visual_act

            # Face quality
            if hasattr(face, "size_score") and hasattr(face, "centrality_score"):
                score += 3.0 * face.size_score
                score += 2.0 * face.centrality_score
            else:
                area_norm = min(face.box.area / 100000.0, 1.0)
                score += 2.5 * area_norm * face.box.confidence
                dist_from_cx = abs(face.box.cx - self.frame_w / 2)
                score += 1.5 * (1.0 - min(dist_from_cx / (self.frame_w / 2), 1.0))

            if hasattr(face, "stability_score"):
                score += 2.0 * face.stability_score
            elif not face.is_coasted:
                score += 1.5 * min(face.age / 30.0, 1.0)

            score += 1.0 * face.box.confidence

            if face.is_coasted:
                score *= 0.25

            scores[face.face_id] = score

        if not scores:
            return fd.dominant_face()

        best_id    = max(scores, key=scores.get)
        best_score = scores[best_id]

        self._target_history.append(best_id)
        if len(self._target_history) > 10:
            self._target_history.pop(0)

        # Hysteresis
        if self._last_target_id is not None and self._last_target_id in scores:
            current_score    = scores[self._last_target_id]
            stability_frames = sum(1 for fid in self._target_history if fid == self._last_target_id)
            stability_bonus  = min(stability_frames / 10.0, 1.0) * 0.3

            if strong_visual_speaker:
                last_act = activity_by_face.get(self._last_target_id, 0.0)
                best_act = activity_by_face.get(best_id, 0.0)
                if last_act < 0.15 and best_act > 0.40:
                    hysteresis_mult = 1.03
                elif last_act > 0.30:
                    hysteresis_mult = 1.25
                else:
                    hysteresis_mult = 1.12
            else:
                if active_speaker_seg and active_speaker_seg.confidence > 0.85:
                    hysteresis_mult = 1.05
                elif active_speaker_seg and active_speaker_seg.confidence > 0.65:
                    hysteresis_mult = 1.12
                else:
                    hysteresis_mult = 1.20

            if best_score < current_score * (1.0 + stability_bonus) * hysteresis_mult:
                best_id = self._last_target_id

        self._last_target_id = best_id

        for face in fd.faces:
            if face.face_id == best_id:
                return face

        return fd.dominant_face()

    def _vertical_crop(self, face) -> float:
        return float(np.clip(face.box.cy - self.target_h * 0.4, 0, self.max_y))

    @staticmethod
    def _smooth_expr(x0: float, x1: float, t_start: float, t_end: float) -> str:
        if x0 == x1:
            return f"{x0:.4f}"

        dur = max(t_end - t_start, 1e-6)
        # clip((t - t_start) / dur, 0, 1) via nested min/max (ffmpeg eval has no clip())
        u = f"min(max((t-{t_start:.4f})/{dur:.4f}\\,0)\\,1)"
        ease = f"(({u})*({u})*(3-2*({u})))"
        return f"({x0:.4f}+({x1:.4f}-{x0:.4f})*{ease})"

    def _keyframes_to_ffmpeg_filter(
        self,
        keyframes: List[ReframeKeyframe],
        transition_sec: float = 0.4,
    ) -> str:
        W, H = self.target_w, self.target_h

        if not keyframes:
            return f"crop={W}:{H}:{self.center_x}:{self.center_y}"

        if len(keyframes) == 1:
            x = int(np.clip(keyframes[0].crop_x, 0, self.max_x))
            y = int(np.clip(keyframes[0].crop_y, 0, self.max_y))
            x -= x % 2
            y -= y % 2
            return f"crop={W}:{H}:{x}:{y}"

        # Snap every keyframe to integer, even-pixel bounds up front.
        pts: List[Tuple[float, int, int]] = []
        for kf in keyframes:
            x = int(np.clip(kf.crop_x, 0, self.max_x)); x -= x % 2
            y = int(np.clip(kf.crop_y, 0, self.max_y)); y -= y % 2
            pts.append((kf.time, x, y))

        # Base case: hold the final position once the last keyframe is reached.
        x_expr = str(pts[-1][1])
        y_expr = str(pts[-1][2])

        for i in range(len(pts) - 2, -1, -1):
            t0, x0, y0 = pts[i]
            t1, x1, y1 = pts[i + 1]

            if transition_sec <= 0.0:
                # Hard cut: hold k0 position until k1.time, then jump.
                x_expr = f"if(lt(t\\,{t1:.4f})\\,{x0}\\,{x_expr})"
                y_expr = f"if(lt(t\\,{t1:.4f})\\,{y0}\\,{y_expr})"
                continue

            window = min(transition_sec, max(t1 - t0, 0.0))
            t_start = t1 - window

            x_move = self._smooth_expr(x0, x1, t_start, t1)
            y_move = self._smooth_expr(y0, y1, t_start, t1)

            x_expr = f"if(lt(t\\,{t_start:.4f})\\,{x0}\\,if(lt(t\\,{t1:.4f})\\,{x_move}\\,{x_expr}))"
            y_expr = f"if(lt(t\\,{t_start:.4f})\\,{y0}\\,if(lt(t\\,{t1:.4f})\\,{y_move}\\,{y_expr}))"

        x_even = f"2*trunc(({x_expr})/2)"
        y_even = f"2*trunc(({y_expr})/2)"

        return f"crop={W}:{H}:{x_even}:{y_even}"


def build_active_speaker_crop_filter(
    face_timeline: List,
    speaker_segments: List,
    frame_w: int,
    frame_h: int,
    clip_duration: float,
    target_aspect: str = "9:16",
    smooth: bool = True,
    fps: float = 25.0,           # kept for API compatibility; easing is time-based (eval `t`), not frame-indexed
    transition_sec: float = 0.4,
) -> str:
    reframer = ActiveSpeakerReframer(
        frame_w=frame_w,
        frame_h=frame_h,
        target_aspect=target_aspect,
        vertical_center=True,
    )
    kf = reframer.build_keyframes(face_timeline, speaker_segments, clip_duration)
    return reframer._keyframes_to_ffmpeg_filter(
        kf,
        transition_sec=transition_sec if smooth else 0.0,
    )