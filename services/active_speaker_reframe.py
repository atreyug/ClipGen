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
        self._last_target_id = None
        # NEW: Track consecutive frames targeting same face for stability
        self._target_stability_counter = 0
        self._target_history: List[Optional[int]] = []

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
            elif fd.faces:
                # Use improved dominant_face() from updated multi_face_tracker
                fallback_face = fd.dominant_face()
                if fallback_face:
                    cx = int(np.clip(fallback_face.crop_x_smoothed, 0, self.max_x))
                    cy = int(self._vertical_crop(fallback_face)) if self.vertical_center else self.center_y
                    prev = (cx, cy)
                elif prev is not None:
                    cx, cy = prev
                else:
                    cx, cy = self.center_x, self.center_y
            elif prev is not None:
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

        stabilized = self._apply_hysteresis(raw)
        keyframes = self._build_adaptive(stabilized, clip_duration)

        # Ensure timeline coverage
        if keyframes and keyframes[-1].time < clip_duration - 0.01:
            keyframes.append(ReframeKeyframe(
                clip_duration, keyframes[-1].crop_x, keyframes[-1].crop_y
            ))
        if keyframes and keyframes[0].time > 0.01:
            keyframes.insert(0, ReframeKeyframe(
                0.0, keyframes[0].crop_x, keyframes[0].crop_y
            ))

        # Clamp times
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
        buf_size = 7

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
            if dx < 2 and dy < 2:
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
        """
        ENHANCED: Active speaker selection with Audio-Visual Consistency + improved metrics.
        
        Now leverages:
        - Improved face quality metrics (centrality_score, size_score, stability_score)
        - Better multi-person discrimination
        - Visual lip-sync validation to prevent audio diarization errors
        """
        
        # Find active audio speaker segment
        active_speaker_seg = None
        for seg in speaker_segments:
            if seg.start <= t <= seg.end and seg.face_id is not None:
                if active_speaker_seg is None or seg.confidence > active_speaker_seg.confidence:
                    active_speaker_seg = seg
            elif seg.start <= t <= seg.end and seg.face_id is None:
                if active_speaker_seg is None:
                    active_speaker_seg = seg

        # Visual validation: detect if someone is CLEARLY speaking (strong MAR)
        mars_in_frame = [f.mar for f in fd.faces if not getattr(f, 'is_profile', False)]
        max_mar_in_frame = max(mars_in_frame) if mars_in_frame else 0.0
        strong_visual_speaker_exists = max_mar_in_frame > 0.30

        # Score each face
        scores = {}
        for face in fd.faces:
            score = 0.0
            
            # ═══════════════════════════════════════════════════════════════
            # 1. AUDIO DIARIZATION ALIGNMENT (if available)
            # ═══════════════════════════════════════════════════════════════
            if active_speaker_seg and face.face_id == active_speaker_seg.face_id:
                # CRITICAL FIX: Validate audio assignment against visual evidence
                if strong_visual_speaker_exists and face.mar < 0.10 and not getattr(face, 'is_profile', False):
                    # Audio says this person, but their mouth isn't moving
                    # and someone else is clearly speaking → likely diarization error
                    score += 2.0 * active_speaker_seg.confidence  # Reduced weight
                    logger.debug(f"Frame {t:.2f}: Audio assigned to face {face.face_id} but MAR={face.mar:.3f} "
                                f"while max_MAR={max_mar_in_frame:.3f} - possible mismatch")
                else:
                    # Audio assignment seems visually consistent
                    score += 12.0 * active_speaker_seg.confidence  # Strong weight
            
            # Audio speaker unknown, check visual speaking
            if active_speaker_seg and active_speaker_seg.face_id is None:
                if face.mar > 0.25:
                    score += 8.0 * active_speaker_seg.confidence
            
            # ═══════════════════════════════════════════════════════════════
            # 2. VISUAL LIP-SYNC (MAR-based speaking detection)
            # ═══════════════════════════════════════════════════════════════
            if not getattr(face, 'is_profile', False):
                if face.mar > 0.25:
                    # Clear mouth opening → likely speaking
                    mar_strength = min(face.mar / 0.5, 1.0)
                    score += 6.0 * mar_strength  # Increased from 5.0
            
            # ═══════════════════════════════════════════════════════════════
            # 3. IMPROVED FACE QUALITY METRICS (from updated tracker)
            # ═══════════════════════════════════════════════════════════════
            
            # Size + centrality (from improved tracker)
            if hasattr(face, 'size_score') and hasattr(face, 'centrality_score'):
                # Prioritize larger, centered faces
                score += 3.0 * face.size_score  # Normalized 0-1
                score += 2.0 * face.centrality_score  # Normalized 0-1
            else:
                # Fallback for older tracker versions
                area_normalized = min(face.box.area / 100000.0, 1.0)
                score += 2.5 * area_normalized * face.box.confidence
                
                # Manual centrality calculation
                frame_center_x = self.frame_w / 2
                distance_from_center = abs(face.box.cx - frame_center_x)
                center_score = 1.0 - min(distance_from_center / (self.frame_w / 2), 1.0)
                score += 1.5 * center_score
            
            # Stability score (from improved tracker)
            if hasattr(face, 'stability_score'):
                score += 2.0 * face.stability_score
            else:
                # Fallback: track age
                if not face.is_coasted:
                    maturity = min(face.age / 30.0, 1.0)
                    score += 1.5 * maturity
            
            # ═══════════════════════════════════════════════════════════════
            # 4. DETECTION QUALITY
            # ═══════════════════════════════════════════════════════════════
            score += 1.0 * face.box.confidence  # MediaPipe detection confidence
            
            # ═══════════════════════════════════════════════════════════════
            # 5. PENALTIES
            # ═══════════════════════════════════════════════════════════════
            if face.is_coasted:
                score *= 0.25  # Heavy penalty for predicted/lost faces
            
            scores[face.face_id] = score

        # ═══════════════════════════════════════════════════════════════
        # SELECT BEST FACE WITH ENHANCED HYSTERESIS
        # ═══════════════════════════════════════════════════════════════
        if scores:
            best_id = max(scores, key=scores.get)
            best_score = scores[best_id]
            
            # Track target stability
            self._target_history.append(best_id)
            if len(self._target_history) > 10:
                self._target_history.pop(0)
            
            # Apply hysteresis to prevent camera ping-ponging
            if self._last_target_id is not None:
                if self._last_target_id in scores:
                    current_score = scores[self._last_target_id]
                    
                    # Calculate stability bonus
                    stability_frames = sum(1 for fid in self._target_history if fid == self._last_target_id)
                    stability_bonus = min(stability_frames / 10.0, 1.0) * 0.3
                    
                    # ADAPTIVE HYSTERESIS based on situation
                    if strong_visual_speaker_exists:
                        # Strong visual evidence exists
                        last_face_mar = next((f.mar for f in fd.faces if f.face_id == self._last_target_id), 0.0)
                        best_face_mar = next((f.mar for f in fd.faces if f.face_id == best_id), 0.0)
                        
                        if last_face_mar < 0.12 and best_face_mar > 0.30:
                            # Current target isn't speaking, new target clearly is → switch quickly
                            hysteresis_multiplier = 1.03
                            logger.debug(f"Frame {t:.2f}: Quick switch from {self._last_target_id} (MAR={last_face_mar:.3f}) "
                                        f"to {best_id} (MAR={best_face_mar:.3f})")
                        elif last_face_mar > 0.25:
                            # Current target is speaking → hold strongly
                            hysteresis_multiplier = 1.25
                        else:
                            hysteresis_multiplier = 1.12
                    else:
                        # No strong visual evidence, rely on audio + stability
                        if active_speaker_seg and active_speaker_seg.confidence > 0.85:
                            hysteresis_multiplier = 1.05
                        elif active_speaker_seg and active_speaker_seg.confidence > 0.65:
                            hysteresis_multiplier = 1.12
                        else:
                            hysteresis_multiplier = 1.20
                    
                    # Apply stability bonus
                    current_score_with_bonus = current_score * (1.0 + stability_bonus)
                    
                    if best_score < current_score_with_bonus * hysteresis_multiplier:
                        best_id = self._last_target_id
                    else:
                        logger.debug(f"Frame {t:.2f}: Switching target {self._last_target_id} → {best_id} "
                                    f"(scores: {current_score:.2f} → {best_score:.2f})")
            
            self._last_target_id = best_id
            
            for face in fd.faces:
                if face.face_id == best_id:
                    return face

        # Fallback: use improved dominant_face() method
        return fd.dominant_face()

    def _vertical_crop(self, face) -> float:
        # Center face in upper 40% of frame
        return float(np.clip(face.box.cy - self.target_h * 0.4, 0, self.max_y))

    def smooth_trajectory(
        self, keyframes: List[ReframeKeyframe], fps: float = 25.0
    ) -> List[ReframeKeyframe]:
        """
        Simplifies keyframes for FFmpeg compatibility WITHOUT destroying 
        the cinematic easing created by _build_adaptive.
        """
        if len(keyframes) < 2:
            return keyframes

        MIN_TIME_DELTA = 0.20
        MAX_KEYFRAMES = 80
        
        simplified = [keyframes[0]]
        for kf in keyframes[1:]:
            last = simplified[-1]
            time_diff = kf.time - last.time
            
            # Always keep last keyframe
            if kf is keyframes[-1]:
                simplified.append(kf)
                continue
                
            # Keep if time gap is significant
            if time_diff >= MIN_TIME_DELTA:
                simplified.append(kf)
            else:
                # Or if spatial movement is large
                dist = float(np.hypot(kf.crop_x - last.crop_x, kf.crop_y - last.crop_y))
                if dist > 150:
                    simplified.append(kf)

        # Downsample if still too many
        if len(simplified) > MAX_KEYFRAMES:
            idx = np.linspace(0, len(simplified) - 1, MAX_KEYFRAMES, dtype=int)
            idx[0] = 0
            idx[-1] = len(simplified) - 1
            simplified = [simplified[i] for i in idx]
            logger.info(f"Simplified trajectory to {len(simplified)} keyframes (was {len(keyframes)}) for FFmpeg compatibility")

        return simplified

    def _keyframes_to_ffmpeg_filter(self, keyframes: List[ReframeKeyframe]) -> str:
        W, H = self.target_w, self.target_h

        if not keyframes:
            return f"crop={W}:{H}:{self.center_x}:{self.center_y}"

        if len(keyframes) == 1:
            x = int(np.clip(keyframes[0].crop_x, 0, self.max_x))
            y = int(np.clip(keyframes[0].crop_y, 0, self.max_y))
            x -= x % 2
            y -= y % 2
            return f"crop={W}:{H}:{x}:{y}"

        # Build time-dependent expression (nested if statements)
        lx = float(np.clip(keyframes[-1].crop_x, 0, self.max_x))
        ly = float(np.clip(keyframes[-1].crop_y, 0, self.max_y))
        x_expr, y_expr = f"{lx:.2f}", f"{ly:.2f}"

        # Build from end to start
        for i in range(len(keyframes) - 2, -1, -1):
            k0, k1 = keyframes[i], keyframes[i + 1]
            dt = k1.time - k0.time
            if dt < 1e-4:
                continue

            x0 = float(np.clip(k0.crop_x, 0, self.max_x))
            x1 = float(np.clip(k1.crop_x, 0, self.max_x))
            y0 = float(np.clip(k0.crop_y, 0, self.max_y))
            y1 = float(np.clip(k1.crop_y, 0, self.max_y))

            sx = (x1 - x0) / dt
            sy = (y1 - y0) / dt

            x_lerp = f"max(0\\,min({x0:.2f}+{sx:.4f}*(t-{k0.time:.4f})\\,{self.max_x}))"
            y_lerp = f"max(0\\,min({y0:.2f}+{sy:.4f}*(t-{k0.time:.4f})\\,{self.max_y}))"

            x_expr = f"if(between(t\\,{k0.time:.4f}\\,{k1.time:.4f})\\,{x_lerp}\\,{x_expr})"
            y_expr = f"if(between(t\\,{k0.time:.4f}\\,{k1.time:.4f})\\,{y_lerp}\\,{y_expr})"

        # Ensure even coordinates for YUV420
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