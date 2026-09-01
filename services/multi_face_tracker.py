"""
services/multi_face_tracker.py
───────────────────────────────
DENSE per-frame face tracking using MediaPipe FaceLandmarker VIDEO mode.

KEY CHANGES:
- Tracks EVERY frame (not sampled) using detect_for_video() — MediaPipe's
  temporal tracker handles motion blur, head turns, partial occlusions.
- Timestamps in output timeline are CLIP-RELATIVE (0.0 .. duration).
  This matches FFmpeg crop expression `t` exactly.
- Coasting: briefly-lost faces hold/predict position instead of vanishing.
- MAR (mouth aspect ratio) computed per face per frame in the SAME pass —
  reused by speaker_detector for lip-sync analysis (no second video pass).
"""

from __future__ import annotations

import math
import logging
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False
    logger.warning("mediapipe not installed – MultiFaceTracker disabled.")


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FaceBox:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float = 1.0

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def area(self) -> float:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)


@dataclass
class TrackedFace:
    face_id: int
    box: FaceBox
    crop_x: float
    crop_x_smoothed: float
    disappeared: int = 0
    age: int = 0
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    mar: float = 0.0            # mouth aspect ratio (lip openness) this frame
    is_coasted: bool = False    # True if position carried forward (not detected)


@dataclass
class FrameFaceData:
    """One entry PER PROCESSED FRAME. timestamp is CLIP-RELATIVE seconds."""
    timestamp: float
    faces: List[TrackedFace]

    def dominant_face(self) -> Optional[TrackedFace]:
        if not self.faces:
            return None
        return max(self.faces, key=lambda f: f.box.area * f.box.confidence)


# ──────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────────────────────────────────────

def _iou(a: FaceBox, b: FaceBox) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


# MediaPipe FaceMesh landmark indices
_INNER_UPPER = [13, 312, 311, 310, 415, 308]
_INNER_LOWER = [14, 317, 402, 318, 324]
_LEFT_CORNER = 61
_RIGHT_CORNER = 291


def _landmarks_to_box(landmarks, w: int, h: int, expand: bool = True) -> Tuple[FaceBox, float]:
    """Tight bbox from 468 landmarks + mean visibility as confidence."""
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

    if expand:
        ex = int((x2 - x1) * 0.10)
        ey = int((y2 - y1) * 0.15)
        x1, y1 = max(0, x1 - ex), max(0, y1 - ey)
        x2, y2 = min(w, x2 + ex), min(h, y2 + ey)

    vis = np.mean([
        lm.visibility if hasattr(lm, "visibility") and lm.visibility is not None else 1.0
        for lm in landmarks
    ])
    conf = float(np.clip(vis, 0.3, 1.0))
    return FaceBox(x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2), confidence=conf), conf


def _compute_mar(landmarks) -> float:
    """Mouth Aspect Ratio — inner lip opening / mouth width (scale-invariant)."""
    try:
        def pt(i):
            return np.array([landmarks[i].x, landmarks[i].y])

        upper = np.mean([pt(i) for i in _INNER_UPPER], axis=0)
        lower = np.mean([pt(i) for i in _INNER_LOWER], axis=0)
        left = pt(_LEFT_CORNER)
        right = pt(_RIGHT_CORNER)

        vertical = np.linalg.norm(upper - lower)
        horizontal = np.linalg.norm(left - right) + 1e-6
        return float(vertical / horizontal)
    except Exception:
        return 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Dense-frame IoU tracker (IDs locked by near-consecutive IoU ≈ 1.0)
# ──────────────────────────────────────────────────────────────────────────────

class FaceIDTracker:
    """
    IoU-based identity tracker for DENSE frames.
    Because consecutive frames barely move, IoU matching is near-perfect.
    Supports coasting: lost faces keep last box + velocity prediction.
    """

    def __init__(
        self,
        max_disappeared: int = 30,
        iou_threshold: float = 0.30,
        smoothing_factor: float = 0.35,
    ):
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold
        self.smoothing = smoothing_factor
        self._next_id = 0
        self._faces: Dict[int, TrackedFace] = OrderedDict()

    def update(
        self,
        detections: List[FaceBox],
        mars: List[float],
        frame_w: int,
        frame_h: int,
        target_w: int,
    ) -> List[TrackedFace]:
        if not self._faces:
            for det, mar in zip(detections, mars):
                self._register(det, mar, frame_w, frame_h, target_w)
            return list(self._faces.values())

        if not detections:
            for tf in self._faces.values():
                tf.disappeared += 1
                tf.is_coasted = True
                self._coast(tf, frame_w, target_w)
            return list(self._faces.values())

        ids = list(self._faces.keys())
        cost = np.zeros((len(ids), len(detections)), dtype=float)

        for i, fid in enumerate(ids):
            for j, det in enumerate(detections):
                cost[i, j] = _iou(self._faces[fid].box, det)

        # Greedy match (dense frames → IoU matrix is nearly diagonal, greedy = optimal here)
        matched_ids, matched_dets = set(), set()
        while cost.max() >= self.iou_threshold:
            i, j = np.unravel_index(cost.argmax(), cost.shape)
            fid = ids[i]
            if fid not in matched_ids and j not in matched_dets:
                self._update_face(fid, detections[j], mars[j], frame_w, frame_h, target_w)
                matched_ids.add(fid)
                matched_dets.add(j)
            cost[i, :] = -1
            cost[:, j] = -1

        for fid in ids:
            if fid not in matched_ids:
                tf = self._faces[fid]
                tf.disappeared += 1
                tf.is_coasted = True
                self._coast(tf, frame_w, target_w)
                if tf.disappeared > self.max_disappeared:
                    del self._faces[fid]

        for j, det in enumerate(detections):
            if j not in matched_dets:
                self._register(det, mars[j], frame_w, frame_h, target_w)

        return list(self._faces.values())

    def _coast(self, tf: TrackedFace, frame_w: int, target_w: int) -> None:
        """Predict position forward using velocity (clamped)."""
        pred_cx = tf.box.cx + tf.velocity_x * tf.disappeared
        half = target_w / 2
        new_crop = float(np.clip(pred_cx - half, 0, frame_w - target_w))
        # light smoothing while coasting to avoid drift jitter
        tf.crop_x = new_crop
        tf.crop_x_smoothed = 0.15 * new_crop + 0.85 * tf.crop_x_smoothed

    def reset(self) -> None:
        self._faces.clear()
        self._next_id = 0

    def _register(self, box: FaceBox, mar: float, frame_w: int, frame_h: int, target_w: int) -> int:
        cx = self._crop_x(box, frame_w, target_w)
        tf = TrackedFace(
            face_id=self._next_id,
            box=box,
            crop_x=cx,
            crop_x_smoothed=cx,
            mar=mar,
        )
        self._faces[self._next_id] = tf
        self._next_id += 1
        return tf.face_id

    def _update_face(self, fid: int, box: FaceBox, mar: float,
                     frame_w: int, frame_h: int, target_w: int) -> None:
        tf = self._faces[fid]
        # velocity from consecutive dense frames
        tf.velocity_x = 0.6 * (box.cx - tf.box.cx) + 0.4 * tf.velocity_x
        tf.velocity_y = 0.6 * (box.cy - tf.box.cy) + 0.4 * tf.velocity_y

        tf.box = box
        tf.disappeared = 0
        tf.age += 1
        tf.is_coasted = False
        tf.mar = mar

        new_cx = self._crop_x(box, frame_w, target_w)
        tf.crop_x = new_cx
        tf.crop_x_smoothed = self.smoothing * new_cx + (1 - self.smoothing) * tf.crop_x_smoothed

    @staticmethod
    def _crop_x(box: FaceBox, frame_w: int, target_w: int) -> float:
        half = target_w / 2
        return float(np.clip(box.cx - half, 0, max(0, frame_w - target_w)))


# ──────────────────────────────────────────────────────────────────────────────
# MultiFaceTracker — dense per-frame VIDEO-mode pipeline
# ──────────────────────────────────────────────────────────────────────────────

class MultiFaceTracker:
    """
    Tracks faces at EVERY frame of the clip window using MediaPipe
    FaceLandmarker in VIDEO running mode (temporal tracking).

    Output timeline:
      - One FrameFaceData per processed frame
      - timestamp = clip-relative seconds (0.0 .. duration)
      - faces include coasted entries (is_coasted=True) so downstream
        crop NEVER falls back to center mid-clip
      - each face carries `mar` for lip-sync analysis
    """

    # Lip landmarks for MAR (module-level reuse)
    _INNER_UPPER = _INNER_UPPER
    _INNER_LOWER = _INNER_LOWER

    def __init__(
        self,
        model_path: Optional[str] = None,
        sample_interval: float = 0.5,      # kept for API compat (ignored in dense mode)
        max_faces: int = 4,
        min_confidence: float = 0.4,
        smoothing_factor: float = 0.35,
        max_disappeared: Optional[int] = None,   # auto-set from fps
        lip_analysis: bool = True,
        infer_width: int = 960,            # downscale for inference speed
    ):
        self.sample_interval = sample_interval
        self.lip_analysis = lip_analysis
        self.infer_width = infer_width
        self._landmarker = None
        self._video_mode = False

        if not _MP_AVAILABLE:
            logger.error("MediaPipe unavailable — tracking disabled.")
            return

        if model_path is None:
            model_path = str(
                Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"
            )
        if not Path(model_path).exists():
            logger.error("face_landmarker.task not found at %s", model_path)
            return

        self._tracker = FaceIDTracker(
            max_disappeared=max_disappeared or 30,
            smoothing_factor=smoothing_factor,
        )
        self._model_path = model_path
        self._max_faces = max_faces
        self._min_confidence = min_confidence

    def _init_landmarker(self, fps: float):
        """VIDEO mode with temporal tracking; fall back to IMAGE mode."""
        try:
            base = python.BaseOptions(model_asset_path=self._model_path)
            opts = vision.FaceLandmarkerOptions(
                base_options=base,
                running_mode=vision.RunningMode.VIDEO,
                num_faces=self._max_faces,
                min_face_detection_confidence=self._min_confidence,
                min_face_presence_confidence=self._min_confidence,
                min_tracking_confidence=0.5,   # the magic: temporal tracking
            )
            self._landmarker = vision.FaceLandmarker.create_from_options(opts)
            self._video_mode = True
            logger.info("FaceLandmarker VIDEO mode (per-frame temporal tracking)")
        except Exception as e:
            logger.warning("VIDEO mode failed (%s) — falling back to IMAGE mode", e)
            try:
                base = python.BaseOptions(model_asset_path=self._model_path)
                opts = vision.FaceLandmarkerOptions(
                    base_options=base,
                    running_mode=vision.RunningMode.IMAGE,
                    num_faces=self._max_faces,
                    min_face_detection_confidence=self._min_confidence,
                )
                self._landmarker = vision.FaceLandmarker.create_from_options(opts)
                self._video_mode = False
            except Exception as e2:
                logger.error("Landmarker init failed entirely: %s", e2)
                self._landmarker = None

        # Auto-tune coasting: ~1 second of frames
        self._tracker.max_disappeared = max(15, int(fps * 1.0))

    def process_video(
        self,
        video_path: str,
        clip_start: float = 0.0,
        clip_end: Optional[float] = None,
        target_aspect: str = "9:16",
    ) -> List[FrameFaceData]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Cannot open video: %s", video_path)
            return []

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if fps <= 0 or fps != fps:  # NaN guard
            fps = 25.0
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if target_aspect == "9:16":
            target_w = int(frame_h * 9 / 16)
        else:
            target_w = frame_w
        target_w = max(2, min(target_w, frame_w))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if total_frames > 0 else 0
        end = min(clip_end, duration) if clip_end else duration

        start_frame = max(0, int(clip_start * fps))
        end_frame = int(end * fps)

        # Adaptive stride: dense (1) for normal clips; larger for very long clips
        frames_in_window = max(1, end_frame - start_frame)
        stride = 1
        if frames_in_window > 2700:  # >~90s @30fps
            stride = int(math.ceil(frames_in_window / 2700))
            logger.info("Long clip — using frame stride %d", stride)

        self._init_landmarker(fps)
        if self._landmarker is None:
            cap.release()
            return []

        self._tracker.reset()
        timeline: List[FrameFaceData] = []

        infer_scale = 1.0
        if frame_w > self.infer_width:
            infer_scale = self.infer_width / frame_w

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_idx = start_frame
        last_ts_ms = -1
        reached_end = False

        while frame_idx <= end_frame and not reached_end:
            ret, frame = cap.read()
            if not ret:
                break

            ts_abs = frame_idx / fps
            ts_rel = max(0.0, ts_abs - clip_start)

            # MediaPipe VIDEO mode needs strictly increasing ms timestamps
            ts_ms = int(round(ts_abs * 1000))
            if ts_ms <= last_ts_ms:
                ts_ms = last_ts_ms + 1
            last_ts_ms = ts_ms

            # Downscale for inference (landmarks are normalized → scale back after)
            if infer_scale < 1.0:
                small = cv2.resize(
                    frame,
                    (int(frame_w * infer_scale), int(frame_h * infer_scale)),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                small = frame
            sw, sh = small.shape[1], small.shape[0]

            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            detections: List[FaceBox] = []
            mars: List[float] = []
            try:
                if self._video_mode:
                    result = self._landmarker.detect_for_video(mp_image, ts_ms)
                else:
                    result = self._landmarker.detect(mp_image)

                if result and result.face_landmarks:
                    for lms in result.face_landmarks:
                        box, conf = _landmarks_to_box(lms, sw, sh)
                        # scale back to full-res coordinates
                        if infer_scale < 1.0:
                            inv = 1.0 / infer_scale
                            box = FaceBox(
                                x1=int(box.x1 * inv), y1=int(box.y1 * inv),
                                x2=int(box.x2 * inv), y2=int(box.y2 * inv),
                                confidence=conf,
                            )
                        detections.append(box)
                        mars.append(_compute_mar(lms) if self.lip_analysis else 0.0)
            except Exception as exc:
                logger.debug("Landmarker error @%.2fs: %s", ts_rel, exc)

            tracked = self._tracker.update(
                detections, mars, frame_w, frame_h, target_w
            )

            # Include coasted faces so timeline NEVER has holes
            visible = [tf for tf in tracked if tf.disappeared <= self._tracker.max_disappeared]

            timeline.append(FrameFaceData(
                timestamp=ts_rel,
                faces=[
                    TrackedFace(
                        face_id=tf.face_id,
                        box=FaceBox(tf.box.x1, tf.box.y1, tf.box.x2, tf.box.y2, tf.box.confidence),
                        crop_x=tf.crop_x,
                        crop_x_smoothed=tf.crop_x_smoothed,
                        disappeared=tf.disappeared,
                        age=tf.age,
                        velocity_x=tf.velocity_x,
                        velocity_y=tf.velocity_y,
                        mar=tf.mar,
                        is_coasted=tf.is_coasted,
                    )
                    for tf in visible
                ],
            ))

            # Advance
            frame_idx += 1
            for _ in range(stride - 1):
                if not cap.grab():
                    reached_end = True
                    break
                frame_idx += 1

        cap.release()
        try:
            self._landmarker.close()
        except Exception:
            pass
        self._landmarker = None

        timeline = self._smooth_timeline(timeline)

        ids_seen = {f.face_id for fd in timeline for f in fd.faces}
        coasted = sum(1 for fd in timeline for f in fd.faces if f.is_coasted)
        logger.info(
            "[Face Tracking] frames=%d unique_ids=%d coasted_entries=%d span=[%.2f..%.2f]s",
            len(timeline), len(ids_seen), coasted,
            timeline[0].timestamp if timeline else -1,
            timeline[-1].timestamp if timeline else -1,
        )
        return timeline

    def _smooth_timeline(self, timeline: List[FrameFaceData]) -> List[FrameFaceData]:
        """Per-face median filter on crop positions (dense data → removes spikes)."""
        if len(timeline) < 5:
            return timeline

        by_face: Dict[int, List[Tuple[int, float]]] = {}
        for idx, fd in enumerate(timeline):
            for face in fd.faces:
                by_face.setdefault(face.face_id, []).append((idx, face.crop_x_smoothed))

        window = max(3, int(len(timeline) * 0.01))
        if window % 2 == 0:
            window += 1
        half = window // 2

        for fid, pts in by_face.items():
            if len(pts) < window:
                continue
            xs = np.array([p[1] for p in pts])
            smoothed = xs.copy()
            for i in range(half, len(xs) - half):
                smoothed[i] = np.median(xs[i - half:i + half + 1])
            for (frame_i, _), nx in zip(pts, smoothed):
                for face in timeline[frame_i].faces:
                    if face.face_id == fid and not face.is_coasted:
                        face.crop_x_smoothed = float(nx)
        return timeline

    def close(self) -> None:
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:
                pass
            self._landmarker = None


# ──────────────────────────────────────────────────────────────────────────────
# Convenience helpers
# ──────────────────────────────────────────────────────────────────────────────

def positions_for_face(
    face_timeline: List[FrameFaceData],
    face_id: int,
) -> List[Tuple[float, float]]:
    """Extract (clip_relative_time, smoothed_crop_x) for one face from a dense timeline."""
    return [
        (fd.timestamp, f.crop_x_smoothed)
        for fd in face_timeline
        for f in fd.faces
        if f.face_id == face_id
    ]


def get_multi_face_crop_positions(
    video_path: str,
    clip_start: float,
    clip_end: float,
    frame_width: int,
    frame_height: int,
    target_aspect: str = "9:16",
    sample_interval: float = 0.5,
    smoothing_factor: float = 0.35,
    max_faces: int = 4,
) -> Dict[int, List[Tuple[float, float]]]:
    """
    Backward-compatible helper. Timestamps returned are CLIP-RELATIVE.
    NOTE: prefer running MultiFaceTracker once and using positions_for_face()
    instead of calling this (it re-decodes the video).
    """
    tracker = MultiFaceTracker(
        sample_interval=sample_interval,
        max_faces=max_faces,
        smoothing_factor=smoothing_factor,
    )
    try:
        timeline = tracker.process_video(
            video_path, clip_start=clip_start, clip_end=clip_end,
            target_aspect=target_aspect,
        )
        positions: Dict[int, List[Tuple[float, float]]] = {}
        for fd in timeline:
            for face in fd.faces:
                positions.setdefault(face.face_id, []).append(
                    (fd.timestamp, face.crop_x_smoothed)
                )
        return positions
    finally:
        tracker.close()


def build_multi_crop_expression(
    positions: List[Tuple[float, float]],
    frame_width: int,
    target_width: int,
    frame_height: int,
    target_height: int,
) -> str:
    """Time-dependent crop expression. Positions are (clip_relative_t, crop_x)."""
    if not positions:
        default_x = max(0, (frame_width - target_width) // 2)
        default_y = max(0, (frame_height - target_height) // 2)
        return f"crop={target_width}:{target_height}:{default_x}:{default_y}"

    if len(positions) == 1:
        x = int(np.clip(positions[0][1], 0, frame_width - target_width))
        y = max(0, (frame_height - target_height) // 2)
        return f"crop={target_width}:{target_height}:{x}:{y}"

    y = max(0, (frame_height - target_height) // 2)
    max_x = frame_width - target_width
    expr_parts = []

    for i in range(len(positions) - 1):
        t0, x0 = positions[i]
        t1, x1 = positions[i + 1]
        x0 = int(np.clip(x0, 0, max_x))
        x1 = int(np.clip(x1, 0, max_x))
        if abs(t1 - t0) < 1e-6:
            continue
        slope = (x1 - x0) / (t1 - t0)
        interp = f"{x0}+({slope:.4f}*(t-{t0:.4f}))"
        clamped = f"max(0\\,min({interp}\\,{max_x}))"
        expr_parts.append(f"if(between(t\\,{t0:.4f}\\,{t1:.4f})\\,{clamped})")

    if not expr_parts:
        x = int(np.clip(positions[-1][1], 0, max_x))
        return f"crop={target_width}:{target_height}:{x}:{y}"

    last_x = int(np.clip(positions[-1][1], 0, max_x))
    expr_parts.append(str(last_x))
    return f"crop={target_width}:{target_height}:(" + "\\,".join(expr_parts) + f"):{y}"