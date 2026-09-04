"""
services/multi_face_tracker.py
───────────────────────────────
DENSE per-frame face tracking using MediaPipe FaceLandmarker VIDEO mode.

KEY CHANGES:
- Fixed confidence scoring to prioritize faces by actual relevance (size, position, stability)
- Better dominant face selection in multi-person scenarios
- Speaking activity now influences face priority
- Added hysteresis to prevent flip-flopping between similar faces
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
    mar: float = 0.0
    is_coasted: bool = False
    # Better priority metrics
    centrality_score: float = 0.0  # How centered the face is
    size_score: float = 0.0        # Relative size of the face
    stability_score: float = 1.0   # How stable tracking has been


@dataclass
class FrameFaceData:
    """One entry PER PROCESSED FRAME. timestamp is CLIP-RELATIVE seconds."""
    timestamp: float
    faces: List[TrackedFace]

    def dominant_face(self, prev_dominant_id: Optional[int] = None, hysteresis: float = 0.25) -> Optional[TrackedFace]:
        """
        Select dominant face with hysteresis to prevent flip-flopping.
        
        Args:
            prev_dominant_id: The face_id that was dominant in the previous frame
            hysteresis: Bonus score given to the previous winner (default 0.25)
        """
        if not self.faces:
            return None

        # Exclude coasted faces if we have real detections
        real_faces = [f for f in self.faces if not f.is_coasted]
        candidates = real_faces if real_faces else self.faces

        # Multi-factor priority scoring with hysteresis
        def priority_score(f: TrackedFace) -> float:
            # Base score from detection quality
            base = f.box.confidence
            
            # Size matters - larger faces are usually the subject (INCREASED)
            size = f.size_score * 2.5  # Increased from 2.0
            
            # Centrality - faces near center are often the subject
            centrality = f.centrality_score * 1.5
            
            # Stability - faces that have been tracked longer are more reliable (INCREASED)
            stability = min(f.stability_score, 1.0) * 1.0  # Increased from 0.8
            
            # Age bonus - prefer established tracks over new detections (INCREASED)
            age_bonus = min(f.age / 30.0, 1.0) * 0.8  # Increased from 0.5
            
            # Penalize disappeared/coasted faces (MORE AGGRESSIVE)
            disappear_penalty = max(0, 1.0 - (f.disappeared * 0.15))  # Increased from 0.1
            
            # REDUCED speaking bonus to prevent wild swings
            speaking_bonus = 0.0
            if f.mar > 0.08:
                speaking_bonus = min((f.mar - 0.08) / 0.22, 1.0) * 2.0  # REDUCED from 5.0
            
            # HYSTERESIS: Give previous winner a sticky bonus to prevent flip-flopping
            winner_bonus = hysteresis if (prev_dominant_id is not None and f.face_id == prev_dominant_id) else 0.0

            return (base + size + centrality + stability + age_bonus + speaking_bonus + winner_bonus) * disappear_penalty

        return max(candidates, key=priority_score)


# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────

def _iou(a: FaceBox, b: FaceBox) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


# Mouth landmarks for MAR calculation
_INNER_UPPER = [13, 312, 311, 310, 415, 308]
_INNER_LOWER = [14, 317, 402, 318, 324]
_LEFT_CORNER = 61
_RIGHT_CORNER = 291


def _landmarks_to_box(landmarks, w: int, h: int, expand: bool = True) -> Tuple[FaceBox, Dict[str, float]]:
    """
    Tight bbox from 468 landmarks + quality metrics.
    
    Returns:
        (FaceBox, metrics_dict) where metrics contains:
        - 'detection_confidence': base detection quality
        - 'centrality': how centered the face is (0-1)
        - 'size_ratio': relative size of face in frame (0-1)
    """
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

    if expand:
        ex = int((x2 - x1) * 0.10)
        ey = int((y2 - y1) * 0.15)
        x1, y1 = max(0, x1 - ex), max(0, y1 - ey)
        x2, y2 = min(w, x2 + ex), min(h, y2 + ey)

    # Base detection confidence from landmark visibility
    vis = np.mean([
        lm.visibility if hasattr(lm, "visibility") and lm.visibility is not None else 1.0
        for lm in landmarks
    ])
    detection_conf = float(np.clip(vis, 0.3, 1.0))
    
    # Calculate quality metrics
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    
    # Centrality: how close to frame center (1.0 = perfectly centered)
    frame_cx, frame_cy = w / 2.0, h / 2.0
    dist_from_center = np.hypot(cx - frame_cx, cy - frame_cy)
    max_dist = np.hypot(w / 2.0, h / 2.0)
    centrality = 1.0 - min(dist_from_center / max_dist, 1.0)
    
    # Size ratio: face area relative to frame area
    face_area = max(0, x2 - x1) * max(0, y2 - y1)
    frame_area = w * h
    size_ratio = min(face_area / frame_area, 1.0)
    
    metrics = {
        'detection_confidence': detection_conf,
        'centrality': centrality,
        'size_ratio': size_ratio,
    }
    
    box = FaceBox(
        x1=int(x1), 
        y1=int(y1), 
        x2=int(x2), 
        y2=int(y2), 
        confidence=detection_conf
    )
    
    return box, metrics


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
# Enhanced face tracking with better multi-person handling
# ──────────────────────────────────────────────────────────────────────────────

class FaceIDTracker:
    """
    Enhanced IoU-based identity tracker with better multi-face discrimination.
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
        self._speaking_score: Dict[int, float] = {}

    def update(
        self,
        detections: List[FaceBox],
        mars: List[float],
        metrics: List[Dict[str, float]],
        frame_w: int,
        frame_h: int,
        target_w: int,
    ) -> List[TrackedFace]:
        if not self._faces:
            for det, mar, met in zip(detections, mars, metrics):
                self._register(det, mar, met, frame_w, frame_h, target_w)
            return list(self._faces.values())

        if not detections:
            for tf in self._faces.values():
                tf.disappeared += 1
                tf.is_coasted = True
                tf.stability_score *= 0.9  # Decay stability when coasting
                self._coast(tf, frame_w, target_w)
                self._speaking_score[tf.face_id] *= 0.95
            return list(self._faces.values())

        ids = list(self._faces.keys())
        
        # Build cost matrix for Hungarian matching
        cost = np.zeros((len(ids), len(detections)), dtype=float)
        
        for i, fid in enumerate(ids):
            tf = self._faces[fid]
            for j, det in enumerate(detections):
                iou_score = _iou(tf.box, det)
                
                # Spatial proximity for lost tracks
                if iou_score < self.iou_threshold:
                    pred_cx = tf.box.cx + tf.velocity_x
                    pred_cy = tf.box.cy + tf.velocity_y
                    centroid_dist = np.hypot(det.cx - pred_cx, det.cy - pred_cy)
                    max_centroid_dist = min(200, max(frame_w, frame_h) * 0.15)
                    
                    if centroid_dist < max_centroid_dist:
                        proximity_score = (1.0 - centroid_dist / max_centroid_dist) * 0.4
                        iou_score = max(iou_score, proximity_score)
                
                # Size consistency
                size_ratio = min(det.area, tf.box.area) / (max(det.area, tf.box.area) + 1e-6)
                size_bonus = size_ratio * 0.2
                
                # Position prediction
                pred_cx = tf.box.cx + tf.velocity_x
                pred_cy = tf.box.cy + tf.velocity_y
                dist = np.hypot(det.cx - pred_cx, det.cy - pred_cy)
                max_dist = np.hypot(frame_w, frame_h)
                position_bonus = (1.0 - min(dist / max_dist, 1.0)) * 0.15
                
                cost[i, j] = iou_score + size_bonus + position_bonus
        
        # Hungarian assignment
        try:
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(-cost)
            matched_pairs = [
                (ids[i], j) for i, j in zip(row_ind, col_ind)
                if cost[i, j] >= self.iou_threshold
            ]
        except ImportError:
            # Greedy fallback
            matched_pairs = []
            matched_ids_set, matched_dets_set = set(), set()
            while cost.max() >= self.iou_threshold:
                i, j = np.unravel_index(cost.argmax(), cost.shape)
                fid = ids[i]
                if fid not in matched_ids_set and j not in matched_dets_set:
                    matched_pairs.append((fid, j))
                    matched_ids_set.add(fid)
                    matched_dets_set.add(j)
                cost[i, :] = -1
                cost[:, j] = -1

        matched_ids = {fid for fid, _ in matched_pairs}
        matched_dets = {j for _, j in matched_pairs}

        # Update matched faces
        for fid, j in matched_pairs:
            self._update_face(fid, detections[j], mars[j], metrics[j], frame_w, frame_h, target_w)

        # Handle unmatched existing faces
        for fid in ids:
            if fid not in matched_ids:
                tf = self._faces[fid]
                tf.disappeared += 1
                tf.is_coasted = True
                tf.stability_score *= 0.85  # Decay stability
                self._coast(tf, frame_w, target_w)
                if tf.disappeared > self.max_disappeared:
                    del self._faces[fid]
                    if fid in self._speaking_score:
                        del self._speaking_score[fid]

        # Register new faces
        for j, det in enumerate(detections):
            if j not in matched_dets:
                self._register(det, mars[j], metrics[j], frame_w, frame_h, target_w)

        return list(self._faces.values())

    def _coast(self, tf: TrackedFace, frame_w: int, target_w: int) -> None:
        """Predict position forward using velocity (clamped)."""
        pred_cx = tf.box.cx + tf.velocity_x * tf.disappeared
        half = target_w / 2
        new_crop = float(np.clip(pred_cx - half, 0, frame_w - target_w))
        tf.crop_x = new_crop
        tf.crop_x_smoothed = 0.15 * new_crop + 0.85 * tf.crop_x_smoothed

    def reset(self) -> None:
        self._faces.clear()
        self._next_id = 0
        self._speaking_score.clear()

    def _register(self, box: FaceBox, mar: float, metrics: Dict[str, float],
                  frame_w: int, frame_h: int, target_w: int) -> int:
        cx = self._crop_x(box, frame_w, target_w)
        tf = TrackedFace(
            face_id=self._next_id,
            box=box,
            crop_x=cx,
            crop_x_smoothed=cx,
            mar=mar,
            centrality_score=metrics.get('centrality', 0.5),
            size_score=metrics.get('size_ratio', 0.5),
            stability_score=1.0,  # New tracks start stable
        )
        self._faces[self._next_id] = tf
        self._speaking_score[self._next_id] = 0.0
        self._next_id += 1
        return tf.face_id

    def _update_face(self, fid: int, box: FaceBox, mar: float, metrics: Dict[str, float],
                     frame_w: int, frame_h: int, target_w: int) -> None:
        tf = self._faces[fid]
        
        # Update velocity
        tf.velocity_x = 0.7 * (box.cx - tf.box.cx) + 0.3 * tf.velocity_x
        tf.velocity_y = 0.7 * (box.cy - tf.box.cy) + 0.3 * tf.velocity_y

        tf.box = box
        tf.disappeared = 0
        tf.age += 1
        tf.is_coasted = False
        tf.mar = mar
        
        # Update quality metrics
        tf.centrality_score = metrics.get('centrality', tf.centrality_score)
        tf.size_score = metrics.get('size_ratio', tf.size_score)
        
        # Improve stability with successful updates
        tf.stability_score = min(1.0, tf.stability_score * 0.95 + 0.05)

        # Update speaking score
        current_speaking = 1.0 if mar > 0.25 else 0.0
        self._speaking_score[fid] = 0.8 * self._speaking_score.get(fid, 0.0) + 0.2 * current_speaking

        new_cx = self._crop_x(box, frame_w, target_w)
        tf.crop_x = new_cx
        tf.crop_x_smoothed = self.smoothing * new_cx + (1 - self.smoothing) * tf.crop_x_smoothed

    def get_speaking_score(self, face_id: int) -> float:
        """Get cumulative speaking activity score for a face."""
        return self._speaking_score.get(face_id, 0.0)

    @staticmethod
    def _crop_x(box: FaceBox, frame_w: int, target_w: int) -> float:
        half = target_w / 2
        return float(np.clip(box.cx - half, 0, max(0, frame_w - target_w)))


# ──────────────────────────────────────────────────────────────────────────────
# Main tracker class
# ──────────────────────────────────────────────────────────────────────────────

class MultiFaceTracker:
    """
    Tracks faces at EVERY frame of the clip window using MediaPipe
    FaceLandmarker in VIDEO running mode (temporal tracking).
    """

    _INNER_UPPER = _INNER_UPPER
    _INNER_LOWER = _INNER_LOWER

    def __init__(
        self,
        model_path: Optional[str] = None,
        sample_interval: float = 0.5,
        max_faces: int = 4,
        min_confidence: float = 0.25,
        smoothing_factor: float = 0.35,
        max_disappeared: Optional[int] = None,
        lip_analysis: bool = True,
        infer_width: int = 1600,
    ):
        self.sample_interval = sample_interval
        self.lip_analysis = lip_analysis
        self.infer_width = infer_width
        self._landmarker = None
        self._fallback_detector = None
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
                min_tracking_confidence=0.5,
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

        self._tracker.max_disappeared = max(15, int(fps * 1.0))

    def _init_fallback_detector(self) -> None:
        """Initialize the far-face detector used when landmarks are missed."""
        if self._fallback_detector is not None:
            return
        model_path = (
            Path(__file__).resolve().parent.parent / "models" / "large_far.tflite"
        )
        if not model_path.exists():
            logger.warning("Far-face detector model not found at %s", model_path)
            return
        try:
            base = python.BaseOptions(model_asset_path=str(model_path))
            options = vision.FaceDetectorOptions(
                base_options=base,
                running_mode=vision.RunningMode.IMAGE,
                min_detection_confidence=self._min_confidence,
            )
            self._fallback_detector = vision.FaceDetector.create_from_options(options)
            logger.info("Far-face detector enabled for landmark misses")
        except Exception as exc:
            logger.warning("Could not initialize far-face detector: %s", exc)
            self._fallback_detector = None

    def _detect_far_faces(
        self,
        image,
        width: int,
        height: int,
    ) -> List[FaceBox]:
        self._init_fallback_detector()
        if self._fallback_detector is None:
            return []
        try:
            result = self._fallback_detector.detect(image)
        except Exception as exc:
            logger.debug("Far-face detector failed: %s", exc)
            return []

        boxes = []
        for detection in getattr(result, "detections", []) or []:
            bbox = detection.bounding_box
            x1 = max(0, int(bbox.origin_x))
            y1 = max(0, int(bbox.origin_y))
            x2 = min(width, x1 + int(bbox.width))
            y2 = min(height, y1 + int(bbox.height))
            if x2 <= x1 or y2 <= y1:
                continue
            categories = getattr(detection, "categories", None) or []
            confidence = float(categories[0].score) if categories else 1.0
            boxes.append(FaceBox(x1, y1, x2, y2, confidence))
        return boxes

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
        if fps <= 0 or fps != fps:
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

        frames_in_window = max(1, end_frame - start_frame)
        stride = 1
        if frames_in_window > 2700:
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

            ts_ms = int(round(ts_abs * 1000))
            if ts_ms <= last_ts_ms:
                ts_ms = last_ts_ms + 1
            last_ts_ms = ts_ms

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
            metrics_list: List[Dict[str, float]] = []
            
            try:
                if self._video_mode:
                    result = self._landmarker.detect_for_video(mp_image, ts_ms)
                else:
                    result = self._landmarker.detect(mp_image)

                if result and result.face_landmarks:
                    for lms in result.face_landmarks:
                        box, metrics = _landmarks_to_box(lms, sw, sh)
                        
                        # Scale back to original resolution
                        if infer_scale < 1.0:
                            inv = 1.0 / infer_scale
                            box = FaceBox(
                                x1=int(box.x1 * inv), y1=int(box.y1 * inv),
                                x2=int(box.x2 * inv), y2=int(box.y2 * inv),
                                confidence=box.confidence,
                            )
                        
                        detections.append(box)
                        mars.append(_compute_mar(lms) if self.lip_analysis else 0.0)
                        metrics_list.append(metrics)
                        
            except Exception as exc:
                logger.debug("Landmarker error @%.2fs: %s", ts_rel, exc)

            # FaceLandmarker often misses smaller faces in wide interview
            # shots. Fill missing detections with the dedicated far-face
            # detector so the crop never silently falls back to empty center.
            if len(detections) < self._max_faces:
                for fallback_box in self._detect_far_faces(mp_image, sw, sh):
                    metrics = {
                        "detection_confidence": fallback_box.confidence,
                        "centrality": 1.0 - min(
                            np.hypot(fallback_box.cx - sw / 2, fallback_box.cy - sh / 2)
                            / max(np.hypot(sw / 2, sh / 2), 1.0),
                            1.0,
                        ),
                        "size_ratio": min(fallback_box.area / max(sw * sh, 1), 1.0),
                    }

                    if infer_scale < 1.0:
                        inv = 1.0 / infer_scale
                        fallback_box = FaceBox(
                            x1=int(fallback_box.x1 * inv),
                            y1=int(fallback_box.y1 * inv),
                            x2=int(fallback_box.x2 * inv),
                            y2=int(fallback_box.y2 * inv),
                            confidence=fallback_box.confidence,
                        )

                    if any(_iou(fallback_box, existing) > 0.35 for existing in detections):
                        continue

                    detections.append(fallback_box)
                    mars.append(0.0)
                    metrics_list.append(metrics)
                    if len(detections) >= self._max_faces:
                        break

            tracked = self._tracker.update(
                detections, mars, metrics_list, frame_w, frame_h, target_w
            )

            # Keep only recently visible faces
            visible = [tf for tf in tracked if tf.disappeared < 3]

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
                        centrality_score=tf.centrality_score,
                        size_score=tf.size_score,
                        stability_score=tf.stability_score,
                    )
                    for tf in visible
                ],
            ))

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
        if self._fallback_detector is not None:
            try:
                self._fallback_detector.close()
            except Exception:
                pass
            self._fallback_detector = None

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
        if self._fallback_detector is not None:
            try:
                self._fallback_detector.close()
            except Exception:
                pass
            self._fallback_detector = None


# ──────────────────────────────────────────────────────────────────────────────
# Public API
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
    """
    Time-dependent crop expression with proper nested if() syntax.
    """
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
    
    last_x = int(np.clip(positions[-1][1], 0, max_x))
    last_x -= last_x % 2
    x_expr = str(last_x)
    
    for i in range(len(positions) - 2, -1, -1):
        t0, x0 = positions[i]
        t1, x1 = positions[i + 1]
        
        x0 = int(np.clip(x0, 0, max_x))
        x0 -= x0 % 2
        x1 = int(np.clip(x1, 0, max_x))
        x1 -= x1 % 2
        
        dt = t1 - t0
        if dt < 1e-6:
            continue
        
        slope = (x1 - x0) / dt
        interp = f"{x0}+({slope:.4f}*(t-{t0:.4f}))"
        clamped = f"max(0\\,min({interp}\\,{max_x}))"
        x_expr = f"if(between(t\\,{t0:.4f}\\,{t1:.4f})\\,{clamped}\\,{x_expr})"
    
    x_even = f"2*trunc(({x_expr})/2)"
    
    return f"crop={target_width}:{target_height}:{x_even}:{y}"