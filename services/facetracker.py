from pathlib import Path

import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class FaceTracker:

    def __init__(self, detection_confidence: float = 0.5):
        model_path = (
            Path(__file__).resolve().parent.parent
            / "models"
            / "large_far.tflite"
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe model not found: {model_path}"
            )

        base_options = python.BaseOptions(
            model_asset_path=str(model_path)
        )

        options = vision.FaceDetectorOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=detection_confidence,
        )

        self.detector = vision.FaceDetector.create_from_options(options)

    def detect_largest_face(self, frame):
        frame_height, frame_width = frame.shape[:2]

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        try:
            result = self.detector.detect(mp_image)
        except Exception as exc:
            print(f"[FaceTracker] detection failed on frame: {exc}")
            return None

        if not result.detections:
            return None

        largest_face = None
        largest_area = 0

        for detection in result.detections:
            box = detection.bounding_box

            x = max(0, box.origin_x)
            y = max(0, box.origin_y)
            width = min(box.width, frame_width - x)
            height = min(box.height, frame_height - y)

            if width <= 0 or height <= 0:
                continue

            area = width * height

            if area > largest_area:
                largest_area = area
                largest_face = (
                    x + width / 2,
                    y + height / 2,
                    width,
                    height,
                )

        return largest_face

    def close(self):
        self.detector.close()


def calculate_crop_x(
    frame_width: int,
    frame_height: int,
    face_center_x: float,
):
    crop_width = int(frame_height * 9 / 16)
    crop_width -= crop_width % 2  

    if crop_width >= frame_width:
        return 0

    crop_x = int(face_center_x - crop_width / 2)
    crop_x = max(0, min(crop_x, frame_width - crop_width))

    return crop_x


def get_face_crop_positions(
    video_path: str,
    start: float,
    end: float,
    tracker: FaceTracker,
    sample_interval: float = 0.05,
):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    positions = []

    try:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        current_time = start

        while current_time < end:
            cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)

            success, frame = cap.read()

            if not success:
                break

            face = tracker.detect_largest_face(frame)

            if face is not None:
                face_center_x = face[0]

                crop_x = calculate_crop_x(
                    frame_width=frame_width,
                    frame_height=frame_height,
                    face_center_x=face_center_x,
                )

                positions.append((round(current_time - start, 3), crop_x))

            current_time += sample_interval

    finally:
        cap.release()

    return positions


def smooth_positions(positions, smoothing_factor=0.25):
    if not positions:
        return positions

    smoothed = []
    previous_x = positions[0][1]

    for timestamp, x in positions:
        current_x = previous_x + smoothing_factor * (x - previous_x)
        smoothed.append((timestamp, current_x))
        previous_x = current_x

    return smoothed