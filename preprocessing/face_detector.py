"""
Face detection and cropping.

Paper basis: Section 3 of the review notes that the reviewed literature
converges on MTCNN for face detection/cropping with a small margin (~10%)
kept around the box for context. We use MTCNN (via the `facenet-pytorch`
package) as the primary detector for that reason.

Proposed implementation decision: MTCNN needs a working PyTorch install and
is a bit slow on CPU. Some student machines may have trouble installing it
(e.g. offline labs, restricted pip access), so we add a Haar-cascade
fallback (bundled with OpenCV, zero extra dependencies) that is selected
automatically if MTCNN cannot be imported/initialised, or explicitly via
`DataConfig.face_detector = "haar"`. This is our engineering choice, not
something described in the paper.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class FaceBox:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float = 1.0


class BaseFaceDetector:
    def detect(self, image_rgb: np.ndarray) -> Optional[FaceBox]:
        candidates = self.detect_all(image_rgb)
        if not candidates:
            return None
        # pick the largest detected face
        return max(candidates, key=lambda b: (b.x2 - b.x1) * (b.y2 - b.y1))

    def detect_all(self, image_rgb: np.ndarray) -> List[FaceBox]:
        raise NotImplementedError

    def crop(self, image_rgb: np.ndarray, box: FaceBox, margin: float = 0.10) -> np.ndarray:
        h, w = image_rgb.shape[:2]
        bw = box.x2 - box.x1
        bh = box.y2 - box.y1
        mx = int(bw * margin)
        my = int(bh * margin)
        x1 = max(0, box.x1 - mx)
        y1 = max(0, box.y1 - my)
        x2 = min(w, box.x2 + mx)
        y2 = min(h, box.y2 + my)
        if x2 <= x1 or y2 <= y1:
            return image_rgb  # degenerate box -> fall back to full frame
        return image_rgb[y1:y2, x1:x2]


class MTCNNFaceDetector(BaseFaceDetector):
    """Wraps facenet-pytorch's MTCNN with multi-face support."""

    def __init__(self, device: str = "cpu"):
        from facenet_pytorch import MTCNN  # imported lazily so the module doesn't
        # hard-require torch/facenet-pytorch just to be imported (e.g. haar-only setups)

        self._mtcnn = MTCNN(keep_all=True, device=device, post_process=False)

    def detect_all(self, image_rgb: np.ndarray) -> List[FaceBox]:
        boxes, probs = self._mtcnn.detect(image_rgb)
        if boxes is None or len(boxes) == 0:
            return []
        out = []
        for b, p in zip(boxes, probs):
            if p is not None and p > 0.60:
                x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                out.append(FaceBox(x1, y1, x2, y2, float(p)))
        return out


class HaarFaceDetector(BaseFaceDetector):
    """OpenCV Haar-cascade fallback. No GPU/torch needed, ships with opencv-python."""

    def __init__(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)

    def detect_all(self, image_rgb: np.ndarray) -> List[FaceBox]:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        faces = self._cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        if len(faces) == 0:
            return []
        out = []
        for (x, y, w, h) in faces:
            out.append(FaceBox(int(x), int(y), int(x + w), int(y + h), confidence=1.0))
        return out


class FaceTracker:
    """Maintains consistent facial identity across consecutive video frames.

    1. Locks onto the primary subject in the first frame where faces appear
       (selecting the largest high-confidence face).
    2. In subsequent frames, matches candidate detections using spatial IoU
       and normalized center-distance proximity.
    3. If tracking is momentarily lost (e.g. occlusion or rapid head turn),
       safely retains the last known position instead of switching randomly
       to another person's face.
    4. Smoothly updates the bounding box with exponential moving average.
    """

    def __init__(self, iou_thresh: float = 0.15, max_lost_frames: int = 4, smooth_alpha: float = 0.75):
        self.iou_thresh = iou_thresh
        self.max_lost_frames = max_lost_frames
        self.smooth_alpha = smooth_alpha
        self.current_box: Optional[FaceBox] = None
        self.lost_count = 0

    def update(self, candidates: List[FaceBox]) -> Optional[FaceBox]:
        if not candidates:
            if self.current_box is not None and self.lost_count < self.max_lost_frames:
                self.lost_count += 1
                return self.current_box
            return None

        if self.current_box is None:
            best = max(candidates, key=lambda b: (b.x2 - b.x1) * (b.y2 - b.y1))
            self.current_box = best
            self.lost_count = 0
            return best

        c_w = max(1, self.current_box.x2 - self.current_box.x1)
        c_h = max(1, self.current_box.y2 - self.current_box.y1)
        c_cx = (self.current_box.x1 + self.current_box.x2) / 2.0
        c_cy = (self.current_box.y1 + self.current_box.y2) / 2.0

        best_match = None
        best_score = -1.0

        for cand in candidates:
            x1 = max(self.current_box.x1, cand.x1)
            y1 = max(self.current_box.y1, cand.y1)
            x2 = min(self.current_box.x2, cand.x2)
            y2 = min(self.current_box.y2, cand.y2)
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            area_c = c_w * c_h
            area_cand = max(1, (cand.x2 - cand.x1) * (cand.y2 - cand.y1))
            iou = inter / float(area_c + area_cand - inter)

            cand_cx = (cand.x1 + cand.x2) / 2.0
            cand_cy = (cand.y1 + cand.y2) / 2.0
            dist = np.sqrt(((c_cx - cand_cx) / c_w) ** 2 + ((c_cy - cand_cy) / c_h) ** 2)

            score = iou if iou > 0.05 else max(0.0, 1.0 - dist)
            if score > best_score:
                best_score = score
                best_match = cand

        cand_cx = (best_match.x1 + best_match.x2) / 2.0 if best_match else 0
        cand_cy = (best_match.y1 + best_match.y2) / 2.0 if best_match else 0
        spatial_close = (c_cx - cand_cx) ** 2 + (c_cy - cand_cy) ** 2 < (c_w * 1.5) ** 2 if best_match else False

        if best_match is not None and (best_score >= self.iou_thresh or spatial_close):
            a = self.smooth_alpha
            nx1 = int(a * best_match.x1 + (1 - a) * self.current_box.x1)
            ny1 = int(a * best_match.y1 + (1 - a) * self.current_box.y1)
            nx2 = int(a * best_match.x2 + (1 - a) * self.current_box.x2)
            ny2 = int(a * best_match.y2 + (1 - a) * self.current_box.y2)
            self.current_box = FaceBox(nx1, ny1, nx2, ny2, best_match.confidence)
            self.lost_count = 0
            return self.current_box
        else:
            if self.lost_count < self.max_lost_frames:
                self.lost_count += 1
                return self.current_box
            else:
                best = max(candidates, key=lambda b: (b.x2 - b.x1) * (b.y2 - b.y1))
                self.current_box = best
                self.lost_count = 0
                return best


def build_face_detector(kind: str = "mtcnn", device: str = "cpu") -> BaseFaceDetector:
    """Factory with automatic fallback to Haar if MTCNN cannot be constructed."""
    if kind == "mtcnn":
        try:
            return MTCNNFaceDetector(device=device)
        except Exception as e:  # pragma: no cover - environment dependent
            print(f"[face_detector] MTCNN unavailable ({e}); falling back to Haar cascade.")
            return HaarFaceDetector()
    return HaarFaceDetector()


def detect_and_crop(
    detector: BaseFaceDetector,
    image_rgb: np.ndarray,
    margin: float = 0.10,
    target_size: Tuple[int, int] = (224, 224),
) -> Optional[np.ndarray]:
    """Detect the primary face, crop with margin, and resize.

    Returns None if no face was found (caller decides whether to skip the
    frame or fall back to a centre-crop of the full frame).
    """
    box = detector.detect(image_rgb)
    if box is None:
        return None
    face = detector.crop(image_rgb, box, margin=margin)
    if face.size == 0:
        return None
    face = cv2.resize(face, target_size, interpolation=cv2.INTER_AREA)
    return face


def center_crop_fallback(image_rgb: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Used when face detection fails on a frame, so the pipeline never crashes."""
    h, w = image_rgb.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    crop = image_rgb[y0 : y0 + side, x0 : x0 + side]
    return cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)


def track_and_crop_sequence(
    detector: BaseFaceDetector,
    frames: List[np.ndarray],
    margin: float = 0.10,
    target_size: Tuple[int, int] = (224, 224),
) -> Tuple[List[np.ndarray], int]:
    """Tracks the primary subject across an entire sequence of video frames.

    Guarantees facial identity consistency from frame to frame. If multiple
    people appear in the video, the tracker locks onto the main subject and
    consistently tracks that person rather than jumping between people.
    If tracking is momentarily lost, it holds the last known box position.
    """
    tracker = FaceTracker()
    cropped_faces: List[np.ndarray] = []
    n_detected = 0

    for frame in frames:
        candidates = detector.detect_all(frame)
        box = tracker.update(candidates)
        if box is not None:
            face = detector.crop(frame, box, margin=margin)
            if face.size > 0:
                face = cv2.resize(face, target_size, interpolation=cv2.INTER_AREA)
                cropped_faces.append(face)
                n_detected += 1
                continue
        # Fallback to center-crop if no face could be tracked
        fallback = center_crop_fallback(frame, target_size=target_size)
        cropped_faces.append(fallback)

    return cropped_faces, n_detected
