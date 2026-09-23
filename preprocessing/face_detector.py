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
from typing import Optional, Tuple

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
    """Wraps facenet-pytorch's MTCNN. Picks the highest-confidence face."""

    def __init__(self, device: str = "cpu"):
        from facenet_pytorch import MTCNN  # imported lazily so the module doesn't
        # hard-require torch/facenet-pytorch just to be imported (e.g. haar-only setups)

        self._mtcnn = MTCNN(keep_all=True, device=device, post_process=False)

    def detect(self, image_rgb: np.ndarray) -> Optional[FaceBox]:
        boxes, probs = self._mtcnn.detect(image_rgb)
        if boxes is None or len(boxes) == 0:
            return None
        best_idx = int(np.argmax(probs))
        x1, y1, x2, y2 = boxes[best_idx]
        return FaceBox(int(x1), int(y1), int(x2), int(y2), float(probs[best_idx]))


class HaarFaceDetector(BaseFaceDetector):
    """OpenCV Haar-cascade fallback. No GPU/torch needed, ships with opencv-python."""

    def __init__(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)

    def detect(self, image_rgb: np.ndarray) -> Optional[FaceBox]:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        faces = self._cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        if len(faces) == 0:
            return None
        # pick the largest detected face (most likely the main subject)
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        return FaceBox(int(x), int(y), int(x + w), int(y + h), confidence=1.0)


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
