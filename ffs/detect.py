"""Face detection via FaceFusion-compatible YOLO Face (default) or RetinaFace ONNX.

Outputs FaceHit with 5-point landmarks (eyes, nose, mouth corners) used by ArcFace / inswapper,
plus bbox and confidence. No MediaPipe.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

DEFAULT_MIN_CONF = 0.50
DEFAULT_MIN_FACE_FRAC = 0.025
DEFAULT_DETECTOR_SIZE = 640  # FaceFusion default for yolo_face


@dataclass
class DetectOpts:
    min_confidence: float = DEFAULT_MIN_CONF
    min_face_frac: float = DEFAULT_MIN_FACE_FRAC
    max_faces: int = 8
    detector: str = "yolo"  # yolo | retina
    detector_size: int = DEFAULT_DETECTOR_SIZE


@dataclass
class FaceHit:
    """One accepted face with 5-point landmarks."""
    kps: np.ndarray              # (5, 2) float32 — left_eye, right_eye, nose, left_mouth, right_mouth
    confidence: float
    gender: Optional[str] = None
    age: Optional[int] = None
    bbox: tuple = (0.0, 0.0, 0.0, 0.0)
    pts: np.ndarray = field(default=None, repr=False)  # alias for tracking (Nx2); defaults to kps

    def __post_init__(self):
        self.kps = np.asarray(self.kps, np.float32).reshape(5, 2)
        if self.pts is None:
            self.pts = self.kps.copy()
        if self.bbox == (0.0, 0.0, 0.0, 0.0):
            x0, y0 = self.kps.min(0); x1, y1 = self.kps.max(0)
            # expand bbox from kps
            w, h = x1 - x0, y1 - y0
            self.bbox = (float(x0 - 0.3 * w), float(y0 - 0.45 * h), float(x1 + 0.3 * w), float(y1 + 0.25 * h))

    @property
    def pts468(self) -> np.ndarray:
        """Compatibility shim: return kps (callers that expected MediaPipe pts use kps5())."""
        return self.kps

    @property
    def pts5(self) -> np.ndarray:
        return self.kps


def _nms(boxes, scores, thr=0.4):
    if len(boxes) == 0:
        return []
    boxes = np.asarray(boxes, np.float32)
    scores = np.asarray(scores, np.float32)
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0]); keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= thr]
    return keep


def _letterbox(img, size):
    """Resize keeping aspect, pad to size×size (FaceFusion prepare_detect_frame style)."""
    h, w = img.shape[:2]
    s = min(size / h, size / w)
    nh, nw = int(round(h * s)), int(round(w * s))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), np.uint8)
    canvas[:nh, :nw] = resized
    return canvas, s, nh, nw


class FaceDetector:
    """ONNX face detector (YOLO Face or RetinaFace) run through the shared Engine lock."""

    def __init__(self, engine, model_name="yoloface_8n", size=640):
        self.engine = engine
        self.model_name = model_name
        self.size = size
        self._session_ready = False

    def _ensure(self):
        if not self._session_ready:
            self.engine.session(self.model_name)
            self._session_ready = True

    def detect(self, img_bgr, opts: DetectOpts | None = None) -> list[FaceHit]:
        opts = opts or DetectOpts()
        self._ensure()
        h0, w0 = img_bgr.shape[:2]
        size = opts.detector_size or self.size
        if self.model_name.startswith("yolo"):
            return self._detect_yolo(img_bgr, opts, size)
        return self._detect_retina(img_bgr, opts, size)

    def _detect_yolo(self, img_bgr, opts, size):
        # FaceFusion detect_with_yolo_face
        h0, w0 = img_bgr.shape[:2]
        # restrict then pad
        s = min(1.0, size / max(h0, w0))
        if s < 1:
            temp = cv2.resize(img_bgr, (int(round(w0 * s)), int(round(h0 * s))), interpolation=cv2.INTER_AREA)
        else:
            temp = img_bgr
            s = 1.0
        th, tw = temp.shape[:2]
        ratio_h = h0 / th
        ratio_w = w0 / tw
        canvas = np.zeros((size, size, 3), np.float32)
        canvas[:th, :tw] = temp.astype(np.float32)
        inp = (canvas.transpose(2, 0, 1)[None] / 255.0).astype(np.float32)
        out = self.engine.run(self.model_name, {"input": inp})[0]
        det = np.squeeze(out).T  # (N, 4+1+15) = box xywh, score, 5*(x,y,vis)
        if det.ndim != 2 or det.shape[1] < 5:
            return []
        boxes_raw = det[:, :4]
        scores = det[:, 4]
        lms_raw = det[:, 5:5 + 15] if det.shape[1] >= 20 else None
        keep = np.where(scores > opts.min_confidence)[0]
        if keep.size == 0:
            return []
        boxes, sc, lms = [], [], []
        for i in keep:
            cx, cy, bw, bh = boxes_raw[i]
            x0 = (cx - bw / 2) * ratio_w
            y0 = (cy - bh / 2) * ratio_h
            x1 = (cx + bw / 2) * ratio_w
            y1 = (cy + bh / 2) * ratio_h
            boxes.append([x0, y0, x1, y1])
            sc.append(float(scores[i]))
            if lms_raw is not None:
                lm = lms_raw[i].reshape(-1, 3)[:, :2].copy()
                lm[:, 0] *= ratio_w
                lm[:, 1] *= ratio_h
                lms.append(lm)
            else:
                # synthesize 5 pts from bbox
                lms.append(np.array([
                    [x0 + 0.3 * (x1 - x0), y0 + 0.35 * (y1 - y0)],
                    [x0 + 0.7 * (x1 - x0), y0 + 0.35 * (y1 - y0)],
                    [x0 + 0.5 * (x1 - x0), y0 + 0.55 * (y1 - y0)],
                    [x0 + 0.35 * (x1 - x0), y0 + 0.75 * (y1 - y0)],
                    [x0 + 0.65 * (x1 - x0), y0 + 0.75 * (y1 - y0)],
                ], np.float32))
        idx = _nms(boxes, sc, 0.4)
        short = min(h0, w0)
        hits = []
        for i in idx[:opts.max_faces]:
            b = boxes[i]
            fw, fh = b[2] - b[0], b[3] - b[1]
            if min(fw, fh) / short < opts.min_face_frac:
                continue
            hits.append(FaceHit(kps=lms[i], confidence=sc[i], bbox=tuple(b)))
        # left-to-right
        hits.sort(key=lambda h: (h.bbox[0] + h.bbox[2]) / 2)
        return hits

    def _detect_retina(self, img_bgr, opts, size):
        # Simplified RetinaFace 10G path (FaceFusion feature strides 8/16/32)
        from . import models as M
        h0, w0 = img_bgr.shape[:2]
        s = min(1.0, size / max(h0, w0))
        if s < 1:
            temp = cv2.resize(img_bgr, (int(round(w0 * s)), int(round(h0 * s))), interpolation=cv2.INTER_AREA)
        else:
            temp = img_bgr
        th, tw = temp.shape[:2]
        ratio_h = h0 / max(th, 1)
        ratio_w = w0 / max(tw, 1)
        canvas = np.zeros((size, size, 3), np.float32)
        canvas[:th, :tw] = temp.astype(np.float32)
        inp = ((canvas.transpose(2, 0, 1)[None] - 127.5) / 128.0).astype(np.float32)
        detection = self.engine.run(self.model_name, {"input": inp})
        # RetinaFace outputs 9 tensors: scores×3, boxes×3, lms×3
        feature_strides = [8, 16, 32]
        boxes, sc, lms = [], [], []
        for index, stride in enumerate(feature_strides):
            scores_raw = detection[index]
            keep = np.where(scores_raw.reshape(-1) >= opts.min_confidence)[0]
            if keep.size == 0:
                continue
            sh, sw = size // stride, size // stride
            # anchors
            ay, ax = np.mgrid[0:sh, 0:sw].astype(np.float32)
            anchors = np.stack([ax.ravel() * stride, ay.ravel() * stride], 1)
            # duplicate for 2 anchors
            anchors = np.repeat(anchors, 2, axis=0)
            bbox_raw = detection[index + 3].reshape(-1, 4) * stride
            lm_raw = detection[index + 6].reshape(-1, 10) * stride
            scores_flat = scores_raw.reshape(-1)
            for ki in keep:
                if ki >= len(anchors):
                    continue
                a = anchors[ki]
                # distance to box
                d = bbox_raw[ki]
                x0 = (a[0] - d[0]) * ratio_w
                y0 = (a[1] - d[1]) * ratio_h
                x1 = (a[0] + d[2]) * ratio_w
                y1 = (a[1] + d[3]) * ratio_h
                boxes.append([x0, y0, x1, y1])
                sc.append(float(scores_flat[ki]))
                lm = lm_raw[ki].reshape(5, 2).copy()
                lm[:, 0] = (a[0] + lm[:, 0]) * ratio_w
                lm[:, 1] = (a[1] + lm[:, 1]) * ratio_h
                # FaceFusion distance_to_face_landmark_5: landmark = anchor + distance
                # Actually FF does: distance_to_face_landmark_5(anchors, raw) then * ratio
                # Re-read: face_landmarks_5_raw = detection[...] * feature_stride
                # then distance_to_face_landmark_5(anchors, face_landmarks_5_raw)
                lms.append(lm)
        if not boxes:
            return []
        idx = _nms(boxes, sc, 0.4)
        short = min(h0, w0)
        hits = []
        for i in idx[:opts.max_faces]:
            b = boxes[i]
            if min(b[2] - b[0], b[3] - b[1]) / short < opts.min_face_frac:
                continue
            hits.append(FaceHit(kps=lms[i], confidence=sc[i], bbox=tuple(b)))
        hits.sort(key=lambda h: (h.bbox[0] + h.bbox[2]) / 2)
        return hits


# Process-global default detector (shared across worker threads; Engine.run is lock-serialised)
_default_detector: FaceDetector | None = None
_default_lock = threading.Lock()


def set_default_engine(engine, model_name="yoloface_8n"):
    global _default_detector
    with _default_lock:
        _default_detector = FaceDetector(engine, model_name)


def detect_faces(img_bgr, opts: DetectOpts | None = None, engine=None) -> list[FaceHit]:
    """Detect faces. Uses process-global detector or a one-shot engine= override."""
    opts = opts or DetectOpts()
    if engine is not None:
        name = "yoloface_8n" if opts.detector != "retina" else "retinaface_10g"
        return FaceDetector(engine, name, opts.detector_size).detect(img_bgr, opts)
    det = _default_detector
    if det is None:
        raise RuntimeError("Face detector not initialised — call set_default_engine or pass engine=")
    return det.detect(img_bgr, opts)


def faces_as_pts(hits: list[FaceHit]) -> list[np.ndarray]:
    """Return list of landmark arrays for tracking / pairing (5×2)."""
    return [h.kps.copy() for h in hits]


def load_image(path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def detect_photo(img, opts: DetectOpts | None = None, engine=None) -> list[FaceHit]:
    return detect_faces(img, opts, engine=engine)


def detect_frame(frame, expected=2, opts: DetectOpts | None = None, engine=None) -> list[np.ndarray]:
    """Return list of landmark arrays (5×2) for tracking."""
    hits = detect_faces(frame, opts, engine=engine)
    # Prefer up to `expected` largest faces if many
    if len(hits) > expected:
        hits = sorted(hits, key=lambda h: (h.bbox[2] - h.bbox[0]) * (h.bbox[3] - h.bbox[1]), reverse=True)[:expected]
        hits.sort(key=lambda h: (h.bbox[0] + h.bbox[2]) / 2)
    return [h.kps.copy() for h in hits]


def detect_frame_hits(frame, expected=2, opts: DetectOpts | None = None, engine=None) -> list[FaceHit]:
    hits = detect_faces(frame, opts, engine=engine)
    if len(hits) > expected:
        hits = sorted(hits, key=lambda h: (h.bbox[2] - h.bbox[0]) * (h.bbox[3] - h.bbox[1]), reverse=True)[:expected]
        hits.sort(key=lambda h: (h.bbox[0] + h.bbox[2]) / 2)
    return hits


def reject_stats(img, opts: DetectOpts) -> dict:
    """Compare acceptance under given opts vs a very loose baseline."""
    loose = DetectOpts(min_confidence=0.15, min_face_frac=0.005, max_faces=opts.max_faces)
    all_hits = detect_faces(img, loose)
    accepted = detect_faces(img, opts)
    return dict(raw=len(all_hits), accepted=len(accepted),
                confidences=[round(h.confidence, 3) for h in all_hits])
