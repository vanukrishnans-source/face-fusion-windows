"""InsightFace genderage ONNX — male/female estimate for same-gender pairing."""
from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from .core import ARCFACE_112, kps5, warp

GENDER_FILE = "gender_age.onnx"
GENDER_SHA256 = "4fde69b1c810857b88c64a335084f1c3fe8f01246c9a191b48c7bb756d6652fb"
GENDER_BYTES = 1_322_532


def bundled_path() -> Optional[Path]:
    here = Path(__file__).resolve().parent
    cands = [here / "resources" / GENDER_FILE]
    if getattr(sys, "_MEIPASS", None):
        cands.insert(0, Path(sys._MEIPASS) / "ffs" / "resources" / GENDER_FILE)
    for c in cands:
        if c.is_file() and c.stat().st_size == GENDER_BYTES:
            return c
    return None


class GenderEstimator:
    def __init__(self, path: Optional[Path] = None):
        import onnxruntime as ort
        p = path or bundled_path()
        if p is None:
            raise FileNotFoundError("gender_age.onnx not found (bundle or model store)")
        so = ort.SessionOptions(); so.log_severity_level = 3
        self.session = ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"])
        self.iname = self.session.get_inputs()[0].name
        self.lock = threading.Lock()

    def estimate(self, img_bgr, pts) -> tuple[str, int, float]:
        crop, _ = warp(img_bgr, kps5(pts), ARCFACE_112, 96)
        x = ((crop[:, :, ::-1].astype(np.float32) - 127.5) / 127.5).transpose(2, 0, 1)[None]
        with self.lock:
            out = self.session.run(None, {self.iname: np.ascontiguousarray(x)})[0][0]
        female, male = float(out[0]), float(out[1])
        m = max(female, male)
        ef, em = np.exp(female - m), np.exp(male - m)
        conf = float(em / (ef + em)) if male >= female else float(ef / (ef + em))
        gender = "male" if male >= female else "female"
        age = int(np.clip(round(float(out[2]) * 100), 0, 100))
        return gender, age, conf


_tls = threading.local()


def get_estimator(path: Optional[Path] = None) -> Optional[GenderEstimator]:
    try:
        est = getattr(_tls, "est", None)
        if est is None:
            est = GenderEstimator(path); _tls.est = est
        return est
    except Exception:  # noqa: BLE001
        return None


def annotate_hits(img, hits, path: Optional[Path] = None) -> list:
    est = get_estimator(path)
    if est is None:
        return hits
    for h in hits:
        try:
            pts = h.kps if hasattr(h, "kps") else h.pts
            g, a, c = est.estimate(img, pts)
            h.gender, h.age = g, a
            h._gender_conf = c  # noqa: SLF001
        except Exception:  # noqa: BLE001
            pass
    return hits
