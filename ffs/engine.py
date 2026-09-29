"""ONNX Runtime engine: DirectML (Radeon 780M) with automatic CPU fallback."""
from __future__ import annotations

import logging
import mmap
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import models as M

log = logging.getLogger("ffs")

FILES = {
    "yoloface_8n": M.YOLOFACE,
    "retinaface_10g": M.RETINAFACE,
    "arcface_w600k_r50": M.ARCFACE,
    "inswapper_128_fp16": M.SWAPPER,
    "gpen_bfr_256": M.ENHANCER_LIGHT,
    "gpen_bfr_512": M.ENHANCER_HQ,
    "gfpgan_1.4": M.GFPGAN,
}
WARMUP = {
    "yoloface_8n": {"input": (1, 3, 640, 640)},
    "retinaface_10g": {"input": (1, 3, 640, 640)},
    "arcface_w600k_r50": {"input": (1, 3, 112, 112)},
    "inswapper_128_fp16": {"target": (1, 3, 128, 128), "source": (1, 512)},
    "gpen_bfr_256": {"input": (1, 3, 256, 256)},
    "gpen_bfr_512": {"input": (1, 3, 512, 512)},
    "gfpgan_1.4": {"input": (1, 3, 512, 512)},
}


def _varint(b, i):
    r = 0; s = 0
    while True:
        c = b[i]; i += 1; r |= (c & 0x7F) << s; s += 7
        if c < 0x80: return r, i


def _fields(b, i, end):
    while i < end:
        key, i = _varint(b, i); fn, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(b, i); yield fn, wt, v, None
        elif wt == 1:
            yield fn, wt, i, i + 8; i += 8
        elif wt == 2:
            ln, i = _varint(b, i); yield fn, wt, i, i + ln; i += ln
        elif wt == 5:
            yield fn, wt, i, i + 4; i += 4
        else:
            raise ValueError(f"unsupported wire type {wt}")


def read_last_initializer(path) -> np.ndarray:
    with open(path, "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as b:
        graph = None
        for fn, wt, a, e in _fields(b, 0, len(b)):
            if fn == 7 and wt == 2: graph = (a, e)
        if graph is None: raise ValueError("no graph in model")
        last = None
        for fn, wt, a, e in _fields(b, graph[0], graph[1]):
            if fn == 5 and wt == 2: last = (a, e)
        if last is None: raise ValueError("no initializers")
        dims, dtype, raw, floats, i32 = [], 1, None, None, None
        for fn, wt, a, e in _fields(b, last[0], last[1]):
            if fn == 1:
                if wt == 0: dims.append(a)
                else:
                    j = a
                    while j < e: v, j = _varint(b, j); dims.append(v)
            elif fn == 2: dtype = a
            elif fn == 9: raw = bytes(b[a:e])
            elif fn == 4 and wt == 2: floats = np.frombuffer(bytes(b[a:e]), "<f4")
            elif fn == 5 and wt == 2:
                vals = []; j = a
                while j < e: v, j = _varint(b, j); vals.append(v)
                i32 = np.array(vals, np.uint32)
    if dtype == 1:
        arr = np.frombuffer(raw, "<f4") if raw is not None else floats
    elif dtype == 10:
        arr = np.frombuffer(raw, "<f2") if raw is not None else i32.astype(np.uint16).view(np.float16)
    else:
        raise ValueError(f"unexpected initializer dtype {dtype}")
    return np.array(arr, dtype=np.float32).reshape(dims)


@dataclass
class DeviceInfo:
    requested: str = "auto"
    active: str = "CPU"
    adapter: str = ""
    fallback_reason: str = ""
    per_model: dict = field(default_factory=dict)

    def label(self):
        if self.active == "DirectML":
            return f"GPU · DirectML{(' · ' + self.adapter) if self.adapter else ''}"
        return "CPU" + (f" (GPU unavailable: {self.fallback_reason})" if self.fallback_reason else "")


def gpu_adapter_name() -> str:
    if os.name != "nt":
        return ""
    try:
        import subprocess
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name) -join '; '"],
            capture_output=True, text=True, timeout=15, creationflags=0x08000000).stdout.strip()
        return out
    except Exception:  # noqa: BLE001
        return ""


class Engine:
    def __init__(self, store: M.ModelStore, device: str = "auto", threads: int = 0):
        import onnxruntime as ort
        self.ort = ort
        self.store = store
        self.device = device
        self.threads = threads or int(os.environ.get("ORT_THREADS", "0") or 0)
        self.lock = threading.Lock()
        self.sessions: dict = {}
        self._emap = None
        self.info = DeviceInfo(requested=device)
        avail = ort.get_available_providers()
        self.dml_available = "DmlExecutionProvider" in avail
        if device in ("auto", "dml") and not self.dml_available:
            self.info.fallback_reason = "DirectML not in this onnxruntime build"
        self.info.active = "DirectML" if (device in ("auto", "dml") and self.dml_available) else "CPU"
        if self.info.active == "DirectML":
            self.info.adapter = gpu_adapter_name()

    def _options(self, dml: bool):
        so = self.ort.SessionOptions()
        so.log_severity_level = 3
        if dml:
            so.enable_mem_pattern = False
            so.execution_mode = self.ort.ExecutionMode.ORT_SEQUENTIAL
        else:
            so.enable_cpu_mem_arena = False
            if self.threads:
                so.intra_op_num_threads = self.threads; so.inter_op_num_threads = 1
        return so

    def _open(self, name):
        spec = FILES[name]
        # GFPGAN: accept size-matched file even if placeholder SHA differs
        if name == "gfpgan_1.4":
            p = self.store.path(spec)
            if not p.is_file() or p.stat().st_size != spec.bytes:
                raise FileNotFoundError(f"model not downloaded: {spec.file}")
        elif not self.store.is_installed(spec):
            raise FileNotFoundError(f"model not downloaded: {spec.file}")
        path = str(self.store.path(spec))
        if self.info.active == "DirectML":
            try:
                s = self.ort.InferenceSession(path, self._options(True),
                                              providers=[("DmlExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"])
                used = s.get_providers()
                if not used or used[0] != "DmlExecutionProvider":
                    raise RuntimeError(f"DirectML provider not used (got {used})")
                self._warm(s, name)
                self.info.per_model[name] = "DirectML"
                return s
            except Exception as e:  # noqa: BLE001
                if self.device == "dml":
                    raise
                log.warning("DirectML failed for %s: %s -> CPU", name, e)
                self.info.fallback_reason = f"{type(e).__name__}: {str(e)[:160]}"
                self.info.active = "CPU"
        s = self.ort.InferenceSession(path, self._options(False), providers=["CPUExecutionProvider"])
        self.info.per_model[name] = "CPU"
        return s

    def _warm(self, s, name):
        shape = WARMUP.get(name)
        if not shape:
            return
        feeds = {k: np.zeros(v, np.float32) for k, v in shape.items()}
        if name == "inswapper_128_fp16":
            feeds["source"][:] = 0.04
        try:
            s.run(None, feeds)
        except Exception:  # noqa: BLE001 — warmup best-effort (input name may differ)
            pass

    def session(self, name):
        s = self.sessions.get(name)
        if s is None:
            with self.lock:
                s = self.sessions.get(name)
                if s is None:
                    s = self._open(name)
                    self.sessions[name] = s
        return s

    def prepare(self, enhance_mode=None, detector="yolo"):
        det = "yoloface_8n" if detector != "retina" else "retinaface_10g"
        names = [det, "arcface_w600k_r50", "inswapper_128_fp16"]
        if enhance_mode:
            names.append({"gpen256": "gpen_bfr_256", "gpen512": "gpen_bfr_512", "gfpgan": "gfpgan_1.4"}[enhance_mode])
        for n in names:
            self.session(n)
        return self.info

    def run(self, name, feeds):
        s = self.session(name)
        with self.lock:
            return s.run(None, feeds)

    def emap(self):
        if self._emap is None:
            self._emap = read_last_initializer(self.store.path(M.SWAPPER))
        return self._emap

    def close(self):
        self.sessions.clear()

    def benchmark(self, modes=("off", "gpen256", "gpen512"), reps=3):
        res = {}
        order = [("swap", "inswapper_128_fp16"), ("gpen256", "gpen_bfr_256"), ("gpen512", "gpen_bfr_512")]
        for key, name in order:
            if key != "swap" and key not in modes:
                continue
            if not self.store.is_installed(FILES[name]):
                continue
            s = self.session(name)
            feeds = {k: np.random.default_rng(0).random(v, dtype=np.float32) for k, v in WARMUP[name].items()}
            ts = []
            for _ in range(reps + 1):
                t = time.perf_counter()
                with self.lock:
                    s.run(None, feeds)
                ts.append(time.perf_counter() - t)
            res[key] = float(np.median(ts[1:]))
        return res
