"""Face Fusion Studio — touch-first Qt UI for the ROG Ally X (7" 1080p @ 150%).

Wizard flow (big steps, large buttons):
  1 · Choose Photo or Video and pick the target
  2 · Pick the photo with the faces to use
  3 · Map faces (big thumbnails + Flip)
  4 · Swap

Heavy work always runs in QThreads. DirectML inference is serialised in Engine.
"""
from __future__ import annotations

import logging
import os
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QSettings, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QFont, QIcon, QImage, QKeyEvent, QPixmap
from PySide6.QtWidgets import (
    QApplication, QButtonGroup, QCheckBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout,
    QLabel, QMainWindow, QMessageBox, QProgressBar, QPushButton, QScrollArea, QSizePolicy,
    QSlider, QStackedWidget, QVBoxLayout, QWidget,
)

from .. import __version__, detect, media, vcore
from ..job import (
    ENHANCE_LABEL, ENHANCE_SPECS, MAX_CLIP_S, Cancelled, Job, Settings,
    default_out_dir, default_pictures_dir, load_photo,
)
from ..models import ENHANCER_LIGHT, ENHANCER_HQ, ModelStore, REQUIRED_BYTES
from .theme import DARK_QSS

log = logging.getLogger("ffs")

TEAL = (169, 184, 0)      # BGR of #00b8a9
ORANGE = (61, 138, 255)
PAGE_SETUP, PAGE_MAIN, PAGE_PREVIEW, PAGE_OPTIONS, PAGE_PROGRESS, PAGE_DONE = range(6)
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi", ".3gp"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def pix(bgr, w, h):
    if bgr is None or getattr(bgr, "size", 0) == 0:
        return QPixmap()
    dpr = QApplication.instance().devicePixelRatio() if QApplication.instance() else 1.0
    W, H = int(w * dpr), int(h * dpr)
    ih, iw = bgr.shape[:2]
    if ih < 1 or iw < 1:
        return QPixmap()
    s = min(W / iw, H / ih)
    img = cv2.resize(bgr, (max(1, int(iw * s)), max(1, int(ih * s))),
                     interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    p = QPixmap.fromImage(QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0],
                                 QImage.Format.Format_RGB888).copy())
    p.setDevicePixelRatio(dpr)
    return p


def draw_faces(img, faces, labels, colors=None):
    out = img.copy(); th = max(2, img.shape[1] // 400)
    for i, f in enumerate(faces):
        x0, y0, x1, y1 = [int(v) for v in vcore.bbox(f)]
        c = (colors[i] if colors else TEAL)
        cv2.rectangle(out, (x0, y0), (x1, y1), c, th)
        fs = max(0.7, img.shape[1] / 1400)
        (tw, tht), _ = cv2.getTextSize(labels[i], cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        cv2.rectangle(out, (x0, max(0, y0 - tht - 12)), (x0 + tw + 10, y0), c, -1)
        cv2.putText(out, labels[i], (x0 + 5, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, fs, (20, 16, 4), th, cv2.LINE_AA)
    return out


def face_crop(img, f, size=120):
    x0, y0, x1, y1 = vcore.bbox(f); cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; r = max(x1 - x0, y1 - y0) * 0.7
    a, b = int(max(0, cx - r)), int(max(0, cy - r)); c, d = int(min(img.shape[1], cx + r)), int(min(img.shape[0], cy + r))
    if d <= b or c <= a:
        return np.zeros((size, size, 3), np.uint8)
    return cv2.resize(img[b:d, a:c], (size, size), interpolation=cv2.INTER_AREA)


class Worker(QThread):
    progressed = Signal(object)
    done = Signal(object)
    failed = Signal(str, str)

    def __init__(self, fn, parent=None):
        super().__init__(parent); self.fn = fn; self.cancelled = False

    def run(self):
        try:
            self.done.emit(self.fn(lambda: self.cancelled, self.progressed.emit))
        except Cancelled:
            self.failed.emit("cancelled", "")
        except Exception as e:  # noqa: BLE001
            if self.cancelled or str(e) == "cancelled":
                self.failed.emit("cancelled", "")
            else:
                tb = traceback.format_exc()
                log.error("worker failed: %s\n%s", e, tb)
                try:
                    from .. import crashlog
                    crashlog.record_current(where="worker")
                except Exception:  # noqa: BLE001
                    pass
                self.failed.emit(str(e) or type(e).__name__, tb)


def button(text, kind=None, min_w=0):
    b = QPushButton(text)
    if kind: b.setObjectName(kind)
    if min_w: b.setMinimumWidth(min_w)
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    b.setMinimumHeight(56)
    return b


def label(text="", kind=None, wrap=False):
    l = QLabel(text)
    if kind: l.setObjectName(kind)
    l.setWordWrap(wrap)
    return l


def card(oid="card"):
    f = QFrame(); f.setObjectName(oid); return f


def image_label(h):
    im = QLabel(); im.setAlignment(Qt.AlignmentFlag.AlignCenter); im.setFixedHeight(h)
    im.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed); im.setMinimumWidth(120)
    return im


class Segmented(QWidget):
    changed = Signal(object)

    def __init__(self, items):
        super().__init__()
        lay = QHBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(8)
        self.group = QButtonGroup(self); self.group.setExclusive(True); self.buttons = {}
        for text, value in items:
            b = button(text); b.setObjectName("seg"); b.setCheckable(True); b.setMinimumHeight(56)
            self.group.addButton(b); lay.addWidget(b, 1); self.buttons[value] = b
            b.clicked.connect(lambda _=False, v=value: self.changed.emit(v))
        if items:
            self.buttons[items[0][1]].setChecked(True)

    def set(self, value):
        b = self.buttons.get(value)
        if b: b.setChecked(True)

    def value(self):
        for v, b in self.buttons.items():
            if b.isChecked(): return v
        return None


class ImageSlot(QFrame):
    clicked = Signal()
    dropped = Signal(str)

    def __init__(self, title, hint, w=600, h=240):
        super().__init__(); self.setObjectName("card"); self.w, self.h = w, h
        self.setAcceptDrops(True); self.setCursor(Qt.CursorShape.PointingHandCursor)
        lay = QVBoxLayout(self); lay.setContentsMargins(12, 10, 12, 10); lay.setSpacing(6)
        self.title = label(title, "section"); lay.addWidget(self.title)
        self.image = QLabel(hint); self.image.setObjectName("slot")
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setMinimumHeight(h); self.image.setWordWrap(True); lay.addWidget(self.image, 1)
        self.info = label("", "hint", True); lay.addWidget(self.info)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton: self.clicked.emit()

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls(): e.acceptProposedAction()

    def dropEvent(self, e):
        for u in e.mimeData().urls(): self.dropped.emit(u.toLocalFile())

    def set_image(self, bgr):
        self.image.setPixmap(pix(bgr, self.image.width() or self.w, self.h))

    def clear_image(self, hint="Tap to choose"):
        self.image.clear(); self.image.setText(hint)


class MainWindow(QMainWindow):
    def __init__(self, store: ModelStore, device="auto"):
        super().__init__()
        self.setWindowTitle(f"Face Fusion Studio {__version__}")
        self.cfg = QSettings("vanu", "FaceFusionStudio")
        self.store = store
        dev = str(self.cfg.value("device", device)); dev = dev if dev in ("auto", "dml", "cpu") else "auto"
        self.job = Job(store, device if device != "auto" else dev)
        self.info = None; self.photo = None; self.thumb = None; self.vfaces = []
        self.rotation = 0; self.mode = "video"; self.target_photo = None
        self.worker = None; self.result = None; self.bench_done = False
        self._busy = False
        enh = self.cfg.value("enhance", "auto")
        enh = None if enh in (None, "off", "None", "") else (enh if enh in ("auto", "gpen256", "gpen512") else "auto")
        self.opt = dict(
            max_short=int(self.cfg.value("max_short", 1080)), fps=float(self.cfg.value("fps", 30.0)),
            enhance=enh, out_dir=str(self.cfg.value("out_dir", str(default_out_dir()))),
            min_confidence=float(self.cfg.value("min_confidence", 0.62)),
            same_gender=self.cfg.value("same_gender", "true") not in (False, "false", "0", 0),
            color_match=self.cfg.value("color_match", "true") not in (False, "false", "0", 0),
            temporal_smooth=float(self.cfg.value("temporal_smooth", 0.18)),
            seamless=self.cfg.value("seamless", "false") in (True, "true", "1", 1),
        )
        root = QWidget(); rl = QVBoxLayout(root); rl.setContentsMargins(0, 0, 0, 0); rl.setSpacing(0)
        rl.addWidget(self._topbar())
        self.stack = QStackedWidget(); rl.addWidget(self.stack, 1)
        for build in (self._page_setup, self._page_main, self._page_preview, self._page_options,
                      self._page_progress, self._page_done):
            self.stack.addWidget(build())
        self.setCentralWidget(root)
        self.setAcceptDrops(True)
        self._refresh_setup(); self.set_chip()
        if self.store.all_installed():
            self.go(PAGE_MAIN); QTimer.singleShot(200, self._start_bench)
        else:
            self.go(PAGE_SETUP)

    # ------------------------------------------------------------------ chrome
    def _topbar(self):
        bar = QFrame(); bar.setObjectName("topbar"); lay = QHBoxLayout(bar); lay.setContentsMargins(16, 10, 12, 10)
        t = label("Face Fusion Studio", "apptitle"); lay.addWidget(t); lay.addSpacing(12)
        self.chip = label("…", "chip"); lay.addWidget(self.chip); lay.addStretch(1)
        self.btn_opts_top = button("Options", None, 140); self.btn_opts_top.clicked.connect(lambda: self.go(PAGE_OPTIONS)); lay.addWidget(self.btn_opts_top)
        about = button("About", None, 120); about.clicked.connect(self._about); lay.addWidget(about)
        return bar

    def set_chip(self):
        eng = self.job.engine
        if eng is None:
            self.chip.setText("Ready"); self.chip.setProperty("state", ""); return
        i = eng.info
        if i.active == "DirectML":
            self.chip.setText("GPU · DirectML"); self.chip.setProperty("state", "")
        elif i.fell_back or i.fallback_reason:
            self.chip.setText("CPU · GPU failed"); self.chip.setProperty("state", "cpu")
            self.chip.setToolTip(i.fallback_reason or "DirectML unavailable — using CPU")
        else:
            self.chip.setText("CPU"); self.chip.setProperty("state", "cpu")
        self.chip.style().unpolish(self.chip); self.chip.style().polish(self.chip)

    def _toast_gpu_fallback(self, reason=""):
        self.set_chip()
        # Non-blocking-ish: show after the current event finishes so we never raise on the ORT thread
        def _show():
            try:
                QMessageBox.information(
                    self, "Using CPU",
                    "GPU (DirectML) failed, so this PC is using CPU instead.\n\n"
                    "Everything still works — just slower.\n\n"
                    + (reason[:300] if reason else "")
                    + "\n\nOptions → Retry GPU to try again, or leave Processor on CPU.")
            except Exception:  # noqa: BLE001
                pass
        QTimer.singleShot(0, _show)

    def _ensure_engine_hooks(self):
        eng = self.job.engine
        if eng is None:
            return
        if getattr(eng, "_gui_hooked", False):
            return
        eng._gui_hooked = True
        eng.on_fallback(lambda reason: QTimer.singleShot(0, lambda: self._toast_gpu_fallback(reason)))
        if eng.info.fell_back and not getattr(self, "_fallback_toasted", False):
            self._fallback_toasted = True
            QTimer.singleShot(200, lambda: self._toast_gpu_fallback(eng.info.fallback_reason))

    def _about(self):
        QMessageBox.information(
            self, "About",
            f"Face Fusion Studio {__version__}\n\n"
            "Touch-friendly face swap for Windows / ROG Ally X.\n"
            "YOLO Face + ArcFace + inswapper + optional GPEN.\n"
            "Not affiliated with the FaceFusion desktop app.\n\n"
            f"Crash log: %LOCALAPPDATA%\\FaceFusionStudio\\crash.log")

    def go(self, page):
        self.stack.setCurrentIndex(page)
        focus = {PAGE_SETUP: "btn_dl", PAGE_MAIN: "btn_start", PAGE_PROGRESS: "btn_cancel"}.get(page)
        if focus and hasattr(self, focus): getattr(self, focus).setFocus()
        self.btn_opts_top.setVisible(page in (PAGE_MAIN, PAGE_PREVIEW, PAGE_DONE))
        if page == PAGE_OPTIONS: self._refresh_options()
        if page == PAGE_MAIN: self._update_summary()

    # ------------------------------------------------------------------ setup
    def _page_setup(self):
        w = QWidget(); outer = QVBoxLayout(w); outer.setContentsMargins(24, 18, 24, 18); outer.setSpacing(12)
        outer.addWidget(label("One-time setup", "title"))
        outer.addWidget(label(
            "Download the AI models (~465 MB required). They stay on this PC. "
            "InsightFace models are for personal / non-commercial use only.", "subtitle", True))
        self.setup_list = label("", "hint", True); outer.addWidget(self.setup_list)
        self.setup_bar = QProgressBar(); self.setup_bar.setRange(0, 1000); self.setup_bar.setTextVisible(False)
        self.setup_bar.setMinimumHeight(28); outer.addWidget(self.setup_bar)
        self.setup_text = label("", "subtitle", True); outer.addWidget(self.setup_text)
        outer.addStretch(1)
        row = QHBoxLayout()
        self.btn_dl = button("Download models", "primary", 260); self.btn_dl.clicked.connect(self._download); row.addWidget(self.btn_dl)
        self.btn_cancel_dl = button("Cancel", "danger", 140); self.btn_cancel_dl.clicked.connect(self._cancel); self.btn_cancel_dl.setEnabled(False); row.addWidget(self.btn_cancel_dl)
        row.addStretch(1)
        self.btn_setup_back = button("Back"); self.btn_setup_back.clicked.connect(lambda: self.go(PAGE_OPTIONS)); row.addWidget(self.btn_setup_back)
        outer.addLayout(row)
        return w

    def _refresh_setup(self):
        miss = self.store.missing_required()
        self.setup_list.setText(
            "All required models are installed." if not miss else
            "Still needed:\n" + "\n".join(f"  · {s.file}  ({s.bytes/1e6:.0f} MB)" for s in miss))
        self.btn_dl.setEnabled(bool(miss) and not (self.worker and self.worker.isRunning()))
        self.btn_setup_back.setVisible(self.store.all_installed())

    def _download(self):
        if self.worker and self.worker.isRunning(): return
        self.btn_dl.setEnabled(False); self.btn_cancel_dl.setEnabled(True)
        def work(cancelled, emit):
            miss = self.store.missing_required()
            def prog(f, done, total, bps, verifying):
                emit(dict(file=f, done=done, total=total, bps=bps, verifying=verifying))
            self.store.ensure(miss, progress=prog, cancelled=cancelled)
            return True
        self.worker = Worker(work, self)
        self.worker.progressed.connect(self._setup_progress)
        self.worker.done.connect(lambda _: (self._refresh_setup(), self.btn_cancel_dl.setEnabled(False),
                                            self.go(PAGE_MAIN), QTimer.singleShot(100, self._start_bench)))
        self.worker.failed.connect(self._failed)
        self.worker.start()

    def _setup_progress(self, d):
        done, total = d.get("done", 0), max(1, d.get("total", 1))
        self.setup_bar.setValue(int(1000 * done / total))
        tag = "Checking" if d.get("verifying") else "Downloading"
        self.setup_text.setText(f"{tag} {d.get('file','')} · {done/1e6:.0f}/{total/1e6:.0f} MB")

    # ------------------------------------------------------------------ main wizard
    def _page_main(self):
        w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(20, 12, 20, 12); lay.setSpacing(10)

        # Mode toggle — obvious Photo | Video
        mode_row = QHBoxLayout(); mode_row.setSpacing(10)
        mode_row.addWidget(label("1", "stepnum"))
        mode_row.addWidget(label("What are you swapping?", "steplabel"))
        mode_row.addStretch(1)
        self.btn_mode_video = button("Video", "mode", 160); self.btn_mode_video.setCheckable(True); self.btn_mode_video.setChecked(True)
        self.btn_mode_photo = button("Photo", "mode", 160); self.btn_mode_photo.setCheckable(True)
        self._mode_group = QButtonGroup(self); self._mode_group.setExclusive(True)
        self._mode_group.addButton(self.btn_mode_video); self._mode_group.addButton(self.btn_mode_photo)
        self.btn_mode_video.clicked.connect(lambda: self._set_mode("video"))
        self.btn_mode_photo.clicked.connect(lambda: self._set_mode("photo"))
        mode_row.addWidget(self.btn_mode_video); mode_row.addWidget(self.btn_mode_photo)
        lay.addLayout(mode_row)

        # Step 2 — pick target + faces (side by side, huge tap targets)
        pick = QHBoxLayout(); pick.setSpacing(12)
        left = QVBoxLayout(); left.setSpacing(6)
        r = QHBoxLayout(); r.addWidget(label("2", "stepnum")); r.addWidget(label("Pick the target", "steplabel")); r.addStretch(1); left.addLayout(r)
        self.vslot = ImageSlot("Target video / photo", "Tap here to choose\nthe video or photo to change", 600, 220)
        self.vslot.clicked.connect(self._pick_target); self.vslot.dropped.connect(self.open_path)
        left.addWidget(self.vslot, 1)
        pick.addLayout(left, 1)

        right = QVBoxLayout(); right.setSpacing(6)
        r2 = QHBoxLayout(); r2.addWidget(label("3", "stepnum")); r2.addWidget(label("Pick the faces photo", "steplabel")); r2.addStretch(1); right.addLayout(r2)
        self.pslot = ImageSlot("Faces to use", "Tap here to choose\na clear photo of the face(s)", 600, 220)
        self.pslot.clicked.connect(self._pick_photo); self.pslot.dropped.connect(self.open_path)
        right.addWidget(self.pslot, 1)
        pick.addLayout(right, 1)
        lay.addLayout(pick, 1)

        # Step 4 — map faces with BIG thumbnails
        map_card = card("stepcard"); ml = QVBoxLayout(map_card); ml.setContentsMargins(14, 10, 14, 10); ml.setSpacing(8)
        mh = QHBoxLayout(); mh.addWidget(label("4", "stepnum")); mh.addWidget(label("Map faces  (who gets which face)", "steplabel")); mh.addStretch(1)
        self.btn_flip = button("Flip pairing", None, 180); self.btn_flip.clicked.connect(self._flip); self.btn_flip.setEnabled(False); mh.addWidget(self.btn_flip)
        ml.addLayout(mh)
        self.pair_hint = label("Add a target and a faces photo to see the mapping.", "subtitle", True); ml.addWidget(self.pair_hint)
        self.pair_row = QHBoxLayout(); self.pair_row.setSpacing(12); ml.addLayout(self.pair_row)
        # trim (video only)
        trim = QHBoxLayout(); trim.setSpacing(10)
        self.l_start = label("Start 0:00"); self.l_len = label("Length 10 s")
        self.s_start = QSlider(Qt.Orientation.Horizontal); self.s_len = QSlider(Qt.Orientation.Horizontal)
        for s in (self.s_start, self.s_len): s.setMinimumHeight(40); s.setEnabled(False)
        self.s_start.valueChanged.connect(self._trim_changed); self.s_len.valueChanged.connect(self._trim_changed)
        self.s_start.sliderReleased.connect(self._refresh_target_faces)
        trim.addWidget(self.l_start); trim.addWidget(self.s_start, 2); trim.addWidget(self.l_len); trim.addWidget(self.s_len, 2)
        ml.addLayout(trim)
        lay.addWidget(map_card)

        # Bottom actions
        bottom = QHBoxLayout(); bottom.setSpacing(12)
        self.summary = label("", "subtitle", True); bottom.addWidget(self.summary, 1)
        self.btn_preview = button("Preview", None, 160); self.btn_preview.clicked.connect(self._preview); bottom.addWidget(self.btn_preview)
        self.btn_start = button("Swap", "primary", 240); self.btn_start.clicked.connect(self._start); bottom.addWidget(self.btn_start)
        lay.addLayout(bottom)
        # keep btn_pause attr so older cancel path is safe
        self.btn_pause = QPushButton(); self.btn_pause.hide()
        return w

    def _set_mode(self, mode):
        self.mode = mode
        self.btn_mode_video.setChecked(mode == "video")
        self.btn_mode_photo.setChecked(mode == "photo")
        self.btn_start.setText("Swap video" if mode == "video" else "Swap photo")
        self.vslot.title.setText("Target video" if mode == "video" else "Target photo")
        for w in (self.s_start, self.s_len, self.l_start, self.l_len):
            w.setVisible(mode == "video")
        self.btn_preview.setVisible(mode == "video")
        # Clear opposite target when switching modes to avoid confusion
        if mode == "photo":
            self.info = None
        else:
            self.target_photo = None
        self._update_summary()

    def _pick_target(self):
        if self.mode == "photo":
            p, _ = QFileDialog.getOpenFileName(self, "Choose the target photo", str(Path.home() / "Pictures"),
                                               "Images (*.jpg *.jpeg *.png *.webp *.bmp);;All files (*)")
            if p: self.set_target_photo(p)
        else:
            self._pick_video()

    def set_target_photo(self, p):
        try:
            img = detect.load_image(p)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Photo", str(e)); return
        self.target_photo = p; self.info = None; self.thumb = img; self.vfaces = []
        self.vslot.set_image(img)
        self.vslot.info.setText(f"{Path(p).name}\n{img.shape[1]}×{img.shape[0]}")
        self._refresh_target_faces()
        self._update_summary()

    def _pick_video(self):
        p, _ = QFileDialog.getOpenFileName(self, "Choose the video", str(Path.home() / "Videos"),
                                           "Videos (*.mp4 *.mov *.m4v *.mkv *.webm *.avi *.3gp);;All files (*)")
        if p: self.set_video(p)

    def _pick_photo(self):
        p, _ = QFileDialog.getOpenFileName(self, "Choose the photo with the faces", str(Path.home() / "Pictures"),
                                           "Images (*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff);;All files (*)")
        if p: self.set_photo(p)

    def open_path(self, p):
        ext = Path(p).suffix.lower()
        if ext in VIDEO_EXT:
            self._set_mode("video"); self.set_video(p)
        elif ext in IMAGE_EXT:
            if self.mode == "photo" and (self.photo is not None and self.target_photo is None):
                self.set_target_photo(p)
            elif self.mode == "photo" and self.target_photo is None:
                self.set_target_photo(p)
            else:
                self.set_photo(p)
        else:
            QMessageBox.warning(self, "Unsupported file", f"{Path(p).name} isn't a video or photo this app can open.")

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls(): e.acceptProposedAction()

    def dropEvent(self, e):
        for u in e.mimeData().urls(): self.open_path(u.toLocalFile())

    def set_video(self, p):
        try:
            info = media.probe(p)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Video", str(e)); return
        self.info = info; self.target_photo = None; self.job.analysis = None; self.vfaces = []
        dur = info.duration
        self.s_start.blockSignals(True); self.s_len.blockSignals(True)
        self.s_start.setEnabled(True); self.s_len.setEnabled(True)
        self.s_start.setRange(0, max(0, int((dur - 0.5) * 10))); self.s_start.setValue(0)
        self.s_len.setRange(5, max(5, int(min(MAX_CLIP_S, dur) * 10))); self.s_len.setValue(int(min(10.0, dur) * 10))
        self.s_start.blockSignals(False); self.s_len.blockSignals(False)
        warn = "  ·  HDR: colours may look flat" if info.hdr else ""
        self.vslot.info.setText(f"{Path(p).name}\n{info.summary()}{warn}")
        self._trim_changed(); self._refresh_target_faces()

    def _busy_worker(self):
        return bool(self.worker and self.worker.isRunning())

    def _refresh_target_faces(self):
        """Detect faces in the target (video frame or target photo) off the UI thread."""
        if self._busy_worker():
            # Still show a thumbnail if we can
            if self.mode == "video" and self.info:
                fr = media.read_frame_at(self.info.path, self.s_start.value() / 10)
                if fr is not None: self.thumb = fr; self._redraw()
            return
        st = self.settings()
        if self.mode == "video" and self.info:
            path = self.info.path; t0 = self.s_start.value() / 10; kind = "video"
            fr0 = media.read_frame_at(path, t0)
            if fr0 is None: return
            self.thumb = fr0
        elif self.mode == "photo" and self.target_photo:
            path = self.target_photo; t0 = 0; kind = "photo"
            try:
                self.thumb = detect.load_image(path)
            except Exception as e:  # noqa: BLE001
                QMessageBox.warning(self, "Photo", str(e)); return
        else:
            return

        self.pair_hint.setText("Finding faces…")
        thumb = self.thumb

        def work(cancelled, emit):
            eng = self.job.get_engine(); eng.prepare(None, detector=st.detector)
            detect.set_default_engine(eng, "yoloface_8n" if st.detector != "retina" else "retinaface_10g")
            if kind == "video":
                frame = media.read_frame_at(path, t0) or thumb
            else:
                frame = detect.load_image(path)
            faces = detect.detect_frame(frame, 2, st.detect_opts(), engine=eng)
            faces = sorted(faces, key=lambda f: f[:, 0].mean())
            return frame, faces

        self.worker = Worker(work, self)

        def done(res):
            self._ensure_engine_hooks()
            frame, faces = res
            self.thumb = frame; self.vfaces = faces; self._redraw()

        def failed(msg, tb):
            log.warning("target face refresh failed: %s", msg)
            self.vfaces = []
            self.pair_hint.setText(f"Could not find faces yet: {msg}")
            self._redraw()

        self.worker.done.connect(done); self.worker.failed.connect(failed)
        self.worker.start()

    def set_photo(self, p):
        """Load source faces photo — always off the UI thread (avoids hang→crash on DirectML)."""
        if self._busy_worker():
            QMessageBox.information(self, "Busy", "Wait for the current step to finish, then try again."); return
        self.pslot.info.setText(f"{Path(p).name}\nFinding faces…")
        self.pair_hint.setText("Finding faces in the faces photo…")
        opts = self.settings().detect_opts()
        device = self.job.device

        def work(cancelled, emit):
            return load_photo(p, opts=opts, store=self.store, device=device)

        self.worker = Worker(work, self)

        def done(ph):
            if not ph.faces:
                QMessageBox.warning(self, "Photo", "No face found in that photo. Use a clear, front-facing photo.")
                self.pslot.info.setText(f"{Path(p).name}\nNo face found"); return
            self.photo = ph; self.rotation = 0; self.job.analysis = None
            genders = ", ".join((g or "?") for g in (ph.genders or []))
            confs = ", ".join(f"{h.confidence:.0%}" for h in (ph.hits or []))
            extra = (f" · {genders}" if genders else "") + (f" · conf {confs}" if confs else "")
            self.pslot.info.setText(f"{Path(p).name}\n{len(ph.faces)} face{'s' if len(ph.faces) != 1 else ''} found{extra}")
            self._redraw()

        def failed(msg, tb):
            QMessageBox.warning(self, "Photo", msg)
            self.pslot.info.setText("Could not read that photo")

        self.worker.done.connect(done); self.worker.failed.connect(failed)
        self.worker.start()

    def _assign(self):
        n = len(self.photo.faces) if self.photo else 0
        return vcore.pair_single_frame(self.vfaces, n, self.rotation) if n else [-1] * len(self.vfaces)

    def _clear_pair_row(self):
        while self.pair_row.count():
            it = self.pair_row.takeAt(0)
            if it.widget(): it.widget().deleteLater()

    def _redraw(self):
        letters = "ABCDEF"
        if self.photo:
            n = len(self.photo.faces)
            plabels = []
            for i in range(n):
                lab = letters[i]
                if getattr(self.photo, "hits", None) and i < len(self.photo.hits):
                    h = self.photo.hits[i]
                    g = (h.gender or "?")[0].upper() if h.gender else "?"
                    lab = f"{letters[i]}  {h.confidence:.0%}  {g}"
                plabels.append(lab)
            self.pslot.set_image(draw_faces(self.photo.img, self.photo.faces, plabels, [ORANGE] * n))
        if self.thumb is not None:
            asg = self._assign()
            labels = [f"{i + 1}" + (f" ← {letters[a]}" if a >= 0 else "") for i, a in enumerate(asg)]
            self.vslot.set_image(draw_faces(self.thumb, self.vfaces, labels) if self.vfaces else self.thumb)

        self._clear_pair_row()
        if self.photo and self.vfaces:
            self.pair_hint.hide()
            for i, a in enumerate(self._assign()):
                if a < 0: continue
                # Target face (big)
                box = QVBoxLayout(); box.setSpacing(4)
                timg = QLabel(); timg.setPixmap(pix(face_crop(self.thumb, self.vfaces[i], 120), 120, 120))
                timg.setAlignment(Qt.AlignmentFlag.AlignCenter)
                box.addWidget(timg); box.addWidget(label(f"Target {i + 1}", "hint"))
                wrap = QWidget(); wrap.setLayout(box); self.pair_row.addWidget(wrap)
                arrow = label("←", "title"); arrow.setAlignment(Qt.AlignmentFlag.AlignCenter); self.pair_row.addWidget(arrow)
                # Source face (big)
                box2 = QVBoxLayout(); box2.setSpacing(4)
                simg = QLabel(); simg.setPixmap(pix(face_crop(self.photo.img, self.photo.faces[a], 120), 120, 120))
                simg.setAlignment(Qt.AlignmentFlag.AlignCenter)
                box2.addWidget(simg); box2.addWidget(label(f"Face {letters[a]}", "hint"))
                wrap2 = QWidget(); wrap2.setLayout(box2); self.pair_row.addWidget(wrap2)
                self.pair_row.addSpacing(24)
            self.pair_row.addStretch(1)
        else:
            self.pair_hint.show()
            if not (self.info or self.target_photo):
                self.pair_hint.setText("Step 2: pick a target.  Step 3: pick a faces photo.")
            elif not self.photo:
                self.pair_hint.setText("Step 3: pick a faces photo to map.")
            elif not self.vfaces:
                self.pair_hint.setText("Looking for faces in the target…")
            else:
                self.pair_hint.setText("Faces are paired left → right. Tap Flip if the mapping is wrong.")
        self.btn_flip.setEnabled(bool(self.photo and len(self.photo.faces) >= 2))
        self._update_summary()

    def _flip(self):
        if self.photo and len(self.photo.faces) >= 2:
            self.rotation = (self.rotation + 1) % len(self.photo.faces); self._redraw()

    def _trim_changed(self):
        if not self.info: return
        st, ln = self.s_start.value() / 10, self.s_len.value() / 10
        ln = min(ln, max(0.5, self.info.duration - st))
        self.l_start.setText(f"Start {media.fmt_time(st)}"); self.l_len.setText(f"Length {ln:.1f} s")
        self._update_summary()

    def settings(self) -> Settings:
        enh = self.opt["enhance"]
        if enh == "auto":
            enh = self.job.recommended_enhance() if self.bench_done else (
                "gpen256" if self.store.is_installed(ENHANCER_LIGHT) else None)
        if enh and not self.store.is_installed(ENHANCE_SPECS[enh]): enh = None
        st = self.s_start.value() / 10 if self.info else 0.0
        ln = self.s_len.value() / 10 if self.info else 10.0
        out = self.opt["out_dir"]
        if self.mode == "photo":
            out = str(default_pictures_dir()) if "FaceFusion" not in out else out
        return Settings(
            start=st, length=ln, fps=self.opt["fps"], max_short=self.opt["max_short"], enhance=enh,
            rotation=self.rotation, device=self.job.device, out_dir=out,
            min_confidence=float(self.opt.get("min_confidence", 0.62)),
            same_gender=bool(self.opt.get("same_gender", True)),
            color_match=bool(self.opt.get("color_match", True)),
            temporal_smooth=float(self.opt.get("temporal_smooth", 0.18)),
            seamless=bool(self.opt.get("seamless", False)),
        )

    def _update_summary(self):
        st = self.settings()
        parts = []
        if self.mode == "video" and self.info:
            W, H, _ = vcore.out_size(self.info.width, self.info.height, st.max_short, st.align)
            fps = st.effective_fps(self.info.fps)
            ln = min(st.length, self.info.duration - st.start)
            n = int(ln * fps); faces = min(2, len(self.photo.faces)) if self.photo else 2
            est = self.job.estimate(n, faces, st.enhance, W, H)
            parts.append(f"{W}×{H} · {fps:.3g} fps · {n} frames · {ENHANCE_LABEL[st.enhance]}")
            parts.append(f"About {media.fmt_time(est)[:-2]}" + ("" if self.bench_done else " (estimate)"))
        elif self.mode == "photo":
            parts.append(f"Photo swap · Enhance {ENHANCE_LABEL[st.enhance]}")
            if self.target_photo: parts.append(Path(self.target_photo).name)
        else:
            parts.append(f"Up to {st.max_short}p · {st.fps:.0f} fps · {ENHANCE_LABEL[st.enhance]}")
        self.summary.setText("\n".join(parts))
        # CRITICAL: photo mode must enable Swap when target_photo + photo are set
        if self.mode == "photo":
            ready = bool(self.target_photo and self.photo and self.photo.faces)
        else:
            ready = bool(self.info and self.photo and self.photo.faces)
        self.btn_start.setEnabled(ready and not self._busy_worker())
        self.btn_preview.setEnabled(bool(self.mode == "video" and self.info and self.photo) and not self._busy_worker())

    # ------------------------------------------------------------------ bench
    def _start_bench(self):
        if self.bench_done or self._busy_worker(): return
        self.chip.setText("Measuring speed…")
        def work(cancelled, emit):
            eng = self.job.get_engine(); 
            return self.job.benchmark()
        self.bw = Worker(work, self)
        def _bench_ok(b):
            self._ensure_engine_hooks(); self._bench_done(b)
        self.bw.done.connect(_bench_ok)
        self.bw.failed.connect(lambda m, t: (self.set_chip(), self.chip.setToolTip(m)))
        self.bw.start()

    def _bench_done(self, b):
        self.bench_done = True; self.set_chip(); self._update_summary()
        if self.stack.currentIndex() == PAGE_OPTIONS: self._refresh_options()

    # ------------------------------------------------------------------ preview
    def _page_preview(self):
        w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(16, 12, 16, 12); lay.setSpacing(10)
        self.prev_title = label("Preview", "title"); lay.addWidget(self.prev_title)
        row = QHBoxLayout(); row.setSpacing(12)
        self.prev_imgs = []
        for t in ("Before", "After"):
            c = card(); cl = QVBoxLayout(c); cl.setContentsMargins(10, 8, 10, 10); cl.addWidget(label(t, "section"))
            im = image_label(420); cl.addWidget(im, 1); row.addWidget(c, 1); self.prev_imgs.append(im)
        lay.addLayout(row, 1)
        self.prev_info = label("", "subtitle", True); lay.addWidget(self.prev_info)
        b = QHBoxLayout()
        back = button("Back", None, 150); back.clicked.connect(lambda: self.go(PAGE_MAIN)); b.addWidget(back)
        fl = button("Flip", None, 150); fl.clicked.connect(lambda: (self._flip(), self._preview())); b.addWidget(fl)
        b.addStretch(1)
        go = button("Swap video", "primary", 230); go.clicked.connect(self._start); b.addWidget(go)
        lay.addLayout(b)
        return w

    def _preview(self):
        if not (self.info and self.photo) or self._busy_worker(): return
        st = self.settings()
        self.btn_preview.setText("Working…"); self.btn_preview.setEnabled(False)
        def work(cancelled, emit):
            t0 = time.perf_counter(); r = self.job.preview(self.info, self.photo, st, self.s_start.value() / 10)
            return r, time.perf_counter() - t0, st
        self.worker = Worker(work, self)
        self.worker.done.connect(self._preview_done); self.worker.failed.connect(self._failed)
        self.worker.start()

    def _preview_done(self, r):
        (before, after, dets, asg), secs, st = r
        letters = "ABCDEF"
        labels = [f"{i + 1}" + (f" ← {letters[a]}" if a >= 0 else "") for i, a in enumerate(asg)]
        self.prev_imgs[0].setPixmap(pix(draw_faces(before, dets, labels) if dets else before, 600, 420))
        self.prev_imgs[1].setPixmap(pix(after, 600, 420))
        self.prev_title.setText(f"Preview at {media.fmt_time(st.start)}")
        self.prev_info.setText(
            f"{after.shape[1]}×{after.shape[0]} · {ENHANCE_LABEL[st.enhance]} · "
            f"{len([a for a in asg if a >= 0])} face(s) · {secs:.1f} s · {self.job.engine.info.label()}")
        self.btn_preview.setText("Preview"); self.btn_preview.setEnabled(True)
        self.set_chip(); self.go(PAGE_PREVIEW)

    # ------------------------------------------------------------------ options
    def _page_options(self):
        w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(24, 14, 24, 14); lay.setSpacing(12)
        lay.addWidget(label("Options", "title"))
        sc = QScrollArea(); sc.setWidgetResizable(True); sc.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget(); il = QVBoxLayout(inner); il.setSpacing(14)

        def add_seg(title, items, key, cast=lambda x: x):
            il.addWidget(label(title, "section"))
            seg = Segmented(items); seg.set(self.opt.get(key))
            seg.changed.connect(lambda v, k=key, c=cast: self._set_opt(k, c(v)))
            il.addWidget(seg); return seg

        self.seg_res = add_seg("Max short side", [("480p", 480), ("720p", 720), ("1080p", 1080)], "max_short", int)
        self.seg_fps = add_seg("FPS", [("15", 15), ("24", 24), ("30", 30), ("Source", 0)], "fps", float)
        self.seg_enh = add_seg("Enhance", [("Off", None), ("Auto", "auto"), ("Light", "gpen256"), ("HQ", "gpen512")], "enhance")
        il.addWidget(label("Processor", "section"))
        self.seg_dev = Segmented([("Auto", "auto"), ("GPU (DirectML)", "dml"), ("CPU", "cpu")])
        self.seg_dev.set(self.job.device); self.seg_dev.changed.connect(self._set_device); il.addWidget(self.seg_dev)
        self.dev_info = label("", "hint", True); il.addWidget(self.dev_info)
        self.enh_info = label("", "hint", True); il.addWidget(self.enh_info)

        self.chk_gender = QCheckBox("Prefer same gender when mapping"); self.chk_gender.setChecked(self.opt["same_gender"])
        self.chk_gender.stateChanged.connect(lambda _=0: self._set_opt("same_gender", self.chk_gender.isChecked())); il.addWidget(self.chk_gender)
        self.chk_color = QCheckBox("Match skin colour"); self.chk_color.setChecked(self.opt["color_match"])
        self.chk_color.stateChanged.connect(lambda _=0: self._set_opt("color_match", self.chk_color.isChecked())); il.addWidget(self.chk_color)
        self.chk_seam = QCheckBox("Seamless blend (slower)"); self.chk_seam.setChecked(self.opt["seamless"])
        self.chk_seam.stateChanged.connect(lambda _=0: self._set_opt("seamless", self.chk_seam.isChecked())); il.addWidget(self.chk_seam)

        r = QHBoxLayout(); self.out_label = label(self.opt["out_dir"], "hint", True); r.addWidget(self.out_label, 1)
        ch = button("Change folder…"); ch.clicked.connect(self._change_out); r.addWidget(ch); il.addLayout(r)
        mm = button("Manage models…"); mm.clicked.connect(lambda: (self._refresh_setup(), self.go(PAGE_SETUP))); il.addWidget(mm)
        il.addStretch(1); sc.setWidget(inner); lay.addWidget(sc, 1)
        done = button("Done", "primary", 200); done.clicked.connect(lambda: self.go(PAGE_MAIN)); lay.addWidget(done, 0, Qt.AlignmentFlag.AlignLeft)
        return w

    def _set_opt(self, k, v):
        self.opt[k] = v; self.cfg.setValue(k, v if v is not None else "off"); self._update_summary()

    def _set_device(self, v):
        self.job.device = v; self.cfg.setValue("device", v)
        if self.job.engine: self.job.engine.close(); self.job.engine = None
        self.bench_done = False; self.set_chip(); QTimer.singleShot(100, self._start_bench)

    def _change_out(self):
        d = QFileDialog.getExistingDirectory(self, "Output folder", self.opt["out_dir"])
        if d: self._set_opt("out_dir", d); self.out_label.setText(d)

    def _retry_gpu(self):
        from .. import dml_probe
        dml_probe.clear_status()
        os.environ.pop("FFS_FORCE_DML_FAIL", None)
        os.environ["FFS_DML_REPROBE"] = "1"
        self._fallback_toasted = False
        if self.job.engine:
            self.job.engine.close(); self.job.engine = None
        self.job.device = "auto"; self.cfg.setValue("device", "auto")
        self.seg_dev.set("auto")
        self.bench_done = False
        self.chip.setText("Retrying GPU…")
        QTimer.singleShot(100, self._start_bench)

    def _refresh_options(self):
        lines = []
        n = 300
        W, H = (1920, 1080) if self.opt["max_short"] >= 1080 else ((1280, 720) if self.opt["max_short"] >= 720 else (854, 480))
        for m in (None, "gpen256", "gpen512"):
            if m and not self.store.is_installed(ENHANCE_SPECS[m]): continue
            lines.append(f"{ENHANCE_LABEL[m]} ≈ {media.fmt_time(self.job.estimate(n, 2, m, W, H))[:-2]}")
        rec = self.job.recommended_enhance()
        self.enh_info.setText(("10 s clip estimate: " + " · ".join(lines)) +
                              (f".  Auto picks {ENHANCE_LABEL[rec]}." if self.bench_done else "."))
        eng = self.job.engine
        self.dev_info.setText(("Now using: " + eng.info.label()) if eng else "")
        self.out_label.setText(self.opt["out_dir"])

    # ------------------------------------------------------------------ progress / run
    def _page_progress(self):
        w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(24, 14, 24, 14); lay.setSpacing(10)
        self.prog_title = label("Working…", "title"); lay.addWidget(self.prog_title)
        self.steps = label("", "subtitle"); lay.addWidget(self.steps)
        self.prog_bar = QProgressBar(); self.prog_bar.setRange(0, 1000); self.prog_bar.setTextVisible(False)
        self.prog_bar.setMinimumHeight(28); lay.addWidget(self.prog_bar)
        self.prog_text = label("", "section"); lay.addWidget(self.prog_text)
        row = QHBoxLayout(); row.setSpacing(12)
        self.prog_imgs = []
        for t in ("Original", "Result"):
            c = card(); cl = QVBoxLayout(c); cl.setContentsMargins(10, 8, 10, 10); cl.addWidget(label(t, "hint"))
            im = image_label(330); cl.addWidget(im, 1); row.addWidget(c, 1); self.prog_imgs.append(im)
        lay.addLayout(row, 1)
        b = QHBoxLayout(); self.prog_dev = label("", "hint"); b.addWidget(self.prog_dev, 1)
        self.btn_cancel = button("Cancel", "danger", 200); self.btn_cancel.clicked.connect(self._cancel); b.addWidget(self.btn_cancel)
        lay.addLayout(b)
        return w

    def _start(self):
        if self._busy_worker(): return
        if self.mode == "photo":
            if not (self.target_photo and self.photo): return
        elif not (self.info and self.photo):
            return
        st = self.settings()
        self.prog_bar.setValue(0); self.prog_text.setText("Starting…"); self.btn_cancel.setEnabled(True)
        for im in self.prog_imgs: im.clear()
        self._steps("detect")
        self.go(PAGE_PROGRESS); self.run_t0 = time.time()
        mode = self.mode; target = self.target_photo; info = self.info; photo = self.photo

        def work(cancelled, emit):
            if mode == "photo":
                return self.job.run_photo(target, photo, st, progress=emit, cancel=cancelled)
            return self.job.run(info, photo, st, progress=emit, cancel=cancelled)

        self.worker = Worker(work, self)
        self.worker.progressed.connect(self._progress)
        self.worker.done.connect(self._finished)
        self.worker.failed.connect(self._failed)
        self.worker.start()

    def _steps(self, cur):
        if self.mode == "photo":
            names = [("detect", "1  Find faces"), ("swap", "2  Swap"), ("mux", "3  Save photo")]
        else:
            names = [("detect", "1  Find & track faces"), ("swap", "2  Swap faces"), ("mux", "3  Save MP4 + sound")]
        order = [n for n, _ in names]; ci = order.index(cur) if cur in order else len(order)
        self.steps.setText("     ".join(("✓ " if i < ci else ("● " if i == ci else "○ ")) + t for i, (n, t) in enumerate(names)))

    def _progress(self, d):
        stage = d.get("stage"); done, total = d.get("done", 0), max(1, d.get("total", 1))
        if stage in ("detect", "swap", "mux"): self._steps(stage)
        if stage == "detect":
            self.prog_title.setText("Finding faces…")
            self.prog_bar.setValue(int(150 * done / total))
            self.prog_text.setText(d.get("detail") or f"Finding faces · {done} of {total}")
        elif stage == "swap":
            self.prog_title.setText("Swapping faces…")
            self.prog_bar.setValue(150 + int(830 * done / total))
            eta = d.get("eta"); rate = d.get("rate") or 0
            self.prog_text.setText(d.get("detail") or (
                f"Frame {done} of {total} · {rate:.1f} frames/s" + (f" · about {media.fmt_time(eta)[:-2]} left" if eta else "")))
        elif stage == "mux":
            self.prog_title.setText("Final — saving…")
            frac = done / total if total else 0.0
            self.prog_bar.setValue(980 + int(20 * min(1.0, frac)))
            self.prog_text.setText(d.get("detail") or "Saving…")
        if "thumb" in d:
            self.prog_imgs[0].setPixmap(pix(d.get("before"), 600, 330))
            self.prog_imgs[1].setPixmap(pix(d["thumb"], 600, 330))
        if self.job.engine:
            self.prog_dev.setText(f"{self.job.engine.info.label()} · elapsed {media.fmt_time(time.time() - self.run_t0)[:-2]}")
            self.set_chip()

    def _cancel(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancelled = True; self.prog_text.setText("Cancelling…"); self.btn_cancel.setEnabled(False)
            if hasattr(self, "btn_pause"): self.btn_pause.setEnabled(False)

    def _failed(self, msg, tb):
        self.btn_preview.setText("Preview"); self.btn_preview.setEnabled(True)
        self.btn_cancel.setEnabled(True)
        if msg == "cancelled":
            self.go(PAGE_MAIN); self._update_summary(); return
        box = QMessageBox(self); box.setIcon(QMessageBox.Icon.Warning); box.setWindowTitle("Something went wrong")
        hint = ""
        try:
            from .. import crashlog
            hint = f"\n\nA log was saved to:\n{crashlog.crash_path()}"
        except Exception:  # noqa: BLE001
            pass
        box.setText((msg or "Unknown error") + hint)
        box.setDetailedText(tb or "")
        box.exec()
        self.go(PAGE_MAIN if self.store.all_installed() else PAGE_SETUP)
        self._update_summary()

    # ------------------------------------------------------------------ done
    def _page_done(self):
        w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(24, 14, 24, 14); lay.setSpacing(10)
        self.done_title = label("Done!", "title"); lay.addWidget(self.done_title)
        self.done_path = label("", "section", True)
        self.done_path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse); lay.addWidget(self.done_path)
        self.done_img = image_label(390); lay.addWidget(self.done_img, 1)
        self.done_info = label("", "subtitle", True); lay.addWidget(self.done_info)
        b = QHBoxLayout(); b.setSpacing(12)
        play = button("Open result", "primary", 200)
        play.clicked.connect(lambda: self.result and QDesktopServices.openUrl(QUrl.fromLocalFile(self.result["path"])))
        b.addWidget(play)
        fold = button("Open folder", None, 180)
        fold.clicked.connect(lambda: self.result and self._open_folder(self.result["path"], select=True)); b.addWidget(fold)
        redo = button("Flip & redo", None, 180); redo.clicked.connect(lambda: (self._flip(), self._start())); b.addWidget(redo)
        b.addStretch(1)
        new = button("New", None, 140); new.clicked.connect(lambda: self.go(PAGE_MAIN)); b.addWidget(new)
        lay.addLayout(b)
        return w

    def _finished(self, res):
        self.result = res
        self.done_path.setText(res["path"])
        is_photo = res.get("mode") == "photo"
        self.done_title.setText("Done! Your photo is saved." if is_photo else "Done! Your video is saved.")
        if is_photo:
            fr = res.get("after")
            if fr is None:
                try: fr = detect.load_image(res["path"])
                except Exception:  # noqa: BLE001
                    fr = None
            if fr is not None and res.get("before") is not None and res["before"].shape == fr.shape:
                fr = np.hstack([res["before"], np.full((fr.shape[0], 12, 3), 18, np.uint8), fr])
        else:
            fr = media.read_frame_at(res["path"], max(0.01, res.get("duration", 1) / 2))
            if fr is not None and self.info:
                s0 = media.read_frame_at(self.info.path, self.s_start.value() / 10 + res.get("duration", 1) / 2)
                if s0 is not None:
                    W, H, s = vcore.out_size(self.info.width, self.info.height, self.settings().max_short, 2)
                    src = vcore.prep(s0, W, H, s)
                    if src is not None and src.shape == fr.shape:
                        fr = np.hstack([src, np.full((src.shape[0], 12, 3), 18, np.uint8), fr])
        if fr is not None: self.done_img.setPixmap(pix(fr, 1200, 390))
        mb = res.get("size", 0) / 1e6
        if is_photo:
            self.done_info.setText(
                f"{res.get('W', '?')}×{res.get('H', '?')} · {mb:.1f} MB · Enhance {res.get('enhance')}\n"
                f"Took {media.fmt_time(res.get('total_s', 0))[:-2]} on {res.get('device', '?')}")
        else:
            self.done_info.setText(
                f"{res.get('W','?')}×{res.get('H','?')} · {res.get('fps', 0):.3g} fps · "
                f"{media.fmt_time(res.get('duration', 0))} · {mb:.1f} MB · Enhance {res.get('enhance')} · {res.get('audio', '')}\n"
                f"Took {media.fmt_time(res.get('total_s', 0))[:-2]} on {res.get('device', '?')} "
                f"(encoder {res.get('encoder', '?')})")
        self.go(PAGE_DONE)

    def _open_folder(self, p, select=False):
        p = str(p)
        if os.name == "nt" and select:
            import subprocess
            subprocess.Popen(["explorer", "/select,", os.path.normpath(p)]); return
        Path(p if not select else Path(p).parent).mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(p if not select else str(Path(p).parent)))

    def keyPressEvent(self, e: QKeyEvent):
        page = self.stack.currentIndex(); k = e.key()
        if k == Qt.Key.Key_Escape:
            if page == PAGE_PROGRESS: self._cancel()
            elif page in (PAGE_PREVIEW, PAGE_OPTIONS, PAGE_DONE): self.go(PAGE_MAIN)
            return
        if k in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and page in (PAGE_MAIN, PAGE_PREVIEW):
            self._start(); return
        if e.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if k == Qt.Key.Key_O: self._pick_video(); return
            if k == Qt.Key.Key_P: self._pick_photo(); return
        if k == Qt.Key.Key_F and page in (PAGE_MAIN, PAGE_PREVIEW): self._flip(); return
        super().keyPressEvent(e)

    def closeEvent(self, e):
        if self.worker and self.worker.isRunning():
            if QMessageBox.question(self, "Quit?", "A job is running. Cancel it and quit?") != QMessageBox.StandardButton.Yes:
                e.ignore(); return
            self.worker.cancelled = True; self.worker.wait(15000)
        e.accept()


EXTRA_QSS = """
QFrame#topbar { background:#0a0c10; border-bottom:1px solid #2a3140; }
QLabel#apptitle { font-size:20px; font-weight:700; color:#fff; background:transparent; }
QLabel#chip { background:#243b39; color:#7fe3d9; border-radius:14px; padding:6px 14px; font-size:14px; font-weight:600; }
QLabel#chip[state="cpu"] { background:#3b3324; color:#ffcf8a; }
QLabel#slot { background:#12151c; border:2px dashed #3c4454; border-radius:14px; color:#9aa0a6; font-size:16px; }
QPushButton#seg { background:#1f232c; border:1px solid #3c4454; border-radius:12px; font-size:16px; }
QPushButton#seg:checked { background:#00b8a9; color:#041014; border:none; font-weight:700; }
QPushButton:focus, QCheckBox:focus, QSlider:focus { outline:none; border:2px solid #ff8a3d; }
QMessageBox QLabel { background:transparent; }
"""


def make_app(argv=None):
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    # Prefer fonts that exist on every Windows 11 / Ally X image (avoids tofu □□□)
    os.environ.setdefault("QT_QPA_FONTDIR", "")
    app = QApplication.instance() or QApplication(argv or sys.argv)
    app.setApplicationName("Face Fusion Studio"); app.setOrganizationName("vanu")
    app.setStyle("Fusion")
    font = QFont("Segoe UI", 11)
    font.setStyleHint(QFont.StyleHint.SansSerif)
    app.setFont(font)
    app.setStyleSheet(DARK_QSS + EXTRA_QSS)
    ico = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[2] / "packaging")) / "icon.ico"
    if ico.is_file(): app.setWindowIcon(QIcon(str(ico)))
    return app


def run_gui(args=None) -> int:
    from .. import crashlog
    crashlog.install()
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("vanu.FaceFusionStudio")
        except Exception:  # noqa: BLE001
            pass
    if args is not None and getattr(args, "screenshots", None):
        from .screens import take_screenshots
        return take_screenshots(args)
    app = make_app()
    try:
        from PySide6.QtCore import QtMsgType, qInstallMessageHandler
        def _qt_msg(mode, context, message):
            try:
                if mode in (QtMsgType.QtFatalMsg, QtMsgType.QtCriticalMsg):
                    crashlog.record(RuntimeError, RuntimeError(f"Qt {mode}: {message}"), None, where="qt")
            except Exception:  # noqa: BLE001
                pass
        qInstallMessageHandler(_qt_msg)
    except Exception:  # noqa: BLE001
        pass
    store = ModelStore(Path(args.models) if args is not None and args.models else None)
    win = MainWindow(store, getattr(args, "device", "auto") if args is not None else "auto")
    from .gamepad import Gamepad
    win.gamepad = Gamepad(win)
    win.resize(1280, 720)
    if QApplication.primaryScreen() and QApplication.primaryScreen().availableGeometry().width() <= 1400:
        win.showMaximized()
    else:
        win.show()
    for p in (getattr(args, "video", None), getattr(args, "photo", None)):
        if p: win.open_path(p)
    return app.exec()
