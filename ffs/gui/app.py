"""Face Fusion Studio — Qt (PySide6) GUI for Windows, touch-first for the ROG Ally X (7" 1080p, 150 % scaling).

Pages: Setup (model download) · Main (video + faces photo, detected faces, pairing + Flip, trim) ·
Preview (before / after) · Options · Progress · Done.  All heavy work runs in QThreads.
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QSettings, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QIcon, QImage, QKeyEvent, QPixmap
from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout,
                               QLabel, QMainWindow, QMessageBox, QProgressBar, QPushButton, QScrollArea, QSizePolicy,
                               QSlider, QStackedWidget, QVBoxLayout, QWidget)

from .. import __version__, detect, media, vcore
from ..job import (ENHANCE_LABEL, ENHANCE_SPECS, MAX_CLIP_S, Cancelled, Job, Settings, default_out_dir, default_pictures_dir, load_photo)
from ..models import ARCFACE, ENHANCER_HQ, ENHANCER_LIGHT, SWAPPER, YOLOFACE, RETINAFACE, ModelStore, REQUIRED_BYTES
from .theme import DARK_QSS

TEAL = (169, 184, 0)      # BGR of #00b8a9
ORANGE = (61, 138, 255)
PAGE_SETUP, PAGE_MAIN, PAGE_PREVIEW, PAGE_OPTIONS, PAGE_PROGRESS, PAGE_DONE = range(6)
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi", ".3gp"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def pix(bgr, w, h):
    """BGR ndarray -> QPixmap fitted into w x h (logical px; rendered at device pixel ratio)."""
    dpr = QApplication.instance().devicePixelRatio() if QApplication.instance() else 1.0
    W, H = int(w * dpr), int(h * dpr)
    ih, iw = bgr.shape[:2]
    s = min(W / iw, H / ih)
    img = cv2.resize(bgr, (max(1, int(iw * s)), max(1, int(ih * s))), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    p = QPixmap.fromImage(QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888).copy())
    p.setDevicePixelRatio(dpr)
    return p


def draw_faces(img, faces, labels, colors=None):
    out = img.copy(); th = max(2, img.shape[1] // 400)
    for i, f in enumerate(faces):
        x0, y0, x1, y1 = [int(v) for v in vcore.bbox(f)]
        c = (colors[i] if colors else TEAL)
        cv2.rectangle(out, (x0, y0), (x1, y1), c, th)
        fs = max(0.6, img.shape[1] / 1400)
        (tw, tht), _ = cv2.getTextSize(labels[i], cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        cv2.rectangle(out, (x0, max(0, y0 - tht - 12)), (x0 + tw + 10, y0), c, -1)
        cv2.putText(out, labels[i], (x0 + 5, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, fs, (20, 16, 4), th, cv2.LINE_AA)
    return out


def face_crop(img, f, size=96):
    x0, y0, x1, y1 = vcore.bbox(f); cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; r = max(x1 - x0, y1 - y0) * 0.7
    a, b = int(max(0, cx - r)), int(max(0, cy - r)); c, d = int(min(img.shape[1], cx + r)), int(min(img.shape[0], cy + r))
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
                self.failed.emit(str(e), traceback.format_exc())


def button(text, kind=None, min_w=0):
    b = QPushButton(text)
    if kind: b.setObjectName(kind)
    if min_w: b.setMinimumWidth(min_w)
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    return b


def label(text="", kind=None, wrap=False):
    l = QLabel(text)
    if kind: l.setObjectName(kind)
    l.setWordWrap(wrap)
    return l


def card():
    f = QFrame(); f.setObjectName("card"); return f


def image_label(h):
    im = QLabel(); im.setAlignment(Qt.AlignmentFlag.AlignCenter); im.setFixedHeight(h)
    im.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed); im.setMinimumWidth(120)
    return im


class Segmented(QWidget):
    """Row of large checkable buttons (touch-friendly radio group)."""
    changed = Signal(object)

    def __init__(self, items):
        super().__init__()
        lay = QHBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(8)
        self.group = QButtonGroup(self); self.group.setExclusive(True); self.buttons = {}
        for text, value in items:
            b = button(text); b.setObjectName("seg"); b.setCheckable(True); b.setMinimumHeight(56)
            self.group.addButton(b); lay.addWidget(b, 1); self.buttons[value] = b
            b.clicked.connect(lambda _=False, v=value: self.changed.emit(v))

    def set(self, value):
        if value in self.buttons: self.buttons[value].setChecked(True)

    def value(self):
        for v, b in self.buttons.items():
            if b.isChecked(): return v


class ImageSlot(QFrame):
    """Drop target + tap-to-open card with a thumbnail."""
    clicked = Signal()
    dropped = Signal(str)

    def __init__(self, title, empty_text, w=560, h=300):
        super().__init__(); self.setObjectName("card"); self.setAcceptDrops(True); self.w, self.h = w, h
        lay = QVBoxLayout(self); lay.setContentsMargins(14, 12, 14, 12); lay.setSpacing(6)
        top = QHBoxLayout(); self.title = label(title, "section"); top.addWidget(self.title); top.addStretch(1)
        self.open_btn = button("Open…"); self.open_btn.clicked.connect(self.clicked.emit); top.addWidget(self.open_btn)
        lay.addLayout(top)
        self.image = QLabel(empty_text); self.image.setObjectName("slot"); self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setFixedHeight(h); self.image.setMinimumWidth(200); self.image.setWordWrap(True)
        self.image.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        lay.addWidget(self.image, 1)
        self.info = label("", "hint", True); lay.addWidget(self.info)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton: self.clicked.emit()

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls(): e.acceptProposedAction()

    def dropEvent(self, e):
        urls = e.mimeData().urls()
        if urls: self.dropped.emit(urls[0].toLocalFile())

    def set_image(self, bgr):
        self.image.setPixmap(pix(bgr, self.image.width() or self.w, self.h))


class MainWindow(QMainWindow):
    def __init__(self, store: ModelStore, device="auto"):
        super().__init__()
        self.setWindowTitle(f"Face Fusion Studio {__version__}")
        self.cfg = QSettings("vanu", "FaceFusionStudio")
        self.store = store
        dev = str(self.cfg.value("device", device)); dev = dev if dev in ("auto", "dml", "cpu") else "auto"
        self.job = Job(store, device if device != "auto" else dev)
        self.info = None; self.photo = None; self.thumb = None; self.vfaces = []
        self.rotation = 0; self.mode = "video"; self.target_photo = None; self.worker = None; self.result = None; self.bench_done = False
        enh = self.cfg.value("enhance", "auto")
        enh = None if enh in (None, "off", "None", "") else (enh if enh in ("auto", "gpen256", "gpen512") else "auto")
        self.opt = dict(max_short=int(self.cfg.value("max_short", 1080)), fps=float(self.cfg.value("fps", 30.0)),
                        enhance=enh, out_dir=str(self.cfg.value("out_dir", str(default_out_dir()))),
                        min_confidence=float(self.cfg.value("min_confidence", 0.62)),
                        same_gender=self.cfg.value("same_gender", "true") not in (False, "false", "0", 0),
                        color_match=self.cfg.value("color_match", "true") not in (False, "false", "0", 0),
                        temporal_smooth=float(self.cfg.value("temporal_smooth", 0.18)),
                        seamless=self.cfg.value("seamless", "false") in (True, "true", "1", 1))
        root = QWidget(); rl = QVBoxLayout(root); rl.setContentsMargins(0, 0, 0, 0); rl.setSpacing(0)
        rl.addWidget(self._topbar())
        self.stack = QStackedWidget(); rl.addWidget(self.stack, 1)
        for build in (self._page_setup, self._page_main, self._page_preview, self._page_options, self._page_progress, self._page_done):
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
        bar = QFrame(); bar.setObjectName("topbar"); lay = QHBoxLayout(bar); lay.setContentsMargins(16, 8, 12, 8)
        t = label("Face Fusion Studio", "apptitle"); lay.addWidget(t); lay.addSpacing(12)
        self.chip = label("…", "chip"); lay.addWidget(self.chip); lay.addStretch(1)
        self.btn_opts_top = button("⚙  Options"); self.btn_opts_top.clicked.connect(lambda: self.go(PAGE_OPTIONS)); lay.addWidget(self.btn_opts_top)
        about = button("About"); about.clicked.connect(self._about); lay.addWidget(about)
        return bar

    def set_chip(self):
        eng = self.job.engine
        if eng is None:
            self.chip.setText("Processor: starts with the first job"); return
        i = eng.info
        if i.active == "DirectML":
            self.chip.setText("GPU · DirectML" + (f" · {i.adapter.split(';')[0]}" if i.adapter else "")); self.chip.setProperty("state", "gpu")
        else:
            self.chip.setText("CPU" + (" (GPU unavailable)" if i.fallback_reason and i.requested != "cpu" else "")); self.chip.setProperty("state", "cpu")
            self.chip.setToolTip(i.fallback_reason)
        self.chip.style().unpolish(self.chip); self.chip.style().polish(self.chip)

    def go(self, page):
        self.stack.setCurrentIndex(page)
        focus = {PAGE_SETUP: "btn_dl", PAGE_MAIN: "btn_start", PAGE_PROGRESS: "btn_cancel"}.get(page)
        if focus and getattr(self, focus, None) is not None and getattr(self, focus).isEnabled():
            getattr(self, focus).setFocus()
        self.btn_opts_top.setVisible(page in (PAGE_MAIN, PAGE_PREVIEW, PAGE_DONE))
        if page == PAGE_OPTIONS: self._refresh_options()
        if page == PAGE_MAIN: self._update_summary()

    def _about(self):
        QMessageBox.about(self, "About Face Fusion Studio",
            f"<b>Face Fusion Studio {__version__}</b> for Windows (x64)<br>Desktop port of the Android app by vanu krishnan.<br><br>"
            "AI models: InsightFace ArcFace w600k_r50 + inswapper_128 (fp16), genderage, GPEN-BFR-256/512 — downloaded from the "
            "FaceFusion model releases. <b>InsightFace models are licensed for personal, non-commercial research use only.</b> "
            "Don't use this app to impersonate or deceive anyone; only swap faces of people who agreed to it.<br><br>"
            "Uses ONNX Runtime + DirectML (MIT), OpenCV (Apache-2.0), Qt 6 / PySide6 (LGPLv3), "
            "FFmpeg (LGPL build). See THIRD_PARTY.md next to the app.")

    # ------------------------------------------------------------------ setup
    def _page_setup(self):
        w = QWidget(); outer = QVBoxLayout(w); outer.setContentsMargins(32, 20, 32, 20); outer.setSpacing(12)
        outer.addWidget(label("One-time setup: download FaceFusion-compatible AI models", "title"))
        outer.addWidget(label("They are downloaded from the FaceFusion model releases on GitHub (Hugging Face mirror as fallback), "
                              "checked with SHA-256 and kept in your user folder. Downloads can be paused and resumed.", "subtitle", True))
        c = card(); g = QGridLayout(c); g.setContentsMargins(16, 12, 16, 12); g.setHorizontalSpacing(16); g.setVerticalSpacing(6)
        self.setup_rows = {}
        rows = [(YOLOFACE, "required · FaceFusion detector"), (ARCFACE, "required"), (SWAPPER, "required"), (ENHANCER_LIGHT, "Light enhancer (recommended)"), (ENHANCER_HQ, "HQ enhancer (best on GPU)")]
        for r, (spec, note) in enumerate(rows):
            cb = QCheckBox(spec.label); cb.setChecked(True); cb.setEnabled(note != "required")
            cb.stateChanged.connect(self._refresh_setup)
            g.addWidget(cb, r, 0); g.addWidget(label(note, "hint"), r, 1)
            g.addWidget(label(f"{spec.bytes / 1e6:.1f} MB"), r, 2); st = label("", "hint"); g.addWidget(st, r, 3)
            self.setup_rows[spec.file] = (spec, cb, st)
        g.setColumnStretch(1, 1)
        outer.addWidget(c)
        self.setup_total = label("", "section"); outer.addWidget(self.setup_total)
        self.setup_bar = QProgressBar(); self.setup_bar.setRange(0, 1000); self.setup_bar.setTextVisible(False); outer.addWidget(self.setup_bar)
        self.setup_status = label("", "subtitle", True); outer.addWidget(self.setup_status)
        row = QHBoxLayout()
        self.btn_dl = button("Download", "primary", 220); self.btn_dl.clicked.connect(self._download); row.addWidget(self.btn_dl)
        self.btn_pause = button("Pause", None, 140); self.btn_pause.clicked.connect(self._cancel); self.btn_pause.setEnabled(False); row.addWidget(self.btn_pause)
        imp = button("Import from folder…"); imp.clicked.connect(self._import_models); row.addWidget(imp)
        row.addStretch(1)
        self.btn_setup_back = button("Back"); self.btn_setup_back.clicked.connect(lambda: self.go(PAGE_OPTIONS)); row.addWidget(self.btn_setup_back)
        outer.addLayout(row); outer.addStretch(1)
        outer.addWidget(label("Licence: InsightFace models (ArcFace, inswapper) are for personal / non-commercial use only. "
                              "GPEN has no published licence (research release).", "hint", True))
        return w

    def _selected_specs(self):
        return [spec for spec, cb, _ in self.setup_rows.values() if cb.isChecked() and not self.store.is_installed(spec)]

    def _refresh_setup(self):
        for spec, cb, st in self.setup_rows.values():
            st.setText("✓ installed" if self.store.is_installed(spec) else "")
        need = self._selected_specs(); n = sum(s.bytes for s in need)
        self.setup_total.setText(f"To download: {n / 1e6:.0f} MB" if need else "Everything selected is installed.")
        self.btn_setup_back.setVisible(self.store.all_installed())
        if not self.worker or not self.worker.isRunning():
            self.btn_dl.setText("Download" if need else "Continue")

    def _download(self):
        need = self._selected_specs()
        if not need:
            self.go(PAGE_MAIN); QTimer.singleShot(100, self._start_bench); return
        total = sum(s.bytes for s in need)
        self.btn_dl.setEnabled(False); self.btn_pause.setEnabled(True)
        self.setup_t0 = time.time()

        def work(cancelled, emit):
            done_before = [0]
            for spec in need:
                def prog(file, done, tot, bps, verifying, base=done_before[0]):
                    emit(dict(done=base + done, total=total, file=file, bps=bps, verifying=verifying))
                self.store.ensure([spec], progress=prog, cancelled=cancelled)
                done_before[0] += spec.bytes
            return True
        self.worker = Worker(work, self)
        self.worker.progressed.connect(self._setup_progress)
        self.worker.done.connect(lambda _: (self._refresh_setup(), self.btn_dl.setEnabled(True), self.btn_pause.setEnabled(False),
                                            self.go(PAGE_MAIN), QTimer.singleShot(100, self._start_bench)))
        self.worker.failed.connect(self._setup_failed)
        self.worker.start()

    def _setup_progress(self, d):
        self.setup_bar.setValue(int(1000 * d["done"] / max(1, d["total"])))
        if d["verifying"]:
            self.setup_status.setText(f"Checking {d['file']} (SHA-256)…")
        else:
            eta = (d["total"] - d["done"]) / d["bps"] if d["bps"] > 0 else None
            self.setup_status.setText(f"Downloading {d['file']} · {d['done'] / 1e6:.0f} of {d['total'] / 1e6:.0f} MB"
                                      + (f" · {d['bps'] / 1e6:.1f} MB/s · about {media.fmt_time(eta)[:-2]} left" if eta else ""))

    def _setup_failed(self, msg, tb):
        self.btn_dl.setEnabled(True); self.btn_pause.setEnabled(False); self._refresh_setup()
        if msg == "cancelled":
            self.setup_status.setText("Paused — tap Download to resume."); self.btn_dl.setText("Resume")
        else:
            self.setup_status.setText(f"Download failed: {msg}\nCheck the internet connection and tap Download to retry (it resumes).")

    def _import_models(self):
        d = QFileDialog.getExistingDirectory(self, "Folder that contains the .onnx model files")
        if not d: return
        got = self.store.import_from(d, [s for s, _, _ in self.setup_rows.values()])
        self._refresh_setup()
        QMessageBox.information(self, "Import", f"Imported: {', '.join(got)}" if got else "No matching model files (name + SHA-256) in that folder.")

    # ------------------------------------------------------------------ main
    def _page_main(self):
        w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(16, 12, 16, 12); lay.setSpacing(10)
        mode_row = QHBoxLayout(); mode_row.setSpacing(8)
        mode_row.addWidget(label("Mode", "section"))
        self.btn_mode_video = button("Video swap", "primary", 160)
        self.btn_mode_photo = button("Photo swap", None, 160)
        self.btn_mode_video.clicked.connect(lambda: self._set_mode("video"))
        self.btn_mode_photo.clicked.connect(lambda: self._set_mode("photo"))
        mode_row.addWidget(self.btn_mode_video); mode_row.addWidget(self.btn_mode_photo); mode_row.addStretch(1)
        lay.addLayout(mode_row)
        row = QHBoxLayout(); row.setSpacing(12)
        self.vslot = ImageSlot("1 · Target video / photo", "Tap to choose a video or photo\nor drop it here", 600, 262)
        self.vslot.clicked.connect(self._pick_target); self.vslot.dropped.connect(self.open_path)
        self.pslot = ImageSlot("2 · Photo with the faces", "Tap to choose a photo\nor drop it here", 600, 262)
        self.pslot.clicked.connect(self._pick_photo); self.pslot.dropped.connect(self.open_path)
        row.addWidget(self.vslot, 1); row.addWidget(self.pslot, 1); lay.addLayout(row)
        # pairing strip + trim
        mid = QHBoxLayout(); mid.setSpacing(12)
        pc = card(); pl = QHBoxLayout(pc); pl.setContentsMargins(14, 8, 14, 8)
        self.pair_box = QHBoxLayout(); self.pair_box.setSpacing(8); pl.addLayout(self.pair_box, 1)
        self.pair_hint = label("Faces are paired left → right.", "hint", True); self.pair_box.addWidget(self.pair_hint)
        self.btn_flip = button("⇄  Flip", None, 120); self.btn_flip.clicked.connect(self._flip); pl.addWidget(self.btn_flip)
        mid.addWidget(pc, 3)
        tc = card(); tl = QGridLayout(tc); tl.setContentsMargins(14, 8, 14, 8); tl.setVerticalSpacing(2)
        self.s_start = QSlider(Qt.Orientation.Horizontal); self.s_len = QSlider(Qt.Orientation.Horizontal)
        for s in (self.s_start, self.s_len): s.setMinimumHeight(40); s.setEnabled(False)
        self.l_start = label("Start 0:00.0"); self.l_len = label("Length 10 s")
        tl.addWidget(self.l_start, 0, 0); tl.addWidget(self.s_start, 0, 1); tl.addWidget(self.l_len, 1, 0); tl.addWidget(self.s_len, 1, 1)
        self.s_start.valueChanged.connect(self._trim_changed); self.s_len.valueChanged.connect(self._trim_changed)
        self.s_start.sliderReleased.connect(self._refresh_video_faces)
        mid.addWidget(tc, 2)
        lay.addLayout(mid)
        bottom = QHBoxLayout(); bottom.setSpacing(12)
        self.summary = label("", "subtitle", True); bottom.addWidget(self.summary, 1)
        self.btn_preview = button("Preview", None, 170); self.btn_preview.clicked.connect(self._preview); bottom.addWidget(self.btn_preview)
        self.btn_start = button("▶  Swap video", "primary", 230); self.btn_start.clicked.connect(self._start); bottom.addWidget(self.btn_start)
        lay.addLayout(bottom)
        return w

    def _set_mode(self, mode):
        self.mode = mode
        self.btn_mode_video.setObjectName("primary" if mode == "video" else "")
        self.btn_mode_photo.setObjectName("primary" if mode == "photo" else "")
        for b in (self.btn_mode_video, self.btn_mode_photo):
            b.style().unpolish(b); b.style().polish(b)
        self.btn_start.setText("▶  Swap video" if mode == "video" else "▶  Swap photo")
        self.vslot.title.setText("1 · Couple video" if mode == "video" else "1 · Target photo")
        self.s_start.setVisible(mode == "video"); self.s_len.setVisible(mode == "video")
        self.l_start.setVisible(mode == "video"); self.l_len.setVisible(mode == "video")
        self.btn_preview.setVisible(mode == "video")
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
        self.target_photo = p; self.info = None
        self.vslot.set_image(img)
        self.vslot.info.setText(f"{Path(p).name}\n{img.shape[1]}×{img.shape[0]}")
        self._update_summary()

    def _pick_video(self):
        p, _ = QFileDialog.getOpenFileName(self, "Choose the couple video", str(Path.home() / "Videos"),
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
            if self.mode == "photo" and self.target_photo is None and self.photo is not None:
                # already have source faces — treat as target
                self.set_target_photo(p)
            elif self.mode == "photo" and self.vslot.pixmap() is None:
                self.set_target_photo(p)
            else:
                self.set_photo(p)
        else: QMessageBox.warning(self, "Unsupported file", f"{Path(p).name} isn't a video or photo this app can open.")

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls(): e.acceptProposedAction()

    def dropEvent(self, e):
        for u in e.mimeData().urls(): self.open_path(u.toLocalFile())

    def set_video(self, p):
        try:
            info = media.probe(p)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Video", str(e)); return
        self.info = info; self.job.analysis = None
        dur = info.duration
        self.s_start.blockSignals(True); self.s_len.blockSignals(True)
        self.s_start.setEnabled(True); self.s_len.setEnabled(True)
        self.s_start.setRange(0, max(0, int((dur - 0.5) * 10))); self.s_start.setValue(0)
        self.s_len.setRange(5, max(5, int(min(MAX_CLIP_S, dur) * 10))); self.s_len.setValue(int(min(10.0, dur) * 10))
        self.s_start.blockSignals(False); self.s_len.blockSignals(False)
        warn = "  ·  ⚠ HDR video: colours may look flat" if info.hdr else ""
        self.vslot.info.setText(f"{Path(p).name}\n{info.summary()}{warn}")
        self._trim_changed(); self._refresh_video_faces()

    def _refresh_video_faces(self):
        if not self.info: return
        fr = media.read_frame_at(self.info.path, self.s_start.value() / 10)
        if fr is None: return
        self.thumb = fr
        try:
            eng = self.job.get_engine(); eng.prepare(None)
            detect.set_default_engine(eng)
            self.vfaces = sorted(detect.detect_frame(fr, 2, self.settings().detect_opts(), engine=eng), key=lambda f: f[:, 0].mean())
        except Exception:  # noqa: BLE001
            self.vfaces = []
        self._redraw()

    def set_photo(self, p):
        try:
            ph = load_photo(p, opts=self.settings().detect_opts(), store=self.store, device=self.job.device)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Photo", str(e)); return
        if not ph.faces:
            QMessageBox.warning(self, "Photo", "No face found in that photo. Use a clear, front-facing photo."); return
        self.photo = ph; self.rotation = 0; self.job.analysis = None
        genders = ", ".join((g or "?") for g in (ph.genders or []))
        confs = ", ".join(f"{h.confidence:.0%}" for h in (ph.hits or []))
        extra = (f" · {genders}" if genders else "") + (f" · conf {confs}" if confs else "")
        self.pslot.info.setText(f"{Path(p).name}\n{len(ph.faces)} face{'s' if len(ph.faces) != 1 else ''} found{extra}")
        self._redraw()

    def _assign(self):
        n = len(self.photo.faces) if self.photo else 0
        return vcore.pair_single_frame(self.vfaces, n, self.rotation) if n else [-1] * len(self.vfaces)

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
                    lab = f"{letters[i]} {h.confidence:.0%} {g}"
                plabels.append(lab)
            self.pslot.set_image(draw_faces(self.photo.img, self.photo.faces, plabels, [ORANGE] * n))
        if self.thumb is not None:
            asg = self._assign()
            labels = [f"{i + 1}" + (f" ← {letters[a]}" if a >= 0 else "") for i, a in enumerate(asg)]
            self.vslot.set_image(draw_faces(self.thumb, self.vfaces, labels) if self.vfaces else self.thumb)
        # pairing strip
        while self.pair_box.count():
            it = self.pair_box.takeAt(0)
            if it.widget() and it.widget() is not self.pair_hint: it.widget().deleteLater()
        if self.photo and self.vfaces:
            self.pair_hint.hide()
            for i, a in enumerate(self._assign()):
                if a < 0: continue
                for img, f, tag in ((self.thumb, self.vfaces[i], f"{i + 1}"), (self.photo.img, self.photo.faces[a], letters[a])):
                    l = QLabel(); l.setPixmap(pix(face_crop(img, f), 56, 56)); l.setToolTip(tag); self.pair_box.addWidget(l)
                    if img is self.thumb: self.pair_box.addWidget(label("←", "arrow"))
                self.pair_box.addSpacing(18)
            self.pair_box.addStretch(1)
        else:
            self.pair_box.addWidget(self.pair_hint); self.pair_hint.show()
            self.pair_hint.setText("Faces are paired left → right. Pick a video and a photo to see who gets which face.")
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
        if enh == "auto": enh = self.job.recommended_enhance() if self.bench_done else ("gpen256" if self.store.is_installed(ENHANCER_LIGHT) else None)
        if enh and not self.store.is_installed(ENHANCE_SPECS[enh]): enh = None
        st = self.s_start.value() / 10 if self.info else 0.0
        ln = self.s_len.value() / 10 if self.info else 10.0
        return Settings(start=st, length=ln, fps=self.opt["fps"], max_short=self.opt["max_short"], enhance=enh,
                        rotation=self.rotation, device=self.job.device, out_dir=self.opt["out_dir"],
                        min_confidence=float(self.opt.get("min_confidence", 0.62)),
                        same_gender=bool(self.opt.get("same_gender", True)),
                        color_match=bool(self.opt.get("color_match", True)),
                        temporal_smooth=float(self.opt.get("temporal_smooth", 0.18)),
                        seamless=bool(self.opt.get("seamless", False)))

    def _update_summary(self):
        st = self.settings()
        parts = []
        if self.info:
            W, H, _ = vcore.out_size(self.info.width, self.info.height, st.max_short, st.align)
            fps = st.effective_fps(self.info.fps)
            ln = min(st.length, self.info.duration - st.start)
            n = int(ln * fps); faces = min(2, len(self.photo.faces)) if self.photo else 2
            est = self.job.estimate(n, faces, st.enhance, W, H)
            parts.append(f"{W}×{H} · {fps:.3g} fps · {n} frames · Enhance {ENHANCE_LABEL[st.enhance]}"
                         + ("" if self.opt["enhance"] != "auto" else " (auto)"))
            parts.append(f"≈ {media.fmt_time(est)[:-2]} on {'GPU' if self.job.engine and self.job.engine.info.active == 'DirectML' else 'this PC'}"
                         + ("" if self.bench_done else " (estimate before speed test)"))
        else:
            parts.append(f"Output up to {st.max_short}p · {st.fps:.0f} fps · Enhance {ENHANCE_LABEL[st.enhance]}")
        self.summary.setText("\n".join(parts))
        ready = bool(self.info and self.photo)
        self.btn_start.setEnabled(ready); self.btn_preview.setEnabled(ready)

    # ------------------------------------------------------------------ benchmark
    def _start_bench(self):
        if self.bench_done or (self.worker and self.worker.isRunning()): return
        self.chip.setText("Measuring speed…")

        def work(cancelled, emit):
            return self.job.benchmark()
        self.bw = Worker(work, self)
        self.bw.done.connect(self._bench_done); self.bw.failed.connect(lambda m, t: (self.set_chip(), self.chip.setToolTip(m)))
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
            im = image_label(420); cl.addWidget(im, 1)
            row.addWidget(c, 1); self.prev_imgs.append(im)
        lay.addLayout(row, 1)
        self.prev_info = label("", "subtitle", True); lay.addWidget(self.prev_info)
        b = QHBoxLayout()
        back = button("‹  Back", None, 150); back.clicked.connect(lambda: self.go(PAGE_MAIN)); b.addWidget(back)
        fl = button("⇄  Flip", None, 150); fl.clicked.connect(lambda: (self._flip(), self._preview())); b.addWidget(fl)
        b.addStretch(1)
        go = button("▶  Swap video", "primary", 230); go.clicked.connect(self._start); b.addWidget(go)
        lay.addLayout(b)
        return w

    def _preview(self):
        if not (self.info and self.photo) or (self.worker and self.worker.isRunning()): return
        st = self.settings()
        self.btn_preview.setText("Working…"); self.btn_preview.setEnabled(False)
        t = self.s_start.value() / 10

        def work(cancelled, emit):
            t0 = time.perf_counter(); r = self.job.preview(self.info, self.photo, st, t); return r, time.perf_counter() - t0, st
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
        self.prev_info.setText(f"{after.shape[1]}×{after.shape[0]} · Enhance {ENHANCE_LABEL[st.enhance]} · {len([a for a in asg if a >= 0])} face(s) · "
                               f"{secs:.1f} s incl. model loading · {self.job.engine.info.label()}")
        self.btn_preview.setText("Preview"); self.btn_preview.setEnabled(True)
        self.set_chip(); self.go(PAGE_PREVIEW)

    # ------------------------------------------------------------------ options
    def _page_options(self):
        w = QWidget(); outer = QVBoxLayout(w); outer.setContentsMargins(0, 0, 0, 0)
        sc = QScrollArea(); sc.setWidgetResizable(True); inner = QWidget(); lay = QVBoxLayout(inner)
        lay.setContentsMargins(24, 14, 24, 14); lay.setSpacing(8)
        lay.addWidget(label("Options", "title"))
        lay.addWidget(label("Output resolution (short side; never upscales)", "section"))
        self.seg_res = Segmented([("480p", 480), ("720p", 720), ("1080p", 1080)]); lay.addWidget(self.seg_res)
        self.seg_res.changed.connect(lambda v: self._set_opt("max_short", v))
        lay.addWidget(label("Frame rate (never above the source)", "section"))
        self.seg_fps = Segmented([("15 fps", 15.0), ("24 fps", 24.0), ("30 fps", 30.0), ("Original", 0.0)]); lay.addWidget(self.seg_fps)
        self.seg_fps.changed.connect(lambda v: self._set_opt("fps", v))
        lay.addWidget(label("Enhance face detail", "section"))
        self.seg_enh = Segmented([("Auto", "auto"), ("Off", None), ("Light", "gpen256"), ("HQ", "gpen512")]); lay.addWidget(self.seg_enh)
        self.seg_enh.changed.connect(lambda v: self._set_opt("enhance", v))
        self.enh_info = label("", "hint", True); lay.addWidget(self.enh_info)
        lay.addWidget(label("Processor", "section"))
        self.seg_dev = Segmented([("Auto (GPU if possible)", "auto"), ("GPU (DirectML)", "dml"), ("CPU only", "cpu")]); lay.addWidget(self.seg_dev)
        self.seg_dev.changed.connect(self._set_device)
        self.dev_info = label("", "hint", True); lay.addWidget(self.dev_info)
        lay.addWidget(label("Face detection (v2)", "section"))
        self.seg_conf = Segmented([("Strict 70%", 0.70), ("Default 62%", 0.62), ("Loose 50%", 0.50)])
        lay.addWidget(self.seg_conf)
        self.seg_conf.changed.connect(lambda v: self._set_opt("min_confidence", v))
        self.chk_gender = QCheckBox("Same-gender match (recommended)")
        self.chk_gender.setChecked(True)
        self.chk_gender.stateChanged.connect(lambda _=0: self._set_opt("same_gender", self.chk_gender.isChecked()))
        lay.addWidget(self.chk_gender)
        self.chk_color = QCheckBox("Match skin / body colour + soft hairline blend")
        self.chk_color.setChecked(True)
        self.chk_color.stateChanged.connect(lambda _=0: self._set_opt("color_match", self.chk_color.isChecked()))
        lay.addWidget(self.chk_color)
        self.chk_seam = QCheckBox("Seamless (Poisson) blend — slower, stronger hairline")
        self.chk_seam.stateChanged.connect(lambda _=0: self._set_opt("seamless", self.chk_seam.isChecked()))
        lay.addWidget(self.chk_seam)
        lay.addWidget(label("Temporal smooth (cuts flicker; landmarks still follow expression)", "section"))
        self.seg_smooth = Segmented([("Off", 0.0), ("Light", 0.12), ("Medium", 0.18), ("Strong", 0.30)])
        lay.addWidget(self.seg_smooth)
        self.seg_smooth.changed.connect(lambda v: self._set_opt("temporal_smooth", v))
        lay.addWidget(label("Expression follows the video person every frame (mouth open/smile/blink). "
                            "Source photo is identity only — not a frozen face paste.", "hint", True))
        lay.addWidget(label("Save to", "section"))

        r = QHBoxLayout(); self.out_label = label("", None, True); r.addWidget(self.out_label, 1)
        ch = button("Change…"); ch.clicked.connect(self._change_out); r.addWidget(ch)
        op = button("Open folder"); op.clicked.connect(lambda: self._open_folder(self.opt["out_dir"])); r.addWidget(op)
        lay.addLayout(r)
        r2 = QHBoxLayout(); mm = button("Manage models…"); mm.clicked.connect(lambda: (self._refresh_setup(), self.go(PAGE_SETUP))); r2.addWidget(mm); r2.addStretch(1)
        lay.addLayout(r2)
        lay.addStretch(1)
        sc.setWidget(inner); outer.addWidget(sc, 1)
        bar = QHBoxLayout(); bar.setContentsMargins(24, 6, 24, 12)
        done = button("‹  Done", "primary", 200); done.clicked.connect(lambda: self.go(PAGE_MAIN)); bar.addWidget(done); bar.addStretch(1)
        outer.addLayout(bar)
        return w

    def _set_opt(self, k, v):
        self.opt[k] = v; self.cfg.setValue(k, v if v is not None else "off"); self.job.analysis = None if k in ("max_short", "fps", "min_confidence", "same_gender") else self.job.analysis
        self._refresh_options()

    def _set_device(self, v):
        self.cfg.setValue("device", v); self.job.get_engine(v); self.bench_done = False
        self.set_chip(); self._refresh_options(); self._start_bench()

    def _change_out(self):
        d = QFileDialog.getExistingDirectory(self, "Save videos to", self.opt["out_dir"])
        if d: self._set_opt("out_dir", d)

    def _refresh_options(self):
        if self.opt["enhance"] == "off": self.opt["enhance"] = None
        self.seg_res.set(self.opt["max_short"]); self.seg_fps.set(self.opt["fps"]); self.seg_enh.set(self.opt["enhance"])
        self.seg_dev.set(self.job.device)
        if hasattr(self, "seg_conf"):
            # pick nearest preset
            c = float(self.opt.get("min_confidence", 0.62))
            best = min(self.seg_conf.buttons.keys(), key=lambda v: abs(v - c))
            self.seg_conf.set(best)
            self.chk_gender.setChecked(bool(self.opt.get("same_gender", True)))
            self.chk_color.setChecked(bool(self.opt.get("color_match", True)))
            self.chk_seam.setChecked(bool(self.opt.get("seamless", False)))
            sm = float(self.opt.get("temporal_smooth", 0.18))
            bests = min(self.seg_smooth.buttons.keys(), key=lambda v: abs(v - sm))
            self.seg_smooth.set(bests)
        for m, spec in ENHANCE_SPECS.items():
            b = self.seg_enh.buttons[m]; inst = self.store.is_installed(spec)
            b.setText(f"{ENHANCE_LABEL[m]}" + ("" if inst else f"  (download {spec.bytes / 1e6:.0f} MB)"))
            b.setEnabled(inst)
        n = 300; lines = []
        if self.info:
            fps = Settings(fps=self.opt["fps"]).effective_fps(self.info.fps); n = int(min(10, self.info.duration) * fps)
        W, H = (1920, 1080) if self.opt["max_short"] >= 1080 else ((1280, 720) if self.opt["max_short"] >= 720 else (854, 480))
        for m in (None, "gpen256", "gpen512"):
            if m and not self.store.is_installed(ENHANCE_SPECS[m]): continue
            lines.append(f"{ENHANCE_LABEL[m]} ≈ {media.fmt_time(self.job.estimate(n, 2, m, W, H))[:-2]}")
        rec = self.job.recommended_enhance()
        self.enh_info.setText(("Estimated time for a 10 s clip with 2 faces: " + " · ".join(lines)) +
                              (f".  Auto picks {ENHANCE_LABEL[rec]} on this device." if self.bench_done else ".  (Speed test pending.)") +
                              "\nLight = GPEN-256 (sharper face, small cost). HQ = GPEN-512 (best detail; fast on the GPU, slow on CPU).")
        eng = self.job.engine
        self.dev_info.setText(("Now using: " + eng.info.label()) if eng else "")
        self.out_label.setText(self.opt["out_dir"])

    # ------------------------------------------------------------------ run
    def _page_progress(self):
        w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(24, 14, 24, 14); lay.setSpacing(10)
        self.prog_title = label("Swapping faces…", "title"); lay.addWidget(self.prog_title)
        self.steps = label("", "subtitle"); lay.addWidget(self.steps)
        self.prog_bar = QProgressBar(); self.prog_bar.setRange(0, 1000); self.prog_bar.setTextVisible(False); self.prog_bar.setMinimumHeight(28); lay.addWidget(self.prog_bar)
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
        if self.worker and self.worker.isRunning(): return
        if self.mode == "photo":
            if not (self.target_photo and self.photo): return
        elif not (self.info and self.photo):
            return
        st = self.settings()
        if self.mode == "photo":
            from ..job import default_pictures_dir
            st.out_dir = st.out_dir if "FaceFusion" in st.out_dir else str(default_pictures_dir())
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
        self.worker.progressed.connect(self._progress); self.worker.done.connect(self._finished); self.worker.failed.connect(self._failed)
        self.worker.start()

    def _steps(self, cur):
        names = [("detect", "1  Find & track faces"), ("swap", "2  Swap faces"), ("mux", "3  Save MP4 + sound")]
        order = [n for n, _ in names]; ci = order.index(cur) if cur in order else 3
        self.steps.setText("     ".join(("✓ " if i < ci else ("● " if i == ci else "○ ")) + t for i, (n, t) in enumerate(names)))

    def _progress(self, d):
        stage = d.get("stage"); done, total = d.get("done", 0), max(1, d.get("total", 1))
        if stage in ("detect", "swap", "mux"): self._steps(stage)
        if stage == "detect":
            self.prog_bar.setValue(int(150 * done / total)); self.prog_text.setText(f"Finding faces · frame {done} of {total}")
        elif stage == "swap":
            self.prog_bar.setValue(150 + int(830 * done / total))
            eta = d.get("eta"); rate = d.get("rate") or 0
            self.prog_text.setText(f"Frame {done} of {total} · {rate:.1f} frames/s" + (f" · about {media.fmt_time(eta)[:-2]} left" if eta else ""))
        elif stage == "mux":
            self.prog_bar.setValue(990); self.prog_text.setText("Saving MP4 and copying the sound…")
        if "thumb" in d:
            self.prog_imgs[0].setPixmap(pix(d["before"], 600, 330)); self.prog_imgs[1].setPixmap(pix(d["thumb"], 600, 330))
        if self.job.engine: self.prog_dev.setText(f"{self.job.engine.info.label()} · elapsed {media.fmt_time(time.time() - self.run_t0)[:-2]}"); self.set_chip()

    def _cancel(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancelled = True; self.prog_text.setText("Cancelling…"); self.btn_cancel.setEnabled(False)
            self.btn_pause.setEnabled(False)

    def _failed(self, msg, tb):
        self.btn_preview.setText("Preview"); self.btn_preview.setEnabled(True)
        if msg == "cancelled":
            self.go(PAGE_MAIN); return
        box = QMessageBox(self); box.setIcon(QMessageBox.Icon.Warning); box.setWindowTitle("Something went wrong")
        box.setText(msg); box.setDetailedText(tb); box.exec()
        self.go(PAGE_MAIN if self.store.all_installed() else PAGE_SETUP)

    # ------------------------------------------------------------------ done
    def _page_done(self):
        w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(24, 14, 24, 14); lay.setSpacing(10)
        lay.addWidget(label("Done! Your video is saved.", "title"))
        self.done_path = label("", "section", True); self.done_path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse); lay.addWidget(self.done_path)
        self.done_img = image_label(390); lay.addWidget(self.done_img, 1)
        self.done_info = label("", "subtitle", True); lay.addWidget(self.done_info)
        b = QHBoxLayout(); b.setSpacing(12)
        play = button("▶  Play", "primary", 180); play.clicked.connect(lambda: self.result and QDesktopServices.openUrl(QUrl.fromLocalFile(self.result["path"]))); b.addWidget(play)
        fold = button("Open folder", None, 180); fold.clicked.connect(lambda: self.result and self._open_folder(self.result["path"], select=True)); b.addWidget(fold)
        redo = button("⇄  Flip && redo", None, 180); redo.clicked.connect(lambda: (self._flip(), self._start())); b.addWidget(redo)
        b.addStretch(1)
        new = button("New video", None, 160); new.clicked.connect(lambda: self.go(PAGE_MAIN)); b.addWidget(new)
        lay.addLayout(b)
        return w

    def _finished(self, res):
        self.result = res
        self.done_path.setText(res["path"])
        if res.get("mode") == "photo":
            fr = res.get("after") or detect.load_image(res["path"])
        else:
            fr = media.read_frame_at(res["path"], res.get("duration", 1) / 2)
        src = None
        if fr is not None and self.info:
            s0 = media.read_frame_at(self.info.path, self.s_start.value() / 10 + res["duration"] / 2)
            if s0 is not None:
                W, H, s = vcore.out_size(self.info.width, self.info.height, self.settings().max_short, 2)
                src = vcore.prep(s0, W, H, s)
                if src.shape == fr.shape:
                    fr = np.hstack([src, np.full((src.shape[0], 12, 3), 18, np.uint8), fr])
        if fr is not None: self.done_img.setPixmap(pix(fr, 1200, 390))
        mb = res["size"] / 1e6
        self.done_info.setText(f"{res['W']}×{res['H']} · {res['fps']:.3g} fps · {media.fmt_time(res['duration'])} · {mb:.1f} MB · "
                               f"Enhance {res['enhance']} · {res['audio']}\nTook {media.fmt_time(res['total_s'])[:-2]} on {res['device']} "
                               f"(encoder {res['encoder']})")
        self.go(PAGE_DONE)

    def _open_folder(self, p, select=False):
        p = str(p)
        if os.name == "nt" and select:
            import subprocess
            subprocess.Popen(["explorer", "/select,", os.path.normpath(p)]); return
        Path(p if not select else Path(p).parent).mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(p if not select else str(Path(p).parent)))

    # ------------------------------------------------------------------ keys (keyboard + gamepad)
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
QFrame#topbar { background:#0d0f14; border-bottom:1px solid #2a2f3a; }
QLabel#apptitle { font-size:19px; font-weight:600; color:#fff; background:transparent; }
QLabel#chip { background:#243b39; color:#7fe3d9; border-radius:13px; padding:5px 12px; font-size:13px; }
QLabel#chip[state="cpu"] { background:#3b3324; color:#ffcf8a; }
QLabel#section { font-size:16px; font-weight:600; color:#e8eaed; background:transparent; }
QLabel#slot { background:#12151c; border:2px dashed #3c4454; border-radius:12px; color:#80868b; font-size:16px; }
QLabel#arrow { font-size:22px; color:#9aa0a6; background:transparent; }
QFrame#card QLabel { background:transparent; }
QPushButton#seg { background:#1f232c; border:1px solid #3c4454; border-radius:10px; font-size:15px; }
QPushButton#seg:checked { background:#00b8a9; color:#041014; border:none; font-weight:600; }
QPushButton#seg:disabled { color:#5f6368; }
QCheckBox { spacing:12px; font-size:15px; background:transparent; min-height:40px; }
QCheckBox::indicator { width:28px; height:28px; }
QPushButton:focus, QCheckBox:focus, QSlider:focus { outline:none; border:2px solid #ff8a3d; }
QMessageBox QLabel { background:transparent; }
"""


def make_app(argv=None):
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    app = QApplication.instance() or QApplication(argv or sys.argv)
    app.setApplicationName("Face Fusion Studio"); app.setOrganizationName("vanu")
    app.setStyle("Fusion"); app.setStyleSheet(DARK_QSS + EXTRA_QSS)
    ico = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[2] / "packaging")) / "icon.ico"
    if ico.is_file(): app.setWindowIcon(QIcon(str(ico)))
    return app


def run_gui(args=None) -> int:
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
    store = ModelStore(Path(args.models) if args is not None and args.models else None)
    win = MainWindow(store, getattr(args, "device", "auto") if args is not None else "auto")
    from .gamepad import Gamepad
    win.gamepad = Gamepad(win)
    win.resize(1280, 720)
    if QApplication.primaryScreen() and QApplication.primaryScreen().availableGeometry().width() <= 1400:
        win.showMaximized()          # Ally X: 1920x1080 @150% = 1280x720 logical
    else:
        win.show()
    for p in (getattr(args, "video", None), getattr(args, "photo", None)):
        if p: win.open_path(p)
    return app.exec()
