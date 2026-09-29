"""Render the real UI states to PNGs (for README / release notes / CI evidence).

Everything shown is real: an actual partial model download into a temporary folder, real face detection,
a real preview, and a real short job (progress captured mid-run, then the Done page with its result).
Run with QT_QPA_PLATFORM=offscreen for headless capture. 1280x720 logical at QT_SCALE_FACTOR=1.5 gives
1920x1080 PNGs, i.e. what the Ally X shows at 150 % scaling.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path

from PySide6.QtWidgets import QApplication

from ..models import ModelStore
from .app import PAGE_MAIN, PAGE_OPTIONS, PAGE_PROGRESS, PAGE_SETUP, MainWindow, make_app


def take_screenshots(args) -> int:
    os.environ.setdefault("QT_SCALE_FACTOR", "1.5")
    out = Path(args.screenshots); out.mkdir(parents=True, exist_ok=True)
    app = make_app(["FaceFusionStudio"])

    def spin(sec=0.05, until=None, timeout=600):
        t0 = time.time()
        while True:
            app.processEvents(); time.sleep(0.02)
            if until is None and time.time() - t0 >= sec: return True
            if until is not None and until(): return True
            if time.time() - t0 > timeout: return False

    def shot(win, name):
        spin(0.2)
        p = out / f"{name}.png"; win.grab().save(str(p)); print("wrote", p)

    # 1-2) Setup: empty store, then a real download for a few seconds (cancelled -> .part kept)
    tmp = Path(tempfile.mkdtemp(prefix="ffs_setup_"))
    w0 = MainWindow(ModelStore(tmp), "cpu"); w0.resize(1280, 720); w0.show()
    w0.go(PAGE_SETUP); shot(w0, "01_setup")
    w0._download()
    spin(until=lambda: w0.setup_bar.value() > 25 or not w0.worker.isRunning(), timeout=60)
    shot(w0, "02_setup_downloading")
    w0._cancel(); spin(until=lambda: not w0.worker.isRunning(), timeout=60)
    w0.close(); shutil.rmtree(tmp, ignore_errors=True)

    store = ModelStore(Path(args.models) if args.models else None)
    win = MainWindow(store, getattr(args, "device", "auto")); win.resize(1280, 720); win.show()
    # measured speed test (the app also does this in the background on start)
    spin(until=lambda: win.bench_done or (hasattr(win, "bw") and not win.bw.isRunning()), timeout=600)
    win.go(PAGE_MAIN); shot(win, "03_main_empty")
    if not (args.video and args.photo):
        return 0
    win.set_video(args.video)
    spin(until=lambda: not win.worker or not win.worker.isRunning(), timeout=600)
    win.set_photo(args.photo)
    spin(until=lambda: win.photo is not None and (not win.worker or not win.worker.isRunning()), timeout=600)
    if args.length: win.s_len.setValue(int(args.length * 10))
    win.go(PAGE_MAIN); shot(win, "04_main_faces_pairing")
    win.go(PAGE_OPTIONS); shot(win, "05_options")
    win.go(PAGE_MAIN)
    win._preview(); spin(until=lambda: not win.worker.isRunning(), timeout=600); spin(0.3)
    shot(win, "06_preview_before_after")
    # real job; capture progress mid-run, then Done
    win._start()
    spin(until=lambda: (win.worker is not None and not win.worker.isRunning()) or
         (win.stack.currentIndex() == PAGE_PROGRESS and win.prog_bar.value() > 400 and win.prog_imgs[1].pixmap() is not None
          and not win.prog_imgs[1].pixmap().isNull()), timeout=1800)
    shot(win, "07_progress")
    spin(until=lambda: not win.worker.isRunning(), timeout=3600); spin(0.5)
    shot(win, "08_done")
    print("result:", win.result)
    return 0
