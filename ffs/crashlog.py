"""Crash log for the windowed app.

Native faults (access violations inside DirectML / OpenCV) and uncaught Python errors are written to
%LOCALAPPDATA%\\FaceFusionStudio\\crash.log. The GUI process has no console, so without this a crash
looks like the window simply closed.
"""
from __future__ import annotations

import faulthandler
import os
import sys
import threading
import traceback
from pathlib import Path

_fh = None
_installed = False


def log_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA")
    root = Path(base) if base else (Path.home() / ".local" / "share")
    return root / "FaceFusionStudio"


def crash_path() -> Path:
    return log_dir() / "crash.log"


def install() -> Path:
    """Idempotent. Call as early as possible, before ONNX or Qt."""
    global _fh, _installed
    path = crash_path()
    if _installed:
        return path
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _fh = open(path, "a", buffering=1, encoding="utf-8")
        _fh.write("\n----- crash log armed -----\n")
        _fh.flush()
        faulthandler.enable(_fh, all_threads=True)
    except Exception:
        _fh = None
    sys.excepthook = _sys_hook
    if hasattr(threading, "excepthook"):
        threading.excepthook = _thread_hook
    if hasattr(sys, "unraisablehook"):
        sys.unraisablehook = _unraisable_hook
    _installed = True
    return path


def record(etype, value, tb, where="uncaught") -> None:
    text = "".join(traceback.format_exception(etype, value, tb))
    line = f"\n===== {where} =====\n{text}\n"
    try:
        if _fh is not None:
            _fh.write(line)
            _fh.flush()
        else:
            p = crash_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "a", encoding="utf-8") as f:
                f.write(line)
    except Exception:
        pass


def record_current(where="error") -> None:
    record(*sys.exc_info(), where=where)


def _sys_hook(etype, value, tb):
    record(etype, value, tb, where="sys.excepthook")
    try:
        sys.__excepthook__(etype, value, tb)
    except Exception:
        pass


def _thread_hook(args):
    record(args.exc_type, args.exc_value, args.exc_traceback,
           where=f"thread {getattr(args.thread, 'name', '?')}")


def _unraisable_hook(args):
    record(args.exc_type, args.exc_value, args.exc_traceback, where="unraisable")
