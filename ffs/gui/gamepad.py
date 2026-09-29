"""Minimal XInput gamepad navigation (Windows only, no extra dependency).

The ROG Ally X's built-in controller is an XInput device. D-pad / left stick move keyboard focus (Tab /
Shift+Tab, or Left/Right on sliders), A = activate (Space), B = back (Escape), Start = start job (Enter).
Polled at 60 Hz with a QTimer; does nothing on other platforms or when no pad is connected.
"""
from __future__ import annotations

import ctypes
import os
import time

from PySide6.QtCore import QObject, QTimer, Qt, QEvent
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QAbstractSlider, QApplication

DPAD_UP, DPAD_DOWN, DPAD_LEFT, DPAD_RIGHT = 0x0001, 0x0002, 0x0004, 0x0008
START, BACK, BTN_A, BTN_B = 0x0010, 0x0020, 0x1000, 0x2000


class _GAMEPAD(ctypes.Structure):
    _fields_ = [("wButtons", ctypes.c_ushort), ("bLeftTrigger", ctypes.c_ubyte), ("bRightTrigger", ctypes.c_ubyte),
                ("sThumbLX", ctypes.c_short), ("sThumbLY", ctypes.c_short), ("sThumbRX", ctypes.c_short), ("sThumbRY", ctypes.c_short)]


class _STATE(ctypes.Structure):
    _fields_ = [("dwPacketNumber", ctypes.c_uint), ("Gamepad", _GAMEPAD)]


class Gamepad(QObject):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.dll = None
        if os.name == "nt":
            for name in ("xinput1_4", "xinput1_3", "xinput9_1_0"):
                try:
                    self.dll = getattr(ctypes.windll, name); break
                except OSError:
                    continue
        self.prev = 0
        self.repeat_at = {}
        self.timer = QTimer(self); self.timer.timeout.connect(self.poll)
        if self.dll is not None:
            self.timer.start(16)

    def _buttons(self):
        st = _STATE()
        if self.dll.XInputGetState(0, ctypes.byref(st)) != 0:
            return 0
        b = st.Gamepad.wButtons; lx, ly = st.Gamepad.sThumbLX, st.Gamepad.sThumbLY
        if ly > 16000: b |= DPAD_UP
        if ly < -16000: b |= DPAD_DOWN
        if lx < -16000: b |= DPAD_LEFT
        if lx > 16000: b |= DPAD_RIGHT
        return b

    def poll(self):
        b = self._buttons(); now = time.monotonic()
        for mask in (DPAD_UP, DPAD_DOWN, DPAD_LEFT, DPAD_RIGHT, BTN_A, BTN_B, START):
            down, was = bool(b & mask), bool(self.prev & mask)
            if down and not was:
                self.repeat_at[mask] = now + 0.4; self._fire(mask)
            elif down and mask & 0x000F and now >= self.repeat_at.get(mask, 1e18):
                self.repeat_at[mask] = now + 0.12; self._fire(mask)
        self.prev = b

    def _fire(self, mask):
        w = QApplication.focusWidget() or QApplication.activeWindow()
        if w is None: return
        slider = isinstance(w, QAbstractSlider)
        key, mod = {
            DPAD_UP: (Qt.Key.Key_Backtab, Qt.KeyboardModifier.ShiftModifier),
            DPAD_DOWN: (Qt.Key.Key_Tab, Qt.KeyboardModifier.NoModifier),
            DPAD_LEFT: (Qt.Key.Key_Left if slider else Qt.Key.Key_Backtab, Qt.KeyboardModifier.NoModifier if slider else Qt.KeyboardModifier.ShiftModifier),
            DPAD_RIGHT: (Qt.Key.Key_Right if slider else Qt.Key.Key_Tab, Qt.KeyboardModifier.NoModifier),
            BTN_A: (Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier),
            BTN_B: (Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier),
            START: (Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier),
        }[mask]
        for t in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
            QApplication.sendEvent(w, QKeyEvent(t, key, mod))
