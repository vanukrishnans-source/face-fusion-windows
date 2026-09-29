"""Dark theme for a 7-inch 1080p touchscreen at ~150% Windows scaling. Tap targets ≥ 48 CSS-px."""

DARK_QSS = """
QWidget { background:#12141a; color:#e8eaed; font-size:15px; font-family:"Segoe UI","Noto Sans",sans-serif; }
QMainWindow, QDialog { background:#12141a; }
QLabel#title { font-size:22px; font-weight:600; color:#fff; }
QLabel#subtitle { font-size:13px; color:#9aa0a6; }
QLabel#hint { font-size:12px; color:#80868b; }
QPushButton { background:#2a2f3a; color:#e8eaed; border:1px solid #3c4454; border-radius:10px;
              padding:12px 18px; min-height:28px; font-size:15px; }
QPushButton:hover { background:#343b4a; }
QPushButton:pressed { background:#1e222b; }
QPushButton:disabled { color:#5f6368; background:#1a1d24; }
QPushButton#primary { background:#00b8a9; color:#041014; border:none; font-weight:600; }
QPushButton#primary:hover { background:#1ad1c1; }
QPushButton#primary:disabled { background:#1a4a45; color:#6a8a86; }
QPushButton#danger { background:#c5221f; color:#fff; border:none; }
QPushButton#danger:hover { background:#e33b38; }
QFrame#card { background:#1a1e27; border:1px solid #2a2f3a; border-radius:14px; }
QProgressBar { background:#1a1e27; border:1px solid #2a2f3a; border-radius:8px; text-align:center;
               min-height:22px; color:#e8eaed; }
QProgressBar::chunk { background:#00b8a9; border-radius:7px; }
QSlider::groove:horizontal { height:8px; background:#2a2f3a; border-radius:4px; }
QSlider::handle:horizontal { width:28px; height:28px; margin:-10px 0; background:#00b8a9; border-radius:14px; }
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
    background:#1a1e27; border:1px solid #3c4454; border-radius:8px; padding:10px 12px; min-height:24px; }
QComboBox::drop-down { width:36px; }
QComboBox QAbstractItemView { background:#1a1e27; selection-background-color:#00b8a9; selection-color:#041014; }
QScrollArea { border:none; }
QStatusBar { background:#0d0f14; color:#9aa0a6; font-size:12px; }
"""
