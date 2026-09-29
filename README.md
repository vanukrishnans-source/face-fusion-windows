# Face Fusion Studio for Windows

Touch-friendly **Windows x64** face-swap app for the **ASUS ROG Ally X** (Windows 11 handheld: AMD Ryzen Z1 Extreme, Radeon 780M, 24 GB LPDDR5X, 7″ 1920×1080 @ 120 Hz).

**Powered by FaceFusion-compatible ONNX models** (not the full FaceFusion GPL desktop app). Custom PySide6 UI + the same community processors FaceFusion documents: YOLO Face detector, ArcFace identity, inswapper_128 (fp16), optional GPEN / GFPGAN enhancer.

| | This app |
|---|---|
| Package | Face Fusion Studio **1.1.3** (portable zip) |
| Acceleration | **ONNX Runtime DirectML** on the Radeon 780M, automatic CPU fallback |
| Modes | **Photo swap** + **Video swap** |
| Default resolution | ≤ **1080p** short side (never upscales) |
| Default fps | **30** (or source if lower) |
| Max clip length | **5 minutes** |
| Output | `%USERPROFILE%\Videos\FaceFusion` (video) · `%USERPROFILE%\Pictures\FaceFusion` (photo) |

**Tech:** Python 3.13 + PySide6 + onnxruntime-directml + OpenCV, packaged with PyInstaller on GitHub Actions `windows-latest`. Bundled LGPL FFmpeg for H.264 encode + audio mux. **No MediaPipe.**

> Not affiliated with, endorsed by, or a fork of the FaceFusion project. We reuse redistributable model weights and processor ideas under their respective licences.

## Download

### What’s new in 1.1.3
- **Fix:** app no longer exits while “Finding faces” (DirectML was being called from the wrong thread / apartment).
- **Fix:** “Final / Finalize” (encode + audio mux) always finishes or fails with a clear timeout — no more hanging on FFmpeg.
- Real progress text during Final; Cancel still works; crashes write `%LOCALAPPDATA%\FaceFusionStudio\crash.log`.
- **Fix:** Photo mode Swap button now enables (was stuck requiring a video) and Done page no longer crashes after a photo swap.
- **UI:** Ally X wizard — big Photo|Video toggle, numbered steps, large face-map thumbnails, Segoe UI fonts (no tofu).
- **Fix:** GPU (DirectML) crash on Radeon 780M — probes DirectML in a child process; on any failure stays alive on CPU and shows “GPU failed, using CPU”.


Grab the latest **`FaceFusionStudio-*-win64.zip`** from the [Releases](https://github.com/vanukrishnans-source/face-fusion-windows/releases) page. No Python install needed.

### First launch (SmartScreen)

The build is **not code-signed**. Windows SmartScreen will say “Windows protected your PC”:

1. Click **More info**
2. Click **Run anyway**

Verify the zip:

```powershell
Get-FileHash .\FaceFusionStudio-1.1.3-win64.zip -Algorithm SHA256
```

### Install / run

1. Unzip anywhere (e.g. `C:\Games\FaceFusionStudio\`).
2. Double-click **`FaceFusionStudio.exe`**.
3. First run downloads **~465 MB** required models (YOLO Face 8n + ArcFace + inswapper_128 fp16). Optional: Light GPEN ≈ 76 MB, HQ GPEN ≈ 284 MB, RetinaFace ≈ 17 MB. InsightFace models are for **personal, non-commercial** use only.

Models stay in `%LOCALAPPDATA%\FaceFusionStudio\models`.

## Features

- **Photo mode** and **Video mode** (touch toggle)
- FaceFusion-class stack: **YOLO Face** detector (5-point landmarks every frame → expression follow), **ArcFace**, **inswapper_128 fp16**, optional **GPEN** enhancer
- Multi-face select / map (left→right + Flip), confidence threshold, same-gender (InsightFace genderage), colour match (Reinhard LAB), temporal EMA smooth for video
- Resolution / fps caps, DirectML + CPU fallback chip in the UI
- First-run model download with **SHA-256**, pause/resume, GitHub → Hugging Face mirror
- Touch-friendly dark UI for 7″ 1080p @ 150 % scaling; mouse + keyboard; basic XInput

## FaceFusion pieces included

| Role | Model | Size | Notes |
|---|---|---|---|
| Detector | `yoloface_8n.onnx` | ≈ 12.7 MB | FaceFusion models-3.0.0 (derronqi, GPL-3.0) |
| Detector (opt) | `retinaface_10g.onnx` | ≈ 16.9 MB | InsightFace / non-commercial |
| Identity | `arcface_w600k_r50.onnx` | ≈ 174 MB | InsightFace / non-commercial |
| Swapper | `inswapper_128_fp16.onnx` | ≈ 278 MB | InsightFace / non-commercial |
| Enhancer | `gpen_bfr_256.onnx` / `_512` | ≈ 76 / 284 MB | Alibaba DAMO research |
| Gender | `gender_age.onnx` | ≈ 1.3 MB | Bundled; InsightFace |

**Not included vs full FaceFusion desktop:** live preview webcam, face editor / expression restorer / lip sync, CodeFormer / GFPGAN in the default download (GFPGAN optional path exists), frame colorizer, many detectors at once, voice extractor, face occluder/parser, batch “jobs” UI, Linux/macOS/CUDA/TensorRT providers.

## Speed estimates (ASUS ROG Ally X · Radeon 780M · DirectML)

**Estimates only** — no Ally X hardware was available for measurement. CI verifies on `windows-latest` CPU (+ DirectML smoke which may use WARP).

| 10 s clip, 30 fps, 2 faces @ 720p | Off | Light | HQ |
|---|---|---|---|
| DirectML (est.) | **0.5–1.5 min** | **0.8–2 min** | **2–5 min** |
| CPU Zen 4 (est.) | 2–3 min | 2.5–4.5 min | 7.5–13 min |

Photo swap on Ally X: typically **1–4 s** per image with Light enhancer (est.).

## Licence / models

- **App code:** MIT (see `LICENSE`).
- **InsightFace models** (ArcFace, inswapper, RetinaFace, genderage): **personal / non-commercial research only**. Commercial use needs a licence from [insightface.ai](https://insightface.ai).
- **YOLO Face:** GPL-3.0 (derronqi).
- **GPEN-BFR:** research release; treat as personal / non-commercial.
- **FFmpeg:** LGPL build (see `THIRD_PARTY.md`).
- Don’t use this app to impersonate or deceive anyone; only swap faces of people who agreed to it.

## Build from source

Needs Windows x64 (or the GitHub Actions workflow).

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-win.txt
powershell -File packaging\download_ffmpeg.ps1
$env:FFS_MODELS = "$PWD\.models-cache"
python scripts/prefetch_models.py light
python -m ffs --selftest --video testdata\sample_2s.mp4 --photo testdata\faces_e5.jpg --models $env:FFS_MODELS --device cpu --enhance gpen256 --length 2
pyinstaller FaceFusionStudio.spec --noconfirm
powershell -File packaging\make_zip.ps1
```

## Related

- Prior Ally X app (MediaPipe pipeline): [face-swap-video-windows](https://github.com/vanukrishnans-source/face-swap-video-windows) — **kept**; this repo does not delete those releases.
