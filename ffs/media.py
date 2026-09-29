"""Video I/O: OpenCV (its bundled LGPL FFmpeg) for decoding — the same decoder the reference used, so decoded
frames are identical — and the bundled LGPL FFmpeg executable for H.264 encoding + audio passthrough.

LGPL FFmpeg has no libx264, so the encoder is picked at run time from:
  h264_amf (AMD VCN hardware encoder on the Radeon 780M) -> h264_nvenc / h264_qsv (other PCs)
  -> h264_mf (Windows Media Foundation) -> libopenh264 (Cisco OpenH264, software, always present).
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

log = logging.getLogger("ffs")

import cv2
import numpy as np

NO_WINDOW = 0x08000000 if os.name == "nt" else 0
MP4_AUDIO_OK = {"aac", "mp3", "alac", "opus", "flac", "ac3", "eac3", "mp2"}


def _bin(name):
    exe = name + (".exe" if os.name == "nt" else "")
    env = os.environ.get("FFS_" + name.upper())
    if env and Path(env).is_file():
        return env
    roots = []
    if getattr(sys, "frozen", False):
        roots.append(Path(sys.executable).resolve().parent)
    if getattr(sys, "_MEIPASS", None):
        roots.append(Path(sys._MEIPASS))
    roots.append(Path(__file__).resolve().parent.parent)
    for r in roots:
        for c in (r / "ffmpeg" / exe, r / "ffmpeg" / "bin" / exe):
            if c.is_file():
                return str(c)
    w = shutil.which(exe)
    if w:
        return w
    raise FileNotFoundError(f"{exe} not found (expected in the app's ffmpeg folder)")


def ffmpeg():
    return _bin("ffmpeg")


def ffprobe():
    return _bin("ffprobe")


def run(cmd, timeout=None, check=False):
    try:
        return subprocess.run(cmd, capture_output=True, timeout=timeout, creationflags=NO_WINDOW, check=check)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"FFmpeg timed out after {timeout}s ({' '.join(str(x) for x in cmd[:4])}…). "
            "The Final step was stopped so the app would not hang. Try a shorter clip or CPU encode."
        ) from e


@dataclass
class VideoInfo:
    path: str
    width: int           # upright (after rotation) as decoded by OpenCV
    height: int
    fps: float           # OpenCV CAP_PROP_FPS (what the reference uses)
    duration: float
    frames: int
    vcodec: str
    has_audio: bool
    acodec: str
    color_space: str
    color_primaries: str
    color_trc: str
    hdr: bool
    rotation: int

    def summary(self):
        a = f"sound: {self.acodec}" if self.has_audio else "no sound"
        return f"{self.width}×{self.height} · {self.fps:.3g} fps · {fmt_time(self.duration)} · {a}"


def fmt_time(s):
    s = max(0.0, float(s)); m = int(s // 60)
    return f"{m}:{s - 60 * m:04.1f}"


def probe(path) -> VideoInfo:
    cp = run([ffprobe(), "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)], timeout=60)
    if cp.returncode != 0:
        raise ValueError("Couldn't read that video: " + cp.stderr.decode("utf-8", "replace").strip()[-300:])
    meta = json.loads(cp.stdout.decode("utf-8", "replace") or "{}")
    streams = meta.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video" and not (s.get("disposition") or {}).get("attached_pic")), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if v is None:
        raise ValueError("That file has no video track.")
    rot = 0
    for sd in v.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(round(float(sd["rotation"])))
    if "rotate" in (v.get("tags") or {}):
        rot = int(v["tags"]["rotate"])
    cap = open_capture(path)
    if not cap.isOpened():
        raise ValueError("This video codec isn't supported by the decoder.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    ok, fr = cap.read()
    cap.release()
    if not ok:
        raise ValueError("Couldn't decode the first frame of that video.")
    h, w = fr.shape[:2]
    if fps <= 0:
        num, _, den = (v.get("avg_frame_rate") or "30/1").partition("/")
        fps = float(num) / float(den or 1)
    dur = float(meta.get("format", {}).get("duration") or v.get("duration") or (n / fps if fps else 0))
    trc = v.get("color_transfer", "") or ""
    return VideoInfo(str(path), w, h, float(fps), dur, n, v.get("codec_name", "?"), a is not None,
                     (a or {}).get("codec_name", ""), v.get("color_space", "") or "",
                     v.get("color_primaries", "") or "", trc, trc in ("smpte2084", "arib-std-b67"), rot)


_ascii_alias = {}


def open_capture(path):
    path = str(path)
    cap = cv2.VideoCapture(_ascii_alias.get(path, path), cv2.CAP_FFMPEG)
    if cap.isOpened() or path.isascii():
        return cap
    # Non-ASCII path that this OpenCV build can't open: try the 8.3 short name, then an ASCII hard link / copy.
    alias = None
    if os.name == "nt":
        try:
            import ctypes
            buf = ctypes.create_unicode_buffer(1024)
            if ctypes.windll.kernel32.GetShortPathNameW(path, buf, 1024) and buf.value.isascii():
                alias = buf.value
        except Exception:  # noqa: BLE001
            pass
    if alias is None:
        d = Path(tempfile.gettempdir()) / "FaceFusionStudio"; d.mkdir(parents=True, exist_ok=True)
        alias = str(d / f"input_{abs(hash(path)) & 0xFFFFFFFF:08x}{Path(path).suffix.lower()}")
        if not os.path.exists(alias):
            try:
                os.link(path, alias)
            except OSError:
                shutil.copyfile(path, alias)
    _ascii_alias[path] = alias
    return cv2.VideoCapture(alias, cv2.CAP_FFMPEG)


def read_frame_at(path, t):
    cap = open_capture(path)
    if t > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
    ok, fr = cap.read()
    cap.release()
    return fr if ok else None


def iter_frames(path, start, end, fps, cancel=None, sequential=False):
    """Yield (k, t, frame_bgr) for the frames selected by the slot rule ('first frame at/after each 1/fps
    slot') using real timestamps. Seeks close to `start` for long trims (sequential=True disables it, which
    is what the reference does)."""
    from .vcore import SlotSelector
    cap = open_capture(path)
    sel = SlotSelector(start, end, fps)
    pending = False                       # a frame is already grabbed (after a seek check)
    if start > 2.0 and not sequential:
        cap.set(cv2.CAP_PROP_POS_MSEC, (start - 1.0) * 1000.0)
        if cap.grab() and cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 <= start + 1e-6:
            pending = True
        else:                             # seek overshot or failed: read from the beginning
            cap.release(); cap = open_capture(path)
    k = 0
    try:
        while True:
            if cancel is not None and cancel():
                return
            if pending:
                pending = False
            elif not cap.grab():
                break
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            r = sel.accept(t)
            if r == 2: break
            if r == 1:
                ok, fr = cap.retrieve()
                if not ok: break
                yield k, t, fr
                k += 1
    finally:
        cap.release()


def count_selected(info: VideoInfo, start, end, fps):
    from .vcore import frame_times
    return len(frame_times(info.duration if not info.frames else info.frames / info.fps, info.fps, start, end, fps))


# ---------------------------------------------------------------- encoding
ENCODER_ORDER = ["h264_amf", "h264_nvenc", "h264_qsv", "h264_mf", "libopenh264", "libx264"]
_encoder_cache = {}


def encoder_args(name, W, H, fps):
    bpf = 0.15 if name != "libopenh264" else 0.2        # bits per pixel per frame
    br = int(max(1.0e6, bpf * W * H * fps))
    g = str(int(round(fps * 2)))
    if name == "h264_amf":
        return ["-c:v", name, "-usage", "transcoding", "-quality", "quality", "-rc", "vbr_peak",
                "-b:v", str(br), "-maxrate", str(int(br * 1.5)), "-profile:v", "high", "-g", g]
    if name == "h264_nvenc":
        return ["-c:v", name, "-preset", "p5", "-rc", "vbr", "-b:v", str(br), "-maxrate", str(int(br * 1.5)), "-g", g]
    if name == "h264_qsv":
        return ["-c:v", name, "-preset", "slow", "-b:v", str(br), "-g", g]
    if name == "h264_mf":
        return ["-c:v", name, "-rate_control", "pc_vbr", "-b:v", str(br), "-hw_encoding", "1", "-g", g]
    if name == "libopenh264":
        return ["-c:v", name, "-rc_mode", "bitrate", "-b:v", str(br), "-profile:v", "high", "-g", g]
    return ["-c:v", "libx264", "-preset", "medium", "-crf", "20", "-g", g]   # dev boxes with GPL ffmpeg only


def pick_encoder(prefer=None):
    """First H.264 encoder that actually works on this machine (1-frame test encode)."""
    key = prefer or "auto"
    if key in _encoder_cache:
        return _encoder_cache[key]
    names = run([ffmpeg(), "-hide_banner", "-encoders"], timeout=30).stdout.decode("utf-8", "replace")
    order = ([prefer] if prefer else []) + [e for e in ENCODER_ORDER if e != prefer]
    tried = []
    for e in order:
        if f" {e} " not in names:
            continue
        cmd = [ffmpeg(), "-v", "error", "-f", "lavfi", "-i", "color=c=gray:s=320x240:r=30", "-frames:v", "5",
               "-pix_fmt", "yuv420p", *encoder_args(e, 320, 240, 30), "-f", "null", "-"]
        try:
            cp = run(cmd, timeout=30)
            ok = cp.returncode == 0
        except subprocess.TimeoutExpired:
            ok = False
        tried.append((e, ok))
        if ok:
            _encoder_cache[key] = e
            return e
    raise RuntimeError(f"No working H.264 encoder in the bundled FFmpeg (tried {tried})")


def color_args(info: VideoInfo):
    """OpenCV decodes with the stream's own matrix (BT.709 for HD, BT.601 if untagged); encode with the same
    matrix and copy the tags so untouched pixels keep their exact colours."""
    cs = info.color_space if info.color_space in ("bt709", "bt470bg", "smpte170m", "bt2020nc") else ""
    matrix = {"bt709": "bt709", "bt470bg": "bt601", "smpte170m": "bt601", "bt2020nc": "bt2020"}.get(cs, "bt601")
    vf = f"scale=out_color_matrix={matrix}:out_range=tv:flags=bicubic,format=yuv420p"
    tags = ["-colorspace", cs or "smpte170m", "-color_range", "tv"]
    if info.color_primaries and info.color_primaries != "unknown":
        tags += ["-color_primaries", info.color_primaries]
    if info.color_trc and info.color_trc != "unknown":
        tags += ["-color_trc", info.color_trc]
    return vf, tags


class Encoder:
    def __init__(self, out_path, W, H, fps, info: VideoInfo, encoder=None, finalize_timeout=None):
        if W < 2 or H < 2 or not fps or fps <= 0:
            raise ValueError(f"Cannot encode {W}×{H} @ {fps} fps.")
        self.encoder = encoder or pick_encoder()
        self.out_path = out_path
        self.W, self.H, self.fps = int(W), int(H), float(fps)
        self.frame_bytes = self.W * self.H * 3
        try:
            default_to = float(os.environ.get("FFS_ENCODE_TIMEOUT", "300") or 300)
        except ValueError:
            default_to = 300.0
        self.finalize_timeout = max(30.0, float(finalize_timeout if finalize_timeout is not None else default_to))
        vf, tags = color_args(info)
        rate = Fraction(fps).limit_denominator(1001000)
        cmd = [ffmpeg(), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{self.W}x{self.H}",
               "-r", f"{rate.numerator}/{rate.denominator}", "-i", "-", "-vf", vf,
               *encoder_args(self.encoder, self.W, self.H, self.fps), *tags, "-an", "-movflags", "+faststart",
               str(out_path)]
        self.err = tempfile.TemporaryFile()
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.err,
                                     creationflags=NO_WINDOW)
        self.frames = 0
        self._closed = False

    def write(self, frame):
        if self._closed:
            raise RuntimeError("Encoder already closed.")
        arr = np.ascontiguousarray(frame)
        if arr.ndim != 3 or arr.shape[0] != self.H or arr.shape[1] != self.W or arr.shape[2] != 3:
            raise RuntimeError(f"Encoder expected {self.W}×{self.H}×3 frames, got {getattr(arr, 'shape', None)}.")
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        try:
            self.proc.stdin.write(arr.tobytes())
        except (BrokenPipeError, OSError) as e:
            raise RuntimeError("The video encoder stopped: " + self._err()) from e
        self.frames += 1

    def _err(self):
        try:
            self.err.seek(0)
            return self.err.read().decode("utf-8", "replace")[-400:]
        except Exception:  # noqa: BLE001
            return ""

    def close(self, progress=None, cancel=None):
        """Flush stdin and wait for FFmpeg. Never blocks forever — kills on timeout / cancel."""
        if self._closed:
            return
        self._closed = True
        if progress:
            progress(dict(stage="mux", done=0, total=1, detail="Closing the video encoder…"))
        # Closing a large pipe can block if FFmpeg is stuck; do it on a helper thread.
        close_err = []

        def _close_stdin():
            try:
                if self.proc.stdin:
                    self.proc.stdin.close()
            except OSError as e:
                close_err.append(e)

        t = threading.Thread(target=_close_stdin, name="ffs-enc-stdin", daemon=True)
        t.start()
        deadline = time.perf_counter() + self.finalize_timeout
        while t.is_alive():
            if cancel and cancel():
                self.kill()
                raise RuntimeError("cancelled")
            left = deadline - time.perf_counter()
            if left <= 0:
                self.kill()
                raise RuntimeError(
                    f"Encoder {self.encoder} hung while finishing the MP4 (stdin close > "
                    f"{self.finalize_timeout:.0f}s). Try another encoder or a shorter clip. {self._err()}"
                )
            t.join(timeout=min(0.5, left))
            if progress:
                progress(dict(stage="mux", done=0, total=1, detail="Finishing encode…"))
        if close_err and self.proc.poll() is None:
            log.warning("stdin close: %s", close_err[0])
        # Wait for the process itself (mp4 remux / faststart).
        while self.proc.poll() is None:
            if cancel and cancel():
                self.kill()
                raise RuntimeError("cancelled")
            left = deadline - time.perf_counter()
            if left <= 0:
                self.kill()
                raise RuntimeError(
                    f"Encoder {self.encoder} timed out after {self.finalize_timeout:.0f}s while writing "
                    f"the MP4. {self._err()}"
                )
            try:
                self.proc.wait(timeout=min(0.5, left))
            except subprocess.TimeoutExpired:
                if progress:
                    progress(dict(stage="mux", done=0, total=1, detail="Waiting for FFmpeg to finish…"))
        rc = self.proc.returncode
        if rc != 0:
            raise RuntimeError(f"Encoder {self.encoder} failed ({rc}): {self._err()}")
        if progress:
            progress(dict(stage="mux", done=1, total=2, detail="Video stream ready"))
        try:
            self.err.close()
        except Exception:  # noqa: BLE001
            pass

    def kill(self):
        try:
            if self.proc.stdin:
                try:
                    self.proc.stdin.close()
                except OSError:
                    pass
            self.proc.kill()
            self.proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            pass
        self._closed = True


def mux_timeout(length_s, floor=60.0):
    try:
        env = float(os.environ.get("FFS_MUX_TIMEOUT", "0") or 0)
    except ValueError:
        env = 0.0
    if env > 0:
        return max(30.0, env)
    # Copying audio is usually seconds; allow plenty for long clips / busy disks, never infinite.
    return max(floor, min(600.0, 45.0 + float(length_s) * 2.0))


def mux_audio(video_only, source, start, length, out_path, info: VideoInfo, progress=None, cancel=None):
    """Copy the trimmed original audio next to the new video. Returns a note for the UI.

    Always finishes or raises within `mux_timeout` — never hangs on a stuck FFmpeg remux.
    """
    out_path = Path(out_path)
    video_only = Path(video_only)
    if not video_only.is_file() or video_only.stat().st_size < 64:
        raise RuntimeError("Temporary video is missing or empty — encode did not produce a file.")
    # Refuse to overwrite a destination that looks locked by another process (Windows Explorer preview, etc.).
    if out_path.exists():
        try:
            with open(out_path, "a+b"):
                pass
        except OSError as e:
            alt = out_path.with_name(out_path.stem + "_new" + out_path.suffix)
            log.warning("output locked (%s) — writing to %s", e, alt.name)
            out_path = alt
            if progress:
                progress(dict(stage="mux", done=1, total=2, detail=f"Output was locked; saving as {alt.name}"))
    if not info.has_audio:
        if progress:
            progress(dict(stage="mux", done=1, total=1, detail="No sound in the source — saving video"))
        shutil.move(str(video_only), str(out_path))
        return "no sound in the source", out_path
    attempts = []
    if info.acodec in MP4_AUDIO_OK:
        attempts.append((["-c:a", "copy"], f"original sound copied ({info.acodec})"))
    attempts.append((["-c:a", "aac", "-b:a", "192k"], f"sound converted {info.acodec} → AAC"))
    to = mux_timeout(length)
    for i, (aargs, note) in enumerate(attempts):
        if cancel and cancel():
            raise RuntimeError("cancelled")
        if progress:
            progress(dict(stage="mux", done=1, total=2, detail=f"Muxing sound ({i + 1}/{len(attempts)})…"))
        # -ss on the audio input after -i would re-decode; keep our previous filter placement but add -vn
        # avoidance and a hard timeout. Prefer a quiet unique temp then rename so a half-written file
        # is never left as the user's final path.
        tmp_out = out_path.with_name(out_path.stem + f".mux{i}.tmp.mp4")
        cmd = [ffmpeg(), "-v", "error", "-y",
               "-i", str(video_only),
               "-ss", f"{start:.6f}", "-t", f"{length:.6f}", "-i", str(source),
               "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", *aargs, "-shortest",
               "-movflags", "+faststart", str(tmp_out)]
        try:
            cp = run(cmd, timeout=to)
        except RuntimeError as e:
            log.warning("mux attempt %s timed out/failed: %s", i, e)
            try:
                tmp_out.unlink(missing_ok=True)
            except OSError:
                pass
            continue
        if cp.returncode == 0 and tmp_out.is_file() and tmp_out.stat().st_size > 64:
            try:
                os.replace(tmp_out, out_path)
            except OSError:
                shutil.move(str(tmp_out), str(out_path))
            try:
                video_only.unlink(missing_ok=True)
            except OSError:
                pass
            if progress:
                progress(dict(stage="mux", done=2, total=2, detail=note))
            return note, out_path
        log.warning("mux attempt %s rc=%s stderr=%s", i, cp.returncode,
                    (cp.stderr or b"").decode("utf-8", "replace")[-200:])
        try:
            tmp_out.unlink(missing_ok=True)
        except OSError:
            pass
    if progress:
        progress(dict(stage="mux", done=2, total=2, detail="Saving without sound"))
    shutil.move(str(video_only), str(out_path))
    return "sound could not be copied (saved without sound)", out_path


def probe_output(path):
    cp = run([ffprobe(), "-v", "error", "-show_entries",
              "stream=codec_type,codec_name,width,height,duration,nb_frames,r_frame_rate:format=duration,size",
              "-of", "json", str(path)], timeout=60)
    return json.loads(cp.stdout.decode("utf-8", "replace") or "{}")
