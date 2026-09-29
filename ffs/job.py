"""Two-pass video face-swap job (same algorithm as the Android app / Python reference), tuned for the Ally X.

Pass 1: decode -> YOLO Face detection per selected frame -> tracking,
        forward/backward One-Euro smoothing, left-to-right pairing (cached: Flip only re-runs pass 2).
Pass 2: per frame swap (+ enhancer) in worker threads (model calls serialised on the GPU, pre/post-processing
        overlapped), frames written in order to the H.264 encoder; then the trimmed original audio is muxed in.
"""
from __future__ import annotations

import collections
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from . import core, detect, media, vcore
from .engine import Engine
from .models import ENHANCER_HQ, ENHANCER_LIGHT, GFPGAN, ModelStore

log = logging.getLogger("ffs")

MAX_CLIP_S = 300.0        # 5 minutes — see README "Limits" for the reasoning
MAX_FPS = 60.0
ENHANCE_SPECS = {"gpen256": ENHANCER_LIGHT, "gpen512": ENHANCER_HQ}
ENHANCE_LABEL = {None: "Off", "gpen256": "Light (GPEN 256)", "gpen512": "HQ (GPEN 512)", "gfpgan": "GFPGAN 1.4"}


class Cancelled(Exception):
    pass


@dataclass
class Settings:
    start: float = 0.0
    length: float = 10.0
    fps: float = 30.0            # 0 = original (capped at 60)
    max_short: int = 1080        # short-side cap
    align: int = 2
    enhance: Optional[str] = "gpen256"
    rotation: int = 0
    device: str = "auto"         # auto | dml | cpu
    out_dir: str = ""
    sequential_decode: bool = False
    # --- v2 advanced ---
    min_confidence: float = 0.62
    min_face_frac: float = 0.035
    same_gender: bool = True
    color_match: bool = True
    seamless: bool = False       # Poisson blend (slower; good hairline)
    temporal_smooth: float = 0.18  # EMA on output frames 0..0.45
    emb_track: bool = True       # ArcFace ID affinity in tracker
    detector: str = "yolo"      # yolo | retina

    def effective_fps(self, src_fps):
        f = self.fps if self.fps and self.fps > 0 else min(src_fps, MAX_FPS)
        return min(f, src_fps)

    def detect_opts(self):
        from .detect import DetectOpts
        return DetectOpts(min_confidence=self.min_confidence, min_face_frac=self.min_face_frac, detector=self.detector)


@dataclass
class Analysis:
    key: tuple
    W: int
    H: int
    s: float
    fps: float
    times: list
    dets: list
    tracks: list
    smooth: list
    n_src: int
    detect_s: float


@dataclass
class PhotoFaces:
    path: str
    img: np.ndarray
    faces: list                  # 478-pt arrays, left -> right
    hits: list = field(default_factory=list)   # FaceHit with confidence/gender
    genders: list = field(default_factory=list)


def default_out_dir() -> Path:
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes
            import uuid
            fid = uuid.UUID("{18989B1D-99B5-455B-841C-AB7C74E4DDFC}")      # FOLDERID_Videos
            guid = (ctypes.c_byte * 16).from_buffer_copy(fid.bytes_le)
            p = ctypes.c_wchar_p()
            if ctypes.windll.shell32.SHGetKnownFolderPath(ctypes.byref(guid), 0, None, ctypes.byref(p)) == 0:
                base = Path(p.value); ctypes.windll.ole32.CoTaskMemFree(p)
                return base / "FaceFusion"
        except Exception:  # noqa: BLE001
            pass
    return Path.home() / "Videos" / "FaceFusion"


def cpu_workers():
    n = os.cpu_count() or 4
    det = int(os.environ.get("FFS_DET_THREADS", "0") or 0) or max(1, min(4, n // 4))
    work = int(os.environ.get("FFS_WORKERS", "0") or 0) or max(1, min(4, n // 4))
    return det, work


def load_photo(path, opts=None, with_gender=True, engine=None, store=None, device="cpu") -> PhotoFaces:
    """Detect faces in a source photo. Creates a temporary CPU Engine if none given."""
    from .engine import Engine
    from .models import ModelStore, REQUIRED
    img = detect.load_image(path)
    eng = engine
    owned = False
    if eng is None:
        st = store or ModelStore()
        eng = Engine(st, device)
        eng.prepare(None, detector=(opts.detector if opts else "yolo"))
        detect.set_default_engine(eng, "yoloface_8n" if (not opts or opts.detector != "retina") else "retinaface_10g")
        owned = True
    try:
        hits = detect.detect_photo(img, opts, engine=eng)
        if with_gender and hits:
            try:
                from . import gender as G
                G.annotate_hits(img, hits)
            except Exception:  # noqa: BLE001
                pass
        faces = [h.kps.copy() for h in hits]
        genders = [h.gender for h in hits]
        return PhotoFaces(str(path), img, faces, hits=hits, genders=genders)
    finally:
        if owned:
            eng.close()


class Job:
    """Holds the engine (sessions stay open between preview / runs) and the pass-1 cache."""

    def __init__(self, store: ModelStore, device="auto"):
        self.store = store
        self.device = device
        self.engine: Optional[Engine] = None
        self.analysis: Optional[Analysis] = None
        self.bench: dict = {}
        self.overhead: dict = {}

    # ------------------------------------------------------------ engine
    def get_engine(self, device=None) -> Engine:
        device = device or self.device
        if self.engine is None or self.engine.device != device:
            if self.engine: self.engine.close()
            self.engine = Engine(self.store, device)
            self.device = device
        return self.engine

    def benchmark(self, device=None):
        eng = self.get_engine(device)
        eng.prepare(None)
        self.bench = eng.benchmark()
        return self.bench

    def recommended_enhance(self):
        """HQ when the GPU makes it cheap, else Light, else Off (only modes whose model is downloaded)."""
        have = {m for m, s in ENHANCE_SPECS.items() if self.store.is_installed(s)}
        b = self.bench
        if "gpen512" in have and "gpen512" in b and b["swap"] + b["gpen512"] <= 0.35:
            return "gpen512"
        if "gpen256" in have:
            return "gpen256"
        return "gpen512" if "gpen512" in have else None

    def estimate(self, n_frames, n_faces, enhance, W=1920, H=1080):
        """Rough seconds for the whole job (detection + swap), from the micro-benchmark."""
        _, work = cpu_workers()
        b = self.bench or {"swap": 0.7, "gpen256": 0.15, "gpen512": 1.3}
        model = b.get("swap", 0.7) + (b.get(enhance, 0.0) if enhance else 0.0)
        scale = max(0.4, (W * H) / (1920 * 1080))
        over = self.overhead.get(enhance, (0.025 + {None: 0, "gpen256": 0.02, "gpen512": 0.06}[enhance]) * scale)
        per_frame = max(model * n_faces, (model + over) * n_faces / work)
        det = n_frames * 0.03 * scale / cpu_workers()[0]
        return det + n_frames * per_frame

    # ------------------------------------------------------------ pass 1
    def analyze(self, info: media.VideoInfo, photo: PhotoFaces, st: Settings, progress=None, cancel=None) -> Analysis:
        eng0 = self.get_engine(st.device)
        eng0.prepare(None, detector=st.detector)
        detect.set_default_engine(eng0, "yoloface_8n" if st.detector != "retina" else "retinaface_10g")
        fps = st.effective_fps(info.fps)
        end = min(st.start + min(st.length, MAX_CLIP_S), info.duration)
        W, H, s = vcore.out_size(info.width, info.height, st.max_short, st.align)
        n_src = len(photo.faces)
        key = (info.path, os.path.getmtime(info.path), round(st.start, 4), round(end, 4), fps, W, H, min(n_src, 2),
               round(st.min_confidence, 3), round(st.min_face_frac, 4), st.emb_track, st.same_gender)
        if self.analysis is not None and self.analysis.key == key:
            return self.analysis
        total = max(1, media.count_selected(info, st.start, end, fps))
        n_det, _ = cpu_workers()
        expected = min(n_src, 2)
        dopts = st.detect_opts()
        t0 = time.perf_counter()
        times, dets = [], []
        # DirectML (and our Engine) only run inference on one thread. Keep a small pool for the
        # CPU-only prep (resize/crop) so decode stays overlapped, then detect serially.
        prep_n = max(1, min(n_det, 2))
        inflight = collections.deque()

        def prep_job(fr):
            return vcore.prep(fr, W, H, s)

        with ThreadPoolExecutor(prep_n, thread_name_prefix="detect-prep") as pool:
            for k, t, fr in media.iter_frames(info.path, st.start, end, fps, cancel, st.sequential_decode):
                times.append(t)
                inflight.append(pool.submit(prep_job, fr))
                while len(inflight) > prep_n + 1:
                    frame = inflight.popleft().result()
                    dets.append(detect.detect_frame(frame, expected, dopts, engine=eng0))
                    self._tick(progress, "detect", len(dets), total, t0)
                if cancel and cancel(): raise Cancelled()
            while inflight:
                frame = inflight.popleft().result()
                dets.append(detect.detect_frame(frame, expected, dopts, engine=eng0))
                self._tick(progress, "detect", len(dets), total, t0)
        if cancel and cancel(): raise Cancelled()
        if not dets:
            raise ValueError("No frames in the selected range.")
        embeddings = None
        if st.emb_track:
            # Sparse ArcFace IDs every ~0.5 s for track affinity (keeps identity stable)
            try:
                eng = self.get_engine(st.device)
                eng.prepare(None)
                embeddings = self._sparse_embeddings(eng, info, an_times=times, dets=dets,
                                                     W=W, H=H, s=s, start=st.start, end=end,
                                                     fps=fps, seq=st.sequential_decode, cancel=cancel)
            except Exception as e:  # noqa: BLE001
                log.warning("embedding track disabled: %s", e); embeddings = None
        tracks = vcore.track(dets, max_gap=int(round(fps)), embeddings=embeddings,
                             emb_weight=0.45 if embeddings else 0.0)
        smooth = [vcore.smooth_track(t, fps) for t in tracks]
        self.analysis = Analysis(key, W, H, s, fps, times, dets, tracks, smooth, n_src, time.perf_counter() - t0)
        return self.analysis

    def _sparse_embeddings(self, eng, info, an_times, dets, W, H, s, start, end, fps, seq, cancel):
        """Compute ArcFace embeddings on a subset of frames; propagate None elsewhere."""
        n = len(dets)
        step = max(1, int(round(fps * 0.5)))
        embs = [[None] * len(dets[i]) for i in range(n)]
        # Re-read only the needed frames is expensive; reuse decode by sampling indices we already have
        # We don't keep frames — re-decode sparse subset.
        want = set(range(0, n, step)) | {n - 1}
        for k, t, fr in media.iter_frames(info.path, start, end, fps, cancel, seq):
            if k >= n: break
            if k not in want: continue
            frame = vcore.prep(fr, W, H, s)
            for j, pts in enumerate(dets[k]):
                try:
                    embs[k][j] = core.embedding(eng, frame, core.kps5(pts))
                except Exception:  # noqa: BLE001
                    embs[k][j] = None
            if cancel and cancel(): raise Cancelled()
        # Leave non-sampled frames as None — tracker falls back to IoU there.
        return embs

    @staticmethod
    def _tick(progress, stage, done, total, t0, extra=None):
        if progress is None: return
        el = time.perf_counter() - t0
        rate = done / el if el > 0 else 0
        eta = (total - done) / rate if rate > 0 else None
        d = dict(stage=stage, done=done, total=total, rate=rate, eta=eta)
        if extra: d.update(extra)
        progress(d)

    # ------------------------------------------------------------ preview (single frame, no tracking)
    def preview(self, info, photo: PhotoFaces, st: Settings, t=None):
        eng = self.get_engine(st.device)
        eng.prepare(st.enhance, detector=st.detector)
        detect.set_default_engine(eng, "yoloface_8n" if st.detector != "retina" else "retinaface_10g")
        W, H, s = vcore.out_size(info.width, info.height, st.max_short, st.align)
        t = st.start if t is None else t
        fr = media.read_frame_at(info.path, t)
        if fr is None: raise ValueError("Couldn't read that part of the video.")
        frame = vcore.prep(fr, W, H, s)
        dets = detect.detect_frame(frame, min(len(photo.faces), 2), st.detect_opts(), engine=eng)
        assign = vcore.pair_single_frame(dets, len(photo.faces), st.rotation)
        if st.same_gender and photo.genders:
            try:
                from . import gender as G
                hits = detect.detect_frame_hits(frame, min(len(photo.faces), 2), st.detect_opts())
                G.annotate_hits(frame, hits)
                # rebuild assign with gender preference on this single frame
                tg = [h.gender for h in hits]
                # map dets order ≈ hits order
                if len(tg) == len(dets):
                    assign = _pair_single_gender(dets, photo.faces, st.rotation, tg, photo.genders, st.same_gender)
            except Exception:  # noqa: BLE001
                pass
        lat = self._latents(eng, photo, assign)
        faces = [(dets[i].astype(np.float32), lat[a]) for i, a in enumerate(assign) if a >= 0]
        t0 = time.perf_counter()
        out = core.process_frame(eng, frame, faces, st.enhance,
                                 color_match=st.color_match, seamless=st.seamless)
        el = time.perf_counter() - t0
        if faces and self.bench:
            model = self.bench.get("swap", 0) + (self.bench.get(st.enhance, 0) if st.enhance else 0)
            self.overhead[st.enhance] = max(0.0, el / len(faces) - model)
        return frame, out, dets, assign

    @staticmethod
    def _latents(eng, photo, assign):
        lat = {}
        for si in sorted(set(a for a in assign if a >= 0)):
            emb = core.embedding(eng, photo.img, core.kps5(photo.faces[si]))
            lat[si] = core.latent_for(eng, emb)
        return lat

    # ------------------------------------------------------------ full job
    def run(self, info: media.VideoInfo, photo: PhotoFaces, st: Settings, out_path=None, progress=None,
            cancel=None, dump_dir=None):
        if not photo.faces: raise ValueError("No face found in the faces photo.")
        if st.enhance and not self.store.is_installed(ENHANCE_SPECS[st.enhance]):
            raise ValueError(f"The {ENHANCE_LABEL[st.enhance]} enhancer isn't downloaded yet.")
        t_start = time.perf_counter()
        an = self.analyze(info, photo, st, progress, cancel)
        assign, pf = vcore.pair_tracks(an.tracks, an.n_src, st.rotation)
        if pf < 0: raise ValueError("No faces found in the selected part of the video.")
        if st.same_gender and photo.genders:
            try:
                track_genders = self._track_genders(info, an, st, pf)
                assign, pf = vcore.pair_tracks_gender(
                    an.tracks, an.n_src, st.rotation, track_genders, photo.genders, True)
            except Exception as e:  # noqa: BLE001
                log.warning("gender pairing skipped: %s", e)
        eng = self.get_engine(st.device)
        dev = eng.prepare(st.enhance, detector=st.detector)
        detect.set_default_engine(eng, "yoloface_8n" if st.detector != "retina" else "retinaface_10g")
        lat = self._latents(eng, photo, assign)
        fb468 = [np.asarray(f, np.float32) for f in photo.faces]

        out_dir = Path(st.out_dir) if st.out_dir else default_out_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        if out_path is None:
            stem = "".join(c for c in Path(info.path).stem if c.isalnum() or c in "-_ ")[:40].strip() or "video"
            out_path = out_dir / f"FaceFusion_{stem}_{time.strftime('%Y%m%d-%H%M%S')}.mp4"
        out_path = Path(out_path)
        tmp_video = out_path.with_name(out_path.stem + ".video.tmp.mp4")
        n = len(an.times)
        enc = media.Encoder(tmp_video, an.W, an.H, an.fps, info)
        _, n_work = cpu_workers()
        dump_idx = {0, 1, 2, n // 2, n - 1} if dump_dir else set()
        if dump_dir: os.makedirs(dump_dir, exist_ok=True)
        t0 = time.perf_counter()
        last_thumb = [0.0]
        # Inference is already serialised on Engine's ORT thread. Do not fan out many concurrent
        # process_frame calls (they all queue on the same lock and only inflate memory). Keep a
        # tiny overlap for CPU prep of the next frame while the current one is swapping.
        ok = False
        try:
            written = 0
            prev_out = [None]
            end_t = an.times[-1] + 1e-3 if an.times else st.start + 1e-3
            for k, t, fr in media.iter_frames(info.path, st.start, end_t, an.fps, cancel, st.sequential_decode):
                if k >= n: break
                if cancel and cancel(): raise Cancelled()
                frame = vcore.prep(fr, an.W, an.H, an.s)
                if frame is None or getattr(frame, "size", 0) == 0:
                    raise RuntimeError(f"Could not decode frame {k} of the selected range.")
                faces = [(an.smooth[ti][k].astype(np.float32), lat[a]) for ti, a in enumerate(assign)
                         if a >= 0 and k in an.smooth[ti]]
                out = core.process_frame(eng, frame, faces, st.enhance,
                                         color_match=st.color_match, seamless=st.seamless)
                if st.temporal_smooth and prev_out[0] is not None and prev_out[0].shape == out.shape:
                    a = float(np.clip(st.temporal_smooth, 0.0, 0.45))
                    out = np.clip((1.0 - a) * out.astype(np.float32) + a * prev_out[0].astype(np.float32),
                                  0, 255).astype(np.uint8)
                prev_out[0] = out
                enc.write(out)
                if k in dump_idx:
                    np.save(f"{dump_dir}/frame{k}_in_rgb.npy", frame[:, :, ::-1].copy())
                    np.save(f"{dump_dir}/frame{k}_out_rgb.npy", out[:, :, ::-1].copy())
                written += 1
                extra = {}
                now = time.perf_counter()
                if now - last_thumb[0] > 0.5 or written == n:
                    last_thumb[0] = now; extra = {"thumb": out, "before": frame}
                self._tick(progress, "swap", written, n, t0, extra)
            # Final / Finalize stage — always completes or raises a clear timeout error.
            if progress:
                progress(dict(stage="mux", done=0, total=2, detail="Finishing MP4 (encode)…"))
            enc.close(progress=progress, cancel=cancel)
            if written == 0:
                raise RuntimeError("No frames were written — nothing to save.")
            if written != n:
                log.warning("wrote %s frames, expected %s — continuing with what we have", written, n)
            length = written / an.fps if an.fps else 0.0
            if progress:
                progress(dict(stage="mux", done=1, total=2, detail="Saving MP4 and copying the sound…"))
            note, out_path = media.mux_audio(tmp_video, info.path, st.start, length, out_path, info,
                                             progress=progress, cancel=cancel)
            n = written  # report the frames we actually produced
            ok = True
        except Cancelled:
            enc.kill()
            raise
        except RuntimeError as e:
            enc.kill()
            if str(e) == "cancelled" or (cancel and cancel()):
                raise Cancelled() from e
            raise
        except BaseException:
            enc.kill()
            raise
        finally:
            if not ok:
                for p in (tmp_video, out_path):
                    try: Path(p).unlink(missing_ok=True)
                    except OSError: pass
        swap_s = time.perf_counter() - t0
        res = dict(path=str(out_path), frames=n, fps=an.fps, W=an.W, H=an.H, duration=n / an.fps,
                   encoder=enc.encoder, audio=note, device=dev.label(), device_active=dev.active,
                   per_model=dict(dev.per_model), enhance=ENHANCE_LABEL[st.enhance], tracks=len(an.tracks),
                   min_confidence=st.min_confidence, same_gender=st.same_gender,
                   color_match=st.color_match, temporal_smooth=st.temporal_smooth,
                   src_genders=list(photo.genders),
                   track_lengths=[len(t) for t in an.tracks], assign=assign, pairing_frame=pf,
                   detect_s=round(an.detect_s, 2), swap_s=round(swap_s, 2),
                   total_s=round(time.perf_counter() - t_start, 2), size=out_path.stat().st_size)
        if dump_dir:
            self._dump(dump_dir, info, an, assign, pf, photo, st)
        if progress: progress(dict(stage="done", done=n, total=n, result=res))
        return res


    def run_photo(self, target_path, source_photo: PhotoFaces, st: Settings, out_path=None, progress=None, cancel=None):
        """Single-image face swap → Pictures/FaceFusion."""
        if not source_photo.faces:
            raise ValueError("No face found in the source faces photo.")
        if st.enhance and st.enhance in ENHANCE_SPECS and not self.store.is_installed(ENHANCE_SPECS[st.enhance]):
            raise ValueError(f"The {ENHANCE_LABEL.get(st.enhance, st.enhance)} enhancer isn't downloaded yet.")
        t0 = time.perf_counter()
        eng = self.get_engine(st.device)
        dev = eng.prepare(st.enhance, detector=st.detector)
        detect.set_default_engine(eng, "yoloface_8n" if st.detector != "retina" else "retinaface_10g")
        tgt = detect.load_image(target_path)
        # optional downscale
        h, w = tgt.shape[:2]
        W, H, s = vcore.out_size(w, h, st.max_short, st.align)
        frame = vcore.prep(tgt, W, H, s) if s < 1 or (W, H) != (w, h) else tgt
        if cancel and cancel(): raise Cancelled()
        if progress: progress(dict(stage="detect", done=0, total=1, detail="Finding faces in the photo…"))
        hits = detect.detect_frame_hits(frame, max(len(source_photo.faces), 2), st.detect_opts(), engine=eng)
        dets = [h.kps for h in hits]
        if st.same_gender and source_photo.genders:
            try:
                from . import gender as G
                G.annotate_hits(frame, hits)
                # Prefer same-gender pairing when labels exist
                try:
                    tg = [h.gender for h in hits]
                    if any(tg) and any(source_photo.genders):
                        assign_g = []
                        src_by_g = {}
                        for si, g in enumerate(source_photo.genders):
                            src_by_g.setdefault(g or "?", []).append(si)
                        used = set()
                        for i, g in enumerate(tg):
                            pool = [s for s in src_by_g.get(g or "?", []) if s not in used] or [
                                s for s in range(len(source_photo.faces)) if s not in used]
                            a = pool[0] if pool else -1
                            if a >= 0: used.add(a)
                            assign_g.append(a)
                        if max(assign_g) >= 0:
                            assign = assign_g
                        else:
                            assign = vcore.pair_single_frame(dets, len(source_photo.faces), st.rotation)
                    else:
                        assign = vcore.pair_single_frame(dets, len(source_photo.faces), st.rotation)
                except Exception:  # noqa: BLE001
                    assign = vcore.pair_single_frame(dets, len(source_photo.faces), st.rotation)
            except Exception:  # noqa: BLE001
                assign = vcore.pair_single_frame(dets, len(source_photo.faces), st.rotation)
        else:
            assign = vcore.pair_single_frame(dets, len(source_photo.faces), st.rotation)
        if not dets or max(assign) < 0:
            raise ValueError("No face found in the target photo. Try a clearer, front-facing picture.")
        if progress: progress(dict(stage="detect", done=1, total=1, detail=f"Found {len(dets)} face(s)"))
        lat = self._latents(eng, source_photo, assign)
        faces = [(dets[i].astype(np.float32), lat[a]) for i, a in enumerate(assign) if a >= 0]
        if not faces:
            raise ValueError("Could not map any faces. Tap Flip and try again, or pick another photo.")
        if progress: progress(dict(stage="swap", done=0, total=1, detail="Swapping faces…"))
        if cancel and cancel(): raise Cancelled()
        out = core.process_frame(eng, frame, faces, st.enhance,
                                 color_match=st.color_match, seamless=st.seamless)
        if progress: progress(dict(stage="mux", done=0, total=1, detail="Saving photo…"))
        out_dir = Path(st.out_dir) if st.out_dir else default_pictures_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        if out_path is None:
            stem = "".join(c for c in Path(target_path).stem if c.isalnum() or c in "-_ ")[:40].strip() or "photo"
            out_path = out_dir / f"FaceFusion_{stem}_{time.strftime('%Y%m%d-%H%M%S')}.jpg"
        out_path = Path(out_path)
        import cv2
        ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ok:
            raise RuntimeError("JPEG encode failed")
        # Atomic write — avoid a half file if the process is killed mid-save
        tmp = out_path.with_suffix(".jpg.tmp")
        buf.tofile(str(tmp))
        os.replace(tmp, out_path)
        if progress: progress(dict(stage="done", done=1, total=1, detail="Done"))
        return dict(path=str(out_path), W=out.shape[1], H=out.shape[0], mode="photo",
                    fps=0.0, duration=0.0, frames=1, encoder="jpeg", audio="photo (no sound)",
                    device=dev.label(), device_active=dev.active, enhance=ENHANCE_LABEL.get(st.enhance),
                    faces=len(faces), total_s=round(time.perf_counter() - t0, 2), size=out_path.stat().st_size,
                    before=frame, after=out, dets=dets, assign=assign)


def default_pictures_dir() -> Path:
    if os.name == "nt":
        try:
            import ctypes, uuid
            from ctypes import wintypes
            fid = uuid.UUID("{33E28130-4E1E-4676-835A-98395C3BC3BB}")  # FOLDERID_Pictures
            guid = (ctypes.c_byte * 16).from_buffer_copy(fid.bytes_le)
            pth = ctypes.c_wchar_p()
            if ctypes.windll.shell32.SHGetKnownFolderPath(ctypes.byref(guid), 0, None, ctypes.byref(pth)) == 0:
                base = Path(pth.value); ctypes.windll.ole32.CoTaskMemFree(pth)
                return base / "FaceFusion"
        except Exception:  # noqa: BLE001
            pass
    return Path.home() / "Pictures" / "FaceFusion"


# Attach dump helper onto Job (kept at module level for clarity after photo helpers).
def _job_dump(self, d, info, an, assign, pf, photo, st):
    """Same layout as reference video_pipeline --dump (for the parity comparison)."""
    sel = [int(round(t * info.fps)) for t in an.times]
    json.dump({"W": an.W, "H": an.H, "fps": an.fps, "src_fps": info.fps, "sel": sel, "assign": assign,
               "pairing_frame": pf, "n_src": len(photo.faces), "rotation": st.rotation}, open(f"{d}/meta.json", "w"))
    np.save(f"{d}/fb.npy", np.stack([np.asarray(f, np.float32) for f in photo.faces]))
    np.save(f"{d}/src_rgb.npy", photo.img[:, :, ::-1].copy())
    with open(f"{d}/dets.json", "w") as fh:
        json.dump([[x[:, :2].astype(float).round(6).tolist() for x in ds] for ds in an.dets], fh)
    with open(f"{d}/tracks.json", "w") as fh:
        json.dump({"raw": [{str(f): p.round(6).tolist() for f, p in t.items()} for t in an.tracks],
                   "smooth": [{str(f): p.round(6).tolist() for f, p in t.items()} for t in an.smooth]}, fh)


Job._dump = _job_dump
