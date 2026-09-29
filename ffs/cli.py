"""CLI modes for FaceFusionStudio(.exe): GUI / --selftest / --run / --run-photo / --bench / --screenshots."""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

from . import __version__

log = logging.getLogger("ffs")


def _setup_logging(verbose=True):
    from . import crashlog
    crashlog.install()
    base = crashlog.log_dir()
    base.mkdir(parents=True, exist_ok=True)
    handlers = [logging.FileHandler(base / "facefusionstudio.log", encoding="utf-8")]
    if verbose and sys.stdout is not None:
        handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=handlers, force=True)


def versions():
    import cv2, onnxruntime as ort
    from . import media
    try:
        ff = media.run([media.ffmpeg(), "-hide_banner", "-version"], timeout=30).stdout.decode().splitlines()[0]
    except Exception as e:  # noqa: BLE001
        ff = f"error: {e}"
    return dict(app=__version__, python=sys.version.split()[0], platform=platform.platform(),
                machine=platform.machine(), cpu_count=os.cpu_count(), onnxruntime=ort.__version__,
                providers=ort.get_available_providers(), opencv=cv2.__version__,
                numpy=np.__version__, ffmpeg=ff, frozen=bool(getattr(sys, "frozen", False)),
                detector="yoloface_8n")


def _progress_printer(prefix="", stages=None):
    last = [0.0]
    seen = stages if stages is not None else []
    def cb(d):
        now = time.time()
        st = d.get("stage")
        if st and (not seen or seen[-1] != st):
            seen.append(st)
        if st in ("done", "mux") or now - last[0] > 2:
            last[0] = now
            eta = d.get("eta")
            detail = d.get("detail") or ""
            log.info("%s%s %s/%s %.2f/s eta %s %s", prefix, st, d.get("done"), d.get("total"),
                     d.get("rate") or 0, f"{eta:.0f}s" if eta else "-", detail)
    return cb


def _models(store_dir, need, cache=None):
    from .models import ModelStore
    store = ModelStore(Path(store_dir) if store_dir else None)
    if cache:
        got = store.import_from(cache, need)
        if got: log.info("imported from cache %s: %s", cache, got)
    missing = [s for s in need if not store.is_installed(s)]
    if missing:
        log.info("downloading %s (%.1f MB)", [s.file for s in missing], sum(s.bytes for s in missing) / 1e6)
        t0 = time.time(); last = [0.0]
        def prog(f, done, total, bps, verifying):
            if time.time() - last[0] > 5:
                last[0] = time.time(); log.info("  %s %s %.0f/%.0f MB %.1f MB/s", "verify" if verifying else "get", f, done / 1e6, total / 1e6, bps / 1e6)
        store.ensure(missing, progress=prog)
        log.info("models ready in %.0f s", time.time() - t0)
    return store


def identity_scores(eng, photo, out_frame, in_frame, dets_out, assign):
    from . import core
    res = []
    for i, a in enumerate(assign):
        if a < 0: continue
        src = core.embedding(eng, photo.img, core.kps5(photo.faces[a]))
        o = core.embedding(eng, out_frame, core.kps5(dets_out[i]))
        n = core.embedding(eng, in_frame, core.kps5(dets_out[i]))
        cos = lambda x, y: float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-9))
        res.append(dict(face=i, src=a, swapped_vs_src=round(cos(o, src), 3), original_vs_src=round(cos(n, src), 3)))
    return res


def selftest(args):
    from . import detect, media, models as M, vcore
    from .job import Job, Settings, load_photo
    report = dict(ok=False, versions=versions(), checks={}, started=time.strftime("%Y-%m-%d %H:%M:%S"))
    out_dir = Path(args.out or tempfile.mkdtemp(prefix="ffs_selftest_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    rep_path = out_dir / "selftest_report.json"
    try:
        log.info("versions %s", json.dumps(report["versions"]))
        enh = None if args.enhance in (None, "off") else args.enhance
        need = list(M.REQUIRED) + ([M.ENHANCER_LIGHT] if enh == "gpen256" else []) + ([M.ENHANCER_HQ] if enh == "gpen512" else [])
        store = _models(args.models, need, args.model_cache)
        report["checks"]["models_sha256_ok"] = all(store.is_installed(s) for s in need)
        work = Path(tempfile.mkdtemp(prefix="ffs_st_"))
        src_video = work / "vidéo test ü.mp4"
        shutil.copy(args.video, src_video)
        info = media.probe(src_video)
        if not info.has_audio:
            log.info("sample has no sound -> adding a 440 Hz AAC tone")
            tmp = work / "with_tone.mp4"
            cp = media.run([media.ffmpeg(), "-v", "error", "-y", "-i", str(src_video), "-f", "lavfi", "-i",
                            f"sine=frequency=440:sample_rate=44100:duration={info.duration}",
                            "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-shortest", str(tmp)], timeout=120)
            assert cp.returncode == 0, cp.stderr.decode()
            os.replace(tmp, src_video); info = media.probe(src_video)
        report["input"] = dict(path=str(src_video), summary=info.summary(), codec=info.vcodec, frames=info.frames)
        photo_path = work / "fotó.jpg"; shutil.copy(args.photo, photo_path)
        photo = load_photo(photo_path, store=store, device=args.device)
        report["checks"]["photo_faces"] = len(photo.faces)
        report["photo_genders"] = list(photo.genders)
        report["photo_confidences"] = [round(h.confidence, 3) for h in photo.hits]
        from .detect import DetectOpts, reject_stats
        from .engine import Engine
        eng_tmp = Engine(store, args.device); eng_tmp.prepare(None)
        detect.set_default_engine(eng_tmp)
        loose = reject_stats(photo.img, DetectOpts(min_confidence=0.25, min_face_frac=0.01))
        strict = reject_stats(photo.img, DetectOpts(min_confidence=0.85, min_face_frac=0.08))
        report["detection_filter"] = dict(loose=loose, strict=strict, default_accepted=len(photo.hits))
        report["checks"]["low_conf_rejects"] = strict["accepted"] <= loose["accepted"]
        report["checks"]["gender_labels"] = bool(photo.genders) and all(g in ("male", "female", None) for g in photo.genders)
        # ---- photo selftest ----
        photo_out = out_dir / "selftest_photo.jpg"
        st_photo = Settings(max_short=args.max_short, enhance=enh, device=args.device, out_dir=str(out_dir),
                            min_confidence=args.min_confidence, same_gender=args.same_gender,
                            color_match=args.color_match, temporal_smooth=0.0)
        job = Job(store, args.device)
        pres = job.run_photo(photo_path, photo, st_photo, out_path=photo_out, progress=_progress_printer("photo "))
        report["photo_result"] = {k: v for k, v in pres.items() if k not in ("before", "after", "dets")}
        report["checks"]["photo_written"] = photo_out.is_file() and photo_out.stat().st_size > 1000
        if "before" in pres and "after" in pres:
            import cv2
            sheet = np.hstack([pres["before"], np.full((pres["before"].shape[0], 8, 3), 32, np.uint8), pres["after"]])
            cv2.imwrite(str(out_dir / "before_after_photo.jpg"), sheet)
        # ---- video selftest (must reach Final/mux and exit cleanly) ----
        st = Settings(start=args.start, length=args.length, fps=args.fps, max_short=args.max_short, enhance=enh,
                      device=args.device, out_dir=str(out_dir),
                      min_confidence=args.min_confidence, min_face_frac=args.min_face_frac,
                      same_gender=args.same_gender, color_match=args.color_match,
                      temporal_smooth=args.temporal_smooth, seamless=args.seamless, emb_track=args.emb_track)
        out = out_dir / "selftest_output.mp4"
        stages = []
        t_job = time.perf_counter()
        res = job.run(info, photo, st, out_path=out, progress=_progress_printer(stages=stages))
        report["result"] = res
        report["stages"] = stages
        report["job_s"] = round(time.perf_counter() - t_job, 2)
        c = report["checks"]
        c["reached_detect"] = "detect" in stages
        c["reached_swap"] = "swap" in stages
        c["reached_final_mux"] = "mux" in stages
        c["reached_done"] = "done" in stages
        c["final_exited_clean"] = out.is_file() and out.stat().st_size > 1000 and "done" in stages
        meta = media.probe_output(out)
        streams = meta.get("streams", [])
        v = [s for s in streams if s["codec_type"] == "video"]; a = [s for s in streams if s["codec_type"] == "audio"]
        exp = res["frames"] / res["fps"]
        vd = float(v[0].get("duration", 0)) if v else 0; ad = float(a[0].get("duration", 0)) if a else 0
        report["output"] = dict(streams=streams, format=meta.get("format"), expected_duration=exp)
        c = report["checks"]
        c["has_video_h264"] = bool(v) and v[0]["codec_name"] == "h264"
        c["has_audio"] = bool(a)
        c["video_duration_ok"] = abs(vd - exp) < 0.15
        c["audio_duration_ok"] = bool(a) and abs(ad - exp) < 0.25
        c["frame_count_ok"] = bool(v) and int(v[0].get("nb_frames", 0)) == res["frames"]
        c["faces_tracked"] = res["tracks"] >= 1 and max(res["assign"]) >= 0
        eng = job.get_engine(args.device)
        mid_t = args.start + (res["frames"] // 2) / res["fps"]
        o = media.read_frame_at(out, (res["frames"] // 2) / res["fps"])
        i_ = vcore.prep(media.read_frame_at(src_video, mid_t), res["W"], res["H"],
                        vcore.out_size(info.width, info.height, args.max_short, 2)[2])
        d = detect.detect_frame(i_, 2, engine=eng)
        asg = vcore.pair_single_frame(d, len(photo.faces), 0)
        ids = identity_scores(eng, photo, o, i_, d, asg)
        report["identity"] = ids
        c["identity_transferred"] = bool(ids) and all(x["swapped_vs_src"] > x["original_vs_src"] + 0.05 for x in ids)
        import cv2
        sheet = np.hstack([i_, np.full((i_.shape[0], 8, 3), 32, np.uint8), o])
        cv2.imwrite(str(out_dir / "before_after_mid.jpg"), sheet)
        report["sample_frames"] = dict(before_after=str(out_dir / "before_after_mid.jpg"), W=res["W"], H=res["H"])
        if args.dml_smoke:
            report["directml"] = dml_smoke(store, i_, d, asg, photo)
            # Simulated DML failure must keep the process alive on CPU
            report["dml_fallback_sim"] = _sim_dml_fallback(store)
            c["dml_fallback_sim_ok"] = bool(report["dml_fallback_sim"].get("ok"))
        report["ok"] = all(bool(x) for k, x in c.items() if k != "photo_faces") and c["photo_faces"] >= 1
    except Exception as e:  # noqa: BLE001
        report["error"] = f"{type(e).__name__}: {e}"
        report["traceback"] = traceback.format_exc()
        log.error("selftest failed: %s", report["traceback"])
    report["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    rep_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    log.info("report %s ok=%s checks=%s", rep_path, report["ok"], report["checks"])
    return 0 if report["ok"] else 1


def _sim_dml_fallback(store):
    """In-process simulation used by --selftest --dml-smoke."""
    from . import dml_probe
    from .engine import Engine
    out = {"ok": False}
    prev = os.environ.get("FFS_FORCE_DML_FAIL")
    try:
        os.environ["FFS_FORCE_DML_FAIL"] = "1"
        dml_probe.clear_status()
        eng = Engine(store, "auto")
        info = eng.prepare(None)
        out.update(active=info.active, fell_back=info.fell_back, reason=info.fallback_reason,
                   label=info.label())
        out["ok"] = info.active == "CPU" and bool(info.fell_back or info.fallback_reason)
        eng.close()
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
        out["ok"] = False
    finally:
        if prev is None:
            os.environ.pop("FFS_FORCE_DML_FAIL", None)
        else:
            os.environ["FFS_FORCE_DML_FAIL"] = prev
        dml_probe.clear_status()  # don't poison later GPU tries on the runner
    log.info("dml fallback sim %s", json.dumps(out, default=str))
    return out


def dml_smoke(store, frame, dets, assign, photo):
    from . import core
    from .engine import Engine
    import onnxruntime as ort
    out = {"available_providers": ort.get_available_providers(),
           "note": "Hosted runners have no discrete/iGPU; DML may use WARP or fall back."}
    try:
        cpu = Engine(store, "cpu"); cpu.prepare(None)
        try:
            dml = Engine(store, "dml")
            t0 = time.perf_counter(); dml.prepare(None); out["dml_init_s"] = round(time.perf_counter() - t0, 2)
            out["mode"] = "forced_dml"
        except Exception as e:  # noqa: BLE001
            out["forced_dml_error"] = f"{type(e).__name__}: {e}"
            dml = Engine(store, "auto")
            t0 = time.perf_counter(); dml.prepare(None); out["dml_init_s"] = round(time.perf_counter() - t0, 2)
            out["mode"] = "auto_after_forced_failed"
        out["device"] = dml.info.label(); out["per_model"] = dict(dml.info.per_model)
        out["adapters"] = dml.info.adapter; out["fallback_reason"] = dml.info.fallback_reason
        out["active"] = dml.info.active
        if dml.info.active == "DirectML":
            out["bench_dml"] = dml.benchmark(modes=("off",)); out["bench_cpu"] = cpu.benchmark(modes=("off",))
            def run(eng):
                lat = {}
                for a in set(x for x in assign if x >= 0):
                    lat[a] = core.latent_for(eng, core.embedding(eng, photo.img, core.kps5(photo.faces[a])))
                faces = [(dets[i].astype(np.float32), lat[a]) for i, a in enumerate(assign) if a >= 0]
                return core.process_frame(eng, frame, faces, None)
            a, b = run(cpu), run(dml)
            mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
            out["frame_psnr_dml_vs_cpu_db"] = round(10 * np.log10(255 ** 2 / mse), 2) if mse > 0 else "inf"
            out["ok"] = True
        else:
            out["ok"] = False
            out["error"] = out.get("forced_dml_error") or dml.info.fallback_reason or "DirectML not active"
    except Exception as e:  # noqa: BLE001
        out["ok"] = False; out["error"] = f"{type(e).__name__}: {e}"
    log.info("directml smoke %s", json.dumps(out, default=str))
    return out


def run_headless(args):
    from . import media, models as M
    from .job import Job, Settings, load_photo
    enh = None if args.enhance in (None, "off") else args.enhance
    need = list(M.REQUIRED) + ([M.ENHANCER_LIGHT] if enh == "gpen256" else []) + ([M.ENHANCER_HQ] if enh == "gpen512" else [])
    store = _models(args.models, need, args.model_cache)
    info = media.probe(args.run[0]); photo = load_photo(args.run[1], store=store, device=args.device)
    st = Settings(start=args.start, length=args.length, fps=args.fps, max_short=args.max_short, align=args.align,
                  enhance=enh, rotation=args.rotation, device=args.device, sequential_decode=args.sequential,
                  min_confidence=args.min_confidence, min_face_frac=args.min_face_frac,
                  same_gender=args.same_gender, color_match=args.color_match,
                  temporal_smooth=args.temporal_smooth, seamless=args.seamless, emb_track=args.emb_track)
    res = Job(store, args.device).run(info, photo, st, out_path=args.run[2], progress=_progress_printer(), dump_dir=args.dump)
    print(json.dumps(res, indent=2, default=str))
    return 0


def run_photo_headless(args):
    from . import models as M
    from .job import Job, Settings, load_photo
    enh = None if args.enhance in (None, "off") else args.enhance
    need = list(M.REQUIRED) + ([M.ENHANCER_LIGHT] if enh == "gpen256" else []) + ([M.ENHANCER_HQ] if enh == "gpen512" else [])
    store = _models(args.models, need, args.model_cache)
    photo = load_photo(args.run_photo[1], store=store, device=args.device)
    st = Settings(max_short=args.max_short, enhance=enh, device=args.device,
                  min_confidence=args.min_confidence, same_gender=args.same_gender, color_match=args.color_match)
    res = Job(store, args.device).run_photo(args.run_photo[0], photo, st, out_path=args.run_photo[2], progress=_progress_printer())
    print(json.dumps({k: v for k, v in res.items() if k not in ("before", "after", "dets")}, indent=2, default=str))
    return 0


def bench(args):
    from . import models as M
    from .job import Job
    store = _models(args.models, list(M.REQUIRED) + [M.ENHANCER_LIGHT], args.model_cache)
    job = Job(store, args.device)
    b = job.benchmark(args.device)
    info = job.engine.info
    res = dict(device=info.label(), per_model=info.per_model, bench_s=b, recommended=job.recommended_enhance())
    print(json.dumps(res, indent=2))
    return 0



def test_dml_fallback(args):
    """Simulate DirectML failure; assert Engine falls back to CPU and process stays alive."""
    from . import dml_probe, models as M
    from .engine import Engine
    report = {"ok": False, "checks": {}}
    try:
        os.environ["FFS_FORCE_DML_FAIL"] = "1"
        dml_probe.clear_status()
        store = _models(args.models, list(M.REQUIRED)[:1] or list(M.REQUIRED), args.model_cache)
        # Even with device=dml / auto, must land on CPU without aborting
        eng = Engine(store, "auto")
        info = eng.prepare(None)
        report["device"] = info.label()
        report["active"] = info.active
        report["fell_back"] = info.fell_back
        report["fallback_reason"] = info.fallback_reason
        report["checks"]["stayed_alive"] = True
        report["checks"]["active_is_cpu"] = info.active == "CPU"
        report["checks"]["marked_fell_back"] = bool(info.fell_back or info.fallback_reason)
        # Run one real inference on CPU path
        import numpy as np
        from . import detect
        detect.set_default_engine(eng)
        img = np.zeros((128, 128, 3), np.uint8)
        img[:] = (40, 60, 80)
        hits = detect.detect_faces(img, engine=eng)
        report["checks"]["detect_on_cpu_ok"] = True
        report["hits"] = len(hits)
        st = dml_probe.read_status()
        report["dml_status"] = st
        report["checks"]["status_file_records_failure"] = st.get("ok") is False or bool(report.get("fell_back"))
        report["ok"] = all(report["checks"].values())
        eng.close()
    except Exception as e:  # noqa: BLE001
        report["error"] = f"{type(e).__name__}: {e}"
        report["traceback"] = traceback.format_exc()
        log.error("test_dml_fallback failed: %s", report.get("traceback"))
    finally:
        os.environ.pop("FFS_FORCE_DML_FAIL", None)
    out = Path(args.out or ".")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "dml_fallback_report.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    log.info("dml fallback report %s ok=%s %s", path, report["ok"], report.get("checks"))
    return 0 if report["ok"] else 1


def main(argv=None):
    try:
        code = _main(argv)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 2)
    except BaseException:  # noqa: BLE001
        traceback.print_exc()
        try:
            from . import crashlog
            crashlog.install()
            crashlog.record_current(where="main")
        except Exception:  # noqa: BLE001
            pass
        code = 1
    logging.shutdown()
    try:
        sys.stdout and sys.stdout.flush(); sys.stderr and sys.stderr.flush()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code or 0)


def _main(argv=None):
    p = argparse.ArgumentParser(prog="FaceFusionStudio")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--dml-probe", metavar="MODEL", help="Child-process DirectML smoke (exit 0=ok)")
    p.add_argument("--test-dml-fallback", action="store_true",
                   help="Simulate DML failure and assert CPU fallback keeps process alive")
    p.add_argument("--run", nargs=3, metavar=("VIDEO", "PHOTO", "OUT"))
    p.add_argument("--run-photo", nargs=3, metavar=("TARGET", "SOURCE", "OUT"))
    p.add_argument("--bench", action="store_true")
    p.add_argument("--screenshots", metavar="DIR")
    p.add_argument("--video"); p.add_argument("--photo")
    p.add_argument("--models")
    p.add_argument("--model-cache")
    p.add_argument("--out")
    p.add_argument("--device", default="auto", choices=["auto", "dml", "cpu"])
    p.add_argument("--enhance", default="gpen256", choices=["off", "gpen256", "gpen512", "gfpgan"])
    p.add_argument("--start", type=float, default=0.0); p.add_argument("--length", type=float, default=10.0)
    p.add_argument("--fps", type=float, default=30.0); p.add_argument("--max-short", type=int, default=1080)
    p.add_argument("--align", type=int, default=2); p.add_argument("--rotation", type=int, default=0)
    p.add_argument("--sequential", action="store_true")
    p.add_argument("--dump"); p.add_argument("--dml-smoke", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--min-confidence", type=float, default=0.50)
    p.add_argument("--min-face-frac", type=float, default=0.025)
    p.add_argument("--no-same-gender", action="store_false", dest="same_gender")
    p.add_argument("--no-color-match", action="store_false", dest="color_match")
    p.add_argument("--temporal-smooth", type=float, default=0.18)
    p.add_argument("--seamless", action="store_true")
    p.add_argument("--no-emb-track", action="store_false", dest="emb_track")
    p.set_defaults(color_match=True, emb_track=True, same_gender=True)
    args, _ = p.parse_known_args(argv)
    if getattr(args, "dml_probe", None):
        from . import dml_probe
        return dml_probe.run_probe_in_this_process(args.dml_probe)
    headless = args.selftest or args.run or args.run_photo or args.bench or getattr(args, "test_dml_fallback", False)
    _setup_logging(verbose=bool(headless) and not args.quiet)
    if getattr(args, "test_dml_fallback", False):
        return test_dml_fallback(args)
    if args.selftest:
        if not args.video or not args.photo:
            p.error("--selftest needs --video and --photo")
        return selftest(args)
    if args.run:
        return run_headless(args)
    if args.run_photo:
        return run_photo_headless(args)
    if args.bench:
        return bench(args)
    from .gui.app import run_gui
    return run_gui(args)
