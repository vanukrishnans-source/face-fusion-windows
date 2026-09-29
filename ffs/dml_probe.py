"""Out-of-process DirectML probe.

Native faults inside onnxruntime-directml / DirectML on Radeon 780M abort the
*whole process* — Python try/except cannot catch them. We therefore create the
first DML session in a short-lived child. If the child dies or returns non-zero,
the parent stays alive and falls back to CPU.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

log = logging.getLogger("ffs")

STATUS_FILE = "dml_status.json"
PROBE_TIMEOUT_S = 90


def _state_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA")
    root = Path(base) if base else (Path.home() / ".local" / "share")
    d = root / "FaceFusionStudio"
    d.mkdir(parents=True, exist_ok=True)
    return d


def status_path() -> Path:
    return _state_dir() / STATUS_FILE


def read_status() -> dict:
    p = status_path()
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def write_status(ok: bool, reason: str = "", adapter: str = "") -> None:
    data = {
        "ok": bool(ok),
        "reason": (reason or "")[:400],
        "adapter": adapter or "",
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pid": os.getpid(),
    }
    try:
        status_path().write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        log.warning("could not write dml status: %s", e)


def force_fail_env() -> bool:
    return os.environ.get("FFS_FORCE_DML_FAIL", "").strip() in ("1", "true", "yes")


def dml_disabled_by_status() -> tuple[bool, str]:
    """Return (disabled, reason) if a previous probe said DML is unsafe."""
    if force_fail_env():
        reason = "FFS_FORCE_DML_FAIL=1 (simulated GPU failure)"
        write_status(False, reason)
        return True, reason
    if os.environ.get("FFS_FORCE_CPU", "").strip() in ("1", "true", "yes"):
        reason = "FFS_FORCE_CPU=1"
        write_status(False, reason)
        return True, reason
    st = read_status()
    if st.get("ok") is False:
        return True, st.get("reason") or "previous DirectML probe failed"
    return False, ""


def clear_status() -> None:
    try:
        status_path().unlink(missing_ok=True)
    except Exception:  # noqa: BLE001
        pass


def run_probe_in_this_process(model_path: str) -> int:
    """Entry for `--dml-probe MODEL`. Exit 0 = DML OK, 2 = soft fail, 1 = hard fail.

    This runs ONLY in the child. Keep it tiny — no Qt, no Engine thread.
    """
    if force_fail_env():
        print("DML_PROBE fail: forced", flush=True)
        return 2
    try:
        import numpy as np
        import onnxruntime as ort
    except Exception as e:  # noqa: BLE001
        print(f"DML_PROBE import fail: {e}", flush=True)
        return 2
    if "DmlExecutionProvider" not in ort.get_available_providers():
        print("DML_PROBE fail: provider missing", flush=True)
        return 2
    path = Path(model_path)
    if not path.is_file():
        print(f"DML_PROBE fail: missing model {model_path}", flush=True)
        return 2
    try:
        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.enable_mem_pattern = False
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        # Safer DirectML EP options for Radeon 780M / iGPU
        last = None
        sess = None
        for dml_opts in ({"device_id": 0}, {"device_id": 0, "disable_metacommands": True}):
            try:
                sess = ort.InferenceSession(
                    str(path), so,
                    providers=[("DmlExecutionProvider", dml_opts), "CPUExecutionProvider"],
                )
                used = sess.get_providers()
                if used and used[0] == "DmlExecutionProvider":
                    break
                last = f"not using DML ({used}) with {dml_opts}"
                sess = None
            except Exception as e:  # noqa: BLE001
                last = f"{type(e).__name__}: {e}"
                sess = None
        if sess is None:
            print(f"DML_PROBE fail: {last}", flush=True)
            return 2
        # Tiny warmup — first InferenceSession.run is where many 780M crashes hit
        inp = sess.get_inputs()[0]
        shape = []
        for d in inp.shape:
            if isinstance(d, int) and d > 0:
                shape.append(d)
            else:
                shape.append(1)
        if len(shape) < 2:
            shape = [1, 3, 64, 64]
        # Cap warmup tensor so probe stays fast/safe
        shape = [min(int(x), 640) for x in shape]
        feed = {inp.name: np.zeros(shape, np.float32)}
        sess.run(None, feed)
        print("DML_PROBE ok", flush=True)
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"DML_PROBE fail: {type(e).__name__}: {e}", flush=True)
        return 2


def _probe_command(model_path: str) -> list[str]:
    """Build argv that re-enters this app as a DML probe child."""
    if getattr(sys, "frozen", False):
        # Prefer the console exe next to the windowed one when present
        exe = Path(sys.executable)
        cli = exe.with_name("FaceFusionStudio_cli.exe")
        target = str(cli if cli.is_file() else exe)
        return [target, "--dml-probe", model_path]
    # Dev: python -m ffs --dml-probe
    return [sys.executable, "-m", "ffs", "--dml-probe", model_path]


def probe_directml(model_path: str, timeout: float = PROBE_TIMEOUT_S) -> tuple[bool, str]:
    """Run DirectML create+warmup in a child process. Never crashes the caller."""
    if force_fail_env():
        reason = "FFS_FORCE_DML_FAIL=1 (simulated GPU failure)"
        write_status(False, reason)
        return False, reason
    disabled, why = dml_disabled_by_status()
    # Re-probe if status missing; if previous ok, trust briefly; if previous fail, stay failed
    # until user clears (Options → retry GPU) or FFS_DML_REPROBE=1
    st = read_status()
    if st.get("ok") is True and os.environ.get("FFS_DML_REPROBE", "").strip() not in ("1", "true", "yes"):
        return True, ""
    if st.get("ok") is False and os.environ.get("FFS_DML_REPROBE", "").strip() not in ("1", "true", "yes"):
        return False, st.get("reason") or why or "previous DirectML probe failed"

    cmd = _probe_command(model_path)
    log.info("DirectML probe: %s", " ".join(cmd))
    creation = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW
    try:
        cp = subprocess.run(
            cmd, capture_output=True, timeout=timeout, creationflags=creation,
        )
    except subprocess.TimeoutExpired:
        reason = f"DirectML probe timed out after {timeout:.0f}s"
        log.error(reason)
        write_status(False, reason)
        return False, reason
    except Exception as e:  # noqa: BLE001
        reason = f"DirectML probe spawn failed: {type(e).__name__}: {e}"
        log.error(reason)
        write_status(False, reason)
        return False, reason

    out = ((cp.stdout or b"") + b"\n" + (cp.stderr or b"")).decode("utf-8", "replace")
    # Non-zero OR missing "DML_PROBE ok" → treat as failure (covers native abort = negative/crash codes)
    if cp.returncode == 0 and "DML_PROBE ok" in out:
        write_status(True, "", adapter="")
        log.info("DirectML probe succeeded")
        return True, ""
    # Child crashed (access violation often 0xC0000005 → large unsigned / -1073741819)
    reason = f"DirectML probe failed (exit {cp.returncode}): {out.strip()[-300:] or 'no output / native crash'}"
    log.error(reason)
    write_status(False, reason)
    return False, reason
