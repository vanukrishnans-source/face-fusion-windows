"""CI helper: download (or verify cached) models into FFS_MODELS with SHA-256 checks.
usage: python scripts/prefetch_models.py [light] [hq] [retina]"""
import sys, time
sys.path.insert(0, '.')
from ffs.models import ModelStore, REQUIRED, ENHANCER_LIGHT, ENHANCER_HQ, RETINAFACE

want = list(REQUIRED)
if 'light' in sys.argv: want.append(ENHANCER_LIGHT)
if 'hq' in sys.argv: want.append(ENHANCER_HQ)
if 'retina' in sys.argv: want.append(RETINAFACE)
s = ModelStore()
last = [0.0]
def prog(f, done, total, bps, verifying):
    if time.time() - last[0] > 5 or done == total:
        last[0] = time.time()
        print(f"{f}: {done / 1e6:.0f}/{total / 1e6:.0f} MB {'verifying' if verifying else f'{bps / 1e6:.1f} MB/s'}", flush=True)
t0 = time.time()
s.ensure(want, progress=prog)
print('models ready in', s.dir, f'{time.time() - t0:.0f}s', [(m.file, s.is_installed(m)) for m in want])
