"""Video core — frame selection, tracking, One-Euro smoothing, pairing.

Verbatim port of reference/test/video_pipeline.py (which the Android VideoCore.kt mirrors: identical track
ids / pairing, smoothed points within 1e-6 px). Only change: out_size() takes the short-side cap and the
dimension alignment as parameters (phone: 720 / 16; Windows default: 1080 / 2, so 1920x1080 stays 1920x1080).
"""
from __future__ import annotations

import math

import cv2
import numpy as np


def out_size(w, h, max_short=720, align=16):
    s = min(1.0, max_short / min(w, h))
    W, H = int(round(w * s)), int(round(h * s))
    return W - W % align, H - H % align, s


def frame_times(duration, src_fps, start, end, fps):
    idx = []; slot = 0; n = int(math.floor(duration * src_fps + 1e-6))
    for i in range(n):
        t = i / src_fps
        if t < start - 1e-9 or t >= end - 1e-9: continue
        if t - start >= slot / fps - 1e-9:
            idx.append(i); slot += 1
            while slot / fps <= t - start + 1e-9: slot += 1
    return idx


class SlotSelector:
    """Streaming form of frame_times() driven by real frame timestamps (PTS) instead of i/src_fps, so
    variable-frame-rate phone videos keep their timing. For constant-rate files it selects exactly the
    same frames as frame_times()."""

    def __init__(self, start, end, fps):
        self.start, self.end, self.fps, self.slot = start, end, fps, 0

    def accept(self, t):
        """-1: before range, 0: skip, 1: take, 2: past the end."""
        if t < self.start - 1e-9: return -1
        if t >= self.end - 1e-9: return 2
        if t - self.start >= self.slot / self.fps - 1e-9:
            self.slot += 1
            while self.slot / self.fps <= t - self.start + 1e-9: self.slot += 1
            return 1
        return 0


def prep(frame, W, H, s):
    if s < 1: frame = cv2.resize(frame, (int(round(frame.shape[1] * s)), int(round(frame.shape[0] * s))), interpolation=cv2.INTER_AREA)
    y0 = (frame.shape[0] - H) // 2; x0 = (frame.shape[1] - W) // 2
    return np.ascontiguousarray(frame[y0:y0 + H, x0:x0 + W])


# ------------------------------------------------------------------ dedupe / tracking
def bbox(p):
    return float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy; u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def is_dup(a, b):
    ba, bb = bbox(a), bbox(b)
    if iou(ba, bb) > 0.3: return True
    ca = ((ba[0] + ba[2]) / 2, (ba[1] + ba[3]) / 2); cb = ((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2)
    return math.hypot(ca[0] - cb[0], ca[1] - cb[1]) < 0.5 * min(ba[2] - ba[0], bb[2] - bb[0])


def dedupe(faces):
    keep = []
    for f in faces:
        if not any(is_dup(f, k) for k in keep): keep.append(f)
    return keep


def match_score(tb, db):
    s = iou(tb, db)
    if s >= 0.1: return s
    tcx, tcy = (tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2; dcx, dcy = (db[0] + db[2]) / 2, (db[1] + db[3]) / 2
    r = 0.5 * max(tb[2] - tb[0], db[2] - db[0]); d = math.sqrt((tcx - dcx) ** 2 + (tcy - dcy) ** 2)
    return 0.1 * (1 - d / r) if d < r else 0.0


def emb_sim(a, b):
    """Cosine similarity of two L2-ish embeddings; 0 if either missing."""
    if a is None or b is None: return 0.0
    a = np.asarray(a, np.float64).ravel(); b = np.asarray(b, np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9: return 0.0
    return float(np.dot(a, b) / (na * nb))


def track(dets, max_gap, embeddings=None, emb_weight=0.45):
    """IoU / centre tracking with optional ArcFace embedding affinity (v2).

    embeddings: parallel to dets — list[list[np.ndarray|None]] per frame per detection.
    Combined score = (1 - emb_weight) * spatial + emb_weight * max(0, cosine).
    Keeps the same person on the same track ID across frames (cuts identity flips / flicker).
    """
    tracks = []; last = []; track_emb = []
    for f, ds in enumerate(dets):
        boxes = [bbox(d) for d in ds]
        embs = (embeddings[f] if embeddings and f < len(embeddings) else [None] * len(ds))
        act = [t for t in range(len(tracks)) if f - last[t][0] <= max_gap]
        cand = []
        for t in act:
            for j, b in enumerate(boxes):
                spatial = match_score(last[t][1], b)
                if spatial <= 0 and emb_weight <= 0: continue
                sim = emb_sim(track_emb[t], embs[j]) if emb_weight > 0 else 0.0
                # allow a weak spatial miss if embedding is strong (crossing / brief occlusion)
                if spatial <= 0 and sim < 0.35: continue
                s = (1.0 - emb_weight) * spatial + emb_weight * max(0.0, sim)
                if s > 0: cand.append((-s, t, j))
        cand.sort()
        used_t, used_d = set(), set()
        for _, t, j in cand:
            if t in used_t or j in used_d: continue
            used_t.add(t); used_d.add(j)
            tracks[t][f] = ds[j][:, :2].astype(np.float64); last[t] = (f, boxes[j])
            if embs[j] is not None:
                # EMA of track embedding for stable ID
                te = track_emb[t]
                track_emb[t] = (0.7 * te + 0.3 * np.asarray(embs[j], np.float64)) if te is not None else np.asarray(embs[j], np.float64)
        for j, d in enumerate(ds):
            if j not in used_d:
                tracks.append({f: d[:, :2].astype(np.float64)}); last.append((f, boxes[j]))
                track_emb.append(np.asarray(embs[j], np.float64) if embs[j] is not None else None)
    return tracks


# ------------------------------------------------------------------ smoothing
# Expression-aware One-Euro (v2): keep rigid head motion smoother, but let mouth / eyes
# follow the video with a higher cutoff so open-mouth / smile / blink are not frozen.
MIN_CUTOFF = 1.0; BETA = 3.0; D_CUTOFF = 1.0
EXPR_MIN_CUTOFF = 4.5; EXPR_BETA = 1.2   # lips / eyes: track faster
# 5-point landmark indices: eyes (0,1) + mouth (3,4) stay responsive for expression follow
_EXPR_IDX = [0, 1, 3, 4]


def alpha(cutoff, te):
    tau = 1.0 / (2 * math.pi * cutoff); return 1.0 / (1.0 + tau / te)


def one_euro(seq, te, min_cutoff=MIN_CUTOFF, beta=BETA):
    out = np.empty_like(seq); out[0] = seq[0]; s_hat = 0.0
    ad = alpha(D_CUTOFF, te)
    for i in range(1, len(seq)):
        c1 = seq[i].mean(0); c0 = seq[i - 1].mean(0)
        w = seq[i][:, 0].max() - seq[i][:, 0].min()
        raw = math.sqrt((c1[0] - c0[0]) ** 2 + (c1[1] - c0[1]) ** 2) / te / max(w, 1.0)
        s_hat = ad * raw + (1 - ad) * s_hat
        a = alpha(min_cutoff + beta * s_hat, te)
        out[i] = a * seq[i] + (1 - a) * out[i - 1]
    return out


def one_euro_expression(seq, te):
    """Smooth rigid structure more; keep lips/eyes responsive so swapped mouth follows the video."""
    rigid = one_euro(seq, te, MIN_CUTOFF, BETA)
    expr = one_euro(seq, te, EXPR_MIN_CUTOFF, EXPR_BETA)
    out = rigid.copy()
    idx = [i for i in _EXPR_IDX if i < seq.shape[1]]
    if idx:
        out[:, idx, :] = expr[:, idx, :]
    return out


MAX_FILL = 2


def fill_gaps(tr, max_fill=MAX_FILL):
    frames = sorted(tr); out = dict(tr)
    for a, b in zip(frames, frames[1:]):
        if 1 < b - a <= max_fill + 1:
            for f in range(a + 1, b):
                u = (f - a) / (b - a); out[f] = (1 - u) * tr[a] + u * tr[b]
    return out


def smooth_track(tr, fps):
    tr = fill_gaps(tr)
    te = 1.0 / fps; frames = sorted(tr); out = {}
    runs = []; cur = [frames[0]]
    for f in frames[1:]:
        if f == cur[-1] + 1: cur.append(f)
        else: runs.append(cur); cur = [f]
    runs.append(cur)
    for run in runs:
        seq = np.stack([tr[f] for f in run])
        fw = one_euro_expression(seq, te); bw = one_euro_expression(seq[::-1].copy(), te)[::-1]
        sm = (fw + bw) / 2
        for k, f in enumerate(run): out[f] = sm[k]
    return out


# ------------------------------------------------------------------ pairing
def pair_tracks(tracks, n_src, rotation):
    nf = max(max(t) for t in tracks) + 1 if tracks else 0
    solid_len = min(max(3, nf // 4), 8)
    solid = [k for k, t in enumerate(tracks) if len(t) >= solid_len] or list(range(len(tracks)))
    vis = [[k for k in solid if f in tracks[k]] for f in range(nf)]
    target = min(n_src, max((len(v) for v in vis), default=0))
    assign = [-1] * len(tracks)
    if target == 0: return assign, -1
    pf = next(f for f in range(nf) if len(vis[f]) >= target)
    cx = lambda k, f: tracks[k][f][:, 0].mean()
    wd = lambda k, f: float(np.ptp(tracks[k][f][:, 0]))
    biggest = sorted(vis[pf], key=lambda k: -wd(k, pf))[:max(n_src, 1)] if n_src >= 2 else vis[pf]
    first = sorted(biggest, key=lambda k: cx(k, pf))
    for i, k in enumerate(first):
        assign[k] = (i + rotation) % n_src if n_src >= 2 else 0
    order = sorted([k for k in range(len(tracks)) if assign[k] < 0], key=lambda k: min(tracks[k]))
    for k in order:
        fk = set(tracks[k]); f0 = min(tracks[k]); c = tracks[k][f0].mean(0)
        best = None
        for j in range(len(tracks)):
            if assign[j] < 0 or j == k or fk & set(tracks[j]): continue
            fj = min(tracks[j], key=lambda f: abs(f - f0))
            d = float(np.linalg.norm(tracks[j][fj].mean(0) - c))
            if best is None or d < best[0]: best = (d, j)
        if best is not None:
            assign[k] = assign[best[1]]
    return assign, pf


def pair_single_frame(dets, n_src, rotation):
    """Pairing for a one-frame preview, same rule as pair_tracks on its pairing frame."""
    assign = [-1] * len(dets)
    if not dets or n_src == 0: return assign
    idx = list(range(len(dets)))
    wd = lambda k: float(np.ptp(dets[k][:, 0]))
    biggest = sorted(idx, key=lambda k: -wd(k))[:max(n_src, 1)] if n_src >= 2 else idx
    first = sorted(biggest, key=lambda k: dets[k][:, 0].mean())
    for i, k in enumerate(first):
        assign[k] = (i + rotation) % n_src if n_src >= 2 else 0
    return assign


def pair_tracks_gender(tracks, n_src, rotation, track_genders=None, src_genders=None, same_gender=True):
    """Left-to-right pairing with optional same-gender preference (v2).

    Falls back to pair_tracks when gender unknown or same_gender is False. When enabled,
    assigns each video track the next unused source face of matching gender before filling
    leftovers left-to-right (Flip/rotation still applies within gender groups).
    """
    assign, pf = pair_tracks(tracks, n_src, rotation)
    if not same_gender or not track_genders or not src_genders or pf < 0:
        return assign, pf
    # Rebuild assignment preferring gender match on the pairing frame
    vis = [k for k in range(len(tracks)) if pf in tracks[k] and assign[k] >= 0]
    if not vis:
        return assign, pf
    vis = sorted(vis, key=lambda k: tracks[k][pf][:, 0].mean())
    src_order = list(range(n_src))
    if n_src >= 2:
        src_order = [(i + rotation) % n_src for i in range(n_src)]
    used_src = set()
    new_assign = [-1] * len(tracks)
    # pass 1: same gender
    for k in vis:
        tg = track_genders[k] if k < len(track_genders) else None
        if not tg: continue
        for si in src_order:
            if si in used_src: continue
            sg = src_genders[si] if si < len(src_genders) else None
            if sg and sg == tg:
                new_assign[k] = si; used_src.add(si); break
    # pass 2: leftovers keep spatial order
    for k in vis:
        if new_assign[k] >= 0: continue
        for si in src_order:
            if si not in used_src:
                new_assign[k] = si; used_src.add(si); break
    # keep secondary tracks that shared an ID with a primary
    for k in range(len(tracks)):
        if new_assign[k] < 0 and assign[k] >= 0:
            # map old source slot through gender remap if possible
            new_assign[k] = new_assign[vis[0]] if vis else assign[k]
    return new_assign, pf
