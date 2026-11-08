"""Hücre düzeninin kamera görüntüsündeki doğruluğu ve stabilitesi (pas kuralından bağımsız).

Izgara homografiyle görüntüye izdüşürülür. Bir düzenin her hücre kenarı 0,5 m aralıklı noktalara bölünür; nokta
gerçek bir boyalı saha çizgisinin üstündeyse "çizgide", değilse "çimde" sayılır. Her karede:
  - doğruluk: çizgideki kenar noktalarının görüntüdeki beyaz çizgiye uzaklığı (px). Çimdeki kenarlar için
    görüntüde karşılaştırılacak bir şey yoktur (doğrulanamaz).
  - titreme: kamera hareketi çıkarıldıktan sonra kenar noktasının kareden kareye oynaması (px):
    |G_t X - M_t^-1 G_{t-1} X|  (M_t: t -> t-1 kamera hareketi)
Çevrimdışı (calibration.npz) ve canlı (live_selftest.npz) homografiler ayrı ölçülür.

Kullanım: .venv/Scripts/python.exe scripts/grid_edge_stability.py data/videos/pas_test.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from grid_count_search import full_x, half_x, y_edges  # noqa: E402
from sgi.io.video import FrameReader  # noqa: E402
from sgi.mapping.homography import pitch_to_image  # noqa: E402
from sgi.mapping.quality import QualityMonitor  # noqa: E402
from sgi.registry import build  # noqa: E402

L, W = 105.0, 68.0
LAYOUTS = {
    "eşit 10x5 = 50 (mevcut)": (np.linspace(0, L, 11), np.linspace(0, W, 6)),
    "hizalı 10x5 = 50": (full_x(half_x(False, 1, 4)), y_edges(1, True, 1)),
    "hizalı 10x7 = 70": (full_x(half_x(False, 1, 4)), y_edges(2, True, 1)),
    "eşit 12x9 = 108": (np.linspace(0, L, 13), np.linspace(0, W, 10)),
}


def painted_segments():
    cy = W / 2
    segs = [((0, 0), (L, 0)), ((0, W), (L, W)), ((0, 0), (0, W)), ((L, 0), (L, W)), ((L / 2, 0), (L / 2, W))]
    for x0, sgn in ((0.0, 1), (L, -1)):
        for depth, half in ((16.5, 20.16), (5.5, 9.16)):
            x1 = x0 + sgn * depth
            segs += [((x0, cy - half), (x1, cy - half)), ((x1, cy - half), (x1, cy + half)),
                     ((x0, cy + half), (x1, cy + half))]
    pts = []
    for (ax, ay), (bx, by) in segs:
        t = np.linspace(0, 1, max(2, int(np.hypot(bx - ax, by - ay) / 0.1)))
        pts.append(np.c_[ax + (bx - ax) * t, ay + (by - ay) * t])
    return np.vstack(pts)


def edge_points(xe, ye, step=0.5):
    pts = []
    for x in xe:
        y = np.arange(0, W + 1e-6, step)
        pts.append(np.c_[np.full_like(y, x), y])
    for y in ye:
        x = np.arange(0, L + 1e-6, step)
        pts.append(np.c_[x, np.full_like(x, y)])
    return np.unique(np.round(np.vstack(pts), 3), axis=0)


def project(G, X, w, h):
    q = np.c_[X, np.ones(len(X))] @ G.T
    front = q[:, 2] > 1e-9
    p = q[:, :2] / np.where(front, q[:, 2], 1)[:, None]
    vis = front & (p[:, 0] >= 0) & (p[:, 0] < w) & (p[:, 1] >= 0) & (p[:, 1] < h)
    return p, vis


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--offline", default="outputs/grid_demo/pas_test/calibration.npz")
    ap.add_argument("--live", default="outputs/grid_demo/pas_test/s0_d64/live_selftest.npz")
    ap.add_argument("--config", default="configs/grid_demo.yaml")
    ap.add_argument("--every", type=int, default=3, help="doğruluk ölçümü kaç karede bir")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))

    sources = {}
    for name, path in (("çevrimdışı", args.offline), ("canlı", args.live)):
        if (ROOT / path).exists():
            z = np.load(ROOT / path)
            sources[name] = (int(z["start_frame"]), z["H"])
    s0 = min(s for s, _ in sources.values())
    n = max(s + len(H) for s, H in sources.values()) - s0

    paint = cKDTree(painted_segments())
    lay = {}
    for name, (xe, ye) in LAYOUTS.items():
        X = edge_points(np.asarray(xe, float), np.asarray(ye, float))
        on = paint.query(X)[0] < 0.06
        lay[name] = (X, on)

    motion_est = build("camera_motion", cfg["camera_motion"])
    stats = {(s, l): {"jit_on": [], "jit_off": [], "acc": [], "vis_on": 0, "vis_all": 0} for s in sources for l in lay}
    prevG = {s: None for s in sources}
    qm = None
    for fi, frame in FrameReader(args.video, s0, n):
        h, w = frame.shape[:2]
        M = motion_est.estimate(frame).M
        dist = None
        if (fi - s0) % args.every == 0:
            if qm is None:
                qm = QualityMonitor((w, h), L, W)
            lines = qm.masks(frame)[2]
            sc = w / lines.shape[1]
            dist = cv2.distanceTransform((lines == 0).astype(np.uint8), cv2.DIST_L2, 3) * sc
        for s, (start, Hs) in sources.items():
            k = fi - start
            H = Hs[k] if 0 <= k < len(Hs) else None
            G = None if H is None or np.isnan(H).any() else pitch_to_image(H, (w, h))
            for l, (X, on) in lay.items():
                if G is None:
                    continue
                p, vis = project(G, X, w, h)
                st = stats[(s, l)]
                st["vis_on"] += int(np.sum(vis & on))
                st["vis_all"] += int(np.sum(vis))
                if dist is not None and np.any(vis & on):
                    q = p[vis & on] / sc
                    st["acc"].append(dist[np.clip(q[:, 1].astype(int), 0, dist.shape[0] - 1),
                                          np.clip(q[:, 0].astype(int), 0, dist.shape[1] - 1)])
                if prevG[s] is not None and M is not None:
                    pp, vp = project(prevG[s], X, w, h)
                    Mi = np.linalg.inv(M)
                    pr = np.c_[pp, np.ones(len(pp))] @ Mi.T
                    pr = pr[:, :2] / pr[:, 2:3]
                    ok = vis & vp
                    j = np.hypot(*(p - pr).T)
                    st["jit_on"].append(j[ok & on])
                    st["jit_off"].append(j[ok & ~on])
            prevG[s] = G

    for s in sources:
        print(f"\n=== {s} homografi ===")
        print(f"{'düzen':26s} {'görünen kenarın çizgide olanı':>30s} | {'çizgideki kenarın beyaz çizgiye uzaklığı':>40s} | "
              f"titreme (çizgide / çimde)")
        for l in lay:
            st = stats[(s, l)]
            acc = np.concatenate(st["acc"]) if st["acc"] else np.array([np.nan])
            jo = np.concatenate(st["jit_on"]) if st["jit_on"] else np.array([np.nan])
            jf = np.concatenate(st["jit_off"]) if st["jit_off"] else np.array([np.nan])
            print(f"{l:26s} {100 * st['vis_on'] / max(st['vis_all'], 1):29.1f}% | "
                  f"medyan {np.median(acc):4.1f} px, %90 {np.percentile(acc, 90):5.1f} px{'':12s} | "
                  f"medyan {np.median(jo):.2f} / {np.median(jf):.2f} px, %95 {np.percentile(jo, 95):.1f} / "
                  f"{np.percentile(jf, 95):.1f} px")


if __name__ == "__main__":
    main()
