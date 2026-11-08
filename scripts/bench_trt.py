"""PyTorch ve TensorRT karşılaştırması: süre ve çıktı farkı (saha noktaları / nesne kutuları).

Kullanım: .venv/Scripts/python.exe scripts/bench_trt.py data/videos/pas_test.mp4 [--what kp|det] [--n 150]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sgi.io.video import FrameReader  # noqa: E402
from sgi.registry import build  # noqa: E402


def timed(fn, frames):
    import torch
    out, ts = [], []
    for f in frames:
        torch.cuda.synchronize()
        t = time.perf_counter()
        out.append(fn(f))
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    return out, 1000 * np.median(ts[5:])


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--what", choices=["kp", "det"], default="kp")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--config", default="configs/grid_demo.yaml")
    ap.add_argument("--engine", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    frames = [f for _, f in FrameReader(args.video, 0, 1615)][::1615 // args.n][:args.n]

    if args.what == "kp":
        c = {k: v for k, v in cfg["pitch_keypoint_detector"].items() if k != "trt_engine"}
        a = build("pitch_keypoint_detector", c, root=ROOT)
        b = build("pitch_keypoint_detector", {**c, "trt_engine": args.engine}, root=ROOT)
        ra, ta = timed(a.detect, frames)
        rb, tb = timed(b.detect, frames)
        same, only, d = 0, 0, []
        for x, y in zip(ra, rb):
            px = {k.kp_id: k.image_xy for k in x}
            py = {k.kp_id: k.image_xy for k in y}
            same += len(px.keys() & py.keys())
            only += len(px.keys() ^ py.keys())
            d += [np.hypot(px[k][0] - py[k][0], px[k][1] - py[k][1]) for k in px.keys() & py.keys()]
        d = np.array(d)
        print(f"süre (detect, medyan): PyTorch {ta:.1f} ms | TensorRT {tb:.1f} ms")
        print(f"nokta: ortak {same}, yalnızca birinde {only} | ortak noktalarda fark medyan {np.median(d):.2f} px, "
              f"%95 {np.percentile(d, 95):.2f} px, maks {d.max():.1f} px | ort nokta/kare {same / len(frames):.1f}")
    else:
        c = {k: v for k, v in cfg["objects"]["detector"].items() if k != "trt_engine"}
        a = build("object_detector", c, root=ROOT)
        b = build("object_detector", {**c, "trt_engine": args.engine}, root=ROOT)
        ra, ta = timed(a.detect, frames)
        rb, tb = timed(b.detect, frames)
        from collections import Counter
        ca, cb = Counter(), Counter()
        ious = []
        for x, y in zip(ra, rb):
            ca.update(o.cls for o in x)
            cb.update(o.cls for o in y)
            for o in x:
                best = 0.0
                for p in y:
                    if p.cls != o.cls:
                        continue
                    ix = max(0, min(o.bbox[2], p.bbox[2]) - max(o.bbox[0], p.bbox[0]))
                    iy = max(0, min(o.bbox[3], p.bbox[3]) - max(o.bbox[1], p.bbox[1]))
                    u = ((o.bbox[2] - o.bbox[0]) * (o.bbox[3] - o.bbox[1]) + (p.bbox[2] - p.bbox[0]) * (p.bbox[3] - p.bbox[1])
                         - ix * iy)
                    best = max(best, ix * iy / max(u, 1e-6))
                ious.append(best)
        ious = np.array(ious)
        print(f"süre (detect, medyan): PyTorch {ta:.1f} ms | TensorRT {tb:.1f} ms")
        print(f"tespit sayısı PyTorch {dict(ca)} | TensorRT {dict(cb)}")
        print(f"PyTorch kutusunun TensorRT'de eşi (IoU>0,5): %{100 * np.mean(ious > 0.5):.1f}, IoU medyan {np.median(ious):.3f}")


if __name__ == "__main__":
    main()
