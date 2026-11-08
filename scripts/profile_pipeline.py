"""grid_demo işleme hattının adım adım süre profili.

Kullanım: .venv/Scripts/python scripts/profile_pipeline.py data/videos/iniesta_sample.mp4 --frames 100
Adaptörün iç adımlarını (ön işleme, iki model, son işleme) ayrı ayrı ölçer; GPU adımları senkronize edilir.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sgi.grid.grid import PitchGrid  # noqa: E402
from sgi.mapping.homography import HomographyEstimator, HomographyTracker  # noqa: E402
from sgi.registry import build  # noqa: E402
from sgi.types import PitchKeypoint  # noqa: E402
from sgi.viz.grid_overlay import GridOverlay, OverlayOptions  # noqa: E402


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--config", default="configs/grid_demo.yaml")
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    det = build("pitch_keypoint_detector", cfg["pitch_keypoint_detector"], root=ROOT)
    motion = build("camera_motion", cfg["camera_motion"])
    cap = cv2.VideoCapture(args.video)
    w, h = int(cap.get(3)), int(cap.get(4))
    est = HomographyEstimator(**cfg["homography"], image_size=(w, h))
    tracker = HomographyTracker((w, h), **cfg["tracking"])
    overlay = GridOverlay(PitchGrid.from_yaml(ROOT / "configs/grid.yaml"), (w, h))
    opt = OverlayOptions()
    tmp = ROOT / "outputs" / "profile_tmp.mp4"
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
    hm = det._hm
    sync = torch.cuda.synchronize if det.device.type == "cuda" else (lambda: None)

    times: dict[str, list[float]] = defaultdict(list)

    def tick(name: str, t0: float, gpu: bool = False) -> float:
        if gpu:
            sync()
        t1 = time.perf_counter()
        times[name].append((t1 - t0) * 1000)
        return t1

    for i in range(args.warmup + args.frames):
        if i == args.warmup:
            times.clear()
        t_frame = t = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        t = tick("1 video okuma", t)

        rgb = cv2.cvtColor(cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(det.device)
        x = x.half() / 255.0 if det.half else x.float() / 255.0
        t = tick("2 ön işleme + GPU'ya kopya", t, gpu=True)
        with torch.no_grad():
            a = det.model_kp(x).float()
            t = tick("3 HRNet keypoint modeli", t, gpu=True)
            b = det.model_lines(x).float()
            t = tick("4 HRNet çizgi modeli", t, gpu=True)
        kc = hm.get_keypoints_from_heatmap_batch_maxpool(a[:, :-1])
        lc = hm.get_keypoints_from_heatmap_batch_maxpool_l(b[:, :-1])
        t = tick("5 ısı haritası tepe bulma (maxpool+CPU)", t, gpu=True)
        kd = hm.coords_to_dict(kc, threshold=det.kp_threshold)
        ld = hm.coords_to_dict(lc, threshold=det.line_threshold)
        pts = hm.complete_keypoints(kd, ld, w=960, h=540, normalize=True)[0]
        kps = [PitchKeypoint(k, (p["x"] * w, p["y"] * h), det._world[k]) for k, p in pts.items()
               if k in det._world and k not in (12, 15, 16, 19)]
        t = tick("6 dict + çizgi kesişimleri (Python/scipy)", t)

        mo = motion.estimate(frame)
        t = tick("7 kamera hareketi (LK akış)", t)
        res = tracker.update(est.estimate(i, kps), mo, i)
        t = tick("8 homografi + takip", t)
        vis = overlay.draw(frame, res, opt, i, 25)
        t = tick("9 ızgara çizimi", t)
        writer.write(vis)
        t = tick("10 MP4 yazma", t)
        times["TOPLAM"].append((t - t_frame) * 1000)

    writer.release()
    tmp.unlink(missing_ok=True)
    total = np.mean(times["TOPLAM"])
    print(f"\n{Path(args.video).name}  {w}x{h}  {args.frames} kare  (fp16={det.half})")
    print(f"{'adım':45s} {'ort ms':>8s} {'%':>6s}")
    for k in sorted(times, key=lambda s: (s == "TOPLAM", int(s.split()[0]) if s[0].isdigit() else 0)):
        m = np.mean(times[k])
        print(f"{k:45s} {m:8.1f} {100 * m / total:6.1f}")
    print(f"=> {1000 / total:.2f} kare/sn")


if __name__ == "__main__":
    main()
