"""Canlı modda nesne kutularının çizim doğruluğu ve akıcılığı (tekrarlanabilir simülasyon).

Çevrimdışı `objects.npz` her karede dedektör + takip sonucudur; yer gerçeği olarak kullanılır. Canlıdaki teslim
zamanlaması taklit edilir: k karesinin sonucu (k = 0, period, 2·period, ...) k + lag karesinde gelir. Her karede
çizilen kutunun ayak noktası, o karenin gerçek ayak noktasıyla karşılaştırılır.

  hata    : çizilen ile gerçek ayak noktası arası (px)
  sarsıntı: kareden kareye çizilen hareket ile gerçek hareket farkı (px); takılma/sıçrama hissinin ölçüsü

Kullanım:
  .venv/Scripts/python.exe scripts/live_obj_sim.py data/videos/pas_test.mp4 outputs/grid_demo/pas_test/s28_d20/objects.npz
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
from sgi.mapping.live import MotionHistory  # noqa: E402
from sgi.objects import rows_to_objects  # noqa: E402
from sgi.registry import build  # noqa: E402
from sgi.viz.live_smoother import BoxFlow, LiveObjectSmoother  # noqa: E402


def foot(o):
    return np.array([0.5 * (o.bbox[0] + o.bbox[2]), o.bbox[3]])


def compensate_only(objs, T):
    out = {}
    for o in objs:
        p = T @ np.array([*foot(o), 1.0])
        out[o.track_id] = p[:2] / p[2]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("objects")
    ap.add_argument("--config", default="configs/grid_demo.yaml")
    ap.add_argument("--period", type=int, default=7, help="nesne sonucu kaç karede bir (canlı ölçüm: ~7)")
    ap.add_argument("--lag", type=int, default=8, help="sonuç kaç kare geç gelir (canlı ölçüm: ~8)")
    ap.add_argument("--sweep", action="store_true", help="ayar taraması")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    z = np.load(args.objects)
    rows, s0 = z["rows"], int(z["start_frame"])
    n = int(rows[:, 0].max()) + 1
    gt = [rows_to_objects(rows[rows[:, 0] == i]) for i in range(n)]

    motion_est = build("camera_motion", cfg["camera_motion"])
    frames = [f for _, f in FrameReader(args.video, s0, n)]
    motions = [motion_est.estimate(f).M for f in frames]

    variants = {"eski: yalnızca kamera telafisi": None,
                "yeni: hızla ilerletme + sönümleme": LiveObjectSmoother(),
                "yeni + optik akış": (LiveObjectSmoother(), BoxFlow())}
    if args.sweep:
        for ema in (0.5, 1.0):
            for decay in (0.0, 0.3, 0.5, 0.7):
                for hz in (8, 15):
                    variants[f"ema {ema} sönüm {decay} ufuk {hz}"] = LiveObjectSmoother(hz, decay, ema)
    for name, sm in variants.items():
        flow = None
        if isinstance(sm, tuple):
            sm, flow = sm
        t_flow = 0.0
        history = MotionHistory()
        err = {"oyuncu": [], "top": []}
        jerk = {"oyuncu": [], "top": []}
        prev_shown, prev_gt = {}, {}
        cur_k = None
        for i in range(n):
            history.add(i, motions[i])
            k_arr = i - args.lag
            t0 = time.perf_counter()
            if flow is not None:
                flow.push(i, frames[i])
            if k_arr >= 0 and k_arr % args.period == 0:
                cur_k = k_arr
                if flow is not None:
                    flow.start(cur_k, gt[cur_k])
            disp = flow.displacement(i) if flow is not None else None
            t_flow += time.perf_counter() - t0
            if cur_k is None:
                continue
            tf = lambda k: history.image_transform(k, i)  # noqa: E731
            if sm is None:
                T = tf(cur_k)
                shown = compensate_only(gt[cur_k], T) if T is not None else {o.track_id: foot(o) for o in gt[cur_k]}
            else:
                shown = {o.track_id: foot(o) for o in sm.update(i, gt[cur_k], cur_k, tf, flow=disp)}
            truth = {o.track_id: (foot(o), "top" if o.role == "ball" else "oyuncu") for o in gt[i]}
            for tid, p in shown.items():
                if tid not in truth:
                    continue
                g, kind = truth[tid]
                err[kind].append(np.linalg.norm(p - g))
                if tid in prev_shown and tid in prev_gt:
                    jerk[kind].append(np.linalg.norm((p - prev_shown[tid]) - (g - prev_gt[tid])))
            prev_shown, prev_gt = shown, {t: v[0] for t, v in truth.items()}
        print(f"{name}" + (f"  (optik akış {1000 * t_flow / n:.1f} ms/kare)" if flow is not None else ""))
        for kind in ("oyuncu", "top"):
            e, j = np.array(err[kind]), np.array(jerk[kind])
            if len(e):
                print(f"  {kind:6s} hata ort {e.mean():5.1f} px, %90 {np.percentile(e, 90):5.1f} | "
                      f"sarsıntı ort {j.mean():4.1f} px, %99 {np.percentile(j, 99):5.1f}")


if __name__ == "__main__":
    main()
