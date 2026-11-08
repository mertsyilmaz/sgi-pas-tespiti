"""Canlı modun tekrarlanabilir simülasyonu (D-029).

Canlı modda saha noktası modeli seyrek (her ~12 karede bir) ve gecikmeli (~8 kare) sonuç verir; arada ızgara
kamera hareketiyle taşınır. Gerçek canlı testin zamanlaması makinenin o anki yüküne (ısınma, başka uygulamalar)
göre değişir, bu yüzden varyantlar karşılaştırılamaz. Bu betik aynı zamanlamayı sabit parametrelerle taklit eder:
  - k karesinde (k % calib_every == 0) saha noktası ölçümü alınır, k + calib_lag karesinde teslim edilir ve
    kayıtlı kamera hareketiyle güncel kareye taşınır (canlıdaki MotionHistory.bring_forward ile aynı).
  - Çizgilere oturtma: 'inline' = güncel karede, refine_every karede bir (eski canlı davranış);
    'worker' = k karesinde başlar, k + refine_lag karesinde teslim edilir (arka plan süreci).
Doğruluk etiketsiz çizgi uyumu skoruyla (D-019) ölçülür: 5 karede bir.
Model çıktıları ve kamera hareketi bir kez hesaplanır, tüm varyantlar aynı veriyle çalışır.

Kullanım:  python scripts/live_sim.py data/videos/pas_test.mp4 --start 20 --duration 18
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from grid_demo import make_camera_prior, make_refiner  # noqa: E402
from sgi.io.video import FrameReader  # noqa: E402
from sgi.mapping.homography import HomographyEstimator, HomographyTracker  # noqa: E402
from sgi.mapping.live import MotionHistory  # noqa: E402
from sgi.mapping.quality import QualityMonitor  # noqa: E402
from sgi.registry import build  # noqa: E402
from sgi.types import HomographyResult  # noqa: E402

VARIANTS = {
    # ad: (tracker ayarları, oturtma modu, oturtma aralığı, oturtma gecikmesi)
    "eski (sabit karışım, satır içi /10)": (dict(gap_blend=False, zoom_reanchor=float("inf")), "inline", 10, 0),
    "aralığa göre karışım + zoom": (dict(), "inline", 10, 0),
    "arka plan oturtma /1, gecikme 1": (dict(), "worker", 1, 1),
    "arka plan oturtma /1, gecikme 2": (dict(), "worker", 1, 2),
    "arka plan oturtma /2, gecikme 2": (dict(), "worker", 2, 2),
    "satır içi oturtma /2": (dict(), "inline", 2, 0),
    "üst sınır: satır içi oturtma her kare": (dict(), "inline", 1, 0),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--config", default="configs/grid_demo.yaml")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--calib-every", type=int, default=12)
    ap.add_argument("--calib-lag", type=int, default=8)
    ap.add_argument("--only", default="", help="yalnızca adında bu metin geçen varyantlar")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    pitch = yaml.safe_load((ROOT / "configs/pitch.yaml").read_text(encoding="utf-8"))
    import cv2
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    s0, n = int(args.start * fps), int(args.duration * fps)

    detector = build("pitch_keypoint_detector", cfg["pitch_keypoint_detector"], root=ROOT,
                     pitch_length=pitch["length"], pitch_width=pitch["width"])
    motion_est = build("camera_motion", cfg["camera_motion"])
    print(f"[sim] {n} kare: kamera hareketi + saha noktaları (her {args.calib_every} karede) hesaplanıyor...")
    motions, kps, frames = [], {}, {}
    for fi, f in FrameReader(args.video, s0, n):
        i = fi - s0
        motions.append(motion_est.estimate(f))
        if i % args.calib_every == 0:
            kps[i] = detector.detect(f)
        frames[i] = f
    qm = QualityMonitor((w, h), pitch["length"], pitch["width"])

    for name, (tkw, mode, r_every, r_lag) in VARIANTS.items():
        if args.only and args.only not in name:
            continue
        estimator = HomographyEstimator(**cfg["homography"], image_size=(w, h), camera_prior=make_camera_prior(cfg, (w, h)))
        refiner, _ = make_refiner(cfg, (w, h), pitch, live=True)
        tcfg = dict(cfg["tracking"], **tkw)
        tracker = HomographyTracker((w, h), refiner=refiner if mode == "inline" else None, refine_every=r_every, **tcfg)
        history = MotionHistory()
        pending_meas: dict[int, HomographyResult] = {}
        pending_ref: dict[int, tuple[int, np.ndarray]] = {}
        scores, shown = [], 0
        for i in range(n):
            history.add(i, motions[i].M)
            if i in kps:                                   # ölçüm k'de başlar, k + lag'de gelir
                pending_meas[i + args.calib_lag] = estimator.estimate(i, kps[i])
            meas = pending_meas.pop(i, None)
            if meas is not None:
                meas = history.bring_forward(meas, i)
            if mode == "worker" and i in pending_ref:
                k, H_ref = pending_ref.pop(i)
                fwd = history.bring_forward(HomographyResult(frame_idx=k, H=H_ref), i - 1)
                if fwd is not None:
                    tracker.correct(fwd.H)
            res = tracker.update(meas, motions[i], i, frames.get(i))
            if (mode == "worker" and res.valid and i % r_every == 0
                    and not any(k2 > i for k2 in pending_ref)):
                H_new, ok, _ = refiner.refine(frames[i], res.H)
                if ok:
                    pending_ref[i + r_lag] = (i, H_new)
            shown += res.valid
            if i % 5 == 0 and res.valid:
                scores.append(qm.measure(frames[i], res.H).line_score)
        sc = np.array(scores)
        print(f"{name:38s} ızgara %{100 * shown / n:5.1f} | çizgi uyumu medyan {np.median(sc):.3f}, "
              f"en kötü %10 {np.percentile(sc, 10):.2f}, kötü (<0,65) %{100 * np.mean(sc < 0.65):4.1f}")


if __name__ == "__main__":
    main()
