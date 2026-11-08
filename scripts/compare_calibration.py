"""İki calibration.npz dosyasını karşılaştırır (regresyon testi).

Kullanım: .venv/Scripts/python scripts/compare_calibration.py referans.npz yeni.npz [--tol 0.05]
Her kare için görüntüdeki kontrol noktaları iki homografiyle sahaya izdüşürülür; fark metre cinsindendir.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sgi.mapping.homography import apply_h  # noqa: E402


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("new")
    ap.add_argument("--tol", type=float, default=0.05, help="kabul edilen en büyük fark (m)")
    args = ap.parse_args()
    a, b = np.load(args.ref), np.load(args.new)
    if len(a["H"]) != len(b["H"]):
        print(f"FARKLI kare sayısı: {len(a['H'])} vs {len(b['H'])}")
        return 1
    w, h = a["size"]
    ctrl = np.array([(x, y) for y in np.linspace(.4 * h, .95 * h, 3) for x in np.linspace(.1 * w, .9 * w, 4)])

    src_diff = int(np.sum(a["source"] != b["source"]))
    d = []
    for Ha, Hb in zip(a["H"], b["H"]):
        if np.isnan(Ha).any() or np.isnan(Hb).any():
            continue
        d.append(np.max(np.linalg.norm(apply_h(Ha, ctrl) - apply_h(Hb, ctrl), axis=1)))
    d = np.array(d)
    print(f"kare: {len(a['H'])} | kaynağı farklı kare: {src_diff} | karşılaştırılan: {len(d)}")
    if len(d):
        print(f"fark (m): medyan {np.median(d):.4f}  %95 {np.percentile(d, 95):.4f}  max {d.max():.4f}")
    ok = src_diff == 0 and (len(d) == 0 or d.max() <= args.tol)
    print("SONUÇ:", "GEÇTİ" if ok else f"KALDI (tol={args.tol} m)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
