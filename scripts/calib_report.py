"""Kalibrasyon kalite raporu: kategori dağılımı, sorunlu bölümler, zaman çizelgesi (14_CALIBRATION_QUALITY §3).

Kullanım: .venv/Scripts/python scripts/calib_report.py outputs/grid_demo/test_2min/calibration.npz
Çıktılar npz ile aynı klasöre: calibration_report.json, calibration_timeline.png
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sgi.mapping.diagnostics import CATEGORIES, DiagnosticThresholds, categorize, segments  # noqa: E402

COLORS = {"OK": (80, 200, 80), "OK?": (150, 220, 150), "OK~": (120, 230, 230), "E1": (160, 160, 160), "E2": (0, 200, 255),
          "E3": (0, 140, 255), "E4": (200, 80, 200), "E4!": (0, 0, 255), "E5": (255, 200, 0),
          "E5!": (60, 20, 220), "E6": (255, 120, 120), "E7": (120, 60, 0)}
SRC_COLORS = [(60, 60, 200), (80, 200, 80), (0, 215, 255)]   # none, keypoints, camera_motion (BGR)


def timeline(c, cats, fps: float, path: Path, width: int = 1600) -> None:
    n = len(cats)
    rows = [("kaynak", 26), ("kategori", 26), ("cizgi uyumu", 90), ("yesil orani", 60)]
    left, top, gap = 130, 30, 12
    hgt = top + sum(r[1] + gap for r in rows) + 30 + 22 * ((len(CATEGORIES) + 3) // 4)
    img = np.full((hgt, width + left + 20, 3), 255, np.uint8)
    x = lambda i: left + int(i * width / n)  # noqa: E731
    y = top
    for name, rh in rows:
        cv2.putText(img, name, (8, y + rh // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.rectangle(img, (left, y), (left + width, y + rh), (200, 200, 200), 1)
        if name == "kaynak":
            for i in range(n):
                cv2.line(img, (x(i), y), (x(i), y + rh), SRC_COLORS[int(c["source"][i])], 1)
        elif name == "kategori":
            for i in range(n):
                cv2.line(img, (x(i), y), (x(i), y + rh), COLORS[cats[i]], 1)
        else:
            vals = c["line_score"] if name == "cizgi uyumu" else c["green_ratio"]
            if name == "cizgi uyumu":
                for thr in (DiagnosticThresholds().min_line_score, DiagnosticThresholds().bad_line_score):
                    yt = y + rh - int(thr * rh)
                    cv2.line(img, (left, yt), (left + width, yt), (0, 0, 255), 1)
            for i in range(n):
                if not np.isnan(vals[i]):
                    cv2.circle(img, (x(i), y + rh - int(np.clip(vals[i], 0, 1) * rh)), 1, (90, 60, 0), -1)
        y += rh + gap
    for s in range(0, int(n / fps) + 1, 10):                    # 10 sn'lik zaman ekseni
        cv2.line(img, (x(s * fps), y - gap), (x(s * fps), y - gap + 5), (0, 0, 0), 1)
        cv2.putText(img, f"{s}s", (x(s * fps) - 8, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    y += 30
    for k, (code, desc) in enumerate(CATEGORIES.items()):
        cx, cy = left + (k % 4) * (width // 4), y + (k // 4) * 22
        cv2.rectangle(img, (cx, cy - 10), (cx + 14, cy + 2), COLORS[code], -1)
        cv2.putText(img, f"{code}: {desc}"[:58], (cx + 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1,
                    cv2.LINE_AA)
    cv2.imwrite(str(path), img)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--min-seg", type=float, default=1.0, help="raporlanan sorunlu bölümün en kısa süresi (sn)")
    args = ap.parse_args()
    npz = Path(args.npz)
    c = np.load(npz, allow_pickle=False)
    if "line_score" not in c:
        sys.exit("Bu npz kalite göstergesi içermiyor. `process` komutunu yeniden çalıştırın.")
    fps = float(c["fps"])
    cats = categorize(c)
    n = len(cats)
    cnt = Counter(cats)
    calibrated = c["source"] != 0
    ls = c["line_score"][calibrated & ~np.isnan(c["line_score"])]

    print(f"{npz.parent.name}: {n} kare ({n / fps:.0f} sn)")
    print(f"kalibre: %{100 * calibrated.mean():.1f}  (saha noktası %{100 * np.mean(c['source'] == 1):.1f}, "
          f"kamera hareketi %{100 * np.mean(c['source'] == 2):.1f})")
    if len(ls):
        print(f"çizgi uyumu (kalibre kareler): medyan {np.median(ls):.2f}, %10'luk {np.percentile(ls, 10):.2f}")
    if "goal_found" in c and c["goal_visible"].any():
        print(f"kale (kadrajda olduğu ölçülen kareler): bulundu %{100 * c['goal_found'][c['goal_visible']].mean():.0f}")
    if "refined" in c:
        print(f"çizgilere oturtma uygulanan: kalibre karelerin %{100 * c['refined'][calibrated].mean():.1f}'i")
    print("kategoriler:")
    for code in CATEGORIES:
        if cnt[code]:
            print(f"  {code:4s} %{100 * cnt[code] / n:5.1f}  {CATEGORIES[code]}")
    bad = [s for s in segments(cats, fps) if s[0] in ("E4!", "E5!") and s[2] - s[1] >= args.min_seg]
    if bad:
        print("şüpheli (kalibre ama çizgilerle uyumsuz) bölümler:")
        for code, a, b in bad:
            print(f"  {a:6.1f}-{b:6.1f} sn  {code}")

    report = {"frames": n, "fps": fps, "calibrated_ratio": float(calibrated.mean()),
              "categories": {k: int(v) for k, v in cnt.items()},
              "line_score_median": float(np.median(ls)) if len(ls) else None,
              "suspicious_segments": [{"code": k, "start_s": a, "end_s": b} for k, a, b in bad],
              "segments": [{"code": k, "start_s": round(a, 2), "end_s": round(b, 2)} for k, a, b in segments(cats, fps)]}
    (npz.parent / "calibration_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                                        encoding="utf-8")
    timeline(c, cats, fps, npz.parent / "calibration_timeline.png")
    print(f"[çıktı] {npz.parent / 'calibration_report.json'}")
    print(f"[çıktı] {npz.parent / 'calibration_timeline.png'}")


if __name__ == "__main__":
    main()
