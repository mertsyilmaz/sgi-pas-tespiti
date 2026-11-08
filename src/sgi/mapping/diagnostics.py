"""Kare başına kalibrasyon durumu ve hata kategorisi (14_CALIBRATION_QUALITY §3, D-019).

Girdi: grid_demo `process` çıktısı calibration.npz (kalite göstergeleriyle).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CATEGORIES = {
    "OK":  "kalibre, çizgilerle uyumlu",
    "OK?": "kalibre, çizgi uyumu ölçülemedi (az görünür çizgi)",
    "OK~": "kalibre, çizgi uyumu orta: yaklaşık doğru, ~1-2 m hata olabilir (veya soluk çizgi)",
    "E1":  "saha görünmüyor (yakın çekim, tribün, grafik)",
    "E2":  "yetersiz saha noktası",
    "E3":  "dejenere nokta dağılımı (dar alan)",
    "E4":  "tutarsız çözüm (fiziksel kontrol / ani sıçrama) - reddedildi",
    "E4!": "kalibre AMA çizgilerle uyumsuz (saha noktalarından) - muhtemel yanlış ızgara",
    # E5! aşağıda: kamera hareketiyle taşınmış ve uyumsuz
    "E5":  "kamera hareketi taşıması sınırına ulaşıldı",
    "E5!": "kalibre AMA çizgilerle uyumsuz (kamera hareketi kayması)",
    "E6":  "sahne kesmesi / hareket kestirilemedi",
    "E7":  "yüksek reprojeksiyon hatası - reddedildi",
}


@dataclass
class DiagnosticThresholds:
    min_green: float = 0.30      # bunun altı: saha görünmüyor
    min_line_score: float = 0.65  # bunun üstü: iyi (doğru ızgaralar >= 0.81)
    bad_line_score: float = 0.35  # bunun altı: muhtemelen yanlış; arası: yaklaşık (1.5-3 m kaydırılmış p90 0.43)
    # Test videosunda görsel kontrol: 0.35-0.65 aralığında hem ~1-2 m kaymış hem de soluk çizgili doğru kareler var.
    max_chain: int = 75


def _fill_forward(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Kalite her N karede ölçülür; ölçülmeyen karelere son ölçülen değer verilir."""
    out, last = x.copy(), np.nan
    for i in range(len(x)):
        if valid[i]:
            last = x[i]
        else:
            out[i] = last
    return out


def categorize(c, th: DiagnosticThresholds = DiagnosticThresholds()) -> np.ndarray:
    n = len(c["source"])
    codes = [str(x) for x in c["reject_codes"]]
    measured = c["quality_measured"] if "quality_measured" in c else np.ones(n, bool)
    greens = _fill_forward(c["green_ratio"], measured)
    lines = _fill_forward(c["line_score"], measured)
    out = np.empty(n, dtype=object)
    for i in range(n):
        green, line = greens[i], lines[i]
        rej = codes[int(c["reject"][i])]
        if c["source"][i] != 0:                                   # kalibre kare
            if np.isnan(line):
                out[i] = "OK?"
            elif line >= th.min_line_score:
                out[i] = "OK"
            elif line >= th.bad_line_score:
                out[i] = "OK~"
            else:
                out[i] = "E5!" if c["source"][i] == 2 else "E4!"
        elif not np.isnan(green) and green < th.min_green:
            out[i] = "E1"
        elif rej in ("few_keypoints", "ransac", "few_inliers"):
            out[i] = "E2"
        elif rej == "spread":
            out[i] = "E3"
        elif rej.startswith("cam_") or rej == "jump":
            out[i] = "E4"
        elif rej == "reproj":
            out[i] = "E7"
        elif not c["motion_ok"][i]:
            out[i] = "E6"
        elif c["motion_chain"][i] >= th.max_chain or (i > 0 and c["motion_chain"][i - 1] >= th.max_chain - 1):
            out[i] = "E5"
        else:
            out[i] = "E2"
    return out


def segments(labels: np.ndarray, fps: float, min_len: int = 1) -> list[tuple[str, float, float]]:
    """Ardışık aynı etiketli bölümler: (etiket, başlangıç sn, bitiş sn)."""
    segs, i = [], 0
    while i < len(labels):
        j = i
        while j < len(labels) and labels[j] == labels[i]:
            j += 1
        if j - i >= min_len:
            segs.append((str(labels[i]), i / fps, j / fps))
        i = j
    return segs
