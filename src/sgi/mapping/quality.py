"""Etiketsiz videoda kalibrasyon kalite göstergeleri (14_CALIBRATION_QUALITY §2.2, D-019).

Hepsi klasik görüntü işleme yöntemleridir ve yer gerçeği (ground truth) gerektirmez:
- Çim maskesi: HSV'de videoya uyarlanan yeşil eşik + morfoloji -> yeşil oranı (saha görünüyor mu?)
- Beyaz çizgi maskesi: parlaklığın yerel medyandan farkı + düşük doygunluk, yalnızca saha içinde
- Çizgi hizalama skoru: saha çizgileri H ile görüntüye izdüşürülür; örnek noktaların gerçek beyaz
  çizgiye `tol_px` pikselden yakın olma oranı. Izgaranın gerçek çizgilere oturup oturmadığını ölçer.
- Saha/çim IoU: izdüşürülen saha poligonu ile çim maskesinin örtüşmesi (aynalanmış/kaymış ızgara)
Hesaplar küçültülmüş görüntüde (varsayılan 640 px genişlik) yapılır; kare başına ~15 ms.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from sgi.mapping.homography import pitch_to_image


def pitch_line_samples(length: float = 105.0, width: float = 68.0, step: float = 0.5) -> np.ndarray:
    """FIFA saha çizgileri üzerinde örnek noktalar (metre, configs/pitch.yaml sistemi)."""
    cy = width / 2
    segs = [((0, 0), (length, 0)), ((0, width), (length, width)), ((0, 0), (0, width)),
            ((length, 0), (length, width)), ((length / 2, 0), (length / 2, width))]
    for x0, sgn in ((0.0, 1), (length, -1)):
        for depth, half in ((16.5, 20.16), (5.5, 9.16)):          # ceza sahası, kale alanı
            x1 = x0 + sgn * depth
            segs += [((x0, cy - half), (x1, cy - half)), ((x1, cy - half), (x1, cy + half)),
                     ((x0, cy + half), (x1, cy + half))]
    pts = []
    for (ax, ay), (bx, by) in segs:
        n = max(2, int(np.hypot(bx - ax, by - ay) / step))
        t = np.linspace(0, 1, n)
        pts.append(np.c_[ax + (bx - ax) * t, ay + (by - ay) * t])
    r = 9.15
    a = np.linspace(0, 2 * np.pi, int(2 * np.pi * r / step))
    pts.append(np.c_[length / 2 + r * np.cos(a), cy + r * np.sin(a)])  # orta yuvarlak
    left_arc = np.c_[11.0 + r * np.cos(a), cy + r * np.sin(a)]               # ceza yayları: yalnızca
    right_arc = np.c_[length - 11.0 + r * np.cos(a), cy + r * np.sin(a)]     # ceza sahası dışında kalan kısım
    pts += [left_arc[left_arc[:, 0] > 16.5], right_arc[right_arc[:, 0] < length - 16.5]]
    return np.vstack(pts)


@dataclass
class FrameQuality:
    green_ratio: float
    line_score: float = float("nan")     # NaN: kalibrasyon yok veya yeterli görünür çizgi örneği yok
    pitch_iou: float = float("nan")
    n_line_samples: int = 0


class QualityMonitor:
    def __init__(self, image_size: tuple[int, int], pitch_length: float = 105.0, pitch_width: float = 68.0,
                 work_width: int = 640, tol_px: float = 2.7, min_samples: int = 40) -> None:
        w, h = image_size
        self.full_size = (w, h)
        self.s = min(1.0, work_width / w)
        self.size = (int(round(w * self.s)), int(round(h * self.s)))
        self.S = np.diag([self.s, self.s, 1.0])
        self.tol, self.min_samples = tol_px, min_samples
        self.lines = pitch_line_samples(pitch_length, pitch_width)
        t = np.linspace(0, 1, 30)
        L, W = pitch_length, pitch_width
        self.border = np.vstack([np.c_[L * t, 0 * t], np.c_[L + 0 * t, W * t], np.c_[L * (1 - t), W + 0 * t],
                                 np.c_[0 * t, W * (1 - t)]])
        self._hue: Optional[float] = None        # videoya uyarlanan çim tonu
        from sgi.mapping.goals import GoalDetector
        self.goal_detector = GoalDetector(image_size, pitch_length, pitch_width)   # D-026
        self._k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        self._k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def masks(self, frame: np.ndarray, fast: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(yeşil, saha, beyaz çizgi) maskeleri, küçültülmüş boyutta (`self.size`).
        `fast`: yerel arka plan medyanı yarım çözünürlükte (canlı modda çizgilere oturtma için, ~8 ms kazanç)."""
        return self._masks(frame, fast)

    def _masks(self, frame: np.ndarray, fast: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        small = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        cand = (sat > 40) & (val > 40) & (hue >= 25) & (hue <= 95)
        if cand.mean() > 0.15:                   # yeterince yeşil varsa tonu güncelle (yavaş uyarlama)
            peak = float(np.argmax(np.bincount(hue[cand], minlength=180)))
            self._hue = peak if self._hue is None else 0.9 * self._hue + 0.1 * peak
        c = self._hue if self._hue is not None else 50.0
        green = ((np.abs(hue.astype(np.int16) - c) <= 15) & (sat > 40) & (val > 40)).astype(np.uint8)
        field = cv2.morphologyEx(cv2.morphologyEx(green, cv2.MORPH_CLOSE, self._k_close), cv2.MORPH_OPEN,
                                 self._k_open)
        # beyaz çizgi: yerel arka plandan (11 px medyan) belirgin parlak, doygunluğu çok yüksek değil, çim üzerinde.
        # Ayar, iki videoda (gündüz 1080p / gece 720p) doğru ve 1.5-3 m kaydırılmış ızgaraları en iyi ayıran
        # parametre taramasıyla seçildi: doğru >= 0.81, kaymış %90'lık 0.43 (D-019).
        if fast:
            half = cv2.resize(val, (self.size[0] // 2, self.size[1] // 2), interpolation=cv2.INTER_AREA)
            bg = cv2.resize(cv2.medianBlur(half, 5), self.size, interpolation=cv2.INTER_LINEAR)
        else:
            bg = cv2.medianBlur(val, 11)
        bright = cv2.subtract(val, bg)
        lines = ((bright > 8) & (sat < 160) & (field > 0)).astype(np.uint8)
        return green, field, lines

    def measure(self, frame: np.ndarray, H_img2pitch: Optional[np.ndarray]) -> FrameQuality:
        green, field, lines = self._masks(frame)
        q = FrameQuality(green_ratio=float(field.mean()))
        if H_img2pitch is None:
            return q
        G = self.S @ pitch_to_image(H_img2pitch, self.full_size)   # saha -> küçük görüntü (önde w > 0)
        w, h = self.size

        p = np.c_[self.lines, np.ones(len(self.lines))] @ G.T
        ok = p[:, 2] > 1e-9
        xy = p[ok, :2] / p[ok, 2:3]
        ins = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
        xy = xy[ins].astype(np.int32)
        xy = xy[field[xy[:, 1], xy[:, 0]] > 0]             # yalnızca çim üzerinde görünür olması gereken örnekler
        q.n_line_samples = len(xy)
        if len(xy) >= self.min_samples and lines.any():
            dist = cv2.distanceTransform((1 - lines).astype(np.uint8), cv2.DIST_L2, 3)
            q.line_score = float(np.mean(dist[xy[:, 1], xy[:, 0]] <= self.tol))

        b = np.c_[self.border, np.ones(len(self.border))] @ G.T
        if np.all(b[:, 2] > 1e-9):
            poly = np.clip(b[:, :2] / b[:, 2:3], -4 * w, 4 * w).astype(np.int32)
            pm = np.zeros_like(field)
            cv2.fillPoly(pm, [poly], 1)
            union = np.logical_or(pm, field).sum()
            q.pitch_iou = float(np.logical_and(pm, field).sum() / union) if union else float("nan")
        return q
