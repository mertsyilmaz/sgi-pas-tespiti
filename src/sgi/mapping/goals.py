"""Kalibrasyon yönlendirmeli klasik kale tespiti (D-026). Model eğitimi yoktur.

Kalibrasyon (H) direk diplerinin görüntüdeki yerini zaten verir. Tüm görüntüde aramak yerine:
1. Her direk dibinin üstünde, beklenen direk yüksekliği kadar dar bir dikey şeritte beyaz piksel maskesi
   (yüksek parlaklık, düşük doygunluk) çıkarılır.
2. Şeritteki en uzun dikey beyaz parça direktir: üst ucu direk tepesidir.
3. Beklenen direk boyu, kale genişliğinin görüntüdeki boyuna oranla (2.44 / 7.32) kestirilir; bulunan boy
   bununla tutarlı değilse tespit reddedilir.
4. İki direk tepesi arasında beyaz piksel oranı (üst direk) ölçülür.
Sonuç: direk dipleri/tepeleri (görüntü), tutarlılık skoru. Direk tepeleri 3B noktalardır (z = 2.44 m);
ileride tam kamera modeli (top yüksekliği) için kullanılabilir.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from sgi.mapping.homography import pitch_to_image

GOAL_WIDTH, GOAL_HEIGHT = 7.32, 2.44


@dataclass
class GoalDetection:
    side: str                                         # left | right
    bases: list[tuple[float, float]]                  # iki direk dibi (görüntü, H'den)
    tops: list[Optional[tuple[float, float]]] = field(default_factory=list)   # bulunan direk tepeleri
    crossbar_ratio: float = 0.0                       # üst direk boyunca beyaz piksel oranı
    height_ratio: float = float("nan")                # bulunan direk boyu / beklenen
    found: bool = False


def white_mask(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return ((hsv[..., 2] > 170) & (hsv[..., 1] < 60)).astype(np.uint8)


class GoalDetector:
    def __init__(self, image_size: tuple[int, int], pitch_length: float = 105.0, pitch_width: float = 68.0,
                 strip_half_px: int = 6, min_height_ratio: float = 0.5, max_height_ratio: float = 2.0) -> None:
        self.w, self.h = image_size
        cy = pitch_width / 2
        self.posts = {"left": [(0.0, cy - GOAL_WIDTH / 2), (0.0, cy + GOAL_WIDTH / 2)],
                      "right": [(pitch_length, cy - GOAL_WIDTH / 2), (pitch_length, cy + GOAL_WIDTH / 2)]}
        self.strip, self.min_r, self.max_r = strip_half_px, min_height_ratio, max_height_ratio

    def _project(self, G: np.ndarray, pts) -> Optional[np.ndarray]:
        p = np.c_[np.asarray(pts), np.ones(len(pts))] @ G.T
        if np.any(p[:, 2] <= 1e-9):
            return None
        return p[:, :2] / p[:, 2:3]

    def detect(self, frame: np.ndarray, H_img2pitch: Optional[np.ndarray]) -> list[GoalDetection]:
        if H_img2pitch is None:
            return []
        G = pitch_to_image(H_img2pitch, (self.w, self.h))
        out = []
        for side, posts in self.posts.items():
            bases = self._project(G, posts)
            if bases is None or not all(-20 <= x < self.w + 20 and 0 <= y < self.h + 20 for x, y in bases):
                continue                                       # kale kadrajda değil
            det = GoalDetection(side, [tuple(b) for b in bases])
            tops, ratios = [], []
            for (px, py), (bx, by) in zip(posts, bases):
                # beklenen direk boyu: dibin çevresinde 1 m'nin görüntüdeki en uzun izdüşümü (yerel ölçek) x 2.44.
                # Kale genişliği perspektifte çok kısalabildiği için ölçek ondan alınmaz.
                nb = self._project(G, [(px + 1.0, py), (px, py + 1.0), (px - 1.0, py), (px, py - 1.0)])
                ppm = 0.0 if nb is None else float(np.max(np.linalg.norm(nb - [bx, by], axis=1)))
                exp_h = ppm * GOAL_HEIGHT
                if exp_h < 8:
                    tops.append(None)
                    ratios.append(float("nan"))
                    continue
                top, ratio = self._find_post(frame, bx, by, exp_h)
                tops.append(top)
                ratios.append(ratio)
            det.tops = tops
            if all(t is not None for t in tops):
                det.height_ratio = float(np.mean(ratios))
                det.crossbar_ratio = self._crossbar(frame, tops[0], tops[1])
                det.found = det.crossbar_ratio >= 0.5
            out.append(det)
        return out

    def _find_post(self, frame, bx, by, exp_h) -> tuple[Optional[tuple[float, float]], float]:
        """Direk dibinden yukarı dar şeritte en uzun kesintisiz beyaz dikey parçayı bul."""
        x0, x1 = int(bx - self.strip), int(bx + self.strip + 1)
        y_top, y_bot = int(by - self.max_r * exp_h), int(by + 4)
        if x0 < 0 or x1 > self.w or y_bot <= 0:
            return None, float("nan")
        y_top = max(0, y_top)
        roi = white_mask(frame[y_top:y_bot, x0:x1])
        if roi.size == 0:
            return None, float("nan")
        col = roi.any(axis=1)                       # şeridin her satırında beyaz var mı
        # alttan (dipten) başlayıp yukarı doğru, küçük boşluklara (≤ 3 px) izin vererek ilerle
        gap, top_row = 0, None
        for r in range(len(col) - 1, -1, -1):
            if col[r]:
                top_row, gap = r, 0
            else:
                gap += 1
                if gap > 3 and top_row is not None:
                    break
        if top_row is None:
            return None, float("nan")
        length = y_bot - (y_top + top_row)
        ratio = length / exp_h
        if not (self.min_r <= ratio <= self.max_r):
            return None, ratio
        xs = np.nonzero(roi[top_row])[0]
        return (float(x0 + xs.mean()), float(y_top + top_row)), ratio

    def _crossbar(self, frame, t0, t1, n: int = 40) -> float:
        """İki direk tepesi arasındaki doğru üzerinde beyaz piksel oranı (±2 px tolerans).
        Yalnızca doğruyu çevreleyen küçük bölge işlenir (tam kare maskesi ~10 ms sürüyordu)."""
        x0 = int(max(0, min(t0[0], t1[0]) - 4)); x1 = int(min(self.w, max(t0[0], t1[0]) + 5))
        y0 = int(max(0, min(t0[1], t1[1]) - 4)); y1 = int(min(self.h, max(t0[1], t1[1]) + 5))
        if x1 <= x0 or y1 <= y0:
            return 0.0
        wm = cv2.dilate(white_mask(frame[y0:y1, x0:x1]), np.ones((5, 5), np.uint8))
        xs = np.linspace(t0[0], t1[0], n).astype(int) - x0
        ys = np.linspace(t0[1], t1[1], n).astype(int) - y0
        ok = (xs >= 0) & (xs < wm.shape[1]) & (ys >= 0) & (ys < wm.shape[0])
        return float(wm[ys[ok], xs[ok]].mean()) if ok.any() else 0.0
