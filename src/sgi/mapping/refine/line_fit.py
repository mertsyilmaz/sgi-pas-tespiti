"""Çizgilere oturtma: homografinin görüntüdeki beyaz saha çizgileriyle ince ayarı (D-020).

Chamfer eşleme:
1. Görüntüde beyaz saha çizgisi maskesi (çim üzerinde) çıkarılır ve uzaklık dönüşümü alınır:
   her pikselin en yakın çizgi pikseline uzaklığı.
2. Saha modelinin çizgi örnekleri (ceza sahaları, orta çizgi, yuvarlaklar, kenarlar) mevcut H ile
   görüntüye izdüşürülür.
3. Görüntü tarafında küçük bir homografi düzeltmesi D aranır: D ile kaydırılan örneklerin çizgiye
   uzaklıkları en aza indirilir. D, 4 kontrol noktasının yer değiştirmesiyle (8 parametre) ifade edilir.
   Sağlam (soft-L1) kayıp kullanılır, uzaklık kırpılır. Yalnızca tek yönde çizgi göründüğünde çizgi boyunca
   kayma belirsiz kalır; bu yüzden düzeltme sıfıra doğru düzenlileştirilir.
4. Sonuç ancak şu üç koşulla kabul edilir: çizgi uyumu yeterince artmış, düzeltme küçük (metre cinsinden
   sınırlı) ve fiziksel kamera kontrolünden geçiyor.
Sonuç: H' = H · D⁻¹ (görüntü -> görüntü düzeltmesi, sonra saha).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from sgi.interfaces import PitchRefiner
from sgi.mapping.homography import apply_h, camera_plausibility, pitch_to_image
from sgi.mapping.quality import QualityMonitor, pitch_line_samples

import cv2


class LineFitRefiner(PitchRefiner):
    def __init__(self, image_size: tuple[int, int], pitch_length: float = 105.0, pitch_width: float = 68.0,
                 work_width: int = 640, max_dist_px: float = 12.0, prior_px: float = 15.0,
                 tol_px: float = 2.7, min_samples: int = 60, min_gain: float = 0.03,
                 max_correction_m: float = 4.0, camera_check: bool = True, fast_mask: bool = False,
                 max_nfev: int = 60, **_: object) -> None:
        self.fast_mask, self.max_nfev = fast_mask, max_nfev
        self.w, self.h = image_size
        self.masker = QualityMonitor(image_size, pitch_length, pitch_width, work_width=work_width)
        self.s = self.masker.s
        self.sw, self.sh = self.masker.size
        self.samples = pitch_line_samples(pitch_length, pitch_width, step=0.5)
        self.max_dist, self.prior, self.tol = max_dist_px, prior_px, tol_px
        self.min_samples, self.min_gain = min_samples, min_gain
        self.max_corr_m, self.camera_check = max_correction_m, camera_check
        m = 0.15
        self.ctrl = np.array([[m * self.sw, m * self.sh], [(1 - m) * self.sw, m * self.sh],
                              [(1 - m) * self.sw, (1 - m) * self.sh], [m * self.sw, (1 - m) * self.sh]], np.float32)
        # karşılaştırma için tam çözünürlükte kontrol noktaları (alt 2/3)
        xs, ys = np.linspace(0.1 * self.w, 0.9 * self.w, 4), np.linspace(0.4 * self.h, 0.95 * self.h, 3)
        self._cmp = np.array([(x, y) for y in ys for x in xs])

    def _D(self, p: np.ndarray) -> np.ndarray:
        """4 kontrol noktası -> kaydırılmış konumları homografisi (64 bit DLT; OpenCV'nin 32 bitlik
        getPerspectiveTransform'u sayısal türev adımlarını yutuyor)."""
        src = self.ctrl.astype(np.float64)
        dst = src + p.reshape(4, 2)
        A, b = [], []
        for (x, y), (u, v) in zip(src, dst):
            A += [[x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y]]
            b += [u, v]
        h = np.linalg.solve(np.array(A), np.array(b))
        return np.append(h, 1.0).reshape(3, 3)

    def refine(self, frame_bgr: np.ndarray, H_img2pitch: np.ndarray) -> tuple[np.ndarray, bool, dict]:
        info: dict = {"reason": ""}
        _, field, lines = self.masker.masks(frame_bgr, fast=self.fast_mask)
        if not lines.any():
            info["reason"] = "no_lines"
            return H_img2pitch, False, info
        dist = cv2.distanceTransform((1 - lines).astype(np.uint8), cv2.DIST_L2, 5)
        dist = np.minimum(dist, self.max_dist).astype(np.float32)

        S = np.diag([self.s, self.s, 1.0])
        G = S @ pitch_to_image(H_img2pitch, (self.w, self.h))           # saha -> küçük görüntü
        p = np.c_[self.samples, np.ones(len(self.samples))] @ G.T
        ok = p[:, 2] > 1e-9
        xy = p[ok, :2] / p[ok, 2:3]
        pad = 20
        keep = (xy[:, 0] >= pad) & (xy[:, 0] < self.sw - pad) & (xy[:, 1] >= pad) & (xy[:, 1] < self.sh - pad)
        xy = xy[keep]
        if len(xy):
            xi, yi = xy[:, 0].astype(int), xy[:, 1].astype(int)
            xy = xy[field[yi, xi] > 0]                                    # yalnızca çim üzerindeki örnekler
        info["n_samples"] = len(xy)
        if len(xy) < self.min_samples:
            info["reason"] = "few_samples"
            return H_img2pitch, False, info

        pts = xy.astype(np.float32).reshape(-1, 1, 2)

        def sample_dist(q: np.ndarray) -> np.ndarray:
            # tam hassasiyetli bilineer örnekleme (cv2.remap alt pikseli 1/32'ye yuvarlar, türevi sıfırlar)
            x, y = q[:, 0, 0], q[:, 0, 1]
            out = np.full(len(x), self.max_dist)
            ok = (x >= 0) & (y >= 0) & (x < self.sw - 1) & (y < self.sh - 1)
            x, y = x[ok], y[ok]
            x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
            fx, fy = x - x0, y - y0
            out[ok] = (dist[y0, x0] * (1 - fx) * (1 - fy) + dist[y0, x0 + 1] * fx * (1 - fy)
                       + dist[y0 + 1, x0] * (1 - fx) * fy + dist[y0 + 1, x0 + 1] * fx * fy)
            return out

        def residuals(params: np.ndarray) -> np.ndarray:
            q = cv2.perspectiveTransform(pts.astype(np.float64), self._D(params))
            return np.r_[sample_dist(q), params / self.prior * np.sqrt(len(pts) / 8.0)]

        before = float(np.mean(sample_dist(pts) <= self.tol))
        sol = least_squares(residuals, np.zeros(8), loss="soft_l1", f_scale=2.0, max_nfev=self.max_nfev, x_scale=5.0,
                            diff_step=0.02)
        D = self._D(sol.x)
        after = float(np.mean(sample_dist(cv2.perspectiveTransform(pts, D)) <= self.tol))
        info.update(score_before=before, score_after=after)

        # küçük görüntüdeki düzeltmeyi tam çözünürlüğe taşı: D_full = S⁻¹ D S
        D_full = np.linalg.inv(S) @ D @ S
        H_new = H_img2pitch @ np.linalg.inv(D_full)
        H_new = H_new / H_new[2, 2]
        corr = float(np.median(np.linalg.norm(apply_h(H_new, self._cmp) - apply_h(H_img2pitch, self._cmp), axis=1)))
        info["correction_m"] = corr

        if after < before + self.min_gain:
            info["reason"] = "no_gain"
        elif corr > self.max_corr_m:
            info["reason"] = "too_large"
        elif self.camera_check and not camera_plausibility(H_new, (self.w, self.h)).ok:
            info["reason"] = "implausible"
        else:
            return H_new, True, info
        return H_img2pitch, False, info
