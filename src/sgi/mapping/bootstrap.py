"""Az noktadan başlangıç kalibrasyonu: noktalar + görünen beyaz çizgiler + fiziksel kamera (D-034).

Orta saha görüntülerinde saha modeli karede 0–4 nokta bulur (görünen: uzak taç çizgisi, orta yuvarlak parçası).
Serbest homografi (8 bilinmeyen) bu kadar noktadan güvenilir çıkmaz; 4 noktalı çözüm fiziksel kamera
kontrolünden geçemez. Burada bilinmeyenler gerçek bir yayın kamerasının parametreleridir:
dönüş (3) + odak (1) + konum (3). Bulunan noktalar ve görüntüdeki beyaz çizgi maskesi birlikte kullanılır:
  - nokta artığı: izdüşen nokta ile bulunan nokta arası (px)
  - çizgi artığı: izdüşen saha çizgisi örnekleri (orta yuvarlak ve ceza yayları dahil) ile en yakın beyaz piksel
  - çim artığı: görüntüdeki çim pikselleri saha sınırları içine düşmeli (çözümün sahayı kadraj dışına itmesini önler)
Başlangıç: yayın kamerası taç çizgisinin gerisinde ve yüksekte (birkaç aday konum), noktalara bakar.
Kabul: fiziksel kamera kontrolü + çizgi uyumu + saha/çim örtüşmesi eşikleri. Nedenseldir (yalnızca bu kare).
"""
from __future__ import annotations

from dataclasses import replace
from typing import Optional

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares

from sgi.mapping.camera_prior import _project, pose_to_h
from sgi.mapping.homography import apply_h, camera_plausibility
from sgi.mapping.quality import QualityMonitor, pitch_line_samples
from sgi.types import HomographyResult, PitchKeypoint


class LineCameraBootstrap:
    def __init__(self, image_size: tuple[int, int], pitch_length: float = 105.0, pitch_width: float = 68.0,
                 min_points: int = 3, min_line_score: float = 0.85, min_iou: float = 0.85,
                 max_point_px: float = 12.0, trunc_px: float = 15.0, min_gap: int = 5, camera_prior=None,
                 **_: object) -> None:
        # min_gap: ızgara kayıpken en fazla bu kadar karede bir denenir (çözüm ~0,5–1 sn CPU)
        self.min_gap, self.prior = min_gap, camera_prior
        self._last_try = -10 ** 9
        self.n_accepted = 0
        self.size = image_size
        self.L, self.W = pitch_length, pitch_width
        self.min_points, self.min_line_score, self.min_iou = min_points, min_line_score, min_iou
        self.max_point_px, self.trunc = max_point_px, trunc_px
        self.qm = QualityMonitor(image_size, pitch_length, pitch_width)
        self.samples = pitch_line_samples(pitch_length, pitch_width, step=1.0)
        self.last_info: dict = {}

    # --- başlangıç adayları ---
    def _starts(self, pit: np.ndarray):
        w, _ = self.size
        tgt = np.r_[pit.mean(axis=0), 0.0]
        for cx in (self.L / 2,):
            for back in (25.0, 45.0, 65.0):
                for hgt in (12.0, 22.0, 35.0):
                    for zs in (1.0, -1.0):                 # saha z ekseninin yönü (x × y) belirsiz: ikisi de denenir
                        C = np.array([cx, self.W + back, zs * hgt])
                        fwd = (tgt - C) / np.linalg.norm(tgt - C)
                        right = np.cross(fwd, [0, 0, zs])
                        right /= np.linalg.norm(right)
                        R0 = np.vstack([right, np.cross(fwd, right), fwd])
                        if np.linalg.det(R0) < 0:
                            R0[0] *= -1
                        for f0 in (1.5 * w, 3.0 * w):
                            yield cv2.Rodrigues(R0)[0].ravel(), f0, C

    def solve(self, frame: np.ndarray, kps: list[PitchKeypoint]) -> Optional[np.ndarray]:
        """image->pitch homografisi ya da None. `last_info` neden/skorları taşır."""
        self.last_info = {"reason": "few_points"}
        if len(kps) < self.min_points:
            return None
        img = np.array([k.image_xy for k in kps], float)
        pit = np.array([k.pitch_xy for k in kps], float)
        green, _, lines = self.qm.masks(frame)
        sc = lines.shape[1] / self.size[0]
        dt = cv2.distanceTransform((lines == 0).astype(np.uint8), cv2.DIST_L2, 3) / sc   # tam çözünürlük px
        gy, gx = np.nonzero(green)
        if len(gx) < 50:
            self.last_info = {"reason": "no_grass"}
            return None
        sel = np.random.default_rng(0).choice(len(gx), size=min(300, len(gx)), replace=False)
        grass = np.c_[gx[sel], gy[sel]] / sc
        w, h = self.size
        n_line = len(self.samples)

        def residuals(p, trunc):
            rvec, f, C = p[:3], p[3], p[4:7]
            uv, z = _project(f, rvec, C, pit, self.size)
            r_kp = np.where(np.repeat(z > 0, 2), (uv - img).ravel(), 1e3)
            uvl, zl = _project(f, rvec, C, self.samples, self.size)
            vis = (zl > 0) & (uvl[:, 0] >= 0) & (uvl[:, 0] < w) & (uvl[:, 1] >= 0) & (uvl[:, 1] < h)
            d = np.zeros(n_line)
            if vis.any():
                d[vis] = np.minimum(map_coordinates(dt, [uvl[vis, 1] * sc, uvl[vis, 0] * sc], order=1, mode="nearest"),
                                    trunc)
            r_ln = d * (1.0 / np.sqrt(max(vis.sum(), 1) / 20.0))   # görünen örnek sayısından bağımsız ağırlık
            H = pose_to_h(f, cv2.Rodrigues(rvec)[0], C, self.size)
            q = apply_h(H, grass)
            out_x = np.maximum(0, -q[:, 0]) + np.maximum(0, q[:, 0] - self.L)
            out_y = np.maximum(0, -q[:, 1]) + np.maximum(0, q[:, 1] - self.W)
            r_gr = np.minimum(out_x + out_y, 30.0) * 0.5
            return np.r_[r_kp, r_ln, r_gr]

        # 1. aşama: konum adayda sabit, yalnızca dönüş + odak noktalara oturtulur (4 bilinmeyen, >= 2 nokta)
        stage1 = []
        for rvec0, f0, C0 in self._starts(pit):
            def r_kp(q, C0=C0):
                uv, z = _project(q[3], q[:3], C0, pit, self.size)
                return np.where(np.repeat(z > 0, 2), (uv - img).ravel(), 1e3)
            try:
                s1 = least_squares(r_kp, np.r_[rvec0, f0], loss="soft_l1", f_scale=5.0, max_nfev=100,
                                   x_scale=np.r_[0.05, 0.05, 0.05, 0.2 * w])
            except Exception:                          # noqa: BLE001
                continue
            stage1.append((s1.cost, np.r_[s1.x, C0]))
        stage1.sort(key=lambda x: x[0])
        # 2. aşama: en iyi birkaç başlangıçtan tüm parametreler, noktalar + çizgiler + çim birlikte
        best = None
        for _, x0 in stage1[:4]:
            try:
                sol = None
                for trunc in (3 * self.trunc, self.trunc):    # kaba -> ince: uzak başlangıçta da çizgi çekimi olsun
                    sol = least_squares(residuals, x0 if sol is None else sol.x, args=(trunc,), loss="soft_l1",
                                        f_scale=5.0, max_nfev=80,
                                        x_scale=np.r_[0.05, 0.05, 0.05, 0.2 * w, 5, 5, 5], diff_step=1e-3)
            except Exception:                          # noqa: BLE001
                continue
            if best is None or sol.cost < best.cost:
                best = sol
        if best is None:
            self.last_info = {"reason": "no_solution"}
            return None
        p = best.x
        H = pose_to_h(p[3], cv2.Rodrigues(p[:3])[0], p[4:7], self.size)
        uv, _ = _project(p[3], p[:3], p[4:7], pit, self.size)
        kp_err = float(np.median(np.linalg.norm(uv - img, axis=1)))
        q = self.qm.measure(frame, H)
        cam = camera_plausibility(H, self.size)
        self.last_info = {"kp_err_px": kp_err, "line_score": q.line_score, "iou": q.pitch_iou, "camera": cam.code,
                          "C": p[4:7].round(1).tolist(), "f": round(float(p[3]))}
        ok = (cam.ok and kp_err <= self.max_point_px and q.line_score >= self.min_line_score
              and q.pitch_iou >= self.min_iou)
        self.last_info["reason"] = "" if ok else "rejected"
        return H if ok else None

    def apply(self, meas: Optional[HomographyResult], frame: np.ndarray, lost: bool) -> Optional[HomographyResult]:
        """Izgara kayıpken reddedilmiş (az noktalı) ölçümü bu yöntemle kurtarmayı dener."""
        if (not lost or meas is None or meas.valid or len(meas.keypoints) < self.min_points
                or meas.frame_idx - self._last_try < self.min_gap):
            return meas
        self._last_try = meas.frame_idx
        H = self.solve(frame, meas.keypoints)
        if H is None:
            return meas
        self.n_accepted += 1
        if self.prior is not None:
            self.prior.observe(H)              # tutarlı kamera konumu: sabit kamera önseline de öğret (D-027)
        return replace(meas, H=H, valid=True, reject_reason="", source="keypoints", bootstrapped=True,
                       n_inliers=len(meas.keypoints))
