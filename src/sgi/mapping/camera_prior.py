"""Sabit kamera konumu önseli (D-027).

Yayının ana kamerası tribünde sabit bir yerde durur; maç boyunca yalnızca döner (yatay/dikey açı) ve zoom
yapar. Görüntü değişir ama kameranın sahadaki 3B konumu C değişmez.

Pinhole model: saha düzlemi (z = 0) -> görüntü homografisi  G = K [r1 r2 t],  t = -R C,  K = diag(f, f, 1)
(asal nokta görüntü merkezi). Serbest homografinin 8 bilinmeyeni vardır (>= 4, pratikte 8 sağlam nokta gerekir).
C bilinirse bilinmeyenler 4'e iner: dönüş (3) + odak (1). Her nokta 2 denklem verir, 2-3 nokta yeter.
Bu, saha noktasının az olduğu orta saha görüntülerinde kalibrasyonu mümkün kılar.

C, iyi kalibre edilmiş karelerin homografilerinden ayrıştırılır ve ortanca alınır (sağlam).
Başka bir kameraya geçildiğinde (tekrar, kale arkası) kestirilen C uyuşmaz; öncel o karede kullanılmaz.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from scipy.optimize import least_squares

from sgi.mapping.homography import camera_plausibility
from sgi.types import PitchKeypoint


@dataclass
class CameraPose:
    f: float
    R: np.ndarray          # dünya -> kamera
    C: np.ndarray          # kamera merkezi (saha koordinatı, metre; z ekseni x × y yönünde)


def _K(f: float, size: tuple[int, int]) -> np.ndarray:
    w, h = size
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])


def decompose(H_img2pitch: np.ndarray, size: tuple[int, int]) -> Optional[CameraPose]:
    """image->pitch homografisinden (f, R, C). Fiziksel olarak tutarsızsa None."""
    chk = camera_plausibility(H_img2pitch, size)
    if not chk.ok:
        return None
    f = chk.focal_px
    G = np.linalg.inv(_K(f, size)) @ np.linalg.inv(H_img2pitch)
    s = 2.0 / (np.linalg.norm(G[:, 0]) + np.linalg.norm(G[:, 1]))
    G = G * s
    # işaret: görüntü merkezine karşılık gelen saha noktası kameranın önünde olmalı (kamera z > 0)
    X = H_img2pitch @ np.array([size[0] / 2, size[1] / 2, 1.0])
    X = X / X[2]
    if (G @ np.array([X[0], X[1], 1.0]))[2] < 0:
        G = -G
    r1, r2, t = G[:, 0], G[:, 1], G[:, 2]
    U, _, Vt = np.linalg.svd(np.column_stack([r1, r2, np.cross(r1, r2)]))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        return None
    return CameraPose(f, R, -R.T @ t)


def pose_to_h(f: float, R: np.ndarray, C: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    t = -R @ C
    G = _K(f, size) @ np.column_stack([R[:, 0], R[:, 1], t])
    H = np.linalg.inv(G)
    return H / H[2, 2]


def _project(f, rvec, C, pts_xy, size):
    R, _ = cv2.Rodrigues(rvec)
    Xc = (np.c_[pts_xy, np.zeros(len(pts_xy))] - C) @ R.T
    z = Xc[:, 2]
    w, h = size
    return np.c_[f * Xc[:, 0] / z + w / 2, f * Xc[:, 1] / z + h / 2], z


class CameraPositionPrior:
    """İyi karelerden C'yi öğrenir; az noktalı karelerde C sabitken (dönüş + odak) çözer."""

    def __init__(self, image_size: tuple[int, int], min_samples: int = 8, max_spread_m: float = 6.0,
                 min_points: int = 3, max_rms_px: float = 4.0, inlier_px: float = 8.0, **_: object) -> None:
        self.size = image_size
        self.min_samples, self.max_spread = min_samples, max_spread_m
        self.min_points, self.max_rms, self.inlier_px = min_points, max_rms_px, inlier_px
        self._Cs: list[np.ndarray] = []
        self.C: Optional[np.ndarray] = None
        self.last_pose: Optional[CameraPose] = None

    # --- öğrenme ---
    def observe(self, H_img2pitch: np.ndarray) -> None:
        """Güvenilir (çok noktalı, kontrolden geçmiş) bir kalibrasyondan kamera konumu örneği ekle."""
        pose = decompose(H_img2pitch, self.size)
        if pose is None:
            return
        self.last_pose = pose
        self._Cs.append(pose.C)
        if len(self._Cs) >= self.min_samples:
            Cs = np.array(self._Cs[-200:])
            med = np.median(Cs, axis=0)
            spread = np.median(np.linalg.norm(Cs - med, axis=1))
            self.C = med if spread <= self.max_spread else None

    @property
    def ready(self) -> bool:
        return self.C is not None

    def spread(self) -> float:
        if len(self._Cs) < 2:
            return float("nan")
        Cs = np.array(self._Cs[-200:])
        return float(np.median(np.linalg.norm(Cs - np.median(Cs, axis=0), axis=1)))

    def matches(self, H_img2pitch: np.ndarray, tol_m: float = 10.0) -> Optional[bool]:
        """Bu kalibrasyon aynı kameradan mı (C öğrenilen konuma yakın mı)? Ayrıştırılamazsa None."""
        if self.C is None:
            return None
        pose = decompose(H_img2pitch, self.size)
        return None if pose is None else bool(np.linalg.norm(pose.C - self.C) <= tol_m)

    # --- az noktayla çözüm ---
    def solve(self, kps: list[PitchKeypoint], init: Optional[CameraPose] = None) -> Optional[tuple[np.ndarray, int, float]]:
        """C sabit, (dönüş, odak) bilinmeyen. Dönüş: (H, inlier sayısı, rms px) ya da None."""
        if self.C is None or len(kps) < self.min_points:
            return None
        img = np.array([k.image_xy for k in kps], float)
        pit = np.array([k.pitch_xy for k in kps], float)
        w, _ = self.size
        starts = []
        if init is not None:
            starts.append((cv2.Rodrigues(init.R)[0].ravel(), init.f))
        if self.last_pose is not None:
            starts.append((cv2.Rodrigues(self.last_pose.R)[0].ravel(), self.last_pose.f))
        # bakış yönü: noktaların ağırlık merkezine bakan, yatay eksen saha x'ine paralel kamera
        tgt = np.r_[pit.mean(axis=0), 0.0]
        fwd = (tgt - self.C) / np.linalg.norm(tgt - self.C)
        right = np.cross(fwd, [0, 0, 1.0]) if abs(fwd[2]) < 0.99 else np.array([1.0, 0, 0])
        right /= np.linalg.norm(right)
        R0 = np.vstack([right, np.cross(fwd, right), fwd])
        if np.linalg.det(R0) < 0:
            R0[0] *= -1
        for f0 in (1.0 * w, 2.0 * w, 4.0 * w):
            for flip in (1, -1):
                R1 = R0.copy()
                R1[:2] *= flip
                if np.linalg.det(R1) > 0:
                    starts.append((cv2.Rodrigues(R1)[0].ravel(), f0))

        best = None
        for rvec0, f0 in starts:
            def res(p):
                uv, z = _project(p[3], p[:3], self.C, pit, self.size)
                r = (uv - img).ravel()
                return np.where(np.repeat(z, 2) > 0, r, 1e3)
            try:
                sol = least_squares(res, np.r_[rvec0, f0], loss="soft_l1", f_scale=3.0, max_nfev=200,
                                    x_scale=np.r_[0.1, 0.1, 0.1, w * 0.2])
            except Exception:
                continue
            uv, z = _project(sol.x[3], sol.x[:3], self.C, pit, self.size)
            err = np.linalg.norm(uv - img, axis=1)
            inl = (err <= self.inlier_px) & (z > 0)
            if inl.sum() < self.min_points:
                continue
            rms = float(np.sqrt(np.mean(err[inl] ** 2)))
            key = (-int(inl.sum()), rms)
            if best is None or key < best[0]:
                best = (key, sol.x, int(inl.sum()), rms)
        if best is None or best[3] > self.max_rms:
            return None
        _, x, n_in, rms = best
        R, _ = cv2.Rodrigues(x[:3])
        H = pose_to_h(x[3], R, self.C, self.size)
        if not camera_plausibility(H, self.size).ok:
            return None
        return H, n_in, rms
