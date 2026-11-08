"""Saha noktalarından görüntü -> saha homografisi (M4) ve zaman içinde takibi (F-3, D-012)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from sgi.types import CameraMotion, HomographyResult, PitchKeypoint


def pitch_to_image(H_img2pitch: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    """image->pitch homografisinin tersini, kameranın önündeki noktalar w > 0 olacak şekilde döndürür.

    Homografi ölçek (ve işaret) belirsizdir; H / H[2,2] normalizasyonu tersin işaretini değiştirebilir.
    Görüntü merkezi kameranın önündedir: onun saha noktası G ile görüntüye geri giderken w > 0 olmalı.
    """
    w, h = image_size
    X = H_img2pitch @ np.array([w / 2, h / 2, 1.0])
    G = np.linalg.inv(H_img2pitch)
    return G if X[2] > 0 else -G


def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Nx2 noktalara homografi uygular."""
    p = np.c_[pts, np.ones(len(pts))] @ H.T
    return p[:, :2] / p[:, 2:3]


@dataclass
class CameraCheck:
    ok: bool
    reason: str
    focal_px: float = float("nan")
    height_m: float = float("nan")
    aspect: float = float("nan")
    code: str = ""                    # kısa reddetme kodu (cam_nofocal, cam_focal, cam_ortho, cam_aspect, cam_height)


def camera_plausibility(H_img2pitch: np.ndarray, image_size: tuple[int, int],
                        focal_range: tuple[float, float] = (0.3, 15.0), max_aspect_dev: float = 0.25,
                        max_cos: float = 0.15, height_range: tuple[float, float] = (3.0, 120.0)) -> CameraCheck:
    """Homografinin gerçek bir pinhole kameradan gelip gelemeyeceğini sınar (14_CALIBRATION_QUALITY §2.2).

    Saha düzlemi -> görüntü homografisi G = K [r1 r2 t] (ölçek hariç). Asal nokta görüntü merkezi,
    pikseller kare kabul edilir. Gerçek bir kamerada r1 ⟂ r2 ve |r1| = |r2| olmalıdır. Odak uzaklığı
    bu iki koşulun her birinden ayrı çözülebilir. Biri dejenere olabilir: örneğin kamera saha
    uzunluğuna tam dik bakarken x çizgileri görüntüde paraleldir ve diklik denklemi f'yi belirlemez.
    Bu yüzden iki aday denenir, koşulları en iyi sağlayan seçilir, sonra iki koşul da ayrıca sınanır.
    Az sayıda (yanlış) noktaya oturtulan homografiler genelde bu koşullardan birini bozar.
    """
    w, h = image_size
    G = np.linalg.inv(H_img2pitch)
    T = np.array([[1, 0, -w / 2], [0, 1, -h / 2], [0, 0, 1]])   # asal noktayı orijine taşı
    G = T @ G
    G = G / np.abs(G[:, :2]).max()
    g1, g2 = G[:, 0], G[:, 1]

    def measure(f: float) -> tuple[float, float]:
        r1, r2 = g1 / [f, f, 1.0], g2 / [f, f, 1.0]
        n1, n2 = np.linalg.norm(r1), np.linalg.norm(r2)
        return abs(r1 @ r2) / (n1 * n2), n1 / n2

    cands = []
    eps = 1e-300   # yalnızca sıfıra bölmeyi önler; kararsız aday, koşul skoruyla elenir
    if abs(g1[2] * g2[2]) > eps:                                       # r1 ⟂ r2
        f2 = -(g1[0] * g2[0] + g1[1] * g2[1]) / (g1[2] * g2[2])
        if f2 > 0:
            cands.append(np.sqrt(f2))
    den = g1[2] ** 2 - g2[2] ** 2                                      # |r1| = |r2|
    if abs(den) > eps:
        f2 = (g2[0] ** 2 + g2[1] ** 2 - g1[0] ** 2 - g1[1] ** 2) / den
        if f2 > 0:
            cands.append(np.sqrt(f2))
    if not cands:
        if abs(g1[2]) < 1e-6 and abs(g2[2]) < 1e-6:                    # tam tepeden bakış: f belirsiz
            cands.append(float(w))
        else:
            return CameraCheck(False, "odak uzaklığı çözümü yok", code="cam_nofocal")
    f = float(min(cands, key=lambda c: sum(abs(v - t) for v, t in zip(measure(c), (0.0, 1.0)))))
    cos, aspect = measure(f)
    if not (focal_range[0] * w <= f <= focal_range[1] * w):
        return CameraCheck(False, f"odak uzaklığı makul değil ({f:.0f} px)", focal_px=f, code="cam_focal")
    if cos > max_cos:
        return CameraCheck(False, f"saha eksenleri dik değil (cos={cos:.2f})", focal_px=f, aspect=aspect, code="cam_ortho")
    if abs(aspect - 1) > max_aspect_dev:
        return CameraCheck(False, f"saha en-boy bozulması ({aspect:.2f})", focal_px=f, aspect=aspect, code="cam_aspect")
    Kinv = np.diag([1 / f, 1 / f, 1.0])
    r1, r2, t = Kinv @ g1, Kinv @ g2, Kinv @ G[:, 2]
    n1, n2 = np.linalg.norm(r1), np.linalg.norm(r2)
    s = 2.0 / (n1 + n2)
    r1, r2, t = r1 * s, r2 * s, t * s
    r3 = np.cross(r1, r2)
    R = np.column_stack([r1, r2, r3])
    C = -R.T @ t                                    # kamera merkezi (saha koordinatı; z düzleme dik)
    height = float(abs(C[2]))
    if not (height_range[0] <= height <= height_range[1]):
        return CameraCheck(False, f"kamera yüksekliği makul değil ({height:.1f} m)", f, height, aspect, code="cam_height")
    return CameraCheck(True, "ok", f, height, aspect)


@dataclass
class HomographyEstimator:
    min_keypoints: int = 4
    ransac_threshold_m: float = 1.0
    max_reproj_error_m: float = 1.5
    min_spread_m: float = 8.0
    min_inliers: int = 4
    camera_check: bool = False
    image_size: tuple[int, int] = (0, 0)
    camera_prior: object = None           # CameraPositionPrior (D-027); None = kapalı
    prior_learn_inliers: int = 12         # bu kadar inlier'lı geçerli ölçümler kamera konumunu öğretir
    _PRIOR_ELIGIBLE = ("few_keypoints", "few_inliers", "spread", "cam_nofocal", "cam_focal", "cam_ortho",
                       "cam_aspect", "cam_height")

    def estimate(self, frame_idx: int, kps: list[PitchKeypoint]) -> HomographyResult:
        res = self._estimate(frame_idx, kps)
        prior = self.camera_prior
        if prior is None:
            return res
        if res.valid and res.n_inliers >= self.prior_learn_inliers:
            prior.observe(res.H)                       # güvenilir kalibrasyon: kamera konumunu öğret
        elif not res.valid and res.reject_reason in self._PRIOR_ELIGIBLE and prior.ready:
            sol = prior.solve(kps)                     # sabit kamera konumuyla az noktadan çözüm
            if sol is not None:
                res.H, res.n_inliers, res.reproj_error_m = sol[0], sol[1], float("nan")
                res.valid, res.reject_reason, res.used_prior, res.source = True, "", True, "keypoints"
        return res

    def _estimate(self, frame_idx: int, kps: list[PitchKeypoint]) -> HomographyResult:
        """Reddedilirse `reject_reason` kısa bir kod taşır (hata teşhisi, 14_CALIBRATION_QUALITY §3):
        few_keypoints, ransac, few_inliers, reproj, spread, cam_<neden>."""
        res = HomographyResult(frame_idx=frame_idx, H=None, n_keypoints=len(kps), keypoints=kps)
        if len(kps) < self.min_keypoints:
            res.reject_reason = "few_keypoints"
            return res
        img = np.array([k.image_xy for k in kps], dtype=np.float64)
        pit = np.array([k.pitch_xy for k in kps], dtype=np.float64)

        H, mask = cv2.findHomography(img, pit, cv2.RANSAC, self.ransac_threshold_m)
        if H is None:
            res.reject_reason = "ransac"
            return res
        inl = mask.ravel().astype(bool)
        res.n_inliers = int(inl.sum())
        if res.n_inliers < self.min_keypoints:
            res.reject_reason = "few_inliers"
            return res

        err = np.linalg.norm(apply_h(H, img[inl]) - pit[inl], axis=1)
        res.reproj_error_m = float(err.mean())
        spread = np.ptp(pit[inl], axis=0)
        res.H = H / H[2, 2]
        res.source = "keypoints"
        if res.n_inliers < self.min_inliers:
            res.reject_reason = "few_inliers"
        elif res.reproj_error_m > self.max_reproj_error_m:
            res.reject_reason = "reproj"
        elif max(spread) < self.min_spread_m:
            res.reject_reason = "spread"
        elif self.camera_check:
            chk = camera_plausibility(res.H, self.image_size)
            res.reject_reason = "" if chk.ok else chk.code
        res.valid = res.reject_reason == ""
        return res


class HomographyTracker:
    """Kare kare ölçülen homografiyi kamera hareketiyle birleştirir (F-3).

    - Ölçüm varsa: tahminle karıştırılır (yumuşatma). Tahminden çok farklı ve zayıfsa reddedilir.
    - Ölçüm yoksa: bir önceki homografi kamera hareketiyle taşınır: H_t = H_{t-1} · M_{t→t-1}.
    - Taşıma zinciri `max_motion_chain` sınırını aşarsa veya hareket kestirilemezse (sahne kesmesi)
      homografi geçersiz sayılır. Bu durumda ızgara çizilmez; yanlış hücre gösterilmez (§3).
    Not: Bu çevrimiçi (yalnızca ileri yönlü) sürümdür. Çevrimdışı iki yönlü taşıma ileride eklenecek.
    """

    def __init__(self, image_size: tuple[int, int], measurement_weight: float = 0.35, max_jump_m: float = 4.0,
                 strong_measurement_inliers: int = 10, max_motion_chain: int = 75, refiner=None,
                 refine_every: int = 1, zoom_reanchor: float = 0.10, gap_blend: bool = True, **_: object) -> None:
        w, h = image_size
        # son kabul edilen ölçümden beri: geçen kare sayısı ve kamera zoom'unun mutlak log toplamı.
        # Canlıda ölçüm seyrek gelir (~10 karede bir); karışım ağırlığı aradaki kare sayısına göre büyütülür.
        # Zoom sırasında kamera hareketiyle taşınan tahmin hızla bozulur; zoom > `zoom_reanchor` ise yeni ölçüm
        # (fiziksel kontrolden geçmiş) tahminle tutarsız da olsa kabul edilir.
        self.zoom_reanchor, self.gap_blend = zoom_reanchor, gap_blend
        self._since, self._zoom = 0, 0.0
        self.refiner, self.refine_every = refiner, max(1, int(refine_every))
        # karşılaştırma ve karışım için kontrol noktaları: görüntünün alt 2/3'ünde 4x3 ızgara
        xs, ys = np.linspace(0.1 * w, 0.9 * w, 4), np.linspace(0.4 * h, 0.95 * h, 3)
        self._ctrl = np.array([(x, y) for y in ys for x in xs], dtype=np.float64)
        self.alpha = measurement_weight
        self.max_jump_m = max_jump_m
        self.strong_inliers = strong_measurement_inliers
        self.max_chain = max_motion_chain
        self._H: Optional[np.ndarray] = None
        self._chain = 0

    def _distance_m(self, H1: np.ndarray, H2: np.ndarray) -> float:
        return float(np.median(np.linalg.norm(apply_h(H1, self._ctrl) - apply_h(H2, self._ctrl), axis=1)))

    def _blend(self, H_pred: np.ndarray, H_meas: np.ndarray) -> np.ndarray:
        a = 1.0 - (1.0 - self.alpha) ** max(1, self._since) if self.gap_blend else self.alpha
        target = a * apply_h(H_meas, self._ctrl) + (1 - a) * apply_h(H_pred, self._ctrl)
        H, _ = cv2.findHomography(self._ctrl, target, 0)
        return H / H[2, 2]

    def correct(self, H: np.ndarray) -> None:
        """Dışarıdan gelen mutlak düzeltme (canlı modda arka planda çizgilere oturtma, D-029): tahmini değiştirir."""
        if self._H is not None:
            self._H, self._chain, self._zoom, self._since = H, 0, 0.0, 0

    def update(self, meas: Optional[HomographyResult], motion: CameraMotion, frame_idx: int,
               frame: Optional[np.ndarray] = None) -> HomographyResult:
        out = self._update(meas, motion, frame_idx)
        if (out.valid and self.refiner is not None and frame is not None
                and frame_idx % self.refine_every == 0):
            H, ok, info = self.refiner.refine(frame, out.H)
            out.refined, out.refine_reason = ok, info.get("reason", "")
            if ok:
                # çizgiler mutlak kanıttır: düzeltilmiş H'den devam et, taşıma zinciri sıfırlanır
                out.H = self._H = H
                # çizgiler taze kanıttır: sonraki (gecikmeli, taşınmış) ölçüm bunu ezmesin, normal ağırlıkla karışsın
                self._chain, self._zoom, self._since = 0, 0.0, 0
        return out

    def _update(self, meas: Optional[HomographyResult], motion: CameraMotion, frame_idx: int) -> HomographyResult:
        pred = None
        if self._H is not None and motion.M is not None and self._chain < self.max_chain:
            pred = self._H @ motion.M
            pred /= pred[2, 2]
        self._since += 1
        if motion.M is not None:
            self._zoom += abs(0.5 * np.log(max(abs(np.linalg.det(motion.M[:2, :2])), 1e-6)))

        out = HomographyResult(frame_idx=frame_idx, H=None)
        if meas is not None:
            out.n_keypoints, out.n_inliers = meas.n_keypoints, meas.n_inliers
            out.reproj_error_m, out.keypoints = meas.reproj_error_m, meas.keypoints
            out.reject_reason = meas.reject_reason

        if meas is not None and meas.valid:
            if pred is None:
                H = meas.H
            elif self._distance_m(pred, meas.H) <= self.max_jump_m:
                H = self._blend(pred, meas.H)
            elif meas.n_inliers >= self.strong_inliers or self._zoom > self.zoom_reanchor:
                H = meas.H                      # güçlü ölçüm ya da zoom sonrası: yeniden sabitle (re-anchor)
            else:
                H = None                        # zayıf ve tutarsız ölçüm: tahmine güven
                out.reject_reason = "jump"
            if H is not None:
                self._H, self._chain = H, 0
                self._since, self._zoom = 0, 0.0
                out.H, out.valid, out.source, out.used_prior = H, True, "keypoints", meas.used_prior
                return out

        if pred is not None:
            self._H = pred
            self._chain += 1
            out.H, out.valid, out.source, out.motion_chain_len = pred, True, "camera_motion", self._chain
            return out

        # ne ölçüm ne tahmin: kalibrasyon yok
        if motion.M is None:
            self._H, self._chain = None, 0
        out.motion_chain_len = self._chain
        return out
