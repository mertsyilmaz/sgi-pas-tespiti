"""Ultralytics YOLO poz modeliyle oyuncunun kameraya göre yönü (D-035).

Yalnızca tek oyuncunun kırpılmış görüntüsünde çalışır (tam kare değil): top takibi bunu, topu tutan oyuncu
uzun süre topu gizlediğinde çağırır. Yön, COCO noktalarındaki sol/sağ sırasından okunur: kameraya dönük kişinin
sol omzu görüntünün sağında kalır (x_sol > x_sağ); sırtı dönükte tersi. Omuz ve kalça çiftleri kullanılır.
Yüz noktalarının güveni kullanılmaz: küçük oyuncularda model sırttan bakışta da başa yüz noktası koyuyor.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from sgi.interfaces import PoseEstimator

L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 5, 6, 11, 12


def facing_from_keypoints(kp: np.ndarray, kc: np.ndarray, width: float, kp_conf: float = 0.5,
                          margin: float = 0.08) -> Optional[str]:
    """kp: (17, 2) görüntü noktaları, kc: (17,) güven, width: kişi kutusu genişliği."""
    d = [(kp[a, 0] - kp[b, 0]) / max(width, 1.0) for a, b in ((L_SHOULDER, R_SHOULDER), (L_HIP, R_HIP))
         if kc[a] >= kp_conf and kc[b] >= kp_conf]
    if not d:
        return None
    s = float(np.mean(d))
    return "front" if s > margin else "back" if s < -margin else None


class UltralyticsPoseEstimator(PoseEstimator):
    def __init__(self, weights: str, imgsz: int = 256, kp_conf: float = 0.5, margin: float = 0.08,
                 device: str = "cuda", half: bool = True, root: str | Path = ".", **_: object) -> None:
        from ultralytics import YOLO
        from ultralytics.utils import LOGGER

        LOGGER.setLevel(logging.ERROR)
        path = Path(root) / weights
        self.model = YOLO(str(path) if path.exists() else weights)
        self.imgsz, self.kp_conf, self.margin = imgsz, kp_conf, margin
        self.device, self.half = device, half
        self.n_calls = 0
        self._predict(np.zeros((imgsz, imgsz // 2, 3), np.uint8))   # ısınma: ilk çağrı saniyeler sürüyor

    def _predict(self, img: np.ndarray):
        return self.model.predict(img, imgsz=self.imgsz, device=self.device, half=self.half, verbose=False)[0]

    def facing(self, frame_bgr: np.ndarray, bbox: tuple[float, float, float, float]) -> Optional[str]:
        self.n_calls += 1
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        H, W = frame_bgr.shape[:2]
        X1, X2 = int(max(0, x1 - 0.3 * w)), int(min(W, x2 + 0.3 * w))
        Y1, Y2 = int(max(0, y1 - 0.15 * h)), int(min(H, y2 + 0.1 * h))
        if X2 - X1 < 8 or Y2 - Y1 < 16:
            return None
        r = self._predict(frame_bgr[Y1:Y2, X1:X2])
        if r.keypoints is None or len(r.keypoints) == 0:
            return None
        # kırpıntıda birden çok kişi olabilir: kutunun merkezine en yakın olan
        bx = r.boxes.xyxy.cpu().numpy()
        cx = (bx[:, 0] + bx[:, 2]) / 2 - (x1 + x2) / 2 + X1
        cy = (bx[:, 1] + bx[:, 3]) / 2 - (y1 + y2) / 2 + Y1
        i = int(np.argmin(np.hypot(cx, cy)))
        if abs(cx[i]) > 0.5 * w:
            return None
        return facing_from_keypoints(r.keypoints.xy[i].cpu().numpy(), r.keypoints.conf[i].cpu().numpy(),
                                     float(bx[i, 2] - bx[i, 0]), self.kp_conf, self.margin)
