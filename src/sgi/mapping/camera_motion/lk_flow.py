"""Seyrek optik akış (Lucas-Kanade) ile kareler arası kamera hareketi (M4b, F-3, D-012).

Kamera dönme/zoom yaptığı için hareket bir homografi olarak modellenir. Oyuncular RANSAC ile
aykırı değer olarak elenir. Sahne kesmesinde M = None döner: ya yeterli inlier bulunamaz ya da renk
histogramı ardışık karede birden değişir (D-033). Histogram kontrolü gereklidir: kesmede LK, iki farklı görüntü
arasında da 40+ "inlier" bulabiliyor ve eski ızgara yeni kameranın görüntüsüne taşınıyordu.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from sgi.interfaces import CameraMotionEstimator
from sgi.types import CameraMotion


class LKCameraMotion(CameraMotionEstimator):
    def __init__(self, max_width: int = 640, max_corners: int = 600, min_inliers: int = 40,
                 cut_hist_dist: float = 0.35) -> None:
        # cut_hist_dist: HSV histogram Bhattacharyya uzaklığı; pas_test'te kesmeler >= 0,45, diğer kareler <= 0,17
        self.cut_hist_dist = cut_hist_dist
        self._hist: Optional[np.ndarray] = None
        self.max_width = max_width
        self.max_corners = max_corners
        self.min_inliers = min_inliers
        self._prev: Optional[np.ndarray] = None
        self._scale = 1.0

    def reset(self) -> None:
        self._prev = None
        self._hist = None

    def _scene_cut(self, frame_bgr: np.ndarray) -> bool:
        hsv = cv2.cvtColor(cv2.resize(frame_bgr, (160, 90), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1], None, [30, 16], [0, 180, 0, 256])
        cv2.normalize(h, h, 1, 0, cv2.NORM_L1)
        prev, self._hist = self._hist, h
        return prev is not None and cv2.compareHist(prev, h, cv2.HISTCMP_BHATTACHARYYA) > self.cut_hist_dist

    def _prep(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        self._scale = min(1.0, self.max_width / w)
        small = cv2.resize(frame_bgr, None, fx=self._scale, fy=self._scale, interpolation=cv2.INTER_AREA) \
            if self._scale < 1.0 else frame_bgr
        return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    def estimate(self, frame_bgr: np.ndarray) -> CameraMotion:
        gray = self._prep(frame_bgr)
        prev, self._prev = self._prev, gray
        cut = self.cut_hist_dist > 0 and self._scene_cut(frame_bgr)
        if prev is None or cut:
            return CameraMotion(M=None, scene_cut=cut)

        pts_prev = cv2.goodFeaturesToTrack(prev, self.max_corners, qualityLevel=0.01, minDistance=8)
        if pts_prev is None or len(pts_prev) < self.min_inliers:
            return CameraMotion(M=None)
        pts_cur, status, _ = cv2.calcOpticalFlowPyrLK(prev, gray, pts_prev, None, winSize=(21, 21), maxLevel=3)
        ok = status.ravel() == 1
        if ok.sum() < self.min_inliers:
            return CameraMotion(M=None)

        # M: güncel kare -> önceki kare
        M, mask = cv2.findHomography(pts_cur[ok], pts_prev[ok], cv2.RANSAC, 2.0)
        n_in = int(mask.sum()) if mask is not None else 0
        if M is None or n_in < self.min_inliers:
            return CameraMotion(M=None, n_inliers=n_in)

        # küçültülmüş koordinatlardan orijinal piksel koordinatına
        S = np.diag([self._scale, self._scale, 1.0])
        M = np.linalg.inv(S) @ M @ S
        return CameraMotion(M=M / M[2, 2], n_inliers=n_in)
