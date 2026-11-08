"""roboflow/sports saha noktası modeli (YOLOv8-pose, 32 nokta) adaptörü (D-025).

Kaynak: https://github.com/roboflow/sports (MIT), `football-pitch-detection.pt`.
Model 32 anlamlı saha noktası verir (köşeler, ceza sahası/kale alanı köşeleri, penaltı noktaları, orta çizgi
uçları, orta yuvarlağın orta çizgi ve yatay eksenle kesişimleri). roboflow sahayı 120x70 m varsayar; burada
noktalar **anlamlarıyla** eşlenir ve konumları bizim saha ölçülerimizle (configs/pitch.yaml) hesaplanır.
Orta yuvarlak noktaları (15, 16, 31, 32) orta saha görüntüleri için NBJW'yi tamamlar.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from sgi.interfaces import PitchKeypointDetector
from sgi.types import PitchKeypoint


def roboflow_vertices(L: float = 105.0, W: float = 68.0) -> list[tuple[float, float]]:
    """roboflow SoccerPitchConfiguration.vertices sırası, FIFA ölçüleriyle (metre). 1-tabanlı numara = indeks+1."""
    pb_w, pb_l = 40.32, 16.5          # ceza sahası
    gb_w, gb_l = 18.32, 5.5           # kale alanı
    r, spot = 9.15, 11.0
    cy = W / 2
    return [
        (0, 0), (0, cy - pb_w / 2), (0, cy - gb_w / 2), (0, cy + gb_w / 2), (0, cy + pb_w / 2), (0, W),     # 1-6
        (gb_l, cy - gb_w / 2), (gb_l, cy + gb_w / 2), (spot, cy),                                           # 7-9
        (pb_l, cy - pb_w / 2), (pb_l, cy - gb_w / 2), (pb_l, cy + gb_w / 2), (pb_l, cy + pb_w / 2),         # 10-13
        (L / 2, 0), (L / 2, cy - r), (L / 2, cy + r), (L / 2, W),                                          # 14-17
        (L - pb_l, cy - pb_w / 2), (L - pb_l, cy - gb_w / 2), (L - pb_l, cy + gb_w / 2), (L - pb_l, cy + pb_w / 2),  # 18-21
        (L - spot, cy), (L - gb_l, cy - gb_w / 2), (L - gb_l, cy + gb_w / 2),                               # 22-24
        (L, 0), (L, cy - pb_w / 2), (L, cy - gb_w / 2), (L, cy + gb_w / 2), (L, cy + pb_w / 2), (L, W),     # 25-30
        (L / 2 - r, cy), (L / 2 + r, cy),                                                                    # 31-32
    ]


class UltralyticsPitchKeypointDetector(PitchKeypointDetector):
    ID_OFFSET = 100                    # izlenebilirlik: roboflow noktaları 101..132 (NBJW 1..73 ile karışmasın)

    def __init__(self, weights: str, kp_conf: float = 0.5, imgsz: int = 640, device: str = "cuda",
                 half: bool = True, pitch_length: float = 105.0, pitch_width: float = 68.0,
                 root: str | Path = ".", **_: object) -> None:
        from ultralytics import YOLO
        from ultralytics.utils import LOGGER

        LOGGER.setLevel(logging.ERROR)
        self.model = YOLO(str(Path(root) / weights))
        self.kp_conf, self.imgsz, self.device, self.half = kp_conf, imgsz, device, half
        self.world = roboflow_vertices(pitch_length, pitch_width)

    def detect(self, frame_bgr: np.ndarray) -> list[PitchKeypoint]:
        r = self.model.predict(frame_bgr, imgsz=self.imgsz, conf=0.3, device=self.device, half=self.half,
                               verbose=False)[0]
        if r.keypoints is None or len(r.keypoints) == 0:
            return []
        best = int(r.boxes.conf.argmax()) if r.boxes is not None and len(r.boxes) else 0
        xy = r.keypoints.xy[best].cpu().numpy()
        conf = r.keypoints.conf[best].cpu().numpy() if r.keypoints.conf is not None else np.ones(len(xy))
        out = []
        for i, ((x, y), c) in enumerate(zip(xy, conf)):
            if c >= self.kp_conf and (x > 0 or y > 0):
                out.append(PitchKeypoint(self.ID_OFFSET + i + 1, (float(x), float(y)), self.world[i], float(c)))
        return out


class CompositePitchKeypointDetector(PitchKeypointDetector):
    """Birden fazla saha noktası dedektörünün çıktısını birleştirir (homografi tüm noktalardan hesaplanır)."""

    def __init__(self, detectors: list[PitchKeypointDetector]) -> None:
        self.detectors = detectors

    def detect(self, frame_bgr: np.ndarray) -> list[PitchKeypoint]:
        out: list[PitchKeypoint] = []
        for d in self.detectors:
            out += d.detect(frame_bgr)
        return out
