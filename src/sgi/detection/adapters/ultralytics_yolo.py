"""Ultralytics YOLO nesne dedektörü adaptörü (D-011, D-022).

Varsayılan: COCO ile eğitilmiş YOLO11 (SoccerNet sn-gamestate baseline'ının kullandığı aile).
COCO sınıfları kanonik sınıflara `class_map` ile çevrilir: 0 person -> person, 32 sports ball -> ball.
Futbola özel eğitilmiş bir model (ör. SN-GSR ile ince ayar) aynı adaptörle, farklı `weights` ve
`class_map` vererek kullanılır (ör. {0: ball, 1: goalkeeper, 2: player, 3: referee}).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from sgi.interfaces import ObjectDetector
from sgi.types import CLS_BALL, Detection


class UltralyticsYoloDetector(ObjectDetector):
    def __init__(self, weights: str, class_map: dict, imgsz: int = 1280, conf: float = 0.3,
                 ball_conf: float = 0.15, device: str = "cuda", half: bool = True,
                 root: str | Path = ".", trt_engine: str | None = None, **_: object) -> None:
        from ultralytics import YOLO
        from ultralytics.utils import LOGGER

        LOGGER.setLevel(logging.ERROR)              # kare başına uyarı satırlarını sustur
        path = Path(root) / weights
        # trt_engine: TensorRT motoru (scripts/build_trt.py, aynı imgsz); yoksa .pt ağırlıklarıyla devam
        engine = Path(root) / trt_engine if trt_engine else None
        if engine is not None and engine.exists():
            self.model = YOLO(str(engine), task="detect")
        else:
            self.model = YOLO(str(path) if path.exists() else weights)
        self.class_map = {int(k): str(v) for k, v in class_map.items()}
        self.imgsz, self.conf, self.ball_conf = imgsz, conf, ball_conf
        self.device, self.half = device, half

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        r = self.model.predict(frame_bgr, imgsz=self.imgsz, conf=min(self.conf, self.ball_conf),
                               classes=list(self.class_map), device=self.device, half=self.half,
                               verbose=False)[0]
        out = []
        boxes = r.boxes
        for xyxy, c, cls in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy()):
            name = self.class_map[int(cls)]
            if c < (self.ball_conf if name == CLS_BALL else self.conf):
                continue
            out.append(Detection(name, tuple(float(v) for v in xyxy), float(c)))
        return out
