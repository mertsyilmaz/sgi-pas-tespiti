"""Konfigdeki `type` anahtarından adaptör üreten basit registry (D-011).

Adaptör modülleri tembel (lazy) import edilir: kurulu olmayan bir kütüphane yalnızca
o adaptör seçildiğinde hata verir.
"""
from __future__ import annotations

import importlib
from typing import Any

_REGISTRY: dict[str, dict[str, str]] = {
    "pitch_keypoint_detector": {
        "nbjw": "sgi.detection.keypoints.adapters.nbjw:NBJWKeypointDetector",
        # roboflow/sports saha modeli: pas_test'te noktalar 50-300 px hatalı çıktı, varsayılan kapalı (D-025)
        "ultralytics_pitch": "sgi.detection.keypoints.adapters.ultralytics_pitch:UltralyticsPitchKeypointDetector",
    },
    "camera_motion": {
        "lk_flow": "sgi.mapping.camera_motion.lk_flow:LKCameraMotion",
    },
    "object_detector": {
        "ultralytics_yolo": "sgi.detection.adapters.ultralytics_yolo:UltralyticsYoloDetector",
    },
    "pose_estimator": {
        "ultralytics_pose": "sgi.detection.pose.adapters.ultralytics_pose:UltralyticsPoseEstimator",
    },
    "tracker": {
        "bytetrack": "sgi.tracking.adapters.bytetrack:ByteTrackTracker",
        "botsort": "sgi.tracking.adapters.bytetrack:BoTSortTracker",
    },
    "team_assigner": {
        "color_kmeans": "sgi.teams.adapters.color_kmeans:ColorKMeansTeamAssigner",
    },
    "pitch_refiner": {
        "line_fit": "sgi.mapping.refine.line_fit:LineFitRefiner",
    },
}


def build(kind: str, cfg: dict[str, Any], **extra: Any) -> Any:
    cfg = dict(cfg)
    name = cfg.pop("type")
    try:
        target = _REGISTRY[kind][name]
    except KeyError:
        known = ", ".join(_REGISTRY.get(kind, {}))
        raise ValueError(f"Bilinmeyen {kind} tipi: '{name}'. Kayıtlı olanlar: {known}") from None
    module_name, class_name = target.split(":")
    cls = getattr(importlib.import_module(module_name), class_name)
    return cls(**cfg, **extra)
