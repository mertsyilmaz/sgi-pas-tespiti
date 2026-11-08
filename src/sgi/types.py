"""Modüller arası ortak veri yapıları (docs/06_DATA_MODEL.md).

Adaptörler yalnızca bu tipleri döndürür; kütüphaneye özgü nesneler pipeline'a sızmaz (D-011).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np

# Hücre kimliği: 1..50 ya da özel değerler
CELL_OUT = "OUT"
CELL_UNKNOWN = "UNKNOWN"
CellId = Union[int, str]


# Kanonik nesne sınıfları (D-011): adaptörler model sınıflarını bunlara çevirir
CLS_PERSON, CLS_PLAYER, CLS_GOALKEEPER, CLS_REFEREE, CLS_BALL = "person", "player", "goalkeeper", "referee", "ball"
TEAM_A, TEAM_B, TEAM_NONE, TEAM_UNKNOWN = "A", "B", "-", "?"


@dataclass
class Detection:
    """Tek karede bir nesne tespiti (M2). `person`: rolü henüz belli olmayan kişi."""
    cls: str
    bbox: tuple[float, float, float, float]       # x1, y1, x2, y2 (görüntü pikseli)
    conf: float
    track_id: int = -1                            # M3 takip kimliği (-1: yok)

    @property
    def foot(self) -> tuple[float, float]:
        """Zemin temas noktası: bbox alt-orta (oyuncu) / merkez (top)."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, y2) if self.cls != CLS_BALL else ((x1 + x2) / 2, (y1 + y2) / 2)


@dataclass
class ObjectState:
    """Sahaya yerleştirilmiş nesne (FrameState içeriği, docs/06_DATA_MODEL.md)."""
    track_id: int
    role: str                                     # player | goalkeeper | referee | ball
    team: str                                     # A | B | - | ?
    bbox: tuple[float, float, float, float]
    conf: float
    pitch_xy: Optional[tuple[float, float]] = None
    cell: CellId = "UNKNOWN"
    state: str = "detected"                       # top: detected | held | predicted (D-028)


@dataclass
class PitchKeypoint:
    """Görüntüde bulunan, saha modelinde konumu bilinen bir nokta."""
    kp_id: int                      # kaynak modelin nokta numarası (yalnızca izlenebilirlik için)
    image_xy: tuple[float, float]   # orijinal görüntü pikseli
    pitch_xy: tuple[float, float]   # saha koordinatı (metre, configs/pitch.yaml sistemi)
    conf: float = 1.0


@dataclass
class CameraMotion:
    """Ardışık iki kare arasındaki görüntü dönüşümü: M @ p_t = p_{t-1}."""
    M: Optional[np.ndarray]         # 3x3, None = kestirilemedi (ör. sahne kesmesi)
    n_inliers: int = 0
    scene_cut: bool = False         # renk histogramı birden değişti (D-033)


@dataclass
class HomographyResult:
    """Bir kare için görüntü -> saha dönüşümü (M4)."""
    frame_idx: int
    H: Optional[np.ndarray]                 # 3x3 image -> pitch, None = yok
    valid: bool = False
    source: str = "none"                    # keypoints | camera_motion | none
    n_keypoints: int = 0
    n_inliers: int = 0
    reproj_error_m: float = float("nan")
    motion_chain_len: int = 0
    reject_reason: str = ""                 # ölçüm reddedildiyse nedeni (hata teşhisi, 14_CALIBRATION_QUALITY §3)
    refined: bool = False                   # çizgilere oturtma uygulandı mı (D-020)
    used_prior: bool = False                # sabit kamera konumu önseliyle çözüldü mü (D-027)
    bootstrapped: bool = False              # az nokta + çizgilerden fiziksel kamerayla kuruldu mu (D-034)
    refine_reason: str = ""                 # uygulanmadıysa nedeni (no_gain, too_large, ...)
    keypoints: list[PitchKeypoint] = field(default_factory=list)
