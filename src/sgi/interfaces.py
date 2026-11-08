"""Değiştirilebilir bileşenlerin arayüzleri (D-011, docs/03_ARCHITECTURE.md §1b).

Yeni bir model eklemek = bu arayüzlerden birini uygulayan bir adaptör yazıp registry'ye kaydetmek.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from sgi.types import CameraMotion, Detection, PitchKeypoint


class ObjectDetector(ABC):
    """Karedeki kişileri ve topu bulur; sınıflar kanonik isimlerle döner (D-011)."""

    @abstractmethod
    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        ...


class Tracker(ABC):
    """Kişi tespitlerine kareler arası kalıcı `track_id` atar (M3)."""

    @abstractmethod
    def update(self, detections: list[Detection], frame_bgr: Optional[np.ndarray] = None) -> list[Detection]:
        """`frame_bgr`: kamera hareketi telafisi yapan takipçiler için (ör. BoT-SORT GMC)."""
        ...

    @abstractmethod
    def reset(self) -> None:
        ...


class TeamAssigner(ABC):
    """track_id -> (rol, takım). Takip kimliği ile takım kimliği ayrıdır (şartname §2.2)."""

    @abstractmethod
    def observe(self, frame_bgr: np.ndarray, persons: list[Detection],
                pitch_xy: list[Optional[tuple[float, float]]], roles: Optional[list[str]] = None) -> None:
        """`roles`: dedektör rol veriyorsa (player/goalkeeper/referee) takip bazında rol; yoksa None."""
        ...

    @abstractmethod
    def assign(self, track_id: int) -> tuple[str, str]:
        ...


class PoseEstimator(ABC):
    """Kişinin kameraya göre yönü (D-035): 'back' (sırtı dönük), 'front' ya da None (karar verilemedi)."""

    @abstractmethod
    def facing(self, frame_bgr: np.ndarray, bbox: tuple[float, float, float, float]) -> Optional[str]:
        ...


class PitchKeypointDetector(ABC):
    """Görüntüde saha noktalarını bulur ve her birinin saha koordinatını verir."""

    @abstractmethod
    def detect(self, frame_bgr: np.ndarray) -> list[PitchKeypoint]:
        ...


class PitchRefiner(ABC):
    """Yaklaşık bir image->pitch homografisini görüntü kanıtıyla iyileştirir (D-014, D-020)."""

    @abstractmethod
    def refine(self, frame_bgr: np.ndarray, H_img2pitch: np.ndarray) -> tuple[np.ndarray, bool, dict]:
        """(H, kabul edildi mi, bilgi). Kabul edilmezse girdi H aynen döner."""
        ...


class CameraMotionEstimator(ABC):
    """Ardışık kareler arasındaki kamera hareketini kestirir (F-3)."""

    @abstractmethod
    def estimate(self, frame_bgr: np.ndarray) -> CameraMotion:
        """Bir önceki `estimate` çağrısındaki kareye göre hareketi döndürür."""
        ...

    @abstractmethod
    def reset(self) -> None:
        ...
