"""ByteTrack takip adaptörü (Ultralytics içindeki BYTETracker), M3, D-022.

Not: supervision 0.31 ByteTrack'i kaldırdı; Ultralytics'in uygulaması kullanılıyor. Ultralytics'e özgü
sınıflar yalnızca bu dosyada görünür (D-011).
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from sgi.interfaces import Tracker
from sgi.types import Detection


class _Dets:
    """BYTETracker'ın beklediği numpy tabanlı tespit kümesi (conf, cls, xywh, maske ile indeksleme)."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray) -> None:
        self.xyxy = xyxy.astype(np.float32).reshape(-1, 4)
        self.conf = conf.astype(np.float32)
        self.cls = np.zeros(len(self.conf), np.float32)
        x1, y1, x2, y2 = self.xyxy.T
        self.xywh = np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, idx) -> "_Dets":
        return _Dets(self.xyxy[idx], self.conf[idx])


class ByteTrackTracker(Tracker):
    def __init__(self, frame_rate: float = 25.0, lost_track_buffer: int = 50, track_high_thresh: float = 0.3,
                 track_low_thresh: float = 0.1, new_track_thresh: float = 0.35, match_thresh: float = 0.8,
                 **_: object) -> None:
        from ultralytics.trackers.byte_tracker import BYTETracker

        self._cls = BYTETracker
        self._args = SimpleNamespace(track_high_thresh=track_high_thresh, track_low_thresh=track_low_thresh,
                                     new_track_thresh=new_track_thresh, track_buffer=lost_track_buffer,
                                     match_thresh=match_thresh, fuse_score=True)
        self.tracker = BYTETracker(self._args)

    def reset(self) -> None:
        self.tracker = self._cls(self._args)

    def update(self, detections: list[Detection], frame_bgr=None) -> list[Detection]:
        xyxy = np.array([d.bbox for d in detections], np.float32).reshape(-1, 4)
        conf = np.array([d.conf for d in detections], np.float32)
        out = self.tracker.update(_Dets(xyxy, conf), frame_bgr)
        res = []
        for row in out:                                     # [x1, y1, x2, y2, track_id, score, cls, idx]
            src = detections[int(row[7])] if len(detections) else None
            res.append(Detection(src.cls if src else "person", tuple(float(v) for v in row[:4]),
                                 float(row[5]), int(row[4])))
        return res


class BoTSortTracker(ByteTrackTracker):
    """BoT-SORT (Ultralytics): ByteTrack + kamera hareketi telafisi (GMC). Yayın kamerası dönüp zoom
    yaptığında kutuları kamera hareketine göre öteler; kimlik değişimlerini azaltır (D-024)."""

    def __init__(self, frame_rate: float = 25.0, lost_track_buffer: int = 50, track_high_thresh: float = 0.3,
                 track_low_thresh: float = 0.1, new_track_thresh: float = 0.35, match_thresh: float = 0.8,
                 gmc_method: str = "sparseOptFlow", **_: object) -> None:
        from ultralytics.trackers.bot_sort import BOTSORT

        self._cls = BOTSORT
        self._args = SimpleNamespace(track_high_thresh=track_high_thresh, track_low_thresh=track_low_thresh,
                                     new_track_thresh=new_track_thresh, track_buffer=lost_track_buffer,
                                     match_thresh=match_thresh, fuse_score=True, gmc_method=gmc_method,
                                     proximity_thresh=0.5, appearance_thresh=0.8, with_reid=False, model="auto")
        self.tracker = BOTSORT(self._args)
