"""Saha noktası önbelleği (D-018).

Saha noktası modeli işleme süresinin ~%90'ıdır, ama sonraki adımlardaki değişiklikler (doğrulama kuralları,
takip, kalite göstergeleri, çizim) model çıktısını değiştirmez. Bu yüzden model her video için bir kez
çalıştırılır ve sonuçlar diske yazılır.

Anahtar = video kimliği (ad, boyut, değişiklik zamanı) + dedektör konfigürasyonu. Model, ağırlık veya
eşik değişirse farklı bir önbellek dosyası kullanılır; eski sonuç yanlışlıkla kullanılmaz.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from sgi.types import PitchKeypoint


def cache_key(video: Path, detector_cfg: dict[str, Any]) -> str:
    st = video.stat()
    ident = {"video": video.name, "size": st.st_size, "mtime": int(st.st_mtime), "detector": detector_cfg}
    return hashlib.sha1(json.dumps(ident, sort_keys=True, default=str).encode()).hexdigest()[:12]


class KeypointCache:
    def __init__(self, cache_dir: Path, video: Path, detector_cfg: dict[str, Any]) -> None:
        self.path = cache_dir / f"{video.stem}_{cache_key(video, detector_cfg)}.npz"
        self._data: dict[int, list[PitchKeypoint]] = {}
        self._dirty = False
        if self.path.exists():
            z = np.load(self.path)
            frames, rows = z["frames"], z["rows"]          # rows: frame, kp_id, ix, iy, px, py, conf
            self._data = {int(f): [] for f in frames}
            for r in rows:
                self._data[int(r[0])].append(PitchKeypoint(int(r[1]), (float(r[2]), float(r[3])),
                                                           (float(r[4]), float(r[5])), float(r[6])))

    def __len__(self) -> int:
        return len(self._data)

    def covers(self, frame_indices) -> bool:
        return all(i in self._data for i in frame_indices)

    def get(self, frame_idx: int) -> Optional[list[PitchKeypoint]]:
        return self._data.get(frame_idx)

    def put(self, frame_idx: int, kps: list[PitchKeypoint]) -> None:
        self._data[frame_idx] = kps
        self._dirty = True

    def save(self) -> None:
        if not self._dirty:
            return
        frames = np.array(sorted(self._data), dtype=np.int64)
        rows = [(f, k.kp_id, *k.image_xy, *k.pitch_xy, k.conf) for f in frames for k in self._data[f]]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.path, frames=frames, rows=np.array(rows, dtype=np.float64).reshape(-1, 7))
        self._dirty = False
