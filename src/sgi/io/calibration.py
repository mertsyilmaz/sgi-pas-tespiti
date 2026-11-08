"""`grid_demo.py process` kalibrasyon önbelleğinin (calibration.npz) okunması."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from sgi.types import HomographyResult, PitchKeypoint


class CalibrationCache:
    def __init__(self, path: str | Path) -> None:
        self.c = np.load(path)
        self.start, self.fps, self.n = int(self.c["start_frame"]), float(self.c["fps"]), len(self.c["H"])
        self.size = tuple(int(v) for v in self.c["size"])
        self._sources = [str(s) for s in self.c["sources"]]
        self._kp = self.c["keypoints"]
        self._prior = self.c["used_prior"] if "used_prior" in self.c.files else np.zeros(self.n, bool)

    def H(self, i: int):
        """Kesit içi kare i için görüntü -> saha homografisi; yoksa None."""
        H = self.c["H"][i] if i < self.n else None
        return None if H is None or np.isnan(H).any() else H

    def result(self, i: int) -> HomographyResult:
        c, H = self.c, self.H(i)
        kps = [PitchKeypoint(-1, (float(x), float(y)), (0.0, 0.0)) for _, x, y in self._kp[self._kp[:, 0] == i]]
        return HomographyResult(frame_idx=self.start + i, H=H, valid=H is not None,
                                source=self._sources[int(c["source"][i])], n_inliers=int(c["n_inliers"][i]),
                                reproj_error_m=float(c["reproj_error_m"][i]),
                                motion_chain_len=int(c["motion_chain"][i]), used_prior=bool(self._prior[i]),
                                keypoints=kps)
