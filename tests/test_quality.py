import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from sgi.mapping.quality import QualityMonitor, pitch_line_samples  # noqa: E402
from test_homography import H as IMG_H, W as IMG_W, synthetic_h  # noqa: E402


def _render(H_img2pitch: np.ndarray) -> np.ndarray:
    """Bilinen kameradan sentetik saha görüntüsü: çim + beyaz çizgiler + üstte tribün."""
    img = np.zeros((IMG_H, IMG_W, 3), np.uint8)
    img[:] = (40, 40, 40)
    G = np.linalg.inv(H_img2pitch)
    border = np.array([[0, 0], [105, 0], [105, 68], [0, 68]], float)
    p = np.c_[border, np.ones(4)] @ G.T
    cv2.fillPoly(img, [np.round(p[:, :2] / p[:, 2:3]).astype(np.int32)], (40, 140, 50))
    q = np.c_[pitch_line_samples(step=0.1), np.ones(len(pitch_line_samples(step=0.1)))] @ G.T
    for x, y in np.round(q[:, :2] / q[:, 2:3]).astype(int):
        if 0 <= x < IMG_W and 0 <= y < IMG_H:
            cv2.circle(img, (x, y), 2, (235, 235, 235), -1)
    return img


def test_line_score_separates_correct_from_shifted_grid():
    H = synthetic_h(target=(80.0, 34.0, 0.0))
    frame = _render(H)
    qm = QualityMonitor((IMG_W, IMG_H))
    good = qm.measure(frame, H)
    shifted = qm.measure(frame, np.array([[1, 0, 2.0], [0, 1, 1.0], [0, 0, 1]]) @ H)   # 2 m kaymış ızgara
    assert good.line_score > 0.8, good
    assert shifted.line_score < 0.5, shifted
    assert good.pitch_iou > 0.8 and good.green_ratio > 0.3


def test_no_calibration_gives_only_green_ratio():
    frame = _render(synthetic_h())
    q = QualityMonitor((IMG_W, IMG_H)).measure(frame, None)
    assert q.green_ratio > 0.3 and np.isnan(q.line_score)
