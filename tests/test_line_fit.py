import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from sgi.mapping.homography import apply_h  # noqa: E402
from sgi.mapping.refine.line_fit import LineFitRefiner  # noqa: E402
from test_homography import H as IMG_H, W as IMG_W, synthetic_h  # noqa: E402
from test_quality import _render  # noqa: E402

CTRL = np.array([(x, y) for y in np.linspace(0.4 * IMG_H, 0.95 * IMG_H, 3) for x in np.linspace(0.1 * IMG_W, 0.9 * IMG_W, 4)])


def err_m(Ha, Hb) -> float:
    return float(np.median(np.linalg.norm(apply_h(Ha, CTRL) - apply_h(Hb, CTRL), axis=1)))


def perturb(H, dx, dy, rot_deg=0.0):
    a = np.deg2rad(rot_deg)
    c, s = np.cos(a), np.sin(a)
    cx, cy = 52.5, 34.0                       # saha merkezi etrafında döndür + kaydır
    T = np.array([[c, -s, cx - c * cx + s * cy + dx], [s, c, cy - s * cx - c * cy + dy], [0, 0, 1]])
    return T @ H


def test_shifted_grid_is_pulled_back_onto_lines():
    H_true = synthetic_h(target=(80.0, 34.0, 0.0))
    frame = _render(H_true)
    ref = LineFitRefiner((IMG_W, IMG_H))
    for dx, dy, rot in [(2.0, 1.0, 0.0), (-1.5, 2.0, 1.0), (1.0, -1.0, -1.5)]:
        H_bad = perturb(H_true, dx, dy, rot)
        H_fix, ok, info = ref.refine(frame, H_bad)
        assert ok, info
        assert err_m(H_fix, H_true) < 0.35 * err_m(H_bad, H_true), (err_m(H_bad, H_true), err_m(H_fix, H_true), info)
        assert err_m(H_fix, H_true) < 0.5, info


def test_correct_grid_is_not_made_worse():
    H_true = synthetic_h(target=(30.0, 30.0, 0.0))
    frame = _render(H_true)
    H_fix, ok, info = LineFitRefiner((IMG_W, IMG_H)).refine(frame, H_true)
    assert err_m(H_fix, H_true) < 0.2, info


def test_far_off_grid_is_not_accepted_with_large_jump():
    H_true = synthetic_h(target=(80.0, 34.0, 0.0))
    frame = _render(H_true)
    H_bad = perturb(H_true, 15.0, 0.0)
    H_fix, ok, info = LineFitRefiner((IMG_W, IMG_H), max_correction_m=4.0).refine(frame, H_bad)
    assert not ok or err_m(H_fix, H_bad) <= 4.0 + 1e-6
