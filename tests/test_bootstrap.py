"""Az noktadan başlangıç kalibrasyonu (D-034): bilinen kamerayla çizilmiş sentetik sahada 3 noktadan geri bulma."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sgi.mapping.bootstrap import LineCameraBootstrap  # noqa: E402
from sgi.mapping.camera_prior import pose_to_h  # noqa: E402
from sgi.mapping.homography import apply_h, pitch_to_image  # noqa: E402
from sgi.mapping.quality import pitch_line_samples  # noqa: E402
from sgi.types import PitchKeypoint  # noqa: E402

SIZE = (1280, 720)


def look_at(C, tgt):
    fwd = (tgt - C) / np.linalg.norm(tgt - C)
    right = np.cross(fwd, [0, 0, -1.0])
    right /= np.linalg.norm(right)
    R = np.vstack([right, np.cross(fwd, right), fwd])
    if np.linalg.det(R) < 0:
        R[0] *= -1
    return R


def render(H):
    img = np.zeros((SIZE[1], SIZE[0], 3), np.uint8)
    img[:] = (40, 130, 50)                                    # çim
    img[:150] = (90, 60, 160)                                 # tribün
    G = pitch_to_image(H, SIZE)
    pts = apply_h(G, pitch_line_samples(step=0.2))
    for x, y in pts.astype(int):
        if 0 <= x < SIZE[0] and 150 <= y < SIZE[1]:
            cv2.circle(img, (x, y), 2, (235, 235, 235), -1)
    return img


def test_recovers_camera_from_three_points_and_lines():
    C = np.array([52.6, 114.0, -23.5])                        # pas_test'te kestirilen yayın kamerası
    R = look_at(C, np.array([62.0, 22.0, 0.0]))
    H_true = pose_to_h(4100.0, R, C, SIZE)
    frame = render(H_true)
    G = pitch_to_image(H_true, SIZE)
    world = [(52.5 + 9.15, 34.0), (52.5, 0.0), (52.5 + 6.47, 34.0 - 6.47)]   # orta yuvarlak / orta çizgi noktaları
    kps = [PitchKeypoint(i, tuple(apply_h(G, np.array([p]))[0]), p) for i, p in enumerate(world)]
    bs = LineCameraBootstrap(SIZE)
    H = bs.solve(frame, kps)
    assert H is not None, bs.last_info
    probe = np.array([[30.0, 10.0], [70.0, 30.0], [60.0, 45.0]])
    err = np.linalg.norm(apply_h(pitch_to_image(H, SIZE), probe) - apply_h(G, probe), axis=1)
    assert err.max() < 15, err


def test_rejects_with_too_few_points():
    bs = LineCameraBootstrap(SIZE)
    assert bs.solve(np.zeros((720, 1280, 3), np.uint8), []) is None
