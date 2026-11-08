"""Poz noktalarından oyuncu yönü (D-035)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sgi.detection.pose.adapters.ultralytics_pose import facing_from_keypoints  # noqa: E402


def _kp(l_x, r_x, conf=0.9):
    kp = np.zeros((17, 2))
    kc = np.full(17, 0.1)
    for (a, b), y in (((5, 6), 20.0), ((11, 12), 50.0)):
        kp[a], kp[b] = (l_x, y), (r_x, y)
        kc[a] = kc[b] = conf
    return kp, kc


def test_front_when_left_side_on_image_right():
    assert facing_from_keypoints(*_kp(30, 10), width=40) == "front"


def test_back_when_left_side_on_image_left():
    assert facing_from_keypoints(*_kp(10, 30), width=40) == "back"


def test_side_view_or_low_confidence_undecided():
    assert facing_from_keypoints(*_kp(21, 20), width=40) is None
    assert facing_from_keypoints(*_kp(10, 30, conf=0.2), width=40) is None
