"""Canlı nesne çizimi düzleştiricisi (D-030) birim testleri."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sgi.types import ObjectState  # noqa: E402
from sgi.viz.live_smoother import BoxFlow, LiveObjectSmoother  # noqa: E402

I3 = lambda k: np.eye(3)  # noqa: E731  sabit kamera


def player(tid, x, y=300.0, h=60.0):
    return ObjectState(tid, "player", "A", (x - 10, y - h, x + 10, y), 0.9)


def ball(x, y):
    return ObjectState(0, "ball", "-", (x - 4, y - 8, x + 4, y), 0.5)


def foot(o):
    return 0.5 * (o.bbox[0] + o.bbox[2]), o.bbox[3]


def test_extrapolates_with_velocity_between_results():
    sm = LiveObjectSmoother(decay=0.0)
    sm.update(10, [player(1, 100)], 0, I3)
    sm.update(17, [player(1, 107)], 7, I3)             # 1 px/kare
    out = sm.update(20, [player(1, 107)], 7, I3)
    assert abs(foot(out[0])[0] - 120) < 1e-6            # 107 + 13 kare


def test_extrapolation_limited_by_horizon():
    sm = LiveObjectSmoother(horizon=5, decay=0.0)
    sm.update(0, [player(1, 100)], 0, I3)
    sm.update(7, [player(1, 107)], 7, I3)
    out = sm.update(40, [player(1, 107)], 7, I3)
    assert abs(foot(out[0])[0] - 112) < 1e-6


def test_implausible_speed_not_extrapolated():
    sm = LiveObjectSmoother(decay=0.0)
    sm.update(0, [player(1, 100)], 0, I3)
    sm.update(7, [player(1, 400)], 7, I3)               # kimlik değişimi: 43 px/kare
    out = sm.update(10, [player(1, 400)], 7, I3)
    assert foot(out[0])[0] == 400


def test_jump_is_damped_on_new_result():
    sm = LiveObjectSmoother(decay=0.5)
    sm.update(0, [player(1, 100)], 0, I3)
    sm.update(1, [player(1, 100)], 0, I3)
    out = sm.update(2, [player(1, 110)], 2, I3)         # 10 px düzeltme (hız 5 px/kare)
    assert abs(foot(out[0])[0] - 100) < 1e-6            # geldiği karede sıçramaz
    out = sm.update(3, [player(1, 110)], 2, I3)
    assert 100 < foot(out[0])[0] < 115                  # hedef 115, fark yarıya iner
    out = sm.update(10, [player(1, 110)], 2, I3)
    assert abs(foot(out[0])[0] - 150) < 0.5


def test_camera_transform_applied():
    sm = LiveObjectSmoother()
    T = np.array([[1, 0, 25.0], [0, 1, 0], [0, 0, 1]])
    out = sm.update(5, [player(1, 100)], 0, lambda k: T)
    assert abs(foot(out[0])[0] - 125) < 1e-6


def test_flow_overrides_prediction_and_ball_follows_owner():
    sm = LiveObjectSmoother()
    objs = [player(1, 100), ball(103, 299)]
    out = sm.update(5, objs, 0, I3, flow={1: np.array([12.0, 0.0])})
    pl = [o for o in out if o.role == "player"][0]
    bl = [o for o in out if o.role == "ball"][0]
    assert abs(foot(pl)[0] - 112) < 1e-6 and abs(foot(bl)[0] - 115) < 1e-6


def test_boxflow_tracks_moving_texture():
    rng = np.random.default_rng(0)
    tex = (rng.random((120, 60)) * 255).astype(np.uint8)
    bflow = BoxFlow(scale=1.0)

    def frame(x0):
        f = np.full((400, 600), 90, np.uint8)
        f[200:320, x0:x0 + 60] = tex
        return np.dstack([f] * 3)

    for i in range(6):
        bflow.push(i, frame(100 + 3 * i))
    bflow.start(0, [ObjectState(1, "player", "A", (100, 200, 160, 320), 0.9)])
    bflow.push(6, frame(118))
    d = bflow.displacement(6)
    assert abs(d[1][0] - 18) < 1.0 and abs(d[1][1]) < 1.0
