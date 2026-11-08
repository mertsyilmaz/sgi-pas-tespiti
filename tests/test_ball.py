"""Top takibi (D-028) birim testleri: sentetik sahneler."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sgi.tracking.ball import DETECTED, HELD, PREDICTED, BallCandidate, BallTracker, PlayerRef  # noqa: E402

FPS = 25.0


def cand(x, y, conf=0.5):
    return BallCandidate((x, y), (0.0, 0.0), conf, (0, 0, 1, 1))


def player(tid, x, y):
    return PlayerRef(tid, (x, y), (0.0, 0.0))


def test_penalty_spot_rejected_when_isolated():
    bt = BallTracker(FPS)
    for t in range(10):
        assert bt.update(t, [cand(11.0, 34.0)], [player(1, 30, 30)], True) is None


def test_penalty_spot_with_player_rejected_if_static():
    bt = BallTracker(FPS)
    assert bt.update(0, [cand(94.2, 34.1)], [player(1, 95.0, 34.0)], True) is None


def test_ball_dribbled_onto_spot_kept():
    bt = BallTracker(FPS)
    bt.update(0, [cand(91.5, 34.0)], [player(1, 91.0, 34.0)], True)
    est = bt.update(5, [cand(93.8, 34.0)], [player(1, 93.3, 34.0)], True)
    assert est.state == DETECTED and est.pitch_xy == (93.8, 34.0)


def test_mark_does_not_steal_tracked_ball():
    bt = BallTracker(FPS)
    bt.update(0, [cand(40, 20)], [], True)
    est = bt.update(1, [cand(52.5, 34.0, 0.9), cand(40.3, 20.0, 0.3)], [], True)
    assert est.state == DETECTED and abs(est.pitch_xy[0] - 40.3) < 1e-6


def test_held_by_nearby_player_when_lost():
    bt = BallTracker(FPS)
    bt.update(0, [cand(50, 30)], [player(7, 50.5, 30.2)], True)
    est = bt.update(5, [], [player(7, 52.0, 31.0)], True)
    assert est.state == HELD and est.holder_id == 7 and est.pitch_xy == (52.0, 31.0)


def test_predicted_along_velocity_then_dropped():
    bt = BallTracker(FPS, predict_max_s=0.5)
    bt.update(0, [cand(20, 30)], [], True)
    bt.update(1, [cand(20.8, 30)], [], True)                  # 20 m/sn
    bt.update(2, [cand(21.6, 30)], [], True)
    est = bt.update(5, [], [], True)
    assert est.state == PREDICTED and 23.0 < est.pitch_xy[0] < 24.5
    assert bt.update(2 + 13, [], [], True) is None             # 0,5 sn sonra bırakılır


def test_outlier_outside_gate_ignored_then_switch_after_repeats():
    bt = BallTracker(FPS, switch_after=3)
    bt.update(0, [cand(30, 30)], [], True)
    est = bt.update(1, [cand(80, 10, 0.9)], [], True)         # 1 karede 50 m: imkânsız
    assert est is None or est.state != DETECTED
    bt.update(2, [cand(80, 10, 0.9)], [], True)
    est = bt.update(3, [cand(80, 10, 0.9)], [], True)         # aynı yerde 3. kez: takip oraya geçer
    assert est.state == DETECTED and est.pitch_xy == (80, 10)


def test_spot_not_taken_over_from_far_ball():
    bt = BallTracker(FPS, switch_after=3)
    bt.update(0, [cand(20, 56)], [player(1, 20.3, 56.2), player(2, 11.5, 34.0)], True)
    for t in range(1, 8):
        est = bt.update(t, [cand(11.0, 34.0, 0.7)], [player(1, 20.3, 56.2), player(2, 11.5, 34.0)], True)
        assert est is None or abs(est.pitch_xy[0] - 11.0) > 2


def test_shifted_static_spot_learned_and_rejected():
    # kalibrasyon kayması: penaltı noktası bilinen yerinden 3 m uzakta görünüyor, yanında oyuncu yok
    bt = BallTracker(FPS)
    outs = [bt.update(t, [cand(14.0, 34.2)], [player(1, 30, 30)], True) for t in range(0, 60, 2)]
    assert outs[0] is not None and outs[0].state == DETECTED   # önce top sanılır
    assert all(o is None for o in outs[-10:])                  # ~1,2 sn sonra işaret olarak öğrenilir
    assert bt.update(100, [cand(14.5, 34.0)], [player(1, 30, 30)], True) is None   # kayarak da tanınır


def test_ball_dribbled_onto_learned_spot_kept():
    bt = BallTracker(FPS)
    for t in range(0, 60, 2):
        bt.update(t, [cand(30.0, 20.0)], [], True)
    bt.update(80, [cand(27.5, 20.0)], [player(2, 27.2, 20.0)], True)
    est = bt.update(82, [cand(29.6, 20.0)], [player(2, 29.4, 20.0)], True)
    assert est is not None and est.state == DETECTED


def test_learned_spot_expires():
    bt = BallTracker(FPS, learned_ttl_s=2.0)
    for t in range(0, 60, 2):
        bt.update(t, [cand(14.0, 34.0)], [], True)
    assert bt.update(200, [cand(14.0, 34.0)], [], True) is not None


def test_resting_ball_with_player_not_learned():
    bt = BallTracker(FPS)
    for t in range(0, 80, 2):
        est = bt.update(t, [cand(40.0, 20.0)], [player(5, 41.0, 20.5)], True)
    assert est is not None and est.state == DETECTED


def _held_after(facing, t_end):
    """Top 0. karede oyuncu 1'in dibinde görülür, sonra kaybolur. t_end karesindeki durum."""
    bt = BallTracker(FPS, hold_max_s=1.0, hold_back_max_s=3.0, pose_every=5)
    pl = [player(1, 50.0, 30.0)]
    bt.update(0, [cand(50.3, 30.0)], pl, True)
    calls = []

    def facing_fn(tid):
        calls.append(tid)
        return facing

    est = None
    for t in range(1, t_end + 1):
        est = bt.update(t, [], pl, True, facing_fn)
    return est, calls


def test_hold_extended_when_holder_back_to_camera():
    est, calls = _held_after("back", 60)               # 2,4 sn > hold_max 1 sn
    assert est is not None and est.state == HELD and est.holder_id == 1
    assert calls and set(calls) == {1}
    est, _ = _held_after("back", 80)                   # 3,2 sn > hold_back_max 3 sn
    assert est is None


def test_hold_not_extended_when_facing_camera_or_unknown():
    for facing in ("front", None):
        est, calls = _held_after(facing, 30)
        assert est is None and len(calls) == 1


def test_pose_not_called_within_normal_hold():
    est, calls = _held_after("back", 25)
    assert est.state == HELD and calls == []
