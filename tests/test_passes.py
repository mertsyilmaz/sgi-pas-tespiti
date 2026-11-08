"""Pas tespiti (M6) ve kural motoru (M7) birim testleri: sentetik sahneler (docs/05_PASS_RULES.md)."""
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sgi.events.passes import (END_LOST, END_OUT, END_RECEIVED, END_STOPPED, PassCandidate,  # noqa: E402
                               PassDetector, PlayerSnap)
from sgi.grid.grid import PitchGrid  # noqa: E402
from sgi.rules.engine import FAIL, SUCCESS, RuleEngine, summarize  # noqa: E402
from sgi.types import ObjectState  # noqa: E402

FPS = 25.0
CFG = yaml.safe_load((ROOT / "configs/pass_rules.yaml").read_text(encoding="utf-8"))
GRID = PitchGrid.from_yaml(ROOT / "configs/grid.yaml")


# --- yardımcılar ---
def pl(tid, team, x, y, role="player"):
    return ObjectState(tid, role, team, (0, 0, 1, 1), 1.0, (x, y), GRID.cell_of(x, y))


def ball(x, y):
    return ObjectState(0, "ball", "-", (0, 0, 1, 1), 1.0, (x, y), GRID.cell_of(x, y))


def run(frames):
    det = PassDetector.from_config(GRID, FPS, CFG)
    out = []
    for i, objs in enumerate(frames):
        out += det.update(i, objs)
    return out


def kick(p0, p1, n_hold=5, n_fly=10, n_after=5, others=(), receiver=None):
    """Oyuncu 1 (A) p0'da topu tutar, top n_fly karede p1'e gider; receiver (tid, takım) varsa p1'de durur."""
    frames = []
    rx = [] if receiver is None else [pl(receiver[0], receiver[1], *p1)]
    for _ in range(n_hold):
        frames.append([pl(1, "A", *p0), *rx, *others, ball(*p0)])
    for k in range(1, n_fly + 1):
        a = k / n_fly
        b = (p0[0] + a * (p1[0] - p0[0]), p0[1] + a * (p1[1] - p0[1]))
        frames.append([pl(1, "A", *p0), *rx, *others, ball(*b)])
    for _ in range(n_after):
        frames.append([pl(1, "A", *p0), *rx, *others, ball(*p1)])
    return frames


# --- M6 ---
def test_pass_received_by_teammate():
    c = run(kick((25, 30), (45, 30), n_fly=25, receiver=(2, "A")))
    assert len(c) == 1 and c[0].end_reason == END_RECEIVED
    assert c[0].passer_track_id == 1 and c[0].receiver_track_id == 2
    assert c[0].start_cell == GRID.cell_of(25, 30) and c[0].target_cell == GRID.cell_of(45, 30)
    assert c[0].path_cells[0] == c[0].start_cell and c[0].path_cells[-1] == c[0].target_cell


def test_dribble_same_player_not_a_pass():
    frames = []
    for k in range(30):                               # oyuncu topla koşuyor, top zaman zaman 2 m önüne kaçıyor
        x = 30 + 0.3 * k
        bx = x + (2.0 if k % 10 in (6, 7) else 0.3)
        frames.append([pl(1, "A", x, 30), ball(bx, 30)])
    assert run(frames) == []


def test_ball_out_of_pitch():
    c = run(kick((90, 60), (95, 75), n_after=0))
    assert len(c) == 1 and c[0].end_reason == END_OUT and c[0].target_cell == "OUT"


def test_ball_stops_without_owner():
    c = run(kick((30, 30), (40, 30), n_after=15))
    assert len(c) == 1 and c[0].end_reason == END_STOPPED and c[0].target_cell == GRID.cell_of(40, 30)


def test_short_gap_interpolated_long_gap_ends_at_last_seen():
    f = kick((30, 30), (50, 30), n_fly=20, receiver=(2, "A"))
    for i in range(10, 14):                           # 4 kare top yok (F-1)
        f[i] = [o for o in f[i] if o.role != "ball"]
    c = run(f)
    assert len(c) == 1 and "F-1" in c[0].fallbacks and c[0].quality["interpolated_frames"] == 4
    f = kick((30, 30), (50, 30), n_fly=20, n_after=0)
    f += [[pl(1, "A", 30, 30)] for _ in range(20)]    # top kayboldu, geri gelmiyor (F-2)
    c = run(f)
    assert len(c) == 1 and c[0].end_reason == END_LOST and "F-2" in c[0].fallbacks


# --- M7 ---
def cand(target_xy, players, passer=1, team="A"):
    cell = GRID.cell_of(*target_xy)
    snaps = [PlayerSnap(o.track_id, o.team, o.role, o.pitch_xy, o.cell) for o in players]
    return PassCandidate(0, 10, passer, team, GRID.cell_of(10, 10), cell, END_RECEIVED, (10, 10), target_xy,
                         end_players=snaps, path_cells=[GRID.cell_of(10, 10), cell])


ENG = RuleEngine(CFG)
T = (35.0, 20.0)                                      # hücre 14'ün içi


def test_r1_only_teammates():
    r = ENG.classify(cand(T, [pl(2, "A", 36, 21), pl(3, "B", 45, 21)]))   # rakip komşu hücrede: sayılmaz
    assert (r.label, r.matched_rule_id) == (SUCCESS, "R-1")


def test_r2_only_opponents():
    r = ENG.classify(cand(T, [pl(3, "B", 36, 21)]))
    assert (r.label, r.matched_rule_id) == (FAIL, "R-2")


def test_r3_nearest_decides_and_tie_is_success():
    assert ENG.classify(cand(T, [pl(2, "A", 35.5, 20), pl(3, "B", 38, 20)])).label == SUCCESS
    assert ENG.classify(cand(T, [pl(2, "A", 38, 20), pl(3, "B", 35.5, 20)])).label == FAIL
    r = ENG.classify(cand(T, [pl(2, "A", 36.0, 20), pl(3, "B", 34.1, 20)]))     # 1,0 / 0,9 m: fark < 0,3
    assert (r.label, r.matched_rule_id) == (SUCCESS, "R-3T")


def test_r4_passer_alone_and_referee_ignored():
    r = ENG.classify(cand(T, [pl(1, "A", 35, 20), pl(9, "-", 35.2, 20, role="referee")]))
    assert (r.label, r.matched_rule_id) == (FAIL, "R-4")


def test_goalkeeper_counts_as_opponent_and_out_fails():
    assert ENG.classify(cand(T, [pl(5, "B", 35.5, 20, role="goalkeeper")])).matched_rule_id == "R-2"
    c = cand(T, [pl(2, "A", 36, 21)])
    c.target_cell = "OUT"
    assert (ENG.classify(c).label, ENG.classify(c).matched_rule_id) == (FAIL, "R-OUT")


def test_summary_counts_by_passer_team():
    recs = [ENG.classify(cand(T, [pl(2, "A", 36, 21)])), ENG.classify(cand(T, [], passer=3, team="B"))]
    s = summarize(recs)
    assert s["by_team"]["A"] == {"success": 1, "fail": 0} and s["by_team"]["B"] == {"success": 0, "fail": 1}


def test_summary_target_team_counts_only_that_team():
    # D-040: hedef takım modunda diğer takımın pasları sayılmaz
    recs = [ENG.classify(cand(T, [pl(2, "A", 36, 21)])), ENG.classify(cand(T, [], passer=3, team="B"))]
    s = summarize(recs, "A")
    assert s["target_team"] == "A" and list(s["by_team"]) == ["A"]
    assert s["by_team"]["A"] == {"success": 1, "fail": 0} and s["total"] == 1


def test_sparse_results_same_pass_as_every_frame():
    # D-040: canlıda nesne sonuçları ~2 karede bir gelir; eşikler kare aralığıyla ölçüldüğü için aynı pas bulunur
    frames = kick((25, 30), (45, 30), n_fly=25, receiver=(2, "A"))
    det = PassDetector.from_config(GRID, FPS, CFG)
    sparse = [c for i, objs in enumerate(frames) if i % 2 == 0 for c in det.update(i, objs)]
    full = run(frames)
    assert len(sparse) == len(full) == 1
    key = lambda c: (c.start_cell, c.target_cell, c.receiver_track_id)  # noqa: E731
    assert key(sparse[0]) == key(full[0])


def test_impossible_ball_jump_between_players_not_a_pass():
    frames = []
    for k in range(20):                               # top 1 karede 4,5 m atlayıp geri dönüyor (üst üste iki oyuncu)
        bx = (40, 50) if k % 6 < 3 else (40, 54.5)
        frames.append([pl(1, "A", 40, 50), pl(2, "A", 40, 54.5), ball(*bx)])
    assert run(frames) == []
