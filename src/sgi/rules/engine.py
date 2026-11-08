"""Kural motoru (M7, docs/05_PASS_RULES.md §B). Girdi yalnızca PassCandidate; kurallar configs/pass_rules.yaml.

Değerlendirme sırası (§B.5):
  1. Yedek kurallar zaten aday üretilirken uygulanmış ve kayda yazılmıştır (F-1, F-2, F-5; F-3 kalibrasyonda, F-4
     oyuncu listesinde)
  2. Top OUT -> R-OUT
  3. Hedef hücredeki takım arkadaşları M ve rakipler O (pasör ve hakem hariç, kaleci dahil)
  4. R-1 / R-2 / R-3 (+R-3T) / R-4
  5. Ara hücreler kayda yazılır (R-0), kararda kullanılmaz
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from sgi.events.passes import PassCandidate
from sgi.types import CELL_OUT

SUCCESS, FAIL = "SUCCESS", "FAIL"


@dataclass
class PassRecord:
    pass_id: str
    candidate: PassCandidate
    label: str
    matched_rule_id: str
    reason: str
    rule_trace: list = field(default_factory=list)
    target_cell_players: list = field(default_factory=list)   # [{track_id, team, role, dist_to_ball_m}]

    def to_dict(self, fps: float) -> dict:
        c = self.candidate
        return {
            "pass_id": self.pass_id,
            "frame_start": c.frame_start, "frame_end": c.frame_end,
            "t_start": round(c.frame_start / fps, 2), "t_end": round(c.frame_end / fps, 2),
            "passer_track_id": c.passer_track_id, "team": c.passer_team,
            "receiver_track_id": c.receiver_track_id, "receiver_team": c.receiver_team,
            "start_cell": c.start_cell, "intermediate_cells": c.intermediate_cells, "target_cell": c.target_cell,
            "path_cells": c.path_cells, "end_reason": c.end_reason,
            "ball_start": [round(v, 2) for v in c.ball_start], "ball_end": [round(v, 2) for v in c.ball_end],
            "label": self.label, "matched_rule_id": self.matched_rule_id, "reason": self.reason,
            "rule_trace": self.rule_trace, "fallbacks_applied": c.fallbacks,
            "target_cell_players": self.target_cell_players, "quality": c.quality,
            "ball_path": [[f, round(p[0], 2), round(p[1], 2)] for f, p, _, _ in c.ball_path],
        }


class RuleEngine:
    def __init__(self, cfg: dict[str, Any]) -> None:
        t = cfg["target_cell_rules"]
        self.exclude_passer = bool(t.get("exclude_passer", True))
        self.exclude_roles = set(t.get("exclude_roles", ["referee"]))
        self.include_gk = bool(t.get("include_goalkeepers", True))
        if t.get("neighbor_cells", False):
            raise ValueError("neighbor_cells: true tanımlı değil (Q15: yalnızca hedef hücre)")
        self.out_result = t.get("out_of_pitch", FAIL)
        self.tie_eps = float(t.get("tie_epsilon_m", 0.3))
        self.tie_result = t.get("tie_result", SUCCESS)
        self.outcomes = t["outcomes"]
        self._n = 0

    def classify(self, c: PassCandidate) -> PassRecord:
        self._n += 1
        pid = f"P{self._n:04d}"
        trace = [{"rule_id": "R-0", "matched": False, "note": f"ara hücreler {c.intermediate_cells} (değerlendirilmez)"}]

        def rec(label, rule, reason, players=()):
            trace.append({"rule_id": rule, "matched": True, "note": reason})
            return PassRecord(pid, c, label, rule, reason, trace, list(players))

        if c.target_cell == CELL_OUT:
            return rec(self.out_result, "R-OUT", "top saha dışına çıktı")
        trace.append({"rule_id": "R-OUT", "matched": False, "note": "top sahada"})

        bx, by = c.ball_end
        mates, opps, listed = [], [], []
        for pl in c.end_players:
            if pl.cell != c.target_cell or pl.role in self.exclude_roles:
                continue
            if pl.role == "goalkeeper" and not self.include_gk:
                continue
            if self.exclude_passer and pl.track_id == c.passer_track_id:
                continue
            d = float(np.hypot(pl.pitch_xy[0] - bx, pl.pitch_xy[1] - by))
            listed.append({"track_id": pl.track_id, "team": pl.team, "role": pl.role, "dist_to_ball_m": round(d, 2)})
            (mates if pl.team == c.passer_team else opps).append(d)

        cell = c.target_cell
        if mates and not opps:
            return rec(self.outcomes["only_teammates"], "R-1",
                       f"hedef hücre {cell}: yalnızca takım arkadaşı ({len(mates)})", listed)
        if opps and not mates:
            return rec(self.outcomes["only_opponents"], "R-2", f"hedef hücre {cell}: yalnızca rakip ({len(opps)})",
                       listed)
        if mates and opps:
            dm, do = min(mates), min(opps)
            if abs(dm - do) < self.tie_eps:
                return rec(self.tie_result, "R-3T",
                           f"hedef hücre {cell}: en yakın takım arkadaşı {dm:.1f} m, rakip {do:.1f} m (eşit)", listed)
            label = SUCCESS if dm < do else FAIL
            return rec(label, "R-3", f"hedef hücre {cell}: topa en yakın {'takım arkadaşı' if dm < do else 'rakip'} "
                                     f"({dm:.1f} m / {do:.1f} m)", listed)
        return rec(self.outcomes["none"], "R-4", f"hedef hücre {cell}: pasör dışında oyuncu yok", listed)


def summarize(records: list[PassRecord], target_team: Optional[str] = None) -> dict:
    """§C: takım başına başarılı / başarısız sayısı. Her pas tek kayıttır ve pasörün takımına yazılır.
    target_team verilirse (D-040) yalnızca o takımın pasları sayılır; diğer takımın pasları raporda yer almaz."""
    teams = [target_team] if target_team else ["A", "B"]
    by_team = {t: {SUCCESS: 0, FAIL: 0} for t in teams}
    for r in records:
        if r.candidate.passer_team in by_team:
            by_team[r.candidate.passer_team][r.label] += 1
    return {"target_team": target_team,
            "by_team": {t: {"success": v[SUCCESS], "fail": v[FAIL]} for t, v in by_team.items()},
            "total": sum(v[SUCCESS] + v[FAIL] for v in by_team.values()),
            "counting_rules": "Her pas, sahiplik geçişlerinden tek kayıt olarak üretilir ve pasörün takımına yazılır "
                              "(05_PASS_RULES §C). Aynı oyuncunun topu geri alması (top sürme) pas sayılmaz."}
