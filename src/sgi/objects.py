"""Nesne hattı: tespit (M2) -> takip (M3) -> rol/takım (M3b) -> saha konumu ve hücre (M4, M5). D-022.

Kalibrasyondan bağımsız çalışır; homografi yoksa nesneler görüntüde tespit edilir ama saha konumu/hücre
atanmaz (UNKNOWN). Saha konumu varsa, ayak noktası sahanın `pitch_margin_m` dışında kalan kişiler
(seyirci, yedek kulübesi) atılır; yan hakemler taç çizgisinde olduğu için pay bırakılır.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

from sgi.grid.grid import PitchGrid
from sgi.mapping.homography import apply_h, pitch_to_image
from sgi.registry import build
from sgi.tracking.ball import BallCandidate, BallTracker, PlayerRef
from collections import Counter, defaultdict, deque

from sgi.types import (CELL_UNKNOWN, CLS_BALL, CLS_GOALKEEPER, CLS_PERSON, CLS_PLAYER, CLS_REFEREE, ObjectState,
                       TEAM_NONE)

BALL_TRACK_ID = 0
PERSON_CLASSES = {CLS_PERSON, CLS_PLAYER, CLS_GOALKEEPER, CLS_REFEREE}


class ObjectPipeline:
    def __init__(self, cfg: dict[str, Any], grid: PitchGrid, fps: float, root: Path) -> None:
        self.grid = grid
        self.margin = float(cfg.get("pitch_margin_m", 3.0))
        self.detector = build("object_detector", cfg["detector"], root=root)
        self.tracker = build("tracker", cfg["tracker"], frame_rate=fps)
        self.teams = build("team_assigner", cfg["team_assigner"], pitch_length=grid.length, pitch_width=grid.width)
        self.ball = BallTracker(fps, grid.length, grid.width, **(cfg.get("ball_tracker") or {}))
        pose = dict(cfg.get("pose") or {})
        # poz (D-035): yalnızca top uzun süre oyuncuda gizli kalınca, o oyuncunun kırpıntısında çalışır
        self.pose = build("pose_estimator", pose, root=root) if pose.pop("enabled", False) else None
        self._t = -1
        self._role_votes: dict[int, deque] = defaultdict(lambda: deque(maxlen=30))

    def reset(self) -> None:
        self.tracker.reset()
        self.ball.reset()
        self._t = -1

    def _to_pitch(self, H: Optional[np.ndarray], xy) -> Optional[tuple[float, float]]:
        if H is None:
            return None
        p = apply_h(H, np.array([xy], dtype=np.float64))[0]
        return float(p[0]), float(p[1])

    def _on_pitch(self, p: Optional[tuple[float, float]], margin: float) -> bool:
        return p is None or (-margin <= p[0] <= self.grid.length + margin and -margin <= p[1] <= self.grid.width + margin)

    def process(self, frame: np.ndarray, H: Optional[np.ndarray], frame_idx: Optional[int] = None) -> list[ObjectState]:
        """Nedensel: yalnızca bu ve önceki kareler kullanılır (D-028). `frame_idx` canlı modda seyrek çağrıda
        top hızının doğru hesaplanması içindir; verilmezse ardışık kare varsayılır."""
        self._t = self._t + 1 if frame_idx is None else frame_idx
        dets = self.detector.detect(frame)
        persons = [d for d in dets if d.cls in PERSON_CLASSES]
        balls = [d for d in dets if d.cls == CLS_BALL]

        # saha dışındakileri (seyirci, kulübe) takipten önce at
        persons = [d for d in persons if self._on_pitch(self._to_pitch(H, d.foot), self.margin)]
        tracked = self.tracker.update(persons, frame)
        pitch = [self._to_pitch(H, d.foot) for d in tracked]
        for d in tracked:
            self._role_votes[d.track_id].append(d.cls)
        roles = [self._role(d.track_id) for d in tracked]
        self.teams.observe(frame, tracked, pitch, roles)

        out = []
        for d, p, det_role in zip(tracked, pitch, roles):
            if det_role == CLS_PERSON:                       # dedektör rol vermiyor (COCO): renk sezgiseli
                role, team = self.teams.assign(d.track_id)
            elif det_role == CLS_REFEREE:
                role, team = CLS_REFEREE, TEAM_NONE
            elif det_role == CLS_GOALKEEPER:
                role, team = CLS_GOALKEEPER, self.teams.goalkeeper_team(d.track_id)
            elif self.teams.is_other(d.track_id):
                # dedektör "oyuncu" dese de forma iki takıma da uymuyor: hakem ya da kaleci. Konuma göre ayır
                # (ceza sahasında kalıyorsa kaleci). Futbol modeli sarı formalı hakemi ve kaleciyi zaman zaman
                # "oyuncu" sınıflandırıyor.
                role, team = self.teams.assign(d.track_id)
                if role == CLS_PLAYER:
                    role, team = CLS_REFEREE, TEAM_NONE
            else:
                role, team = CLS_PLAYER, self.teams.team_of(d.track_id)
            cell = self.grid.cell_of(*p) if p is not None else CELL_UNKNOWN
            out.append(ObjectState(d.track_id, role, team, d.bbox, d.conf, p, cell))

        ball = self._track_ball(frame, H, balls, out)
        if ball is not None:
            out.append(ball)
        return out

    def _track_ball(self, frame, H, balls, people: list[ObjectState]) -> Optional[ObjectState]:
        cands = []
        for b in balls:
            p = self._to_pitch(H, b.foot)
            if self._on_pitch(p, 2.0):
                cands.append(BallCandidate(p, b.foot, b.conf, b.bbox))
        players = [PlayerRef(o.track_id, o.pitch_xy, ((o.bbox[0] + o.bbox[2]) / 2, o.bbox[3]))
                   for o in people if o.role in (CLS_PLAYER, CLS_GOALKEEPER)]
        facing_fn = None
        if self.pose is not None:
            boxes = {o.track_id: o.bbox for o in people}
            facing_fn = lambda tid: self.pose.facing(frame, boxes[tid]) if tid in boxes else None  # noqa: E731
        est = self.ball.update(self._t, cands, players, H is not None, facing_fn)
        if est is None:
            return None
        bbox = est.bbox
        if bbox is None and est.pitch_xy is not None:        # tahmini konum: görüntüye geri izdüşür
            h, w = frame.shape[:2]
            q = apply_h(pitch_to_image(H, (w, h)), np.array([est.pitch_xy], dtype=np.float64))[0]
            bbox = (q[0] - 5, q[1] - 5, q[0] + 5, q[1] + 5)
        if bbox is None:
            return None
        p = est.pitch_xy
        return ObjectState(BALL_TRACK_ID, CLS_BALL, TEAM_NONE, bbox, est.conf, p,
                           self.grid.cell_of(*p) if p is not None else CELL_UNKNOWN, est.state)

    def _role(self, track_id: int) -> str:
        """Dedektörün verdiği rol, takip boyunca çoğunluk oyuyla (tek karelik sınıf hatası rolü değiştirmez)."""
        return Counter(self._role_votes[track_id]).most_common(1)[0][0]

    def team_colors(self) -> dict:
        return self.teams.team_colors_bgr() if hasattr(self.teams, "team_colors_bgr") else {}


# --- diske yazma / okuma (FrameState nesneleri) ---
ROLE_CODES = ["player", "goalkeeper", "referee", "ball"]
STATE_CODES = ["detected", "held", "predicted"]
N_COLS = 13
TEAM_CODES = ["A", "B", "-", "?"]


def objects_to_rows(frame_i: int, objs: list[ObjectState]) -> list[list[float]]:
    rows = []
    for o in objs:
        px, py = o.pitch_xy if o.pitch_xy is not None else (np.nan, np.nan)
        cell = o.cell if isinstance(o.cell, int) else -1
        rows.append([frame_i, o.track_id, ROLE_CODES.index(o.role), TEAM_CODES.index(o.team), *o.bbox, o.conf,
                     px, py, cell, STATE_CODES.index(o.state)])
    return rows


def rows_to_objects(rows: np.ndarray) -> list[ObjectState]:
    out = []
    for r in rows:
        p = None if np.isnan(r[9]) else (float(r[9]), float(r[10]))
        cell = int(r[11]) if r[11] >= 0 else CELL_UNKNOWN
        state = STATE_CODES[int(r[12])] if len(r) > 12 else "detected"
        out.append(ObjectState(int(r[1]), ROLE_CODES[int(r[2])], TEAM_CODES[int(r[3])], tuple(r[4:8]), float(r[8]),
                               p, cell, state))
    return out
