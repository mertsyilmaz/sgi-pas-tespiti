"""Sahiplik ve pas adayı tespiti (M6, docs/05_PASS_RULES.md §A). Gözlem katmanıdır: SUCCESS/FAIL kararı vermez.

Kare kare, yalnızca geçmiş karelerle çalışır (D-028). Bir pas, bitiş koşulu gerçekleştiği karede tamamlanır;
F-1 doldurması yalnızca o ana kadar görülmüş iki konum arasındadır (gelecek kare kullanılmaz).

Sahiplik (§A.1): topa en yakın oyuncu (hakem hariç, takımı belli) `radius_m` içinde art arda `min_frames` kare
kalırsa topun sahibi olur.
Pas (§A.2):
- Başlangıç: sahip varken top sahibinden `radius_m` dışına çıkar. Başlangıç karesi ve hücresi, topun sahibinin
  dibinde görüldüğü son karedir. Sahip net değilse (dibinde < min_frames kalmış biri) o oyuncu pasördür (F-5).
- Bitiş (ilk gerçekleşen): başka bir oyuncu sahip olur (bitiş karesi, onun dibine ilk girdiği kare) | top
  `stop_frames` boyunca `stop_speed_mps`'den yavaş ve sahipsiz | top saha dışı (OUT) | top `max_ball_gap_frames`'ten
  uzun görünmez (F-2: son görüldüğü kare).
- Aynı oyuncu topu geri alırsa (arada başka sahip yoksa) top sürmedir, kayıt üretilmez.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from sgi.grid.grid import PitchGrid
from sgi.types import CELL_OUT, CLS_BALL, TEAM_A, TEAM_B, CellId, ObjectState

END_RECEIVED, END_STOPPED, END_OUT, END_LOST = "received", "stopped", "out", "ball_lost"


@dataclass
class PlayerSnap:
    track_id: int
    team: str
    role: str
    pitch_xy: tuple[float, float]
    cell: CellId


@dataclass
class PassCandidate:
    frame_start: int
    frame_end: int
    passer_track_id: int
    passer_team: str
    start_cell: CellId
    target_cell: CellId
    end_reason: str                                   # received | stopped | out | ball_lost
    ball_start: tuple[float, float]
    ball_end: tuple[float, float]
    receiver_track_id: Optional[int] = None
    receiver_team: Optional[str] = None
    path_cells: list = field(default_factory=list)    # zaman sıralı, tekrarlar birleşik ("GAP": konum yok)
    ball_path: list = field(default_factory=list)     # [(kare, (x, y), hücre, doldurma mı)]
    end_players: list[PlayerSnap] = field(default_factory=list)   # bitiş karesindeki oyuncular (kurallar için)
    fallbacks: list[str] = field(default_factory=list)
    quality: dict = field(default_factory=dict)

    @property
    def intermediate_cells(self) -> list:
        return self.path_cells[1:-1]


class PassDetector:
    def __init__(self, grid: PitchGrid, fps: float, possession: dict, pass_end: dict, fallbacks: dict,
                 pass_definition: Optional[dict] = None) -> None:
        self.grid, self.fps = grid, fps
        self.radius = float(possession.get("radius_m", 1.5))
        self.min_frames = int(possession.get("min_frames", 3))
        self.roles = set(possession.get("roles", ["player", "goalkeeper"]))
        self.stop_speed = float(pass_end.get("stop_speed_mps", 1.5))
        self.stop_frames = int(pass_end.get("stop_frames", 10))
        self.speed_win = int(pass_end.get("speed_window_frames", 5))
        self.max_speed = float(pass_end.get("max_ball_speed_mps", 40.0))
        self.n_rejected_fast = 0
        self.run_speed = 9.0                          # m/sn: kimlik değişiminde aynı kişinin koşabileceği hız
        self.max_gap = int(fallbacks.get("max_ball_gap_frames", 15))
        self.dribble = bool((pass_definition or {}).get("same_player_regain_is_dribble", True))
        self.reset()

    @classmethod
    def from_config(cls, grid: PitchGrid, fps: float, cfg: dict[str, Any]) -> "PassDetector":
        return cls(grid, fps, cfg.get("possession", {}), cfg.get("pass_end", {}), cfg.get("fallbacks", {}),
                   cfg.get("pass_definition"))

    def reset(self) -> None:
        self.owner: Optional[int] = None              # onaylı sahip (track_id)
        self.owner_team: Optional[str] = None
        self._owner_last: Optional[tuple[int, tuple, CellId]] = None   # sahibin dibinde son görülme (kare, konum, hücre)
        self._streak: tuple[Optional[int], int, int] = (None, 0, -1)    # (oyuncu, art arda kare, ilk kare)
        self._flight: Optional[dict] = None           # devam eden pas
        self._last_ball: Optional[tuple[int, tuple]] = None   # son görülen top (kare, konum)
        self._players_at: dict[int, list[PlayerSnap]] = {}    # son karelerin oyuncuları (bitiş karesi için)
        self._slow: Optional[int] = None             # topun yavaşlamaya başladığı kare
        self._prev_near: Optional[tuple] = None       # önceki karede topun dibindeki oyuncu (F-5 için)
        self._last_seen: dict[int, tuple[int, tuple]] = {}   # oyuncu -> (son görüldüğü kare, konum)

    # --- yardımcılar ---
    def _players(self, objs: Sequence[ObjectState]) -> list[PlayerSnap]:
        # F-4: takımı belirlenemeyen oyuncu yok sayılır (takım zaten takip boyunca çoğunluk oyuyla gelir)
        return [PlayerSnap(o.track_id, o.team, o.role, o.pitch_xy, o.cell) for o in objs
                if o.role in self.roles and o.team in (TEAM_A, TEAM_B) and o.pitch_xy is not None] + \
               [PlayerSnap(o.track_id, o.team, o.role, o.pitch_xy, o.cell) for o in objs
                if o.role not in self.roles and o.role != CLS_BALL and o.pitch_xy is not None]

    def _nearest(self, p: np.ndarray, players: list[PlayerSnap]) -> tuple[Optional[PlayerSnap], float]:
        best, bd = None, float("inf")
        for pl in players:
            if pl.role not in self.roles:
                continue
            d = float(np.hypot(pl.pitch_xy[0] - p[0], pl.pitch_xy[1] - p[1]))
            if d < bd:
                best, bd = pl, d
        return best, bd

    def _speed(self) -> Optional[float]:
        path = [(f, p) for f, p, _, interp in self._flight["path"] if not interp] if self._flight else []
        if len(path) < 2:
            return None
        f1, p1 = path[-1]
        f0, p0 = next(((f, p) for f, p in reversed(path) if f1 - f >= self.speed_win), path[0])
        if f1 == f0:
            return None
        return float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]) * self.fps / (f1 - f0))

    def _start_flight(self, passer: int, team: str, start: tuple[int, tuple, CellId], fallbacks: list[str]) -> None:
        f, p, c = start
        present = {pl.track_id for pl in self._players_at.get(f, [])}
        self._flight = {"passer": passer, "team": team, "f0": f, "p0": p, "c0": c, "path": [(f, p, c, False)],
                        "fallbacks": list(fallbacks), "missing": 0, "present0": present}
        self._slow = None

    def _finish(self, frame_end: int, ball_end: tuple, cell_end: CellId, reason: str,
                receiver: Optional[PlayerSnap] = None, extra_fb: Sequence[str] = ()) -> PassCandidate:
        fl = self._flight
        path = [x for x in fl["path"] if x[0] <= frame_end]
        cells: list = []                   # boşluklar F-1 ile doldurulduğu için yol kesintisizdir
        for _, _, c, _ in path:
            if not cells or cells[-1] != c:
                cells.append(c)
        if cells[-1] != cell_end:
            cells.append(cell_end)
        n_interp = sum(1 for x in path if x[3])
        fb = fl["fallbacks"] + [x for x in extra_fb if x not in fl["fallbacks"]]
        players = self._players_at.get(frame_end, [])
        cand = PassCandidate(
            frame_start=fl["f0"], frame_end=frame_end, passer_track_id=fl["passer"], passer_team=fl["team"],
            start_cell=fl["c0"], target_cell=cell_end, end_reason=reason, ball_start=fl["p0"], ball_end=ball_end,
            receiver_track_id=receiver.track_id if receiver else None,
            receiver_team=receiver.team if receiver else None,
            path_cells=cells, ball_path=path, end_players=players, fallbacks=fb,
            quality={"interpolated_frames": n_interp, "ball_missing_frames": fl["missing"],
                     "duration_s": round((frame_end - fl["f0"]) / self.fps, 2)})
        self._flight = None
        self._prev_near = None
        return cand

    # --- ana adım ---
    def update(self, frame_idx: int, objs: Sequence[ObjectState]) -> list[PassCandidate]:
        """Bu karede tamamlanan pas adaylarını döndürür."""
        out: list[PassCandidate] = []
        players = self._players(objs)
        self._players_at[frame_idx] = players
        for q in players:
            self._last_seen[q.track_id] = (frame_idx, q.pitch_xy)
        for k in [k for k in self._players_at if k < frame_idx - 3 * self.max_gap - 10]:
            del self._players_at[k]
        ball = next((o for o in objs if o.role == CLS_BALL and o.pitch_xy is not None), None)

        if ball is None:
            fl = self._flight
            if fl is not None:
                fl["missing"] += 1
                if self._last_ball is not None and frame_idx - self._last_ball[0] > self.max_gap:
                    f, p = self._last_ball                         # F-2: son görüldüğü konum ve kare
                    out.append(self._finish(f, p, self.grid.cell_of(*p), END_LOST, extra_fb=["F-2"]))
            self._streak = (None, 0, -1)
            return out

        p = np.asarray(ball.pitch_xy, float)
        pt = (float(p[0]), float(p[1]))
        cell = self.grid.cell_of(*pt)
        # F-1: kısa boşluğu son görülen konumla bu konum arasında doğrusal doldur (yalnızca geçmiş + bu kare)
        if self._flight is not None and self._last_ball is not None and frame_idx - self._last_ball[0] > 1:
            f0, q0 = self._last_ball
            for f in range(f0 + 1, frame_idx):
                a = (f - f0) / (frame_idx - f0)
                q = (q0[0] + a * (pt[0] - q0[0]), q0[1] + a * (pt[1] - q0[1]))
                self._flight["path"].append((f, q, self.grid.cell_of(*q), True))
            if "F-1" not in self._flight["fallbacks"]:
                self._flight["fallbacks"].append("F-1")
        self._last_ball = (frame_idx, pt)

        near, d = self._nearest(p, players)
        near_id = near.track_id if near is not None and d <= self.radius else None
        sid, n, s0 = self._streak
        self._streak = (near_id, n + 1, s0) if near_id is not None and near_id == sid else \
            ((near_id, 1, frame_idx) if near_id is not None else (None, 0, -1))
        sid, n, s0 = self._streak
        # süre kare aralığıyla ölçülür (sonuç sayısıyla değil): çevrimdışında her karede sonuç olduğu için ikisi
        # aynıdır; canlıda nesne sonuçları seyrek gelir (~2 karede bir) ve eşikler yine aynı süreye karşılık gelir
        confirmed = sid if sid is not None and frame_idx - s0 + 1 >= self.min_frames else None

        if self._flight is not None:
            fl = self._flight
            fl["path"].append((frame_idx, pt, cell, False))
            if cell == CELL_OUT:
                out.append(self._finish(frame_idx, pt, cell, END_OUT))
                self.owner = None
            elif confirmed is not None:
                now = {pl.track_id for pl in players}
                # takip kimliği değişimi: pasör artık görünmüyor ve "alıcı" başlangıçta yoktu -> aynı kişi
                # ve top neredeyse yer değiştirmedi (uzun pasta kamera döner: pasör çıkar, alıcı girer; o pas silinmez)
                # ve "alıcı", pasörün son görüldüğü yere koşarak ulaşılabilecek mesafede (top süren oyuncunun
                # kimliği değişir; uzun pasta ise alıcı uzaktadır)
                lf, lp = self._last_seen.get(fl["passer"], (frame_idx, (np.inf, np.inf)))
                rp = recv_now.pitch_xy if (recv_now := next((q for q in players if q.track_id == confirmed),
                                                              None)) else (np.inf, np.inf)
                reach = 2.0 + self.run_speed * (frame_idx - lf) / self.fps
                id_switch = (fl["passer"] not in now and confirmed not in fl["present0"]
                             and float(np.hypot(rp[0] - lp[0], rp[1] - lp[1])) <= reach)
                if (confirmed == fl["passer"] or id_switch) and self.dribble:
                    self._flight = None                            # aynı oyuncu geri aldı: top sürme, kayıt yok
                    self._prev_near = None
                else:
                    recv = next(pl for pl in self._players_at.get(frame_idx, []) if pl.track_id == confirmed)
                    bf = next((x for x in fl["path"] if x[0] == s0), (frame_idx, pt, cell, False))
                    dist = float(np.hypot(bf[1][0] - fl["p0"][0], bf[1][1] - fl["p0"][1]))
                    if dist * self.fps / max(s0 - fl["f0"], 1) > self.max_speed:
                        # top fiziksel olarak imkânsız hızda "gitti": ölçüm sıçraması (üst üste görünen iki
                        # oyuncu arasında top konumu atlıyor), pas değil
                        self._flight = None
                        self._prev_near = None
                        self.n_rejected_fast += 1
                    else:
                        out.append(self._finish(s0, bf[1], bf[2], END_RECEIVED, receiver=recv))
                self.owner, self.owner_team = confirmed, near.team
                self._owner_last = (frame_idx, pt, cell)
            else:
                v = self._speed()
                slow = v is not None and v < self.stop_speed and near_id is None
                self._slow = (frame_idx if self._slow is None else self._slow) if slow else None
                if self._slow is not None and frame_idx - self._slow + 1 >= self.stop_frames:
                    f_stop = self._slow                                     # yavaşlamanın başladığı kare
                    bf = next((x for x in fl["path"] if x[0] >= f_stop), (frame_idx, pt, cell, False))
                    out.append(self._finish(bf[0], bf[1], bf[2], END_STOPPED))
            return out

        # pas yok: sahiplik ve başlangıç
        if confirmed is not None:
            self.owner, self.owner_team = confirmed, near.team
        if self.owner is not None:
            # pas, top sahibinden gerçekten ayrılınca başlar (§A.2): yanındaki oyuncunun ölçüm gürültüsüyle
            # "daha yakın" görünmesi yetmez (yan yana duran iki oyuncu arasında sahte pas zinciri oluşuyordu)
            owner_pl = next((pl for pl in players if pl.track_id == self.owner), None)
            d_owner = (float(np.hypot(owner_pl.pitch_xy[0] - pt[0], owner_pl.pitch_xy[1] - pt[1]))
                       if owner_pl is not None else None)
            if d_owner is not None and d_owner <= self.radius:
                self._owner_last = (frame_idx, pt, cell)
            else:
                if owner_pl is not None and self._owner_last is not None:
                    self._start_flight(self.owner, self.owner_team, self._owner_last, [])
                    self._flight["path"].append((frame_idx, pt, cell, False))
                    self.owner = None
        elif sid is None and self._prev_near is not None:
            # F-5: sahip onaylanmadan (dibinde < min_frames) top ayrıldı -> son dibindeki oyuncu pasör
            pid, team, start = self._prev_near
            self._start_flight(pid, team, start, ["F-5"])
            self._flight["path"].append((frame_idx, pt, cell, False))
        self._prev_near = (near.track_id, near.team, (frame_idx, pt, cell)) if near_id is not None else None
        return out
