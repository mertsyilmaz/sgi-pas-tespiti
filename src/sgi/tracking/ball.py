"""Top takibi (D-028). Yalnızca geçmiş kareleri kullanır (nedensel): canlı ve çevrimdışı aynı sonucu verir.

Dedektör topu her karede bulamaz (hızlı pasta bulanıklık, oyuncu ayağının arkasında kalma). Takip, tespitler
arasında topun nerede olduğunu sabit kurallarla kestirir. Konum saha koordinatındadır (metre): kamera hareketi
homografi içinde zaten telafi edilmiş olur.

Durumlar:
- detected : bu karede dedektör buldu ve takip kapısından (gating) geçti
- held     : top kayboldu, son görüldüğünde bir oyuncunun dibindeydi -> topu o oyuncunun ayağında say.
             hold_max_s dolunca, oyuncunun sırtı kameraya dönükse (poz, D-035) hold_back_max_s'ye kadar uzar:
             top gövdenin arkasında saklı kabul edilir. Yön pose_every karede bir yeniden bakılır.
- predicted: top kayboldu, sahibi yok (pas/şut yolunda) -> son hızla kısa süre ilerlet
Sahte tespitler:
- Saha işaretleri (penaltı noktaları, orta nokta) beyaz ve top boyutundadır. Bu noktalardaki aday, ancak dibinde
  (1 m) oyuncu varsa ve top oraya hareketle geldiyse kabul edilir.
- Takip varken kapı dışındaki adaylar yok sayılır; aynı yerde art arda güçlü tespit gelirse takip oraya geçer.
Topun havada olması (homografi yalnızca zemin içindir) konumu bozar; bu bir kısıttır (13_LIMITATIONS).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

DETECTED, HELD, PREDICTED = "detected", "held", "predicted"


@dataclass
class BallCandidate:
    pitch_xy: Optional[tuple[float, float]]
    image_xy: tuple[float, float]
    conf: float
    bbox: tuple[float, float, float, float]


@dataclass
class PlayerRef:
    track_id: int
    pitch_xy: Optional[tuple[float, float]]
    foot_xy: tuple[float, float]                  # görüntüde ayak noktası


@dataclass
class BallEstimate:
    state: str
    pitch_xy: Optional[tuple[float, float]]
    image_xy: Optional[tuple[float, float]]
    conf: float
    bbox: Optional[tuple[float, float, float, float]] = None
    holder_id: int = -1                           # held: topu tuttuğu varsayılan oyuncu


class BallTracker:
    def __init__(self, fps: float, pitch_length: float = 105.0, pitch_width: float = 68.0,
                 gate_base_m: float = 2.0, max_speed_mps: float = 35.0, max_gate_m: float = 40.0,
                 possess_m: float = 1.5, hold_max_s: float = 3.0, predict_max_s: float = 0.5,
                 mark_radius_m: float = 1.5, mark_player_m: float = 1.0, switch_after: int = 3,
                 far_conf: float = 0.3, static_s: float = 1.2, static_r_m: float = 1.0, static_player_m: float = 2.0,
                 learned_r_m: float = 2.0, learned_ttl_s: float = 20.0, hold_back_max_s: float = 8.0,
                 pose_every: int = 5, **_: object) -> None:
        self.far_conf = far_conf
        self.hold_back_frames = int(round(hold_back_max_s * fps))
        self.pose_every = pose_every
        self.n_pose_checks = self.n_pose_extended = 0
        # öğrenilen sabit işaretler (D-032): kalibrasyon kayınca penaltı noktası bilinen yerinden (marks) uzağa
        # düşer ve filtreden kaçar. Yanında oyuncu olmadan static_s boyunca static_r içinde kıpırdamayan "top"
        # saha işaretidir; konumu öğrenilir ve bilinen işaretler gibi elenir. Görülmeye devam ettikçe tazelenir.
        self.static_frames = int(round(static_s * fps))
        self.static_r, self.static_player = static_r_m, static_player_m
        self.learned_r, self.learned_ttl = learned_r_m, int(round(learned_ttl_s * fps))
        self._learned: list[list] = []            # [konum (saha), son görülme karesi]
        self.fps = fps
        self.L, self.W = pitch_length, pitch_width
        self.gate_base, self.vmax, self.gate_max = gate_base_m, max_speed_mps, max_gate_m
        self.possess_m, self.mark_r, self.mark_player = possess_m, mark_radius_m, mark_player_m
        self.hold_frames = int(round(hold_max_s * fps))
        self.predict_frames = int(round(predict_max_s * fps))
        self.switch_after = switch_after
        cy = pitch_width / 2
        self.marks = np.array([(11.0, cy), (pitch_length - 11.0, cy), (pitch_length / 2, cy)])
        self._now = 0
        self.reset()

    def reset(self) -> None:
        self._p: Optional[np.ndarray] = None      # son tespit edilen konum (saha)
        self._v = np.zeros(2)                     # hız (m/sn)
        self._t: Optional[int] = None             # son tespit karesi
        self._holder = -1                         # son tespitte topun dibindeki oyuncu
        self._img: Optional[tuple[float, float]] = None   # saha konumu yoksa görüntü konumu
        self._challenger: Optional[np.ndarray] = None
        self._challenger_n = 0
        self._recent_holders: dict[int, int] = {}     # oyuncu -> topu en son tuttuğu kare
        self._static: list[tuple[int, np.ndarray]] = []  # oyuncusuz, kıpırdamayan kabul edilmiş tespitler
        self._hold_limit = self.hold_frames       # sırtı dönük sahipte poz ile uzar (D-035)
        self._pose_t: Optional[int] = None        # son yön kontrolü karesi

    # --- yardımcılar ---
    def _gate(self, dt: int) -> float:
        return min(self.gate_max, self.gate_base + self.vmax * dt / self.fps)

    def _predict(self, t: int) -> Optional[np.ndarray]:
        if self._p is None or self._t is None:
            return None
        dt = min(t - self._t, self.predict_frames) / self.fps
        return self._p + self._v * dt

    def _is_mark_artifact(self, p: np.ndarray, players: Sequence[PlayerRef]) -> bool:
        """Saha işaretindeki (penaltı/orta nokta) aday, ancak dibinde oyuncu varsa VE takip edilen top oraya
        başka yerden hareketle geldiyse kabul edilir. İşaret kıpırdamaz; gerçek top oyuncuyla birlikte hareket eder.
        Kısıt: işaretin üzerinde duran gerçek top (ör. penaltı atışı öncesi) vurulana kadar görülmez (13_LIMITATIONS)."""
        marks = list(self.marks) + [m[0] for m in self._learned]
        radii = [self.mark_r] * len(self.marks) + [self.learned_r] * len(self._learned)
        d = np.linalg.norm(np.array(marks) - p, axis=1)
        k = int(np.argmin(d / np.array(radii)))
        if d[k] > radii[k]:
            return False
        if k >= len(self.marks):                  # öğrenilen işaret yeniden görüldü: tazele, kaymayı izle
            m = self._learned[k - len(self.marks)]
            m[0], m[1] = 0.8 * m[0] + 0.2 * p, self._now
        # top hemen yakından (<= 3 m) noktaya doğru hareket etmiş olmalı; uzaktaki takip "gelmiş" sayılmaz
        dm = np.linalg.norm(self._p - marks[k]) if self._p is not None else 0.0
        moving_in = 0.5 * radii[k] < dm <= 3.0
        player_at = any(pl.pitch_xy is not None and np.hypot(pl.pitch_xy[0] - p[0], pl.pitch_xy[1] - p[1]) <= self.mark_player
                        for pl in players)
        return not (moving_in and player_at)

    def _nearest_player(self, p: np.ndarray, players: Sequence[PlayerRef]) -> tuple[int, float]:
        best, bd = -1, float("inf")
        for pl in players:
            if pl.pitch_xy is None:
                continue
            d = float(np.hypot(pl.pitch_xy[0] - p[0], pl.pitch_xy[1] - p[1]))
            if d < bd:
                best, bd = pl.track_id, d
        return best, bd

    def _near_recent_holder(self, p: np.ndarray, players: Sequence[PlayerRef], t: int) -> bool:
        for pl in players:
            last = self._recent_holders.get(pl.track_id)
            if (last is not None and t - last <= self.fps and pl.pitch_xy is not None
                    and np.hypot(pl.pitch_xy[0] - p[0], pl.pitch_xy[1] - p[1]) <= self.possess_m):
                return True
        return False

    def _holder_pos(self, players: Sequence[PlayerRef]) -> Optional[np.ndarray]:
        for pl in players:
            if pl.track_id == self._holder >= 0 and pl.pitch_xy is not None:
                return np.array(pl.pitch_xy)
        return None

    def _accept(self, c: BallCandidate, t: int, players: Sequence[PlayerRef]) -> BallEstimate:
        p = np.array(c.pitch_xy)
        if self._p is not None and self._t is not None and 0 < t - self._t <= self.predict_frames:
            v_new = (p - self._p) * self.fps / (t - self._t)
            self._v = 0.5 * self._v + 0.5 * v_new
        else:
            self._v = np.zeros(2)
        self._p, self._t, self._img = p, t, c.image_xy
        self._hold_limit, self._pose_t = self.hold_frames, None
        holder, d = self._nearest_player(p, players)
        self._holder = holder if d <= self.possess_m else -1
        if self._holder >= 0:
            self._recent_holders[self._holder] = t
        self._challenger, self._challenger_n = None, 0
        if self._learn_static(p, t, d):
            return None
        return BallEstimate(DETECTED, (float(p[0]), float(p[1])), c.image_xy, c.conf, c.bbox)

    def _learn_static(self, p: np.ndarray, t: int, d_player: float) -> bool:
        """Kabul edilen tespit oyuncusuz ve kıpırdamadan static_s sürdüyse: saha işaretidir, öğren ve bırak."""
        if d_player <= self.static_player or (self._static and np.linalg.norm(p - self._static[0][1]) > self.static_r):
            self._static = []
        if d_player <= self.static_player:
            return False
        self._static.append((t, p))
        if len(self._static) >= 3 and t - self._static[0][0] >= self.static_frames:
            self._learned.append([np.mean([q for _, q in self._static], axis=0), t])
            self.reset()
            return True
        return False

    # --- ana adım ---
    def _check_back(self, t: int, players: Sequence[PlayerRef],
                    facing_fn: Optional[Callable[[int], Optional[str]]]) -> None:
        """Oyuncuda sayma süresi dolduysa sahibinin yönüne bak: sırtı dönükse süre uzar, değilse biter."""
        if self._t is None or self._holder < 0 or t - self._t <= self.hold_frames or facing_fn is None:
            return
        if self._pose_t is not None and t - self._pose_t < self.pose_every:
            return
        if not any(pl.track_id == self._holder for pl in players):
            return                                 # sahibi bu karede yok: karar verilemez, süre uzamaz
        self._pose_t = t
        self.n_pose_checks += 1
        back = facing_fn(self._holder) == "back"
        if back and self._hold_limit == self.hold_frames:
            self.n_pose_extended += 1
        self._hold_limit = self.hold_back_frames if back else self.hold_frames

    def update(self, t: int, cands: Sequence[BallCandidate], players: Sequence[PlayerRef],
               H_available: bool, facing_fn: Optional[Callable[[int], Optional[str]]] = None
               ) -> Optional[BallEstimate]:
        """facing_fn(track_id) -> 'back' / 'front' / None: poz modeli (D-035). Yalnızca süre dolunca çağrılır."""
        if not H_available:
            return self._update_image_only(t, cands)
        self._now = t
        self._learned = [m for m in self._learned if t - m[1] <= self.learned_ttl]
        self._check_back(t, players, facing_fn)

        active = self._t is not None and self._p is not None and t - self._t <= self._hold_limit
        if not active:
            self.reset()
        cands = [c for c in cands if c.pitch_xy is not None
                 and not self._is_mark_artifact(np.array(c.pitch_xy), players)]

        if cands:
            if self._p is None:
                return self._accept(max(cands, key=lambda c: c.conf), t, players)
            pred = self._predict(t)
            holder = self._holder_pos(players)
            if holder is not None:                 # top oyuncudaydı: oyuncuyla birlikte hareket etmiş olabilir
                pred = holder
            gate = self._gate(t - self._t)
            dist = [float(np.linalg.norm(np.array(c.pitch_xy) - pred)) for c in cands]
            # düşük güvenli aday uzağa sıçratamaz (ör. krampon); son 1 sn'de topu tutan oyuncunun dibindeki aday
            # ise kapıdan bağımsız kabul edilir (top kısa süre başka bir yerde sanıldıysa hızlı geri dönüş)
            near_holder = [self._near_recent_holder(np.array(c.pitch_xy), players, t) for c in cands]
            inside = [(d / gate - 0.5 * c.conf - nh, c) for d, c, nh in zip(dist, cands, near_holder)
                      if nh or (d <= gate and (d <= 2 * self.gate_base or c.conf >= self.far_conf))]
            if inside:
                return self._accept(min(inside, key=lambda x: x[0])[1], t, players)
            # kapı dışında: tek seferlik sahte tespit olabilir; aynı yerde tekrarlanırsa takip oraya geçer
            best = max(cands, key=lambda c: c.conf)
            bp = np.array(best.pitch_xy)
            if self._challenger is not None and np.linalg.norm(bp - self._challenger) <= self._gate(1) * 2:
                self._challenger_n += 1
            else:
                self._challenger, self._challenger_n = bp, 1
            if self._challenger_n >= self.switch_after:
                self.reset()
                return self._accept(best, t, players)

        return self._coast(t, players)

    def _coast(self, t: int, players: Sequence[PlayerRef]) -> Optional[BallEstimate]:
        """Bu karede kabul edilen tespit yok: sahibine bağla ya da kısa süre ilerlet."""
        if self._t is None or self._p is None:
            return None
        dt = t - self._t
        if self._holder >= 0 and dt <= self._hold_limit:
            for pl in players:
                if pl.track_id == self._holder and pl.pitch_xy is not None:
                    fx, fy = pl.foot_xy
                    return BallEstimate(HELD, pl.pitch_xy, (fx, fy), 0.0, (fx - 4, fy - 8, fx + 4, fy), self._holder)
            return None                            # sahibi kadrajda yok
        if self._holder < 0 and dt <= self.predict_frames:
            p = self._predict(t)
            if -2 <= p[0] <= self.L + 2 and -2 <= p[1] <= self.W + 2:
                return BallEstimate(PREDICTED, (float(p[0]), float(p[1])), None, 0.0)
        return None

    def _update_image_only(self, t: int, cands: Sequence[BallCandidate]) -> Optional[BallEstimate]:
        """Homografi yok: saha konumu, hız ve işaret filtresi kullanılamaz; son görüntü konumuna en yakın aday."""
        self._p, self._v, self._holder = None, np.zeros(2), -1
        if not cands:
            return None
        if self._img is not None and len(cands) > 1:
            lx, ly = self._img
            best = min(cands, key=lambda c: (c.image_xy[0] - lx) ** 2 + (c.image_xy[1] - ly) ** 2 - 1e4 * c.conf)
        else:
            best = max(cands, key=lambda c: c.conf)
        self._img, self._t = best.image_xy, t
        return BallEstimate(DETECTED, None, best.image_xy, best.conf, best.bbox)
