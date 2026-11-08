"""Canlı modda nesne kutularının akıcı çizimi (yalnızca gösterim).

Canlıda nesne dedektörü saha modeliyle aynı GPU'yu paylaştığı için sonuç saniyede ~3–4 kez ve ~8 kare geç gelir.
Önceden kutular yalnızca kamera hareketi kadar kaydırılıyordu: oyuncu koşarken kutu yerinde kalıyor, yeni sonuç
gelince sıçrıyordu (takılma hissi). Burada her kimlik için son iki sonuçtan kamera hareketinden arındırılmış hız
kestirilir ve kutu aradaki karelerde bu hızla ilerletilir. Yeni sonuç gelince kalan fark birkaç karede söndürülür.

Nedenseldir (D-028): yalnızca gelmiş sonuçlar kullanılır. Pas/hücre kararlarını etkilemez; çevrimdışı kayıtlar
dedektörün gerçek çıktısıdır.
"""
from __future__ import annotations

from dataclasses import replace
from collections import deque
from typing import Callable, Optional, Sequence

import cv2
import numpy as np

from sgi.types import ObjectState


def _warp(T: np.ndarray, x: float, y: float) -> np.ndarray:
    p = T @ np.array([x, y, 1.0])
    return p[:2] / p[2]


class LiveObjectSmoother:
    def __init__(self, horizon: int = 15, decay: float = 0.5, speed_ema: float = 0.5,
                 max_speed_h: float = 0.25, max_ball_px: float = 60.0, ball_extrapolate: bool = False) -> None:
        # horizon: en fazla bu kadar kare ileriye kestirilir (sonuç gecikirse kutu sonsuza kaymasın)
        # decay: yeni sonuçtaki sıçramanın her karede kalan oranı
        # max_speed_h: oyuncu için kare başına en fazla kutu yüksekliğinin bu oranı kadar hareket (9 m/sn ~ 0,2)
        self.horizon, self.decay, self.ema = horizon, decay, speed_ema
        self.max_speed_h, self.max_ball_px = max_speed_h, max_ball_px
        # top 7-8 karede yön değiştirebilir (pas, sekme): iki seyrek sonuçtan hız kestirimi çizimi kötüleştiriyor
        self.ball_extrapolate = ball_extrapolate
        self.reset()

    def reset(self) -> None:
        self._k: Optional[int] = None
        self._objs: list[ObjectState] = []
        self._v: dict[int, np.ndarray] = {}          # kimlik -> hız (px/kare, kamera hareketinden arındırılmış)
        self._res: dict[int, np.ndarray] = {}        # kimlik -> söndürülen sıçrama (px)
        self._shown: dict[int, np.ndarray] = {}      # kimlik -> son çizilen ayak noktası
        self._last_idx: Optional[int] = None

    @staticmethod
    def _foot(o: ObjectState) -> tuple[float, float]:
        x1, y1, x2, y2 = o.bbox
        return 0.5 * (x1 + x2), y2

    def _cap(self, o: ObjectState) -> float:
        return self.max_ball_px if o.role == "ball" else self.max_speed_h * max(1.0, o.bbox[3] - o.bbox[1])

    def _ball_owner(self, ball: ObjectState) -> Optional[ObjectState]:
        """Sonuç karesinde ayak noktası topa en yakın oyuncu (kutu yüksekliğinin yarısı içinde)."""
        bx, by = self._foot(ball)
        best, bd = None, float("inf")
        for o in self._objs:
            if o.role == "ball":
                continue
            fx, fy = self._foot(o)
            d = np.hypot(fx - bx, fy - by)
            if d <= max(15.0, 0.5 * (o.bbox[3] - o.bbox[1])) and d < bd:
                best, bd = o, d
        return best

    def _predict(self, o: ObjectState, T: np.ndarray, dt: int) -> np.ndarray:
        p = _warp(T, *self._foot(o))
        v = self._v.get(o.track_id)
        if v is not None and (o.role != "ball" or self.ball_extrapolate):
            p = p + v * min(dt, self.horizon)
        return p

    def update(self, idx: int, objs: Optional[Sequence[ObjectState]], k: Optional[int],
               transform: Callable[[int], Optional[np.ndarray]],
               H_img2pitch: Optional[np.ndarray] = None,
               flow: Optional[dict[int, np.ndarray]] = None) -> Optional[list[ObjectState]]:
        """idx: çizilecek kare; objs/k: elimizdeki en güncel sonuç ve ait olduğu kare;
        transform(k): k karesinin görüntü koordinatlarını idx karesine taşıyan 3x3 (bilinmiyorsa None);
        flow: kimlik -> k'den idx'e kutunun optik akışla ölçülen yer değiştirmesi (BoxFlow). Varsa hız
        kestiriminin yerine geçer: oyuncunun gerçekten nereye gittiği görüntüden okunur."""
        flow = flow or {}
        if objs is None or k is None:
            return None
        if self._last_idx is not None and idx < self._last_idx:      # geri sarma
            self.reset()
        steps = 0 if self._last_idx is None else idx - self._last_idx
        self._last_idx = idx
        for tid in self._res:
            self._res[tid] = self._res[tid] * self.decay ** steps

        T_new = transform(k)
        if k != self._k:                                             # yeni sonuç geldi
            T_old = transform(self._k) if self._k is not None else None
            old = {o.track_id: o for o in self._objs}
            v_new: dict[int, np.ndarray] = {}
            res_new: dict[int, np.ndarray] = {}
            for o in objs:
                po = old.get(o.track_id)
                if T_new is not None and T_old is not None and po is not None and po.role == o.role and k > self._k:
                    v = (_warp(T_new, *self._foot(o)) - _warp(T_old, *self._foot(po))) / (k - self._k)
                    if np.linalg.norm(v) <= self._cap(o):
                        prev = self._v.get(o.track_id)
                        v_new[o.track_id] = v if prev is None else self.ema * v + (1 - self.ema) * prev
                    # sınırı aşan hız: kimlik değişimi ya da sahte tespit -> hız bilinmiyor
                if T_new is not None and o.track_id in self._shown:
                    if o.track_id in flow:
                        pred = np.array(self._foot(o)) + flow[o.track_id]
                    else:
                        pred = _warp(T_new, *self._foot(o))
                        if o.track_id in v_new and (o.role != "ball" or self.ball_extrapolate):
                            pred = pred + v_new[o.track_id] * min(idx - k, self.horizon)
                    jump = self._shown[o.track_id] - pred
                    # top: sönümleme topu geride bırakıyordu (ölçüldü); büyük fark: düzeltme değil, sıçra
                    if o.role != "ball" and np.linalg.norm(jump) <= 2 * self._cap(o) + 10:
                        res_new[o.track_id] = jump
            self._k, self._objs, self._v, self._res = k, list(objs), v_new, res_new

        if T_new is None:                                            # kamera hareketi bilinmiyor: ham sonuç
            self._shown.clear()
            return list(self._objs)
        base = {}
        for o in self._objs:
            if o.role != "ball":
                base[o.track_id] = (np.array(self._foot(o)) + flow[o.track_id] if o.track_id in flow
                                    else self._predict(o, T_new, idx - k))
        for o in self._objs:
            if o.role == "ball":                  # ayağındaki oyuncu varsa onunla birlikte taşınır
                owner = self._ball_owner(o)
                p = _warp(T_new, *self._foot(o))
                if owner is not None and owner.track_id in base:
                    p = p + base[owner.track_id] - _warp(T_new, *self._foot(owner))
                base[o.track_id] = p
        out = []
        shown = {}
        for o in self._objs:
            p = base[o.track_id] + self._res.get(o.track_id, 0.0)
            shown[o.track_id] = p
            x1, y1, x2, y2 = o.bbox
            # kutu boyu kamera zoom'uyla ölçeklenir
            a, b = _warp(T_new, x1, y1), _warp(T_new, x2, y2)
            hw, hh = 0.5 * (b[0] - a[0]), b[1] - a[1]
            pitch = o.pitch_xy
            if H_img2pitch is not None and o.pitch_xy is not None:
                q = H_img2pitch @ np.array([p[0], p[1], 1.0])
                if q[2] != 0:
                    pitch = (float(q[0] / q[2]), float(q[1] / q[2]))
            out.append(replace(o, bbox=(float(p[0] - hw), float(p[1] - hh), float(p[0] + hw), float(p[1])),
                               pitch_xy=pitch))
        self._shown = shown
        return out


class BoxFlow:
    """Kutuların kare kare optik akışla (LK) takibi; nesne sonuçları arasındaki boşluğu doldurur.

    Son kareler yarım çözünürlükte gri olarak saklanır. k karesine ait sonuç geç gelince kutulardaki noktalar
    k'den güncel kareye kadar saklanan karelerde ilerletilir; sonra her yeni karede bir adım ilerler.
    Akış kamera hareketini de içerir; ayrıca telafi gerekmez. Top küçük olduğu için takip edilmez."""

    def __init__(self, scale: float = 0.5, buffer: int = 40, grid: int = 3, min_pts: int = 3,
                 max_err: float = 12.0) -> None:
        self.scale, self.grid, self.min_pts, self.max_err = scale, grid, min_pts, max_err
        self._buf: deque[tuple[int, np.ndarray]] = deque(maxlen=buffer)
        self._lk = dict(winSize=(15, 15), maxLevel=3,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
        self.reset()

    def reset(self) -> None:
        self._buf.clear()
        self._at: Optional[int] = None            # noktaların bulunduğu kare
        self._p0 = self._p = np.empty((0, 2), np.float32)
        self._owner = np.empty(0, int)            # nokta -> kimlik
        self._alive = np.empty(0, bool)

    def _gray(self, frame: np.ndarray) -> np.ndarray:
        small = cv2.resize(frame, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    def _step(self, g0: np.ndarray, g1: np.ndarray) -> None:
        idx = np.flatnonzero(self._alive)
        if len(idx) == 0:
            return
        p1, st, err = cv2.calcOpticalFlowPyrLK(g0, g1, self._p[idx].reshape(-1, 1, 2), None, **self._lk)
        ok = (st.ravel() == 1) & (err.ravel() < self.max_err)
        self._p[idx[ok]] = p1.reshape(-1, 2)[ok]
        self._alive[idx[~ok]] = False

    def push(self, idx: int, frame: np.ndarray) -> None:
        g = self._gray(frame)
        if self._at is not None and self._buf and self._buf[-1][0] == idx - 1 and self._at == idx - 1:
            self._step(self._buf[-1][1], g)
            self._at = idx
        elif self._at is not None and self._at != idx:
            self._at = None                       # kare atlandı ya da geri sarıldı: takip bırakılır
        self._buf.append((idx, g))

    def start(self, k: int, objs: Sequence[ObjectState]) -> None:
        """k karesine ait kutularla takibi başlat ve tampondaki karelerle güncel kareye getir."""
        frames = [(i, g) for i, g in self._buf if i >= k]
        if not frames or frames[0][0] != k:
            self._at = None
            return
        pts, own = [], []
        s, n = self.scale, self.grid
        for o in objs:
            if o.role == "ball":
                continue
            x1, y1, x2, y2 = o.bbox
            w, h = x2 - x1, y2 - y1
            for fx in np.linspace(0.3, 0.7, n):
                for fy in np.linspace(0.2, 0.7, n):    # gövde + üst bacak: ayaklardan kararlı
                    pts.append(((x1 + fx * w) * s, (y1 + fy * h) * s))
                    own.append(o.track_id)
        self._p0 = np.array(pts, np.float32).reshape(-1, 2)
        self._p = self._p0.copy()
        self._owner = np.array(own, int)
        self._alive = np.ones(len(own), bool)
        for (_, g0), (_, g1) in zip(frames[:-1], frames[1:]):
            self._step(g0, g1)
        self._at = frames[-1][0]

    def displacement(self, idx: int) -> dict[int, np.ndarray]:
        """kimlik -> k karesinden idx karesine kutu yer değiştirmesi (tam çözünürlük px)."""
        if self._at != idx or len(self._owner) == 0:
            return {}
        out = {}
        d = (self._p - self._p0) / self.scale
        for tid in np.unique(self._owner):
            m = (self._owner == tid) & self._alive
            if m.sum() >= self.min_pts:
                out[int(tid)] = np.median(d[m], axis=0)
        return out
