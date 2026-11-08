"""Forma rengiyle takım / rol ataması (M3b, D-022). Klasik görüntü işleme: renk uzayı + kümeleme.

1. Öznitelik: bbox'ın gövde bölgesi (yükseklikçe %15-%55, genişlikçe orta %60). Çim pikselleri
   (HSV yeşil) atılır; kalan piksellerin Lab ortalaması forma rengidir.
2. Kümeleme: yeterli örnek birikince K-means (k=2) ile iki takım rengi bulunur. Her takımın kendi
   yayılımına göre, iki merkeze de uzak kalan örnekler "diğer" (hakem / kaleci) sayılır.
   Kümeler periyodik olarak yeniden hesaplanır; etiket kayması olmasın diye yeni merkezler eskilerle eşlenir.
3. Karar takip kimliği bazında çoğunluk oyuyla verilir (tek karedeki renk hatası takımı değiştirmez).
   Genişlik/yükseklik > 0.6 olan kutulardan (muhtemelen iki kişi üst üste) örnek alınmaz.
4. "Diğer" rengindeki kişi (oylarının çoğunluğu, en az 3 oy): zamanının çoğunu bir ceza sahasında geçiriyorsa
   kaleci, aksi halde hakem. Kalecinin takımı: o kaleyi savunan takım. Savunan takım, kaleye en yakın saha
   oyuncusunun takımıdır (ofsayt kuralı); bu gözlem video boyunca birikir.
5. Takım A / B adı: daha koyu forma rengi (Lab L küçük) Takım A. Video başına sabit ve tekrarlanabilir.
Sınırlar: Hakem ve bir takım forması benzer renkteyse karışabilir; iki kaleci rengi aynıysa ayrım konuma dayanır.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from typing import Optional

import cv2
import numpy as np

from sgi.interfaces import TeamAssigner
from sgi.types import CLS_GOALKEEPER, CLS_PLAYER, CLS_REFEREE, TEAM_A, TEAM_B, TEAM_NONE, TEAM_UNKNOWN, Detection

_OTHER = 2


class ColorKMeansTeamAssigner(TeamAssigner):
    def __init__(self, min_samples: int = 150, refit_every: int = 100, outlier_factor: float = 2.5,
                 history: int = 20, pitch_length: float = 105.0, pitch_width: float = 68.0, **_: object) -> None:
        self.min_samples, self.refit_every, self.outlier_factor = min_samples, refit_every, outlier_factor
        self.L, self.W = pitch_length, pitch_width
        self._feats: deque = deque(maxlen=4000)                # (track_id, feature)
        self._votes: dict[int, deque] = defaultdict(lambda: deque(maxlen=history))
        self._pos: dict[int, deque] = defaultdict(lambda: deque(maxlen=history))
        self.centers: Optional[np.ndarray] = None                # 2x3 Lab; [0] = Takım A
        self.radius: Optional[np.ndarray] = None
        self._since_fit = 0
        self._side_votes = {"left": Counter(), "right": Counter()}
        self.min_other_ratio, self.min_other_votes = 0.5, 3
        self.max_aspect = 0.6          # genişlik/yükseklik bundan büyükse kutuda muhtemelen iki kişi var

    # --- öznitelik ---
    @staticmethod
    def jersey_feature(frame: np.ndarray, bbox) -> Optional[np.ndarray]:
        h_img, w_img = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        if bh < 12 or bw < 4:
            return None
        cx1, cx2 = int(max(0, x1 + 0.2 * bw)), int(min(w_img, x2 - 0.2 * bw))
        cy1, cy2 = int(max(0, y1 + 0.15 * bh)), int(min(h_img, y1 + 0.55 * bh))
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size < 12:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        grass = (hsv[..., 0] >= 30) & (hsv[..., 0] <= 90) & (hsv[..., 1] > 50)
        keep = ~grass
        if keep.sum() < 6:
            return None
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3)[keep.ravel()].astype(np.float32)
        return lab.mean(axis=0)

    # --- kümeleme ---
    def _fit(self) -> None:
        X = np.array([f for _, f in self._feats], dtype=np.float32)
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
        _, labels, centers = cv2.kmeans(X, 2, None, crit, 5, cv2.KMEANS_PP_CENTERS)
        labels = labels.ravel()
        # aykırı örnekleri (hakem/kaleci) çıkarıp merkezleri bir kez daha hesapla
        d = np.linalg.norm(X - centers[labels], axis=1)
        keep = np.zeros(len(X), bool)
        for k in range(2):
            m = labels == k
            if m.any():
                keep |= m & (d <= np.percentile(d[m], 85))
        centers = np.array([X[keep & (labels == k)].mean(axis=0) if (keep & (labels == k)).any() else centers[k]
                            for k in range(2)])
        if self.centers is None:
            order = np.argsort(centers[:, 0])                    # koyu forma = Takım A
        else:                                                    # önceki etiketlerle eşle (etiket kayması olmasın)
            same = (np.linalg.norm(centers[0] - self.centers[0]) + np.linalg.norm(centers[1] - self.centers[1]))
            swap = (np.linalg.norm(centers[0] - self.centers[1]) + np.linalg.norm(centers[1] - self.centers[0]))
            order = [0, 1] if same <= swap else [1, 0]
        self.centers = centers[order]
        dist = np.linalg.norm(X[:, None, :] - self.centers[None], axis=2)
        lab = dist.argmin(axis=1)
        self.radius = np.array([np.median(dist[lab == k, k]) if (lab == k).any() else 10.0 for k in range(2)])
        self.radius = np.maximum(self.radius, 4.0)
        self._since_fit = 0

    def _classify(self, f: np.ndarray) -> int:
        d = np.linalg.norm(self.centers - f, axis=1)
        k = int(d.argmin())
        return k if d[k] <= self.outlier_factor * self.radius[k] else _OTHER

    # --- arayüz ---
    def observe(self, frame_bgr, persons: list[Detection], pitch_xy, roles=None) -> None:
        roles = roles or [None] * len(persons)
        for det, xy, role in zip(persons, pitch_xy, roles):
            if det.track_id < 0:
                continue
            if xy is not None:
                self._pos[det.track_id].append(xy)
            if role in (CLS_REFEREE, CLS_GOALKEEPER):
                continue                      # dedektör rolü verdiyse: hakem/kaleci rengi takım kümelemesini bozmasın
            x1, y1, x2, y2 = det.bbox
            if (x2 - x1) > self.max_aspect * max(y2 - y1, 1e-6):
                continue                      # birleşmiş kutu: iki forma rengi karışır, örnek alma
            f = self.jersey_feature(frame_bgr, det.bbox)
            if f is None:
                continue
            self._feats.append((det.track_id, f))
            if self.centers is not None:
                self._votes[det.track_id].append(self._classify(f))
        self._since_fit += 1
        if (self.centers is None and len(self._feats) >= self.min_samples) or \
                (self.centers is not None and self._since_fit >= self.refit_every):
            self._fit()
            # ilk kümelemede birikmiş örnekleri oylara dök
            if all(len(v) == 0 for v in self._votes.values()):
                for tid, f in self._feats:
                    self._votes[tid].append(self._classify(f))
        self._update_team_sides(persons, pitch_xy)

    def _update_team_sides(self, persons: list[Detection], pitch_xy) -> None:
        """Hangi takım hangi kaleyi savunuyor: bir kaleye en yakın saha oyuncusu (kaleci hariç) neredeyse
        her zaman o kaleyi savunan takımdandır (ofsayt kuralı gereği hücumcu son savunmacıyı geçemez).
        Gözlemler video boyunca birikir; karar zamanla oynamaz."""
        if self.centers is None:
            return
        cands = []
        for det, xy in zip(persons, pitch_xy):
            if xy is None or det.track_id < 0:
                continue
            votes = self._votes.get(det.track_id)
            if not votes:
                continue
            k = Counter(votes).most_common(1)[0][0]
            if k != _OTHER:
                cands.append((xy[0], (TEAM_A, TEAM_B)[k]))
        if len(cands) < 6:
            return
        xmin, tmin = min(cands)
        xmax, tmax = max(cands)
        if xmin < self.L / 3:
            self._side_votes["left"][tmin] += 1
        if xmax > 2 * self.L / 3:
            self._side_votes["right"][tmax] += 1

    def defending(self, side: str) -> Optional[str]:
        own = self._side_votes[side]
        other = self._side_votes["right" if side == "left" else "left"]
        if not own and not other:
            return None
        # karşı kalenin gözlemleri de kanıttır: sağı A savunuyorsa solu B savunur
        score_a = own[TEAM_A] + other[TEAM_B]
        score_b = own[TEAM_B] + other[TEAM_A]
        return TEAM_A if score_a >= score_b else TEAM_B

    def _in_penalty_area(self, xy) -> Optional[str]:
        x, y = xy
        if not (self.W / 2 - 20.16 - 2 <= y <= self.W / 2 + 20.16 + 2):
            return None
        if x <= 16.5 + 2:
            return "left"
        if x >= self.L - 16.5 - 2:
            return "right"
        return None

    def assign(self, track_id: int) -> tuple[str, str]:
        votes = self._votes.get(track_id)
        if self.centers is None or not votes:
            return CLS_PLAYER, TEAM_UNKNOWN
        cnt = Counter(votes)
        n_other = cnt.get(_OTHER, 0)
        # "diğer" (hakem/kaleci) için güçlü kanıt iste: iki oyuncunun üst üste gelmesi gibi kısa karışmalar
        # renk karışık olduğu için "diğer"e düşer; bunlar takım oylarının çoğunluğuyla oyuncu kalır
        if not self._other_dominant(n_other, len(votes)):
            team_votes = {k: v for k, v in cnt.items() if k != _OTHER}
            if team_votes:
                return CLS_PLAYER, (TEAM_A, TEAM_B)[max(team_votes, key=team_votes.get)]
            return CLS_PLAYER, TEAM_UNKNOWN
        sides = [self._in_penalty_area(p) for p in self._pos.get(track_id, [])]
        if sides:
            side, n = Counter(sides).most_common(1)[0]
            if side is not None and n >= 0.6 * len(sides):
                return CLS_GOALKEEPER, self.defending(side) or TEAM_UNKNOWN
        return CLS_REFEREE, TEAM_NONE

    def team_of(self, track_id: int) -> str:
        """Yalnızca takım (A/B): dedektör kişinin oyuncu olduğunu söylediğinde "diğer" oyları yok sayılır."""
        votes = self._votes.get(track_id)
        if self.centers is None or not votes:
            return TEAM_UNKNOWN
        team_votes = Counter(v for v in votes if v != _OTHER)
        if not team_votes:
            return TEAM_UNKNOWN
        return (TEAM_A, TEAM_B)[team_votes.most_common(1)[0][0]]

    def is_other(self, track_id: int) -> bool:
        """Forma rengi iki takıma da uymuyor mu (oyların çoğunluğu "diğer", en az `min_other_votes`)?"""
        votes = self._votes.get(track_id)
        if self.centers is None or not votes:
            return False
        return self._other_dominant(sum(1 for v in votes if v == _OTHER), len(votes))

    def _other_dominant(self, n_other: int, n: int) -> bool:
        """"Diğer" oyları çoğunlukta mı? En az min_other_votes oy gerekir; ancak takip yeni başladıysa ve oyların
        hepsi "diğer" ise hemen karar verilir (yoksa hakem ilk karelerde "?" görünüyordu)."""
        return n_other >= self.min_other_ratio * n and (n_other >= self.min_other_votes or n_other == n > 0)

    def goalkeeper_team(self, track_id: int) -> str:
        """Kalecinin takımı: bulunduğu kaleyi savunan takım."""
        pos = list(self._pos.get(track_id, []))
        if not pos:
            return TEAM_UNKNOWN
        side = "left" if np.mean([p[0] for p in pos]) < self.L / 2 else "right"
        return self.defending(side) or TEAM_UNKNOWN

    def team_colors_bgr(self) -> dict[str, tuple[int, int, int]]:
        if self.centers is None:
            return {}
        lab = np.clip(self.centers, 0, 255).astype(np.uint8).reshape(1, 2, 3)
        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR).reshape(2, 3)
        return {TEAM_A: tuple(int(v) for v in bgr[0]), TEAM_B: tuple(int(v) for v in bgr[1])}
