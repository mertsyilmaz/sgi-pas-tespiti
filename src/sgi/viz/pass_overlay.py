"""Pas olaylarının çizimi (M8): top yolu videoda, takım bazında sayaç sağ paneldeki PAS SAYACI bölümünde.

İki mod (D-040): iki takım (varsayılan) ya da hedef takım (`target_team`): yalnızca o takımın pasları çizilir ve sayılır."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from sgi.mapping.homography import pitch_to_image
from sgi.viz.panel import TEXT, TITLE, PanelLayout, put

_OK, _NO = (80, 220, 80), (60, 60, 240)               # BGR: başarılı yeşil, başarısız kırmızı
_FLY = (0, 220, 255)                                  # pas sürüyor


class PassOverlay:
    def __init__(self, passes: list[dict], image_size: tuple[int, int], hold_s: float = 2.0, fps: float = 25.0,
                 target_team: Optional[str] = None) -> None:
        """image_size: videonun kendi boyutu (panelsiz). target_team: None = iki takım, "A"/"B" = yalnızca o takım."""
        self.target_team = target_team
        self.passes: list[dict] = []
        for d in passes:
            self.add(d)
        self.w, self.h = image_size
        self.layout = PanelLayout(self.w, self.h)
        self.scale = self.layout.scale
        self.hold = int(hold_s * fps)

    def add(self, d: dict) -> None:
        """Pas kaydı ekler (canlı modda paslar tespit edildikçe gelir). Hedef takım dışındaki paslar alınmaz."""
        if self.target_team is None or d["team"] == self.target_team:
            self.passes.append(d)
            self.passes.sort(key=lambda x: x["frame_end"])

    def reset(self) -> None:
        self.passes = []

    def draw(self, canvas: np.ndarray, idx: int, H_img2pitch: Optional[np.ndarray], top: int) -> np.ndarray:
        """canvas: GridOverlay çıktısı (panelli tuval). idx: kesit içi kare (passes.json kare numaralarıyla aynı).
        top: sayacın sağ sütundaki başlangıcı (`GridOverlay.panel_bottom`)."""
        L = self.layout
        video = L.video_view(canvas)
        Hi = pitch_to_image(H_img2pitch, (self.w, self.h)) if H_img2pitch is not None else None
        th = max(2, int(round(3 * self.scale * L.zoom)))
        for d in self.passes:
            if Hi is None or not (d["frame_start"] <= idx <= d["frame_end"] + self.hold):
                continue
            done = idx >= d["frame_end"]
            col = (_OK if d["label"] == "SUCCESS" else _NO) if done else _FLY
            pts = np.array([(x, y) for f, x, y in d["ball_path"] if f <= idx], float)
            if len(pts) < 2:
                continue
            q = np.c_[pts, np.ones(len(pts))] @ Hi.T
            ok = q[:, 2] > 1e-9
            q = q[ok, :2] / q[ok, 2:3] * L.zoom                 # video koordinatı -> büyütülmüş görüntü
            if len(q) >= 2:
                cv2.polylines(video, [np.round(q).astype(np.int32)], False, (0, 0, 0), th + 2, cv2.LINE_AA)
                cv2.polylines(video, [np.round(q).astype(np.int32)], False, col, th, cv2.LINE_AA)
                if done:
                    cv2.circle(video, tuple(np.round(q[-1]).astype(int)), 3 * th, col, th, cv2.LINE_AA)
        self._draw_score(canvas, idx, top)
        return canvas

    def _draw_score(self, img: np.ndarray, idx: int, top: int) -> None:
        L = self.layout
        teams = (self.target_team,) if self.target_team else ("A", "B")
        cnt = {t: [0, 0] for t in teams}
        for d in self.passes:
            if d["frame_end"] <= idx:
                cnt[d["team"]][0 if d["label"] == "SUCCESS" else 1] += 1
        title = f"PAS SAYACI  -  HEDEF TAKIM: {self.target_team}" if self.target_team else "PAS SAYACI  -  IKI TAKIM"
        y = L.title(img, title, L.side_x, top, L.side_inner_w)
        fs = 0.65 * self.scale
        x1, x2 = L.side_x + L.px(150), L.side_x + L.px(300)
        put(img, "basarili", (x1, y), fs, TITLE)
        put(img, "basarisiz", (x2, y), fs, TITLE)
        for k, t in enumerate(teams):
            yy = y + (k + 1) * L.px(40)
            put(img, f"Takim {t}", (L.side_x, yy), fs, TEXT, 2)
            put(img, f"{cnt[t][0]:3d}", (x1 + L.px(15), yy), fs, _OK, 2)
            put(img, f"{cnt[t][1]:3d}", (x2 + L.px(20), yy), fs, _NO, 2)
