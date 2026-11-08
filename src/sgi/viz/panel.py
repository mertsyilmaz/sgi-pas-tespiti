"""Bilgi paneli yerleşimi: video solda (1,125 kat büyütülmüş), sağda düz renkli panel.

1280x720 video -> 1440x810 olarak gösterilir, sağında 480 px panel: 1920x810 tuval.
Sağ sütun: durum, mini harita, pas sayacı. Videonun üzerinde yalnızca sahaya ait çizimler kalır
(ızgara, oyuncu etiketleri, kale, top yolu); bunlar videonun kendi çözünürlüğünde çizilip birlikte büyütülür.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import cv2
import numpy as np

BG = (32, 30, 28)                  # BGR: panel arka planı
SEP = (75, 72, 70)                 # bölüm ayırıcı çizgiler
TITLE = (165, 160, 155)            # bölüm başlıkları
TEXT = (235, 235, 235)
FONT = cv2.FONT_HERSHEY_SIMPLEX
ZOOM = 1.125                       # video büyütme oranı (1280 -> 1440)
SIDE = 0.375                       # panel genişliği / video genişliği (1280 -> 480)


@dataclass(frozen=True)
class PanelLayout:
    video_w: int
    video_h: int

    @property
    def scale(self) -> float:
        return self.video_w / 1280.0

    @property
    def zoom(self) -> float:
        return ZOOM

    @property
    def shown_w(self) -> int:
        return int(round(self.video_w * ZOOM))

    @property
    def shown_h(self) -> int:
        return int(round(self.video_h * ZOOM))

    @property
    def side_w(self) -> int:
        return int(round(self.video_w * SIDE))

    @property
    def size(self) -> tuple[int, int]:
        """(genişlik, yükseklik): 1280x720 video için 1920x810."""
        return self.shown_w + self.side_w, self.shown_h

    @property
    def margin(self) -> int:
        return int(round(20 * self.scale))

    def px(self, v: float) -> int:
        return int(round(v * self.scale))

    # --- sağ sütun bölümleri ---
    @property
    def side_x(self) -> int:
        return self.shown_w + self.margin

    @property
    def side_inner_w(self) -> int:
        return self.side_w - 2 * self.margin

    @property
    def status_y(self) -> int:
        return self.margin

    @cached_property
    def _background(self) -> np.ndarray:
        w, h = self.size
        c = np.empty((h, w, 3), np.uint8)
        c[:] = BG
        cv2.line(c, (self.shown_w, 0), (self.shown_w, h), SEP, max(1, self.px(2)))
        return c

    def canvas(self, frame: np.ndarray) -> np.ndarray:
        """Hazır arka planın kopyası + büyütülmüş video solda."""
        c = self._background.copy()
        c[:self.shown_h, :self.shown_w] = cv2.resize(frame, (self.shown_w, self.shown_h),
                                                     interpolation=cv2.INTER_LINEAR)
        return c

    def video_view(self, canvas: np.ndarray) -> np.ndarray:
        """Tuvalde videonun alanı (koordinatlar videonun ZOOM katı)."""
        return canvas[:self.shown_h, :self.shown_w]

    def title(self, img: np.ndarray, txt: str, x: int, y: int, width: int) -> int:
        """Bölüm başlığı + altında ince çizgi. Başlığın altındaki ilk satırın y'sini döndürür."""
        fs = 0.5 * self.scale
        put(img, txt, (x, y + self.px(14)), fs, TITLE, 1)
        cv2.line(img, (x, y + self.px(22)), (x + width, y + self.px(22)), SEP, 1)
        return y + self.px(50)


def put(img: np.ndarray, txt: str, org: tuple[int, int], fs: float, col, th: int = 1) -> None:
    cv2.putText(img, txt, (int(org[0]), int(org[1])), FONT, fs, col, th, cv2.LINE_AA)


def wrap(txt: str, fs: float, max_w: int, th: int = 1) -> list[str]:
    """Metni kelime sınırlarından bölerek her satırı max_w pikselden kısa tutar."""
    lines, cur = [], ""
    for word in txt.split(" "):
        cand = f"{cur} {word}" if cur else word
        if cur and cv2.getTextSize(cand, FONT, fs, th)[0][0] > max_w:
            lines.append(cur)
            cur = word
        else:
            cur = cand
    return lines + [cur] if cur else lines
