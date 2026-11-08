"""50 hücreli ızgaranın video karesine ve mini haritaya çizimi (M8 görselleştirme)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from sgi.grid.grid import N_CELLS, PitchGrid
from sgi.mapping.homography import pitch_to_image
from sgi.types import HomographyResult
from sgi.viz.panel import TEXT as _PANEL_TEXT
from sgi.viz.panel import PanelLayout, put, wrap

_LINE = (255, 255, 255)
_FILL_A = (60, 180, 255)     # BGR, dama tahtası renkleri
_FILL_B = (255, 140, 60)
_TEXT = (255, 255, 255)
_SAMPLES = 24                 # bir kenar üzerinde örnek nokta (perspektifte düzgün eğri için)


@dataclass
class OverlayOptions:
    grid: bool = True             # hücre sınırlarının beyaz çizgileri ('g' tuşu)
    fill: bool = True
    numbers: bool = True
    minimap: bool = True
    keypoints: bool = False
    hud: bool = True
    detected_lines: bool = True   # mini haritada videodan tespit edilen çizgiler (~10 ms/kare)
    objects: bool = True          # oyuncu/hakem/top işaretleri (videoda ve mini haritada)
    goals: bool = True            # kale çerçevesi tespiti ve çizimi (D-026)
    fill_alpha: float = 0.10       # dama dolgusu opaklığı (0,22 oyuncu renklerini soldurdu, D-039)
    line_alpha: float = 0.45       # hücre sınırı çizgilerinin opaklığı (D-039)


def _project(H_inv: np.ndarray, pts: np.ndarray, limit: float) -> tuple[np.ndarray, np.ndarray]:
    """Saha -> görüntü. Kamera arkasına düşen (w<=0) ve aşırı uzak noktalar geçersiz sayılır."""
    p = np.c_[pts, np.ones(len(pts))] @ H_inv.T
    w = p[:, 2]
    ok = w > 1e-9
    xy = np.full((len(pts), 2), np.nan)
    xy[ok] = p[ok, :2] / w[ok, None]
    ok &= np.all(np.abs(xy) < limit, axis=1)
    return xy, ok


def _segments(xy: np.ndarray, ok: np.ndarray) -> list[np.ndarray]:
    segs, cur = [], []
    for p, v in zip(xy, ok):
        if v:
            cur.append(p)
        elif cur:
            segs.append(np.array(cur))
            cur = []
    if cur:
        segs.append(np.array(cur))
    return [np.round(s).astype(np.int32) for s in segs if len(s) >= 2]


class GridOverlay:
    def __init__(self, grid: PitchGrid, image_size: tuple[int, int]) -> None:
        self.grid = grid
        self.w, self.h = image_size
        self.limit = 4.0 * max(self.w, self.h)
        self.scale = self.w / 1280.0
        self.layout = PanelLayout(self.w, self.h)       # bilgi yazıları ve mini harita videonun yanındaki panelde
        self.out_size = self.layout.size
        self._mini_k = 1.0                               # mini harita büyüklüğü / eski video içi boyut (0,30 w)
        self._mini_static = None                         # mini haritanın sabit katmanları (çizgiler, numaralar)
        self._status_bottom = 0                          # durum bölümünün o ana kadarki en alt noktası (zıplamasın)
        self.panel_bottom = 0                            # sağ sütunda dolu kısmın altı: pas sayacı buradan başlar
        self._cell_polys = {c: self._cell_polygon(c) for c in range(1, N_CELLS + 1)}
        # hücre durumu için her hücrede 4x4 iç örnek nokta
        t = np.linspace(0.05, 0.95, 4)
        probes = []
        for c in range(1, N_CELLS + 1):
            x0, y0, x1, y1 = grid.cell_bounds(c)
            probes += [(x0 + (x1 - x0) * a, y0 + (y1 - y0) * b) for a in t for b in t]
        self._cell_probe = np.array(probes)
        from sgi.mapping.quality import pitch_line_samples
        self._pitch_marks = pitch_line_samples(grid.length, grid.width, step=0.25)
        self._masker = None
        self._objects, self._team_colors = None, {}
        from sgi.mapping.goals import GoalDetector
        self._goal_det = GoalDetector(image_size, grid.length, grid.width)
        self._goals = []
        self.min_line_len_px = 25        # mini harita için çizgi parçası en kısa uzunluğu (640 px genişlikte)

    def _cell_polygon(self, cell: int) -> np.ndarray:
        x0, y0, x1, y1 = self.grid.cell_bounds(cell)
        t = np.linspace(0, 1, _SAMPLES)
        return np.vstack([np.c_[x0 + (x1 - x0) * t, np.full_like(t, y0)],
                          np.c_[np.full_like(t, x1), y0 + (y1 - y0) * t],
                          np.c_[x1 - (x1 - x0) * t, np.full_like(t, y1)],
                          np.c_[np.full_like(t, x0), y1 - (y1 - y0) * t]])

    def draw(self, frame: np.ndarray, res: Optional[HomographyResult], opt: OverlayOptions,
             frame_idx: int = 0, fps: float = 25.0, extra: Optional[list[str]] = None,
             objects: Optional[list] = None, team_colors: Optional[dict] = None) -> np.ndarray:
        """Çıktı: `out_size` boyutunda tuval. Saha çizimleri videonun kendi çözünürlüğünde yapılır,
        video büyütülüp sola konur; sağdaki panele durum, mini harita (ve pas sayacı) yazılır."""
        self._objects = objects if opt.objects else None
        self._team_colors = team_colors or {}
        out = frame.copy()
        H = res.H if (res is not None and res.valid) else None
        if H is not None:
            H_inv = pitch_to_image(H, (self.w, self.h))
            if opt.fill:
                self._draw_fill(out, H_inv, opt.fill_alpha)
            if opt.grid:
                self._draw_lines(out, H_inv, opt.line_alpha)
            if opt.numbers:
                self._draw_numbers(out, H_inv)
        self._goals = self._goal_det.detect(frame, H) if (opt.goals and H is not None) else []
        for gl in self._goals:
            if gl.found:
                self._draw_goal_frame(out, gl)
        if self._objects:
            self._draw_objects_on_frame(out)
        if opt.keypoints and res is not None:
            for k in res.keypoints:
                cv2.circle(out, (int(k.image_xy[0]), int(k.image_xy[1])), max(3, int(4 * self.scale)), (0, 0, 255), -1)
        canvas = self.layout.canvas(out)
        # sağ sütun yukarıdan aşağı: durum, mini harita (+ açıklama); pas sayacı `panel_bottom`dan sonra
        y = self.layout.status_y
        if opt.hud:
            self._status_bottom = max(self._status_bottom, self._draw_hud(canvas, res, frame_idx, fps, extra or []))
            y = self._status_bottom + self.layout.px(10)
        if opt.minimap:
            y = self._draw_minimap(canvas, H, frame, opt, y)
        self.panel_bottom = y
        return canvas

    # --- oyuncu / hakem / top işaretleri (D-022) ---
    _REF_BLUE = (255, 80, 0)          # BGR: hakem mavi
    _BALL = (0, 0, 0)                 # top siyah
    _UNKNOWN = (150, 150, 150)

    def _object_color(self, o) -> tuple[int, int, int]:
        if o.role == "ball":
            return self._BALL
        if o.role == "referee":
            return self._REF_BLUE
        return self._team_colors.get(o.team, self._UNKNOWN)

    def _draw_objects_on_frame(self, img: np.ndarray) -> None:
        th = max(1, int(round(2 * self.scale)))
        for o in self._objects:
            x1, y1, x2, y2 = (int(v) for v in o.bbox)
            col = self._object_color(o)
            if o.role == "ball":
                c = ((x1 + x2) // 2, (y1 + y2) // 2)
                r = max(6, (x2 - x1))
                if getattr(o, "state", "detected") != "detected":
                    # görünmüyor, takipten kestirildi (D-028): oyuncuda (held) ya da yörüngede (predicted)
                    cv2.circle(img, c, r + 2, (0, 0, 0), th + 2, cv2.LINE_AA)
                    cv2.circle(img, c, r + 2, (0, 200, 255), th, cv2.LINE_AA)
                    tag = "top?" if o.state == "predicted" else "top (oyuncuda)"
                    fs = 0.4 * max(self.scale, 0.8)
                    cv2.putText(img, tag, (c[0] + r + 4, c[1]), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(img, tag, (c[0] + r + 4, c[1]), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 200, 255), 1,
                                cv2.LINE_AA)
                    continue
                cv2.circle(img, c, r, (255, 255, 255), th + 2, cv2.LINE_AA)
                cv2.circle(img, c, r, (0, 0, 0), th, cv2.LINE_AA)
                continue
            # yalnızca etiket (ayak altı halkası kaldırıldı): takım harfi, hakem H, kaleci KA/KB;
            # saha konumu biliniyorsa yanında metre cinsinden (x, y) koordinatı
            tag = {"referee": "H", "goalkeeper": f"K{o.team}"}.get(o.role, o.team)
            if o.pitch_xy is not None and o.role != "referee":
                tag = f"{tag}: {o.pitch_xy[0]:.0f},{o.pitch_xy[1]:.0f}"
            # okunurluk: takım renginde dolu kutu, üstünde renge göre siyah/beyaz yazı
            fs = 0.45 * max(self.scale, 0.8)
            (tw, tht), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
            pad = max(2, int(round(2 * self.scale)))
            bx1, by2 = (x1 + x2) // 2 - tw // 2 - pad, y1 - 3
            by1 = by2 - tht - 2 * pad
            cv2.rectangle(img, (bx1, by1), (bx1 + tw + 2 * pad, by2), col, -1)
            cv2.rectangle(img, (bx1, by1), (bx1 + tw + 2 * pad, by2), (0, 0, 0), 1)
            luma = 0.114 * col[0] + 0.587 * col[1] + 0.299 * col[2]
            cv2.putText(img, tag, (bx1 + pad, by2 - pad), cv2.FONT_HERSHEY_SIMPLEX, fs,
                        (0, 0, 0) if luma > 140 else (255, 255, 255), 1, cv2.LINE_AA)

    _GOAL = (255, 0, 255)             # BGR: kale çerçevesi (magenta)

    def _draw_goal_frame(self, img: np.ndarray, gl) -> None:
        th = max(2, int(round(3 * self.scale)))
        (b0, b1), (t0, t1) = gl.bases, gl.tops
        for a, b in ((b0, t0), (b1, t1), (t0, t1)):
            cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (0, 0, 0), th + 2, cv2.LINE_AA)
            cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), self._GOAL, th, cv2.LINE_AA)
        fs = 0.5 * max(self.scale, 0.8)
        org = (int(min(t0[0], t1[0])), int(min(t0[1], t1[1])) - 6)
        cv2.putText(img, "KALE", org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, "KALE", org, cv2.FONT_HERSHEY_SIMPLEX, fs, self._GOAL, 1, cv2.LINE_AA)

    def _draw_goals_on_minimap(self, mini: np.ndarray, to_px) -> None:
        g = self.grid
        cy = g.width / 2
        for gl in self._goals:
            x = 0.0 if gl.side == "left" else g.length
            d = -1.5 if gl.side == "left" else 1.5          # kale çizginin dışına doğru küçük kutu
            a, b = to_px((x, cy - 3.66)), to_px((x + d, cy + 3.66))
            p0, p1 = (int(min(a[0], b[0])), int(a[1])), (int(max(a[0], b[0])), int(b[1]))
            col = self._GOAL if gl.found else (150, 150, 150)
            cv2.rectangle(mini, p0, p1, col, -1 if gl.found else 1)

    def _draw_objects_on_minimap(self, mini: np.ndarray, to_px) -> None:
        r = max(3, int(round(4 * self.scale * self._mini_k)))
        for o in sorted(self._objects, key=lambda o: o.role == "ball"):      # top en üstte
            if o.pitch_xy is None:
                continue
            x, y = to_px(o.pitch_xy)
            if not (0 <= x < mini.shape[1] and 0 <= y < mini.shape[0]):
                continue
            col = self._object_color(o)
            if o.role == "ball":
                cv2.circle(mini, (int(x), int(y)), r - 1, (255, 255, 255), -1, cv2.LINE_AA)
                filled = -1 if getattr(o, "state", "detected") == "detected" else 1     # kestirim: içi boş
                cv2.circle(mini, (int(x), int(y)), max(2, r - 2), self._BALL, filled, cv2.LINE_AA)
                continue
            cv2.circle(mini, (int(x), int(y)), r, col, -1, cv2.LINE_AA)
            border = (255, 255, 255) if o.role == "goalkeeper" else (0, 0, 0)
            cv2.circle(mini, (int(x), int(y)), r, border, 2 if o.role == "goalkeeper" else 1, cv2.LINE_AA)

    def _draw_fill(self, img: np.ndarray, H_inv: np.ndarray, alpha: float) -> None:
        layer = img.copy()
        for cell, poly in self._cell_polys.items():
            xy, ok = _project(H_inv, poly, self.limit)
            if ok.sum() < 3:
                continue
            row, col = self.grid.row_col(cell)
            color = _FILL_A if (row + col) % 2 == 0 else _FILL_B
            cv2.fillPoly(layer, [np.round(xy[ok]).astype(np.int32)], color, lineType=cv2.LINE_AA)
        cv2.addWeighted(layer, alpha, img, 1 - alpha, 0, dst=img)

    def _draw_lines(self, img: np.ndarray, H_inv: np.ndarray, alpha: float = 1.0) -> None:
        g, n = self.grid, _SAMPLES * 4
        layer = img.copy() if alpha < 1.0 else img
        lines = [np.c_[np.full(n, x), np.linspace(g.y_edges[0], g.y_edges[-1], n)] for x in g.x_edges]
        lines += [np.c_[np.linspace(g.x_edges[0], g.x_edges[-1], n), np.full(n, y)] for y in g.y_edges]
        thick = max(1, int(round(2 * self.scale)))
        for ln in lines:
            for seg in _segments(*_project(H_inv, ln, self.limit)):
                cv2.polylines(layer, [seg], False, _LINE, thick, cv2.LINE_AA)
        if alpha < 1.0:
            cv2.addWeighted(layer, alpha, img, 1 - alpha, 0, dst=img)

    def _draw_numbers(self, img: np.ndarray, H_inv: np.ndarray) -> None:
        for cell in range(1, N_CELLS + 1):
            c, ok = _project(H_inv, np.array([self.grid.cell_center(cell)]), self.limit)
            if not ok[0]:
                continue
            x, y = c[0]
            if not (0 <= x < self.w and 0 <= y < self.h):
                continue
            # yazı boyutu: hücrenin görüntüdeki yaklaşık genişliğine göre
            x0, y0, x1, y1 = self.grid.cell_bounds(cell)
            e, ok2 = _project(H_inv, np.array([[x0, (y0 + y1) / 2], [x1, (y0 + y1) / 2]]), self.limit)
            px = np.linalg.norm(e[1] - e[0]) if ok2.all() else 60
            fs = float(np.clip(px / 110.0, 0.35, 1.4))
            txt = str(cell)
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
            org = (int(x - tw / 2), int(y + th / 2))
            cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, fs, _TEXT, 2, cv2.LINE_AA)

    def cell_states(self, H: Optional[np.ndarray]) -> np.ndarray:
        """Hücre başına durum: 0 = görünür, 1 = görüş dışında (hesaplanabilir ama kadrajda değil),
        2 = hesaplanamıyor (kalibrasyon yok ya da hücre kameranın görüş geometrisinin dışında/ufkun ötesinde)."""
        states = np.full(N_CELLS, 2, np.int8)
        if H is None:
            return states
        H_inv = pitch_to_image(H, (self.w, self.h))
        p = np.c_[self._cell_probe, np.ones(len(self._cell_probe))] @ H_inv.T
        front = p[:, 2] > 1e-9
        xy = np.full((len(p), 2), np.nan)
        xy[front] = p[front, :2] / p[front, 2:3]
        inside = front & (xy[:, 0] >= 0) & (xy[:, 0] < self.w) & (xy[:, 1] >= 0) & (xy[:, 1] < self.h)
        front, inside = front.reshape(N_CELLS, -1), inside.reshape(N_CELLS, -1)
        states[front.any(axis=1)] = 1
        states[inside.any(axis=1)] = 0
        return states

    def _detected_lines_on_pitch(self, frame: np.ndarray, H: np.ndarray) -> np.ndarray:
        """Videoda tespit edilen beyaz çizgi pikselleri (çim üzerinde), saha koordinatına çevrilmiş."""
        if self._masker is None:
            from sgi.mapping.quality import QualityMonitor
            self._masker = QualityMonitor((self.w, self.h))
        _, _, lines = self._masker.masks(frame)
        # yalnızca uzun parçaları tut: forma/çorap gibi kısa beyaz lekeler gürültü yapar
        n, labels, stats, _ = cv2.connectedComponentsWithStats(lines, connectivity=8)
        long_ = np.zeros(n, bool)
        long_[1:] = np.maximum(stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT]) >= self.min_line_len_px
        ys, xs = np.nonzero(long_[labels])
        if len(xs) == 0:
            return np.zeros((0, 2))
        s = self._masker.s
        img_pts = np.c_[(xs + 0.5) / s, (ys + 0.5) / s]
        Ho = H if (H @ np.array([self.w / 2, self.h / 2, 1.0]))[2] > 0 else -H
        p = np.c_[img_pts, np.ones(len(img_pts))] @ Ho.T
        ok = p[:, 2] > 1e-9
        pts = p[ok, :2] / p[ok, 2:3]
        g = self.grid
        keep = (pts[:, 0] > -2) & (pts[:, 0] < g.length + 2) & (pts[:, 1] > -2) & (pts[:, 1] < g.width + 2)
        return pts[keep]

    def _draw_minimap(self, img: np.ndarray, H: Optional[np.ndarray], frame: np.ndarray, opt: "OverlayOptions",
                      top: int) -> int:
        """Sağ sütunda `top`tan başlar; açıklamanın altındaki y'yi döndürür."""
        g, L = self.grid, self.layout
        y_off = L.title(img, "MINI HARITA", L.side_x, top, L.side_inner_w)
        mw = L.side_inner_w
        ppm = mw / g.length
        mh = int(g.width * ppm)
        x_off = L.side_x
        self._mini_k = mw / (0.30 * self.w)
        mini = np.full((mh, mw, 3), (40, 110, 40), np.uint8)

        def to_px(p):
            return np.round(np.asarray(p) * ppm).astype(np.int32)

        if self._mini_static is None or self._mini_static[0].shape[:2] != (mh, mw):
            self._mini_static = self._minimap_static_layers(mw, mh, to_px)
        under, numbers = self._mini_static

        # hücre durumları: görüş dışı koyu gri, hesaplanamayan kırmızı
        for cell, st in zip(range(1, N_CELLS + 1), self.cell_states(H)):
            if st == 0:
                continue
            x0, y0, x1, y1 = g.cell_bounds(cell)
            color = (60, 70, 60) if st == 1 else (40, 40, 200)
            cv2.rectangle(mini, tuple(to_px((x0, y0))), tuple(to_px((x1, y1))), color, -1)

        cv2.max(mini, under, dst=mini)                  # saha çizgileri + hücre sınırları (sabit katman)

        if H is not None:
            # videoda tespit edilen çizgiler, kuş bakışına çevrilmiş (sarı). Referansla örtüşmesi = doğru kalibrasyon
            if opt.detected_lines:
                det = to_px(self._detected_lines_on_pitch(frame, H))
                ok = (det[:, 0] >= 0) & (det[:, 0] < mw) & (det[:, 1] >= 0) & (det[:, 1] < mh)
                mini[det[ok, 1], det[ok, 0]] = (0, 230, 255)
            # kameranın gördüğü alanın sınırı
            border = np.vstack([np.c_[np.linspace(0, self.w, 20), np.zeros(20)],
                                np.c_[np.full(20, self.w), np.linspace(0, self.h, 20)],
                                np.c_[np.linspace(self.w, 0, 20), np.full(20, self.h)],
                                np.c_[np.zeros(20), np.linspace(self.h, 0, 20)]])
            Ho = H if (H @ np.array([self.w / 2, self.h / 2, 1.0]))[2] > 0 else -H
            p = np.c_[border, np.ones(len(border))] @ Ho.T
            ok = p[:, 2] > 1e-9
            if ok.sum() >= 3:
                fov = np.clip(p[ok, :2] / p[ok, 2:3], [-50, -50], [g.length + 50, g.width + 50])
                cv2.polylines(mini, [to_px(fov)], True, (90, 200, 255), 1, cv2.LINE_AA)

        cv2.max(mini, numbers, dst=mini)                # hücre numaraları (sabit katman, beyaz)
        if self._goals:
            self._draw_goals_on_minimap(mini, to_px)
        if self._objects:
            self._draw_objects_on_minimap(mini, to_px)
        img[y_off:y_off + mh, x_off:x_off + mw] = mini
        cv2.rectangle(img, (x_off, y_off), (x_off + mw - 1, y_off + mh - 1), (255, 255, 255), 1)
        return self._draw_minimap_legend(img, x_off, y_off + mh)

    def _minimap_static_layers(self, mw: int, mh: int, to_px) -> tuple[np.ndarray, np.ndarray]:
        """Her karede aynı kalan mini harita çizimleri bir kez hazırlanır; kareye cv2.max ile eklenir
        (açık renkli çizimler yeşil/gri zeminde aynen görünür)."""
        g = self.grid
        under = np.zeros((mh, mw, 3), np.uint8)
        # hücre sınırları: ince ve silik (saha çizgileriyle karışmasın)
        for x in g.x_edges:
            cv2.line(under, tuple(to_px((x, 0))), tuple(to_px((x, g.width))), (95, 150, 95), 1)
        for y in g.y_edges:
            cv2.line(under, tuple(to_px((0, y))), tuple(to_px((g.length, y))), (95, 150, 95), 1)
        # sahanın gerçek çizgileri: beyaz, 2 px (belirgin)
        ref = to_px(self._pitch_marks)
        ok = (ref[:, 0] >= 0) & (ref[:, 0] < mw) & (ref[:, 1] >= 0) & (ref[:, 1] < mh)
        marks = np.zeros((mh, mw), np.uint8)
        marks[ref[ok, 1], ref[ok, 0]] = 255
        marks = cv2.dilate(marks, np.ones((2, 2), np.uint8))
        under[marks > 0] = (255, 255, 255)
        numbers = np.zeros_like(under)
        fs = 0.28 * max(self.scale, 0.8) * self._mini_k
        for cell in range(1, N_CELLS + 1):
            cx, cy = to_px(g.cell_center(cell))
            (tw, tht), _ = cv2.getTextSize(str(cell), cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
            cv2.putText(numbers, str(cell), (int(cx - tw / 2), int(cy + tht / 2)), cv2.FONT_HERSHEY_SIMPLEX,
                        fs, (255, 255, 255), 1, cv2.LINE_AA)
        return under, numbers

    def _draw_minimap_legend(self, img: np.ndarray, x_off: int, y_top: int) -> int:
        lfs = 0.45 * max(self.scale, 0.8)
        rows = [[("tespit edilen cizgi", (0, 230, 255)), ("gorus disi", (60, 70, 60)),
                 ("hesaplanamiyor", (40, 40, 200))]]
        if self._objects is not None:
            rows.insert(0, [("Takim A", self._team_colors.get("A", self._UNKNOWN)),
                            ("Takim B", self._team_colors.get("B", self._UNKNOWN)),
                            ("hakem", self._REF_BLUE), ("top", self._BALL)])
        line_h = int(22 * max(self.scale, 0.8))
        for r_i, items in enumerate(rows):
            lx, ly = x_off, y_top + int(26 * self.scale) + r_i * line_h
            for txt, col in items:
                cv2.rectangle(img, (lx - 1, ly - 10), (lx + 11, ly + 2), (255, 255, 255), -1)
                cv2.rectangle(img, (lx, ly - 9), (lx + 10, ly + 1), col, -1)
                put(img, txt, (lx + 16, ly), lfs, _PANEL_TEXT)
                lx += 16 + cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, lfs, 1)[0][0] + 18
        return y_top + int(26 * self.scale) + (len(rows) - 1) * line_h + int(30 * self.scale)

    def _draw_hud(self, img: np.ndarray, res: Optional[HomographyResult], frame_idx: int, fps: float,
                  extra: list[str]) -> int:
        """Durum satırları; son satırın altındaki y'yi döndürür."""
        L = self.layout
        fs = 0.55 * max(self.scale, 0.7)
        if res is None or not res.valid:
            status, color = "KALIBRASYON YOK - izgara gizlendi", (0, 0, 255)
        elif res.source == "keypoints" and res.used_prior:
            status, color = f"kalibrasyon: sabit kamera konumu + {res.n_inliers} nokta (D-027)", (255, 200, 0)
        elif res.source == "keypoints":
            status, color = f"kalibrasyon: saha noktalari ({res.n_inliers} nokta, hata {res.reproj_error_m:.2f} m)", (0, 255, 0)
        else:
            status, color = f"kalibrasyon: kamera hareketi tahmini (F-3, {res.motion_chain_len} kare)", (0, 215, 255)
        t = frame_idx / fps if fps else 0
        y = L.title(img, "DURUM", L.side_x, L.status_y, L.side_inner_w)
        step = int(26 * max(self.scale, 0.7))
        for i, txt in enumerate([f"kare {frame_idx}   t = {t:.2f} sn", status] + extra):
            for part in wrap(txt, fs, L.side_inner_w):
                put(img, part, (L.side_x, y), fs, color if i == 1 else _PANEL_TEXT)
                y += step
        return y - step // 2
