"""Hücre düzeni analizi: hücre kenarlarının saha çizgilerine oturması ve hücre boyutunun etkisi.

Gerçek oyuncu/top saha konumları (çevrimdışı objects.npz) üzerinde düzenler karşılaştırılır:
  - çizgi uyumu: ana düz saha çizgilerinin en yakın hücre kenarına uzaklığı
  - konum hatasına duyarlılık: konuma σ metre gürültü eklenince hücresi değişen konum oranı
  - kural duyarlılığı: oyuncunun kendi hücresinde rakip sayısı takım arkadaşından fazla olma oranı
    (o oyuncuya giden pas, oyuncu topu alsa da FAIL sayılırdı; büyük hücre = daha çok çekişme)
  - boş hedef: topun bulunduğu hücrede (topu tutan hariç) hiç oyuncu olmama oranı

Kullanım: .venv/Scripts/python.exe scripts/grid_layout_analysis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
L, W = 105.0, 68.0
# ana düz saha çizgileri (IFAB standart ölçüleri, configs/pitch.yaml sistemi)
X_LINES = {"kale çizgisi": [0, 105], "kale alanı": [5.5, 99.5], "ceza sahası": [16.5, 88.5], "orta çizgi": [52.5]}
Y_LINES = {"taç çizgisi": [0, 68], "ceza sahası yan": [13.84, 54.16], "kale alanı yan": [24.84, 43.16]}

MID = list(np.linspace(16.5, 52.5, 5))                       # ceza sahası ile orta çizgi arası 4 eşit sütun (9 m)
LAYOUTS = {
    "U 10x5 = 50 (mevcut, eşit)": (np.linspace(0, L, 11), np.linspace(0, W, 6)),
    "H 10x5 = 50 (çizgilere hizalı)": ([0] + MID + [105 - v for v in MID[::-1][1:]] + [105],
                                      [0, 13.84, 24.84, 43.16, 54.16, 68]),
    "U 12x5 = 60 (eşit)": (np.linspace(0, L, 13), np.linspace(0, W, 6)),
    "H 12x5 = 60 (kale alanı da hizalı)": ([0, 5.5] + MID + [105 - v for v in MID[::-1][1:]] + [99.5, 105],
                                          [0, 13.84, 24.84, 43.16, 54.16, 68]),
}


def cell_of(xe, ye, p):
    cx = np.clip(np.searchsorted(xe, p[:, 0], side="right") - 1, 0, len(xe) - 2)
    cy = np.clip(np.searchsorted(ye, p[:, 1], side="right") - 1, 0, len(ye) - 2)
    return cy * (len(xe) - 1) + cx


def load(path, ball):
    z = np.load(path)
    r = z["rows"]
    r = r[~np.isnan(r[:, 9])]
    if ball:
        r = r[(r[:, 2] == 3) & ((r[:, 12] == 0) if r.shape[1] > 12 else True)]
    else:
        r = r[(r[:, 2] <= 1) & (r[:, 3] <= 1)]                # oyuncu + kaleci, takımı A/B
    inside = (r[:, 9] >= 0) & (r[:, 9] <= L) & (r[:, 10] >= 0) & (r[:, 10] <= W)
    return r[inside]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    out = ROOT / "outputs/grid_demo/pas_test"
    players = load(out / "objects.npz", ball=False)
    ball = np.vstack([load(out / s / "objects.npz", ball=True) for s in ("s0_d20", "s28_d20")])
    pl_new = np.vstack([load(out / s / "objects.npz", ball=False) for s in ("s0_d20", "s28_d20")])
    print(f"oyuncu konumu {len(players)}, top konumu {len(ball)} (tespit)\n")
    rng = np.random.default_rng(0)
    pp = players[:, 9:11]

    for name, (xe, ye) in LAYOUTS.items():
        xe, ye = np.asarray(xe, float), np.asarray(ye, float)
        n = (len(xe) - 1) * (len(ye) - 1)
        dx, dy = np.diff(xe), np.diff(ye)
        areas = np.outer(dy, dx)
        print(f"== {name}: {n} hücre, en {dx.min():.1f}–{dx.max():.1f} m × boy {dy.min():.1f}–{dy.max():.1f} m, "
              f"alan {areas.min():.0f}–{areas.max():.0f} m²")
        al = []
        for lines, edges in ((X_LINES, xe), (Y_LINES, ye)):
            for ln, vals in lines.items():
                d = max(np.min(np.abs(edges - v)) for v in vals)
                al.append(f"{ln} {'✓' if d < 0.01 else f'{d:.1f} m'}")
        print("   çizgi uyumu: " + ", ".join(al))
        c0 = cell_of(xe, ye, pp)
        flips = []
        for s in (0.5, 1.0, 2.0):
            c1 = cell_of(xe, ye, pp + rng.normal(0, s, pp.shape))
            flips.append(f"σ={s:g} m %{100 * np.mean(c0 != c1):.1f}")
        print("   konum hatasıyla hücre değişimi (oyuncu): " + ", ".join(flips))

        # kural duyarlılığı: oyuncunun hücresinde rakip > takım arkadaşı (kendisi dahil)
        contested, alone, box_cont, box_n = 0, 0, 0, 0
        for f in np.unique(pl_new[:, 0]):
            fr = pl_new[pl_new[:, 0] == f]
            cells = cell_of(xe, ye, fr[:, 9:11])
            for i in range(len(fr)):
                same = cells == cells[i]
                mates = np.sum(same & (fr[:, 3] == fr[i, 3]))
                opp = np.sum(same & (fr[:, 3] != fr[i, 3]))
                c = opp > mates
                contested += c
                alone += (mates == 1 and opp == 0)
                if fr[i, 9] < 16.5 or fr[i, 9] > 88.5:
                    box_cont += c
                    box_n += 1
        tot = len(pl_new)
        print(f"   kural: oyuncunun hücresinde rakip çoğunlukta %{100 * contested / tot:.1f} "
              f"(ceza sahası bölgesinde %{100 * box_cont / max(box_n, 1):.1f}), hücrede yalnız %{100 * alone / tot:.1f}")

        # boş hedef: topun hücresinde, topu tutan (en yakın, <= 1,5 m) hariç oyuncu yok
        empty = 0
        for b in ball:
            fr = pl_new[pl_new[:, 0] == b[0]]
            if len(fr) == 0:
                continue
            cells = cell_of(xe, ye, fr[:, 9:11])
            bc = cell_of(xe, ye, b[None, 9:11])[0]
            d = np.hypot(fr[:, 9] - b[9], fr[:, 10] - b[10])
            others = (cells == bc) & ~((d == d.min()) & (d <= 1.5))
            empty += not others.any()
        print(f"   topun hücresinde (sahibi hariç) oyuncu yok: %{100 * empty / len(ball):.1f}\n")


if __name__ == "__main__":
    main()
