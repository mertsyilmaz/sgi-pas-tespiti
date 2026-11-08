"""Hücre sayısı/düzeni araması: kaç hücre (>= 50) pas kuralının doğruluğunu en yükseğe çıkarır?

Etiketli pas olmadığı için simülasyon: videodaki her gerçek oyuncu konumu (çevrimdışı objects.npz) bir pasın
alıcısı sayılır. İki durum:
  - pası atan alıcının takımından  -> doğru sonuç SUCCESS
  - pası atan rakip takımdan (kesme) -> doğru sonuç FAIL
Topun varış noktasına ve tüm oyunculara konum hatası (σ, metre) eklenir; hedef hücreye kural uygulanır
(D-006: yalnızca hedef hücre, çoğunluk, eşitlikte SUCCESS, boş hücre FAIL). Puan = doğru sonuç oranı.
Büyük hücre: rakipler karışır. Küçük hücre: konum hatasıyla top yanlış/boş hücreye düşer.

Düzenler saha çizgilerinden türetilir (kenarlar ana çizgilerin üstünde, aradaki bölgeler eşit bölünür) ve
karşılaştırma için eşit ızgaralar da denenir.

Kullanım: .venv/Scripts/python.exe scripts/grid_count_search.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
L, W = 105.0, 68.0


def half_x(goal_area: bool, box_parts: int, mid_parts: int) -> list[float]:
    """0 -> 52,5 arası kenarlar: [kale alanı 5,5] + ceza sahası 16,5 + orta bölge eşit parçalar."""
    e = [0.0]
    if goal_area:
        e += [5.5]
        e += list(np.linspace(5.5, 16.5, box_parts + 1)[1:])
    else:
        e += list(np.linspace(0, 16.5, box_parts + 1)[1:])
    e += list(np.linspace(16.5, 52.5, mid_parts + 1)[1:])
    return e


def full_x(h: list[float]) -> np.ndarray:
    return np.array(h + [L - v for v in h[::-1][1:]])


def y_edges(outer_parts: int, goal_band: bool, inner_parts: int) -> np.ndarray:
    """Satırlar: taç -> ceza sahası yan (13,84) bölgesi, iç bölge (kale alanı yanları 24,84/43,16 isteğe bağlı)."""
    top = list(np.linspace(0, 13.84, outer_parts + 1))
    if goal_band:                                  # 13,84-24,84 tek, 24,84-43,16 inner_parts parça
        mid = [24.84] + list(np.linspace(24.84, 43.16, inner_parts + 1)[1:]) + [54.16]
    else:                                          # 13,84-54,16 inner_parts eşit parça
        mid = list(np.linspace(13.84, 54.16, inner_parts + 1)[1:])
    bottom = [W - v for v in top[::-1][1:]]
    return np.array(top + mid + bottom)


def layouts():
    out = {}
    for ga, bp, mp in itertools.product((False, True), (1, 2), range(1, 9)):
        xe = full_x(half_x(ga, bp, mp))
        for op, gb, ip in itertools.product((1, 2), (False, True), range(1, 5)):
            ye = y_edges(op, gb, ip)
            n = (len(xe) - 1) * (len(ye) - 1)
            if 50 <= n <= 200:
                name = (f"hizalı {len(xe) - 1}x{len(ye) - 1}: x[kale alanı {'var' if ga else 'yok'}, ceza {bp}, "
                        f"orta {mp}] y[dış {op}, {'kale alanı bandı' if gb else 'iç'} {ip}]")
                out[name] = (xe, ye)
    for c, r in itertools.product(range(8, 21), range(4, 11)):
        if 50 <= c * r <= 200:
            out[f"eşit {c}x{r}"] = (np.linspace(0, L, c + 1), np.linspace(0, W, r + 1))
    return out


def cell_of(xe, ye, p):
    cx = np.clip(np.searchsorted(xe, p[..., 0], side="right") - 1, 0, len(xe) - 2)
    cy = np.clip(np.searchsorted(ye, p[..., 1], side="right") - 1, 0, len(ye) - 2)
    return cy * (len(xe) - 1) + cx


def score(xe, ye, frames, sigma_ball, sigma_pl, rng):
    ok_s = ok_f = n = 0
    for P, T in frames:
        k = len(P)
        noisy = P + rng.normal(0, sigma_pl, P.shape)
        tgt = P + rng.normal(0, sigma_ball, P.shape)          # her alıcı için topun varış noktası
        cp = cell_of(xe, ye, noisy)                           # (k,)
        ct = cell_of(xe, ye, tgt)                             # (k,)
        inside = ct[:, None] == cp[None, :]                   # alıcı i'nin hedef hücresindeki oyuncular
        same = (inside & (T[None, :] == T[:, None])).sum(1)   # alıcının takımı
        opp = (inside & (T[None, :] != T[:, None])).sum(1)
        any_ = (same + opp) > 0
        ok_s += np.sum(any_ & (same >= opp))                  # pasör = alıcının takımı: SUCCESS doğru
        ok_f += np.sum(~any_ | (same > opp))                  # pasör = rakip: pasörün takımı (opp) < alıcınınki -> FAIL doğru
        n += k
    return ok_s / n, ok_f / n


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    z = np.load(ROOT / "outputs/grid_demo/pas_test/objects.npz")
    r = z["rows"]
    r = r[(r[:, 2] <= 1) & (r[:, 3] <= 1) & ~np.isnan(r[:, 9])]
    r = r[(r[:, 9] >= 0) & (r[:, 9] <= L) & (r[:, 10] >= 0) & (r[:, 10] <= W)]
    frames = [(r[r[:, 0] == f][:, 9:11], r[r[:, 0] == f][:, 3]) for f in np.unique(r[:, 0])[::3]]
    print(f"{len(frames)} kare, {sum(len(p) for p, _ in frames)} alıcı örneği (her biri SUCCESS + FAIL senaryosu)")
    lays = layouts()
    print(f"{len(lays)} düzen deneniyor (50–200 hücre)\n")
    res = []
    for name, (xe, ye) in lays.items():
        row = [name, (len(xe) - 1) * (len(ye) - 1)]
        for sb, sp in ((1.0, 0.5), (2.0, 1.0), (3.0, 1.5)):
            s, f = score(xe, ye, frames, sb, sp, np.random.default_rng(0))
            row += [0.5 * (s + f), s, f]
        res.append(row)
    res.sort(key=lambda x: -(x[2] + x[5] + x[8]))
    hdr = "doğruluk (SUCCESS / FAIL senaryosu) — top hatası 1 m | 2 m | 3 m"
    print(f"{'düzen':78s} {'hücre':>5s}  {hdr}")

    def show(x):
        print(f"{x[0]:78s} {x[1]:5d}  " + " | ".join(f"%{100 * x[i]:.1f} ({100 * x[i + 1]:.0f}/{100 * x[i + 2]:.0f})"
                                                  for i in (2, 5, 8)))
    print("-- en iyi 12 --")
    for x in res[:12]:
        show(x)
    print("-- hücre sayısına göre en iyi (her sayı için) --")
    best = {}
    for x in res:
        best.setdefault(x[1], x)
    for n in sorted(best)[:40]:
        show(best[n])
    print("-- başvuru --")
    for x in res:
        if x[0] in ("eşit 10x5",):
            show(x)


if __name__ == "__main__":
    main()
