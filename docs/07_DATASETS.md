# 07 — Veri Setleri

> Durum: SoccerNet incelemesi tamamlandı (2026-10-21, bkz. `08_REPO_ANALYSIS.md`). Erişim ve indirme henüz yapılmadı.
> Tüm SoccerNet verileri artık **Hugging Face** üzerindedir (`huggingface.co/SoccerNet`, `OpenSportsLab`). İndirme aracı: `pip install SoccerNet`.

## İhtiyaç → veri seti eşlemesi
| İhtiyaç | Birincil kaynak | Yedek / ek | Kullanıldığı modül |
|---|---|---|---|
| Oyuncu, kaleci, hakem bbox | **SN-GSR** | SN-Tracking, SoccerNet-v3 | M2 eğitim + test |
| **Top bbox** | **SN-GSR + SN-Tracking** | Kendi etiketimiz | M2 top modeli |
| Saha çizgisi / keypoint | **SN-Calibration-2023** | SN-GSR `pitch.lines` | M2 keypoint, M4 |
| Saha koordinatında oyuncu konumu (GT) | **SN-GSR `bbox_pitch`** | GAR takip | M4 değerlendirme |
| Takım etiketi (GT) | **SN-GSR `team`** (sol/sağ) | GAR `team` | M3b değerlendirme |
| Takip kimliği (GT) | SN-Tracking | SN-GSR | M3 değerlendirme |
| Pas olayı (zaman + takım) | **SoccerNet-GAR** | SN-BAS-2025 | M6 değerlendirme |
| Saha koordinatında oyuncu + top izleri (GT) | **SoccerNet-GAR `tracking-full`** | — | M5–M7 mantık testi |
| Başarılı/başarısız pas etiketi | **Yok** → kendi elle etiketimiz | GAR'dan yarı-otomatik türetme | M7 değerlendirme |

## Veri seti kartları

### SN-GSR (Game State Reconstruction) — ⭐ birincil
- **Kaynak:** HF `SoccerNet/SN-GSR-2024`, `SN-GSR-2025` · repo `sn-gamestate`
- **Erişim:** Herkese açık
- **Boyut:** 200 klip × 30 sn (train/valid/test/challenge), ana yayın kamerası. Veri sürümü ≥ 1.3 olmalı.
- **Etiketler:** `Labels-GameState.json` (COCO benzeri):
  - nesne: `bbox_image`, `bbox_pitch`, `track_id`, `role ∈ {player, goalkeeper, referee, ball, other}`, `team ∈ {left, right}`, `jersey`
  - `pitch`: görüntü başına saha çizgileri
  - `camera`: kamera parametreleri
- **Bizim şemamızla eşleşme:** player/goalkeeper/referee/ball doğrudan eşleşir. `other` yok sayılır. left/right → A/B (video bazında konfig).
- **Kullanım:** Eğitim (dedektör), doğrulama (takım, homografi), test.
- **Not:** Top için `bbox_pitch` olup olmadığı indirildikten sonra doğrulanacak.

### SN-Tracking-2023
- **Kaynak:** HF `SoccerNet/SN-Tracking-2023` · repo `sn-tracking` · açık
- **Boyut:** 12 maç, 200 klip × 30 sn + 1 tam devre
- **Format:** MOT CSV (`frame, id, x, y, w, h, conf, -1, -1, -1`) + `gameinfo.ini` (ID → sınıf/takım)
- **Kullanım:** Top dedektörü eğitimi (ek veri), takip değerlendirmesi (HOTA)

### SN-Calibration-2023
- **Kaynak:** HF `SoccerNet/SN-Calibration-2023` · repo `sn-calibration` · açık
- **Etiketler:** SoccerNet-v3 görüntülerinde 26 semantik çizgi sınıfının uç noktaları ve daire noktaları
- **Kullanım:** Saha keypoint/çizgi modeli ince ayarı ve değerlendirmesi. Saha şablonu referansı (`soccerpitch.py`).

### SoccerNet-GAR — ⭐ değerlendirme için birincil
- **Kaynak:** HF `OpenSportsLab/SoccerNet-GAR` · repo `sn-gar-2027`
- **Erişim:** **Gated + NDA**, manuel onay. Yalnızca araştırma ve ticari olmayan kullanım, paylaşılamaz.
- **Boyut:** Dünya Kupası 2022'nin 64 maçı, 87.939 olay (train 62.159 / valid 12.091 / test 13.689)
- **Sınıflar:** PASS, HIGH PASS, CROSS, HEADER, OUT, THROW IN, SHOT, PLAYER SUCCESSFUL TACKLE, FREE KICK, GOAL. Olay başına `team` (home/away) ve `position_ms` var.
- **Modaliteler:** yayın videosu (~29.97 fps), kareler, takip (16 kare), **takip-full (144 kare, 30 fps)**
- **Takip:** Saha koordinatında 2B oyuncu konumu (x ∈ [-60, 60], y ∈ [-42, 41] m), **3B top konumu** (z ∈ [-8, 25] m), oyuncu pozisyon rolü. Otomatik çıkarılıp elle düzeltilmiş.
- **Kullanım:**
  - M5–M7'nin görüntüden bağımsız mantık testi
  - M6 pas tespiti recall ve precision ölçümü
  - Uçtan uca karşılaştırma
- **Dönüşüm:** Merkez orijinli koordinatlar → bizim `[0,105]×[0,68]` sistemimize (adaptör)

### SN-BAS-2024 / 2025 (Ball Action Spotting) — opsiyonel
- **Kaynak:** HF `SoccerNet/SN-BAS-2025` · repolar `sn-spotting`, `sn-teamspotting` · **NDA-gated**
- **İçerik:** 12 sınıf yoğun top aksiyonu (Pass, Drive, High Pass, Cross…), 25 fps, 2025'te takım (sol/sağ)
- **Kullanım:** M6 için ek değerlendirme. GAR erişimi olmazsa yedek.

### SoccerNet-v3 — opsiyonel
- Tekil karelerde bbox + takım + saha çizgileri. ~60 GB kare + 1 GB etiket. Ek eğitim verisi olarak kullanılabilir.

## Kendi değerlendirme setimiz (zorunlu, §9)
Hiçbir veri setinde **başarılı/başarısız pas** etiketi yoktur. Plan:
- 3–5 kısa klip (ekip seçecek, D-010). Mümkünse GAR veya SN-GSR videolarından seçilir; böylece GT konumlarla çapraz kontrol yapılabilir.
- Her pas için başlangıç ve bitiş karesi, pasör, alıcı, hedef hücre ve beklenen etiket elle işaretlenir → `data/annotations/passes_<clip>.csv`
- Ayrıntılar `11_EVALUATION.md` dosyasında.
