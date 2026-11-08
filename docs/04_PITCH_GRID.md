# 04 — Saha Koordinat Sistemi ve 50 Hücreli Izgara

> Durum: **ONAYLANDI (2026-10-21, D-004).** 10 sütun × 5 satır eşit ızgara kullanılacak. **Hücre sayısı (50) değiştirilemez.**

## 1. Saha koordinat sistemi
- Saha ölçüsü (varsayılan): **105 m × 68 m** (FIFA standardı). Gerçek ölçü `configs/pitch.yaml` dosyasında parametredir.
- Orijin: Kamera yayın tarafındaki taç çizgisi ile **sol** kale çizgisinin kesiştiği köşe.
- `x` ekseni: Sol kale çizgisinden sağa doğru, saha boyu boyunca `[0, 105]`.
- `y` ekseni: Yayın tarafının karşısındaki taç çizgisinden yayın tarafına doğru `[0, 68]`. Kesin yön, keypoint modelinin şablonuna göre sabitlenecek.
- Birim: metre.

> Not: Sol ve sağ, kameraya göre tanımlıdır. Hücum yönü ve devre arası taraf değişimi **kuralları etkilemez** (D-006). Bu yüzden numaralandırma mutlaktır.

## 2. Düzen: 10 sütun × 5 satır (eşit boyutlu) — onaylı
- Sütun genişliği: 105 / 10 = **10.5 m**
- Satır yüksekliği: 68 / 5 = **13.6 m**
- `col = floor(x / 10.5)` → 0..9, `row = floor(y / 13.6)` → 0..4
- **`cell_id = col * 5 + (4 - row) + 1`** → 1..50 (D-038, hocanın geri bildirimi 2026-10-28)
  - Numaralandırma **sol alt köşeden** (sol kale çizgisi + yayın tarafındaki yakın taç çizgisi) başlar.
  - Her sütunda **alttan yukarı** artar: 1. sütun 1–5, 2. sütun 6–10, …, 10. sütun 46–50.
  - Önceki düzen (D-004 ilk hali: `row * 10 + col + 1`, sol üstten satır satır) bu tarihten önceki çıktılarda geçerlidir.
- Sınır kuralı: Hücre aralıkları **yarı açıktır** `[min, max)`. Yalnızca en sağ ve en alt sınırlar kapalıdır (`x = 105` → col 9, `y = 68` → row 4). Böylece her nokta tam olarak bir hücreye düşer.

```
            x →  0   10.5  21   31.5  42  52.5  63  73.5  84  94.5  105
 y   0    ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐   uzak taç çizgisi
 ↓        │  5 │ 10 │ 15 │ 20 │ 25 │ 30 │ 35 │ 40 │ 45 │ 50 │  row 0
   13.6   ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤
          │  4 │  9 │ 14 │ 19 │ 24 │ 29 │ 34 │ 39 │ 44 │ 49 │  row 1
   27.2   ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤
  Sol     │  3 │  8 │ 13 │ 18 │ 23 │ 28 │ 33 │ 38 │ 43 │ 48 │  row 2   Sağ
  kale    ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤          kale
   40.8   │  2 │  7 │ 12 │ 17 │ 22 │ 27 │ 32 │ 37 │ 42 │ 47 │  row 3
   54.4   ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤
          │  1 │  6 │ 11 │ 16 │ 21 │ 26 │ 31 │ 36 │ 41 │ 46 │  row 4
   68     └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘   yakın taç çizgisi (yayın kamerası)
                              orta saha çizgisi = x 52.5 (25|30 sütunları arası)
```

**Gerekçe:** Basit, simetrik ve tekrarlanabilir. Orta saha çizgisi bir hücre sınırıyla çakışır (col 4 | col 5).

Ceza sahası hücre sınırlarıyla hizalı değildir. Kurallar sabit hücrelere atıf yapmadığı için bu bir sorun oluşturmaz.

## 3. Özel durumlar
| Durum | Değer | Açıklama |
|---|---|---|
| Nokta saha dışında | `OUT` | 50 hücreden biri değildir. Kural motoru "top oyun alanını terk etti" kanıtı olarak kullanabilir |
| Homografi geçersiz / konum yok | `UNKNOWN` | Belirsizlik kaynağıdır. **Hiçbir hücreye zorla atanmaz** |
| Sınır üstü (ör. x = 52.5) | Yarı açık aralık kuralı | Deterministiktir |

Saha dışı için küçük bir tolerans (ör. 0.5 m) `grid.yaml` dosyasında parametredir. Homografi hatası çizgi üstündeki topu yanlışlıkla `OUT` yapmamalıdır.

## 4. Top hücre dizisinin çıkarılması (ara hücreler)
> D-006 gereği ara hücreler **sınıflandırmayı etkilemez**. Şartname §8 gereği yalnızca kayıt ve görselleştirme için tutulur.

1. Pas penceresindeki her kare için topun `cell_id` değeri alınır (`UNKNOWN` olanlar işaretli kalır).
2. **Gürültü filtresi:** Bir hücre en az `min_frames_in_cell` (ör. 2–3) kare boyunca gözlenmedikçe dizide yeni bir geçiş sayılmaz. Bu, tek karelik sıçramaları bastırır.
3. Ardışık tekrarlar birleştirilir: `[12,12,12,13,13,14]` → `[12,13,14]`.
4. **Hücre uydurulmaz:** İki gözlem arasında atlanan bir hücre geometrik olarak arada olsa bile eklenmez. Böyle bir boşluk, dizide `gap` işaretiyle belirtilir (ör. `[12, <gap>, 14]`) ve güven düşürülür.
5. Interpolasyonla üretilmiş top konumlarından gelen hücreler ayrı işaretlenir (`interpolated=True`). Kural motoru bunları kanıt olarak kabul edip etmeyeceğine konfigürasyonla karar verir.

## 5. Uygulama notları
- `grid.yaml` düzeni (sütun ve satır sınırları) açıkça tutar. Kod sınırları buradan okur.
- `grid` modülü saf fonksiyonlardan oluşur ve birim testleriyle tüm sınır durumları doğrulanır.
- Görselleştirme: Hücre numaralarının çizildiği 2B saha görüntüsü ve videonun üzerine homografi tersiyle izdüşürülmüş ızgara. Bu ikincisi eşlemenin doğruluğunu gözle kontrol etmek için en iyi yöntemdir.
