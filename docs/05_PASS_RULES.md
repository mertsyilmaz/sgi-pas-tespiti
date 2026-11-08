# 05 — Pas Tespiti ve Kural Motoru

> Durum: **Kural seti kullanıcı tarafından tanımlandı ve tamamlandı (2026-10-21; D-006, D-008, D-009).**
> Sınıflandırma **ikilidir**: yalnızca **SUCCESS** ve **FAIL**. Belirsiz (UNCERTAIN) sınıfı yoktur. Belirsiz olabilecek durumlar sabit kurallarla çözülür ve bu kurallar `13_LIMITATIONS.md` dosyasında bilinen kısıtlar olarak belgelenir.

---

## A. Pas adayı tespiti (M6) — gözlem katmanı

### A.1 Sahiplik (possession) modeli
- Her karede topa en yakın oyuncu bulunur. Mesafe **saha koordinatında, metre cinsinden** ölçülür.
- `d < possession_radius` (ör. 1.5 m) koşulu en az `possession_min_frames` (ör. 3) kare boyunca sağlanırsa o oyuncu **topun sahibi** olur.
- Hiçbir oyuncu bu koşulu sağlamıyorsa durum `FREE` olur.
- Hakemler sahip olamaz.

### A.2 Pas tanımı (D-009)
- **Her vuruş bir pastır.** Şut, orta, uzaklaştırma ve sekme ayrımı yapılmaz.
- **Başlangıç (release):** Top, sahibi olan oyuncudan ayrıldığı an. **Başlangıç hücresi**, topun bu andaki hücresidir.
- **Bitiş:** Release sonrası aşağıdakilerden **ilki** gerçekleştiğinde pas biter:
  1. Başka bir oyuncu topun sahibi olur
  2. Top durur ya da yavaşlar ve sahipsiz kalır
  3. Top saha dışına çıkar (`OUT`)
- **Hedef hücre**, topun bitiş anındaki hücresidir. Kurallar bitiş karesindeki oyuncu konumlarıyla değerlendirilir.
- **Dribling istisnası:** Topu aynı oyuncu tekrar alırsa ve arada başka hiçbir oyuncu topa sahip olmamışsa bu pas değil, top sürmedir. Kayıt üretilmez.

### A.3 Ara hücreler
Başlangıç ile hedef arasında topun gerçekten girdiği hücreler zaman sırasıyla kaydedilir (bkz. `04_PITCH_GRID.md` §4). **Sınıflandırmada kullanılmaz** (R-0).

---

## B. Kural motoru (M7) — karar katmanı

### B.1 İlkeler
1. Kurallar yalnızca `configs/pass_rules.yaml` dosyasındadır. Kod içinde sabit kural bulunmaz.
2. Kural motoru yalnızca `PassCandidate` girdisini kullanır.
3. Her kararın yanında `reason`, eşleşen kural ID'si ve uygulanan yedek kurallar (`fallbacks_applied`) kaydedilir. Sınıf ikili olsa da kararın nasıl verildiği izlenebilir kalır.

### B.2 Tanımlar
- **Pasör takımı** `T`: Pası atan oyuncunun takımı. Kurallar A ve B için simetriktir.
- **Takım arkadaşları** `M`: Bitiş karesinde hedef hücrede bulunan, takımı `T` olan oyuncular. Kaleci dahildir. **Pasör hariçtir.** Pasör aynı hücrede olsa bile sonucu değiştirmez.
- **Rakipler** `O`: Hedef hücrede bulunan, takımı `T` olmayan oyuncular. Kaleci dahildir.
- **Hakemler** her iki kümeden de hariçtir.
- **Yalnızca hedef hücre** dikkate alınır. Komşu hücrede, topa çok yakın duran oyuncular sayılmaz.
- Oyuncunun hücresi, ayak noktasının (bbox alt-orta) saha koordinatına göre belirlenir.
- Hücum yönü ve devre arası taraf değişimi kuralları etkilemez.

### B.3 Sınıflandırma kuralları

| ID | Koşul (bitiş anında) | Sonuç |
|---|---|---|
| **R-0** | Ara hücreler | Değerlendirilmez, yalnızca kaydedilir |
| **R-OUT** | Top saha dışına çıktı | **FAIL** |
| **R-1** | Hedef hücrede yalnızca takım arkadaşı var | **SUCCESS** |
| **R-2** | Hedef hücrede yalnızca rakip var | **FAIL** |
| **R-3** | Hem takım arkadaşı hem rakip var | Topa en yakın oyuncu takım arkadaşıysa **SUCCESS**, rakipse **FAIL** |
| **R-3T** | R-3'te mesafeler eşit veya fark `tie_epsilon_m` değerinden küçük | **SUCCESS** |
| **R-4** | Hedef hücrede pasör dışında hiç oyuncu yok | **FAIL** |

Örnekler:
- Pas 3 numaralı hücreden başlar, top 7 üzerinden geçip 12'de biter. 7'de kimse olmaması önemsizdir. Karar 12'deki oyunculara göre verilir. (Numaralar D-038 düzenine göredir.)
- Pas 3 numaralı hücrede başlar ve biter. Hücrede pasör dışında bir takım arkadaşı varsa ve top ona daha yakınsa sonuç SUCCESS olur. Hücrede yalnızca pasör varsa sonuç FAIL olur (R-4).

### B.4 Veri eksikliği için yedek kurallar (fallback)
Belirsiz sınıfı olmadığı için, verinin eksik veya güvenilmez olduğu durumlar **sabit ve belgelenmiş** kurallarla çözülür. Her uygulama pas kaydında `fallbacks_applied` alanına yazılır.

| ID | Durum | Uygulanan kural |
|---|---|---|
| F-1 | Top kısa süre görünmüyor (≤ `max_ball_gap_frames`) | Konum saha koordinatında doğrusal interpolasyonla doldurulur. Pas devam eder |
| F-2 | Top uzun süre görünmüyor (kamera kesmesi, tekrar gösterimi vb.) | Pasın bitişi, topun **son görüldüğü** konum ve kare kabul edilir. R-1…R-4 o kareye uygulanır |
| F-3 | Bitiş karesinde saha çizgisi/keypoint yetersiz, homografi doğrudan hesaplanamıyor | Önceki (ve sonraki) karelerdeki geçerli homografi, **kamera hareketi kestirimiyle** bu kareye taşınır. Kamera hareketi de kestirilemezse (sahne kesmesi) zamanca en yakın geçerli homografi kullanılır. Ayrıntı: `03_ARCHITECTURE.md` §4, D-012 |
| F-4 | Bir oyuncunun takımı o karede belirlenemiyor | Takip boyunca çoğunluk oyu ile belirlenen takım kullanılır. Hiç belirlenemediyse oyuncu yok sayılır |
| F-5 | Release anında sahip net değil | Release karesinde topa en yakın oyuncu pasör kabul edilir |

> F-1…F-5 kullanıcı tarafından onaylanmıştır (2026-10-21). F-3, kullanıcı talebiyle kamera hareketi tahminine dayalı hale getirilmiştir.

### B.5 Değerlendirme sırası
```
1. Yedek kurallar (F-1…F-5) ile eksik veriyi tamamla → fallbacks_applied
2. Top OUT ise               → FAIL   (R-OUT)
3. M, O kümelerini hedef hücreden çıkar (pasör ve hakem hariç)
4. |M|>0, |O|=0              → SUCCESS (R-1)
   |M|=0, |O|>0              → FAIL    (R-2)
   |M|>0, |O|>0              → en yakın oyuncuya göre (R-3, eşitlikte R-3T → SUCCESS)
   |M|=0, |O|=0              → FAIL    (R-4)
5. Ara hücreleri kayda yaz   (R-0)
```

### B.6 Konfigürasyon (taslak `configs/pass_rules.yaml`)
```yaml
version: 1.0
labels: [SUCCESS, FAIL]
target_cell_rules:
  exclude_passer: true
  exclude_roles: [referee]
  include_goalkeepers: true
  neighbor_cells: false          # yalnızca hedef hücre
  out_of_pitch: FAIL             # R-OUT
  tie_epsilon_m: 0.3
  tie_result: SUCCESS            # R-3T
  outcomes:
    only_teammates: SUCCESS      # R-1
    only_opponents: FAIL         # R-2
    both: nearest_to_ball        # R-3
    none: FAIL                   # R-4
intermediate_cells:
  evaluate: false                # R-0
fallbacks:
  max_ball_gap_frames: 15        # F-1 / F-2 sınırı
  homography_search_frames: 30   # F-3 (son çare: nearest)
  max_motion_chain: 50           # F-3: kamera hareketiyle en fazla kaç kare taşınabilir
pass_definition:
  every_kick_is_pass: true       # D-009
  same_player_regain_is_dribble: true
```

---

## C. Çıktı ve sayım (D-008)
Raporlanan istatistikler yalnızca şunlardır:

| | Başarılı | Başarısız |
|---|---|---|
| Takım A | n | n |
| Takım B | n | n |

- Her pas tek kayıttır (`pass_id`). Sahiplik segmentlerinden üretildiği için aynı pas birden fazla sayılmaz.
- Pas, **pasörün takımına** yazılır.
- **Hedef takım modu (D-040):** İstenirse yalnızca seçilen takımın (ör. A) pasları sayılır ve gösterilir. Kurallar değişmez; diğer takımın pasları tespit edilir ama sayılmaz.
