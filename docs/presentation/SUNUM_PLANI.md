# Ara Sunum Planı — Futbol Pas Tespiti ve Analiz Sistemi

> Amaç: Hocaya şimdiye kadar yapılanları anlatmak.
> Vurgu: **saha geometrisi → 50 hücre → pas kuralları**. Oyuncu/top tespiti yalnızca 1 slaytta: kullanılan hazır modeller ve bilgileri.
> Slaytlarda ekip içi iş bölümünden söz edilmez.
> Tüm sayılar `pas_test.mp4` (64,6 sn, 1614 kare, 1280×720, Barcelona–Real Madrid) üzerinden ölçülmüştür; kaynak karar numarası parantez içinde (D-xxx) verilmiştir.
> ⚠ işaretli satırlar sunumdan önce doğrulanmalı veya tamamlanmalıdır.

---

## Bölüm 1 — Giriş (2 slayt)

### S1. Başlık
- Proje adı, ders, isimler, tarih
- Arka planda `passes_overlay.mp4`'ten bir kare (ızgara + mini harita + pas paneli)

### S2. Problem ve sistem akışı
- Girdi: yayın kamerası maç videosu. Çıktı: Takım A ve B için başarılı / başarısız pas sayıları
- Akış şeması (tek satır, kutular):
  `Video → Tespit (oyuncu/top) → Takip → Takım → Saha noktaları → Homografi → 50 hücre → Pas tespiti → Kurallar → Sayım`
- Sunumun odak kısmı (Saha noktaları → Sayım) şemada vurgulanır

---

## Bölüm 2 — Tespit: kullanılan hazır modeller (1 slayt)

### S3. Oyuncu / top tespiti — kullanılan hazır modeller
| Bileşen | Kullanılan | Kaynak / eğitim verisi | Bu videodaki sonucu |
|---|---|---|---|
| Oyuncu, kaleci, hakem, top | `roboflow/sports` futbol modeli (YOLOv8x, 1280 px) | GitHub `roboflow/sports` (MIT). Eğitim verisi Roboflow Universe "football-players-detection" (Bundesliga yayın görüntüleri) ⚠ görüntü sayısı ve mAP sayfadan doğrulanacak | Top bulunan kare **%80** (COCO YOLO11l ile %73) (D-024) |
| İlk deneme (bırakıldı) | COCO YOLO11l (`person`, `sports ball`) | Ultralytics, COCO | Top %73, rol ayrımı yok (D-022) |
| Takip | BoT-SORT (Ultralytics) | — | ByteTrack'e göre kimlik sayısı 178 → 168, yanlış kaleci 24 → 0 (D-024) |
| Takım ataması | Forma rengi + K-means (kendi kodumuz, eğitimsiz) | — | Barcelona = A, Real = B, hakemler H doğru (görsel kontrol) (D-022) |

- Görsel: `outputs/grid_demo/pas_test/objects_full.jpg` (etiketli oyuncular)

---

## Bölüm 3 — Saha geometrisi ve 50 hücre (ana bölüm, 5–6 slayt)

### S4. Mimari: model bağımsız tasarım (D-011)
- Her öğrenilmiş bileşen bir arayüzün arkasında: `ObjectDetector`, `PitchKeypointDetector`, `Tracker`, `TeamAssigner`, `CameraMotionEstimator`, `PoseEstimator`
- Model seçimi `configs/grid_demo.yaml` üzerinden; pipeline kodunda model kütüphanesi yok
- Kanıt: dedektör COCO → roboflow, saha modeli ağırlıkları SV → SV_FT_TSWC, takip ByteTrack → BoT-SORT değişimleri yalnızca konfigle yapıldı
- Küçük diyagram: arayüz ← adaptör ← konfig

### S5. Saha noktaları ve homografi
- **NBJW** (No Bells, Just Whistles; HRNet tabanlı, SoccerNet-Calibration ile eğitilmiş, TS-WorldCup ile ince ayarlı `SV_FT_TSWC` ağırlıkları kullanılıyor) ⚠ makale künyesi doğrulanacak
- Model yalnızca **görüntüdeki saha noktalarını** verir. Homografiyi kendimiz hesaplıyoruz:
  - Görüntü noktası ↔ 105×68 m saha noktası eşleşmeleri → **RANSAC ile homografi** (H, 3×3)
  - Oyuncunun ayak noktası (kutunun alt ortası) `H⁻¹` ile metreye çevrilir
- **Doğrulama katmanı (D-017):** 4–7 noktalı çözümler düşük hatayla "geçerli" görünüp tamamen yanlış olabiliyordu (140 m'ye varan sapma). H, gerçek kamera parametrelerine (odak, dönüş, yükseklik) ayrıştırılıp fiziksel olarak mümkün mü diye sınanıyor + en az 8 inlier
- Görsel: `goal_kps.jpg` veya `projected_lines.jpg` (izdüşürülmüş saha çizgileri gerçek çizgilerin üstünde)

### S6. 50 hücreli ızgara
- 105×68 m saha → **10 sütun × 5 satır**, hücre 10,5 m × 13,6 m (D-004)
- Numaralandırma: sol alt köşe 1, sütun sütun alttan yukarı (1–5, 6–10, …, 46–50) (D-038)
- Hücre sahaya bağlı, görüntüye değil: kamera dönünce/zoom yapınca aynı hücre aynı saha alanını gösterir
- Görsel: mini harita (numaralı) + videoda perspektifli ızgara yan yana (`minimap_full.jpg`)

### S7. Saha noktası görünmediğinde: kamera hareketiyle taşıma (F-3, D-012)
- Sorun: orta sahada, yakın planda veya hızlı kamera hareketinde yeterli saha noktası yok
- Çözüm: Ardışık kareler arasında **Lucas-Kanade optik akış + RANSAC** ile kamera hareketi `M` kestirilir; `H_t = H_{t-1} · M`. Nokta görünür olunca yeniden sabitlenir
- **Sahne kesmesi algılama (D-033):** renk histogramı farkı > 0,35 ise taşıma durur; eski ızgara yakın çekimin üstüne çizilmez
- Ölçüm: Her 5 karede bir nokta kullanıp aradakileri taşıma → her karede nokta kullanmaya göre fark **medyan 0,23 m, en çok 0,51 m** (D-013, iniesta örneği)

### S8. Kaymaları düzeltmek: çizgilere oturtma (D-020)
- Taşıma zamanla kayar (1–5 m). Görüntüdeki beyaz çizgiler bulunur (çim maskesi + parlaklık), saha modelinin çizgileri izdüşürülür ve H küçük bir düzeltmeyle çizgilere **oturtulur** (chamfer uzaklığı, `least_squares`)
- Sınırlar: düzeltme ≤ 4 m (paralel çizgiler 5,5 m arayla; yanlış çizgiye oturmayı önler)
- Ölçüm (pas_test): doğru kalibre kare (OK) **%65,6 → %81,7**, yaklaşık (kaymış) **%18,9 → %3,7**
- Görsel: `before_after.jpg`

### S9. Az noktayla kalibrasyon: sabit kamera konumu (D-027, D-034)
- Fikir: yayın kamerası tribünde sabit durur, yalnızca döner ve zoom yapar. Kameranın 3B konumu iyi karelerden **otomatik öğrenilir**; sonra bilinmeyen 8 → 4'e iner, **2–3 nokta yeter**
- Kontrollü test: yalnızca 3 noktayla çözüm, tam kalibrasyondan **medyan 0,35 m** farklı
- Hiç kalibrasyon yokken (video başı, orta saha) kamera parametreleri noktalar + çizgiler + çimden doğrudan çözülür (D-034)
- Ölçüm tablosu (tam video):

  | | Önce | Sonra |
  |---|---|---|
  | Kalibre kare | %87,3 | **%94,8** |
  | Doğru (OK) | %81,7 | **%93,8** |
  | Kaymış (E4!/E5!) | %1,6 | **%0** |
  | İlk ızgaranın gelişi (video başı) | 5,4 sn | **0,0 sn** (canlı 0,76 sn) |

- ⚠ Not: Bu tablo D-027 anındaki ölçüm. D-028 (yalnızca geçmiş kareler) ve D-034 sonrası tam video kalibrasyon raporu bir kez yeniden üretilmeli (`scripts/calib_report.py`)

---

## Bölüm 4 — Top ve oyuncu ile ilgili ekler (2 slayt)

### S10. Top takibi: yalnızca geçmiş kareler (D-028)
- İlke: hiçbir adım ileriki kareleri kullanmaz → çevrimdışı ölçüm canlı performansı yansıtır
- Top durumları: **tespit** / **oyuncuda** (kaybolunca son sahibinde, ≤ 3 sn) / **tahmin** (son hızla ≤ 0,5 sn)
- Hız kapısı (35 m/sn) + saha işareti filtresi: dedektör penaltı noktasını top sanıyordu (D-028, D-032)
- Ölçüm (28–48 sn): penaltı noktasının top sanıldığı kare **302/488 → 0**; topun konumu bilinen kare **%98** (tespit %73 + oyuncuda %21 + tahmin %4)

### S11. Sırtı dönük oyuncu topu saklıyor olabilir (D-035)
- Fikir: top kaybolduğunda sahibinin sırtı kameraya dönükse top gövdenin arkasındadır
- Poz modeli (YOLO11m-pose) yalnızca bu durumda, yalnızca o oyuncunun kırpıntısında çalışır; omuz/kalça sırasından yön bulunur
- Sırtı dönükse "oyuncuda" süresi 3 → 8 sn'ye uzar
- Dürüst sonuç: pas_test'te en uzun kayıp 0,68 sn olduğu için bu videoda sonuca etkisi yok; 50 kırpıntılık görsel kontrolde ~38 kararda ~3 hata

---

## Bölüm 5 — Pas tespiti ve kurallar (2–3 slayt)

### S12. Pas nasıl bulunuyor? (D-037, D-009)
- Sahiplik: top bir oyuncuya ≤ 1,5 m ve en az 3 kare → o oyuncu sahip
- Pas başlangıcı: top sahibinden 1,5 m'den fazla uzaklaşır
- Pas bitişi: başka oyuncuya ulaşma / durma (< 1,5 m/sn, 10 kare) / saha dışı — hangisi önce
- Her vuruş pas (şut/orta ayrımı yok); aynı oyuncu geri alırsa top sürme
- Sahte pasları ayıklama: 40 m/sn üstü "pas" ölçüm sıçraması; takip kimliği değişen oyuncu aynı kişi sayılır → **49 → 27 pas**
- Başlangıç hücresi, ara hücreler, hedef hücre kaydedilir

### S13. Kurallar: hedef hücreye göre (D-006, D-008)
| Kural | Hedef hücrede (pasör hariç) | Sonuç |
|---|---|---|
| R-OUT | Top saha dışında | FAIL |
| R-1 | Yalnızca takım arkadaşı | SUCCESS |
| R-2 | Yalnızca rakip | FAIL |
| R-3 | İkisi de var | Topa en yakın oyuncunun takımı |
| R-3T | Eşit mesafe (< 0,3 m) | SUCCESS |
| R-4 | Kimse yok | FAIL |
| R-0 | Ara hücreler | Kaydedilir, değerlendirilmez |
- Yedek kurallar (eksik veri): F-1 kısa top kaybı doğrusal doldurma, F-2 uzun kayıpta son görülen konum, F-3 kamera hareketiyle homografi, F-4 takımı belirsiz oyuncu yok sayılır, F-5 sahip net değilse en yakın oyuncu pasör
- Kurallar kodda değil `configs/pass_rules.yaml` içinde
- Görsel: mini haritada bir pasın başlangıç / hedef hücresi

### S14. Sonuç ve hedef takım modu (D-040)
- pas_test: **27 pas** — A: 16 başarılı / 7 başarısız, B: 0 / 4
- İki çıktı videosu: iki takımlı (`passes_overlay.mp4`) ve yalnızca hedef takım (`passes_overlay_A.mp4`)
- ⚠ **Doğruluk henüz ölçülmedi.** Sunumdan önce pas_test'teki paslar elle etiketlenirse precision / recall / sonuç doğruluğu bu slayta eklenir. Etiketlenmezse "elle doğrulama sıradaki adım" diye açıkça söylenir

---

## Bölüm 6 — Performans, demo, kapanış (3 slayt)

### S15. Hız ve canlı mod
| | Önce | Sonra |
|---|---|---|
| Saha noktası modeli | 108 ms | **47 ms** (TensorRT, D-036) |
| Dedektör | 80 ms | **63 ms** (TensorRT) |
| Çevrimdışı işleme | 6,8 kare/sn | **11,7 kare/sn** |
| Canlı gösterim | — | **~24 fps** (GPU işleri ayrı süreçte, D-031) |
| Canlı ızgara çizgi uyumu (zoom bölümü) | 0,77 | **0,90** (D-029) |
- Donanım: RTX 3050 Laptop, 4 GB

### S16. Demo
- `passes_overlay.mp4`'ten 20–30 sn (sağ panel: durum, mini harita, pas sayacı)
- İsteğe bağlı: canlı mod (`grid_demo.py live`)

### S17. Kısıtlar ve sonraki adımlar
- Kısıtlar (`13_LIMITATIONS.md`): havadaki topun saha konumu yanlış (yer düzlemi varsayımı); canlı modda kısa paslar kaçabiliyor (ilk 30 sn: çevrimdışı 11, canlı 5); hızlı zoom-out bölümleri zayıf; doğruluk henüz elle doğrulanmadı
- Sonraki adımlar:
  1. Elle etiketli paslarla doğruluk ölçümü
  2. SN-GSR üzerinde hücre doğruluğu (yer gerçeğiyle)
  3. Daha fazla videoyla değerlendirme

---

## Hazırlık listesi
- [ ] ⚠ roboflow veri seti görüntü sayısı / mAP ve NBJW makale künyesini doğrula (S3, S5)
- [ ] ⚠ Güncel tam video kalibrasyon raporunu üret (S9 tablosu güncel olsun)
- [ ] ⚠ pas_test pas listesi elle etiketlensin → S14 doğruluk satırı
- [ ] Görseller: mevcut `outputs/grid_demo/pas_test/*.jpg` içinden seç; eksikler için yeni ekran görüntüsü (ızgara önce/sonra, mini harita, pas anı)
- [ ] Demo video kesiti (20–30 sn)
- [ ] Sunum süresi ve slayt sayısı hocanın beklentisine göre kısaltılacak (S4, S11, S15 birleştirilebilir)
