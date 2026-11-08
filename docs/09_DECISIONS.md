# 09 — Karar Kaydı (Decision Log)

> Her önemli teknik karar buraya eklenir. Kararlar silinmez, değişirse yeni bir kayıtla "supersedes" belirtilir.

## Şablon
```
### D-XXX: <başlık>
- Tarih:
- Durum: önerildi | kabul | reddedildi | değiştirildi (D-YYY ile)
- Bağlam:
- Karar:
- Alternatifler:
- Sonuçlar / etkiler:
```

---

### D-001: Dokümantasyon yapısı
- Tarih: 2026-10-21
- Durum: kabul
- Karar: Tüm tasarım ve bağlam bilgisi `docs/` altında numaralı md dosyalarında tutulur. Kök dizindeki `CLAUDE.md` dosyası giriş noktasıdır. Orijinal şartname `00_SPEC_ORIGINAL.md` dosyasında değiştirilmeden saklanır.

### D-002: Gözlem ve karar katmanlarının ayrılması
- Tarih: 2026-10-21
- Durum: kabul (şartname §7 gereği)
- Karar: M1–M6 yalnızca gözlem üretir. Sınıflandırma yalnızca M7 kural motorunda, `configs/pass_rules.yaml` üzerinden yapılır.

### D-003: Ara temsil olarak diske yazılan `FrameState`
- Tarih: 2026-10-21
- Durum: önerildi
- Karar: Görüntü işleme (M1–M5) ve olay/kural analizi (M6–M8) `frame_states` dosyası üzerinden ayrılır. Böylece kural ayarlamaları videoyu yeniden işlemeden denenebilir.

### D-004: 50 hücre düzeni — 10×5 eşit ızgara
- Tarih: 2026-10-21
- Durum: **kabul (kullanıcı onayı, 2026-10-21)**
- Bağlam: Şartname 50 hücre şart koşuyor ama düzeni vermiyor.
- Karar: 105×68 m saha üzerinde 10 sütun × 5 satır, `cell_id = row*10 + col + 1` (numaralandırma D-038 ile değişti). Ayrıntılar `04_PITCH_GRID.md` dosyasında.
- Alternatifler: 5×10, ceza sahasına hizalı eşit olmayan sütunlar.

### D-005: Kural eşleşmezse varsayılan sonuç UNCERTAIN
- Tarih: 2026-10-21
- Durum: değiştirildi (D-006 ile)
- Not: D-006 kuralları (R-1…R-4) tüm durumları kapsadığı için "eşleşmeyen kural" durumu oluşmaz. Ayrıca D-008 ile UNCERTAIN sınıfı tamamen kaldırıldı.

### D-006: Pas sınıflandırma kuralları — hedef hücre doluluğu
- Tarih: 2026-10-21
- Durum: **kabul (kullanıcı tanımı)**
- Bağlam: Şartname hücre kurallarını tanımlamıyordu (§11).
- Karar: Pasın sonucu yalnızca **hedef hücredeki** oyunculara göre belirlenir:
  - Yalnızca takım arkadaşı varsa SUCCESS, yalnızca rakip varsa FAIL.
  - İkisi de varsa topa en yakın oyuncunun takımı belirler.
  - Hedef hücrede kimse yoksa FAIL.
  - **Ara hücreler değerlendirilmez**, yalnızca kaydedilir.
  - Hücum yönü ve devre arası taraf değişimi önemsizdir.
- Ayrıntılar: `05_PASS_RULES.md` §B.3
- Sonuç: Kalite açısından kritik olan; pasın bitiş anının, hedef hücrenin ve hedef hücredeki oyuncuların takımlarının doğru belirlenmesidir.

### D-007: Geliştirme ortamı
- Tarih: 2026-10-21
- Durum: açık, **repo incelemesinden sonra karar verilecek**
- Not: Makinede Python kurulu değil. Python, kütüphane ve CUDA sürümleri, seçilecek repoların gereksinimlerine göre belirlenecek.

### D-008: İkili sınıflandırma, belirsiz sınıfı yok
- Tarih: 2026-10-21
- Durum: **kabul (kullanıcı kararı)**, şartname §6.6'dan bilinçli sapma
- Karar: Yalnızca SUCCESS / FAIL kullanılır. Belirsiz durumlar sabit kurallarla çözülür:
  - R-3T: eşit mesafede SUCCESS
  - R-OUT: top saha dışına çıkarsa FAIL
  - F-1…F-5: veri eksikliği için yedek kurallar
- Bu kuralların hata olasılığı kabul edilir ve `13_LIMITATIONS.md` dosyasında belgelenir.
- Ek kararlar:
  - Pasör hedef hücredeyse kendisi sayılmaz (başka bir takım arkadaşı gerekir).
  - Yalnızca hedef hücre dikkate alınır, komşu hücreler sayılmaz.
- Çıktı: Takım A ve B için yalnızca başarılı ve başarısız sayıları.

### D-009: Her vuruş pastır
- Tarih: 2026-10-21
- Durum: **kabul (kullanıcı kararı)**, şartname §5'ten bilinçli sapma
- Karar: Şut, orta ve uzaklaştırma ayrımı yapılmaz. Aynı oyuncunun topu tekrar alması dribling sayılır, pas değildir.
- Pasın bitişi: Top başka bir oyuncuya ulaştığında, durduğunda veya saha dışına çıktığında, hangisi önce olursa.

### D-011: Model bağımsız mimari (arayüz + adaptör + registry)
- Tarih: 2026-10-21
- Durum: **kabul (kullanıcı talebi)**
- Bağlam: Başlangıçta repolardaki hazır modeller kullanılacak. İleride kendi eğittiğimiz veya farklı bir hazır modele geçmek isteyebiliriz.
- Karar: Tüm öğrenilmiş ve değiştirilebilir bileşenler (nesne tespiti, saha keypoint tespiti, takip, takım ataması, kamera hareketi) soyut arayüz arkasında, adaptörlerle uygulanır. Hangisinin kullanılacağı konfigden seçilir. Model çıktıları `class_map` / `keypoint_map` ile kanonik şemaya çevrilir. Kütüphane importları yalnızca adaptör dosyalarında yapılır. Her adaptör sözleşme testlerinden geçmelidir.
- Sonuç: Model değişikliği yeni bir adaptör ve konfig değişikliğiyle yapılır, pipeline kodu değişmez. Ayrıntı: `03_ARCHITECTURE.md` §1b.

### D-012: F-3 kamera hareketi ile homografi taşıma
- Tarih: 2026-10-21
- Durum: **kabul (kullanıcı talebi)**
- Karar: Keypoint yetersiz karelerde homografi, önceki/sonraki geçerli karelerden kareler arası kamera hareketi kestirimiyle (`H_t = H_{t-1}·M_{t→t-1}`) taşınır. Keypoint görünür olunca yeniden sabitlenir. Hareket kestirilemezse en yakın geçerli homografi kullanılır.

### D-013: Faz 0 — 50 hücreli ızgara demosu, NBJW + kendi homografimiz
- Tarih: 2026-10-22
- Durum: **kabul (kullanıcı talebi)**, uygulandı
- Karar:
  - İlk hedef yalnızca sahayı 50 hücreye bölüp videoda göstermek. Top ve oyuncu tespiti bu aşamada yok.
  - Saha noktaları NBJW (`SV_kp`, `SV_lines`) ile bulunur. NBJW'nin kendi kamera kalibrasyonu **kullanılmaz**: adaptör yalnızca zemin düzlemindeki noktaları (görüntü ↔ saha) döndürür. Homografiyi `sgi.mapping.homography` RANSAC ile hesaplar (D-011 uyumlu).
  - F-3 için Lucas-Kanade optik akış + RANSAC homografi kullanılır.
- Ortam: Python 3.12 `.venv`, torch 2.14.1+cu126, RTX 3050 Laptop (4 GB). Video yazımı mp4v (opencv-python'da H.264 yok).
- İlk ölçüm (`iniesta_sample.mp4`, 1080p, 153 kare):
  - Saha noktalarıyla kalibrasyon %100, ortalama reprojeksiyon hatası 0,1–0,25 m, işleme hızı 3,5 kare/sn.
  - F-3 testi (keypoint 5 karede bir): taşınan karelerde, her karede keypoint kullanan sonuca göre fark medyan 0,23 m, en fazla 0,51 m.
- Sınır: İşleme gerçek zamanlı değil. Gösterim, önbellekli oynatıcı (`play`) veya üretilen MP4 ile yapılır.

### D-014: Kalibrasyon doğruluk ölçümü ve çim tabanlı iyileştirme planı
- Tarih: 2026-10-22
- Durum: **kabul (plan)**, uygulanmadı
- Karar:
  - Hücre ataması doğruluğu SN-GSR üzerinde **hücre doğruluğu (%)** ve **konum hatası (m)** ile ölçülür.
  - Etiketsiz kliplerde öz-tutarlılık göstergeleri ve hata kategorileri (E1–E7) üretilir.
  - Çim bilgisi üç şekilde kullanılır:
    1. Çim maskesi
    2. Kamera hareketi kestirimine destek
    3. Şerit haritası ile F-3 kayma düzeltmesi (`PitchRefiner` arayüzü)
- Sıra ve ayrıntılar: `14_CALIBRATION_QUALITY.md` §5.

### D-015: İşleme hızı optimizasyonları
- Tarih: 2026-10-22
- Durum: **kabul (kullanıcı onayı)**, uygulandı
- Profil (720p, kare başına 262 ms):

  | Adım | Süre (ms) |
  |---|---|
  | HRNet keypoint modeli | 100 |
  | HRNet çizgi modeli | 99 |
  | Isı haritası tepe bulma | 45 |
  | Python/scipy döngüleri | ~1 |
  | Optik akış | 7,5 |
  | Çizim + yazma | ~6 (1080p'de ~23) |

  Sonuç: Darboğaz model ve tepe bulmadaki büyük max-pool. Python döngüleri önemsiz.
- Uygulananlar:
  1. **Ayrılabilir max-pool** ile tepe bulma: 35 → 5 ms, sonuç birebir aynı (`tests/test_nbjw_peaks.py`).
  2. **Arka plan okuma ve yazma** (`sgi.io.video.FrameReader/AsyncWriter`): MP4 üretiminin maliyeti neredeyse sıfıra indi.
  3. **`--no-video`** seçeneği.
- Denenip reddedilenler:
  - cuDNN benchmark ve toplu çıkarım: kazanç yok.
  - `channels_last`: model başına %11 hız, ama fp16 sonuçlarını değiştiriyor (H farkı medyan 0,13 m). Tekrarlanabilirlik için `channels_last: false` (konfigde isteğe bağlı).
- Sonuç: 3,7 → **4,5 kare/sn**, çıktı birebir aynı (regresyon farkı 0,0000 m). Daha fazlası için model düzeyinde değişiklik (TensorRT) veya `detect_every` gerekir.

### D-016: Canlı mod (`grid_demo.py live`)
- Tarih: 2026-10-22
- Durum: **kabul (kullanıcı talebi)**, uygulandı
- Karar:
  - Model ayrı iş parçacığında en son kareyi işler (`sgi.mapping.live.AsyncCalibrator`).
  - Ana döngü her karede kamera hareketini hesaplar.
  - Geç gelen ölçüm kayıtlı hareketlerle güncel kareye taşınır (`MotionHistory.bring_forward`). Arada sahne kesmesi varsa ölçüm atılır.
- Ölçüm (720p, 60 sn, `--headless-seconds`):
  - Gösterim 24,9 fps (hedef 25), çizim 10–12 ms.
  - Çevrimdışı (her kare model) sonuca göre fark medyan 0,65 m, %95 2,8 m.
  - Kapsama %32 (çevrimdışı %35).
- Kullanım: gösterim için canlı mod, ölçüm ve rapor için `process`.

### D-017: Kalibrasyon doğrulaması — fiziksel kamera kontrolü + en az 8 inlier
- Tarih: 2026-10-22
- Durum: **kabul**, uygulandı
- Bağlam: Canlı mod testinde, 4–7 saha noktasına dayanan homografilerin düşük reprojeksiyon hatasıyla "geçerli" sayıldığı ama tamamen yanlış olduğu görüldü (ızgara aynalanmış veya kaymış, 140 m'ye varan sapma). Az nokta + 8 serbestlik derecesi → az fazlalık → yanlış çözüm de tutarlı görünür. Bu, klibin ilk tam işlemesini de etkilemişti.
- Karar:
  1. `camera_plausibility`: Homografi K[r1 r2 t] olarak ayrıştırılır. Odak uzaklığı iki koşuldan (r1 ⟂ r2, |r1| = |r2|) ayrı ayrı çözülür, iyi olan seçilir. Sonra diklik, en-boy, odak aralığı ve kamera yüksekliği sınanır. Tek denklem kullanan ilk sürüm, karşıdan bakan geçerli kameraları reddediyordu; birim testi yakaladı ve düzeltildi.
  2. `min_inliers: 8`: Doğru ölçümlerin hepsi ≥ 8 inlier'lı.
- Bilinen kör nokta: Tam karşıdan (pan≈0) bakışta x yönündeki gerilme, farklı zoom/mesafeyle açıklanabilir ve kontrol bunu yakalayamaz.
- Sonuç (60 sn bölüm):
  - Yanlış ölçümler elendi. Rastgele örneklerde ızgara doğru (kaleler 21/30'da, ceza yayları doğru hücrede).
  - Kapsama %64,6 → %35,1. Düşen kısım büyük oranda yanlış kalibrasyonlardı.

### D-018: Hızlı deneme döngüsü — saha noktası önbelleği + 2 dk test videosu
- Tarih: 2026-10-22
- Durum: **kabul (kullanıcı talebi)**, uygulandı
- Bağlam: 8 dk'lık klibin her işlenişi ~45 dk sürüyordu. Sürenin ~%90'ı saha noktası modeli, ama sonraki değişiklikler model çıktısını etkilemiyor.
- Karar:
  1. `sgi.detection.keypoints.cache.KeypointCache`: Model her video için bir kez çalışır, noktalar `outputs/cache/keypoints/` altına yazılır. Anahtar = video kimliği + dedektör konfigürasyonu (model/eşik değişirse otomatik yeni önbellek). `--no-cache` ile atlanır. Önbellek doluysa model hiç yüklenmez.
  2. Test videosu `data/videos/test_2min.mp4`: kaynak klipten 5 bölüm (yakın çekim/VAR, geniş orta saha, sol kale, sağ kale, ana kamera), 3000 kare. Bölüm listesi `test_2min.txt`.
- Ölçüm: örnek videoda 36 sn → 4,2 sn (~8,6x), sonuç birebir aynı. 2 dk test videosu: ilk ~11 dk, sonrası ~30–40 sn.
- Kural: Geliştirme 2 dk test videosuyla yapılır. Tam klip yalnızca sonunda işlenir.
- **Güncelleme (2026-10-22, kullanıcı kararı):**
  - Kesilip birleştirilmiş test videosu (çoğunlukla şut/gol anları) bırakıldı. Test videosu artık kullanıcının eklediği `data/videos/pas_test.mp4` (64,6 sn, 1614 kare, daha çok pas içeriyor).
  - Önbellek **varsayılan olarak kapalı** (`--use-cache` ile açılır). Her şey doğrulandıktan sonra önbellekli ve önbelleksiz çıktıların birebir aynı olduğu, veri kaybı olmadığı ayrıca test edilecek.
  - `outputs/` temizlendi (referans sonuç dahil, yeniden üretilecek).

### D-019: Kalite göstergeleri ve hata kategorileri (etiketsiz video)
- Tarih: 2026-10-22
- Durum: **kabul**, uygulandı (`sgi.mapping.quality`, `sgi.mapping.diagnostics`, `scripts/calib_report.py`)
- Göstergeler (kare başına, `calibration.npz` içinde):
  - Yeşil oranı: videoya uyarlanan HSV çim maskesi
  - **Çizgi hizalama skoru:** Saha çizgileri H ile izdüşürülür. Örneklerin gerçek beyaz çizgiye ≤ 3 px yakınlık oranı ölçülür. Beyaz çizgi = parlaklık − 15 px medyan > 8, doygunluk < 120, çim üzerinde.
  - Saha/çim IoU
  - Reddetme kodu: few_keypoints, ransac, few_inliers, reproj, spread, cam_*, jump
  - Hareket kestirildi mi
- Ayar doğrulaması (örnek video):
  - Doğru ızgarada skor 0,87. Izgara 1 m kaydırılınca ~0,2, 2–8 m'de ≤ 0,25.
  - İlk sürüm (640 px, top-hat > 25, doygunluk < 90) çizgileri değil formaları yakalıyordu (doğru karede 0,33). Ayrıca tribün de çizgi sanılıyordu. Değerlendirme yalnızca çim üzerindeki örneklerle yapılacak şekilde düzeltildi.
- Kategoriler: OK / OK? / E1–E7 ve kalibre ama çizgilerle uyumsuz kareler için E4!, E5! (14 §3). Eşik `min_line_score = 0.5`.
- Rapor: `calib_report.py` → kategori dağılımı, şüpheli bölümler, `calibration_timeline.png`.
- Yan bulgu: homografi tersinin işareti belirsizdi (`H/H[2,2]` sonrası). Kameranın önündeki noktalar "arkada" sayılabiliyordu. `pitch_to_image` ile yön sabitlendi (çizim + kalite).

### D-020: Çizgilere oturtma (homografi ince ayarı, `PitchRefiner`)
- Tarih: 2026-10-22
- Durum: **kabul (kullanıcı onayı)**, uygulandı, gerçek videoda ölçüm bekliyor
- Bağlam: `pas_test` raporunda kamera hareketiyle taşınan karelerde 1–5 m kayma görüldü (E5!, OK~). Orta saha karelerinde saha noktası yetersizdi.
- Karar: `sgi.mapping.refine.line_fit.LineFitRefiner` (chamfer eşleme). Yöntem:
  1. Çim maskesi içinde beyaz çizgi tespit edilir ve uzaklık dönüşümü alınır.
  2. Saha modeli çizgileri izdüşürülür.
  3. 4 kontrol noktasıyla tanımlı 8 parametreli görüntü düzeltmesi, soft-L1 kayıp + sıfıra düzenlileştirme ile `least_squares`'te optimize edilir.
  4. Kabul koşulları: çizgi uyumu ≥ 0,03 artmalı, düzeltme ≤ 4 m (paralel çizgiler 5,5 m arayla, yanlış çizgiye oturmayı önler), fiziksel kamera kontrolü geçilmeli.
  5. Kabul edilirse takip düzeltilmiş H'den devam eder ve taşıma zinciri sıfırlanır.
- Ayar: `pitch_refiner` (konfig). `process` her karede, `live` 5 karede bir (~40 ms/çağrı).
- Uygulama notları:
  - OpenCV `getPerspectiveTransform` (32 bit) ve `remap` (1/32 px) sayısal türevleri yutuyordu. 64 bit DLT ve numpy bilineer örnekleme ile çözüldü.
- Sentetik test (`tests/test_line_fit.py`):
  - 1–2 m hata 0,2–0,4 m'ye, 3 m hata 0,5–1,0 m'ye iniyor.
  - ≥ 3,6 m reddediliyor.
  - Doğru ızgara bozulmuyor.
- Sınır: Başlangıç homografisi olmayan kareleri (ör. pas_test'in ilk 5 sn'si) çözmez. Bunun için ayrı bir başlatma yöntemi gerekir.

### D-021: Mini harita — tespit edilen çizgiler ve hücre durumu
- Tarih: 2026-10-22
- Durum: **kabul (kullanıcı talebi)**, uygulandı (`GridOverlay`)
- Karar:
  - Videoda tespit edilen beyaz çizgiler, homografiyle kuş bakışına çevrilip mini haritaya **sarı** çizilir. Referans olarak sahanın gerçek çizgileri silik çizilir. İkisinin örtüşmesi, kalibrasyonun gözle doğrulanmasını sağlar. Yalnızca uzun parçalar (≥ 25 px @640) tutulur, forma lekeleri elenir. `c` tuşuyla açılıp kapanır.
  - Hücre durumu (hücre başına 4×4 örnek nokta):
    - görünür (normal)
    - **görüş dışı** (hesaplanabilir ama kadrajda değil): koyu gri
    - **hesaplanamıyor** (kalibrasyon yok veya hücre kameranın görüş geometrisi dışında): **kırmızı**
  - Kamera görüş alanı sadece sınır çizgisiyle gösterilir.
- Maliyet: çizim ~25 ms/kare (çizgi tespiti dahil).
- Kapsam kararı (kullanıcı): 50 hücreye bölme şimdilik yeterli. İyileştirme fikirleri `10_ROADMAP.md`'de not edildi.

### D-022: Oyuncu / hakem / kaleci / top tespiti, takip ve takım ataması (Adım 1: eğitimsiz)
- Tarih: 2026-10-22
- Durum: **kabul (kullanıcı talebi)**, uygulandı
- Karar (her biri D-011 adaptörü, `configs/grid_demo.yaml` → `objects`):
  - **Tespit:** `UltralyticsYoloDetector`, COCO YOLO11l (`person` + `sports ball`), imgsz 1280, kişi conf 0,30, top 0,15 (~43 ms). sn-gamestate baseline'ı da COCO YOLO11 kullanıyor. Futbola özel model (SN-GSR ince ayarı) aynı adaptörle `weights`/`class_map` değiştirilerek takılır.
  - **Takip:** `ByteTrackTracker` (Ultralytics BYTETracker; supervision 0.31 ByteTrack'i kaldırdı).
  - **Takım/rol:** `ColorKMeansTeamAssigner`
    - Gövde bölgesi, çim pikselleri hariç Lab ortalaması, K-means(2).
    - İki merkeze de uzak kalanlar "diğer". Zamanının ≥ %60'ını bir ceza sahasında geçiren "diğer" kaleci, aksi halde hakem.
    - Kalecinin takımı: o kaleyi savunan takım (oyuncularının ortalama x'i kaleye yakın olan).
    - Karar takip kimliği bazında çoğunluk oyuyla verilir. Takım A = daha koyu forma (Lab L).
  - **Sahaya yerleştirme:** Ayak noktası (bbox alt-orta, top için merkez) H ile sahaya çevrilir, hücresi atanır. Sahanın 3 m dışındaki kişiler atılır (seyirci, kulübe). Top: sahadaki adaylardan son konuma yakın olan.
  - **Çıktı:** `objects.npz` (kare, track_id, rol, takım, bbox, conf, saha x/y, hücre) + takım renkleri. `play` bunu okur.
  - **Görselleştirme:**
    - Videoda ayak altında takım renginde elips + etiket (A/B/KA/KB/H), topta halka
    - Mini haritada takım renginde nokta, hakem mavi, top siyah, kalecide beyaz halka
    - `o` tuşu ile aç/kapa
  - **Canlı mod:**
    - YOLO ayrı iş parçacığında (`AsyncWorker`), saha noktası modeli en fazla 10 karede bir (GPU paylaşımı).
    - Mini haritadaki tespit edilen çizgiler varsayılan kapalı, çizgilere oturtma 10 karede bir.
    - Sonuç 6 → **23 fps**. Oyuncu işaretleri birkaç kare geriden gelir.
- İlk ölçüm (pas_test ilk 24 sn):
  - Barcelona = A, Real = B, sarı formalı hakem ve yan hakem doğru (hakem).
  - Top karelerin %69'unda bulundu. Nesne hattı tek başına 18 kare/sn.
- Bilinen sınırlar:
  - COCO modeli rol ayırmaz (renk sezgiseli)
  - Top küçük/bulanıkken kaçırılır
  - Havadaki topun saha konumu yanlış
  - Hakem rengi bir takıma benzerse karışır
  - Yakın çekime geçişte (pas_test ~24 sn) yanlış ızgara/konum görülebilir
- Adım 2 (gerekirse): SN-GSR (rol + top etiketli) ile YOLO ince ayarı.
- **Tam test (pas_test, 1614 kare) ve düzeltmeler:**
  1. Kalecinin takımı ortada A↔B dönüyordu (ortalama takım konumu sezgisi). Düzeltme: "kaleye en yakın saha oyuncusu o kaleyi savunan takımdandır" (ofsayt kuralı), video boyunca birikimli. Sonuç: sol kale kalecisi baştan sona B (Real).
  2. Üst üste binen iki oyuncunun kutusu karışık renkten "hakem" oluyordu. Düzeltme: genişlik/yükseklik > 0,6 kutulardan renk örneği alınmaz. "Diğer" kararı için oyların çoğunluğu + en az 3 oy gerekir.
  3. Takip kimliği değişince eski takım etiketi taşınıyordu. Düzeltme: oy penceresi 60 → 20.
  - Örnek karelerde Barcelona = A, Real = B, orta hakem + yan hakem = H, kaleci = KB doğru. Top karelerin %73'ünde.
  - İşleme: tam hat 3,3 kare/sn. `objects` komutu (kalibrasyonu yeniden kullanır) 1614 kare ~90 sn.
  - Kalan: 178 takip kimliği (~22 kişi için; kimlik değişimleri sık), "takımı belirsiz" (?) işaretler, havadaki top konumu.

### D-023: Canlı modda GPU paylaşımı — tek kuyruk, ısınma, nesnelerin kamera hareketiyle taşınması
- Tarih: 2026-10-22
- Durum: **kabul**, uygulandı
- Bağlam: Kullanıcı canlı modda ızgaranın uzun süre duraksadığını, oyuncu ve top tespitinin kötü göründüğünü ve ikisinin birbirini engellediğini bildirdi. Ölçüm (penceresiz, 30 sn):
  - İki model ayrı iş parçacıklarında GPU'yu aynı anda kullanıyordu. Saha noktası modeli 214 → ~290 ms yavaşladı.
  - İlk YOLO çağrısı (CUDA ısınması) 5,4 sn sürdü. Bu sürede ne oyuncu işareti ne ilk ızgara geldi.
  - Oyuncu işaretleri ~5 kare geriden geliyordu. Kamera dönerken oyuncuların yanına düşüyordu.
- Karar:
  1. `GpuScheduler`: Tek GPU iş parçacığı. Saha noktası modeli en az `calib_every` (10) karede bir öncelikli çalışır, arada YOLO çalışır. Hatalar yakalanıp sayılır (iş parçacığı sessizce ölmez).
  2. Isınma: Video başlamadan önce iki model de birer kez çalıştırılır.
  3. Geç gelen oyuncu kutuları, aradaki kamera hareketi kadar kaydırılır (`MotionHistory.image_transform`). Mini haritadaki saha konumları zaten mutlak olduğu için etkilenmez.
- Sonuç (aynı 30 sn):
  - Izgara görünen kare %84,5 → %86,1–86,9 (çevrimdışı %87,3)
  - Saha noktası modeli ~290 → ~225 ms
  - Nesne gecikmesi ort. 4,7 kare (telafili). 24,9 fps.
- Not: Çevrimdışı `process` / `objects` çıktıları bu sorundan etkilenmez; adımlar orada sırayla çalışır.

### D-024: Hazır modellerle iyileştirme — BoT-SORT, NBJW TSWC ağırlıkları, futbol dedektörü
- Tarih: 2026-10-22
- Durum: **kabul (kullanıcı talebi: "hazır modeller kullanılamıyor mu")**, uygulandı. Analiz: `15_IMPROVEMENT_ANALYSIS.md`
- Uygulananlar ve ölçümler (pas_test):
  1. **Takip: ByteTrack → BoT-SORT** (Ultralytics, kamera hareketi telafisi GMC). COCO ile: kimlik 178 → 168, "?" %4,3 → %2,6, yanlış kaleci A 24 → 0.
  2. **NBJW ağırlıkları SV → SV_FT_TSWC** (TS-WorldCup yayın verisiyle ince ayarlı). İlk 20 sn:
     - Kalibre %75,8 → %76,2
     - Çizgi uyumu en kötü %10: 0,92 → 0,96
     - E4 %0,6 → 0
     - Orta saha (E2) değişmedi
  3. **Dedektör: COCO YOLO11l → `roboflow/sports` futbol modeli** (MIT, YOLOv8x; sınıflar ball/goalkeeper/player/referee; Google Drive, gdown):
     - Top bulunan kare %73 → **%80**. Kaleci modelden geliyor.
     - Model sarı formalı hakemi zaman zaman "oyuncu" sanıyor. Düzeltme: model "oyuncu" dese de forma rengi iki takıma uymuyorsa hakem (`is_other`). Takım kümelemesine yalnızca oyuncu kutuları katılır.
     - conf 0,40 + `new_track_thresh` 0,5: titreşen kısa takipler azaltıldı (kimliklerin ~yarısı < 10 kare, kayıtların %1,7'si).
     - Süre 43 → 87 ms/kare. `objects` ~160 sn/1614 kare.
     - Canlı: 24,7 fps, ızgara %82,9, nesne gecikmesi ort. 7,4 kare (kamera hareketiyle telafili).
- Değerlendirilip ertelenenler:
  - PnLCalib: ağırlıklar NBJW ile aynı isim/boyutta. Fark kalibrasyon hesabında, onu D-020 ile kendimiz yapıyoruz.
  - roboflow top modeli: bu videoda futbol modelinden zayıf.
  - roboflow saha noktası modeli: indirildi (`models/roboflow_sports/football-pitch-detection.pt`), henüz bağlanmadı. Orta saha için ikinci kaynak adayı.
- Kalan: orta saha başlangıç kalibrasyonu (ilk 5 sn), top izi/interpolasyonu, saha koordinatında takip.

### D-025: roboflow/sports saha noktası modeli — denendi, kullanılmıyor
- Tarih: 2026-10-22
- Durum: **reddedildi** (bu video için). Adaptör kodda duruyor, varsayılan kapalı.
- Ne yapıldı:
  - `UltralyticsPitchKeypointDetector` adaptörü (YOLOv8-pose, 32 nokta). roboflow 120×70 m varsaydığı için noktalar anlamlarıyla eşlenip bizim 105×68 m ölçülerimizle hesaplandı.
  - NBJW ile birleştirmek için `CompositePitchKeypointDetector` yazıldı.
- Ölçüm (pas_test, 7 kare):
  - Eksen yönü doğru (y çevrilince fark 20–75 m'ye çıkıyor).
  - Ama noktalar, NBJW'nin doğrulanmış kalibrasyonundan 50–300 px uzakta. Homografi farkı 6–30 m. Çoğu kare fiziksel kontrolden geçmiyor.
  - Görsel kontrol: NBJW noktaları gerçek saha özelliklerinin üstünde. roboflow noktaları değil (ör. orta yuvarlak kesişimleri ve ceza sahası köşesi kaymış). Görünmeyen noktaları da (sağ ceza sahası, uzak köşe) yüksek güvenle veriyor.
- Yorum: Model büyük olasılıkla farklı yayın kamera açılarıyla (DFL Bundesliga) eğitilmiş ve bu videoya genellenmiyor.
- Sonuç: Orta saha başlangıç kalibrasyonu için bu yol işe yaramadı. Alternatif: orta yuvarlak elips uydurması (klasik, 15 §3).

### D-026: Kalibrasyon yönlendirmeli klasik kale tespiti
- Tarih: 2026-10-22
- Durum: **kabul (kullanıcı talebi)**, uygulandı (`sgi.mapping.goals.GoalDetector`). Model eğitimi yok.
- Ön ölçüm: NBJW direk diplerini iyi buluyor (medyan 3,9 px), direk tepelerini bulamıyor (tepe+dip birlikte karelerin %14'ünde, tepeler bazen skor tabelasında).
- Yöntem:
  1. Direk dipleri H ile izdüşürülür.
  2. Her dipten yukarı dar şeritte beyaz maske (V > 170, S < 60) çıkarılır, en uzun kesintisiz dikey parça direktir.
  3. Beklenen boy, dipteki yerel ölçekten hesaplanır (1 m'nin en uzun izdüşümü × 2,44). Kale genişliği perspektifte çok kısaldığı için ilk denemede ondan alınan ölçek direkleri yanlışlıkla reddediyordu.
  4. Üst direk, iki tepe arasındaki beyaz oranıyla (≥ 0,5) doğrulanır.
- Sonuç (pas_test):
  - Kale kadrajdayken karelerin **%82'sinde** iki direk + üst direk bulundu.
  - Çizilen çerçeve gerçek kalenin üstünde.
  - Kalenin görünmediği karelerde yanlış tespit yok.
  - Direk boyu / zemin ölçeği ≈ 1,37 (kamera eğimi bilgisi, 3B kamera modeli için girdi).
- Kullanım:
  - Videoda magenta kale çerçevesi, mini haritada bulunan kale işareti (`y` tuşu)
  - `calibration.npz`'de `goal_visible/goal_found`, raporda "kale bulundu %"
- Yan düzeltme: Futbol modeli kaleciyi "oyuncu" sınıflandırınca renk yedeği onu hakem yapıyordu. Artık ceza sahasında kalıyorsa kaleci (pas_test: kaleci B 760, A 0).
- Sonraki olası adımlar: direk tepeleriyle tam 3B kamera modeli (top yüksekliği), gol çizgisi olayları.

### D-027: Sabit kamera konumu önseli (kullanıcı fikri: standart saha ölçüleri + kameranın yeri)
- Tarih: 2026-10-22
- Durum: **kabul (kullanıcı onayı)**, uygulandı (`sgi.mapping.camera_prior`)
- Fikir: Ana yayın kamerası tribünde sabit bir yerde durur, yalnızca döner ve zoom yapar. Sabit görüntülü video gerektirmez; kameranın 3B konumu sistem tarafından videodan otomatik öğrenilir.
  - Serbest homografi: 8 bilinmeyen (pratikte 8 sağlam nokta).
  - Kamera konumu C biliniyorsa: 4 bilinmeyen (dönüş 3 + odak 1), 2–3 nokta yeter.
- Yöntem:
  1. Güvenilir (≥ 12 inlier, fiziksel kontrolden geçmiş) karelerin homografisi K[r1 r2 t]'ye ayrıştırılır, C = −Rᵀt. Ortanca alınır.
  2. Çevrimdışı modda video başlamadan tüm videodan seyrek ön tarama yapılır (25 karede bir, ~18 sn). Canlı modda çevrimiçi öğrenilir.
  3. Normal yöntem nokta azlığından/fiziksel kontrolden reddettiğinde, C sabitken dönüş+odak soft-L1 en küçük kareler ile çözülür (çok başlangıçlı). En az 3 inlier, rms ≤ 4 px ve fiziksel kontrol gerekir.
  4. Kestirimlerin yayılımı > 6 m ise (ör. farklı kameralar karışıyor) öncel kapalı kalır.
- Ölçümler (pas_test):
  - **Tutarlılık:** 606 iyi kareden C = (≈51, ≈117, yükseklik ≈24 m), yani orta saha hizası. Karelere göre sapma medyan 3,8 m. Odak 2400–4700 px (zoom).
  - **Kontrollü test:** Kalibrasyonu bilinen karelerde yalnızca 3 nokta ile çözüm, tam kalibrasyondan medyan **0,35 m** (%90: 0,64 m) farklı.
  - **İlk 20 sn:** Kalibre %76,2 → **%100**, OK %76 → **%98**, ilk 5 sn (orta saha) ızgara %0 → **%100**. 40 kare doğrudan öncelle çözüldü, aradaki kareleri kamera hareketiyle taşıma + çizgilere oturtma doldurdu.
  - **Görsel:** Orta yuvarlak, orta çizgi, taç çizgisi ve ceza sahası köşesi izdüşümleri gerçek çizgilerin üstünde.
  - **Tam video (1614 kare, nesneler + kale dahil, 633 sn):**

    | | D-024 sonrası | D-027 sonrası |
    |---|---|---|
    | Kalibre | %87,3 | **%94,8** |
    | OK | %81,7 | **%93,8** |
    | Yaklaşık (OK~) | %3,7 | %0,6 |
    | Kaymış (E4!/E5!) | %1,6 | %0 |
    | Yetersiz nokta (E2) | %11,2 | %3,9 |

    181 kare doğrudan öncelle çözüldü. Kale kadrajdayken %78 bulundu. Oyuncuların %100'ünün saha konumu var.
- Sınırlar:
  - İdeal pinhole varsayımı (kare piksel, asal nokta merkezde): tek karelik dönüşüm farkı medyan 1,24 m. Çizgilere oturtma bunu düzeltir.
  - Başka kameralara geçişte (tekrar, kale arkası) öncel o kamera için geçersizdir. Fiziksel kontrol + takip tutarlılığı (sıçrama reddi) ile korunur. Çok kameralı öncel (kamera başına C) ileride eklenebilir.

### D-028: Yalnızca geçmiş kareler (nedensel işleme) + top takibi
- Tarih: 2026-10-22
- Durum: kabul (kullanıcı kararı)
- Gerekçe (kullanıcı): Sistemin canlı performansı önemlidir. İleriki karelere bakmak (çevrimdışı iki yönlü interpolasyon, tüm videoyu önceden tarama) ölçülen başarıyı canlıda elde edilemeyecek şekilde yükseltir ve yanıltır. **Çevrimdışı işleme de canlıdaki gibi yalnızca o ana kadarki kareleri kullanır.**
- Uygulama:
  1. Kamera konumu ön taraması (D-027 adım 2) kaldırıldı. Kamera konumu her iki modda da işlenirken öğrenilir.
  2. "Çevrimdışı iki yönlü F-3 taşıması" ve "iki yönlü top interpolasyonu" yol haritasından çıkarıldı.
  3. Top takibi (`src/sgi/tracking/ball.py`, saha koordinatında, nedensel):
     - **Kapı (gating):** Yeni tespit, tahmin edilen konuma `2 m + 35 m/sn × geçen süre` içinde olmalı. Dışındaki adaylar yok sayılır. Aynı yerde art arda 3 kez çıkarsa takip oraya geçer.
     - **Oyuncuda (held):** Top kaybolduğunda son görüldüğü anda bir oyuncuya ≤ 1,5 m ise, en fazla 3 sn o oyuncunun ayağında sayılır.
     - **Tahmin (predicted):** Sahipsiz kaybolan top, son hızıyla en fazla 0,5 sn ilerletilir.
     - **Saha işaretleri:** Penaltı noktaları ve orta noktadaki aday, ancak dibinde (≤ 1 m) oyuncu varsa ve top oraya ≤ 3 m yakından hareketle geldiyse kabul edilir.
     - Kestirilen top videoda turuncu halka ve "top?" / "top (oyuncuda)" yazısıyla, mini haritada içi boş daire ile gösterilir. `objects.npz` 13. sütun `state`.
- Ölçüm (pas_test 28–48 sn, penaltı noktası görünen bölüm):
  - Eski tam çalıştırmada top 1614 karenin 533'ünde aslında sol penaltı noktasıydı. Bu bölümde 302/488 → **0**.
  - Top durumu: tespit %73, oyuncuda %21, tahmin %4 (toplam %98 karede top konumu var).
  - Kalan hata: Dedektör bazen beyaz kramponu yüksek güvenle (0,5) top sanıyor. Takip 3 kare sonra gerçek topa döner (son 1 sn'de topu tutan oyuncunun dibindeki aday kapıdan bağımsız kabul edilir; düşük güvenli (< 0,3) aday 4 m'den uzağa sıçratamaz).
  - Ayarlar konum hatası için pay bırakır: işaret yarıçapı 1,5 m (kalibrasyon kayması ~1 m).
- Oyuncu yönü fizibilitesi (kullanıcı fikri): YOLO11m-pose, oyuncu kesitinde (256 px) ~22 ms. Omuz sırası + yüz noktalarıyla ön/arka: net önden ve arkadan görünüşlerde doğru, yandan görünüşlerde kararsız. Henüz bağlanmadı.
- Sonuç (ilk 20 sn): Ön tarama kaldırılınca kalibre %100 → %76,2 (D-027 öncesi değer). Videonun başında kamera konumu henüz öğrenilmemiştir. Bu, canlıdaki gerçek durumdur.

### D-029: Canlı modda zoom/hızlı hareket sonrası ızgara doğruluğu
- Tarih: 2026-10-22
- Durum: kabul
- Sorun (kullanıcı): Canlı modda zoom'da ızgara kayboluyor/kayıyor, zoom çıkışında toparlanma gecikiyor.
- Teşhis (ölçümle):
  1. Çevrimdışı modda sorun yok. Saha ölçümü her karede gelir.
  2. Canlıda saha ölçümü ~12 karede bir, ~8 kare gecikmeyle gelir. Ölçüm kendi karesinde iyi (çizgi uyumu 0,86–0,99), ama kamera hareketiyle güncel kareye taşınınca 0,2–0,7'ye düşer. Optik akış 10 karede ~0,5 m (hızlı hareket/zoom'da 1–3,7 m) kayar. Ölçüm üstüne bir sonraki ölçüme kadar ~20 kare taşıma, ~2 m hata demektir.
  3. Ölçüm, her kare ölçüm gelen çevrimdışı mod için ayarlanmış sabit %35 ağırlıkla karıştırılıyordu.
  4. Çizgilere oturtma canlıda ana döngüde 10 karede bir yapılıyordu.
  5. Arka planda **iş parçacığı** ile sürekli oturtma, Python kilidi (GIL) yüzünden gösterimi 25 → 12 fps'e düşürdü.
- Çözüm:
  1. **Karışım ağırlığı aralığa göre:** `1 − 0,65^n` (n = son kanıttan beri kare). Çevrimdışında n = 1, davranış aynı.
  2. **Zoom farkındalığı:** Son kanıttan beri kamera zoom'u > %10 ise, fiziksel kontrolden geçmiş ölçüm "sıçrama" diye reddedilmez.
  3. **Çizgilere oturtma ayrı süreçte** (`ProcessWorker`), boşaldıkça en güncel karede çalışır. Sonuç 1–2 kare taşınıp `HomographyTracker.correct` ile uygulanır. Oturtma taze kanıt sayılır, ardından gelen gecikmeli ölçüm onu ezmez.
  4. **Canlıya özel hızlı oturtma:** Yarım çözünürlükte arka plan medyanı + en fazla 25 optimizasyon adımı. 28 → 22 ms, doğruluk aynı.
  5. **Izgara kayıpken saha modeli öne alınır:** 10 yerine 2 karede bir.
  6. **Tekrarlanabilir canlı simülasyonu** (`scripts/live_sim.py`). Gerçek canlı testin zamanlaması makinenin yüküne/ısınmasına göre değişiyor (aynı ayarla 18–24 fps). Simülasyon zamanlamayı sabit parametrelerle taklit eder ve varyantları karşılaştırır.
- Ölçüm (pas_test 20–38 sn, yakın çekim + zoom + hızlı hareket):

  | | Çizgi uyumu medyan | Kötü kare (< 0,65) |
  |---|---|---|
  | Önce (canlı) | 0,77 | %35 |
  | Sonra (canlı, 24 fps) | **0,90** | **%23** |
  | Çevrimdışı (referans) | 0,96 | — |

  - Simülasyon (aynı bölüm): eski 0,76/%39 → yeni (alt süreç, 1 kare gecikme) 0,90/%14. Üst sınır (her karede ana döngüde oturtma, ~18 fps'e düşürür) 0,97.
  - Yakın çekim çıkışında ızgara 28,1 sn → 27,8–27,96 sn'de geri geliyor. Çevrimdışından 0,3–0,45 sn geride; kalan fark saha modelinin süresi (~280 ms).
  - Çevrimdışı sonuç değişmedi (28–48 sn: kalibre %100, çizgi uyumu 0,953 / %10'luk 0,86, birebir aynı).
- Kalan: Yakın çekime geçişteki hızlı zoom-out (23–24 sn) canlıda hâlâ kötü (çevrimdışında da 0,80).

### D-030: Canlı modda nesne kutularının akıcı çizimi
- Tarih: 2026-10-23
- Durum: kabul
- Sorun (kullanıcı): Canlı modda oyuncu/top işaretleri takılıyormuş gibi görünüyor.
- Teşhis (ölçümle, pas_test 28–43 sn):
  1. Nesne sonucu saniyede yalnızca ~3,5 kez geliyor (375 karede 53), ortalama 8 kare (en fazla 17) geç.
  2. Sebep: Saha modeli (iki HRNet-W48, saf GPU ~195 ms, canlıda ~280 ms) 10 karede bir çalışıp GPU zamanının ~%70'ini alıyor.
  3. Arada kutular yalnızca kamera hareketi kadar kaydırılıyordu: koşan oyuncunun kutusu geride kalıyor, yeni sonuçta sıçrıyordu.
- Denenen ve reddedilen: Stabil kalibrasyonda saha modelini seyrekleştirmek (12 → 25 karede bir). Simülasyonda (`live_sim.py`, 20–38 sn) kötü kare %22 → %43 çıktı. Saha modeli 10 karede bir kalıyor.
- Çözüm (yalnızca gösterim; kayıtlar ve pas kararları dedektörün gerçek çıktısıdır):
  1. **Kutuların optik akışla takibi** (`BoxFlow`, `src/sgi/viz/live_smoother.py`): Son 40 kare yarım çözünürlükte gri saklanır. Gecikmeli sonuç gelince her oyuncu kutusundan 9 nokta (gövde + üst bacak) saklanan kareler üzerinden güncel kareye LK ile ilerletilir, sonra her kare bir adım. Kutunun yer değiştirmesi noktaların medyanıdır. Maliyet ~2,5 ms/kare.
  2. **Yedek: hızla ilerletme** (`LiveObjectSmoother`): Akış yoksa son iki sonuçtan kamera hareketinden arındırılmış hız (en fazla 15 kare ileri, kutu yüksekliğinin 0,25'i/kare sınırı).
  3. **Sönümleme:** Yeni sonuçtaki fark her karede yarıya iner; büyük farkta (kimlik değişimi) doğrudan sıçrar.
  4. **Top:** Hızla ilerletilmez (7–8 karede yön değiştirebilir; ölçümde kötüleştirdi), sönümlenmez. Sonuç karesinde bir oyuncunun ayağındaysa o oyuncuyla birlikte taşınır.
  5. Mini harita konumu, çizilen ayak noktasının güncel homografiyle izdüşümüdür.
  6. Nedenseldir (D-028). `live.smooth_objects: false` ile kapatılır.
- Ölçüm (`scripts/live_obj_sim.py`: çevrimdışı s28_d20 tespitleri yer gerçeği, sonuç 7 karede bir, 8 kare gecikmeyle):

  | | Oyuncu hata ort / %90 | Oyuncu sarsıntı %99 | Top hata ort |
  |---|---|---|---|
  | Önce (yalnızca kamera telafisi) | 26,7 / 56,7 px | 39,5 px | 99 px |
  | Hızla ilerletme + sönümleme | 20,0 / 40,7 px | 28,5 px | 94 px |
  | **+ optik akış (seçilen)** | **9,2 / 20,3 px** | **18,7 px** | **91 px** |

  - Gerçek canlı test: 24,6 fps (önce 24,75), çizim 21 ms.
- Kalan: Top hâlâ ~8 kare geriden geliyor. Bu, GPU kapasitesiyle (sonuç sıklığı) sınırlı. Bir çözüm topun kendi kısa adımlı takibi ya da saha modelinin hızlandırılması (TensorRT vb.) olabilir.

### D-031: Canlı mod performansı: maliyet/fayda analizi
- Tarih: 2026-10-23
- Durum: kabul
- Sorun (kullanıcı): Canlıda tespitler atlamalı; kayıtlı video akıcı. Hangi hesaplama ne kadar fayda sağlıyor?
- Ölçüm (tek başına, RTX 3050): saha modeli nokta ağı 98 ms + çizgi ağı 98 ms; YOLOv8x 1280 80 ms; BoT-SORT 8,7 ms; kamera hareketi 7,8 ms; çizim 11,6 ms; kutu akışı 0,4 ms. Canlıda GPU zamanının ~%70'i saha modelindeydi; aynı süreçteki GPU iş parçacığı Python kilidi (GIL) yüzünden saha modelini 109 → ~200 ms, çizimi 12 → 24 ms'ye şişiriyordu.
- Kararlar:
  1. **Saha modelinin çizgi ağı kapatıldı** (`use_lines: false`). Kare başına nokta sayısı 10,2 → 10,1 (pas_test, 40 karede bir örnek). Canlı simülasyonda çizgi uyumu aynı (20–38 sn 0,846 / 0,846; 0–20 sn 0,908 / 0,908; 40–60 sn 0,923 / 0,930). Süre 211 → 109 ms. Çevrimdışı da kapalı (canlı = çevrimdışı ilkesi).
  2. **GPU işleri ayrı süreçte** (`ProcessGpuScheduler`): Ana süreç her kareyi paylaşımlı belleğe yazar, alt süreç boşaldığı anda en güncelini alır.
- Sonuç (canlı, pas_test 28–48 sn, 20 sn):

  | | Nesne sonucu | Nesne gecikmesi | Saha modeli | Çizim |
  |---|---|---|---|---|
  | Önce | 69 (3,5/sn) | 8,0 kare | 291 ms | 23,5 ms |
  | Çizgi ağı kapalı | 114 | 6,3 | 197 ms | 24,4 ms |
  | + GPU alt süreci | **123 (6,2/sn)** | **5,8** | 158 ms | 15 ms |

- Araştırılan, henüz uygulanmayan: TensorRT (YOLO ve HRNet; kurulu değil, ~1–2 GB indirme), RF-DETR, top için TrackNet/WASB türü ısı haritası modelleri.

### D-032: Penaltı noktası filtresi: kalibrasyondan bağımsız öğrenilen işaretler
- Tarih: 2026-10-23
- Durum: kabul
- Sorun (kullanıcı): Canlıda penaltı noktası yer yer hâlâ top seçiliyor.
- Neden: D-028 filtresi adayı işaretin bilinen saha konumuyla (1,5 m) karşılaştırır. Kalibrasyon kayınca nokta haritada 1,5 m'den uzağa düşer ve filtreden kaçar.
- Çözüm: Yanında oyuncu olmadan (2 m) 1,2 sn boyunca 1 m içinde kıpırdamayan kabul edilmiş top, saha işareti sayılır. Konumu öğrenilir, takip bırakılır. Sonraki adaylar bilinen işaretler gibi elenir (2 m; oyuncu topu oraya sürerse kabul). Öğrenilen işaret yeniden görüldükçe konumu güncellenir (kaymayı izler), 20 sn görülmezse unutulur. Nedenseldir.
- Ölçüm: pas_test 28–48 sn çevrimdışı: top karesi 487 → 487, değişen kare yok (gerçek topu elemiyor). 4 yeni birim testi. Hata bu çalıştırmalarda yeniden üretilemediği için canlıdaki etkisi kullanıcının gözlemiyle doğrulanacak.
- Kalan: Izgara yokken (homografi yok) top yalnızca görüntü konumuyla seçilir, işaret filtresi çalışmaz.

### D-033: Sahne kesmesi algılama
- Tarih: 2026-10-23
- Durum: kabul
- Sorun: Ana kameradan çim seviyesindeki kameraya kesmede (pas_test 23,88 sn) LK kamera hareketi iki farklı görüntü arasında 40+ eşleşme bulup "geçerli" hareket döndürüyordu. Eski ızgara birkaç kare yakın çekimin üstüne yanlış çiziliyordu.
- Çözüm: Ardışık karelerin HSV renk histogramı (Bhattacharyya) karşılaştırılır. Eşik 0,35 (pas_test: kesmeler >= 0,45, diğer kareler <= 0,17). Kesmede hareket None: ızgara bırakılır, gecikmeli ölçümler kesmeden öteye taşınmaz.
- Sonuç (canlı): ızgara tam kesmede (23,88) kaybolur, ana kameraya dönüşte (27,52) 0,12 sn sonra geri gelir.
- Yakın çekim sırasında ızgara çizilmez: başka bir kameradır (çim seviyesi), saha çizgisi görünmez, ana kameranın önceki karelerinden geometrik olarak türetilemez.

### D-034: Az noktadan başlangıç kalibrasyonu (orta saha)
- Tarih: 2026-10-23
- Durum: kabul
- Sorun (kullanıcı): Video başında ızgara 3–5 sn sonra geliyor. Ölçüm: ilk 4,4 sn saha modeli karede 0–3 nokta buluyor (görünen: uzak taç çizgisi, orta yuvarlak parçası, ceza sahası köşesi). 4 noktalı çözüm fiziksel kamera kontrolünden geçmiyor. Eşik düşürmek ve çizgi ağı işe yaramadı.
- Çözüm (`src/sgi/mapping/bootstrap.py`): Izgara kayıpken ve >= 3 nokta varken, bilinmeyenler gerçek yayın kamerasının parametreleri olarak çözülür: dönüş (3) + odak (1) + konum (3). Artıklar: noktalar (px), izdüşen saha çizgisi örneklerinin (orta yuvarlak ve yaylar dahil) beyaz çizgi maskesine uzaklığı (kaba 45 px -> ince 15 px), çim piksellerinin saha içinde kalması. İki aşama: konum adaylarda sabit, noktalara dönüş+odak; sonra hepsi birlikte. Kabul: fiziksel kamera + çizgi uyumu >= 0,85 + saha/çim örtüşmesi >= 0,85 + nokta hatası <= 12 px. Kabul edilen kamera konumu sabit kamera önseline (D-027) de öğretilir. En fazla 5 karede bir denenir (~0,5–1 sn CPU). Sonraki karelerde kamera hareketi + çizgilere oturtma ile taşınır. Nedenseldir.
- Ölçüm (pas_test):
  - Kestirilen kamera konumu kareler arasında tutarlı: (52,7, 114, −23,7) m civarı; odak ~4100 px.
  - Çevrimdışı 0–10 sn: ilk ızgara 5,4 sn → 0,0 sn; görünür %100; saniyelik çizgi uyumu medyanı 0,70–1,00 (en zayıf 4. sn, hiç nokta yok, yalnızca taşıma).
  - Canlı 0–12 sn: ilk ızgara 5,4 → 0,76 sn; görünür %93,7.
  - 20–40 sn: yakın çekimde yanlış ızgara kurulmadı (yalnızca kesme geçiş karesi 23,88); diğer bölümler çizgi uyumu 0,98 / 0,94.
  - 2 sentetik birim testi.

### D-035: Poz modeli ile topu gizleyen oyuncu (oyuncu yönü)
- Tarih: 2026-10-23
- Durum: kabul
- Fikir (kullanıcı): Top kaybolduğunda topu tutan oyuncunun sırtı kameraya dönükse top gövdenin arkasında saklıdır, onda sayılır.
- Ön ölçüm (pas_test, 0–20 ve 28–48 sn): top oyuncuda sayılan ("held") 71 aralık; medyan 0,08 sn, en uzun 0,68 sn; 66'sı top yeniden görülerek bitiyor. Mevcut kural (son görüldüğü oyuncuda 3 sn) bu videoda zaten yetiyor. Held karelerinin çoğunda top görüntüde görünür ama dedektör kaçırmış; bu yüzden "önü dönük ve top görünmüyor -> top onda değil" kuralı yanlış olurdu, eklenmedi.
- Karar: Poz yalnızca süreyi **uzatır**: 3 sn dolunca sahibin yönüne bakılır; sırtı dönükse 8 sn'ye kadar (`hold_back_max_s`) onda sayılır, 5 karede bir yeniden bakılır; önü dönük ya da karar verilemezse süre biter. Model (`yolo11m-pose`, 256 px) tam karede değil, yalnızca sahibin kırpıntısında ve yalnızca bu durumda çalışır: normal akışta maliyeti sıfır (crop başına ~15–35 ms).
- Yön: COCO sol/sağ omuz ve kalça sırası (kameraya dönükte sol omuz görüntünün sağında). Yüz noktası güveni kullanılmaz (küçük oyuncuda sırttan bakışta da başa yüz noktası koyuyor).
- Görsel kontrol (50 rastgele oyuncu kırpıntısı, 28–48 sn, elle bakıldı, yer gerçeği değil): ~38 karar, ~3 açık hata; hataların ikisi sırtında numara görünen oyuncuya "önü dönük" (yani kaçırma yönünde, yanlış uzatma yönünde değil). 40 px altı oyuncularda çoğunlukla karar yok.
- Etki: pas_test'te 3 sn'yi aşan held aralığı olmadığı için çıktı değişmez. Kod yolu kısa süre sınırıyla (0,2 sn) gerçek karelerde ve canlı alt süreçte çalıştırıldı, hata yok. Arayüz `PoseEstimator`, adaptör `ultralytics_pose`; 6 birim testi.

### D-036: TensorRT ile GPU hızlandırma
- Tarih: 2026-10-23
- Durum: kabul (kullanıcı onayıyla kuruldu: tensorrt-cu12 11.4, onnx, onnxslim)
- Uygulama: `scripts/build_trt.py` saha noktası ağını (fp16 ONNX -> motor; TensorRT 11'de ağ güçlü tipli, fp16 bayrağı yok) ve dedektörü (ultralytics export, 1280 px, fp16) `models/trt/` altına derler. Adaptörlerde `trt_engine` ayarı; motor dosyası yoksa PyTorch'a düşülür. `src/sgi/detection/trt_runner.py`: torch tensörleriyle kopyasız çalıştırma, ayrı CUDA akışı. Motorlar bu GPU'ya ve TensorRT sürümüne özeldir; başka makinede yeniden derlenir (~10 dk).
- Ölçüm (`scripts/bench_trt.py`, pas_test 150 kare):
  - Saha noktası ağı: 108 -> 47 ms. Noktaların %99,8'i ikisinde de var; ortak noktada fark medyan 0 px, %95 3,8 px, en çok 8 px (ısı haritası 2 px adımlı).
  - Dedektör: 80 -> 63 ms. PyTorch kutularının %98,4'ünün eşi var (IoU medyan 0,986); top 204 -> 200, hakem 105 -> 94.
- Canlı (30 sn, penceresiz): saha modeli 119 -> 55 ms, gecikme 3,9 -> 2,2 kare; nesne sonucu 185 -> 307 (6,2 -> 10,2/sn), nesne gecikmesi ort 11,4 -> 3,8 kare, en çok 101 -> 11 kare. fps 24,2–24,3 (değişmedi).
- Çevrimdışı 20–30 sn: 6,8 -> 11,7 kare/sn. Homografi birebir aynı değil (fark medyan 0,18 m, %95 0,6 m); çizgi uyumu medyanı 0,968 (PyTorch) / 0,975 (TensorRT): doğruluk düşmüyor. Eski PyTorch çıktılarıyla regresyon karşılaştırması artık tolerans dışı kalır.

### D-037: Pas tespiti ve kural motoru (Faz 1, M6–M8)
- Tarih: 2026-10-23
- Durum: kabul (ölçüm bekliyor: elle etiketli pas listesi)
- Uygulama: `src/sgi/events/passes.py` (M6, kare kare ve nedensel), `src/sgi/rules/engine.py` (M7, R-OUT, R-1…R-4, R-3T; R-0 kayda yazılır), `configs/pass_rules.yaml`, `scripts/passes.py` (objects.npz -> passes.json, summary.json, `--video` ile passes_overlay.mp4), `src/sgi/viz/pass_overlay.py`. Yedek kurallar: F-1 (boşluk <= 15 kare doğrusal doldurma, yalnızca iki görülmüş konum arasında), F-2 (son görülen konumda biter), F-4 (takımı belirsiz oyuncu yok sayılır), F-5 (sahip onaylanmadan ayrılan topta son dibindeki oyuncu pasör). F-3 kalibrasyon katmanında.
- Ölçüm katmanı eşikleri (kural değil, `pass_rules.yaml` `possession`/`pass_end`): sahiplik 1,5 m ve 3 kare; durma 1,5 m/sn altında 10 kare.
- pas_test'te bulunan sahte pas kaynakları ve önlemleri:
  - Yan yana iki oyuncu arasında "en yakın oyuncu" gidip geliyordu: pas ancak top sahibinden 1,5 m'den fazla uzaklaşınca başlar (§A.2'nin doğrudan uygulaması).
  - Görüntüde üst üste duran (sahada ~4,5 m derinlik farkı olan) iki oyuncu arasında top konumu kareden kareye atlıyor: ortalama hızı 40 m/sn'yi aşan "pas" ölçüm sıçramasıdır, kaydedilmez (`max_ball_speed_mps`).
  - Top süren oyuncunun takip kimliği değişiyor: pasör artık görünmüyor, "alıcı" başlangıçta yoktu ve pasörün son görüldüğü yere koşarak (2 m + 9 m/sn) ulaşılabilecek yerde ise aynı kişidir, top sürme sayılır.
  - Sonuç: 64,6 sn'de 49 -> 27 pas. Kalan bilinen hata: takip kutusunun yandaki oyuncuya atlaması (40,8 sn).
- Doğruluk henüz ölçülmedi (şartname §9: elle doğrulanmış örnek olmadan "doğru" denmez).

### D-038: Hücre numaralandırması sol alt köşeden, sütun sütun alttan yukarı
- Tarih: 2026-10-28
- Durum: kabul (hocanın geri bildirimi)
- Karar: `cell_id = col * 5 + (4 - row) + 1`. 1 numara sol alt köşe (sol kale çizgisi + yakın taç çizgisi). İlk sütun alttan yukarı 1–5, ikinci sütun 6–10, …, son sütun 46–50. Hücre sayısı, sınırlar ve kurallar değişmedi; yalnızca etiketler değişti.
- Uygulama: `PitchGrid.cell_id` / `row_col` (`src/sgi/grid/grid.py`). Tüm modüller numarayı buradan aldığı için başka kod değişmedi. `tests/test_grid.py` yeni numaralara göre güncellendi.
- Mevcut çıktılar: pas_test `objects.npz` dosyalarındaki hücre sütunu eski → yeni olarak dönüştürüldü (konumdan yeniden hesaplanan hücreyle %100 uyumlu), `passes.json` / `summary.json` / `passes_overlay.mp4` yeniden üretildi. Pas sayıları ve sonuçları değişmedi (27 pas; A 16/7, B 0/4). Önceden yazılmış `grid_overlay.mp4` dosyalarında eski numaralar vardır; `play` ve `live` ızgarayı yeniden çizdiği için yeni numaraları gösterir.

### D-039: Bilgi yazıları ve mini harita videonun yanında panelde (16:9)
- Tarih: 2026-10-28
- Durum: kabul (hocanın geri bildirimi: yazılar ve mini harita videonun büyük kısmını kapatıyordu)
- Karar: Çıktı tuvali videonun 1,5 katı (1280x720 -> 1920x1080, 16:9). Video ölçeklenmeden sol üstte durur. Sağ sütunda DURUM (kare, zaman, kalibrasyon kaynağı, canlı ölçümler), MİNİ HARİTA (600 px genişlik, eskisinin ~1,6 katı) + renk açıklaması ve PAS SAYACI var. Alt şeritte ŞU ANKİ PAS ve son 6 pasın tablosu (zaman, takım, hücre, sonuç, kural) yer alır. Videonun üzerinde yalnızca sahaya ait çizimler kalır: ızgara, hücre numaraları, oyuncu etiketleri, kale ve top yolu. Kullanıcı üç seçenek (L şekli / yalnızca sağ / yalnızca alt) arasından L şeklini seçti.
- Uygulama: `src/sgi/viz/panel.py` (`PanelLayout`), `GridOverlay.draw` artık tuval döndürür (`out_size`), `PassOverlay` paneli kullanır. `play` aynı klasörde `passes.json` varsa pas panelini de gösterir. `scripts/passes.py --video` pas videosunu eski `grid_overlay.mp4` üzerine değil, orijinal videodan (`--source`, varsayılan `data/videos/<klasör>.mp4`) ızgara + oyuncular + paslarla birlikte çizer; böylece yeniden işleme gerektirmez (pas_test 64,6 sn: ~70 sn). `calibration.npz` okuma `sgi.io.calibration.CalibrationCache` içinde toplandı.
- Performans: tuvalin arka planı ve mini haritanın sabit katmanları (saha çizgileri, hücre sınırları, numaralar) bir kez hazırlanıp her karede kopyalanır. Tuval 9 -> 1,4 ms. Canlı test (pas_test, 30 sn penceresiz): 21,7 fps, çizim ort 24,5 ms (D-030 ölçümü 21 ms). Çizimdeki en büyük kalem, eskiden de olan mini haritadaki "tespit edilen çizgi" katmanı (~12 ms, `c` tuşuyla kapatılabilir).
- Önceden üretilmiş `grid_overlay.mp4` dosyaları eski düzendedir; bir sonraki `process` çalıştırmasında panelli üretilir.
- **Güncelleme (aynı gün, kullanıcı geri bildirimi):** Alt şerit (şu anki pas, son paslar) gereksiz yer kapladığı için kaldırıldı. Video 1,125 kat büyütülerek 1440x810 gösterilir, sağ panel 480 px; tuval 1920x810. Saha çizimleri videonun kendi çözünürlüğünde yapılır, sonra video ile birlikte büyütülür (homografi koordinatları değişmez). Sağ sütun: DURUM, MİNİ HARİTA (440 px), PAS SAYACI. Hücrelerin beyaz sınır çizgileri başlangıçta kapalı (`OverlayOptions.grid = False`, `g` tuşuyla açılır); dama tahtası renkleri hücreleri ayırmaya devam eder. Canlı test (30 sn penceresiz): 24,2 fps, çizim ort 11,4 ms.
- **Güncelleme 2 (aynı gün):** Sağ sütun sabit konumlar yerine yukarıdan aşağı akar: DURUM satırları, hemen altında MİNİ HARİTA + açıklama, onun altında PAS SAYACI (`GridOverlay.panel_bottom`). Canlı moddaki fazladan satırlar için durum bölümü o ana kadarki en uzun hali kadar yer tutar, böylece harita kareden kareye zıplamaz. Mini haritada saha çizgileri beyaz ve 2 px, hücre sınırları ince ve koyu yeşil (önce hücre sınırları saha çizgilerinden daha parlaktı, ayırt edilemiyordu).
- **Güncelleme 3 (aynı gün):** Videodaki dama dolgusunun opaklığı 0,22 -> 0,10 (`OverlayOptions.fill_alpha`). 0,22 oyuncuların forma renklerini soluklaştırıyordu; 0,07'de hücreler zor seçiliyordu (karşılaştırma görseliyle seçildi). `f` tuşuyla dolgu tamamen kapatılabilir.
- **Güncelleme 4 (aynı gün):** Hücre sınırı çizgileri yeniden başlangıçta açık (`OverlayOptions.grid = True`), ama yarı saydam: `line_alpha = 0,45` (1,0 / 0,6 / 0,45 / 0,3 karşılaştırıldı; 0,3'te uzak taraftaki çizgiler kayboluyor). Canlı test (30 sn): 24,2 fps, çizim ort 14,9 ms.

### D-040: Hedef takım modu ve canlı modda pas sayımı
- Tarih: 2026-10-28
- Durum: kabul (hocanın isteği: iki takımlı videonun yanında yalnızca seçilen hedef takımın paslarını sayan ikinci bir video)
- Karar: İki mod. **İki takım** (varsayılan, §C) ve **hedef takım** (`--team A|B`). Hedef takım modunda pas tespiti ve kurallar aynen çalışır; yalnızca sayım ve gösterim süzülür. Diğer takımın pasları sayılmaz, top yolları çizilmez, sayaçta yalnızca hedef takım satırı ve "HEDEF TAKIM: A" başlığı bulunur. Şimdilik hedef takım komut satırından verilir (A); başlangıçta nasıl seçileceği sonra belirlenecek (Q17).
- Uygulama: `summarize(records, target_team)` (`src/sgi/rules/engine.py`), `PassOverlay(target_team=...)`. `scripts/passes.py --team A [--video]` -> `summary_A.json`, `passes_overlay_A.mp4`; iki takımlı `summary.json` / `passes_overlay.mp4` ayrı dosyalar olarak kalır. `passes.json` her zaman tüm pasları içerir (izlenebilirlik). `grid_demo.py play|live --team A`.
- **Canlı modda pas sayımı (yeni):** Canlı modda önceden pas tespiti yoktu. Artık gelen her nesne sonucu `PassDetector`'a verilir, kural motoruyla sınıflandırılır ve panelde sayılır (nedensel). Atlamada (`j`/`l`) ve video başa sardığında sayaç sıfırlanır.
- Dedektör eşikleri (sahiplik 3 kare, durma 10 kare) artık sonuç sayısıyla değil, kapsanan kare aralığıyla ölçülür. Çevrimdışında her karede sonuç olduğu için sonuç **birebir aynıdır** (27 pas, tüm alanlar karşılaştırıldı). Canlıda nesne sonuçları ~2 karede bir gelir; eski sayım canlıda eşikleri iki katına çıkarıyordu.
- Ölçüm (pas_test ilk 30 sn, `live --headless-seconds 30`): çevrimdışı 11 pas (A 9 + B 2), canlı 5 pas (A 4 + B 1). Ortak 4 pas, birinde hedef hücre farklı. Canlıda kaçanlar çoğunlukla 0,1–0,4 sn süren kısa paslar ve hızlı el değiştirmeler (takip canlıda seyrek çalışır). Canlı mod gösterim içindir; raporlanan sayılar çevrimdışı çıktıdan alınır. Kısıt: `13_LIMITATIONS.md`.
- Canlı test: 24,2 fps, çizim ort 14,9 ms (pas tespitinin ek yükü ölçülemeyecek kadar küçük).

### D-010: Veri ve video kaynağı
- Tarih: 2026-10-21
- Durum: kabul
- Karar: Videoları ekip seçecek. Önce paylaşılacak repolardaki hazır etiketli veri setleri değerlendirilecek. İhtiyaca uymazlarsa kendi veri setimiz oluşturulup etiketlenecek.
