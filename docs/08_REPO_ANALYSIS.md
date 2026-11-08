# 08 — GitHub Repo İncelemesi: SoccerNet Organizasyonu

> Kaynak: https://github.com/SoccerNet (22 repo). İnceleme tarihi: 2026-10-21.
> Yöntem: Her reponun README dosyası, gerekli yerlerde kod ağacı, konfigürasyon dosyaları (`pyproject.toml`, `soccernet.yaml`), veri yükleyicileri ve Hugging Face veri seti kartları okundu.
> Veri setlerinin ayrıntısı `07_DATASETS.md` dosyasında.

## 0. Yönetici özeti

| Öncelik | Repo | Bizim modül | Ne sağlıyor | Karar |
|---|---|---|---|---|
| ⭐⭐⭐ | **sn-gamestate** (+ TrackLab) | M2, M3, M3b, M4 | Oyuncu, kaleci ve hakem tespiti + takip + takım ataması + saha kalibrasyonu (NBJW / PnLCalib / TVCalib). **GSR veri seti:** bbox, **saha koordinatı**, rol (oyuncu/kaleci/hakem/**top**), takım, kare başına saha çizgisi | **UYARLA**: model ve ağırlıkları adaptörle kullan, veri setini eğitim/değerlendirme için kullan |
| ⭐⭐⭐ | **sn-calibration** | M2 (keypoint/çizgi), M4 | 26 sınıflı saha çizgisi şeması, `soccerpitch.py` saha modeli, homografi ve kamera kodu, kalibrasyon veri seti | **REFERANS + VERİ**: saha şablonu ve çizgi etiketleri |
| ⭐⭐⭐ | **sn-gar-2027** (SoccerNet-GAR) | M6, M7 değerlendirmesi | Dünya Kupası 2022'nin 64 maçı: **PASS** vb. olay etiketleri, **takım**, video ve **saha koordinatında oyuncu + 3B top izleri** (30 fps) | **VERİ (değerlendirme)**: pas tespiti ve hücre mantığı için en değerli yer gerçeği (ground truth) kaynağı. NDA gerekli |
| ⭐⭐ | **sn-tracking** | M2, M3 | 200 klip × 30 sn, MOT formatında bbox + track ID (oyuncu, kaleci, hakem, **top**) | **VERİ**: dedektör (özellikle top) eğitimi ve takip değerlendirmesi |
| ⭐⭐ | **sn-spotting** + **sn-teamspotting** | M6 değerlendirmesi | Ball Action Spotting: **Pass**, Drive, High Pass, Cross, Out… zaman damgalı, takım (sol/sağ) bilgili. T-DEED modeli | **VERİ (değerlendirme)**: video üzerinde pas anları. NDA gerekli. Model **kullanılmayacak** (D-002) |
| ⭐⭐ | **SoccerNet** (pip paketi) | Altyapı | Tüm veri setlerini indirme aracı (artık Hugging Face üzerinden) | **KULLAN** |
| ⭐ | **sn-trackeval** | Değerlendirme | HOTA / GS-HOTA metrikleri | Opsiyonel |
| ⭐ | **sn-reid**, **sn-jersey** | M3 (dolaylı) | Yeniden tanıma ve forma numarası | Opsiyonel. Projede forma numarası gerekmiyor |
| ⭐ | **SoccerNet-v3** | M2 | Tek karelerde bbox + takım + saha çizgisi | Opsiyonel ek eğitim verisi |
| — | sn-banner | (M4 dolaylı) | NBJW kalibrasyon ağırlıklarının indirme linkleri | REFERANS |
| ✗ | sn-mvfoul, sn-foulret, sn-caption, sn-echoes, sn-grounding, sn-vqa-2027, sn-depth, sn-nvs, ActiveSpotting, PTS-baseline | — | Faul, altyazı, yorum sesi, tekrar, VQA, derinlik, yeni görünüm sentezi, aktif öğrenme | **KULLANMA**: kapsam dışı |

**Kritik bulgular:**
1. **Hazır modellerin hiçbiri bizim için topu güvenilir şekilde tespit etmiyor.** GSR baseline'ı genel amaçlı COCO `yolo11m.pt` kullanıyor ve yalnızca sporcularla ilgileniyor. Top için **GSR + Tracking verisinde bulunan top bbox'larıyla kendi YOLO modelimizi ince ayarlamamız** gerekecek. Bu, D-011'deki adaptör yapısıyla uyumludur.
2. **Saha keypoint ve kalibrasyon için hazır, güçlü bir model var: NBJW (No Bells, Just Whistles).** HRNet tabanlı keypoint ve çizgi modelleri, ağırlıkları GitHub release'te. GSR pipeline'ında varsayılan kalibrasyon modülü bu.
3. **Hiçbir repo pası başarılı/başarısız olarak sınıflandırmıyor.** Bizim M5–M7 katmanımız özgün katkı olarak kalıyor.
4. **SoccerNet-GAR**, saha koordinatında oyuncu ve top izleri ile PASS etiketlerini birlikte veriyor. Bu sayede M6–M7 mantığı **görüntü işleme hatalarından bağımsız** olarak test edilebilir (bkz. §3).
5. **Sürüm çakışması riski:** sn-gamestate, **Python 3.9 + torch 1.13.1 + mmcv 2.0.1** sürümlerine sabitlenmiş. Bizim pipeline'ı bu ortama bağlamamalıyız. Önerilen çözüm §4'te.

---

## 1. Ayrıntılı repo incelemeleri

### 1.1 SoccerNet/sn-gamestate ⭐⭐⭐
- **URL:** https://github.com/SoccerNet/sn-gamestate · ⭐468 · son güncelleme 2026-05 · **GPL-3.0**
- **Amaç:** Game State Reconstruction (GSR). Tek, hareketli yayın kamerasından tüm sporcuların **saha üzerindeki 2B konumu**, rolü (oyuncu/kaleci/hakem/diğer), takımı (kameraya göre sol/sağ) ve forma numarası çıkarılır ve mini haritada gösterilir.
- **Çerçeve:** [TrackLab](https://github.com/TrackingLaboratory/tracklab), Hydra konfigürasyonlu ve modüler bir yapı. Modüller `soccernet.yaml` dosyasından değiştirilebilir. Bu yapı bizim D-011 tasarımımıza çok benzer.
- **Pipeline ve kullanılan modeller:**

  | Adım | Varsayılan modül | Model | Bizdeki karşılık |
  |---|---|---|---|
  | bbox_detector | `yolo_ultralytics` | `yolo11m.pt` (COCO, genel) | M2 `ObjectDetector`. Alternatifler: rtdetr, yolox, rtmdet |
  | reid | `prtreid` | PRTReid (ReID + takım + rol çok görevli) | M3 yardımcı |
  | track | `bpbreid_strong_sort` | StrongSORT + BPBreID | M3 `Tracker` |
  | pitch | `nbjw_calib` | No-Bells-Just-Whistles keypoint + çizgi (HRNetV2-w48) | M2 `PitchKeypointDetector` |
  | calibration | `nbjw_calib` | Keypoint'lerden kamera parametreleri | M4 homografi |
  | jersey_number_detect | `mmocr` | MMOCR | Gerekmez |
  | tracklet_agg | `voting_role_jn` | Rol ve numara oylaması | M3b (çoğunluk oyu fikri) |
  | team | `kmeans_embeddings` | ReID gömmeleri üzerinde KMeans (k=2), track bazında | M3b `TeamAssigner` |
  | team_side | `mean_position` | Takım ortalama konumuna göre sol/sağ | M3b (A/B etiketleme) |

  Alternatif kalibrasyon modülleri: `tvcalib`, `pnlcalib`, `baseline` (sn-calibration), `bbox2pitch`.
- **Veri seti (SoccerNetGS, sürüm 1.3):** 200 klip × 30 sn, train/valid/test/challenge. `Labels-GameState.json` dosyası **COCO benzeri** bir formattır:
  - `object` süper kategorisi: `bbox_image`, `bbox_pitch` (saha koordinatı, metre), `track_id`, `attributes{role ∈ player/goalkeeper/referee/ball/other, team ∈ left/right, jersey}`
  - `pitch` süper kategorisi: **her görüntü için saha çizgisi** (`lines`) etiketleri
  - `camera` süper kategorisi: kamera parametreleri
  - Hugging Face: `SoccerNet/SN-GSR-2024`, `SN-GSR-2025`, **herkese açık**
- **Hazır çıktılar:** Baseline "tracker state" dosyaları (valid/test/challenge) Zenodo'da. Önceden hesaplanmış tespit, takip ve kalibrasyon sonuçlarını içerir. Bizim `CachedDetector` adaptörümüz için hazır bir girdi olabilir.
- **Bağımlılıklar:** `python >=3.9,<3.10`, `torch==1.13.1`, `mmcv==2.0.1`, `mmocr`, `mmdet~=3.1`, `tracklab==1.3.24`, `numpy==1.26.4`. **Eski ve katı sürümler.** Windows'ta mmcv kurulumu zahmetli olabilir.
- **Lisans notu:** GPL-3.0. Kodunu projemize kopyalarsak projemiz de GPL olur. Ödev için sorun değil ama **ayrı ortamda çalıştırıp çıktısını okumak** daha temizdir.
- **Bizim için kullanım:**
  - ✅ **Veri seti:** M2 dedektörünü eğitmek (oyuncu, kaleci, hakem, **top**), M3b takım atamasını değerlendirmek, M4 homografiyi `bbox_pitch` ile **metre cinsinden** değerlendirmek.
  - ✅ **NBJW kalibrasyonu:** M2 keypoint + M4 için ana aday.
  - ✅ **Takım ataması fikri:** Track bazında gömme ortalaması + KMeans(2) + konuma göre taraf. Bizim M3b tasarımıyla aynı.
  - ⚠️ **Tam pipeline:** Ağır (ReID, OCR). Bizim ihtiyacımız için fazla. İsteğe bağlı bir "GSR adaptörü" olarak dışarıda çalıştırılıp çıktısı okunabilir.
  - ❌ **Top:** GSR pipeline'ı topu tespit etmiyor ve izlemiyor. Bunu biz eklemeliyiz.
- **Karar: UYARLA.**

### 1.2 SoccerNet/sn-calibration ⭐⭐⭐
- **URL:** https://github.com/SoccerNet/sn-calibration · ⭐111 · 2024-06 · lisans belirtilmemiş
- **Amaç:** (1) Saha çizgisi uç noktalarının tespiti, (2) kamera kalibrasyonu.
- **Veri:** SoccerNet-v3 görüntülerinde **26 semantik çizgi sınıfı** (büyük ve küçük ceza sahası, orta çizgi, kenar çizgileri, daireler, kale direkleri). Her çizginin uç noktaları etiketli, bazılarında çizgi üzerinde ek noktalar da var. HF: `SN-Calibration-2023` (açık).
- **Kod:** `soccerpitch.py` (FIFA ölçülerinde 3B saha modeli), `camera.py` (homografi ayrıştırma, projeksiyon), `baseline_cameras.py`, `evaluate_camera.py` (JaC@t metriği).
- **Baseline model:** DeepLabv3 çizgi segmentasyonu (ağırlık Google Drive'da). Zayıf: JaC@5 = %11.7. **NBJW çok daha iyi** olduğu için model olarak tercih edilmez.
- **Bizim için kullanım:**
  - ✅ `soccerpitch.py` ile çizgi sınıf şeması → `configs/pitch.yaml` kanonik keypoint/çizgi şablonumuz için referans.
  - ✅ Kalibrasyon veri seti → keypoint modelini ince ayar veya doğrulama.
  - ✅ JaC metriği → M4 değerlendirmesi (opsiyonel).
  - ⚠️ "Kale arkası kamera" belirsizliği (sol/sağ ayrılamaz) burada açıkça belgelenmiş. `13_LIMITATIONS.md` dosyasına eklenecek.
- **Karar: REFERANS + VERİ.**

### 1.3 SoccerNet/sn-gar-2027 (SoccerNet-GAR) ⭐⭐⭐
- **URL:** https://github.com/SoccerNet/sn-gar-2027 · 2026-10 · veri: HF `OpenSportsLab/SoccerNet-GAR` (**gated + NDA**, yalnızca araştırma ve ticari olmayan kullanım)
- **İçerik:** FIFA Dünya Kupası 2022'nin 64 maçı. **87.939 olay** (train 62.159 / valid 12.091 / test 13.689).
  - 10 sınıf: **PASS** (valid'de 7.762), HIGH PASS, CROSS, HEADER, OUT, THROW IN, SHOT, PLAYER SUCCESSFUL TACKLE, FREE KICK, GOAL
  - Her olayda: `game_id`, `position_ms` (zaman damgası), **`team` (home/away)**
  - 4 modalite: `videos` (yayın MP4, ~29.97 fps), `frames`, `tracking` (16 kare örneklenmiş), `tracking-full` (144 kare, tam fps)
  - **Takip verisi:** Saha koordinatında **2B oyuncu konumları** (x ∈ [-60, 60] m, y ∈ [-42, 41] m) ve **3B top konumu** (z yüksekliği ∈ [-8, 25] m), 30 fps. Yayından otomatik çıkarılıp **elle düzeltilmiş**. Oyuncu pozisyon rolü de var.
- **Baseline:** GIN (graph neural network) ile olay sınıflandırma. **Bizim için kullanılmaz** (pas tespitini öğrenilmiş modelle değil, kuralla yapıyoruz).
- **Bizim için değeri (çok yüksek):**
  - ✅ **M5–M7'nin görüntü işlemeden bağımsız testi:** Gerçek saha koordinatları → 50 hücre → sahiplik → pas adayı → kural motoru. Olay etiketleriyle (PASS + zaman) karşılaştırılır. Böylece "kural motoru ve olay tespiti doğru mu?" sorusu, "tespit doğru mu?" sorusundan ayrılır (şartname §9).
  - ✅ **Uçtan uca değerlendirme:** Aynı olayın videosu da var. Videodan bizim pipeline'ımızın ürettiği sonuç, takip verisinden üretilenle karşılaştırılabilir.
  - ✅ **Top yüksekliği (z):** "Havadaki top" kısıtının etkisi ölçülebilir.
  - ⚠️ **Sınır:** Etiket "başarılı/başarısız pas" değildir. Ama sonraki sahiplik takip verisinden çıkarılabilir. Bu, bizim kuralımızla üretilen bir **yarı-otomatik referans** olur. Rapor için elle doğrulanmış bir alt küme gerekir.
  - ⚠️ Koordinat sistemi merkez orijinli (-52.5..52.5 × -34..34 civarı). Bizim 0..105 × 0..68 sistemine dönüştürme adaptörü gerekir.
- **Karar: VERİ (değerlendirme).** Erişim başvurusu **erkenden** yapılmalı (manuel onay ve NDA gerekiyor).

### 1.4 SoccerNet/sn-tracking ⭐⭐
- **URL:** https://github.com/SoccerNet/sn-tracking · ⭐113 · 2023-07
- **Veri:** 12 maç, ana kamera. **200 klip × 30 sn** (+1 tam devre). MOT formatında CSV: `frame, track_id, x, y, w, h, conf`. Nesneler: oyuncu, kaleci, hakem, **top**, diğer. Sınıflar değerlendirmede kullanılmıyor ama `gameinfo.ini` içinde track ID → sınıf/takım eşlemesi var. HF: `SN-Tracking-2023` (açık).
- **Baseline'lar:** ByteTrack, DeepSORT, FairMOT (`Benchmarks/` altında).
- **Bizim için:** ✅ **Top dedektörü eğitimi** (her karede top bbox'ı), ✅ M3 takip değerlendirmesi (HOTA), ✅ ByteTrack referansı.
- **Not:** GSR veri seti bu verinin üzerine kuruludur (aynı 200 klip + saha konumu ve rol). Mümkünse doğrudan GSR kullanılmalı.
- **Karar: VERİ.**

### 1.5 SoccerNet/sn-spotting ve sn-teamspotting ⭐⭐
- **URL:** https://github.com/SoccerNet/sn-spotting (⭐105), https://github.com/SoccerNet/sn-teamspotting (⭐19, GPL-3.0)
- **Veri:**
  - *Action Spotting (v2):* 500 maç, 17 seyrek sınıf (gol, kart, korner…). Pas yok.
  - *Ball Action Spotting (BAS 2024/2025):* 7 (+) maç, **12 yoğun sınıf**: **Pass**, Drive, Header, **High Pass**, Out, Cross, Throw In, Shot, Ball Player Block, Player Successful Tackle, Free Kick, Goal. 2025 sürümünde **takım (sol/sağ)** bilgisi var. 25 fps, ±1 sn tolerans.
  - HF: `SN-BAS-2024/2025` **NDA-gated**. `SN-PCBAS-2026` (oyuncu merkezli) da NDA-gated.
- **Model:** T-DEED (2024 kazananı), checkpoint Google Drive'da.
- **Bizim için:**
  - ✅ **Değerlendirme:** Yayın videosunda pas anlarının elle etiketlenmiş zamanları. M6 pas tespitimizin recall ve precision değerini ölçmek için kullanılabilir.
  - ❌ **Model:** T-DEED pası öğrenilmiş modelle "tespit eder". Bizim kural tabanlı ve açıklanabilir yaklaşımımızla (D-002, D-009) çelişir. Kullanılmaz. En fazla karşılaştırma için anılabilir.
  - ⚠️ GAR ile kısmen örtüşür (aynı sınıf isimleri). GAR'da saha koordinatları da olduğu için **GAR öncelikli**.
- **Karar: VERİ (opsiyonel değerlendirme).**

### 1.6 SoccerNet/SoccerNet (pip paketi) ⭐⭐
- **URL:** https://github.com/SoccerNet/SoccerNet · MIT · 2026-10 (aktif)
- **İşlev:** `SoccerNetDownloader` ile tüm görev verilerinin indirilmesi. **Veriler artık Hugging Face'te.** Açık setler doğrudan inebilir. Gated setler için HF hesabı + NDA onayı + `hf auth login` gerekir.
- **Açık veri setleri:** SN-GSR-2024/2025, SN-Tracking-2023, SN-Calibration-2023, SN-ReID-2023, SN-Jersey-2023.
- **Gated veri setleri:** SN-BAS-*, `SoccerNet_raw_HQ` (tam maç videoları), SoccerNet-GAR.
- **Karar: KULLAN** (veri indirme aracı, hafif bağımlılık).

### 1.7 Diğerleri
| Repo | Özet | Projeye katkı | Karar |
|---|---|---|---|
| sn-trackeval | TrackEval fork'u, GS-HOTA metriği | M3 / M3b değerlendirmesinde opsiyonel | Opsiyonel |
| sn-reid | Çoklu görüş oyuncu ReID veri seti ve baseline | Takip kimliği kalitesine dolaylı katkı | Opsiyonel |
| sn-jersey | Forma numarası tanıma | Gereksiz (takım yeterli) | Kullanma |
| SoccerNet-v3 | Tekil karelerde bbox + takım + saha çizgisi + tekrar eşleşmeleri, ~60 GB | Ek dedektör/keypoint eğitim verisi | Opsiyonel |
| sn-banner | Reklam panosu değiştirme. NBJW ağırlık linklerini veriyor (`SV_kp`, `SV_lines`) | Ağırlık kaynağı referansı | Referans |
| sn-depth | Video oyunlarından sentetik derinlik verisi | Kapsam dışı | Kullanma |
| sn-nvs | Yeni görünüm sentezi (Blender) | Kapsam dışı | Kullanma |
| sn-mvfoul / sn-foulret | Faul sınıflandırma ve geri getirme | Kapsam dışı | Kullanma |
| sn-caption / sn-echoes / sn-vqa-2027 | Altyazı, yorum sesi, soru-cevap | Kapsam dışı | Kullanma |
| sn-grounding / ActiveSpotting / PTS-baseline | Tekrar eşleme, aktif öğrenme, olay tespiti | Kapsam dışı | Kullanma |

---

## 2. Modül → kaynak eşleme matrisi

| Bizim modül (D-011 arayüzü) | Hazır model / adaptör adayı | Eğitim / ince ayar verisi | Değerlendirme verisi |
|---|---|---|---|
| M2 `ObjectDetector` (oyuncu, kaleci, hakem) | YOLO11 (ultralytics) — GSR'nin kullandığı | **SN-GSR** (rol etiketli bbox), SN-Tracking | SN-GSR test |
| M2 `ObjectDetector` (**top**) | **Hazır uygun model yok** → YOLO11 ince ayar (yüksek çözünürlük / SAHI) | **SN-GSR + SN-Tracking top bbox'ları** | SN-GSR test (top recall) |
| M2 `PitchKeypointDetector` | **NBJW** (HRNetV2-w48 keypoint + çizgi), alternatif PnLCalib, TVCalib | SN-Calibration-2023, SN-GSR `pitch.lines` | SN-Calibration (JaC), SN-GSR |
| M3 `Tracker` | ByteTrack (hafif) veya StrongSORT (GSR) | — | SN-Tracking (HOTA) |
| M3b `TeamAssigner` | Gömme + KMeans(2) + `mean_position` (GSR yaklaşımı). Basit alternatif: forma rengi HSV + KMeans | — | SN-GSR `team` etiketleri |
| M4 homografi | NBJW `utils_calib` / sn-calibration `camera.py` | — | **SN-GSR `bbox_pitch`** (metre hatası) |
| M4b `CameraMotionEstimator` | OpenCV (ORB + RANSAC). SoccerNet'te hazır model yok | — | SN-GSR (keypoint'leri gizleyip taşıma hatasını ölçme) |
| M5 grid | Kendi kodumuz | — | Birim testleri, **GAR koordinatları** |
| M6 pas adayı | Kendi kodumuz (kural tabanlı) | — | **SoccerNet-GAR** (PASS + takip), SN-BAS (opsiyonel) |
| M7 kural motoru | Kendi kodumuz | — | GAR'dan türetilen referans + elle doğrulanmış klipler |

---

## 3. Önerilen değerlendirme stratejisi (şartname §9 ile uyumlu)
1. **Katman ayrımı:**
   - (a) **M5–M7:** GAR **gerçek** takip verisiyle test edilir. Görüntü hatası sıfırdır, yalnızca mantık test edilir.
   - (b) **M2–M4:** SN-GSR ile test edilir (tespit, takım, saha konumu hatası).
   - (c) **Uçtan uca:** GAR videoları + bizim pipeline. Sonuç (a) ile karşılaştırılır.
2. Başarılı/başarısız etiketi hiçbir veri setinde doğrudan yok. Rapor için elle doğrulanmış küçük bir pas seti (ör. 50–100 pas) oluşturulmalı (`11_EVALUATION.md`).

## 4. Ortam ve kurulum önerisi (D-007 için girdi)
- **Ana ortam (bizim pipeline):** Güncel Python (3.11 veya 3.12) + güncel PyTorch (CUDA) + ultralytics + OpenCV + numpy/pandas. NBJW modelleri yalnızca PyTorch + HRNet koduna ihtiyaç duyar ve yeni torch ile çalışır (sn-banner README'si "NBJW yeni torch ile daha iyi" diyor).
- **NBJW** kodu ve ağırlıkları bizim `PitchKeypointDetector` adaptörümüze **taşınır**. Kaynak: `mguti97/No-Bells-Just-Whistles` ve `sn-gamestate/plugins/calibration/nbjw_calib`. Lisans kontrol edilecek.
- **sn-gamestate tam pipeline'ı gerekirse** ayrı bir ortamda (Python 3.9 + torch 1.13.1) çalıştırılır. Çıktısı (tracker state / JSON) `CachedDetector` adaptörüyle okunur. **Ana pipeline bu eski sürümlere bağlanmaz.**
- Makinede NVIDIA sürücüsü mevcut (PATH'te NVIDIA bileşenleri görülüyor). GPU modeli ve CUDA sürümü kurulum aşamasında kontrol edilecek.

## 5. Açık eylemler
- [ ] **SoccerNet-GAR** için Hugging Face erişim başvurusu + NDA (onay süresi belirsiz, erken başlanmalı)
- [ ] (Opsiyonel) SN-BAS-2025 erişim başvurusu
- [ ] NBJW lisansının ve ağırlık indirme linklerinin doğrulanması
- [ ] SN-GSR `Labels-GameState.json` örneğinin indirilip alanların doğrulanması (top `bbox_pitch` var mı?)
- [ ] GAR `tracking-full` parquet şemasının erişim sonrası incelenmesi
