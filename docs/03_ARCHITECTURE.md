# 03 — Mimari Tasarım

> Durum: **Taslak**. Teknoloji seçimleri GitHub repo incelemesinden sonra kesinleşecek (bkz. `08_REPO_ANALYSIS.md`, `09_DECISIONS.md`).

## 1. Genel akış (pipeline)

```
 Video
   │
   ▼
[M1] Video I/O ──────────► Frame(idx, t, image)
   │
   ▼
[M2] Detection ──────────► Detections(players, goalkeepers, referees, ball)
   │                       PitchKeypoints(image_xy ↔ pitch_id, conf)
   ▼
[M3] Tracking ───────────► Tracks(track_id, bbox, class)   + ball track (interpolasyon)
   │
   ├──► [M3b] Team Assignment ─► track_id → team_id ∈ {A, B, REF, UNKNOWN}
   ▼
[M4] Pitch Mapping ──────► H(frame) + source ∈ {keypoints, camera_motion, nearest} ; pitch_xy (metre)
   │   └─ [M4b] Camera Motion ─► kareler arası görüntü dönüşümü (keypoint yetersizse H'yi taşır, F-3)
   │
   ▼
[M5] 50-Cell Grid ───────► cell_id ∈ {1..50} | OUT | UNKNOWN
   │
   ▼  (FrameState zaman serisi)
[M6] Pass Event Detection ► PassCandidate(start, cells[], end, passer, receiver, evidence)
   │
   ▼
[M7] Rule Engine ────────► PassRecord(label ∈ {SUCCESS, FAIL}, reason, rule_id, fallbacks_applied)
   │
   ▼
[M8] Reporting / Viz ────► JSON/CSV kayıtlar, özet istatistik, anotasyonlu video, 2B saha görünümü
```

**Temel ilkeler:**
1. M1–M6 adımları **gözlem** üretir. Neyin olduğuna yalnızca M7 **karar** verir. Kurallar M2–M6 içine gömülmez.
2. **Model bağımsızlık (D-011):** Hiçbir modül belirli bir modele veya kütüphaneye bağımlı değildir. Modeller **arayüz + adaptör** yapısıyla takılıp çıkarılır, seçim yalnızca konfigden yapılır (bkz. §7).

## 1b. Model bağımsız katman (D-011)

### Amaç
Oyuncu, top ve hakem tespiti başta olmak üzere tüm öğrenilmiş bileşenler şu şekillerde değiştirilebilir olmalıdır:
- Hazır bir modelin (ör. GitHub reposundaki ağırlık) yerine **kendi eğittiğimiz** model konabilmeli
- **Farklı bir hazır model** (başka mimari veya kütüphane) kullanılabilmeli
- Bu değişiklikler **yalnızca konfig değişikliğiyle** yapılabilmeli, pipeline kodu değişmemeli

### Yapı
```
                    ┌──────────────────────────────────┐
  pipeline.yaml ──► │  Registry / Factory              │
  detector:         │  build("detector", cfg)          │
    type: yolo      └──────────────┬───────────────────┘
    weights: ...                   │ seçilen adaptörü üretir
    class_map: ...                 ▼
          ┌─────────────────────────────────────────────────┐
          │ «interface» ObjectDetector                      │
          │   detect(frames: list[Frame]) -> list[list[Detection]] │
          └───────▲───────────────▲────────────────▲────────┘
                  │               │                │
       UltralyticsYoloDetector  RepoXDetector   CachedDetector
       (hazır/kendi ağırlık)    (başka repo     (önceden hesaplanmış
                                 adaptörü)       sonuçları dosyadan okur)
```

### Arayüzler (`src/sgi/interfaces.py`)
| Arayüz | Metot (özet) | Çıktı (ortak tip, `types.py`) | Kullanan |
|---|---|---|---|
| `ObjectDetector` | `detect(frames)` | `Detection[]`: bbox, **kanonik sınıf**, conf | M2 |
| `PitchKeypointDetector` | `detect(frames)` | `PitchKeypoint[]`: **kanonik keypoint ID**, image_xy, conf | M2 |
| `Tracker` | `update(detections, frame)` | `Track[]` | M3 |
| `TeamAssigner` | `fit(crops)`, `predict(track_crops)` | `track_id → team_id` | M3b |
| `CameraMotionEstimator` | `estimate(prev_frame, frame, mask)` | 3×3 görüntü→görüntü dönüşümü + kalite | M4b |
| `PitchRefiner` (planlı, D-014) | `refine(frame, H_pred)` | Düzeltilmiş H + kalite (ör. çim şeridi haritası) | M4 |

### Kurallar
1. **Kanonik şema:** Pipeline yalnızca kendi sınıf adlarımızı (`player, goalkeeper, referee, ball`) ve `configs/pitch.yaml` dosyasındaki keypoint numaralarını bilir. Her modelin kendi sınıf ve keypoint numaralandırması, adaptörde konfigdeki `class_map` / `keypoint_map` ile **kanonik şemaya çevrilir**.
   ```yaml
   detector:
     type: ultralytics_yolo          # registry anahtarı
     weights: models/repoX_players.pt
     conf: 0.3
     class_map: {0: ball, 1: goalkeeper, 2: player, 3: referee}
   ```
2. **Ortak çıktı tipi:** Adaptörler kütüphaneye özgü nesneler (ultralytics `Results`, `supervision.Detections` vb.) döndürmez. Yalnızca `types.py` dataclass'larını döndürür.
3. **Bağımlılık izolasyonu:** Bir kütüphanenin importu yalnızca kendi adaptör dosyasında yapılır (ör. `detection/adapters/ultralytics_yolo.py`). O kütüphane kurulu değilse yalnızca o adaptör kullanılamaz.
4. **Ayrı modeller desteklenir:** Top için ayrı bir model kullanılabilir. `CompositeDetector` birden fazla dedektörün çıktısını birleştirir (ör. oyuncu modeli + top modeli).
5. **Önbellek / tekrar oynatma:** `CachedDetector` önceden kaydedilmiş tespitleri okur. Hem hızlı deney yapmayı hem de modelleri adil karşılaştırmayı sağlar.
6. **Takım bilgisi tespitten ayrıdır:** Bir model doğrudan "Takım A / Takım B" sınıfı üretse bile bu bilgi `TeamAssigner` arayüzü üzerinden ele alınır. Böylece tespit modeli değişince takım mantığı bozulmaz.
7. **Sözleşme testleri:** Her adaptör, aynı test setiyle (`tests/contracts/`) çıktı tipinin, kanonik sınıfların ve koordinat sisteminin doğru olduğunu doğrular. Yeni bir model eklemek için yeni bir adaptör yazıp bu testlerden geçirmek yeterlidir.
8. **Model karşılaştırma:** `scripts/eval_detector.py --config <cfg>` aynı değerlendirme setinde (`11_EVALUATION.md`) farklı adaptörleri ölçer.

## 2. Modüller, girdiler ve çıktılar

| # | Modül | Girdi | Çıktı | Bağımsız test |
|---|---|---|---|---|
| M1 | `io` – Video I/O | Video yolu, kesit (start/end) | `Frame` akışı (idx, timestamp, BGR) | Kare sayısı/fps doğrulama |
| M2 | `detection` – Nesne ve anahtar nokta tespiti | `Frame` | `Detection[]` (bbox, class, conf), `Keypoint[]` | Etiketli test seti üzerinde mAP ve keypoint hatası |
| M3 | `tracking` – Oyuncu ve top takibi | `Detection[]` zaman serisi | `Track[]` (track_id sabit), top izi (eksik kareler interpolasyonla doldurulur, `interpolated=True` bayrağıyla) | ID switch sayısı, MOTA/HOTA (opsiyonel) |
| M3b | `teams` – Takım ataması | Oyuncu kırpıntıları + `track_id` | `track_id → team_id` (+ güven) | Elle etiketli kırpıntılarda doğruluk |
| M4 | `mapping` – Saha koordinat eşlemesi | `Keypoint[]`, saha modeli, M4b çıktısı | Kare başına `H` matrisi, `source` (keypoints / camera_motion / nearest), `reproj_error`, `valid`; nesneler için `pitch_xy` | Bilinen noktalarda reprojeksiyon hatası (metre) |
| M4b | `mapping.camera_motion` – Kamera hareketi | Ardışık kareler + oyuncu/top maskesi | Kareler arası görüntü dönüşümü `M(t→t-1)` + kalite | Sentetik kaydırma/zoom ile; keypoint'li karelerde sürüklenme (drift) ölçümü |
| M5 | `grid` – 50 hücre | `pitch_xy` | `cell_id` / `OUT` / `UNKNOWN` | Saf birim testleri (sınır durumları) |
| M6 | `events` – Pas olayı tespiti | `FrameState[]` zaman serisi | `PassCandidate[]` | Elle işaretlenmiş olay zaman damgalarıyla eşleştirme |
| M7 | `rules` – Kural motoru | `PassCandidate` + `pass_rules.yaml` | `PassRecord` (label, reason, rule_trace) | Sentetik aday paslarla birim testleri |
| M8 | `reporting` / `viz` | `PassRecord[]`, `FrameState[]` | `passes.json`, `summary.json/csv`, anotasyonlu MP4, radar/saha görüntüsü | Sayım kurallarının birim testi |

## 3. Ara veri temsili: `FrameState`
Her kare için M5 sonunda tek bir yapı oluşur (ayrıntılar `06_DATA_MODEL.md` içinde):
- `frame_idx`, `timestamp`
- `homography_valid`, `homography_quality`
- `players[]`: `track_id`, `team_id`, `role` (player/goalkeeper/referee), `image_xy` (bbox alt-orta noktası), `pitch_xy`, `cell_id`, `conf`
- `ball`: `image_xy`, `pitch_xy`, `cell_id`, `conf`, `interpolated`, `visible`

`FrameState` dizisi diske yazılır (ör. parquet/JSON). Böylece M6–M8 videoyu ve modelleri yeniden çalıştırmadan, **çevrimdışı** ve hızlı test edilebilir. Kural ayarlarını değiştirip denemek için bu önemlidir.

## 4. Önemli tasarım noktaları
- **Oyuncu konumu:** Bbox'ın **alt-orta** noktası (ayak) homografiyle dönüştürülür.
- **Top konumu:** Top havadayken homografi topun yere izdüşümünü değil, yanlış bir noktayı verir. Bu sınırlılık `13_LIMITATIONS.md` dosyasında belgelenmiştir.
- **Homografi ve kamera hareketi (F-3, D-012):** Her kare için H şu öncelik sırasıyla elde edilir:
  1. **keypoints:** Yeterli keypoint var (`≥ 4`, reprojeksiyon hatası eşik altında) → doğrudan `findHomography`.
  2. **camera_motion:** Keypoint yetersiz → bir önceki karenin H'si kamera hareketiyle taşınır: `H_t = H_{t-1} · M_{t→t-1}`. `M`, ardışık kareler arasında **oyuncu ve top maskelenerek** (yalnızca çim, çizgi, tribün gibi statik sahne) özellik eşleme (ORB/SIFT + RANSAC) veya optik akış ile kestirilir. Kamera dönme/zoom yaptığı için `M` bir homografi olarak modellenir.
  3. **nearest:** Kamera hareketi de kestirilemezse (sahne kesmesi, tekrar gösterimi) zamanca en yakın geçerli H kullanılır.
  - **Sürüklenme (drift) kontrolü:** Zincirleme taşımada hata birikir. Keypoint tekrar görünür görünmez H yeniden sabitlenir (re-anchor). Ardışık taşıma sayısı `max_motion_chain` sınırını aşarsa `source=nearest` olur.
  - Çevrimdışı çalıştığımız için taşıma **iki yönlü** yapılabilir: önceki ve sonraki keypoint'li karelerden taşınan iki tahminden zincir uzunluğu kısa olan (veya ağırlıklı ortalaması) seçilir.
  - Kısa titreşimler için kareler arasında yumuşatma yapılır.
- **Takım tutarlılığı:** Takım ataması `track_id` bazında çoğunluk oyu ile yapılır. Böylece tek karedeki renk hatası takımı değiştirmez.
- **Hakem:** Ayrı sınıf olarak tespit edilir ve takım kümelemesine dahil edilmez.
- **Hata ayrımı:** Her modül kendi kalite bayraklarını üretir (`det_conf`, `track_interpolated`, `homography_valid`, `cell_unknown`). Pas kaydı bu bayrakları taşır. Böylece bir hatanın tespit, takip, eşleme veya kural kaynaklı olduğu ayırt edilebilir.

## 5. Önerilen klasör yapısı
```
sgi_project/
├── CLAUDE.md                 # Claude için bağlam giriş noktası
├── docs/                     # Bu dokümanlar
├── configs/
│   ├── pipeline.yaml         # model yolları, eşikler, kesit bilgisi
│   ├── pitch.yaml            # saha ölçüleri ve keypoint tanımları
│   ├── grid.yaml             # 50 hücre düzeni
│   └── pass_rules.yaml       # kural tanımları (M7)
├── src/sgi/
│   ├── types.py              # ortak veri yapıları (dataclass), kanonik sınıflar
│   ├── interfaces.py         # ObjectDetector, PitchKeypointDetector, Tracker, TeamAssigner, CameraMotionEstimator
│   ├── registry.py           # konfigdeki "type" → adaptör sınıfı (factory)
│   ├── io/
│   ├── detection/
│   │   ├── adapters/         # ultralytics_yolo.py, <repo>_detector.py, cached.py, composite.py
│   │   └── keypoints/adapters/
│   ├── tracking/adapters/    # bytetrack.py, ...
│   ├── teams/adapters/       # color_kmeans.py, embedding_kmeans.py, ...
│   ├── mapping/              # homography.py, camera_motion/ (adapters), pitch_model.py
│   ├── grid/  events/  rules/
│   └── reporting/  viz/
├── scripts/                  # run_pipeline.py, train_*.py, eval_*.py
├── tests/                    # modül bazlı birim testleri
│   └── contracts/            # her adaptörün geçmesi gereken sözleşme testleri
├── data/                     # (git dışı) raw/ interim/ processed/ annotations/
├── models/                   # (git dışı) eğitilmiş ağırlıklar
├── notebooks/                # keşif ve doğrulama
└── outputs/                  # (git dışı) çalıştırma çıktıları
```

## 6. Aday teknoloji yığını (kesin değil — adaptör olarak eklenir, D-011)
| İş | Aday | Not |
|---|---|---|
| Dil | Python 3.10+ | |
| Oyuncu/top/hakem tespiti | Ultralytics YOLO (v8/v11) | Hazır etiketli futbol veri setleriyle ince ayar |
| Top tespiti | Ayrı küçük-nesne modeli veya yüksek çözünürlük / SAHI | Top küçük ve bulanık |
| Takip | ByteTrack (`supervision`) | |
| Takım ataması | Forma rengi kümeleme (HSV + KMeans) veya gömme + KMeans | Hakem ayrı tutulur |
| Saha keypoint | YOLO-pose (saha keypoint modeli) | Hazır veri seti aranacak |
| Homografi | OpenCV `findHomography` + RANSAC | |
| Kamera hareketi | OpenCV ORB/SIFT eşleme + RANSAC homografi, veya `calcOpticalFlowPyrLK` | Oyuncu/top maskelenir |
| Konfig | YAML + pydantic/dataclass doğrulama | |
| Görselleştirme | OpenCV, `supervision`, matplotlib | |
