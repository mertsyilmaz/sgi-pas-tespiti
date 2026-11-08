# 15 — İyileştirme Analizi: Tespit, Kalibrasyon ve Kalelerin Destekleyici Kullanımı

> Tarih: 2026-10-22. Kullanıcı sorusu: Tespit ve 50 hücre performansı başka repolardan parça alınarak artırılabilir mi? Kaleler, çim renkleri gibi destekleyici olarak kullanılabilir mi?
> Bu bir analizdir, uygulama kararı değildir. Seçilen maddeler `09_DECISIONS.md`'ye ayrı karar olarak girer.
> Durum tabanı: D-020 (çizgilere oturtma), D-022 (YOLO11 COCO + ByteTrack + renk kümeleme), D-023 (canlı mod GPU kuyruğu).

---

## 0. Mevcut darboğazlar (ölçülmüş)

| Alan | Gözlem (pas_test) | Kök neden |
|---|---|---|
| Rol ayrımı | Hakem/kaleci renk sezgiseliyle. Bazı kısa takipler "?" | COCO modeli yalnızca "person" biliyor |
| Takip | ~22 kişi için **178 kimlik** | ByteTrack görüntü koordinatında çalışıyor, kamera dönünce eşleşme kopuyor |
| Top | Karelerin %73'ünde bulundu | COCO "sports ball" küçük ve bulanık topta zayıf, top takibi yok |
| Kalibrasyon kapsamı | %87 (ilk 5 sn orta saha yok) | Orta sahada NBJW 8 inlier üretemiyor |
| Kalibrasyon hassasiyeti | OK %82, büyük kaymalar (>4 m) düzelmiyor | Tek model, tek kare; çizgilere oturtma 4 m ile sınırlı |

---

## 1. Tespit (oyuncu / kaleci / hakem / top)

### 1.1 Futbola özel tespit modeli — en büyük kazanç
| Seçenek | Kaynak | Ne kazandırır | Maliyet / risk |
|---|---|---|---|
| **A. YOLO'yu SN-GSR ile ince ayarlamak** | `sn-gamestate` veri seti (açık): rol etiketli bbox (player/goalkeeper/referee/ball) | Rol doğrudan modelden gelir (renk sezgiseli yalnızca takım için kalır). Top recall belirgin artar (SoccerNet topları ayrıca etiketli) | Veri boyutu indirmeden önce kontrol edilmeli (onlarca GB olabilir). 4 GB GPU'da YOLO11s/m eğitimi saatler sürer (imgsz 960–1280, batch 2–4). Adaptör değişmez, sadece `weights` + `class_map` |
| B. Hazır futbol modeli (repo dışı) | Ör. `roboflow/sports` (futbol oyuncu/top/hakem/kaleci YOLOv8 ağırlıkları + saha keypoint modeli) | Eğitimsiz, hemen denenebilir | SoccerNet dışı. Lisans ve ağırlık indirme yolu doğrulanmalı. Kendi verimizle değerlendirilmeli |
| C. Top için ayrı dedektör / döşemeli çıkarım | SAHI benzeri döşeme (tiling) veya topa özel yüksek çözünürlük | Küçük top recall'u artar | Kare başına süre artar (çevrimdışı kabul edilebilir, canlıda değil) |

**Öneri:** Önce **B** ile hızlı karşılaştırma (1–2 saat). Kalıcı çözüm **A**. Veri boyutu ve eğitim süresi öğrenildikten sonra karar verilir. Her iki durumda da `UltralyticsYoloDetector` adaptörü aynen kullanılır (D-011).

### 1.2 Takip (kimlik değişimleri) — ucuz ve etkili
| Seçenek | Kaynak | Ne kazandırır | Maliyet |
|---|---|---|---|
| **A. Saha koordinatında takip** | Kendi kodumuz (homografimiz var) | Kalman filtresi ve eşleme metre cinsinden yapılır. Kamera dönmesi/zoom kimlik değişimine yol açmaz. Oyuncu hızları fiziksel olarak sınırlıdır (≤ 10 m/s) | Orta. Homografi olmayan karelerde görüntü uzayına düşülür |
| **B. BoT-SORT + kamera hareketi telafisi (GMC)** | Ultralytics içinde hazır (`BOTSORT`). sn-tracking benchmark'larındaki yöntemlerin devamı | Kamera hareketi telafisi ID değişimini azaltır. İsteğe bağlı ReID | Düşük. Yeni bir `Tracker` adaptörü |
| C. ReID (görünüm benzerliği) | `sn-gamestate`: PRTReid / BPBReID (+ `sn-reid` veri seti) | Uzun örtülmelerden sonra aynı kişiyi tanır. PRTReid aynı ağda takım ve rol de verir | Yüksek. Eski ortam (torch 1.13, Python 3.9). Ayrı ortamda çalıştırıp çıktıyı dosyadan okumak gerekir |
| D. Forma numarası | `sn-jersey` / sn-gamestate MMOCR modülü | Kalıcı kimlik (numara) | Yüksek. Yayın görüntüsünde numara çoğu karede okunamaz |

**Öneri:** **B** (hemen) → **A** (sonra). C ve D projenin hedefi (pas sayımı) için şart değil.

### 1.3 Takım ataması
- Mevcut renk kümeleme bu videoda iyi çalışıyor. Rol modelden gelirse (1.1-A) en zayıf parçası olan hakem/kaleci sezgiseli ortadan kalkar.
- Alternatif: Görüntü gömme (SigLIP/CLIP) + kümeleme (`roboflow/sports` yaklaşımı) veya PRTReid takım başlığı. Benzer renkli formalarda daha sağlam, ama ek model ve maliyet getirir. Şu an gerek yok.

### 1.4 Top
- Dedektör iyileştirmesine (1.1) ek olarak **top izi**: kısa kayıplarda interpolasyon (F-1 ile uyumlu) ve fiziksel tutarlılık (ani sıçrama → aykırı). Kendi kodumuz, ucuz. Pas tespiti için zaten gerekecek.

---

## 2. Kalibrasyon / 50 hücre

| Seçenek | Kaynak | Ne kazandırır | Maliyet |
|---|---|---|---|
| **A. NBJW'nin ince ayarlı ağırlıkları** | NBJW release: `SV_FT_TSWC_*` (TS-WorldCup, yayın videosu), `SV_FT_WC14_*`, `MV_*` (çok görüşlü) | Sadece ağırlık değişimi. Yayın görüntüsünde nokta sayısı ve doğruluk artabilir | Çok düşük. Konfigde `weights_kp/lines` değişir, kalite göstergesiyle A/B karşılaştırma |
| **B. PnLCalib** | `mguti97/PnLCalib` (NBJW'nin devamı, aynı yazar). sn-gamestate'te `pnlcalib` modülü olarak hazır | Nokta + çizgi birlikte kullanılır. Daha az nokta ile çözüm üretebilir: orta saha kapsamına doğrudan etki eder | Orta. Yeni `PitchKeypointDetector` adaptörü (D-011). Lisans ve ağırlıklar kontrol edilmeli |
| C. TVCalib | sn-gamestate `tvcalib` modülü (optimizasyon tabanlı, segment eşleme) | Çizgi segmentleri + daire üzerinden kamera parametresi optimizasyonu. Zor görüntülerde sağlam | Yüksek. Yavaş (kare başına optimizasyon), eski ortam |
| D. Yer gerçeğiyle ölçüm | SN-GSR (kare başına çizgi + kamera etiketi) | A/B/C'nin hangisinin gerçekten iyi olduğunu sayıyla gösterir (`14_CALIBRATION_QUALITY` §2.1) | Orta. Veri indirme |
| E. Hız | NBJW'yi ONNX/TensorRT'ye çevirmek | 1,5–2 kat. Canlı modda ızgara daha sık güncellenir | Orta–yüksek. Windows kurulumu zahmetli |

**Öneri:** **A** (yarım saat) → **B** (adaptör) → mümkünse **D** ile hangisinin gerçekten daha iyi olduğunu ölçmek.

---

## 3. Kaleler destekleyici olarak kullanılabilir mi?

Evet, ama çim şeritlerinden farklı türden bir destek sağlar. Kale **standart ölçülüdür** (7,32 × 2,44 m) ve **tektir** (sahada yalnızca iki tane, konumu kesin: x = 0 / 105, y = 30,34–37,66). Çim şeritlerinin aksine periyodik belirsizliği yoktur.

| Kullanım | Açıklama | Değer |
|---|---|---|
| 1. Sol/sağ belirsizliğini çözme | Kale arkası ve yakın açılarda görüntü tek başına sahanın hangi yarısı olduğunu söylemez (sn-calibration'da belgelenmiş). Görünen kale ve kalenin kendi ceza sahasıyla ilişkisi bunu çözer | Yüksek. E4 (aynalanmış ızgara) türü hataları önler |
| 2. Ek zemin noktası | Direklerin dibi (z = 0) zemin düzlemindedir. NBJW bu noktaları zaten veriyor ve homografide kullanıyoruz. Direklerin tepesi ve üst direk 3B noktalardır (z = 2,44), düzlemsel homografide kullanılamaz | Kısmen var |
| 3. **Dikey referans → tam kamera modeli** | Direkler gerçek dünyada diktir ve yüksekliği bilinir. Zemin homografisi + dikey kaybolma noktası ile tam kamera matrisi (K, R, t) çıkarılabilir. Bu: (a) odak uzaklığını sağlamlaştırır, fiziksel kontrolü güçlendirir; (b) **havadaki topun yüksekliğinin** tahmini için gerekli altyapıyı verir (bilinen kısıt, 13_LIMITATIONS) | Orta–yüksek. Pas analizinde uzun toplar için önemli |
| 4. Kalite göstergesi | İzdüşürülen kale çerçevesi ile görüntüde tespit edilen direklerin (beyaz dikey çizgiler, Hough) örtüşmesi. Çizgi uyumu gibi etiketsiz bir doğrulama | Orta |
| 5. Kale tespiti | Klasik: beyaz dikey direkler + yatay üst direk (renk eşikleme, Hough, geometri). Ya da NBJW'nin direk noktaları | Klasik yöntem derse uygun |

**Sınırlar:**
- Kale yalnızca ceza sahası çevresindeki görüntülerde görünür (pas_test'te karelerin ~%30–40'ı).
- Sorunlu olduğumuz **orta saha görüntülerinde kale yoktur**. Orada destek **orta yuvarlak** (elips uydurma, klasik) ve **orta çizgiden** gelmelidir. Bu, `10_ROADMAP` → "orta saha başlangıç kalibrasyonu" maddesiyle aynı iştir.
- Benzer standart nesneler: penaltı noktası (11 m), köşe bayrakları (dikey, konumu kesin), orta nokta.

**Çim şeritleriyle karşılaştırma:**

| | Çim şeritleri | Kale | Orta yuvarlak |
|---|---|---|---|
| Ölçü standart mı | Hayır | Evet | Evet (r = 9,15 m) |
| Benzersiz mi | Hayır (periyodik) | Evet (2 tane) | Evet |
| Ne zaman görünür | Hemen her karede | Ceza sahası çevresinde | Orta sahada |
| Ne sağlar | Kayma düzeltme, hareket | Sol/sağ, 3B kamera, top yüksekliği | Orta saha başlangıç kalibrasyonu |

---

## 4. Önerilen yol haritası (öncelik sırası)

| # | İş | Beklenen etki | Süre (tahmini) |
|---|---|---|---|
| 1 | NBJW ince ayarlı ağırlıklarını A/B denemek (2-A) | Kalibrasyon kapsamı/hassasiyeti | ~0,5 saat |
| 2 | BoT-SORT + GMC takip adaptörü (1.2-B) | Kimlik değişimleri ↓ | ~1 saat |
| 3 | Hazır futbol modeli denemesi (1.1-B) | Rol + top | ~1–2 saat |
| 4 | Top izi: interpolasyon + aykırı eleme (1.4) | Top kapsamı ↑, pas tespitine hazırlık | ~1–2 saat |
| 5 | PnLCalib adaptörü (2-B) | Orta saha kapsamı ↑ | ~0,5–1 gün |
| 6 | Orta yuvarlak/kale ile destekleyici kalibrasyon (3) | Orta saha + sol/sağ + 3B kamera | ~1 gün |
| 7 | SN-GSR: indirme + değerlendirme + YOLO ince ayarı (1.1-A, 2-D) | Kalıcı tespit kalitesi + gerçek skor | ~1–2 gün (çoğu bekleme) |
| 8 | Saha koordinatında takip (1.2-A) | Kimlik değişimleri ↓↓ | ~0,5 gün |

Her adım `pas_test` üzerinde mevcut ölçütlerle karşılaştırılır:
- **Kalibrasyon:** kategoriler / çizgi uyumu
- **Tespit:** kimlik sayısı, top bulunan kare, "?" oranı

D-011 mimarisi sayesinde 1, 2, 3, 5 ve 7 sadece yeni adaptör ve konfig değişikliğidir; pipeline kodu değişmez.
