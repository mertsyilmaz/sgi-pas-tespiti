# 14 — Kalibrasyon ve Hücre Doğruluğu: Ölçüm, Hata Teşhisi ve İyileştirme Planı

> Durum: **Plan (2026-10-22, D-014).** Kullanıcıyla yapılan tartışmanın sonucu. Faz 0.5 demosu (D-013) çalışıyor. Bu dosya, doğruluğunun sayısal olarak nasıl ölçüleceğini ve nasıl iyileştirileceğini tanımlar.
> İlgili: `03_ARCHITECTURE.md` (M4, M4b), `04_PITCH_GRID.md`, `11_EVALUATION.md`, `13_LIMITATIONS.md`.

---

## 1. Mevcut yöntem ve etkileri

**Yöntem:**
1. NBJW saha noktalarını (zemin düzlemindeki görüntü ↔ saha eşleşmeleri) bulur.
2. `cv2.findHomography` + RANSAC ile 3×3 düzlemsel **perspektif homografi** hesaplanır: görüntü pikseli → saha metresi.
3. Izgara, bu homografinin tersiyle görüntüye çizilir.
4. Saha noktası yetersizse homografi kamera hareketiyle taşınır (F-3).

| Varsayım / özellik | Etki |
|---|---|
| Saha düz bir düzlemdir | Zemindeki noktalar (çizgiler, oyuncu ayağı) için doğrudur. **Havadaki top** için yanlış konum verir |
| Lens bozulması yok sayılır | Yayın kameralarında genelde küçüktür. Çok geniş açıda görüntü kenarlarında birkaç piksel sapma olabilir |
| Piksel hatası mesafeyle büyür | Yakın tarafta 1 px ≈ birkaç cm, **uzak taç çizgisi tarafında 1 px ≈ 0,3–0,5 m**. Uzak taraftaki hücre sınırları daha az hassastır |
| Noktaların dağılımı | Noktalar dar bir bölgede toplanırsa (ör. yalnızca bir ceza sahası köşesi) bölgeden uzaklaştıkça hata büyür |
| Alternatif | NBJW tam kamera kalibrasyonu (K, R, t) da verebilir. Topun yüksekliği gerekirse kullanılabilir. Hücre ataması için homografi yeterlidir |

Hata, hücre boyutuyla birlikte yorumlanmalıdır. Hücre 10,5 × 13,6 m'dir. 0,25 m hata hücre genişliğinin ~%2'sidir. **Hata pratikte yalnızca hücre sınırına yakın noktalar için önemlidir.**

---

## 2. Sayısal doğruluk ölçümü

### 2.1 Yer gerçeği (ground truth) olan veriyle — birincil yöntem
Veri: **SN-GSR** (açık). Her sporcu için `bbox_image` ve `bbox_pitch` (saha konumu, metre), her kare için saha çizgisi etiketleri. Bkz. `07_DATASETS.md`.

| Metrik | Tanım | Ne gösterir |
|---|---|---|
| **Konum hatası (m)** | Ayak noktası (bbox alt-orta) → bizim H → saha konumu, `bbox_pitch` ile farkı. Ortalama, medyan, %95 | Homografinin metre cinsinden doğruluğu |
| **Hücre doğruluğu (%)** | `cell_of(tahmin) == cell_of(gerçek)` oranı | **50 hücreye bölmenin doğrudan başarısı.** Rapordaki ana skor |
| Sınıra uzaklığa göre hücre doğruluğu | Hücre doğruluğu, gerçek noktanın en yakın hücre sınırına mesafesine göre gruplanır (<0,5 m, 0,5–1 m, >1 m) | Hatanın sınır kaynaklı olup olmadığı |
| Uzak / yakın taraf hatası | Konum hatası satıra (y) göre ayrı raporlanır | §1'deki perspektif etkisi |
| JaC@5 (opsiyonel) | sn-calibration metriği: izdüşürülen çizgilerin etiketli çizgilere 5 px'ten yakın olma oranı | Literatürle karşılaştırılabilirlik |
| Kapsama (%) | Geçerli homografi üretilen kare oranı | Kalibrasyonun ne sıklıkla kaybolduğu |

Ayrıca F-3 değerlendirmesi: Keypoint'li karelerde saha noktası modeli yapay olarak N kare kapatılır. Kamera hareketiyle taşınan H, gerçek konumla karşılaştırılır. Sonuç, **zincir uzunluğuna göre hata eğrisi** (drift) olarak raporlanır.

Betik (yapılacak): `scripts/eval_calibration.py --dataset SN-GSR --split valid`

### 2.2 Etiketsiz videoda öz-tutarlılık göstergeleri
Kendi seçtiğimiz kliplerde yer gerçeği yoktur. Aşağıdaki göstergeler kare başına hesaplanıp `calibration.npz` dosyasına yazılır:

| Gösterge | Hesap | Yakaladığı sorun |
|---|---|---|
| Reprojeksiyon hatası, inlier sayısı | Zaten var | Saha noktalarının birbiriyle tutarsızlığı |
| **Çizgi hizalama skoru** ✅ (D-019) | Saha çizgileri H⁻¹ ile görüntüye izdüşürülür. Çizgi boyunca örneklenen piksellerin "beyaz çizgi" maskesine (yüksek parlaklık + düşük doygunluk + top-hat) düşme oranı | Izgaranın gerçek çizgilerden kayması. **Etiket gerektirmez** |
| Zamansal tutarlılık | Ölçülen H ile kamera hareketinden tahmin edilen H arasındaki fark (kontrol noktalarında medyan metre) | Ani sıçrama, titreme |
| Kamera geometrisi ✅ (D-017) | `camera_plausibility`: K[r1 r2 t] ayrıştırması. Diklik, en-boy, odak aralığı, yükseklik (3–120 m). Kör nokta: tam karşıdan bakışta x-gerilmesi | Fiziksel olarak imkânsız çözümler. **Uygulandı. Az noktalı yanlış çözümleri eliyor** |
| Çim maskesi uyumu | İzdüşürülen saha poligonu ile çim maskesinin IoU'su (bkz. §4.1) | Aynalanmış veya kaymış ızgara |
| Yeşil oranı | Karedeki çim piksellerinin oranı | Saha dışı kareler |

### 2.3 Küçük bir elle etiketlenmiş set (yedek)
Kendi kliplerimizden ~50 karede, kare başına en az 4 saha noktası elle işaretlenir. Böylece referans homografi elde edilir. §2.1 metrikleri bu set için de hesaplanır. Basit bir tıklama aracı yazılabilir.

---

## 3. Hata teşhisi: kategoriler

Her kare için §2.2 göstergeleri kaydedilir ve kurallarla bir kategoriye atanır:

| Kod | Kategori | Tipik belirti |
|---|---|---|
| E1 | **Saha görünmüyor** (yakın çekim, reklam, logo geçişi) | Yeşil oranı düşük |
| E2 | **Yetersiz saha noktası** (orta saha, dar açı) | Nokta sayısı < 4 veya dar dağılım |
| E3 | **Dejenere konfigürasyon** | Noktalar neredeyse aynı doğru üzerinde |
| E4 | **Yanlış nokta kimliği** (ör. sol/sağ ceza sahası karışması) | İnlier'ı yüksek ama önceki H'den büyük sıçrama, aynalanmış ızgara, çizgi hizalama skoru düşük |
| E5 | **Kamera hareketi kayması** (F-3 drift) | Zincir uzun, çizgi hizalama skoru düşüyor |
| E6 | **Sahne kesmesi** | Optik akış inlier'ı çöker, yeşil oranı ve histogram aniden değişir |
| E7 | Kalite eşikleri aşıldı (yüksek reprojeksiyon hatası) | Kalibrasyon reddedildi |

**Çıktılar:**
- `calibration_report.json`: kategori dağılımı ve kapsama
- Zaman çizelgesi grafiği: kare ekseninde kalibrasyon kaynağı, kategoriler ve göstergeler. Sunumda kullanılabilir.

---

## 4. Çim (saha zemini) bilgisinin kullanılması — iyileştirme fikirleri

**Fikir (kullanıcı):** Sahalarda çim genelde açık/koyu yeşil şeritler hâlindedir. Saha çizgilerinin net görünmediği yerlerde bu desen, önceki kareler ve kamera hareketine ek olarak koordinat hesabına yardımcı olabilir.

**Değerlendirme:** Çim deseni tek başına mutlak koordinat vermez:
- Şerit genişliği, sayısı ve başlangıç yeri stadyuma göre değişir. Genelde 5–6 m'dir ama standart değildir.
- Desen periyodiktir (hangi şerit olduğu belirsizdir).
- Bazı sahalarda dama, daire veya çapraz desen vardır.
- Işık, gölge ve kamera açısına göre görünürlük değişir.

Ancak aşağıdaki şekillerde yardımcı bilgi olarak çok değerlidir. Kolaydan zora sıralanmıştır:

### 4.1 Çim maskesi (öncelik: yüksek, maliyet: düşük)
- **Yöntem:** HSV uzayında yeşil eşikleme + morfolojik açma/kapama + en büyük bağlı bileşen. Eşikler video başına otomatik uyarlanabilir (ör. ilk karelerin ton histogramının tepe noktası).
- **Kullanım:**
  1. Saha dışı karelerin tespiti (E1). Bu karelerde kalibrasyon denenmez ve F-3 zinciri kesilir.
  2. Kamera hareketi kestiriminde tribün, reklam panosu ve skor tabelası gibi bağımsız hareket eden bölgelerin maskelenmesi.
  3. Kalite göstergesi: izdüşürülen saha ile çim maskesinin uyumu (§2.2).

### 4.2 Çim dokusuyla kamera hareketini güçlendirme (öncelik: orta)
- Orta sahada çizgiler azken görüntüdeki en belirgin doku çim şeritleridir. Şerit kenarları optik akışa ek özellik sağlar.
- **Sınır:** Kenarlar tek yönlüdür (aperture problemi). Kamera şeritlere paralel kayarsa bu hareket kenarlardan ölçülemez. Bu yüzden diğer doku noktalarıyla birlikte kullanılmalıdır.

### 4.3 Şerit yönünden dönüş kısıtı (öncelik: düşük)
- Şerit sınırları genelde kale çizgisine paraleldir. Görüntüde Hough dönüşümüyle bulunan şerit kenarları bir kaybolma noktasında birleşir. Bu, kameranın dönüşü için ek bir geometrik kısıttır.

### 4.4 Şerit haritası ile kayma düzeltme (öncelik: orta–yüksek, özgün katkı adayı)
1. **Harita çıkarımı:** Saha noktalarıyla iyi kalibre edilmiş karelerde, kenar tespiti + Hough ile şerit sınırları bulunur ve H ile sahaya izdüşürülür. Sahadaki konumları birikerek o videoya/stadyuma özel bir **şerit haritası** oluşur (ör. sınırlar x = 5,3 / 10,9 / 16,4 … m).
2. **Kullanım:** Saha noktası yetersiz karelerde F-3 yaklaşık H verir (hata birkaç metre). Görüntüdeki şerit sınırları bu H ile sahaya izdüşürülür ve haritadaki **en yakın** sınıra eşlenir. Tahmin hatası yarım şerit genişliğinden küçük olduğu sürece periyodik belirsizlik çözülür.
3. **Düzeltme:** Eşlenen sınırlar, x = sabit doğru eşleşmeleri olarak H'yi yeniden kestirmede kullanılır (doğru-tabanlı homografi düzeltmesi). Böylece F-3 kayması sıfırlanır.
4. **Mimari:** Yeni bir değiştirilebilir bileşen olarak eklenir: `PitchRefiner` arayüzü (`refine(frame, H_pred) -> H, kalite`). `HomographyTracker` içinde F-3 tahmininden sonra çağrılır. Konfigden açılıp kapatılabilir (D-011).
5. **Sınırlar:**
   - Şeritler görünmüyorsa veya desen kale çizgisine paralel değilse modül devre dışı kalır.
   - Haritanın oluşması için önce iyi kalibre edilmiş kareler gerekir.
   - Sahne kesmesinden sonra ilk karelerde kullanılamaz.

**Ders bağlamı:** 4.1–4.4, klasik sayısal görüntü işleme yöntemlerinin derin öğrenme tabanlı keypoint modelini tamamlayacak şekilde kullanılmasıdır: renk uzayları, eşikleme, morfoloji, kenar tespiti, Hough dönüşümü, kaybolma noktası, optik akış. Raporda projenin özgün katkılarından biri olarak sunulabilir.

---

## 5. Uygulama sırası
1. **§2.2 göstergeleri + §3 kategorileri** → `grid_demo.py` raporu ve zaman çizelgesi (etiketsiz kliplerde hemen kullanılabilir)
2. **§4.1 çim maskesi** → E1 tespiti, hareket kestiriminde maskeleme, kalite göstergesi
3. **§2.1 SN-GSR değerlendirmesi** → hücre doğruluğu ve konum hatası. Bundan sonraki her iyileştirme bu skorla kıyaslanır.
4. **§4.4 şerit haritası düzeltmesi** (+ §4.2) → F-3 drift eğrisinde iyileşme ölçülür
5. (Opsiyonel) §4.3, JaC@5, elle etiketlenmiş set
