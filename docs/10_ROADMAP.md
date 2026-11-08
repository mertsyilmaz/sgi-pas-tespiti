# 10 — Yol Haritası ve İlerleme

> Durum işaretleri: ⬜ yapılmadı · 🟨 devam ediyor · ✅ tamamlandı · ⛔ engelli
> Her oturum sonunda güncellenir.

## Faz 0 — Hazırlık
- ✅ Şartnamenin okunması ve `docs/` yapısının kurulması
- ✅ SoccerNet repolarının incelenmesi → `08_REPO_ANALYSIS.md`
- ✅ Veri setlerinin belirlenmesi → `07_DATASETS.md`
- ⬜ Repo ve model seçiminin kullanıcı tarafından onaylanması
- ⬜ SoccerNet-GAR erişim başvurusu (HF + NDA)
- ⬜ SN-GSR örnek etiketinin indirilip alanların doğrulanması
- ✅ Kuralların netleştirilmesi (D-004, D-006, D-008, D-009)
- ⬜ Python ve kütüphane kurulumu: repo seçiminden **sonra**, seçilen repoların sürümlerine göre (D-007)
- ⬜ Teknoloji yığınının kesinleştirilmesi → `09_DECISIONS.md`
- ⬜ Python ortamı, proje iskeleti, konfig dosyaları

## Faz 0.5 — 50 hücreli ızgara demosu (D-013)
- ✅ Ortam: `.venv` (Python 3.12), torch+CUDA, NBJW ağırlıkları
- ✅ `PitchKeypointDetector` arayüzü + NBJW adaptörü
- ✅ Homografi (RANSAC) + kamera hareketiyle takip (F-3)
- ✅ 50 hücre (`configs/grid.yaml`, `sgi.grid`) + birim testleri (7/7 geçti)
- ✅ Görselleştirme: perspektifli ızgara, hücre numaraları, mini harita, durum bilgisi
- ✅ `scripts/grid_demo.py process | play`
- ✅ Örnek videoda doğrulama (`iniesta_sample.mp4`)
- ✅ Ekibin klibiyle test (Lille–Galatasaray özeti, 8 dk). Yanlış kalibrasyon sorunu bulundu ve düzeltildi (D-017)
- ✅ Hız optimizasyonları: 3,7 → 4,5 kare/sn, sonuç birebir aynı (D-015)
- ✅ Canlı mod `live`: 25 fps gösterim (D-016)
- ✅ Fiziksel kamera kontrolü + birim testleri (18/18)
- ✅ Hızlı deneme döngüsü: saha noktası önbelleği (şimdilik kapalı) + test videosu `pas_test.mp4` (D-018)
- ✅ Kalite göstergeleri + hata kategorileri + rapor (D-019)
- 🟨 Çizgilere oturtma (D-020): kod ve sentetik testler hazır, `pas_test` üzerinde ölçüm bekliyor
- ✅ Çizgilere oturtma pas_test'te ölçüldü: OK %65,6 → %81,7, yaklaşık %18,9 → %3,7; görsel doğrulandı (D-020)
- ✅ Mini harita: tespit edilen çizgiler + hücre durumu (hesaplanamayan kırmızı, görüş dışı gri) (D-021)

## Faz 0.6 — Oyuncu / hakem / kaleci / top (D-022)
- ✅ YOLO11 (COCO) adaptörü, ByteTrack adaptörü, forma rengiyle takım/rol ataması
- ✅ Sahaya yerleştirme + hücre ataması, `objects.npz`
- ✅ Video ve mini harita işaretleri (takım renkleri, hakem mavi, top siyah)
- ✅ Canlı mod: arka planda YOLO, 23 fps
- ✅ pas_test tam işleme ve görsel kontrol; kaleci/hakem/kimlik değişimi düzeltmeleri (D-022)
- ✅ Hazır modellerle iyileştirme (D-024): BoT-SORT, NBJW TSWC ağırlıkları, roboflow/sports futbol dedektörü (top %73 → %80, kaleci/hakem modelden)
- ✗ roboflow saha noktası modeli: denendi, bu videoda 50–300 px hatalı, kullanılmıyor (D-025)
- ✅ Orta saha başlangıç kalibrasyonu (D-034): az nokta + çizgiler + fiziksel kamera; ilk ızgara 5,4 → 0,0 sn (çevrimdışı), 0,76 sn (canlı)
- ✅ Sahne kesmesi algılama (D-033): eski ızgara yakın çekime taşınmıyor
  - 2026-10-23 ölçüm: pas_test ilk 4,4 sn saha modeli karede 0–3 nokta buluyor (görünen: uzak taç çizgisi + orta yuvarlak parçası); eşik düşürmek ve çizgi ağı işe yaramadı. Izgara canlıda 5,4 sn'de geliyor
- ✅ Görsel: oyuncu ayak halkası kaldırıldı; etiket takım harfi (hakem H, kaleci KA/KB) + saha koordinatı (m); etiket takım renginde kutuda. Yeni takipte hakem ilk karelerde "?" yerine hemen H (oyların hepsi "diğer" ise)
- ✅ Kale tespiti (D-026): kadrajdayken %82, video + mini haritada
- ✅ **Sabit kamera konumu varsayımı** — uygulandı (D-027): ilk 20 sn %76 → %100, orta saha çözüldü. (kullanıcı fikri, 2026-10-22, standart saha ölçüleri ile): Yayın kamerası video boyunca aynı yerde durur, yalnızca döner ve zoom yapar. İyi karelerden kamera konumu bir kez kestirilirse, sonraki karelerde 8 yerine 3 bilinmeyen kalır (yatay açı, dikey açı, odak). 2–3 nokta (ör. orta yuvarlak + orta çizgi) yeter. Orta saha sorununa doğrudan çözüm adayı.
- ⬜ Görünür bölge önsel bilgisi (kullanıcı fikri): Görülen özelliklere göre (orta çizgi / ceza sahası / kale) kameranın sahanın neresine baktığı kısıtlanır. İmkânsız nokta kombinasyonları (ör. orta yuvarlak + karşı ceza sahası köşesi) elenir.
- ✅ Top takibi (D-028): kapı + aykırı eleme, oyuncuda/tahmin durumları, penaltı noktası filtresi. Yalnızca geçmiş kareler (interpolasyon yok)
- ✅ Nedensel işleme (D-028): çevrimdışı da canlı gibi; kamera konumu ön taraması kaldırıldı
- ✅ Oyuncu yönü (D-035): sırtı kameraya dönük sahipte kaybolan top 3 sn yerine 8 sn'ye kadar onda sayılır; poz yalnızca bu durumda tek kırpıntıda çalışır. pas_test'te bu durum oluşmuyor (en uzun kayıp 0,7 sn)
- ⬜ Oyuncu yönü/hızı ile pas yönü tahmini (pas tespiti aşamasında, gerekirse)
- ✅ Canlı mod akıcı nesne çizimi (D-030): kutular optik akışla takip, oyuncu hatası 26,7 → 9,2 px. Saha modelini seyrekleştirme denendi, ızgarayı bozdu (reddedildi)
- ✅ Canlı performans analizi (D-031): çizgi ağı kapatıldı (katkısı ~%1, GPU'nun yarısı), GPU işleri ayrı süreçte; nesne sonucu 3,5 → 6,2/sn
- ✅ TensorRT (D-036): saha modeli 108 -> 47 ms, dedektör 80 -> 63 ms; canlıda nesne gecikmesi 11,4 -> 3,8 kare
- ✅ Canlıda penaltı noktası (D-032): öğrenilen sabit işaret filtresi; kullanıcı canlıda doğruladı (2026-10-23)
- ✅ Canlıda top gecikmesi: TensorRT ile ort 3,8 kare (D-036)
- ✅ Zoom / canlı ızgara doğruluğu (D-029): canlı çizgi uyumu 0,77 → 0,90, ayrı süreçte çizgilere oturtma, zoom'da yeniden sabitleme, kayıpken hızlı kalibrasyon, `scripts/live_sim.py`
- ⬜ Yakın çekime geçişteki hızlı zoom-out (pas_test 23–24 sn): canlı ve çevrimdışı ikisinde de zayıf
- ⬜ Saha koordinatında takip (kimlik değişimleri)
- ⬜ Adım 2 (gerekirse): SN-GSR ile YOLO ince ayarı (rol + top), top takibi/interpolasyonu
- ✅ Yakın çekime geçişte (pas_test ~24 sn) yanlış kalibrasyon: sahne kesmesi algılama ile çözüldü (D-033)

### 50 hücre — ileride bakılabilecekler (kullanıcı kararı 2026-10-22: şimdilik yeterli, sadece not)
- ✅ (D-034) Orta saha görüntülerinde başlangıç kalibrasyonu: orta yuvarlak + orta çizgiden ek nokta (pas_test ilk 5 sn, %11)
- ⬜ 4 m'den büyük kaymaların düzeltilmesi (pas_test kare 880; çok ölçekli / kaba-ince oturtma)
- ⬜ Önbelleğin doğrulanması (önbellekli ve önbelleksiz çıktılar birebir aynı mı, veri kaybı var mı)
- ⬜ Çim şeridi haritası ile kayma düzeltme (14 §4.4) ve hareket kestiriminde çim maskesi (14 §4.1)
- ✗ Çevrimdışı iki yönlü F-3 taşıması: D-028 ile iptal (gelecek kare kullanılmaz)
- ⬜ SN-GSR üzerinde yer gerçeğiyle hücre doğruluğu ölçümü (14 §2.1). Rapor için gerçek skor
- ✅ Canlı modda ince ayar maliyeti: ayrı süreç + hızlı ayar (D-029)
- ⬜ 8 dk'lık tam klibin son sürümle işlenmesi
- ⬜ 8 dk'lık klibin tam yeniden işlenmesi: en sona bırakıldı
- ⬜ Kalite göstergeleri + hata kategorileri E1–E7 + zaman çizelgesi (`14_CALIBRATION_QUALITY.md` §2.2, §3)
- ⬜ Çim maskesi (§4.1)
- ⬜ SN-GSR değerlendirmesi: hücre doğruluğu ve konum hatası (§2.1, `scripts/eval_calibration.py`)
- ⬜ Şerit haritası ile F-3 kayma düzeltmesi, `PitchRefiner` (§4.4)

## Faz 1 — İlk kilometre taşı: uçtan uca prototip (§11)
Hedef: **Tek kısa klipte** (10–30 sn) akışın tamamını çalıştırmak ve izlenebilir bir pas kaydı üretmek. Kalite ikinci planda kalır, önemli olan tüm halkaların bağlı olmasıdır.
1. ✅ M1 Video I/O + kesit seçimi
2. ✅ M2 Hazır veya ince ayarlı model ile oyuncu/top/hakem tespiti
3. ✅ M2 Saha keypoint tespiti
4. ✅ M3 ByteTrack + top interpolasyonu
5. ✅ M3b Takım ataması (renk kümeleme)
6. ✅ M4 Homografi + kalite bayrağı. **Doğrulama:** izdüşürülmüş ızgara overlay
7. ✅ M5 50 hücre ataması + birim testleri
8. ✅ M6 Sahiplik + pas adayları (D-037)
9. ✅ M7 Kural motoru: R-OUT, R-1…R-4, R-3T, F-1/F-2/F-4/F-5 (D-037)
10. ✅ M8 `passes.json`, `summary.json`, `passes_overlay.mp4` (D-037)
11. 🟨 Klipteki pasların elle etiketlenmesi ve ilk karşılaştırma (kullanıcıdan liste bekleniyor)

## Faz 2 — Doğrulama ve iyileştirme
- ⬜ Tespit modeli değerlendirmesi (mAP, özellikle top recall)
- ⬜ Homografi hatası ölçümü
- ⬜ Takım ataması doğruluğu
- ⬜ Pas adayı tespiti precision/recall (elle etiketli olaylara göre)
- ⬜ `pass_rules.yaml` v1.0 yazılması ve birim testleri (kurallar tamamlandı)
- ⬜ Yedek kuralların (F-1…F-5) uygulanma sıklığının ölçülmesi → `13_LIMITATIONS.md`

## Faz 3 — Genişletme ve teslim
- ⬜ Daha fazla klip ile değerlendirme
- ⬜ Hata analizi (hata hangi modülde oluşuyor?)
- ⬜ Rapor ve sunum materyali, demo videosu

## Oturum günlüğü
| Tarih | Yapılan | Sonraki adım |
|---|---|---|
| 2026-10-21 | Şartname okundu, docs/ oluşturuldu | GitHub repo incelemesi |
| 2026-10-21 | 10×5 ızgara onaylandı, pas kuralları tanımlandı (D-004, D-006) | Kalan açık sorular |
| 2026-10-21 | Belirsiz sınıfı kaldırıldı, her vuruş pas, yedek kurallar F-1…F-5, 13_LIMITATIONS.md (D-008…D-010) | — |
| 2026-10-21 | F-1…F-5 onaylandı, F-3 kamera hareketi tabanlı oldu (D-012). Model bağımsız mimari eklendi (D-011) | — |
| 2026-10-21 | SoccerNet (22 repo) incelendi. GSR, NBJW, GAR ve Tracking öne çıktı | — |
| 2026-10-22 | Faz 0.5: ortam kuruldu, 50 hücreli ızgara demosu çalışıyor (D-013) | Ekibin klibiyle test, sonra Faz 1 |
| 2026-10-22 | Klip testi. Profil + optimizasyon (D-015), canlı mod (D-016), yanlış kalibrasyon düzeltmesi (D-017). Not: ölçüm sırasında klibin ilk tam çıktısının üzerine yazıldı; kısmi çalıştırmalar artık alt klasöre yazıyor | Tam yeniden işleme sonucu → kalite göstergeleri (14 §2.2) |
| 2026-10-22 | Nedensel işleme + top takibi (D-028), canlı zoom/ızgara doğruluğu (D-029) | Canlı modda akıcı oyuncu çizimi, poz modeli |
| 2026-10-23 | Penaltı noktası öğrenilen işaret (D-032), sahne kesmesi (D-033), az noktadan başlatma (D-034), oyuncu etiketleri | Poz modeli |
| 2026-10-23 | Canlı performans analizi, çizgi ağı kapatıldı, GPU alt süreci (D-031) | Penaltı noktası, TensorRT, hücre düzeni kararı |
| 2026-10-23 | Canlı akıcı nesne çizimi: optik akış + hız + sönümleme (D-030), `scripts/live_obj_sim.py` | Poz modeli (oyuncu yönü) |
| 2026-10-23 | Poz modeli: topu gizleyen sırtı dönük oyuncu (D-035) | TensorRT denemesi, sonra pas tespiti |
| 2026-10-23 | TensorRT: motor derleme, karşılaştırma, canlı test (D-036) | Pas tespiti (Faz 1: sahiplik + pas adayları) |
| 2026-10-23 | Görsel etiket kutuları + hakem düzeltmesi; pas tespiti ve kural motoru (D-037), tam video yeniden işlendi | Elle etiketli paslarla doğruluk ölçümü |
| 2026-10-28 | Hoca geri bildirimi (görsel düzenlemeler) 1: hücre numaralandırması sol alttan, sütun sütun yukarı (D-038) | Hocanın diğer görsel geri bildirimleri |
| 2026-10-28 | Hoca geri bildirimi 2: bilgi yazıları, mini harita ve pas paneli videonun sağında/altında, 1920x1080 tuval (D-039). Ardından alt şerit kaldırıldı, video büyütüldü (1920x810), beyaz ızgara çizgileri varsayılan kapalı; mini harita yukarı alındı, saha çizgileri belirginleştirildi; dolgu 0,10, sınır çizgileri 0,45 opaklık |
| 2026-10-28 | Hoca geri bildirimi 3: hedef takım modu (`--team A`), ikinci video `passes_overlay_A.mp4`; canlı modda pas sayımı; dedektör eşikleri kare aralığıyla (D-040) | Hedef takım seçim yöntemi (Q17); canlı pas sayımının çevrimdışıya yaklaştırılması | Hocanın diğer görsel geri bildirimleri |
