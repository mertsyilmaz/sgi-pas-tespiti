# 13 — Varsayımlar ve Bilinen Kısıtlar

> Bu dosya **final proje raporunda** "Varsayımlar ve Kısıtlar" bölümüne doğrudan aktarılacaktır. Kullanıcının talebiyle (2026-10-21): belirsiz olabilecek durumlar için sabit kurallar belirlenmiştir. Bu kuralların hata olasılığı taşıdığı **bilinmekte ve kabul edilmektedir**.

## 1. Şartnameden bilinçli sapmalar (kullanıcı onaylı)
| # | Şartname | Projede | Gerekçe |
|---|---|---|---|
| S-1 | §6.6: belirsiz olaylar ayrı raporlanmalı | Belirsiz sınıfı yok. Yalnızca SUCCESS / FAIL | Kullanıcı kararı (D-008): sabit kurallarla deterministik ikili sonuç |
| S-2 | §5: pas; şut, uzaklaştırma, sekme ve sapmadan ayrılmalı | Her vuruş pas kabul edilir | Kullanıcı kararı (D-009): kapsam sadeleştirme |
| S-3 | §8: toplam ve oranlar | Takım bazında yalnızca başarılı ve başarısız sayıları raporlanır | Kullanıcı kararı (D-008) |

## 2. Sabit kurallar ve olası hata kaynakları
| Kural | Olası hata |
|---|---|
| Her vuruş pas (D-009) | Şutlar ve uzaklaştırmalar çoğunlukla FAIL olarak sayılır ve başarısız pas sayısını artırır |
| Yalnızca hedef hücre (B.2) | Hücre sınırına yakın alımlarda, komşu hücredeki alıcı sayılmaz ve pas FAIL olur |
| R-3T: eşit mesafede SUCCESS | Ölçüm hatası sınırındaki çekişmeler pasör takımı lehine sonuçlanır |
| R-4: hedef hücre boş ise FAIL | Oyuncu tespit edilemediyse (örtülme, kadraj dışı) doğru pas FAIL sayılabilir |
| F-2: top kaybolursa son görülen konum | Kamera kesmesi veya tekrar gösterimlerinde hedef hücre yanlış olabilir |
| F-3: kamera hareketiyle homografi taşıma | Zincirleme kestirimde hata birikir (drift). Kalabalık veya hareketli tribün eşlemeyi bozabilir. Sahne kesmesinde en yakın homografiye düşülür ve hücre kayabilir |
| F-4: takımı belirlenemeyen oyuncu yok sayılır | Hedef hücredeki tek alıcı yok sayılırsa sonuç değişir |
| F-5: release anında en yakın oyuncu pasör | Kalabalık pozisyonlarda pasör yanlış atanabilir |
| Dribling istisnası | Oyuncunun topu önüne atıp tekrar alması pas sayılmaz. Bu davranış istenen yöndedir ama belgelenmiştir |

## 3. Görüntü işleme kaynaklı kısıtlar
- **Rol ve takım ataması (D-022):** Hazır COCO modeli yalnızca "kişi" tespit eder. Takım, hakem ve kaleci ayrımı forma rengi kümelemesi ve konum sezgisiyle yapılır. Hakem forması bir takıma benziyorsa ya da iki kalecinin rengi aynıysa hata olasılığı vardır. Takım A/B adı daha koyu formaya göre verilir.
- **Top tespiti:** Top küçük ve hareket bulanıklığı fazla olduğu için bazı karelerde bulunamaz (pas_test ilk 24 sn: karelerin %69'unda bulundu).
- **Nedensel işleme (D-028):** Hem canlı hem çevrimdışı modda yalnızca o ana kadarki kareler kullanılır. Böylece ölçülen başarı canlı performansı yansıtır. Bedeli: videonun başında kamera konumu henüz öğrenilmemiştir. Ayrıca kaybolan top geriye dönük doldurulmaz.
- **Top kestirimi (D-028), sabit kurallar:**
  - Kaybolan top, son görüldüğünde dibinde (≤ 1,5 m) oyuncu varsa en fazla 3 sn o oyuncuda sayılır. Top bu sürede oyuncudan ayrıldıysa (görünmeyen hızlı pas) yanlış oyuncuda gösterilir; yeniden görülünce düzelir.
  - Sahipsiz kaybolan top, en fazla 0,5 sn son hızıyla ilerletilir.
  - Penaltı noktası / orta nokta üzerinde duran gerçek top (ör. penaltı atışı öncesi) vurulana kadar görülmez.
- **Havadaki top:** Homografi topun yere izdüşümünü değil, görüntüdeki konumunu dönüştürür. Bu yüzden havadaki topun hücresi hatalı olabilir. Hedef hücre, top alındığında veya yere indiğinde belirlendiği için etkisi sınırlıdır. Ara hücreler sınıflandırmayı etkilemez.
- **Kamera:** Yayın kamerası sahanın tamamını göstermez. Kadraj dışındaki oyuncular hedef hücrede "yok" sayılır.
- **Canlı mod doğruluğu:** Canlı modda model sonucu birkaç kare geç gelir ve kamera hareketiyle telafi edilir. D-029 sonrası çizgi uyumu medyanı canlıda 0,90, çevrimdışında 0,96 (pas_test 20–38 sn, zoom ve yakın çekim içeren bölüm). Canlı doğruluk `scripts/live_sim.py` ile tekrarlanabilir şekilde ölçülür.
- **Canlı nesne çizimi (D-030):** Canlıda nesne sonucu saniyede ~3,5 kez, ~8 kare geç gelir. Aradaki karelerde oyuncu kutuları optik akışla ilerletilir (yalnızca gösterim). Top ilerletilmez, ~8 kare geriden çizilir (ayağındaki oyuncuyla taşınması hariç).
- **Kalibrasyon kapsamı:** Yanlış ızgara göstermemek için katı doğrulama uygulanır (D-017). Az saha çizgisi görünen orta saha görüntülerinde ızgara gizlenebilir. Çim şeridi fikri (14 §4.4) bu boşluğu hedefler.
- **Perspektif hassasiyeti:** Bir piksellik hata yakın tarafta birkaç cm'ye, uzak taç çizgisi tarafında 0,3–0,5 m'ye karşılık gelir. Uzak taraftaki hücre sınırları daha az hassastır.
- **Lens bozulması:** Düzlemsel homografi lens bozulmasını modellemez. Çok geniş açıda görüntü kenarlarında küçük sapmalar olabilir.
- **Kale arkası kamera belirsizliği:** Kale arkasından çekilen görüntülerde, bağlam olmadan sahanın sol mu sağ mı tarafı olduğu ayırt edilemez (sn-calibration'da belgelenmiş). Bu durumda F-3 (kamera hareketiyle taşıma) bağlam sağlar. Sahne kesmesinden sonra hata olasılığı vardır.
- **Saha ölçüsü:** Varsayılan 105×68 m kabul edilir. Gerçek saha ölçüsü farklıysa hücre sınırları orantılı olarak kayar.
- **Pas tespiti (D-037):** Takip kutusu yandaki oyuncuya atlarsa (kimlik karışması) top sürme sahte pas olarak çıkabilir; o oyuncunun takımı da yanlış görünebilir. Görüntüde üst üste duran iki oyuncudan hangisinde top olduğu tek kameradan ayırt edilemeyebilir. Top sahibinden 1,5 m'den fazla uzaklaşmadan el değiştirirse (ör. top kapma) pas kaydı oluşmaz.
- **Canlı modda pas sayımı (D-040):** Canlı modda nesne sonuçları ~2 karede bir ve birkaç kare geç gelir; takip de seyrek çalışır. Bu yüzden canlı sayaç çevrimdışı sonuçtan farklıdır (pas_test ilk 30 sn: çevrimdışı 11 pas, canlı 5; ortak 4). Özellikle 0,1–0,4 sn süren kısa paslar ve hızlı el değiştirmeler canlıda kaçabilir. Raporlanan sayılar çevrimdışı çıktıdan (`passes.py`) alınır; canlı sayaç gösterim amaçlıdır.
- **TensorRT (D-036):** Motorlar bu GPU'ya/TensorRT sürümüne özeldir; `models/trt/` yoksa sistem PyTorch ile (daha yavaş) çalışır. TensorRT sonuçları PyTorch ile birebir aynı değildir (homografi farkı medyan 0,18 m); ölçülen doğruluk aynı.
- **Oyuncu yönü (D-035):** Sırtı dönük sahipte top en fazla 8 sn onda sayılır. Yön poz modelinin sol/sağ omuz-kalça sırasından okunur; 40 px'ten küçük oyuncularda çoğu zaman karar verilemez, o zaman uzatma yapılmaz. Sırtı numaralı oyuncuyu zaman zaman "önü dönük" sanıyor. Yer gerçeğiyle ölçülmedi.
- **Yakın çekim / başka kamera (D-033):** Sahne kesmesinden sonra gelen yakın çekimlerde (çim seviyesi kamera, saha çizgisi yok) ızgara ve saha koordinatı yoktur. Ana kameranın önceki kareleri bu görüntüye geometrik olarak taşınamaz.
