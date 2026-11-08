# 01 — Proje Özeti

> Kaynak: `00_SPEC_ORIGINAL.md`. Bu dosya şartnamenin Türkçe, uygulamaya dönük özetidir. Çelişki olursa orijinal metin geçerlidir.

## Bağlam
- **Ders:** Sayısal Görüntü İşleme (Yüksek Lisans), dönem projesi
- **Proje adı:** Futbol Pas Tespiti ve Analiz Sistemi
- **Ana hedef:** Maç videosunda pasları, topun ve oyuncuların **50 hücreli saha ızgarası** üzerindeki hareketine ve **açıkça tanımlanmış kurallara** göre **başarılı / başarısız** olarak sınıflandırmak. (Belirsiz sınıfı kullanılmaz, bkz. D-008.)

> ⚠️ Şartnameden kullanıcı onaylı sapmalar vardır: belirsiz sınıfı yok, her vuruş pas sayılır. Ayrıntılar `13_LIMITATIONS.md` dosyasında. Kesin kurallar `05_PASS_RULES.md` dosyasında.

> Asıl amaç nesne tespiti değildir. Tespit, takip ve koordinat dönüşümü, kural motoruna güvenilir **gözlem** sağlamak için kullanılır.

## Tespit edilecekler
| Nesne | Not |
|---|---|
| A Takımı oyuncuları | Her karede tespit, kareler arası takip |
| B Takımı oyuncuları | Her karede tespit, kareler arası takip |
| Kaleci | Takımına atanmalı. Forma rengi farklı olduğu için ayrıca ele alınır |
| Hakem | **Hiçbir takıma atanmamalı** |
| Top | Oyunculardan bağımsız konum. Örtülme, bulanıklık ve küçük boyut dikkate alınmalı |
| Saha çizgileri ve anahtar noktalar | Yalnızca uzamsal eşleme için. Pas kararını tek başına belirlemez |

**Kimlik ayrımı:** Bir oyuncunun `track_id` (takip kimliği) ve `team_id` (takım kimliği) bilgileri **ayrı** tutulur.

## Uzamsal temsil
1. Görüntü koordinatları **homografi** ile saha koordinatlarına (metre) dönüştürülür.
2. Kamera açısı değişince homografi güncellenir.
3. Homografi güvenilir değilse ilgili kare `homography_valid=False` olarak işaretlenir. Pas kararında yedek kural F-3 uygulanır.
4. Saha **tam olarak 50 hücreye** bölünür. Hücre sayısı izinsiz değiştirilemez.
5. Hücreler görüntü bölgelerini değil, **saha konumlarını** temsil eder.

## Pas mantığı
- Pas = sahiplik geçişi (possession transition) + top hareketi + zamansal kanıt.
- Bir pas kaydında şunlar bulunur: pasör, pasörün takımı, **başlangıç hücresi**, **ara hücreler** (gerçek zaman sırasıyla), **hedef hücre**, alıcı, alıcının takımı, sınıf, gerekçe ve güven bilgisi.
- Dribling, uzaklaştırma, şut, sekme ve sapma gibi olaylar mümkün olduğunca pastan ayırt edilir.
- Ara hücre listesine, top o hücreye girmediyse geometrik olarak arada kalsa bile hücre **eklenmez**.
- Topun kısa süreli kaybolması pasın başarısız olduğu anlamına **gelmez**.

## Sınıflandırma
- **Successful:** Tanımlı başarı koşulları kanıtla sağlanmışsa
- **Unsuccessful:** Tanımlı başarısızlık koşulu güvenilir kanıtla sağlanmışsa
- **Uncertain:** Kanıt yetersizse. Ayrı raporlanır ve diğer sayıları şişirmez.

Kurallar **konfigürasyonda açıkça** tanımlanır (bkz. `05_PASS_RULES.md`). **Tanımlanmamış kural uydurulmaz**. Böyle kurallar "unresolved" olarak belgelenir.

## Çıktılar
- Pas başına yapılandırılmış kayıt (bkz. `06_DATA_MODEL.md`)
- Özet istatistikler: **Takım A** ve **Takım B** için başarılı ve başarısız pas sayıları
- Sayım kuralları açık olmalı, aynı pas birden fazla sayılmamalı
- Görselleştirme: anotasyonlu video ve 2B saha görünümü

## Geliştirme ilkeleri
- Modüller ayrı tutulur ve her birinin girdi/çıktısı tanımlıdır, böylece bağımsız test edilebilir.
- Geliştirme artımlı ilerler: tespit ve eşleme doğrulanmadan pas sınıflandırmasına güvenilmez.
- Tespit, takip, eşleme ve kural ihlali hataları ayrı ayrı izlenebilir olmalıdır.
- Elle doğrulanmış örneklerle değerlendirilmeden "doğru sınıflandırıyor" denmez.
- **İlk kilometre taşı:** Kısa bir video kesitinde uçtan uca akışın tamamı çalışmalıdır (bkz. `10_ROADMAP.md`).
- Kapsam dışı özellik eklenmez.
