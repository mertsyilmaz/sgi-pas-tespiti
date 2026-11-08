# CLAUDE.md

Futbol Pas Tespiti ve Analiz Sistemi — Sayısal Görüntü İşleme (Yüksek Lisans) dönem projesi.
Kullanıcı ile **Türkçe** iletişim kurulur.

## Her oturumun başında
1. `docs/README.md` → doküman dizini
2. `docs/10_ROADMAP.md` → mevcut durum ve sıradaki adım
3. `docs/12_OPEN_QUESTIONS.md` → çözülmemiş konular

## Değişmez kurallar (şartnameden)
- Saha **tam olarak 50 hücreye** bölünür. Bu sayı izinsiz değiştirilmez.
- Izgara 10 sütun × 5 satırdır (D-004).
- Pas sonucu yalnızca **hedef hücredeki** oyunculara göre belirlenir. **Ara hücreler değerlendirilmez** (D-006, `05_PASS_RULES.md` §B.3).
- Pas kuralları **uydurulmaz**. Tanımsız ayrıntılar `12_OPEN_QUESTIONS.md` içinde açık kalır.
- Kurallar yalnızca `configs/pass_rules.yaml` + `src/sgi/rules/` içinde bulunur. Tespit ve takip koduna gömülmez.
- Sınıflar **yalnızca SUCCESS / FAIL**. Belirsiz sınıfı yok. Eksik veri sabit yedek kurallarla (F-1…F-5) çözülür (D-008).
- Her vuruş pas sayılır, şut/orta ayrımı yok (D-009). Çıktı: A ve B için başarılı ve başarısız sayıları.
- Şartnameden sapmalar ve kısıtlar `docs/13_LIMITATIONS.md` dosyasında tutulur. Yeni bir sabit kural eklenirse oraya da yazılır.
- `track_id` ile `team_id` ayrı tutulur. Hakem hiçbir takıma atanmaz.
- Elle doğrulanmış örneklerle ölçülmeden "doğru çalışıyor" denmez.
- Kapsam dışı özellik eklenmez.

## Mimari ilkeler
- **Model bağımsız (D-011):** Tespit, keypoint, takip, takım ataması ve kamera hareketi bileşenleri `src/sgi/interfaces.py` arayüzleri ve `adapters/` ile uygulanır. Seçim konfigden yapılır (`registry.py`). Pipeline kodunda belirli bir modele veya kütüphaneye (ultralytics vb.) doğrudan bağımlılık **olmaz**. Model sınıfları `class_map` ile kanonik sınıflara çevrilir.
- Keypoint yetersizse homografi kamera hareketiyle taşınır (F-3, D-012).
- **Nedensel işleme (D-028):** Hiçbir adım ileriki kareleri kullanmaz (iki yönlü interpolasyon, videoyu önceden tarama yok). Çevrimdışı mod canlıdaki gibi çalışır; ölçülen başarı canlı performansı yansıtmalıdır.

## Ortam ve çalıştırma
- Python: `.venv/Scripts/python.exe` (3.12). `python` komutu Store yönlendirmesine düşer; venv dışında `py -3.12` kullanılır.
- Testler: `.venv/Scripts/python.exe -m pytest -q tests`
- Izgara demosu:
  - Çevrimdışı: `.venv/Scripts/python.exe scripts/grid_demo.py process <video> [--show] [--no-video]`, ardından `... play <video>`
  - Canlı: `... live <video>`. Penceresiz test: `... live <video> --headless-seconds 60`
  - Canlı doğruluk (tekrarlanabilir, makine yükünden bağımsız): `.venv/Scripts/python.exe scripts/live_sim.py <video> --start 20 --duration 18`. Gerçek canlı test fps'i ısınma/yük ile 18–24 arasında oynar; varyant karşılaştırması simülasyonla yapılır.
  - Canlı nesne çizimi (D-030): `.venv/Scripts/python.exe scripts/live_obj_sim.py <video> outputs/grid_demo/pas_test/s28_d20/objects.npz`
- Pas tespiti (D-037): `.venv/Scripts/python.exe scripts/passes.py <objects.npz> [--video]` -> aynı klasörde passes.json, summary.json, passes_overlay.mp4 (orijinal videodan, sağ panelli 1920x810; D-039). Hedef takım (D-040): `--team A` -> summary_A.json, passes_overlay_A.mp4; `play`/`live` da `--team A` alır
- TensorRT (D-036): motorlar `.venv/Scripts/python.exe scripts/build_trt.py` (~10 dk, `models/trt/`), karşılaştırma `scripts/bench_trt.py <video> --what kp|det --engine <motor>`. Motor yoksa PyTorch kullanılır.
- Regresyon: `scripts/compare_calibration.py <referans npz> <yeni npz>`. Referans (`outputs/baseline/`) 2026-10-22'de temizlendi; yeniden üretilecek. Profil: `scripts/profile_pipeline.py`.
- **Geliştirme/test videosu:** `data/videos/pas_test.mp4` (64,6 sn, kullanıcının seçtiği; D-018 güncellemesi). Saha noktası önbelleği **varsayılan kapalı**; kullanıcı onaylayana kadar `--use-cache` kullanma. 5 dakikadan uzun bir çalıştırma başlatmadan önce süreyi ve gerekçeyi kullanıcıya söyle.
- Kısmi çalıştırmalar (`--start/--duration`) `outputs/grid_demo/<video>/s<start>_d<dur>/` altına yazar. Tam çıktıların üzerine yazma.
- Model ağırlıkları `models/nbjw/`, NBJW kaynak kodu `third_party/nbjw/` (GPL-2.0), çıktılar `outputs/` altında (git dışı).

## Dokümantasyon disiplini
- Teknik karar → `docs/09_DECISIONS.md`
- İlerleme → `docs/10_ROADMAP.md` (oturum günlüğü dahil)
- Repo veya veri seti bulgusu → `docs/08_REPO_ANALYSIS.md`, `docs/07_DATASETS.md`
- `docs/00_SPEC_ORIGINAL.md` dosyası **düzenlenmez**.
