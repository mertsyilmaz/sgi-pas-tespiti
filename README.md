# Futbol Pas Tespiti ve Analiz Sistemi

Sayısal Görüntü İşleme (YL) dönem projesi. Tasarım dokümanları: [docs/README.md](docs/README.md).

## Kurulum (Windows, NVIDIA GPU)
```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\python -m pip install -r requirements.txt
git clone --depth 1 https://github.com/mguti97/No-Bells-Just-Whistles.git third_party/nbjw
# NBJW ağırlıkları (TS-WorldCup ince ayarlı) -> models/nbjw/SV_FT_TSWC_kp, models/nbjw/SV_FT_TSWC_lines
#   https://github.com/mguti97/No-Bells-Just-Whistles/releases/download/v1.0.0/SV_FT_TSWC_kp
#   https://github.com/mguti97/No-Bells-Just-Whistles/releases/download/v1.0.0/SV_FT_TSWC_lines
# Futbol oyuncu/top dedektörü (roboflow/sports, MIT) -> models/roboflow_sports/
.venv\Scripts\pip install ultralytics gdown
.venv\Scripts\gdown -O models/roboflow_sports/football-player-detection.pt "https://drive.google.com/uc?id=17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q"
```

## 50 hücreli ızgara demosu

### Canlı (önceden işleme yok)
```powershell
.venv\Scripts\python scripts\grid_demo.py live data\videos\mac.mp4
#   başlangıç saniyesi: --start 260
```
Video normal hızda oynar ve ızgara anlık hesaplanır. Saha noktası modeli arka planda çalışır, aradaki kareler kamera hareketiyle güncellenir. Çevrimdışı moda göre biraz daha az hassastır (medyan ~0,65 m fark).

### Sadece oyuncu/hakem/top adımını yeniden çalıştırma
Kalibrasyon (`calibration.npz`) zaten varsa saha noktası modeli tekrar çalıştırılmaz (~90 sn / 1 dk video):
```powershell
.venv\Scripts\python scripts\grid_demo.py objects data\videos\mac.mp4
```

### Çevrimdışı (ölçüm ve rapor için)
```powershell
# 1) Videoyu işle (RTX 3050 ile ~4.5 kare/sn). --show ile işlerken izlenebilir, --no-video ile MP4 üretilmez.
.venv\Scripts\python scripts\grid_demo.py process data\videos\mac.mp4 --show
#    Sadece bir bölüm için:  --start 120 --duration 30   (saniye) -> outputs/grid_demo/<video>/s120_d30/

# 2) Gerçek hızda oynat
.venv\Scripts\python scripts\grid_demo.py play data\videos\mac.mp4
```
`--use-cache` verilirse saha noktası modeli her video için bir kez çalışır ve sonuçlar `outputs/cache/keypoints/` altında saklanır. Aynı videonun sonraki işlemeleri modeli atlar (~8x hızlı). Doğrulama tamamlanana kadar varsayılan olarak kapalıdır.

Çıktılar `outputs/grid_demo/<video adı>/` altına yazılır:
- `grid_overlay.mp4`: Izgaralı video. Herhangi bir oynatıcıda (VLC önerilir) açılabilir.
- `calibration.npz`: Kare başına homografi ve kalite bilgisi.

Oynatıcı tuşları:

| Tuş | İşlev |
|---|---|
| `SPACE` | Duraklat |
| `a` / `d` | Bir kare geri / ileri |
| `j` / `l` | 5 sn geri / ileri |
| `g` | Izgara |
| `f` | Dolgu |
| `n` | Hücre numaraları |
| `m` | Mini harita |
| `c` | Mini haritada tespit edilen çizgiler (sarı) |
| `o` | Oyuncu / hakem / top işaretleri |
| `k` | Saha noktaları |
| `h` | Bilgi satırı |
| `q` | Çıkış |

Bilgi satırındaki kalibrasyon kaynağı:
- **Yeşil:** Saha noktalarından hesaplandı.
- **Sarı:** Kamera hareketiyle tahmin edildi (F-3).
- **Kırmızı:** Kalibrasyon yok. Izgara bilerek gizlenir, yanlış hücre gösterilmez.

## Testler
```powershell
.venv\Scripts\python -m pytest -q tests
```
