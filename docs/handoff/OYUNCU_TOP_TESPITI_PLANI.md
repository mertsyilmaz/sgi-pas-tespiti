# Futbol Videosunda Oyuncu, Kaleci, Hakem ve Top Tespiti — Proje Planı

Bu doküman, futbol maçı yayın videolarında **oyuncu, kaleci, hakem ve topu** bulan bir nesne tespit modelinin sıfırdan nasıl hazırlanacağını adım adım anlatır. Bağımsız bir projedir. Başka bir kod tabanı bilmeniz gerekmez.

---

## 0. Hedef ve teslim edilecekler

**Hedef:** Yayın kamerasından alınmış (geniş açı, hareketli kamera) bir futbol videosunun her karesinde aşağıdaki 4 sınıfı kutu (bbox) olarak bulan bir model.

| Sınıf no | Sınıf adı | Açıklama |
|---|---|---|
| 0 | `ball` | Top |
| 1 | `goalkeeper` | Kaleci (iki takımın kalecisi) |
| 2 | `player` | Kaleci dışındaki oyuncular (iki takım; takım ayrımı **yapılmaz**) |
| 3 | `referee` | Orta hakem ve yan hakemler |

> Takım ayrımı (A/B) bu modelin işi değildir. Model yalnızca "oyuncu" der, takım başka bir adımda forma renginden bulunur.

**Proje sonunda teslim edilecekler:**
1. `best.pt` — eğitilmiş model ağırlığı (Ultralytics YOLO formatı)
2. Sınıf listesi: `{0: ball, 1: goalkeeper, 2: player, 3: referee}` (sıra değişirse açıkça yazılmalı)
3. Eğitim çözünürlüğü (`imgsz`, ör. 1280) ve önerilen güven eşikleri (oyuncu ve top için ayrı)
4. Kullanılan Ultralytics ve PyTorch sürümleri (`pip freeze > requirements.txt`)
5. Kısa bir sonuç raporu: test setinde sınıf bazında precision / recall / mAP50, özellikle **topun recall değeri**
6. `data.yaml` ve eğitim komutu (sonucun tekrarlanabilmesi için)

---

## 1. Genel yaklaşım (neden bu yol?)

- Her kareyi elle etiketlemek **gereksiz**. 25 fps videoda ardışık kareler neredeyse aynıdır. 750 karelik bir klip, modele ancak 30–50 farklı kare kadar bilgi verir.
- Bunun yerine **hazır, etiketli ve açık bir veri seti** kullanılır: **SoccerNet SN-GSR** (yaklaşık 160 klip × 30 sn, her karede oyuncu/kaleci/hakem/top kutuları).
- Bunun yanına **kendi videolarımızdan küçük bir etiketli set** hazırlanır:
  - **Test seti (zorunlu):** Modelin bizim videolarımızda ne kadar iyi olduğunu ölçmek için. Eğitimde **asla** kullanılmaz.
  - **İnce ayar seti (gerekirse):** Model bizim videolarımızda topu kaçırırsa, birkaç yüz kareyle ek eğitim yapılır.
- Model: **YOLO11** (Ultralytics). Hazır COCO ağırlıklarından başlanıp (transfer öğrenme) futbol verisiyle eğitilir.

Akış:

```
SN-GSR indir ──> YOLO formatına dönüştür ──> YOLO11 eğit ──> kendi test setinde ölç
                                                                  │
                                         top zayıf mı? ──evet──> kendi karelerinle ince ayar ──> tekrar ölç
                                                                  │
                                                                hayır ──> teslim
```

---

## 2. Ortamın kurulması

### 2.1 Donanım
| Seçenek | Ne zaman | Not |
|---|---|---|
| Kendi bilgisayarın (NVIDIA GPU, ≥ 8 GB) | En rahatı | 4–6 GB GPU'da da olur, ama küçük model + küçük batch gerekir |
| **Kaggle Notebooks** | GPU yoksa önerilen | Ücretsiz, haftada ~30 saat GPU (T4/P100), 20 GB disk, oturum 12 saate kadar |
| Google Colab | Alternatif | Ücretsiz sürümde oturum sık kopar, disk ve süre sınırlı |

### 2.2 Yazılım
1. **Python 3.10–3.12** kur.
2. Proje klasörü ve sanal ortam oluştur:
   ```bash
   mkdir futbol_tespit && cd futbol_tespit
   python -m venv .venv
   # Windows: .venv\Scripts\activate    Linux/Mac: source .venv/bin/activate
   ```
3. GPU'lu PyTorch kur (komutu https://pytorch.org/get-started/locally/ sayfasından kendi CUDA sürümüne göre al), ardından:
   ```bash
   pip install ultralytics opencv-python tqdm
   ```
4. Kontrol:
   ```bash
   python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   yolo checks
   ```
   `True` görmelisin. `False` ise model CPU'da eğitilir (çok yavaş).
5. Yardımcı araç: **ffmpeg** (videodan kare çıkarmak için) — https://ffmpeg.org/download.html

### 2.3 Önerilen klasör yapısı
```
futbol_tespit/
├── raw/                 # indirilen zip'ler ve açılmış SN-GSR (eğitimden sonra silinebilir)
├── videos/              # kendi maç videoların
├── datasets/
│   ├── gsr_yolo/        # SN-GSR'nin YOLO formatına çevrilmiş hali
│   └── own/             # kendi etiketlediğin kareler (test + ince ayar)
├── scripts/             # dönüştürme ve yardımcı betikler
├── runs/                # Ultralytics eğitim çıktıları (otomatik oluşur)
└── README.md            # ne yaptığının kısa kaydı
```

---

## 3. Hazır veri seti: SoccerNet SN-GSR

### 3.1 İndirme
Sayfa: https://huggingface.co/datasets/SoccerNet/SN-GSR-2025

| Dosya | Boyut | Kullanım |
|---|---|---|
| `train.zip` | 9,8 GB | Eğitim |
| `valid.zip` | 11,2 GB | Doğrulama (eğitim sırasında ölçüm) |
| `test.zip` | 8,8 GB | İsteğe bağlı ek test / ek eğitim |
| `challenge.zip` | 5,3 GB | **İndirme**: etiketleri gizli, işe yaramaz |

Kesilirse kaldığı yerden devam eden indirme:
```bash
curl -L -C - -o raw/train.zip https://huggingface.co/datasets/SoccerNet/SN-GSR-2025/resolve/main/train.zip
```

> **Lisans:** SoccerNet verisi yalnızca araştırma/eğitim amaçlı kullanılabilir, başkalarına dağıtılmaz. Raporda kaynak olarak belirtilmelidir.

### 3.2 Yapıyı tanıma (dönüştürmeden önce mutlaka yap)
Zip açıldığında her klip ayrı bir klasördür (ör. `SNGS-060/`). Klasörün içinde kareler (`img1/000001.jpg` …) ve etiket dosyası `Labels-GameState.json` bulunur.

İlk iş olarak bir `Labels-GameState.json` dosyasını aç ve şunları kontrol et:
- `images`: her kare için `image_id`, `file_name`, `width`, `height`
- `annotations`: her kutu için `image_id`, `supercategory` (`object` = oyuncu/top vb.; `pitch` ve `camera` başka şeylerdir, bizim için gereksiz), `bbox_image` (`x`, `y`, `w`, `h` piksel), `attributes.role` (`player`, `goalkeeper`, `referee`, `ball`, `other`)
- `categories`: sınıf numarası → ad listesi

Hızlı inceleme betiği:
```python
import json, collections
d = json.load(open("raw/train/SNGS-060/Labels-GameState.json", encoding="utf-8"))
print(d.keys())
print(d["images"][0])
obj = [a for a in d["annotations"] if a.get("supercategory") == "object"]
print(obj[0])
print(collections.Counter((a.get("attributes") or {}).get("role") for a in obj))
```
> Alan adları burada yazılandan farklı çıkarsa, 3.3'teki betikte yalnızca ilgili satırları değiştir.

**Kontrol listesi:**
- [ ] `ball` rolünde kutular var mı, kaç tane? (Top her karede etiketli olmayabilir)
- [ ] Kutular piksel cinsinden mi, sol üst köşe + genişlik/yükseklik mi?
- [ ] Görüntü çözünürlüğü nedir (beklenen 1920×1080)?
- [ ] 5–10 kareyi kutularla birlikte çizip gözle kontrol et (3.4).

### 3.3 YOLO formatına dönüştürme
**YOLO formatı nedir?** Her görüntü için aynı adda bir `.txt` dosyası olur. Her satır bir kutudur:
```
<sınıf_no> <x_merkez> <y_merkez> <genişlik> <yükseklik>
```
Değerler görüntü boyutuna bölünmüştür (0–1 arası). Örnek: `0 0.512 0.430 0.006 0.011` (küçük bir top).

**Önemli kararlar:**
- **Her 5. kareyi al** (`STEP = 5`). Ardışık kareler çok benzer; hepsini almak diski doldurur, faydası azdır.
- **`other` rolünü at** (seyirci, yedek, kameraman vb.).
- **Bölmeyi klip bazında yap.** Aynı klibin kareleri hem eğitimde hem doğrulamada olmamalı; yoksa skorlar yapay olarak yüksek çıkar. SN-GSR bunu zaten `train/valid/test` olarak klip bazında ayırmıştır, bu ayrımı koru.

`scripts/gsr_to_yolo.py`:
```python
"""SN-GSR (Labels-GameState.json) -> Ultralytics YOLO formatı.
Kullanım: python scripts/gsr_to_yolo.py raw/train datasets/gsr_yolo train
"""
import json, shutil, sys
from collections import defaultdict
from pathlib import Path

ROLE2CLS = {"ball": 0, "goalkeeper": 1, "player": 2, "referee": 3}
STEP = 5  # her 5. kare

def convert(split_dir: Path, out_dir: Path, split: str) -> None:
    img_out = out_dir / "images" / split
    lbl_out = out_dir / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    counts = defaultdict(int)

    for seq in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        lab = seq / "Labels-GameState.json"
        if not lab.exists():
            continue
        data = json.loads(lab.read_text(encoding="utf-8"))
        cats = {c["id"]: c["name"] for c in data.get("categories", [])}

        boxes = defaultdict(list)
        for a in data["annotations"]:
            if a.get("supercategory") != "object" or "bbox_image" not in a:
                continue
            role = (a.get("attributes") or {}).get("role") or cats.get(a.get("category_id"))
            if role not in ROLE2CLS:
                continue
            b = a["bbox_image"]
            boxes[a["image_id"]].append((ROLE2CLS[role], b["x"], b["y"], b["w"], b["h"]))

        images = sorted(data["images"], key=lambda im: im["file_name"])
        for k, im in enumerate(images):
            if k % STEP:
                continue
            W, H = im.get("width", 1920), im.get("height", 1080)
            src = seq / "img1" / im["file_name"]
            if not src.exists():
                continue
            name = f"{seq.name}_{Path(im['file_name']).stem}"
            shutil.copy(src, img_out / f"{name}.jpg")
            lines = []
            for c, x, y, w, h in boxes.get(im["image_id"], []):
                x0, y0 = max(0.0, x), max(0.0, y)            # görüntü dışına taşan kutuları kırp
                x1, y1 = min(W, x + w), min(H, y + h)
                if x1 - x0 < 1 or y1 - y0 < 1:
                    continue
                lines.append(f"{c} {(x0+x1)/2/W:.6f} {(y0+y1)/2/H:.6f} {(x1-x0)/W:.6f} {(y1-y0)/H:.6f}")
                counts[c] += 1
            (lbl_out / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")

    print(split, "kutu sayıları:", {k: counts[v] for k, v in ROLE2CLS.items()})

if __name__ == "__main__":
    convert(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])
```
Çalıştırma:
```bash
python scripts/gsr_to_yolo.py raw/train datasets/gsr_yolo train
python scripts/gsr_to_yolo.py raw/valid datasets/gsr_yolo val
```
Beklenen: train ve val için birkaç bin görüntü. Top kutusu sayısı oyunculardan çok daha az olacaktır (karede 1 top, ~20 oyuncu); bu normaldir.

### 3.4 Gözle kontrol
Dönüştürülmüş birkaç görüntüyü kutularla çiz (`scripts/draw_labels.py`):
```python
import cv2, random, sys
from pathlib import Path
COLORS = {0: (0, 0, 255), 1: (0, 255, 255), 2: (0, 255, 0), 3: (255, 0, 0)}
root = Path(sys.argv[1])  # ör. datasets/gsr_yolo
imgs = random.sample(sorted((root / "images/train").glob("*.jpg")), 8)
for p in imgs:
    im = cv2.imread(str(p)); H, W = im.shape[:2]
    for line in (root / "labels/train" / f"{p.stem}.txt").read_text().splitlines():
        c, xc, yc, w, h = line.split(); c = int(c)
        xc, yc, w, h = float(xc)*W, float(yc)*H, float(w)*W, float(h)*H
        cv2.rectangle(im, (int(xc-w/2), int(yc-h/2)), (int(xc+w/2), int(yc+h/2)), COLORS[c], 2)
    cv2.imwrite(f"check_{p.stem}.jpg", im)
```
Kutular nesnelerin üzerine oturmuyorsa dönüşümde hata vardır. **Eğitime geçmeden düzelt.**

### 3.5 `data.yaml`
`datasets/gsr_yolo/data.yaml`:
```yaml
path: /tam/yol/futbol_tespit/datasets/gsr_yolo
train: images/train
val: images/val
names:
  0: ball
  1: goalkeeper
  2: player
  3: referee
```

---

## 4. Kendi veri setin (test + gerekirse ince ayar)

### 4.1 Video seçimi
- 2–3 farklı maçtan **yayın kamerası** görüntüsü (geniş açı; tribün, reklam, yakın çekim ağırlıklı kısımlar değil).
- **Test videosu ile ince ayar videoları farklı maçlardan olmalı.** Test videosu eğitimin hiçbir aşamasına girmez.

### 4.2 Kare çıkarma
Saniyede 2 kare yeterli:
```bash
ffmpeg -i videos/test_mac.mp4 -vf fps=2 -q:v 2 datasets/own/test_frames/%05d.jpg
```
Sonra elle ayıkla: yakın çekim, tekrar (replay), reklam, tribün karelerini sil.
Hedef büyüklükler:
- **Test seti:** ~150 kare
- **İnce ayar seti:** 300–500 kare (yalnızca gerekirse, bkz. §6)

### 4.3 Etiketleme aracı
| Araç | Artısı | Eksisi |
|---|---|---|
| **CVAT** (önerilen) — https://app.cvat.ai | Ücretsiz çevrimiçi sürüm, YOLO formatında içe/dışa aktarım, hazır etiketleri yükleyip düzeltme, video için kutu taşıma (interpolasyon) | Arayüze alışmak 30 dk sürer |
| Label Studio | Kurulumu kolay (`pip install label-studio`), yerel çalışır | Video ve YOLO içe aktarma CVAT kadar rahat değil |
| Roboflow Annotate | Çok kolay, YOLO dışa aktarımı tek tık | Ücretsiz planda veri seti herkese açık olur |

### 4.4 Etiketleme süresini kısaltma: önce model çizsin, sen düzelt
Sıfırdan kutu çizmek çok yavaştır. Bunun yerine:
1. Önce SN-GSR ile eğittiğin modeli (§5) bu karelerde çalıştır:
   ```bash
   yolo predict model=runs/detect/gsr_v1/weights/best.pt source=datasets/own/test_frames imgsz=1280 conf=0.25 save_txt=True
   ```
   Her kare için YOLO formatında bir `.txt` oluşur (`runs/detect/predict*/labels/`).
2. Kareleri ve bu `.txt` dosyalarını CVAT'a **YOLO formatında içe aktar** (görevde "Upload annotations").
3. CVAT'ta yalnızca **hataları düzelt**: eksik topu ekle, yanlış sınıfı değiştir, fazla kutuyu sil.

Bu yöntem 3–5 kat hızlıdır.

> **Dikkat:** Test setini düzeltirken modelin önerisine körü körüne güvenme. Her kareye bak; test seti "gerçek doğru" olmalı.

### 4.5 Etiketleme kuralları (tutarlılık için yazılı olsun)
- Kutu nesneyi **sıkıca** sarar (fazladan boşluk yok).
- **Top:** Görünüyorsa her zaman etiketle; bulanık, havada veya yarısı görünüyor olsa da etiketle. Tamamen gizliyse etiketleme.
- **Oyuncu:** Yarısından azı görünüyorsa da etiketle (ör. başka oyuncunun arkasında). Kutu yalnızca görünen kısmı kapsar.
- **Kaleci:** Formasından ayrılır (farklı renk); `player` değil `goalkeeper`.
- **Hakem:** Orta hakem ve yan hakemler `referee`. Dördüncü hakem, yedekler, teknik ekip, top toplayıcı, seyirci **etiketlenmez**.
- Saha dışındaki kişiler etiketlenmez.
- Kararsız kaldığın durumları bir listeye yaz, sonradan tek bir kurala bağla.

### 4.6 Dışa aktarma
CVAT'ta "Export dataset" → **Ultralytics YOLO Detection** (veya "YOLO 1.1") seç. Çıkan dosyaları şu düzene koy:
```
datasets/own/
├── images/test/   labels/test/     # test videosundan
├── images/train/  labels/train/    # ince ayar videolarından (gerekirse)
└── test.yaml
```
`test.yaml`, `data.yaml` ile aynı yapıdadır; yalnızca `val: images/test` olur. Sınıf numaraları §0'daki tabloyla **aynı** olmalıdır (CVAT'ta etiketleri bu sırayla tanımla: ball, goalkeeper, player, referee).

---

## 5. Model eğitimi

### 5.1 Model seçimi
| Model | Boyut | Ne zaman |
|---|---|---|
| `yolo11s.pt` | Küçük, hızlı | GPU ≤ 6 GB veya ilk deneme |
| **`yolo11m.pt`** | Orta (önerilen) | GPU ≥ 8 GB veya Kaggle/Colab |
| `yolo11l.pt` | Büyük | Zaman ve GPU bolsa; küçük kazanç |

> **Neden yüksek çözünürlük (`imgsz=1280`)?** Yayın görüntüsünde top yalnızca 5–15 piksel. 640'a küçültülünce top 2–5 piksele düşer ve model göremez. Oyuncular için 640 yeterli olurdu, top için 1280 gerekir.

### 5.2 İlk eğitim (yalnızca SN-GSR)
```bash
yolo detect train model=yolo11m.pt data=datasets/gsr_yolo/data.yaml imgsz=1280 epochs=50 batch=-1 patience=15 name=gsr_v1
```
- `batch=-1`: GPU belleğine göre otomatik seçilir. Bellek hatası alırsan `batch=4` veya `batch=2` yap, gene olmazsa `yolo11s.pt` kullan.
- `patience=15`: 15 epoch boyunca iyileşme olmazsa erken durur.
- Kaggle'da 12 saatlik oturum sınırına dikkat et; yarıda kalırsa `yolo detect train resume model=runs/detect/gsr_v1/weights/last.pt` ile devam edilir.
- Süre: Görüntü sayısına ve GPU'ya göre birkaç saat. Önce `epochs=3` ile kısa bir deneme yapıp her şeyin çalıştığını gör, sonra uzun eğitimi başlat.

Çıktılar `runs/detect/gsr_v1/` altında: `weights/best.pt`, `results.png` (kayıp ve mAP eğrileri), `confusion_matrix.png`, örnek tahmin görselleri.

### 5.3 Eğitim çıktısını okuma
- **`results.png`:** Eğitim kaybı düşerken doğrulama kaybı yükselmeye başlıyorsa aşırı öğrenme (overfitting) vardır; erken durma bu yüzden önemlidir.
- **mAP50:** Kutuların gerçeğe en az %50 örtüştüğü durumdaki ortalama doğruluk. Oyuncu için 0,9 üstü beklenir, top için çok daha düşük çıkması normaldir.
- **Recall (top):** Gerçek topların yüzde kaçını buldu? **Bu projede en önemli sayı budur.**
- **Confusion matrix:** Hangi sınıflar karışıyor? (Ör. hakem ↔ oyuncu, kaleci ↔ oyuncu.)

---

## 6. Değerlendirme (kendi test setinde)

### 6.1 Sayısal ölçüm
```bash
yolo detect val model=runs/detect/gsr_v1/weights/best.pt data=datasets/own/test.yaml imgsz=1280 conf=0.001 split=val
```
Sınıf bazında P, R, mAP50, mAP50-95 tablosunu rapora koy.

### 6.2 Görsel kontrol
Test videosunun tamamında modeli çalıştır ve videoyu izle:
```bash
yolo predict model=runs/detect/gsr_v1/weights/best.pt source=videos/test_mac.mp4 imgsz=1280 conf=0.25 save=True
```
Bak: Top hangi durumlarda kayboluyor (hızlı şut, oyuncunun ayağında, havada)? Hakem oyuncu sanılıyor mu? Seyirci/yedek oyuncu olarak bulunuyor mu?

### 6.3 Güven eşiği seçimi
Top için düşük (0,1–0,2), oyuncular için daha yüksek (0,3–0,5) bir eşik genelde iyi çalışır. Test setinde birkaç değeri dene ve önerdiğin eşikleri rapora yaz.

### 6.4 Karar: ince ayar gerekiyor mu?
- Test setinde **top recall'u yeterliyse** (hedef: en az %80) → §8'e geç, teslim et.
- Yeterli değilse → §7.

---

## 7. (Gerekirse) Kendi verinle ince ayar
1. İnce ayar videolarından 300–500 kare çıkar ve §4.4'teki yöntemle etiketle.
2. SN-GSR ile kendi verini birleştir. Kendi verin az olduğu için kaybolmasın diye `data.yaml`'da kendi `train` klasörünü 2–3 kez listeleyebilirsin:
   ```yaml
   train:
     - /tam/yol/datasets/gsr_yolo/images/train
     - /tam/yol/datasets/own/images/train
     - /tam/yol/datasets/own/images/train
   val: /tam/yol/datasets/gsr_yolo/images/val
   ```
3. Önceki en iyi modelden devam ederek kısa bir eğitim:
   ```bash
   yolo detect train model=runs/detect/gsr_v1/weights/best.pt data=datasets/mix.yaml imgsz=1280 epochs=20 lr0=0.002 name=mix_v1
   ```
4. §6'yı tekrarla ve iki modeli **aynı test setinde** karşılaştır. Daha iyi olanı seç.

---

## 8. Teslim
`teslim/` klasörü:
```
teslim/
├── best.pt
├── data.yaml
├── requirements.txt          # pip freeze çıktısı
└── RAPOR.md                  # aşağıdaki başlıklar
```
`RAPOR.md` içeriği:
- Kullanılan veri (SN-GSR bölümleri, kare örnekleme adımı, kendi veri setinin büyüklüğü ve hangi videolardan geldiği)
- Model, `imgsz`, epoch sayısı, eğitim komutu, eğitim süresi ve GPU
- Test setinde sınıf bazında P / R / mAP50 tablosu
- Önerilen güven eşikleri (top ve diğerleri için ayrı)
- Bilinen zayıflıklar (ör. "hızlı şutlarda top kayboluyor", "sarı formalı hakem bazen oyuncu sanılıyor")

Hızlı kullanım testi (teslimden önce teslim alan kişi de bunu çalıştıracak):
```python
from ultralytics import YOLO
m = YOLO("teslim/best.pt")
print(m.names)   # {0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'} olmalı
r = m.predict("ornek_kare.jpg", imgsz=1280, conf=0.1)[0]
print(r.boxes.cls, r.boxes.conf, r.boxes.xyxy)
```

---

## 9. Önerilen zaman planı
| Adım | İş | Tahmini süre |
|---|---|---|
| 1 | Ortam kurulumu, SN-GSR indirme | 0,5–1 gün (indirme bağlantıya bağlı) |
| 2 | Yapıyı inceleme, dönüştürme, gözle kontrol | 0,5 gün |
| 3 | Kısa deneme eğitimi (3 epoch), ardından tam eğitim | 1 gün (çoğu bekleme) |
| 4 | Test videosundan kare çıkarma + etiketleme (150 kare) | 0,5–1 gün |
| 5 | Değerlendirme + rapor | 0,5 gün |
| 6 | (Gerekirse) ince ayar verisi + eğitim + yeniden değerlendirme | 1–2 gün |

---

## 10. Sık karşılaşılan sorunlar
| Sorun | Çözüm |
|---|---|
| `CUDA out of memory` | `batch` küçült; olmazsa `yolo11s.pt`; son çare `imgsz=960` (top recall düşebilir) |
| `torch.cuda.is_available()` = False | PyTorch'u CUDA sürümüyle yeniden kur (pytorch.org'daki komut) |
| Eğitimde top mAP'i çok düşük | `imgsz=1280` kullanıldığından emin ol; dönüşümde top kutularının gerçekten yazıldığını kontrol et (§3.4) |
| Eğitim skoru çok yüksek, gerçek videoda kötü | Aynı klibin kareleri hem train hem val'de olabilir; klip bazında böl |
| Disk doldu | `STEP` değerini büyüt (ör. 10); açılmış zip'leri dönüştürmeden sonra sil |
| Kaggle oturumu kapandı | `resume` ile `last.pt`'den devam et; çıktıları düzenli olarak Kaggle Dataset'e kaydet |

---

## 11. Faydalı bağlantılar
- Ultralytics YOLO belgeleri: https://docs.ultralytics.com
  - Eğitim: https://docs.ultralytics.com/modes/train/
  - Veri formatı: https://docs.ultralytics.com/datasets/detect/
- SN-GSR veri seti: https://huggingface.co/datasets/SoccerNet/SN-GSR-2025
- SN-GSR referans kodu ve veri açıklaması: https://github.com/SoccerNet/sn-gamestate
- CVAT: https://app.cvat.ai · belgeler: https://docs.cvat.ai
- ffmpeg: https://ffmpeg.org
