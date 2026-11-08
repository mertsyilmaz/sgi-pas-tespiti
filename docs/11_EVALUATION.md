# 11 — Değerlendirme Planı

> Şartname §9: "Elle doğrulanmış örneklerle değerlendirilmeden sistemin pasları doğru sınıflandırdığı iddia edilmez."

Her modül **ayrı** değerlendirilir. Böylece hatanın kaynağı (tespit, takip, eşleme veya kural) ayırt edilebilir.

| Seviye | Ölçüt | Veri |
|---|---|---|
| M2 tespit | mAP@0.5, sınıf bazında precision/recall (**top recall** kritik) | Veri setinin test bölümü + kendi kliplerimizden elle etiketli kareler |
| M2 keypoint | Ortalama piksel hatası, görünür keypoint oranı | Keypoint veri setinin test bölümü |
| M3 takip | ID switch sayısı, (opsiyonel) HOTA/MOTA | Kısa elle etiketli dizi |
| M3b takım | Takım ataması doğruluğu, hakemin takıma atanma oranı (0 olmalı) | Elle etiketli `track_id → team` |
| M4 eşleme | Konum hatası (m; ortalama, medyan, %95; uzak/yakın taraf ayrı), kapsama, F-3 drift eğrisi, (ops.) JaC@5 | SN-GSR `bbox_pitch` + çizgi etiketleri; elle işaretlenmiş kareler |
| M5 hücre | **Hücre doğruluğu (%)**, sınıra uzaklığa göre kırılım | SN-GSR (gerçek konumun hücresi) + birim testleri |
| M4 (etiketsiz) | Öz-tutarlılık göstergeleri ve hata kategorileri E1–E7 | Kendi kliplerimiz (`14_CALIBRATION_QUALITY.md` §2.2–3) |
| M6 olay | Pas adayı precision/recall (±N kare toleransla eşleşme), başlangıç/hedef hücre doğruluğu | `passes_<clip>.csv` |
| M7 sınıf | Karışıklık matrisi SUCCESS/FAIL, takım bazında sayıların elle sayımla farkı | `passes_<clip>.csv` |

## Elle pas etiketi formatı (`data/annotations/passes_<clip>.csv`)
```
pass_id,frame_start,frame_end,passer_team,passer_jersey_or_track,receiver_team,receiver_jersey_or_track,start_cell,target_cell,label,notes
```

## Raporlanacaklar
- Her seviye için metrikler ve hata örnekleri (görsel)
- Yedek kuralların (F-1…F-5) ne sıklıkla uygulandığı ve uygulandığında hata oranı (`13_LIMITATIONS.md` için kanıt)
- Kullanılan konfig hash'i (tekrarlanabilirlik için)
