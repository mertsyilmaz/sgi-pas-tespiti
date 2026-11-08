# 06 — Veri Modeli (Modüller Arası Sözleşmeler)

> Kod karşılığı: `src/sgi/types.py` (dataclass). Alan adları burada değişirse kod da güncellenir.

## Enum'lar
```
ObjectClass  = player | goalkeeper | referee | ball
TeamId       = A | B | REF | UNKNOWN
CellId       = int 1..50 | "OUT" | "UNKNOWN"
PassLabel    = SUCCESS | FAIL          (D-008: belirsiz sınıfı yok)
ReceiverType = teammate | opponent | none | out | unknown
```

## M2 → M3: `Detection`
| Alan | Tip | Açıklama |
|---|---|---|
| frame_idx | int | |
| cls | ObjectClass | |
| bbox_xyxy | float[4] | görüntü pikseli |
| conf | float | |

## M2 → M4: `PitchKeypoint`
| Alan | Tip | Açıklama |
|---|---|---|
| frame_idx | int | |
| kp_id | int | saha şablonundaki nokta numarası (`configs/pitch.yaml`) |
| image_xy | float[2] | |
| conf | float | |

## M4: `HomographyResult`
| Alan | Tip | Açıklama |
|---|---|---|
| frame_idx | int | |
| H | float[3x3] \| null | image → pitch |
| n_inliers | int | |
| reproj_error_m | float | inlier'ların ortalama hatası (metre) |
| valid | bool | eşiklere göre |
| source | str | `keypoints` \| `camera_motion` \| `nearest` (F-3) |
| motion_chain_len | int | keypoint'li kareden bu yana kaç kare taşındı (drift göstergesi) |

## M5 sonu: `FrameState`
```
FrameState
├── frame_idx, timestamp
├── homography_valid: bool, homography_error_m: float
├── objects: [ObjectState]
│     track_id, cls, team_id, team_conf,
│     image_xy (oyuncu: bbox alt-orta, top: merkez),
│     pitch_xy | null, cell: CellId, det_conf
└── ball: BallState | null
      image_xy, pitch_xy | null, cell: CellId,
      det_conf, visible: bool, interpolated: bool
```

## M6: `PassCandidate`
| Alan | Tip |
|---|---|
| candidate_id | str |
| t_start, t_end / frame_start, frame_end | float / int |
| passer_track_id, passer_team | int \| null, TeamId |
| receiver_track_id, receiver_team | int \| null, TeamId |
| receiver_type | ReceiverType |
| start_cell, target_cell | CellId |
| path_cells | list[CellId \| "GAP"] (zaman sıralı, tekrarlar birleştirilmiş) |
| intermediate_cells | `path_cells[1:-1]` |
| ball_path | list[(frame_idx, pitch_xy, cell, interpolated)] |
| passer_pos, receiver_pos | pitch_xy |
| event_hints | list[str] (ör. `shot_like`, `deflection_like`) |
| quality | {ball_missing_frames, interpolated_ratio, invalid_homography_ratio, receiver_distance_m} |

## M7: `PassRecord` (şartname §8 alanları)
`PassCandidate` alanlarına ek olarak:
| Alan | Tip |
|---|---|
| pass_id | str (ör. `P0001`) |
| label | PassLabel |
| reason | str (insan okunur) |
| matched_rule_id | str \| null |
| rule_trace | list[{rule_id, matched: bool, note}] |
| fallbacks_applied | list[str] (ör. `["F-1","F-3"]`, izlenebilirlik için) |
| target_cell_players | list[{track_id, team_id, dist_to_ball_m}] |

## M8: `Summary`
```
by_team: { A: {success, fail}, B: {success, fail} }
counting_rules: <metin, sayım tanımı>
config_hash: <pass_rules.yaml ve grid.yaml özet hash'i, tekrarlanabilirlik için>
```

## Dosya çıktıları (çalıştırma başına `outputs/<run_id>/`)
- `frame_states.parquet` (veya `.jsonl`)
- `passes.json`: `PassRecord[]`
- `summary.json` ve `summary.csv`
- `annotated.mp4`, `pitch_view.mp4` (opsiyonel)
- `run_config.yaml`: kullanılan tüm konfiglerin kopyası
