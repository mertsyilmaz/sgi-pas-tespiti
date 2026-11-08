"""Pas tespiti ve sınıflandırma (M6 + M7 + M8): objects.npz -> passes.json, summary.json.

objects.npz, `grid_demo.py process` çıktısıdır (oyuncu/top saha konumları ve hücreleri). Kare kare, nedensel işlenir.
Kullanım: .venv/Scripts/python.exe scripts/passes.py outputs/grid_demo/pas_test/s0_d20/objects.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sgi.events.passes import PassDetector  # noqa: E402
from sgi.grid.grid import PitchGrid  # noqa: E402
from sgi.objects import rows_to_objects  # noqa: E402
from sgi.rules.engine import RuleEngine, summarize  # noqa: E402


def run(objects_npz: Path, rules_cfg: dict, grid: PitchGrid):
    z = np.load(objects_npz)
    rows, fps = z["rows"], float(z["fps"])
    start = int(z["start_frame"])
    frames = rows[:, 0].astype(int)
    det = PassDetector.from_config(grid, fps, rules_cfg)
    eng = RuleEngine(rules_cfg)
    records = []
    lo, hi = frames.min(), frames.max()
    order = np.argsort(frames, kind="stable")
    rows, frames = rows[order], frames[order]
    bounds = np.searchsorted(frames, np.arange(lo, hi + 2))
    for k, f in enumerate(range(lo, hi + 1)):
        objs = rows_to_objects(rows[bounds[k]:bounds[k + 1]])
        for cand in det.update(f, objs):
            records.append(eng.classify(cand))
    return records, fps, start


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("objects")
    ap.add_argument("--rules", default="configs/pass_rules.yaml")
    ap.add_argument("--grid", default="configs/grid.yaml")
    ap.add_argument("--video", action="store_true",
                    help="ızgara, oyuncular ve paslarla birlikte panelli passes_overlay.mp4 üret")
    ap.add_argument("--source", help="orijinal video (varsayılan: data/videos/<klasör adı>.mp4)")
    ap.add_argument("--team", choices=["A", "B"],
                    help="hedef takım (D-040): yalnızca bu takımın pasları sayılır -> summary_<takım>.json, "
                         "passes_overlay_<takım>.mp4. Verilmezse iki takım")
    args = ap.parse_args()
    path = Path(args.objects)
    cfg = yaml.safe_load((ROOT / args.rules).read_text(encoding="utf-8"))
    grid = PitchGrid.from_yaml(ROOT / args.grid)
    records, fps, start = run(path, cfg, grid)

    out = path.parent
    recs = [r.to_dict(fps) for r in records]
    for d in recs:                                   # kare numaraları npz'deki gibi (kesitte göreli); video saniyesi ayrıca
        d["video_t_start"] = round((start + d["frame_start"]) / fps, 2)
        d["video_t_end"] = round((start + d["frame_end"]) / fps, 2)
    (out / "passes.json").write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
    suffix = f"_{args.team}" if args.team else ""
    summ = summarize(records, args.team)
    (out / f"summary{suffix}.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")

    for d in recs:
        if args.team and d["team"] != args.team:
            continue
        rcv = f"{d['receiver_team']}#{d['receiver_track_id']}" if d["receiver_track_id"] is not None else "-"
        print(f"{d['pass_id']} {d['video_t_start']:6.2f}-{d['video_t_end']:6.2f}s {d['team']}#{d['passer_track_id']:<4d} "
              f"-> {rcv:7s} {str(d['start_cell']):>3s}->{str(d['target_cell']):<3s} {d['end_reason']:9s} "
              f"{d['label']:7s} {d['matched_rule_id']:5s} {','.join(d['fallbacks_applied'])}")
    print(f"\n{'hedef takım: ' + args.team if args.team else 'iki takım'}\n           Başarılı  Başarısız")
    for t, v in summ["by_team"].items():
        print(f"Takım {t}:   {v['success']:5d}   {v['fail']:5d}")
    print(f"[çıktı] {out / 'passes.json'}\n[çıktı] {out / f'summary{suffix}.json'}")
    if args.video:
        render_video(out, recs, grid, Path(args.source) if args.source else source_video(out), args.team)


def source_video(out: Path) -> Path:
    """outputs/grid_demo/<video>/[s<start>_d<dur>/] -> data/videos/<video>.mp4"""
    stem = out.resolve().relative_to((ROOT / "outputs" / "grid_demo").resolve()).parts[0]
    return ROOT / "data" / "videos" / f"{stem}.mp4"


def render_video(out: Path, recs: list[dict], grid: PitchGrid, video: Path, team: str | None = None) -> None:
    """Orijinal videodan ızgara + oyuncular (GridOverlay) ve paslar (PassOverlay), panelli tuvalde."""
    import cv2

    from sgi.io.calibration import CalibrationCache
    from sgi.io.video import FrameReader
    from sgi.viz.grid_overlay import GridOverlay, OverlayOptions
    from sgi.viz.pass_overlay import PassOverlay

    cal = CalibrationCache(out / "calibration.npz")
    z = np.load(out / "objects.npz")
    rows = z["rows"]
    frames = rows[:, 0].astype(int)
    team_cols = {t: tuple(int(v) for v in z["team_colors"][k]) for k, t in enumerate("AB")}
    overlay, opt = GridOverlay(grid, cal.size), OverlayOptions()
    passes = PassOverlay(recs, cal.size, fps=cal.fps, target_team=team)
    name = f"passes_overlay_{team}.mp4" if team else "passes_overlay.mp4"
    wr = cv2.VideoWriter(str(out / name), cv2.VideoWriter_fourcc(*"mp4v"), cal.fps, overlay.out_size)
    n = 0
    for frame_idx, frame in FrameReader(str(video), cal.start, cal.n):
        i = frame_idx - cal.start
        objs = rows_to_objects(rows[frames == i])
        vis = overlay.draw(frame, cal.result(i), opt, frame_idx, cal.fps, objects=objs, team_colors=team_cols)
        wr.write(passes.draw(vis, i, cal.H(i), overlay.panel_bottom))
        n += 1
    wr.release()
    print(f"[çıktı] {out / name} ({n} kare)")


if __name__ == "__main__":
    main()
