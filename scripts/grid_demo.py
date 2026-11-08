"""Faz 0 demosu: maç videosu üzerinde 50 hücreli saha ızgarası (D-013).

Kullanım:
  # 1) Videoyu işle: ızgaralı MP4 + kalibrasyon önbelleği üretir (isteğe bağlı canlı önizleme)
  .venv/Scripts/python scripts/grid_demo.py process data/videos/mac.mp4 --show

  # 2) Önbellekle gerçek hızda oynat (tuşlarla katmanları aç/kapa)
  .venv/Scripts/python scripts/grid_demo.py play data/videos/mac.mp4

  # Canlı: önceden işleme yok, video oynarken ızgara anlık hesaplanır (model arka planda, D-016)
  .venv/Scripts/python scripts/grid_demo.py live data/videos/mac.mp4

Oynatıcı tuşları: SPACE duraklat | a/d bir kare geri/ileri (duraklatılmışken) | j/l 5 sn geri/ileri
                  g ızgara | f dolgu | n numaralar | m mini harita | c mini haritada tespit edilen çizgiler | y kale
                  k saha noktaları | h bilgi | q/ESC çıkış
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sgi.grid.grid import PitchGrid  # noqa: E402
from sgi.detection.keypoints.cache import KeypointCache  # noqa: E402
from sgi.events.passes import PassDetector  # noqa: E402
from sgi.io.calibration import CalibrationCache  # noqa: E402
from sgi.io.video import AsyncWriter, FrameReader  # noqa: E402
from sgi.mapping.camera_prior import CameraPositionPrior  # noqa: E402
from sgi.mapping.homography import HomographyEstimator, HomographyTracker  # noqa: E402
from sgi.mapping.live import MotionHistory, ProcessGpuScheduler, ProcessWorker  # noqa: E402
from sgi.mapping.quality import QualityMonitor  # noqa: E402
from sgi.objects import N_COLS, ROLE_CODES, STATE_CODES, TEAM_CODES, ObjectPipeline, objects_to_rows, rows_to_objects  # noqa: E402
from sgi.registry import build  # noqa: E402
from sgi.rules.engine import RuleEngine  # noqa: E402
from sgi.types import HomographyResult  # noqa: E402
from sgi.viz.grid_overlay import GridOverlay, OverlayOptions  # noqa: E402
from sgi.viz.live_smoother import BoxFlow, LiveObjectSmoother  # noqa: E402
from sgi.viz.pass_overlay import PassOverlay  # noqa: E402

SOURCES = ["none", "keypoints", "camera_motion"]
REJECT_CODES = ["", "few_keypoints", "ransac", "few_inliers", "reproj", "spread", "cam_nofocal", "cam_focal",
                "cam_ortho", "cam_aspect", "cam_height", "jump"]
WINDOW = "SGI - 50 hucreli saha izgarasi"


def out_dir_for(video: Path, start: float = 0.0, duration: float = 0.0) -> Path:
    """Tam çalıştırma: outputs/grid_demo/<video>/, kısmi çalıştırma: .../<video>/s<start>_d<duration>/
    (kısmi bir çalıştırma, tam çalıştırmanın çıktılarının üzerine yazmamalı)."""
    d = ROOT / "outputs" / "grid_demo" / video.stem
    if start or duration:
        d = d / f"s{start:g}_d{duration:g}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def open_window(size: tuple[int, int]) -> None:
    """Panelli tuval (video + sağ/alt panel) için yeniden boyutlanabilir pencere; ekrana sığacak kadar."""
    w, h = size
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, min(w, 1600), int(min(w, 1600) * h / w))


def open_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    # opencv-python tekerleği H.264 (OpenH264) içermiyor; MPEG-4 (mp4v) VLC ve Windows oynatıcılarında açılır
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not wr.isOpened():
        raise RuntimeError(f"Video yazıcı açılamadı: {path}")
    return wr


def make_camera_prior(cfg: dict, size: tuple[int, int]):
    """Sabit kamera konumu önseli (D-027); konfigde kapalıysa None."""
    pcfg = dict(cfg.get("camera_prior") or {})
    if not pcfg.pop("enabled", False):
        return None
    return CameraPositionPrior(size, **pcfg)


def make_refiner(cfg: dict, size: tuple[int, int], pitch: dict, live: bool):
    """Çizgilere oturtma adımı (D-020) ve kaç karede bir uygulanacağı; konfigde kapalıysa (None, 1)."""
    rcfg = dict(cfg.get("pitch_refiner") or {})
    if not (rcfg.pop("enabled", True) and rcfg):
        return None, 1
    every = int(rcfg.pop("live_every" if live else "every", 1))
    rcfg.pop("every", None), rcfg.pop("live_every", None)
    live_opts = rcfg.pop("live", None) or {}
    if live:
        rcfg.update(live_opts)                    # canlıya özel hızlı ayarlar (D-029)
    return build("pitch_refiner", rcfg, image_size=size, pitch_length=pitch["length"],
                 pitch_width=pitch["width"]), every


def _refiner_fn(cfg: dict, size: tuple[int, int], pitch: dict):
    """Canlı mod alt süreci için: çizgilere oturtma işlevi (ProcessWorker fabrikası, D-029)."""
    cv2.setNumThreads(1)                      # alt süreç ana süreçle çekirdek yarışına girmesin
    refiner, _ = make_refiner(cfg, size, pitch, live=True)
    return refiner.refine


def _gpu_fn(cfg: dict, size: tuple[int, int], pitch: dict, fps: float, calib_gap: int, has_objects: bool,
            lost_gap: int = 2):
    """Canlı mod GPU alt süreci (D-031): saha noktası modeli + nesne hattı. Hangisinin çalışacağını burada seçer
    (D-023 zamanlaması): saha modeli en az `calib_gap` karede bir, ızgara yokken `lost_gap` karede bir (D-029)."""
    detector = build("pitch_keypoint_detector", cfg["pitch_keypoint_detector"], root=ROOT,
                     pitch_length=pitch["length"], pitch_width=pitch["width"])
    estimator = HomographyEstimator(**cfg["homography"], image_size=size,
                                    camera_prior=make_camera_prior(cfg, size))   # canlıda çevrimiçi öğrenir
    boot = make_bootstrap(cfg, size, pitch, estimator.camera_prior)
    grid = PitchGrid.from_yaml(ROOT / "configs/grid.yaml")
    objects = ObjectPipeline(cfg["objects"], grid, fps, ROOT) if has_objects else None
    state = {"last_calib": -10 ** 9}

    def step(idx, frame, H, cmd):
        if cmd == "warmup":
            detector.detect(frame)
            if objects is not None:
                objects.detector.detect(frame)
            return "ready", None
        if cmd == "reset":
            if objects is not None:
                objects.reset()
            state["last_calib"] = -10 ** 9
        gap = lost_gap if H is None else calib_gap
        if objects is None or idx - state["last_calib"] >= gap:
            state["last_calib"] = idx
            meas = estimator.estimate(idx, detector.detect(frame))
            if boot is not None:
                meas = boot.apply(meas, frame, lost=H is None)     # ızgara yokken az noktadan başlatma (D-034)
            return "calib", meas
        return "obj", (objects.process(frame, H, idx), objects.team_colors())

    return step


def make_bootstrap(cfg: dict, size: tuple[int, int], pitch: dict, prior=None):
    """Az noktadan başlangıç kalibrasyonu (D-034); kapalıysa None."""
    bcfg = cfg.get("bootstrap") or {}
    if not bcfg.get("enabled", True):
        return None
    from sgi.mapping.bootstrap import LineCameraBootstrap
    return LineCameraBootstrap(size, pitch["length"], pitch["width"], camera_prior=prior,
                               **{k: v for k, v in bcfg.items() if k != "enabled"})


def make_tracker(cfg: dict, size: tuple[int, int], pitch: dict, live: bool) -> HomographyTracker:
    """Homografi takibi + (varsa) çizgilere oturtma adımı (D-020)."""
    refiner, every = make_refiner(cfg, size, pitch, live)
    return HomographyTracker(size, refiner=refiner, refine_every=every, **cfg["tracking"])


def cmd_process(args: argparse.Namespace) -> None:
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    pitch = yaml.safe_load((ROOT / "configs/pitch.yaml").read_text(encoding="utf-8"))
    grid = PitchGrid.from_yaml(ROOT / "configs/grid.yaml")
    video = Path(args.video)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        sys.exit(f"Video açılamadı: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    start = int(args.start * fps)
    end = min(total, start + int(args.duration * fps)) if args.duration else total

    print(f"[bilgi] {video.name}: {w}x{h} @ {fps:.2f} fps, kareler {start}..{end - 1}")
    every = max(1, int(cfg["tracking"].get("detect_every", 1)))
    det_cfg = dict(cfg["pitch_keypoint_detector"], pitch_length=pitch["length"], pitch_width=pitch["width"])
    cache = None if not args.use_cache else KeypointCache(ROOT / "outputs" / "cache" / "keypoints", video, det_cfg)
    needed = [f for f in range(start, end) if (f - start) % every == 0]
    detector = None
    if cache is not None and cache.covers(needed):
        print(f"[bilgi] saha noktaları önbellekten okunuyor, model yüklenmeyecek ({cache.path.name})")
    else:
        if cache is not None:
            print(f"[bilgi] önbellekte {len(cache)} kare var; eksik kareler için model çalışacak")
        print("[bilgi] modeller yükleniyor...")
        det_cfg.pop("pitch_length"), det_cfg.pop("pitch_width")
        detector = build("pitch_keypoint_detector", det_cfg, root=ROOT,
                         pitch_length=pitch["length"], pitch_width=pitch["width"])

    def detect(frame_idx: int, frame: np.ndarray):
        kps = cache.get(frame_idx) if cache is not None else None
        if kps is None:
            kps = detector.detect(frame)
            if cache is not None:
                cache.put(frame_idx, kps)
        return kps

    motion_est = build("camera_motion", cfg["camera_motion"])
    prior = make_camera_prior(cfg, (w, h))
    estimator = HomographyEstimator(**cfg["homography"], image_size=(w, h), camera_prior=prior)
    boot = make_bootstrap(cfg, (w, h), pitch, prior)
    tracker = make_tracker(cfg, (w, h), pitch, live=False)
    # kamera konumu canlı moddaki gibi işlenirken öğrenilir; ileriki karelere bakan ön tarama yok (D-028)
    overlay = GridOverlay(grid, (w, h))
    opt = OverlayOptions(keypoints=args.keypoints)
    ocfg = cfg.get("objects") or {}
    objects = None
    if ocfg.get("enabled", True) and ocfg and not args.no_objects:
        print("[bilgi] nesne dedektörü yükleniyor (oyuncu/hakem/top)...")
        objects = ObjectPipeline(ocfg, grid, fps, ROOT)

    out_dir = out_dir_for(video, args.start, args.duration)
    writer = None
    if not args.no_video:
        # çizim + kodlama arka planda; --show ise çizim ana iş parçacığında yapılıp hazır kare gönderilir
        render = None if args.show else (lambda f, r, idx, objs, cols: overlay.draw(f, r, opt, idx, fps,
                                                                                   objects=objs, team_colors=cols))
        writer = AsyncWriter(open_writer(out_dir / "grid_overlay.mp4", fps, overlay.out_size), render)
    n = end - start
    rec = {"H": np.full((n, 3, 3), np.nan), "source": np.zeros(n, np.int8), "n_inliers": np.zeros(n, np.int16),
           "reproj_error_m": np.full(n, np.nan), "motion_chain": np.zeros(n, np.int16),
           "n_keypoints": np.full(n, -1, np.int16),          # -1: bu karede model çalışmadı (detect_every)
           "reject": np.zeros(n, np.int8), "motion_ok": np.zeros(n, bool),
           "green_ratio": np.full(n, np.nan), "line_score": np.full(n, np.nan), "pitch_iou": np.full(n, np.nan),
           "quality_measured": np.zeros(n, bool), "refined": np.zeros(n, bool),
           "goal_visible": np.zeros(n, bool), "goal_found": np.zeros(n, bool), "used_prior": np.zeros(n, bool)}
    kp_rows: list[tuple[int, float, float]] = []
    obj_rows: list[list[float]] = []
    quality = None if args.no_quality else QualityMonitor((w, h), pitch["length"], pitch["width"])

    reader = FrameReader(str(video), start, n)
    t0 = time.time()
    done = 0
    try:
        done = _process_loop(reader, start, n, every, detect, motion_est, estimator, tracker, quality,
                             int(cfg.get("quality", {}).get("every", 5)), overlay, opt, fps, writer, args.show, rec,
                             kp_rows, t0, objects, obj_rows, boot=boot)
    finally:
        if cache is not None:
            cache.save()                  # yarıda kesilse bile hesaplanan kareler kaybolmaz
    print()
    n = done
    reader.close()
    if writer:
        writer.close()
    cv2.destroyAllWindows()

    kp = np.array(kp_rows, dtype=np.float32).reshape(-1, 3)
    np.savez_compressed(out_dir / "calibration.npz", start_frame=start, fps=fps, size=(w, h), keypoints=kp,
                        sources=np.array(SOURCES), reject_codes=np.array(REJECT_CODES),
                        **{k: v[:n] for k, v in rec.items()})
    s = rec["source"][:n]
    print(f"[özet] kare: {n} | saha noktalarıyla: %{100 * np.mean(s == 1):.1f} | "
          f"kamera hareketiyle (F-3): %{100 * np.mean(s == 2):.1f} | kalibrasyonsuz: %{100 * np.mean(s == 0):.1f}")
    if objects is not None:
        cols = objects.team_colors()
        np.savez_compressed(out_dir / "objects.npz", start_frame=start, fps=fps,
                            rows=np.array(obj_rows, dtype=np.float64).reshape(-1, N_COLS),
                            columns=np.array(["frame", "track_id", "role", "team", "x1", "y1", "x2", "y2", "conf",
                                              "pitch_x", "pitch_y", "cell", "state"]),
                            state_codes=np.array(STATE_CODES),
                            role_codes=np.array(ROLE_CODES), team_codes=np.array(TEAM_CODES),
                            team_colors=np.array([cols.get("A", (150, 150, 150)), cols.get("B", (150, 150, 150))]))
        r = np.array(obj_rows).reshape(-1, N_COLS)
        if len(r):
            print(f"[nesneler] kişi takibi: {len(np.unique(r[r[:, 2] != 3, 1]))} kimlik | {_ball_summary(r, n)} | "
                  f"takım renkleri (BGR): {cols}")
        if objects.pose is not None:
            print(f"[poz] {objects.ball.n_pose_checks} yön kontrolü, {objects.ball.n_pose_extended} kez "
                  f"oyuncuda sayma uzatıldı (sırtı dönük)")
    print(f"[süre] {time.time() - t0:.1f} sn ({n / max(time.time() - t0, 1e-6):.2f} kare/sn)")
    if writer:
        print(f"[çıktı] {out_dir / 'grid_overlay.mp4'}")
    print(f"[çıktı] {out_dir / 'calibration.npz'}")
    if objects is not None:
        print(f"[çıktı] {out_dir / 'objects.npz'}")


def _process_loop(reader, start, n, every, detect, motion_est, estimator, tracker, quality, quality_every, overlay,
                  opt, fps, writer, show, rec, kp_rows, t0, objects=None, obj_rows=None, boot=None) -> int:
    done = 0
    prev_valid = False
    for frame_idx, frame in reader:
        i = frame_idx - start
        motion = motion_est.estimate(frame)
        meas = estimator.estimate(frame_idx, detect(frame_idx, frame)) if i % every == 0 else None
        if boot is not None:
            meas = boot.apply(meas, frame, lost=not prev_valid)     # ızgara yokken az noktadan başlatma (D-034)
        res = tracker.update(meas, motion, frame_idx, frame)
        prev_valid = res.valid
        rec["refined"][i] = res.refined
        rec["used_prior"][i] = res.used_prior

        if res.valid:
            rec["H"][i] = res.H
        rec["source"][i] = SOURCES.index(res.source) if res.valid else 0
        rec["n_inliers"][i], rec["reproj_error_m"][i] = res.n_inliers, res.reproj_error_m
        rec["motion_chain"][i], rec["motion_ok"][i] = res.motion_chain_len, motion.M is not None
        if meas is not None:
            rec["n_keypoints"][i] = meas.n_keypoints
        rec["reject"][i] = REJECT_CODES.index(res.reject_reason) if res.reject_reason in REJECT_CODES else 0
        if quality is not None and res.valid and i % quality_every == 0:
            goals = quality.goal_detector.detect(frame, res.H)
            rec["goal_visible"][i] = bool(goals)
            rec["goal_found"][i] = any(g.found for g in goals)
        if quality is not None and i % quality_every == 0:
            q = quality.measure(frame, res.H if res.valid else None)
            rec["green_ratio"][i], rec["line_score"][i], rec["pitch_iou"][i] = q.green_ratio, q.line_score, q.pitch_iou
            rec["quality_measured"][i] = True
        kp_rows += [(i, *k.image_xy) for k in res.keypoints]
        objs, cols = None, None
        if objects is not None:
            objs = objects.process(frame, res.H if res.valid else None, frame_idx)
            cols = objects.team_colors()
            obj_rows += objects_to_rows(i, objs)
        done = i + 1

        if show:
            vis = overlay.draw(frame, res, opt, frame_idx, fps, objects=objs, team_colors=cols)
            if writer:
                writer.put(vis)
            cv2.imshow(WINDOW, vis)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
        elif writer:
            writer.put(frame, res, frame_idx, objs, cols)
        if i % 25 == 0 or i == n - 1:
            el = time.time() - t0
            print(f"\r[işleniyor] {i + 1}/{n}  {(i + 1) / max(el, 1e-6):5.1f} kare/sn", end="", flush=True)
    return done


def _ball_summary(r: np.ndarray, n: int) -> str:
    """Top durumlarının kare oranları (D-028): tespit / oyuncuda varsayıldı / yörüngeden tahmin."""
    b = r[r[:, 2] == ROLE_CODES.index("ball")]
    parts = [f"{name} %{100 * np.sum(b[:, 12] == k) / max(n, 1):.0f}"
             for k, name in enumerate(["tespit", "oyuncuda", "tahmin"])]
    return "top: " + ", ".join(parts)


def cmd_objects(args: argparse.Namespace) -> None:
    """Mevcut kalibrasyonu (calibration.npz) yeniden kullanıp yalnızca nesne adımını çalıştırır (D-022).
    Saha noktası modeli çalışmaz; takım/rol ayarlarını hızlı denemek içindir."""
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    grid = PitchGrid.from_yaml(ROOT / "configs/grid.yaml")
    video = Path(args.video)
    out_dir = out_dir_for(video)
    cal = CalibrationCache(out_dir / "calibration.npz")
    start, fps, n = cal.start, cal.fps, cal.n
    w, h = cal.size
    print(f"[bilgi] kalibrasyon yeniden kullanılıyor: {out_dir / 'calibration.npz'} ({n} kare)")
    objects = ObjectPipeline(cfg["objects"], grid, fps, ROOT)
    overlay, opt = GridOverlay(grid, (w, h)), OverlayOptions()
    writer = None if args.no_video else AsyncWriter(
        open_writer(out_dir / "grid_overlay.mp4", fps, overlay.out_size),
        lambda f, r, idx, objs, cols: overlay.draw(f, r, opt, idx, fps, objects=objs, team_colors=cols))
    reader = FrameReader(str(video), start, n)
    rows, t0 = [], time.time()
    for frame_idx, frame in reader:
        i = frame_idx - start
        res = cal.result(i)
        objs = objects.process(frame, res.H, frame_idx)
        rows += objects_to_rows(i, objs)
        if writer:
            writer.put(frame, res, frame_idx, objs, objects.team_colors())
        if i % 25 == 0 or i == n - 1:
            print(f"\r[işleniyor] {i + 1}/{n}  {(i + 1) / max(time.time() - t0, 1e-6):5.1f} kare/sn", end="", flush=True)
    print()
    reader.close()
    if writer:
        writer.close()
    cols = objects.team_colors()
    r = np.array(rows, dtype=np.float64).reshape(-1, N_COLS)
    np.savez_compressed(out_dir / "objects.npz", start_frame=start, fps=fps, rows=r,
                        role_codes=np.array(ROLE_CODES), team_codes=np.array(TEAM_CODES),
                        team_colors=np.array([cols.get("A", (150, 150, 150)), cols.get("B", (150, 150, 150))]))
    print(f"[nesneler] {_ball_summary(r, n)} | takım renkleri (BGR): {cols}")
    print(f"[süre] {time.time() - t0:.1f} sn")
    print(f"[çıktı] {out_dir / 'objects.npz'}" + (f"\n[çıktı] {out_dir / 'grid_overlay.mp4'}" if writer else ""))


def cmd_play(args: argparse.Namespace) -> None:
    video = Path(args.video)
    cache = Path(args.cache) if args.cache else out_dir_for(video) / "calibration.npz"
    if not cache.exists():
        sys.exit(f"Önbellek yok: {cache}\nÖnce: python scripts/grid_demo.py process {video}")
    cal = CalibrationCache(cache)
    start, fps, n = cal.start, cal.fps, cal.n
    grid = PitchGrid.from_yaml(ROOT / "configs/grid.yaml")

    cap = cv2.VideoCapture(str(video))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    overlay = GridOverlay(grid, (w, h))
    opt = OverlayOptions()
    obj_by_frame, team_cols = {}, None
    obj_file = cache.parent / "objects.npz"
    if obj_file.exists():
        o = np.load(obj_file)
        rows = o["rows"]
        for fi in np.unique(rows[:, 0]).astype(int):
            obj_by_frame[fi] = rows_to_objects(rows[rows[:, 0] == fi])
        team_cols = {"A": tuple(int(v) for v in o["team_colors"][0]), "B": tuple(int(v) for v in o["team_colors"][1])}
    pass_file = cache.parent / "passes.json"           # scripts/passes.py çıktısı varsa pas paneli de gösterilir
    passes = None
    if pass_file.exists():
        passes = PassOverlay(json.loads(pass_file.read_text(encoding="utf-8")), (w, h), fps=fps,
                             target_team=args.team)
    open_window(overlay.out_size)

    i, paused, frame = 0, False, None
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    next_t = time.time()
    while True:
        if frame is None or not paused:
            ok, frame = cap.read()
            if not ok or i >= n:
                i = 0
                cap.set(cv2.CAP_PROP_POS_FRAMES, start)
                continue
        vis = overlay.draw(frame, cal.result(i), opt, start + i, fps,
                           objects=obj_by_frame.get(i, []) if obj_file.exists() else None, team_colors=team_cols)
        cv2.imshow(WINDOW, passes.draw(vis, i, cal.H(i), overlay.panel_bottom) if passes is not None else vis)

        wait = max(1, int((next_t - time.time()) * 1000)) if not paused else 30
        key = cv2.waitKey(wait) & 0xFF
        next_t = max(next_t + 1.0 / fps, time.time()) if not paused else time.time()
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1 or key in (ord("q"), 27):
            break
        toggles = {ord("g"): "grid", ord("f"): "fill", ord("n"): "numbers", ord("m"): "minimap",
                   ord("k"): "keypoints", ord("h"): "hud", ord("c"): "detected_lines",
                   ord("o"): "objects", ord("y"): "goals"}
        if key in toggles:
            setattr(opt, toggles[key], not getattr(opt, toggles[key]))
        elif key == ord(" "):
            paused = not paused
        elif key in (ord("a"), ord("d"), ord("j"), ord("l")):
            step = {ord("a"): -1, ord("d"): 1, ord("j"): -int(5 * fps), ord("l"): int(5 * fps)}[key]
            i = int(np.clip(i + step, 0, n - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, start + i)
            ok, frame = cap.read()
            continue
        if not paused:
            i += 1
    cap.release()
    cv2.destroyAllWindows()


def _compensate(objs, T):
    """Geç gelen nesne kutularını kamera hareketi kadar kaydır (T: o karenin görüntüsü -> güncel kare)."""
    if T is None or np.allclose(T, np.eye(3)):
        return objs
    out = []
    for o in objs:
        x1, y1, x2, y2 = o.bbox
        p = np.array([[x1, y1, 1.0], [x2, y2, 1.0]]) @ T.T
        p = p[:, :2] / p[:, 2:3]
        out.append(replace(o, bbox=(float(p[0, 0]), float(p[0, 1]), float(p[1, 0]), float(p[1, 1]))))
    return out


def cmd_live(args: argparse.Namespace) -> None:
    """Önceden işleme olmadan, video oynarken anlık ızgara (D-016)."""
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    pitch = yaml.safe_load((ROOT / "configs/pitch.yaml").read_text(encoding="utf-8"))
    grid = PitchGrid.from_yaml(ROOT / "configs/grid.yaml")
    video = Path(args.video)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        sys.exit(f"Video açılamadı: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print("[bilgi] modeller yükleniyor (GPU alt süreci)...")
    motion_est = build("camera_motion", cfg["camera_motion"])
    history = MotionHistory()
    overlay = GridOverlay(grid, (w, h))
    opt = OverlayOptions(detected_lines=False)     # canlıda varsayılan kapalı (~10 ms); 'c' ile açılır
    # canlıda çizgilere oturtma ana döngüde değil, arka planda sürekli en güncel karede çalışır (D-029)
    new_tracker = lambda: HomographyTracker((w, h), **cfg["tracking"])  # noqa: E731
    line_refiner, refine_gap = make_refiner(cfg, (w, h), pitch, live=True)
    last_refine_submit = -10 ** 9
    refine_worker = ProcessWorker(_refiner_fn, (cfg, (w, h), pitch)) if line_refiner is not None else None
    line_refiner = None                       # ana süreçte kullanılmıyor; alt süreç kendi örneğini kurar
    tracker = new_tracker()
    ocfg = cfg.get("objects") or {}
    has_objects = bool(ocfg) and bool(ocfg.get("enabled", True))
    objs, objs_idx, cols = None, None, None
    # pas tespiti ve kural motoru (D-037) gelen nesne sonuçlarıyla, nedensel; hedef takım seçilebilir (D-040)
    rules_cfg = yaml.safe_load((ROOT / "configs/pass_rules.yaml").read_text(encoding="utf-8"))
    pass_det, rule_eng = PassDetector.from_config(grid, fps, rules_cfg), RuleEngine(rules_cfg)
    live_passes = PassOverlay([], (w, h), fps=fps, target_team=args.team)
    pass_from, pass_last = int(args.start * fps), -1      # bu aralığın dışındaki (atlamadan önceki) sonuçlar alınmaz
    # nesne sonuçları seyrek ve geç gelir: aradaki karelerde kutular optik akışla/hızla ilerletilir (D-030)
    smooth = bool((cfg.get("live") or {}).get("smooth_objects", True))
    smoother, box_flow = LiveObjectSmoother(), BoxFlow()
    calib_gap = max(1, int((cfg.get("live") or {}).get("calib_every", 10)))
    # tek GPU kuyruğu (D-023), ayrı süreçte (D-031): saha noktası modeli zamanı gelince öncelikli, arada nesneler
    gpu = ProcessGpuScheduler(_gpu_fn, (cfg, (w, h), pitch, fps, calib_gap, has_objects), (h, w, 3))
    # ısınma: ilk CUDA çağrıları birkaç saniye sürer; video başlamadan önce yap
    print("[bilgi] modeller ısıtılıyor...")
    _cap = cv2.VideoCapture(str(video))
    _cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.start * fps))
    _ok, _f = _cap.read()
    _cap.release()
    if _ok and not gpu.wait_ready(_f):
        print("[uyarı] GPU alt süreci zamanında hazır olmadı")

    headless = args.headless_seconds > 0
    record: list[np.ndarray] = []
    draw_ms: list[float] = []
    stats: dict[str, list] = {"calib_ms": [], "calib_lag": [], "source": [], "obj_lag": [], "meas": [], "meas_acc": [], "ref": [], "obj_rows": [], "passes": []}
    if headless:
        print(f"[bilgi] pencere yok (test): {args.headless_seconds:g} sn gerçek zaman hızında çalışılacak")
    else:
        open_window(overlay.out_size)
        print("[bilgi] oynatılıyor. Tuşlar: SPACE g f n m k h j l q")

    def open_at(frame_no: int):
        return FrameReader(str(video), max(0, min(frame_no, total - 1)))

    reader = open_at(int(args.start * fps))
    frames = iter(reader)
    paused, frame, res, idx = False, None, None, 0
    lag, obj_lag, disp_fps, t_prev = 0, 0, fps, time.perf_counter()
    next_t = t_start = time.perf_counter()
    try:
        while True:
            if frame is None or not paused:
                item = next(frames, None)
                if item is None:                       # video bitti: başa sar
                    reader.close()
                    reader, tracker = open_at(0), new_tracker()
                    gpu.request_reset()
                    frames = iter(reader)
                    motion_est.reset()
                    history.reset()
                    smoother.reset()
                    box_flow.reset()
                    pass_det.reset()
                    live_passes.reset()
                    pass_from, pass_last = 0, -1
                    continue
                idx, frame = item
                motion = motion_est.estimate(frame)
                history.add(idx, motion.M)
                if smooth and has_objects:
                    box_flow.push(idx, frame)
                meas = gpu.poll_calibration()
                if meas is not None:
                    lag = idx - meas.frame_idx
                    raw = meas
                    meas = history.bring_forward(meas, idx)
                    if headless and raw.valid:
                        stats["meas"].append((raw.frame_idx, idx, raw.H, meas.H if meas is not None else np.full((3, 3), np.nan),
                                              raw.n_inliers, raw.refined))
                if refine_worker is not None:
                    got_ref = refine_worker.poll()
                    if got_ref is not None:
                        k_ref, (H_ref, ok_ref, _info) = got_ref
                        if headless:
                            stats["ref"].append((idx - k_ref, bool(ok_ref), _info.get("reason", "")))
                        fwd = history.bring_forward(HomographyResult(frame_idx=k_ref, H=H_ref), idx - 1) if ok_ref else None
                        if fwd is not None:
                            tracker.correct(fwd.H)
                res = tracker.update(meas, motion, idx, frame)
                # CPU bütçesi: sürekli çalıştırmak GIL'i tüketip gösterimi 12 fps'e düşürüyordu; `live_every` karede bir
                if refine_worker is not None and res.valid and idx - last_refine_submit >= refine_gap:
                    if refine_worker.submit(idx, frame, res.H):
                        last_refine_submit = idx
                if headless and meas is not None and meas.valid and stats["meas"]:
                    stats["meas_acc"].append((idx, res.source == "keypoints", res.reject_reason))
                gpu.submit(idx, frame, res.H if res.valid else None)
                got = gpu.poll_objects()
                if got is not None:
                    objs_idx, (objs, cols) = got[0], got[1]
                    if smooth:
                        box_flow.start(objs_idx, objs)
                    if headless and objs:
                        stats["obj_rows"].extend(objects_to_rows(objs_idx, objs))
                    if objs is not None and pass_from <= objs_idx <= idx and objs_idx > pass_last:
                        pass_last = objs_idx
                        for cand in pass_det.update(objs_idx, objs):
                            rec_ = rule_eng.classify(cand).to_dict(fps)
                            live_passes.add(rec_)
                            if headless:
                                stats["passes"].append(rec_)
                obj_lag = idx - objs_idx if objs_idx is not None else 0

                now = time.perf_counter()
                disp_fps = 0.9 * disp_fps + 0.1 / max(now - t_prev, 1e-6)
                t_prev = now

            extra = [f"CANLI | gosterim {disp_fps:4.1f} fps | saha modeli {1000 * gpu.calib_latency_s:4.0f} ms, "
                     f"gecikme {lag} kare (kamera hareketiyle telafi)"]
            if has_objects:
                extra.append(f"nesneler: {1000 * gpu.obj_latency_s:4.0f} ms, {obj_lag} kare geriden "
                             f"(kamera hareketiyle telafi)")
            t_draw = time.perf_counter()
            tf = lambda k: history.image_transform(k, idx)  # noqa: E731
            if not objs:
                shown_objs = objs
            elif smooth:
                shown_objs = smoother.update(idx, objs, objs_idx, tf, res.H if res.valid else None,
                                             box_flow.displacement(idx))
            else:
                shown_objs = _compensate(objs, tf(objs_idx))
            vis = overlay.draw(frame, res, opt, idx, fps, extra, team_colors=cols, objects=shown_objs)
            if has_objects:
                live_passes.draw(vis, idx, res.H if res.valid else None, overlay.panel_bottom)
            if headless:
                draw_ms.append(1000 * (time.perf_counter() - t_draw))
                record.append(res.H if res.valid else np.full((3, 3), np.nan))
                stats["calib_ms"].append(1000 * gpu.calib_latency_s)
                stats["calib_lag"].append(lag)
                stats["source"].append(res.source if res.valid else "none")
                if has_objects and objs_idx is not None:
                    stats["obj_lag"].append(obj_lag)
                next_t += 1.0 / fps
                time.sleep(max(0.0, next_t - time.perf_counter()))
                if len(record) >= int(args.headless_seconds * fps):
                    break
                continue
            cv2.imshow(WINDOW, vis)

            if paused:
                key = cv2.waitKey(30) & 0xFF
                next_t = time.perf_counter()
            else:
                next_t += 1.0 / fps
                key = cv2.waitKey(max(1, int((next_t - time.perf_counter()) * 1000))) & 0xFF
                next_t = max(next_t, time.perf_counter() - 0.5 / fps)   # gerideysek birikmeyi önle
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1 or key in (ord("q"), 27):
                break
            toggles = {ord("g"): "grid", ord("f"): "fill", ord("n"): "numbers", ord("m"): "minimap",
                       ord("k"): "keypoints", ord("h"): "hud", ord("c"): "detected_lines",
                   ord("o"): "objects", ord("y"): "goals"}
            if key in toggles:
                setattr(opt, toggles[key], not getattr(opt, toggles[key]))
            elif key == ord(" "):
                paused = not paused
            elif key in (ord("j"), ord("l")):
                step = int(5 * fps) * (1 if key == ord("l") else -1)
                reader.close()
                reader, tracker = open_at(idx + step), new_tracker()
                gpu.request_reset()
                frames = iter(reader)
                motion_est.reset()
                history.reset()
                smoother.reset()
                box_flow.reset()
                pass_det.reset()
                live_passes.reset()
                pass_from, pass_last = max(0, idx + step), -1
                frame = None
    finally:
        reader.close()
        gpu.close()
        if refine_worker is not None:
            refine_worker.close()
        cv2.destroyAllWindows()
    if headless:
        el = time.perf_counter() - t_start
        out = out_dir_for(video, args.start, args.headless_seconds) / "live_selftest.npz"
        np.savez_compressed(out, H=np.array(record), start_frame=int(args.start * fps), fps=fps, size=(w, h),
                            source=np.array(stats["source"]), calib_lag=np.array(stats["calib_lag"]),
                            meas_k=np.array([m[0] for m in stats["meas"]]), meas_at=np.array([m[1] for m in stats["meas"]]),
                            meas_H=np.array([m[2] for m in stats["meas"]]), meas_Hfwd=np.array([m[3] for m in stats["meas"]]),
                            meas_inl=np.array([m[4] for m in stats["meas"]]), meas_ref=np.array([m[5] for m in stats["meas"]]),
                            acc_at=np.array([a[0] for a in stats["meas_acc"]]), acc_ok=np.array([a[1] for a in stats["meas_acc"]]),
                            acc_rej=np.array([a[2] for a in stats["meas_acc"]]),
                            obj_rows=np.array(stats["obj_rows"], dtype=float).reshape(-1, N_COLS))
        print(f"[test] {len(record)} kare / {el:.1f} sn = {len(record) / el:.2f} fps (hedef {fps:g}), "
              f"çizim ort {np.mean(draw_ms):.1f} ms")
        src = np.array(stats["source"])
        print(f"[test] ızgara görünür: %{100 * np.mean(src != 'none'):.1f} (saha noktası %{100 * np.mean(src == 'keypoints'):.1f}, "
              f"kamera hareketi %{100 * np.mean(src == 'camera_motion'):.1f}) | saha noktası modeli: "
              f"ort {np.nanmean(stats['calib_ms']):.0f} ms, gecikme ort {np.mean(stats['calib_lag']):.1f} kare")
        if stats["obj_lag"]:
            ol = np.array(stats["obj_lag"])
            print(f"[test] nesneler: gecikme ort {ol.mean():.1f} kare, max {ol.max()} (en uzun donma karesi "
                  f"{int(np.argmax(ol))}) | sonuç sayısı {gpu.n_obj_results}, hata {gpu.errors}")
        if stats["ref"]:
            rl = np.array([r[0] for r in stats["ref"]]); rok = np.array([r[1] for r in stats["ref"]])
            from collections import Counter
            print(f"[test] çizgilere oturtma (alt süreç): {len(rl)} sonuç ({len(rl) / len(record):.2f}/kare), "
                  f"gecikme ort {rl.mean():.1f} kare, başarılı %{100 * rok.mean():.0f}, "
                  f"{1000 * refine_worker.last_latency_s:.0f} ms | nedenler {Counter(r[2] for r in stats['ref'] if not r[1])}")
        summ = {t: [sum(1 for d in stats["passes"] if d["team"] == t and d["label"] == lab) for lab in ("SUCCESS", "FAIL")]
                for t in ([args.team] if args.team else ["A", "B"])}
        print(f"[test] paslar ({'hedef takım ' + args.team if args.team else 'iki takım'}): "
              + ", ".join(f"{t} başarılı {v[0]} / başarısız {v[1]}" for t, v in summ.items()))
        (out.parent / "live_passes.json").write_text(json.dumps(stats["passes"], ensure_ascii=False, indent=1),
                                                     encoding="utf-8")
        print(f"[test] kayıt: {out}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap =argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("process", help="videoyu işle, ızgaralı MP4 ve kalibrasyon önbelleği üret")
    p.add_argument("video")
    p.add_argument("--config", default="configs/grid_demo.yaml")
    p.add_argument("--start", type=float, default=0.0, help="başlangıç (sn)")
    p.add_argument("--duration", type=float, default=0.0, help="süre (sn), 0 = sonuna kadar")
    p.add_argument("--show", action="store_true", help="işlerken canlı önizleme")
    p.add_argument("--keypoints", action="store_true", help="MP4'e saha noktalarını da çiz")
    p.add_argument("--no-video", action="store_true", help="MP4 üretme, yalnızca kalibrasyon önbelleği (daha hızlı)")
    p.add_argument("--use-cache", action="store_true",
                   help="saha noktası önbelleğini kullan (varsayılan kapalı; doğrulanana kadar, D-018)")
    p.add_argument("--no-quality", action="store_true", help="kalite göstergelerini hesaplama")
    p.add_argument("--no-objects", action="store_true", help="oyuncu/hakem/top tespitini atla")
    p.set_defaults(func=cmd_process)
    q = sub.add_parser("play", help="önbellekle gerçek hızda oynat")
    q.add_argument("video")
    q.add_argument("--cache", default="", help="varsayılan: tam çalıştırmanın önbelleği; kısmi için .../s<start>_d<dur>/calibration.npz")
    q.add_argument("--team", choices=["A", "B"], help="hedef takım: yalnızca bu takımın pasları (D-040)")
    q.set_defaults(func=cmd_play)
    ob = sub.add_parser("objects", help="mevcut kalibrasyonla yalnızca oyuncu/hakem/top adımını yeniden çalıştır")
    ob.add_argument("video")
    ob.add_argument("--config", default="configs/grid_demo.yaml")
    ob.add_argument("--no-video", action="store_true")
    ob.set_defaults(func=cmd_objects)
    lv = sub.add_parser("live", help="önceden işlemeden, oynatırken anlık ızgara")
    lv.add_argument("video")
    lv.add_argument("--config", default="configs/grid_demo.yaml")
    lv.add_argument("--start", type=float, default=0.0, help="başlangıç (sn)")
    lv.add_argument("--headless-seconds", type=float, default=0.0,
                    help="test: pencere açmadan bu kadar saniye gerçek zaman hızında çalış ve sonucu kaydet")
    lv.add_argument("--team", choices=["A", "B"], help="hedef takım: yalnızca bu takımın pasları (D-040)")
    lv.set_defaults(func=cmd_live)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
