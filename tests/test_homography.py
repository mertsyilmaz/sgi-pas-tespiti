import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sgi.mapping.homography import apply_h, camera_plausibility  # noqa: E402

W, H = 1920, 1080


def synthetic_h(f=2000.0, cam=(52.5, 110.0, 20.0), target=(52.5, 34.0, 0.0)) -> np.ndarray:
    """Bilinen bir pinhole kameradan image->pitch homografisi üretir."""
    C, T = np.array(cam, float), np.array(target, float)
    fwd = (T - C) / np.linalg.norm(T - C)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.vstack([right, down, fwd])
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
    P = K @ np.c_[R, -R @ C]
    G = P[:, [0, 1, 3]]                       # pitch (x, y, 1) -> image
    Hm = np.linalg.inv(G)
    return Hm / Hm[2, 2]


def test_real_camera_passes_and_is_recovered():
    chk = camera_plausibility(synthetic_h(), (W, H))
    assert chk.ok, chk.reason
    assert abs(chk.focal_px - 2000) < 1
    assert abs(chk.height_m - 20) < 0.01
    assert abs(chk.aspect - 1) < 1e-6


def test_panned_and_near_degenerate_views_pass():
    # sağ ceza sahasına dönük, hafif dönük ve tam karşıdan bakan kameralar
    for target in [(90.0, 34.0, 0.0), (54.0, 34.0, 0.0), (52.5, 34.0, 0.0), (20.0, 20.0, 0.0)]:
        for f in (1200.0, 2000.0, 6000.0):
            chk = camera_plausibility(synthetic_h(f=f, target=target), (W, H))
            assert chk.ok, (target, f, chk.reason)
            assert abs(chk.focal_px - f) / f < 1e-3 and abs(chk.height_m - 20) < 0.05


def test_pitch_to_image_sign_is_independent_of_scaling():
    from sgi.mapping.homography import pitch_to_image
    Hm = synthetic_h()
    pitch_pts = np.c_[[[52.5, 34.0], [80.0, 60.0], [10.0, 10.0]], np.ones(3)]
    for s in (1.0, -1.0, -3.7):                                # H / H[2,2] işareti değiştirebilir
        w = (pitch_pts @ pitch_to_image(s * Hm, (W, H)).T)[:, 2]
        assert np.all(w > 0)


def test_projection_roundtrip_center_of_image_on_pitch():
    p = apply_h(synthetic_h(), np.array([[W / 2, H / 2]]))[0]
    assert np.allclose(p, [52.5, 34.0], atol=1e-6)


def test_distorted_pitch_is_rejected():
    # Açılı bakışta saha x yönünde 2.5 kat gerilirse hiçbir pinhole kamera bunu açıklayamaz.
    # Not: Tam karşıdan (pan=0) bakışta bu bozulma farklı zoom/mesafeyle birebir açıklanabilir;
    # kontrolün kör noktasıdır (14_CALIBRATION_QUALITY §2.2).
    for target in [(85.0, 34.0, 0.0), (20.0, 20.0, 0.0)]:
        squash = np.diag([2.5, 1.0, 1.0]) @ synthetic_h(target=target)
        assert not camera_plausibility(squash, (W, H)).ok


def test_camera_below_ground_height_is_rejected():
    assert not camera_plausibility(synthetic_h(cam=(52.5, 110.0, 0.5)), (W, H)).ok


# --- HomographyTracker: seyrek (canlı) ölçüm davranışı (D-029) ---
from sgi.mapping.homography import HomographyTracker  # noqa: E402
from sgi.types import CameraMotion, HomographyResult  # noqa: E402

I3 = np.eye(3)


def _meas(H, idx, inliers=8):
    return HomographyResult(frame_idx=idx, H=H, valid=True, n_inliers=inliers)


def _shift(H, dx_px):
    """Görüntüde dx piksel kaydırılmış ölçüm (aynı kamera, biraz farklı H)."""
    T = np.array([[1, 0, -dx_px], [0, 1, 0], [0, 0, 1.0]])
    return H @ T


def _dist(tr, H1, H2):
    return tr._distance_m(H1, H2)


def test_sparse_measurement_gets_full_weight():
    H0 = synthetic_h()
    tr = HomographyTracker((W, H), measurement_weight=0.35, max_jump_m=50)
    tr.update(_meas(H0, 0), CameraMotion(None), 0)
    for i in range(1, 12):                                     # 11 kare ölçümsüz (canlı: model meşgul)
        tr.update(None, CameraMotion(I3), i)
    Hm = _shift(H0, 20)
    out = tr.update(_meas(Hm, 12), CameraMotion(I3), 12)
    assert _dist(tr, out.H, Hm) < 0.05 * _dist(tr, H0, Hm)       # ~%99 ağırlık


def test_every_frame_measurement_keeps_base_weight():
    H0 = synthetic_h()
    tr = HomographyTracker((W, H), measurement_weight=0.35, max_jump_m=50)
    tr.update(_meas(H0, 0), CameraMotion(None), 0)
    Hm = _shift(H0, 20)
    out = tr.update(_meas(Hm, 1), CameraMotion(I3), 1)
    r = _dist(tr, out.H, H0) / _dist(tr, H0, Hm)
    assert 0.25 < r < 0.45                                     # çevrimdışı davranış değişmedi (~0,35)


def test_weak_inconsistent_measurement_accepted_after_zoom():
    H0 = synthetic_h()
    Z = np.diag([1.2, 1.2, 1.0])                               # %20 zoom
    Hm = _shift(H0, 300)
    tr = HomographyTracker((W, H), max_jump_m=1.0, strong_measurement_inliers=10)
    tr.update(_meas(H0, 0), CameraMotion(None), 0)
    assert tr.update(_meas(Hm, 1, inliers=6), CameraMotion(I3), 1).reject_reason == "jump"
    tr2 = HomographyTracker((W, H), max_jump_m=1.0, strong_measurement_inliers=10)
    tr2.update(_meas(H0, 0), CameraMotion(None), 0)
    out = tr2.update(_meas(Hm, 1, inliers=6), CameraMotion(Z), 1)
    assert out.source == "keypoints" and out.reject_reason == ""


def test_correct_resets_evidence_age():
    H0 = synthetic_h()
    tr = HomographyTracker((W, H), measurement_weight=0.35, max_jump_m=50)
    tr.update(_meas(H0, 0), CameraMotion(None), 0)
    for i in range(1, 12):
        tr.update(None, CameraMotion(I3), i)
    tr.correct(H0)                                             # taze çizgi oturtması
    Hm = _shift(H0, 20)
    out = tr.update(_meas(Hm, 12), CameraMotion(I3), 12)
    r = _dist(tr, out.H, H0) / _dist(tr, H0, Hm)
    assert r < 0.45                                            # gecikmeli ölçüm taze düzeltmeyi ezmez
