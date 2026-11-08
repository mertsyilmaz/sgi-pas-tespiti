import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sgi.detection.keypoints.cache import KeypointCache  # noqa: E402
from sgi.types import PitchKeypoint  # noqa: E402


def _video(tmp_path: Path) -> Path:
    v = tmp_path / "clip.mp4"
    v.write_bytes(b"fake video")
    return v


def test_roundtrip_including_empty_frames(tmp_path):
    video, cfg = _video(tmp_path), {"type": "nbjw", "kp_threshold": 0.15}
    c = KeypointCache(tmp_path / "cache", video, cfg)
    c.put(0, [PitchKeypoint(1, (10.5, 20.25), (0.0, 0.0), 0.9), PitchKeypoint(30, (100.0, 200.0), (105.0, 68.0))])
    c.put(1, [])                                   # saha noktası bulunamayan kare de önbelleğe girer
    c.save()

    c2 = KeypointCache(tmp_path / "cache", video, cfg)
    assert c2.covers([0, 1]) and not c2.covers([2])
    assert c2.get(1) == []
    k = c2.get(0)
    assert [p.kp_id for p in k] == [1, 30]
    assert k[0].image_xy == (10.5, 20.25) and k[1].pitch_xy == (105.0, 68.0) and k[0].conf == 0.9


def test_detector_config_change_uses_new_cache(tmp_path):
    video = _video(tmp_path)
    c = KeypointCache(tmp_path / "cache", video, {"kp_threshold": 0.15})
    c.put(0, [])
    c.save()
    assert not KeypointCache(tmp_path / "cache", video, {"kp_threshold": 0.20}).covers([0])
