"""Hızlandırılmış ısı haritası tepe bulma, NBJW orijinaliyle birebir aynı sonuç vermeli."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NBJW = ROOT / "third_party" / "nbjw"
sys.path.insert(0, str(ROOT / "src"))
torch = pytest.importorskip("torch")
if not NBJW.exists():
    pytest.skip("third_party/nbjw yok", allow_module_level=True)
sys.path.insert(0, str(NBJW))

from utils.utils_heatmap import (  # noqa: E402
    get_keypoints_from_heatmap_batch_maxpool, get_keypoints_from_heatmap_batch_maxpool_l)

from sgi.detection.keypoints.adapters.nbjw import _heatmap_peaks  # noqa: E402


def _heatmaps(seed: int, channels: int) -> "torch.Tensor":
    g = torch.Generator().manual_seed(seed)
    hm = torch.rand(1, channels, 270, 480, generator=g) ** 8      # çoğu küçük, birkaç belirgin tepe
    hm[:, ::3] *= 0.0                                              # tamamen boş kanallar (topk sıfırlar)
    return hm


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_keypoint_peaks_identical(seed):
    hm = _heatmaps(seed, 57)
    ref = get_keypoints_from_heatmap_batch_maxpool(hm)
    new = _heatmap_peaks(hm, max_keypoints=1, min_dist=15, border_pad=True)
    assert torch.allclose(ref.float(), new, atol=0, rtol=0)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_line_peaks_identical(seed):
    hm = _heatmaps(seed, 23)
    ref = get_keypoints_from_heatmap_batch_maxpool_l(hm)
    new = _heatmap_peaks(hm, max_keypoints=2, min_dist=10, border_pad=False)
    assert torch.allclose(ref.float(), new, atol=0, rtol=0)
