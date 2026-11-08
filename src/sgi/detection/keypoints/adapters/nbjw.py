"""NBJW (No Bells, Just Whistles) saha keypoint adaptörü.

Kaynak: https://github.com/mguti97/No-Bells-Just-Whistles (GPL-2.0), third_party/nbjw altında.
Model iki HRNetV2-w48 ağıdır: 57 saha noktası + 23 çizgi uç noktası. Çizgi kesişimlerinden
ek noktalar (58..73) üretilir. Bu adaptör yalnızca **zemin düzlemindeki** noktaları döndürür;
homografiyi M4 (sgi.mapping.homography) hesaplar.

NBJW saha sistemi bizimkiyle aynı eksen yönündedir (x: sol kale -> sağ, y: üst/uzak taç -> alt/yakın taç);
`keypoint_world_coords_2D` 0..105 x 0..68 aralığında tanımlanıp merkeze kaydırılmıştır.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from sgi.interfaces import PitchKeypointDetector
from sgi.types import PitchKeypoint

_INPUT_W, _INPUT_H = 960, 540
_OFF_GROUND_KP = {12, 15, 16, 19}   # kale üst direği noktaları (z = -2.44 m)


def _heatmap_peaks(heatmap, max_keypoints: int, min_dist: int, border_pad: bool, scale: int = 2):
    """NBJW `get_keypoints_from_heatmap_batch_maxpool(_l)` ile birebir aynı çıktı, daha hızlı.

    - Max-pool ayrılabilir olduğu için (2k+1)x(2k+1) pencere, (2k+1)x1 + 1x(2k+1) olarak uygulanır (~7x hızlı).
    - Kanal başına Python döngüsü yerine vektörel indeksleme kullanılır.
    border_pad=True: keypoint modeli (kenar 1.0 ile doldurulur), False: çizgi modeli (-inf dolgu).
    Dönüş: CPU tensörü [B, C, K, 3] = (x, y, skor), x/y ısı haritası * scale.
    """
    import torch
    import torch.nn.functional as F

    b, c, _, width = heatmap.shape
    k = 2 * min_dist + 1
    if border_pad:
        padded = F.pad(heatmap, (min_dist, min_dist, min_dist, min_dist), mode="constant", value=1.0)
        pooled = F.max_pool2d(F.max_pool2d(padded, (k, 1), stride=1), (1, k), stride=1)
    else:
        pooled = F.max_pool2d(F.max_pool2d(heatmap, (k, 1), stride=1, padding=(min_dist, 0)),
                              (1, k), stride=1, padding=(0, min_dist))
    heatmap = heatmap * (pooled == heatmap)
    scores, idx = torch.topk(heatmap.view(b, c, -1), max_keypoints, sorted=True)
    rows = torch.div(idx, width, rounding_mode="floor")
    cols = idx % width
    out = torch.stack([(cols * scale).float(), (rows * scale).float(), scores.float()], dim=-1)
    return out.cpu()


class NBJWKeypointDetector(PitchKeypointDetector):
    def __init__(self, repo_dir: str, weights_kp: str, weights_lines: str,
                 kp_threshold: float = 0.1486, line_threshold: float = 0.3880,
                 device: str = "cuda", half: bool = True, channels_last: bool = False, pitch_length: float = 105.0,
                 pitch_width: float = 68.0, root: str | Path = ".", use_lines: bool = True,
                 trt_engine: str | None = None) -> None:
        import torch
        import yaml

        root = Path(root)
        repo = (root / repo_dir).resolve()
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from model.cls_hrnet import get_cls_net
        from model.cls_hrnet_l import get_cls_net as get_cls_net_l
        from utils import utils_calib, utils_heatmap

        self._torch = torch
        self._hm = utils_heatmap
        self.device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        self.half = half and self.device.type == "cuda"
        self.channels_last = channels_last and self.device.type == "cuda"
        self.kp_threshold = kp_threshold
        self.line_threshold = line_threshold
        # use_lines: false -> çizgi ağı çalışmaz (GPU süresi yarıya iner); çizgi kesişiminden gelen ek noktalar olmaz
        self.use_lines = use_lines

        def load(builder, cfg_name: str, weights: str):
            cfg = yaml.safe_load((repo / "config" / cfg_name).read_text())
            net = builder(cfg)
            net.load_state_dict(torch.load(root / weights, map_location=self.device))
            net.to(self.device).eval()
            net = net.half() if self.half else net
            # channels_last: model başına ~%11 hız, ancak fp16 sonuçları değişir ve keypoint'ler 1-2 px kayabilir
            # (regresyon: H farkı medyan 0,13 m). Tekrarlanabilirlik için varsayılan kapalı (D-015).
            return net.to(memory_format=torch.channels_last) if self.channels_last else net

        # trt_engine: nokta ağı TensorRT motoruyla (scripts/build_trt.py); verilmezse ya da yoksa PyTorch
        engine = root / trt_engine if trt_engine else None
        if engine is not None and engine.exists():
            from sgi.detection.trt_runner import TrtModule
            self.model_kp = TrtModule(engine)
        else:
            self.model_kp = load(get_cls_net, "hrnetv2_w48.yaml", weights_kp)
        self.model_lines = load(get_cls_net_l, "hrnetv2_w48_l.yaml", weights_lines) if use_lines else None

        # NBJW dünya koordinatları (merkez orijinli) -> bizim sistem (köşe orijinli)
        main = np.asarray(utils_calib.keypoint_world_coords_2D) + [pitch_length / 2, pitch_width / 2]
        aux = np.asarray(utils_calib.keypoint_aux_world_coords_2D) + [pitch_length / 2, pitch_width / 2]
        self._world = {i + 1: tuple(p) for i, p in enumerate(main)}
        self._world.update({i + 58: tuple(p) for i, p in enumerate(aux)})

    def detect(self, frame_bgr: np.ndarray) -> list[PitchKeypoint]:
        import cv2
        torch = self._torch

        h0, w0 = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(cv2.resize(frame_bgr, (_INPUT_W, _INPUT_H), interpolation=cv2.INTER_AREA),
                           cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        x = x.half() / 255.0 if self.half else x.float() / 255.0
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)

        with torch.no_grad():
            hm_kp = self.model_kp(x).float()
            kp_coords = _heatmap_peaks(hm_kp[:, :-1], max_keypoints=1, min_dist=15, border_pad=True)
            if self.model_lines is not None:
                hm_lines = self.model_lines(x).float()
                line_coords = _heatmap_peaks(hm_lines[:, :-1], max_keypoints=2, min_dist=10, border_pad=False)
        kp_dict = self._hm.coords_to_dict(kp_coords, threshold=self.kp_threshold)
        lines_dict = (self._hm.coords_to_dict(line_coords, threshold=self.line_threshold)
                      if self.model_lines is not None else [{} for _ in kp_dict])
        points = self._hm.complete_keypoints(kp_dict, lines_dict, w=_INPUT_W, h=_INPUT_H, normalize=True)[0]

        out = []
        for kp_id, p in points.items():
            if kp_id in _OFF_GROUND_KP or kp_id not in self._world:
                continue
            out.append(PitchKeypoint(kp_id=kp_id,
                                     image_xy=(float(p["x"]) * w0, float(p["y"]) * h0),
                                     pitch_xy=self._world[kp_id],
                                     conf=float(p.get("p", 1.0))))
        return out
