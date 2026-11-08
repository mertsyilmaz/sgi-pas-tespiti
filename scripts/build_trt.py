"""TensorRT motorlarını üretir: saha noktası ağı (NBJW HRNet) ve nesne dedektörü (YOLO). fp16.

Motorlar bu GPU'ya ve TensorRT sürümüne özeldir; `models/trt/` altına yazılır (git dışı).
Kullanım: .venv/Scripts/python.exe scripts/build_trt.py [--only kp|det]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "models" / "trt"


def build_engine(onnx_path: Path, engine_path: Path, workspace_gb: float = 1.5) -> None:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    # TensorRT 11: ağ güçlü tiplidir; hassasiyet ONNX'teki tiplerden gelir (fp16 ONNX -> fp16 motor)
    net = builder.create_network(0)
    parser = trt.OnnxParser(net, logger)
    if not parser.parse(onnx_path.read_bytes()):
        raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    data = builder.build_serialized_network(net, cfg)
    if data is None:
        raise RuntimeError("TensorRT motoru üretilemedi")
    engine_path.write_bytes(bytes(data))


def build_kp(cfg: dict) -> None:
    import torch

    from sgi.detection.keypoints.adapters.nbjw import _INPUT_H, _INPUT_W, NBJWKeypointDetector

    kcfg = dict(cfg["pitch_keypoint_detector"])
    kcfg.pop("type")
    kcfg.pop("trt_engine", None)
    kcfg.update(half=True, use_lines=False)
    det = NBJWKeypointDetector(**kcfg, root=ROOT)
    onnx_path = OUT / "nbjw_kp.onnx"
    x = torch.zeros(1, 3, _INPUT_H, _INPUT_W, device=det.device, dtype=torch.float16)
    with torch.no_grad():
        torch.onnx.export(det.model_kp, x, str(onnx_path), input_names=["x"], output_names=["hm"], opset_version=17,
                          dynamo=False)
    build_engine(onnx_path, OUT / "nbjw_kp_fp16.engine")
    print(f"[çıktı] {OUT / 'nbjw_kp_fp16.engine'}")


def build_det(cfg: dict) -> None:
    from ultralytics import YOLO

    d = cfg["objects"]["detector"]
    src = ROOT / d["weights"]
    eng = YOLO(str(src)).export(format="engine", imgsz=d.get("imgsz", 1280), half=True, batch=1, workspace=1.5,
                                verbose=False)
    dst = OUT / (src.stem + f"_{d.get('imgsz', 1280)}_fp16.engine")
    shutil.move(eng, dst)
    print(f"[çıktı] {dst}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grid_demo.yaml")
    ap.add_argument("--only", choices=["kp", "det"])
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    if args.only in (None, "kp"):
        build_kp(cfg)
    if args.only in (None, "det"):
        build_det(cfg)


if __name__ == "__main__":
    main()
