"""TensorRT motoru ile tek girdili / tek çıktılı ağ çalıştırma (torch CUDA tensörleriyle, kopyasız).

Motor (`.engine`) `scripts/build_trt.py` ile bu makinede üretilir; başka GPU / TensorRT sürümüne taşınamaz.
"""
from __future__ import annotations

from pathlib import Path


class TrtModule:
    def __init__(self, engine_path: str | Path) -> None:
        import tensorrt as trt
        import torch

        self._torch = torch
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"TensorRT motoru yüklenemedi: {engine_path}")
        self.ctx = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inp = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        self.out = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
        dt = {trt.DataType.HALF: torch.float16, trt.DataType.FLOAT: torch.float32}
        self.in_dtype = dt[self.engine.get_tensor_dtype(self.inp)]
        self.out_dtype = dt[self.engine.get_tensor_dtype(self.out)]
        self.out_shape = tuple(self.engine.get_tensor_shape(self.out))
        self._buf = torch.empty(self.out_shape, dtype=self.out_dtype, device="cuda")
        self._stream = torch.cuda.Stream()        # varsayılan akışta TensorRT her çağrıda ek senkronizasyon yapıyor

    def __call__(self, x):
        torch = self._torch
        x = x.to(self.in_dtype).contiguous()
        self.ctx.set_tensor_address(self.inp, x.data_ptr())
        self.ctx.set_tensor_address(self.out, self._buf.data_ptr())
        cur = torch.cuda.current_stream()
        self._stream.wait_stream(cur)
        self.ctx.execute_async_v3(self._stream.cuda_stream)
        cur.wait_stream(self._stream)
        return self._buf
