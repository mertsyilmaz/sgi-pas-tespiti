"""Canlı mod için asenkron kalibrasyon (D-016).

Saha noktası modeli video hızından yavaştır (~4.5 kare/sn). Bu yüzden:
- Model ayrı bir iş parçacığında, boşaldıkça **en son** kareyi işler (arada kalan kareler atlanır).
- Ana döngü her karede kamera hareketini hesaplar ve kaydeder.
- Model sonucu k karesine aittir ve birkaç kare geç gelir; kayıtlı kamera hareketleriyle güncel kareye taşınır:
  H_c = H_k · M_{k+1} · ... · M_c   (M_t: t karesi -> t-1 karesi)
  Aradaki kareler arasında hareket kestirilemediyse (sahne kesmesi) sonuç atılır.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import replace
from typing import Optional

import numpy as np

from sgi.interfaces import PitchKeypointDetector
from sgi.mapping.homography import HomographyEstimator
from sgi.types import HomographyResult


class AsyncCalibrator:
    def __init__(self, detector: PitchKeypointDetector, estimator: HomographyEstimator) -> None:
        self.detector, self.estimator = detector, estimator
        self._cv = threading.Condition()
        self._pending: Optional[tuple[int, np.ndarray]] = None
        self._result: Optional[HomographyResult] = None
        self._stop = False
        self.last_latency_s = float("nan")
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def submit(self, frame_idx: int, frame: np.ndarray) -> None:
        """Model boştaysa bu kareyi işler; meşgulse bekleyen kare en yenisiyle değiştirilir."""
        with self._cv:
            self._pending = (frame_idx, frame)
            self._cv.notify()

    def poll(self) -> Optional[HomographyResult]:
        with self._cv:
            res, self._result = self._result, None
            return res

    def _run(self) -> None:
        while True:
            with self._cv:
                while self._pending is None and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                idx, frame = self._pending
                self._pending = None
            t0 = time.perf_counter()
            res = self.estimator.estimate(idx, self.detector.detect(frame))
            self.last_latency_s = time.perf_counter() - t0
            with self._cv:
                self._result = res

    def close(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify()
        self._t.join(timeout=2.0)


class AsyncWorker:
    """Genel arka plan işçisi: en son gönderilen işi işler, arada gelenleri atlar (canlı mod, D-022).

    `fn(*args)` ayrı bir iş parçacığında çalışır; `poll()` en son sonucu (ve hangi kareye ait olduğunu) verir.
    """

    def __init__(self, fn) -> None:
        self.fn = fn
        self._cv = threading.Condition()
        self._pending = None
        self._result = None
        self._stop = False
        self.last_latency_s = float("nan")
        self.errors = 0
        self.n_results = 0
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def submit(self, frame_idx: int, *args) -> None:
        with self._cv:
            self._pending = (frame_idx, args)
            self._cv.notify()

    def busy(self) -> bool:
        with self._cv:
            return self._pending is not None

    def poll(self):
        """(kare, sonuç) ya da None."""
        with self._cv:
            r, self._result = self._result, None
            return r

    def _run(self) -> None:
        while True:
            with self._cv:
                while self._pending is None and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                idx, args = self._pending
                self._pending = None
            t0 = time.perf_counter()
            try:
                res = self.fn(*args)
            except Exception:                         # iş parçacığı sessizce ölmesin: kaydet, devam et
                import traceback
                self.errors += 1
                if self.errors <= 3:
                    traceback.print_exc()
                continue
            self.last_latency_s = time.perf_counter() - t0
            self.n_results += 1
            with self._cv:
                self._result = (idx, res)

    def close(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify()
        self._t.join(timeout=2.0)


class GpuScheduler:
    """Canlı modda iki GPU modelini tek kuyrukta sırayla çalıştırır (D-023).

    İki model ayrı iş parçacıklarında GPU'yu aynı anda kullanınca birbirini yavaşlatıyordu (saha noktası
    modeli 214 -> ~290 ms). Burada tek bir iş parçacığı vardır: kalibrasyon zamanı geldiyse (en az `calib_gap`
    kare geçtiyse) önce saha noktası modeli, değilse nesne dedektörü en son kare üzerinde çalışır.
    """

    def __init__(self, calib_fn, objects_fn=None, calib_gap: int = 10, lost_gap: int = 2) -> None:
        # lost_gap: ızgara yokken (yakın çekimden/zoom'dan çıkış) saha modeli bu kadar karede bir öne alınır (D-029)
        self.calib_fn, self.objects_fn, self.calib_gap, self.lost_gap = calib_fn, objects_fn, calib_gap, lost_gap
        self._cv = threading.Condition()
        self._latest = None                      # (kare, görüntü, H)
        self._calib_res = None
        self._obj_res = None
        self._last_calib_idx = -10 ** 9
        self._stop = False
        self.calib_latency_s = self.obj_latency_s = float("nan")
        self.errors = 0
        self.n_obj_results = 0
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def submit(self, frame_idx: int, frame: np.ndarray, H: Optional[np.ndarray]) -> None:
        with self._cv:
            self._latest = (frame_idx, frame, H)
            self._cv.notify()

    def poll_calibration(self):
        with self._cv:
            r, self._calib_res = self._calib_res, None
            return r

    def poll_objects(self):
        with self._cv:
            r, self._obj_res = self._obj_res, None
            return r

    def _run(self) -> None:
        while True:
            with self._cv:
                while self._latest is None and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                idx, frame, H = self._latest
                self._latest = None
                gap = self.lost_gap if H is None else self.calib_gap
                do_calib = idx - self._last_calib_idx >= gap or self.objects_fn is None
                if do_calib:
                    self._last_calib_idx = idx
            t0 = time.perf_counter()
            try:
                if do_calib:
                    res = self.calib_fn(idx, frame)
                    self.calib_latency_s = time.perf_counter() - t0
                    with self._cv:
                        self._calib_res = res
                else:
                    res = self.objects_fn(idx, frame, H)
                    self.obj_latency_s = time.perf_counter() - t0
                    self.n_obj_results += 1
                    with self._cv:
                        self._obj_res = (idx, res)
            except Exception:
                import traceback
                self.errors += 1
                if self.errors <= 3:
                    traceback.print_exc()

    def close(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify()
        self._t.join(timeout=2.0)


class MotionHistory:
    """Son kareler için kamera hareketi kaydı; geç gelen ölçümü güncel kareye taşır."""

    def __init__(self, maxlen: int = 250) -> None:
        self._m: deque[tuple[int, Optional[np.ndarray]]] = deque(maxlen=maxlen)

    def reset(self) -> None:
        self._m.clear()

    def add(self, frame_idx: int, M: Optional[np.ndarray]) -> None:
        self._m.append((frame_idx, M))

    def image_transform(self, k: int, current_idx: int) -> Optional[np.ndarray]:
        """k karesinin görüntü koordinatlarını güncel kareye taşıyan 3x3 dönüşüm (kamera hareketi telafisi).
        M_t: t -> t-1 olduğundan k -> c = (M_{k+1} ··· M_c)⁻¹. Arada hareket kestirilemediyse None."""
        if k == current_idx:
            return np.eye(3)
        chain = [M for idx, M in self._m if k < idx <= current_idx]
        if len(chain) != current_idx - k or any(M is None for M in chain):
            return None
        P = np.eye(3)
        for M in chain:
            P = P @ M
        T = np.linalg.inv(P)
        return T / T[2, 2]

    def bring_forward(self, res: HomographyResult, current_idx: int) -> Optional[HomographyResult]:
        if res.H is None:
            return res
        k = res.frame_idx
        if k == current_idx:
            return res
        chain = [M for idx, M in self._m if k < idx <= current_idx]
        if len(chain) != current_idx - k or any(M is None for M in chain):
            return None                      # kayıt eksik ya da arada sahne kesmesi: ölçüm bayat
        H = res.H.copy()
        for M in chain:
            H = H @ M
        return replace(res, H=H / H[2, 2], frame_idx=current_idx)


def _process_worker_main(factory, factory_args, jobs, results) -> None:
    """Alt süreç döngüsü: işlevi bir kez kur, gelen işleri sırayla işle (None = dur)."""
    fn = factory(*factory_args)
    while True:
        job = jobs.get()
        if job is None:
            return
        idx, args = job
        t0 = time.perf_counter()
        try:
            res = fn(*args)
        except Exception as e:                       # noqa: BLE001 — alt süreç ölmesin
            results.put((idx, None, repr(e), 0.0))
            continue
        results.put((idx, res, "", time.perf_counter() - t0))


class ProcessWorker:
    """CPU ağırlıklı işi ayrı bir süreçte çalıştırır (D-029).

    Aynı süreçteki bir iş parçacığı Python kilidini (GIL) paylaşır: çizgilere oturtma arka planda sürekli
    çalışınca canlı gösterim 25 -> 12 fps'e düştü. Ayrı süreç kendi çekirdeğinde çalışır. Aynı anda en fazla
    bir iş bekler; meşgulken gelen gönderimler atlanır (en güncel kare zaten kısa süre sonra gönderilecek).
    `factory(*factory_args)` alt süreçte çağrılır ve işi yapan işlevi döndürür (Windows 'spawn': seçilebilir olmalı).
    """

    def __init__(self, factory, factory_args: tuple = ()) -> None:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        self._jobs, self._results = ctx.Queue(), ctx.Queue()
        self._proc = ctx.Process(target=_process_worker_main, args=(factory, factory_args, self._jobs, self._results),
                                 daemon=True)
        self._proc.start()
        self._busy = False
        self.last_latency_s = float("nan")
        self.errors = 0
        self.n_results = 0

    def busy(self) -> bool:
        return self._busy

    def submit(self, frame_idx: int, *args) -> bool:
        if self._busy:
            return False
        self._jobs.put((frame_idx, args))
        self._busy = True
        return True

    def poll(self):
        """(kare, sonuç) ya da None."""
        import queue
        try:
            idx, res, err, dt = self._results.get_nowait()
        except queue.Empty:
            return None
        self._busy = False
        if err:
            self.errors += 1
            if self.errors <= 3:
                print(f"[uyarı] arka plan süreci hatası: {err}")
            return None
        self.last_latency_s, self.n_results = dt, self.n_results + 1
        return idx, res

    def close(self) -> None:
        try:
            self._jobs.put(None)
            self._proc.join(timeout=2.0)
        finally:
            if self._proc.is_alive():
                self._proc.terminate()


def _gpu_proc_main(factory, factory_args, shm_name, shape, meta, lock, new_frame, results) -> None:
    """GPU alt süreci: boşaldıkça paylaşımlı bellekteki en güncel kareyi işler (D-031)."""
    from multiprocessing import shared_memory
    shm = shared_memory.SharedMemory(name=shm_name)
    buf = np.ndarray(shape, np.uint8, buffer=shm.buf)
    step = factory(*factory_args)
    ready = False
    while True:
        if not new_frame.wait(timeout=0.5):
            continue
        with lock:
            new_frame.clear()
            if meta[0] < 0:                                  # dur
                break
            idx, cmd = int(meta[1]), int(meta[2])
            meta[2] = 0
            H = None if np.isnan(meta[3]) else np.array(meta[3:12]).reshape(3, 3)
            frame = buf.copy()
        t0 = time.perf_counter()
        try:
            if not ready:
                step(idx, frame, None, "warmup")
                ready = True
                results.put(("ready", idx, None, 0.0))
                continue
            kind, res = step(idx, frame, H, "reset" if cmd == 1 else "")
            results.put((kind, idx, res, time.perf_counter() - t0))
        except Exception as e:                               # noqa: BLE001 — alt süreç ölmesin
            results.put(("error", idx, repr(e), 0.0))
    shm.close()


class ProcessGpuScheduler:
    """GpuScheduler'ın ayrı süreçte çalışan karşılığı (D-031).

    Aynı süreçteki GPU iş parçacığı (YOLO ön/son işleme, BoT-SORT, takım ataması Python'da) ana döngüyle Python
    kilidini paylaşıyordu: saha modeli tek başına 109 ms iken canlıda ~200 ms, çizim 12 ms iken ~24 ms sürüyordu.
    Modeller alt süreçte kurulur. Ana süreç her kareyi paylaşımlı belleğe yazar (kopyalama ~0,3 ms); alt süreç
    boşaldığı anda en güncel kareyi alır (beklemesiz). `step(idx, frame, H, cmd)` saha modeli mi nesne dedektörü
    mü çalışacağına kendisi karar verir ve ("calib", HomographyResult) ya da ("obj", (nesneler, renkler)) döndürür.
    """

    def __init__(self, factory, factory_args: tuple, frame_shape: tuple[int, int, int]) -> None:
        import multiprocessing as mp
        from multiprocessing import shared_memory
        ctx = mp.get_context("spawn")
        self._shape = tuple(frame_shape)
        self._shm = shared_memory.SharedMemory(create=True, size=int(np.prod(self._shape)))
        self._buf = np.ndarray(self._shape, np.uint8, buffer=self._shm.buf)
        self._meta = ctx.Array("d", 12, lock=False)          # [durum, kare, komut, H(9)]
        self._lock, self._new, self._results = ctx.Lock(), ctx.Event(), ctx.Queue()
        self._proc = ctx.Process(target=_gpu_proc_main, daemon=True,
                                 args=(factory, factory_args, self._shm.name, self._shape, self._meta,
                                       self._lock, self._new, self._results))
        self._proc.start()
        self._calib = self._obj = None
        self._reset = False
        self.calib_latency_s = self.obj_latency_s = float("nan")
        self.n_obj_results = 0
        self.errors = 0

    def _write(self, idx: int, frame: np.ndarray, H: Optional[np.ndarray]) -> None:
        with self._lock:
            self._buf[...] = frame
            self._meta[0], self._meta[1] = 0, idx
            if self._reset:
                self._meta[2], self._reset = 1, False
            self._meta[3:12] = [float("nan")] * 9 if H is None else list(np.asarray(H, float).ravel())
            self._new.set()

    def wait_ready(self, frame: np.ndarray, timeout_s: float = 180.0) -> bool:
        """Modeller kurulup ısınana kadar bekle (ilk CUDA çağrıları birkaç saniye sürer)."""
        import queue
        self._write(-1, frame, None)
        try:
            kind, *_ = self._results.get(timeout=timeout_s)
        except queue.Empty:
            return False
        return kind == "ready"

    def request_reset(self) -> None:
        self._reset = True                       # bir sonraki kareyle iletilir (geri sarma/atlama)

    def _drain(self) -> None:
        import queue
        while True:
            try:
                kind, idx, res, dt = self._results.get_nowait()
            except queue.Empty:
                return
            if kind == "calib":
                self._calib, self.calib_latency_s = res, dt
            elif kind == "obj":
                self._obj, self.obj_latency_s = (idx, res), dt
                self.n_obj_results += 1
            elif kind == "error":
                self.errors += 1
                if self.errors <= 3:
                    print(f"[uyarı] GPU alt süreci hatası: {res}")

    def submit(self, frame_idx: int, frame: np.ndarray, H: Optional[np.ndarray]) -> None:
        self._write(frame_idx, frame, H)

    def poll_calibration(self):
        self._drain()
        r, self._calib = self._calib, None
        return r

    def poll_objects(self):
        self._drain()
        r, self._obj = self._obj, None
        return r

    def close(self) -> None:
        with self._lock:
            self._meta[0] = -1
            self._new.set()
        self._proc.join(timeout=3.0)
        if self._proc.is_alive():
            self._proc.terminate()
        self._shm.close()
        self._shm.unlink()
