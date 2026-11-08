"""Video giriş/çıkışı (M1): arka plan iş parçacıklarıyla okuma ve yazma.

OpenCV kod çözme/kodlama sırasında GIL'i bıraktığı için, okuma ve yazma ayrı iş parçacıklarında
GPU çıkarımıyla paralel yürür.
"""
from __future__ import annotations

import queue
import threading
from typing import Callable, Iterator, Optional

import cv2
import numpy as np

_END = object()


class FrameReader:
    """Kareleri önceden okuyan iterator: `for idx, frame in FrameReader(path, start, n)`."""

    def __init__(self, path: str, start: int = 0, count: Optional[int] = None, prefetch: int = 16) -> None:
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise IOError(f"Video açılamadı: {path}")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        self.start, self.count = start, count
        self._q: queue.Queue = queue.Queue(maxsize=prefetch)
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self) -> None:
        i = 0
        while not self._stop.is_set() and (self.count is None or i < self.count):
            ok, frame = self.cap.read()
            if not ok:
                break
            self._q.put((self.start + i, frame))
            i += 1
        self._q.put(_END)

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        while True:
            item = self._q.get()
            if item is _END:
                return
            yield item

    def close(self) -> None:
        self._stop.set()
        while self._t.is_alive():          # kuyruk doluysa okuyucunun çıkabilmesi için boşalt
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            self._t.join(timeout=0.05)
        self.cap.release()


class AsyncWriter:
    """İsteğe bağlı bir `render(*item) -> frame` fonksiyonunu ve kodlamayı arka planda çalıştırır."""

    def __init__(self, writer: cv2.VideoWriter, render: Optional[Callable[..., np.ndarray]] = None,
                 maxsize: int = 16) -> None:
        self.writer, self.render = writer, render
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is _END:
                break
            frame = self.render(*item) if self.render else item[0]
            self.writer.write(frame)

    def put(self, *item) -> None:
        self._q.put(item)

    def close(self) -> None:
        self._q.put(_END)
        self._t.join()
        self.writer.release()
