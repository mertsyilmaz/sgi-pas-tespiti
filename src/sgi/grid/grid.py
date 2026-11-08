"""50 hücreli saha ızgarası (M5, D-004, docs/04_PITCH_GRID.md)."""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

import yaml

from sgi.types import CELL_OUT, CellId

N_CELLS = 50


@dataclass(frozen=True)
class PitchGrid:
    x_edges: tuple[float, ...]
    y_edges: tuple[float, ...]
    out_tolerance_m: float = 0.5

    def __post_init__(self) -> None:
        if self.n_cols * self.n_rows != N_CELLS:
            raise ValueError(f"Izgara {N_CELLS} hücre olmalı, {self.n_cols}x{self.n_rows} verildi (D-004)")
        for edges in (self.x_edges, self.y_edges):
            if any(b <= a for a, b in zip(edges, edges[1:])):
                raise ValueError("Hücre sınırları kesin artan olmalı")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PitchGrid":
        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(tuple(cfg["x_edges"]), tuple(cfg["y_edges"]), cfg.get("out_tolerance_m", 0.5))

    @property
    def n_cols(self) -> int:
        return len(self.x_edges) - 1

    @property
    def n_rows(self) -> int:
        return len(self.y_edges) - 1

    @property
    def length(self) -> float:
        return self.x_edges[-1] - self.x_edges[0]

    @property
    def width(self) -> float:
        return self.y_edges[-1] - self.y_edges[0]

    def cell_id(self, row: int, col: int) -> int:
        """Sütun sıralı, sol alt köşeden başlar: her sütunda alttan (yakın taç, row = n_rows-1) yukarı artar."""
        return col * self.n_rows + (self.n_rows - 1 - row) + 1

    def row_col(self, cell_id: int) -> tuple[int, int]:
        if not 1 <= cell_id <= N_CELLS:
            raise ValueError(f"Geçersiz hücre: {cell_id}")
        col, k = divmod(cell_id - 1, self.n_rows)
        return self.n_rows - 1 - k, col

    def cell_of(self, x: float, y: float) -> CellId:
        """Saha noktasını hücreye atar. Yarı açık aralık; saha dışı -> OUT."""
        x0, x1 = self.x_edges[0], self.x_edges[-1]
        y0, y1 = self.y_edges[0], self.y_edges[-1]
        tol = self.out_tolerance_m
        if not (x0 - tol <= x <= x1 + tol and y0 - tol <= y <= y1 + tol):
            return CELL_OUT
        # tolerans bandındaki noktaları en yakın kenar hücresine sıkıştır
        x = min(max(x, x0), x1)
        y = min(max(y, y0), y1)
        col = min(bisect_right(self.x_edges, x) - 1, self.n_cols - 1)
        row = min(bisect_right(self.y_edges, y) - 1, self.n_rows - 1)
        return self.cell_id(row, col)

    def cell_bounds(self, cell_id: int) -> tuple[float, float, float, float]:
        """(x_min, y_min, x_max, y_max)"""
        row, col = self.row_col(cell_id)
        return self.x_edges[col], self.y_edges[row], self.x_edges[col + 1], self.y_edges[row + 1]

    def cell_center(self, cell_id: int) -> tuple[float, float]:
        x0, y0, x1, y1 = self.cell_bounds(cell_id)
        return (x0 + x1) / 2, (y0 + y1) / 2
