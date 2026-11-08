import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sgi.grid.grid import N_CELLS, PitchGrid  # noqa: E402
from sgi.types import CELL_OUT  # noqa: E402


@pytest.fixture(scope="module")
def grid() -> PitchGrid:
    return PitchGrid.from_yaml(ROOT / "configs/grid.yaml")


def test_layout_is_10x5(grid):
    assert (grid.n_cols, grid.n_rows) == (10, 5)
    assert grid.length == 105.0 and grid.width == 68.0


def test_corner_cells(grid):
    # D-038: sol alt köşeden başlar, her sütunda alttan yukarı artar
    assert grid.cell_of(0.1, 67.9) == 1         # sol kale, yakın taç
    assert grid.cell_of(0.0, 0.0) == 5          # sol kale, uzak taç
    assert grid.cell_of(15.0, 67.0) == 6        # ikinci sütun yine alttan başlar
    assert grid.cell_of(15.0, 1.0) == 10
    assert grid.cell_of(105.0, 68.0) == 46      # sağ kale, yakın taç (kapalı son sınır)
    assert grid.cell_of(104.9, 0.1) == 50       # sağ kale, uzak taç


def test_spec_example_cells(grid):
    # docs/05_PASS_RULES.md örneği: 3 -> 7 -> 12 (1.-3. sütunlar)
    assert grid.cell_of(5.0, 30.0) == 3
    assert grid.cell_of(15.0, 45.0) == 7
    assert grid.cell_of(25.0, 45.0) == 12


def test_half_open_boundaries(grid):
    assert grid.cell_of(52.5, 10.0) == 30       # orta çizgi sağdaki hücreye aittir
    assert grid.cell_of(52.49, 10.0) == 25
    assert grid.cell_of(10.0, 13.6) == 4        # satır sınırı alttaki (yakın taç tarafı) hücreye aittir


def test_out_of_pitch(grid):
    assert grid.cell_of(-0.4, 30.0) == 3        # tolerans içinde: kenar hücre
    assert grid.cell_of(-1.0, 30.0) == CELL_OUT
    assert grid.cell_of(50.0, 69.0) == CELL_OUT


def test_every_cell_roundtrip(grid):
    for c in range(1, N_CELLS + 1):
        assert grid.cell_of(*grid.cell_center(c)) == c
        assert grid.cell_id(*grid.row_col(c)) == c


def test_cell_count_is_enforced():
    with pytest.raises(ValueError):
        PitchGrid((0, 50, 105), (0, 68))
