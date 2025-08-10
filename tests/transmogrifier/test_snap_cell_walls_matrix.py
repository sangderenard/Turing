import random
import pytest

from src.transmogrifier.cells.simulator import Simulator
from src.transmogrifier.cells.cell_consts import Cell, LEFT_WALL, RIGHT_WALL
from src.transmogrifier.bitbitbuffer import CellProposal


def build_cells(config: str, variant: str, seed: int | None = None):
    if seed is not None:
        random.seed(seed)

    if config == 'prime':
        strides = [5, 7, 11]
    elif config == 'coprime':
        strides = [4, 9, 25]
    elif config == 'power2':
        strides = [8, 16, 32]
    elif config == 'mix':
        strides = [7, 9, 16]
    elif config == 'random':
        pool = [3, 4, 5, 6, 7, 8, 9, 10]
        strides = [random.choice(pool) for _ in range(3)]
    else:
        raise ValueError(f"Unknown config: {config}")

    # Base contiguous layout; widths scaled to stride
    cells = []
    left = 0
    for i, s in enumerate(strides):
        width = 8 * s
        right = left + width
        leftmost = left + s
        rightmost = right - (s + 1)
        label = chr(ord('A') + i)
        cells.append(Cell(label=label, left=left, right=right, stride=s, leftmost=leftmost, rightmost=rightmost))
        left = right

    if variant == 'normal':
        return cells
    if variant == 'collapsed':
        a = cells[0]
        # Fully collapsed A triggers PID domain assertion on Simulator construction
        return [Cell(label=a.label, left=a.left, right=a.left, stride=a.stride, leftmost=a.left, rightmost=a.left - 1)] + cells[1:]
    if variant == 'stride_bound':
        adj = []
        for c in cells:
            left = c.stride  # stride-aligned but not necessarily LCM aligned
            right = left + 6 * c.stride
            leftmost = left + max(0, c.stride // 2)
            rightmost = right - 1 - max(0, c.stride // 2)
            adj.append(Cell(label=c.label, left=left, right=right, stride=c.stride, leftmost=leftmost, rightmost=rightmost))
        return adj
    if variant == 'unbound':
        adj = []
        for c in cells:
            left = c.stride + 1
            right = left + 6 * c.stride + 3
            leftmost = left + 1
            rightmost = right - 2
            adj.append(Cell(label=c.label, left=left, right=right, stride=c.stride, leftmost=leftmost, rightmost=rightmost))
        return adj
    raise ValueError(f"Unknown variant: {variant}")


def shrink_proposals(sim, cells):
    props = []
    for c in cells:
        p = CellProposal(c)
        new_width = max(c.stride, (c.right - c.left) // 2)
        p.right = p.left + sim.bitbuffer.intceil(new_width, c.stride)
        props.append(p)
    return props


def widen_proposals(cells):
    return [CellProposal(c) for c in cells]


def expect_exception_for(variant: str) -> tuple[type[BaseException] | tuple, str | None]:
    if variant == 'collapsed':
        return (AssertionError,), 'domain_left must be < domain_right'
    if variant in ('stride_bound', 'unbound'):
        return (IndexError, AssertionError, ValueError), None
    return tuple(), None


@pytest.mark.matrix
@pytest.mark.parametrize(
    'config,variant,seed',
    [
        *[(cfg, var, 4242) for cfg in ['prime','coprime','power2','mix'] for var in ['normal','collapsed','stride_bound','unbound']],
        ('random','normal', 9001),
    ]
)
def test_snap_cell_walls_matrix(config, variant, seed):
    cells = build_cells(config, variant, seed)

    exc_types, match = expect_exception_for(variant)

    # 1) Simulator construction stage
    if exc_types:
        if variant == 'collapsed':
            with pytest.raises(exc_types, match=match):
                Simulator(cells)
            return

    sim = Simulator(cells)

    # 2) shrink path
    sprops = shrink_proposals(sim, cells)
    if exc_types:
        with pytest.raises(exc_types):
            sim.snap_cell_walls(cells, sprops)
    else:
        sim.snap_cell_walls(cells, sprops)
        for p in sprops:
            assert p.right - p.left >= p.stride > 0

    # 3) lcm alignment + contiguity with baseline proposals
    props = widen_proposals(cells)
    if exc_types:
        with pytest.raises(exc_types):
            sim.snap_cell_walls(cells, props)
        return

    sim.snap_cell_walls(cells, props)
    lcm = sim.system_lcm
    for p in props:
        assert p.left % lcm == 0
        assert p.right % lcm == 0

    chain = [LEFT_WALL] + props
    for i in range(len(chain) - 1):
        assert chain[i].right == chain[i+1].left


if __name__ == "__main__":
    import sys
    import pytest as _pytest
    sys.exit(_pytest.main([__file__]))
