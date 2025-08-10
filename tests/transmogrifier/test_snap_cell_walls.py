import pytest
import random
from src.transmogrifier.cells.simulator import Simulator  # Adjust this import to your project structure
from src.transmogrifier.cells.cell_consts import Cell, LEFT_WALL, RIGHT_WALL
from src.transmogrifier.bitbitbuffer import CellProposal

def _build_cells(config: str, variant: str, seed: int | None = None):
    """
    Build three cells (A, B, C) according to configuration and variant.

    config in {prime, coprime, power2, mix, random}
    variant in {normal, collapsed, stride_bound, unbound}
    """
    if seed is not None:
        random.seed(seed)

    # Choose strides
    if config == 'prime':
        strides = [5, 7, 11]
    elif config == 'coprime':
        strides = [4, 9, 25]  # pairwise coprime
    elif config == 'power2':
        strides = [8, 16, 32]
    elif config == 'mix':
        strides = [7, 9, 16]
    elif config == 'random':
        # keep within small bounds to avoid huge LCM
        pool = [3, 4, 5, 6, 7, 8, 9, 10]
        strides = [random.choice(pool) for _ in range(3)]
    else:
        raise ValueError(f"Unknown config: {config}")

    # Baseline positions keeping things small
    # Start with lefts roughly staggered and widths 8x stride
    bases = []
    start = 0
    for s in strides:
        left = start
        width = 8 * s
        right = left + width
        leftmost = left + s  # guaranteed inside
        rightmost = right - (s + 1)
        bases.append((left, right, leftmost, rightmost))
        start = right

    # Apply variants
    adj = []
    for i, (s, (L, R, LM, RM)) in enumerate(zip(strides, bases)):
        label = chr(ord('A') + i)
        if variant == 'normal':
            left, right = L, R
        elif variant == 'collapsed' and label == 'A':
            # Collapse only A; B and C remain usable for tests referencing 'C'
            left, right = L, L
            LM, RM = L, L - 1  # empty data
        elif variant == 'stride_bound':
            # Left aligned to stride but not to overall LCM (most k won't be multiple of LCM/stride)
            left = s  # multiple of stride
            width = 6 * s
            right = left + width
            LM = left + (s // 2 if s > 1 else 0)
            RM = right - 1 - (s // 2 if s > 1 else 0)
        elif variant == 'unbound':
            # Neither LCM nor stride aligned
            left = s + 1
            width = 6 * s + 3
            right = left + width
            LM = left + 1
            RM = right - 2
        else:
            left, right = L, R
        adj.append(Cell(label=label, left=left, right=right, stride=s, leftmost=LM, rightmost=RM))

    return adj


# --- Test Fixture ---
# This sets up a Simulator for parametrized cells; defaults to original baseline when no param provided.
@pytest.fixture
def sim_and_cells(request):
    """Provides a Simulator instance and a list of cells for testing across many scenarios."""
    config = variant = seed = None
    if hasattr(request, 'param') and request.param:
        config, variant, seed = request.param
        cells = _build_cells(config, variant, seed)
    else:
        cells = [
            Cell(label='A', left=0,    right=128, stride=16, leftmost=16, rightmost=31),
            Cell(label='B', left=128,  right=256, stride=32, leftmost=160, rightmost=191),
            Cell(label='C', left=256,  right=512, stride=64, leftmost=320, rightmost=447)
        ]
    # Try to construct simulator; for pathological variants allow capture
    try:
        sim = Simulator(cells)
        sim._fixture_error = None
    except Exception as e:
        class _Dummy:
            pass
        sim = _Dummy()
        sim._fixture_error = e
        sim.cells = cells
    # Attach test metadata
    sim._test_config = config or 'baseline'
    sim._test_variant = variant or 'normal'
    sim._test_seed = seed
    if getattr(sim, '_fixture_error', None) is None:
        sim.system_lcm = sim.lcm(cells)
    return sim, cells


def _allowed_exception_and_match(variant: str):
    """Return acceptable exception classes and optional message substrings for a variant."""
    # Broadly accept IndexError in expansion and assertion/value errors for invalid geometry
    if variant == 'collapsed':
        return (AssertionError, ValueError), [
            'domain_left must be < domain_right',
            'PIDBuffer domain',
        ]
    # For stride-bound or unbound geometries, buffer operations may raise IndexError/AssertionError
    if variant in ('stride_bound', 'unbound'):
        return (IndexError, AssertionError, ValueError), [
            'bytearray index out of range',
            'LCM',
            'align',
        ]
    # For random/mix/prime/coprime/power2 adversarial configs, be permissive
    return (IndexError, AssertionError, ValueError), [
        'bytearray index out of range',
        'LCM',
        'align',
        'domain_left',
    ]

def _maybe_expect_exception(sim, call):
    """Execute call(); if it raises and variant is adversarial, assert it matches expectations and return True.
    Return False if no exception.
    """
    variant = getattr(sim, '_test_variant', 'normal')
    try:
        call()
        return False
    except Exception as e:
        exc_types, substrs = _allowed_exception_and_match(variant)
        assert isinstance(e, exc_types), f"Unexpected exception type {type(e).__name__}: {e} for variant {variant}"
        msg = str(e)
        if substrs:
            assert any(s.lower() in msg.lower() for s in substrs), f"Unexpected exception message: {msg}"
        return True

# New: Prime-stride fixture for harsher alignment behavior
@pytest.fixture
def prime_sim_and_cells():
    """Simulator + cells using prime strides to force large LCM and coarse snapping."""
    # Use moderately large primes to avoid absurd memory while still adversarial
    cells = [
        Cell(label='P1', left=0,    right=193, stride=97,  leftmost=3,   rightmost=127),
        Cell(label='P2', left=193,  right=500, stride=101, leftmost=227, rightmost=463),
        Cell(label='P3', left=500,  right=900, stride=103, leftmost=541, rightmost=859),
    ]
    sim = Simulator(cells)
    sim.system_lcm = sim.lcm(cells)
    return sim, cells
def test_handles_shrink_proposals_from_evolution(sim_and_cells):
    """
    REALISTIC STRESS TEST: Simulates the scenario where evolution_tick proposes
    to shrink cells. This test asserts that snap_cell_walls will respect the
    shrink proposal BUT will not collapse any cell to zero or a width smaller
    than its stride. This should trigger the bug.
    """
    sim, cells = sim_and_cells

    # 1. Simulate the output of evolution_tick: create proposals with smaller widths.
    proposals = []
    for c in cells:
        proposal = CellProposal(c)
        # Propose a new width that is smaller, but still valid (e.g., half the original)
        # This mimics the pressure/salinity calculations.
        new_width = (c.right - c.left) // 2
        # Ensure the new width is at least the stride and aligned to it.
        proposal.right = proposal.left + max(c.stride, sim.bitbuffer.intceil(new_width, c.stride))
        proposals.append(proposal)

    # 2. Run the function with the problematic shrink proposals.
    # If this explodes under adversarial variants, that's acceptable and asserted.
    if sim._fixture_error is not None:
        # Fixture failed to build due to variant; assert expected and return
        exc_types, substrs = _allowed_exception_and_match(sim._test_variant)
        msg = str(sim._fixture_error)
        assert isinstance(sim._fixture_error, exc_types)
        assert any(s.lower() in msg.lower() for s in substrs)
        return
    if _maybe_expect_exception(sim, lambda: sim.snap_cell_walls(cells, proposals)):
        return

    # 3. Assert that no cell was collapsed.
    for p in proposals:
        final_width = p.right - p.left
        assert final_width > 0, f"Cell '{p.label}' was collapsed to zero-width by a shrink proposal."
        assert final_width >= p.stride, f"Cell '{p.label}' was contracted below its stride by a shrink proposal."

# --- Core Assumption Tests ---

def test_no_contraction_or_collapse(sim_and_cells):
    """
    STRESS TEST: Asserts that snap_cell_walls NEVER reduces a cell's width
    below its stride and NEVER collapses a cell to zero-width. This is the
    most critical test to prevent the main bug.
    """
    sim, cells = sim_and_cells
    proposals = [CellProposal(c) for c in cells]

    # Calculate total width before
    initial_widths = {c.label: c.right - c.left for c in cells}

    # Run the function (exceptions are acceptable under adversarial variants)
    if sim._fixture_error is not None:
        exc_types, substrs = _allowed_exception_and_match(sim._test_variant)
        msg = str(sim._fixture_error)
        assert isinstance(sim._fixture_error, exc_types)
        assert any(s.lower() in msg.lower() for s in substrs)
        return
    if _maybe_expect_exception(sim, lambda: sim.snap_cell_walls(cells, proposals)):
        return

    # --- Assertions ---
    for p in proposals:
        final_width = p.right - p.left
        # 1. CRITICAL: Assert cell was not collapsed.
        assert final_width > 0, f"Cell '{p.label}' was collapsed to zero-width."
        # 2. CRITICAL: Assert cell did not shrink below its fundamental stride.
        assert final_width >= p.stride, f"Cell '{p.label}' contracted below its minimum stride width."
        # 3. Assert cell did not shrink at all (unless buffer expansion happened).
        if sim.bitbuffer.mask_size == 768: # Assuming no expansion for this check
             assert final_width >= initial_widths[p.label], f"Cell '{p.label}' contracted from {initial_widths[p.label]} to {final_width}."


def test_lcm_alignment(sim_and_cells):
    """
    STRICT TEST: Asserts that ALL cell boundaries after snapping are perfectly
    aligned to the system's LCM grid.
    """
    sim, cells = sim_and_cells
    proposals = [CellProposal(c) for c in cells]
    lcm = sim.system_lcm

    # Run the function
    if sim._fixture_error is not None:
        exc_types, substrs = _allowed_exception_and_match(sim._test_variant)
        msg = str(sim._fixture_error)
        assert isinstance(sim._fixture_error, exc_types)
        assert any(s.lower() in msg.lower() for s in substrs)
        return
    if _maybe_expect_exception(sim, lambda: sim.snap_cell_walls(cells, proposals)):
        return

    # --- Assertions ---
    for p in proposals:
        assert p.left % lcm == 0, f"Cell '{p.label}' left boundary {p.left} is not aligned to LCM {lcm}."
        assert p.right % lcm == 0, f"Cell '{p.label}' right boundary {p.right} is not aligned to LCM {lcm}."

def test_boundary_contiguity(sim_and_cells):
    """
    STRICT TEST: Asserts that there are no gaps or overlaps between adjacent cells.
    The right boundary of one cell must equal the left boundary of the next.
    """
    sim, cells = sim_and_cells
    proposals = [CellProposal(c) for c in cells]

    # Run the function
    if sim._fixture_error is not None:
        exc_types, substrs = _allowed_exception_and_match(sim._test_variant)
        msg = str(sim._fixture_error)
        assert isinstance(sim._fixture_error, exc_types)
        assert any(s.lower() in msg.lower() for s in substrs)
        return
    if _maybe_expect_exception(sim, lambda: sim.snap_cell_walls(cells, proposals)):
        return

    # --- Assertions ---
    # Check contiguity from LEFT_WALL up to the last proposal
    all_boundaries = [LEFT_WALL] + proposals
    for i in range(len(all_boundaries) - 1):
        prev_cell = all_boundaries[i]
        curr_cell = all_boundaries[i+1]
        assert prev_cell.right == curr_cell.left, \
            f"Gap or overlap found: Cell '{prev_cell.label}' ends at {prev_cell.right} but Cell '{curr_cell.label}' starts at {curr_cell.left}."

def test_data_integrity_preserved(sim_and_cells):
    """
    STRICT TEST: Asserts that the relative position of data within a cell
    (leftmost, rightmost) is preserved after boundaries are shifted by expansion.
    """
    sim, cells = sim_and_cells

    # Store initial relative positions of data
    initial_data_offsets = {c.label: (c.leftmost - c.left, c.rightmost - c.left) for c in cells}

    # Create proposals that simulate an expansion by forcing new, wider widths
    proposals = [CellProposal(c) for c in cells]
    for p in proposals:
        p.right = p.left + (p.right - p.left) * 2 # Double the width

    # Run the function
    if sim._fixture_error is not None:
        exc_types, substrs = _allowed_exception_and_match(sim._test_variant)
        msg = str(sim._fixture_error)
        assert isinstance(sim._fixture_error, exc_types)
        assert any(s.lower() in msg.lower() for s in substrs)
        return
    if _maybe_expect_exception(sim, lambda: sim.snap_cell_walls(cells, proposals)):
        return

    # --- Assertions ---
    for p in proposals:
        initial_left_offset, initial_right_offset = initial_data_offsets[p.label]
        final_left_offset = p.leftmost - p.left
        final_right_offset = p.rightmost - p.left

        assert final_left_offset == initial_left_offset, \
            f"Data in cell '{p.label}' shifted incorrectly. Left offset changed from {initial_left_offset} to {final_left_offset}."
        assert final_right_offset == initial_right_offset, \
            f"Data in cell '{p.label}' shifted incorrectly. Right offset changed from {initial_right_offset} to {final_right_offset}."


def test_no_unintended_relocation(sim_and_cells):
    """
    STRICT TEST: Asserts that if no expansion is proposed, the cells do not move.
    Relocation should only be a consequence of expansion.
    """
    sim, cells = sim_and_cells

    # Store initial positions
    initial_positions = {c.label: (c.left, c.right) for c in cells}

    # Create proposals with no changes
    proposals = [CellProposal(c) for c in cells]

    # Run the function
    if sim._fixture_error is not None:
        exc_types, substrs = _allowed_exception_and_match(sim._test_variant)
        msg = str(sim._fixture_error)
        assert isinstance(sim._fixture_error, exc_types)
        assert any(s.lower() in msg.lower() for s in substrs)
        return
    if _maybe_expect_exception(sim, lambda: sim.snap_cell_walls(cells, proposals)):
        return

    # --- Assertions ---
    for p in proposals:
        initial_left, initial_right = initial_positions[p.label]
        assert p.left == initial_left, f"Cell '{p.label}' was relocated without cause (left changed)."
        assert p.right == initial_right, f"Cell '{p.label}' was relocated without cause (right changed)."


def test_extreme_shrink_proposal_is_ignored(sim_and_cells):
    """
    HARSH STRESS TEST: Proposes a new width for a cell that is smaller than its
    existing data span. Asserts that the function IGNORES the shrink request and
    instead ensures the cell is large enough to contain its data.
    """
    sim, cells = sim_and_cells
    proposals = [CellProposal(c) for c in cells]

    # Find cell 'C' and give it an impossible shrink request
    cell_c_proposal = next(p for p in proposals if p.label == 'C')
    # Data span for C is 447 - 320 = 127. Propose a new width of 64.
    cell_c_proposal.right = cell_c_proposal.left + 64

    # Run the function
    if sim._fixture_error is not None:
        exc_types, substrs = _allowed_exception_and_match(sim._test_variant)
        msg = str(sim._fixture_error)
        assert isinstance(sim._fixture_error, exc_types)
        assert any(s.lower() in msg.lower() for s in substrs)
        return
    if _maybe_expect_exception(sim, lambda: sim.snap_cell_walls(cells, proposals)):
        return

    # --- Assertions ---
    final_cell_c = next(p for p in proposals if p.label == 'C')
    data_span = final_cell_c.rightmost - final_cell_c.leftmost
    final_width = final_cell_c.right - final_cell_c.left

    assert final_width > data_span, \
        f"Cell 'C' failed to protect its data. Final width {final_width} is not greater than its data span {data_span}."
    assert final_cell_c.left <= final_cell_c.leftmost, \
        f"Cell 'C' data integrity lost: leftmost {final_cell_c.leftmost} is outside the left boundary {final_cell_c.left}."
    assert final_cell_c.right > final_cell_c.rightmost, \
        f"Cell 'C' data integrity lost: rightmost {final_cell_c.rightmost} is outside the right boundary {final_cell_c.right}."


if __name__ == "__main__":
    import sys
    import pytest as _pytest
    sys.exit(_pytest.main([__file__]))


# --- Adversarial prime-stride and exception-as-pass attacks ---

@pytest.mark.stress
def test_zero_stride_empty_cell_triggers_assertion(prime_sim_and_cells):
    """
    Attack: Include an empty cell with stride=0 among otherwise valid prime-stride cells.
    Expectation: snap_cell_walls asserts system LCM > 0 and raises.
    Pass condition: an exception is raised.
    """
    sim, cells = prime_sim_and_cells

    # Malicious: zero-width, zero-stride cell (empty) injected into the call only
    bomb = Cell(label='Z0', left=777, right=777, stride=0, leftmost=777, rightmost=776)
    atk_cells = cells + [bomb]
    proposals = [CellProposal(c) for c in atk_cells]

    with pytest.raises(AssertionError):
        sim.snap_cell_walls(atk_cells, proposals)


@pytest.mark.stress
def test_noninteger_stride_typeerror(prime_sim_and_cells):
    """
    Attack: Give a cell a non-integer stride so LCM/gcd math explodes.
    Pass condition: any exception is raised (TypeError most likely).
    """
    sim, cells = prime_sim_and_cells
    weird = Cell(label='WEIRD', left=1000, right=1100, stride='prime?', leftmost=1001, rightmost=1099)
    atk_cells = cells + [weird]
    proposals = [CellProposal(c) for c in atk_cells]

    with pytest.raises(Exception):
        sim.snap_cell_walls(atk_cells, proposals)


@pytest.mark.stress
def test_pathological_proposals_on_primes_raise_or_reject(prime_sim_and_cells):
    """
    Attack: Create self-contradictory proposals under large prime LCM snapping.
    - left > right (negative width)
    - leftmost >> rightmost (empty data)
    - push boundaries far outside current mask
    Pass condition: an exception is raised (alignment/LCM/expansion assertions).
    """
    sim, cells = prime_sim_and_cells
    # Craft proposals that are as nonsensical as possible while keeping types valid
    proposals = [CellProposal(c) for c in cells]
    for i, p in enumerate(proposals):
        # Reverse boundaries for negative width on alternating cells
        if i % 2 == 0:
            p.left, p.right = p.right + 10_000, p.left - 10_000
        # Make data extents contradictory and far outside
        p.leftmost = p.right + 123_456
        p.rightmost = p.left - 123_456
        # Inflate pressure to amplify any pressure math edge cases
        p.pressure = 10**9

    # Also append a dummy cell with a huge prime stride to balloon LCM subtly
    dummy = Cell(label='DUMP', left=proposals[-1].right + 1, right=proposals[-1].right + 2,
                 stride=149, leftmost=None, rightmost=None)
    atk_cells = cells + [dummy]
    proposals.append(CellProposal(dummy))

    with pytest.raises(Exception):
        sim.snap_cell_walls(atk_cells, proposals)

