import string


def print_system(sim, width=80):
    """Print a textual overview of the simulator's cell layout.

    The rendering scales the complete address space to ``width`` characters.
    Whenever a column intersects ``N`` cells, ``N`` glyphs are emitted—
    lower‑case if the slice holds no data and ramp‑mapped otherwise.
    """
    cells = sim.cells
    total_bits = sim.bitbuffer.mask_size
    if total_bits == 0:
        print("<empty>")
        return

    # 1) Build bit_info: (bit_index, cell_idx_or_None, mask_bit)
    bit_info = []
    for b in range(total_bits):
        cell_idx = None
        for idx, cell in enumerate(cells):
            if cell.left <= b < cell.right:
                cell_idx = idx
                break
        mask_bit = bool(int(sim.bitbuffer[b]))
        bit_info.append((b, cell_idx, mask_bit))

    # 2) Fragmentation (unchanged)
    free_bits = sum(1 for _, idx, m in bit_info if idx is not None and not m)
    runs, run = [], 0
    for _, idx, m in bit_info:
        if idx is not None and not m:
            run += 1
        elif run:
            runs.append(run); run = 0
    if run: runs.append(run)
    max_run = max(runs) if runs else 0
    frag_pct = (1 - max_run / free_bits) * 100 if free_bits else 0.0

    size_string = (
        f"Total size: {total_bits} bits "
        f"({total_bits/8:.2f} bytes, mask bits: {sim.bitbuffer.mask_size})"
    )
    free_string = f"Free: {free_bits} bits; fragmentation: {frag_pct:.2f}%"

    # 3) Prepare labels & ramp
    labels     = string.ascii_lowercase
    alpha_len  = len(labels)
    glyph_bases = [
            ord('a'),    # level 1
            ord('A'),    # level 2
            0x03B1,      # level 3 α
            0x0391,      # level 4 Α
            0x0430,      # level 5 а
            0x0410,      # level 6 А
    ]
    max_level = len(glyph_bases)

    # 4) Columnize
    bits_per_col = (total_bits + width - 1) // width
    output = []
    for col in range(width):
        start = col * bits_per_col
        end   = min(start + bits_per_col, total_bits)

        # track which cells appear, and how many data‐hits each has
        cell_presence = set()
        cell_counts   = {}
        for b, idx, m in bit_info[start:end]:
            if idx is None:
                continue
            cell_presence.add(idx)
            # only count “data” at stride‐anchor positions
            cell = cells[idx]
            if m and ((b - cell.left) % cell.stride) == 0:
                cell_counts[idx] = cell_counts.get(idx, 0) + 1

        if not cell_presence:
            # entirely outside all cells
            output.append('.')
        else:
            # for each cell in this column, emit one glyph
            for idx in sorted(cell_presence):
                cnt = cell_counts.get(idx, 0)
                if cnt == 0:
                    # inside a cell but no data → lower‐case label
                    ch = labels[idx % alpha_len]
                else:
                    # data present → map 1→level1, 2→level2, ..., clamp
                    level = min(cnt, max_level) - 1
                    base  = glyph_bases[level]
                    ch    = chr(base + (idx % alpha_len))
                output.append(ch)

    # 5) Print map + stats
    print(''.join(output))
    print(size_string, free_string)

def bar(sim, number=2, width=80):
    """Emit ``number`` rows of ``#`` characters for quick visual separators."""
    for _ in range(number):
        print("#" * width)


# ────────────────────────────────────────────────────────────────
# Live Pygame Visualiser driven by the Simulator
# ----------------------------------------------------------------
try:  # pragma: no cover - optional dependency
    import pygame, sys, time
    VISUALISE = True
except Exception:  # pragma: no cover - pygame not available
    pygame = None  # type: ignore
    sys = time = None  # type: ignore
    VISUALISE = False
SCALE_X     = 1.0           # pixels per byte  (auto-scaled below)
ROW_H       = 28            # pixels per cell row
GRID_COLOUR = (180,180,180) # light grey grid lines
COL_SOLVENT = (200,225,255) # pale blue
COL_DATA    = ( 30,144,255) # bright blue
FPS         = 60

# ────────────────────────────────────────────────────────────────
# Live injection driver (manual keys 0-7 or auto every N seconds)
# ----------------------------------------------------------------
if VISUALISE:  # pragma: no branch - trivial runtime guard
    INJECT_KEYS = {
        pygame.K_0: 0,
        pygame.K_1: 1,
        pygame.K_2: 2,
        pygame.K_3: 3,
        pygame.K_4: 4,
        pygame.K_5: 5,
        pygame.K_6: 6,
        pygame.K_7: 7,
    }
else:
    INJECT_KEYS = {}
AUTO_INJECT_EVERY = 0.10          # seconds (set 0 to disable)


class _LCVisual:
    """Simple pygame-based visualiser for the simulator's cells."""

    def __init__(self, sim):
        self.sim = sim
        pygame.init()
        self.clock = pygame.time.Clock()

        # geometry is derived from the cell layout and may change as cells
        # expand/shrink; keep a cached copy and recalc when needed
        self._prev_span = None
        self._prev_rows = None
        self.screen = None  # assigned by _recalc_geometry()
        self._recalc_geometry()

    def _recalc_geometry(self) -> None:
        """Recompute scale factors and window size based on current cells."""
        cells = self.sim.cells
        tot_span = max(c.right for c in cells) - min(c.left for c in cells)
        global SCALE_X
        SCALE_X = 1200 / max(1, tot_span)       # fit into ~1200 px window
        w = int(tot_span * SCALE_X) + 20
        h = len(cells) * ROW_H + 20
        self.screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption("Simulator memory layout")
        self._prev_span = tot_span
        self._prev_rows = len(cells)

    def draw(self):
        cells = self.sim.cells
        # if any cell moved/expanded, recompute scaling and window size so
        # content stays within view.  This keeps the visualisation responsive
        # to dynamic layout changes.
        span = max(c.right for c in cells) - min(c.left for c in cells)
        if span != self._prev_span or len(cells) != self._prev_rows:
            self._recalc_geometry()

        self.screen.fill((0, 0, 0))
        base_left = min(c.left for c in cells)

        for row, c in enumerate(cells):
            y0 = 10 + row * ROW_H
            stride = max(1, c.stride)  # avoid div-by-zero

            # Fetch the PID mask for this cell and draw each stride slot
            pb = self.sim.bitbuffer.pid_buffers.get(c.label)
            if pb:
                mask = pb.pids
                mask_slots = mask.mask_size
            else:
                mask = None
                mask_slots = 0

            # Number of stride buckets that cover the cell's span
            cell_slots = ((c.right - c.left) + stride - 1) // stride

            for slot_idx in range(cell_slots):
                slot_left = c.left + slot_idx * stride
                if slot_left >= c.right:
                    break
                slot_right = min(slot_left + stride, c.right)

                x0 = 10 + int((slot_left - base_left) * SCALE_X)
                x1 = 10 + int((slot_right - base_left) * SCALE_X)
                w = max(1, x1 - x0)

                bit_active = int(mask[slot_idx]) if mask and slot_idx < mask_slots else 0
                colour = COL_DATA if bit_active else COL_SOLVENT
                pygame.draw.rect(
                    self.screen,
                    colour,
                    pygame.Rect(x0, y0 + 4, w, ROW_H - 8),
                )

            # cell boundary lines (after quanta so they stay visible)
            xL = 10 + int((c.left  - base_left) * SCALE_X)
            xR = 10 + int((c.right - base_left) * SCALE_X)
            pygame.draw.line(self.screen, GRID_COLOUR, (xL, y0), (xL, y0 + ROW_H - 1))
            pygame.draw.line(self.screen, GRID_COLOUR, (xR, y0), (xR, y0 + ROW_H - 1))

            # optional: light stride grid inside region
            # comment out if busy
            if stride and stride * SCALE_X > 4:
                x = xL + int(stride * SCALE_X)
                while x < xR - 1:
                    pygame.draw.line(self.screen, (60, 60, 60),
                                     (x, y0 + 4), (x, y0 + ROW_H - 5))
                    x += int(stride * SCALE_X)

            # label at left edge
            font = pygame.font.Font(None, 18)
            self.screen.blit(
                font.render(str(c.label), True, (255, 255, 255)),
                (xL + 4, y0 + 4),
            )

        pygame.display.flip()
        self.clock.tick(FPS)


_vis = None


def visualise_step(sim, cells):
    """Wrapper for ``Simulator.step`` that updates the pygame window."""
    global _vis
    if VISUALISE and _vis is None:
        _vis = _LCVisual(sim)

    sim.run_saline_sim()
    sp, mask = sim.step(cells)

    if VISUALISE:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        _vis.draw()

    return sp, mask

    # --------------------------------------------------------------------
    # DEMO HARNESS – basic pygame loop using the Simulator
    # --------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import random
    
    from ..cell_consts import Cell
    from ..simulator import Simulator

    specs = [
        dict(left=0,   right=128,  label="0", len=128, stride=128),
        dict(left=128, right=256,  label="1", len=128, stride=64),
        dict(left=256, right=512,  label="2", len=256, stride=32),
        dict(left=512, right=768,  label="3", len=256, stride=16),
    ]

    cells = [Cell(**s) for s in specs]
    sim = Simulator(cells)
    sim.run_balanced_saline_sim()

    vis = _LCVisual(sim)

    next_auto = time.time() + AUTO_INJECT_EVERY if AUTO_INJECT_EVERY else None

    while True:
        now = time.time()
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if INJECT_KEYS and ev.type == pygame.KEYDOWN and ev.key in INJECT_KEYS:
                idx = INJECT_KEYS[ev.key]
                target = cells[idx]
                data_len = (target.stride * sim.bitbuffer.bitsforbits + 7) // 8
                # queue the payload
                sim.input_queues.setdefault(target.label, []).append((os.urandom(data_len), target.stride))
                # tell the system there’s one more item to inject
                target.injection_queue += 1
        if next_auto and now >= next_auto:
            idx = random.randrange(len(cells))
            target = cells[idx]
            data_len = (target.stride * sim.bitbuffer.bitsforbits + 7) // 8
            # queue the payload
            sim.input_queues.setdefault(target.label, []).append((os.urandom(data_len), target.stride))
            # tell the system there’s one more item to inject
            target.injection_queue += 1
            next_auto = now + AUTO_INJECT_EVERY

        visualise_step(sim, cells)



def bar(sim, number=2, width=80):
    """Emit ``number`` rows of ``#`` characters for quick visual separators."""
    for _ in range(number):
        print("#" * width)
