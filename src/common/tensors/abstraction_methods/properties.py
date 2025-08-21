from __future__ import annotations

from typing import Any, Tuple, Union


from .. import DEBUG

def numel(self) -> int:
    return self.numel_()

def item(self) -> Union[int, float, bool]:
    return self.item_()

def shape(self) -> Tuple[int, ...]:
    """Return the shape of the tensor as a tuple (NumPy/PyTorch style)."""
    return self.shape_()

def shape_(self) -> Tuple[int, ...]:
    """Return the shape of the tensor as a tuple (backend hook)."""
    return self.get_shape()


def ndim(self):
    """Return the number of dimensions (property, NumPy style)."""
    return self.get_ndims(self.data)


def dim(self) -> int:
    """Return the number of dimensions (method, torch style)."""
    return self.get_ndims(self.data)


def ndims(self) -> int:
    """Return the number of dimensions (method, project style)."""
    return self.get_ndims(self.data)


def datastring(self, data: Any) -> str:
    """Return a pretty string representation of ``data`` for console output."""
    if data is None:
        return "AbstractTensor (None)"

    try:
        shape = self.get_shape()
    except Exception:
        shape = ()

    try:
        dtype = self.get_dtype(data)
    except Exception:
        dtype = getattr(data, "dtype", None)

    try:
        device = self.get_device(data)
    except Exception:
        device = getattr(data, "device", None)

    header = f"shape={shape} dtype={dtype} device={device}"

    try:
        from colorama import Fore, Style  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        class _NoColor:
            RED = BLUE = CYAN = YELLOW = GREEN = MAGENTA = WHITE = RESET_ALL = ""
        Fore = Style = _NoColor()  # type: ignore

    if hasattr(data, "tolist"):
        values = data.tolist()
    else:
        values = data

    if not isinstance(values, list):
        values = [values]

    if shape and len(shape) == 1:
        values = [values]

    rows = len(values)
    cols = len(values[0]) if rows and isinstance(values[0], list) else 1

    flat_vals = [
        float(x)
        for row in values
        for x in (row if isinstance(row, list) else [row])
        if isinstance(x, (int, float))
    ]
    if flat_vals:
        min_val, max_val = min(flat_vals), max(flat_vals)
        spread = max_val - min_val or 1.0
    else:
        min_val, max_val, spread = 0.0, 0.0, 1.0

    def colorize(v: Any) -> str:
        if not isinstance(v, (int, float)):
            return str(v)
        norm = (float(v) - min_val) / spread
        palette = [Fore.BLUE, Fore.CYAN, Fore.GREEN, Fore.YELLOW, Fore.RED]
        idx = int(norm * (len(palette) - 1))
        return f"{palette[idx]}{v:.4e}{Style.RESET_ALL}"

    cell_w = 10
    col_cap = 6
    row_cap = 10
    lines = []
    border = "+" + "+".join(["-" * cell_w] * min(cols, col_cap)) + "+"
    lines.append(border)
    for r in range(min(rows, row_cap)):
        row = values[r] if isinstance(values[r], list) else [values[r]]
        cells = []
        for c in range(min(cols, col_cap)):
            if c < len(row):
                cell = colorize(row[c]).ljust(cell_w)
            else:
                cell = "".ljust(cell_w)
            cells.append(cell)
        lines.append("|" + "|".join(cells) + "|")
    if rows > row_cap or cols > col_cap:
        ell = "...".center(cell_w)
        lines.append("|" + "|".join([ell] * min(cols, col_cap)) + "|")
    lines.append(border)

    table = "\n".join(lines)
    return f"\n\n{header}\n{table}\n\n"


def __str__(self):
    return self.datastring(self.data)


def __format__(self, format_spec: str) -> str:
    return f"AbstractTensor (shape={self.get_shape()}, dtype={self.get_dtype()}, device={self.get_device()})"


def __repr__(self):
    backend_class = (
        type(self.data).__name__ if self.data is not None else "NoneType"
    )
    backend_data_repr = repr(self.data)
    return f"AbstractTensor ({backend_class} ({backend_data_repr}))"


def __len__(self):
    if DEBUG:
        print(f"__len__ called on {self.__class__.__name__}")
    data = self.data
    if data is None:
        raise ValueError("__len__ called on empty tensor")
    return len(data)

