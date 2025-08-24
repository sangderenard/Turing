"""Console utilities for screen manipulation and color initialization."""
from __future__ import annotations

from colorama import Cursor, ansi, just_fix_windows_console as _just_fix_windows_console

# Provide no-op fallbacks if colorama is not installed. These stubs ensure the
# module continues to function, albeit without colored output.
class _MockAnsi:
    def __getattr__(self, name: str) -> str:
        return ""

Cursor = Cursor if 'Cursor' in locals() else _MockAnsi()  # type: ignore
ansi = ansi if 'ansi' in locals() else _MockAnsi()  # type: ignore
_just_fix_windows_console = (
    _just_fix_windows_console if '_just_fix_windows_console' in locals() else lambda: None
)  # type: ignore

def init_colorama_for_windows() -> None:
    """Initialize Colorama for proper ANSI escape code handling on Windows."""
    _just_fix_windows_console()

def full_clear_and_reset_cursor() -> None:
    """Clear the terminal screen and move the cursor to the top-left."""
    print(ansi.clear_screen() + Cursor.POS(0, 0), end="", flush=True)

def reset_cursor_to_top() -> None:
    """Move the cursor to the top-left of the terminal screen."""
    print(Cursor.POS(0, 0), end="", flush=True)
