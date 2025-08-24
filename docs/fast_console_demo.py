"""Demonstration for the fast console printer.

This script exercises :class:`src.common.fast_console.cffiPrinter`
to show it can emit formatted text to the terminal on any supported
platform.
"""

from __future__ import annotations

from src.common.fast_console import cffiPrinter


def main() -> None:
    value = 7 * 6
    message = f"The answer is {value:02d}!\n"
    printer = cffiPrinter()
    try:
        printer.print(message)
        printer.flush()
    finally:
        printer.stop()


if __name__ == "__main__":
    main()
