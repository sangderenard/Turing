"""Tests for the ASCII image viewer menu screen clearing."""

from unittest import mock

from src.rendering.ascii_diff import image_viewer


def test_menu_clears_screen_once(monkeypatch):
    """Ensure the menu clears the screen before displaying options."""

    # Simulate user immediately exiting the menu.
    monkeypatch.setattr("builtins.input", lambda *_: "9")

    with mock.patch.object(image_viewer, "full_clear_and_reset_cursor") as clear:
        image_viewer.menu()

    # The clear function should be invoked at least once per menu iteration.
    assert clear.call_count >= 1

