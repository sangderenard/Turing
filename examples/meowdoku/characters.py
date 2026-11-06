"""The character tile set of the platformer-style games (the climber, the runner, Delve).

The player's sprite is not a fixed frog: the page shows these characters as a column of glass tiles to the left of the
board (the power-up tray's cubes and glyph pieces, ``web/app.js``), a tap chooses one, and the choice is a profile
setting (``profiles.SETTING_KEYS``: ``avatar``), so it persists with the player.  The games draw their player with the
chosen character's mesh (``EntityGame.hero_mesh``); the meshes come from the same glyph builder as every other emoji
(``web_export``), under the names ``hero_<key>``.
"""

from __future__ import annotations

#: key -> emoji, in tile order (top to bottom).  The first is the default.
CHARACTERS = {"frog": "🐸", "cat": "🐱", "dog": "🐶", "fox": "🦊", "panda": "🐼", "rabbit": "🐰", "bear": "🐻", "owl": "🦉"}
DEFAULT = "frog"

#: The glyph meshes web_export builds, by mesh name.
EMOJI = {f"hero_{key}": glyph for key, glyph in CHARACTERS.items()}

#: Material colours (the palette entries the page draws a mesh with), by mesh name.
PALETTE = {
    "hero_frog": [0.45, 0.9, 0.35], "hero_cat": [0.95, 0.72, 0.35], "hero_dog": [0.82, 0.6, 0.36],
    "hero_fox": [0.96, 0.5, 0.18], "hero_panda": [0.92, 0.92, 0.9], "hero_rabbit": [0.93, 0.84, 0.88],
    "hero_bear": [0.62, 0.42, 0.26], "hero_owl": [0.7, 0.56, 0.38],
}


def mesh_of(key: str) -> str:
    return f"hero_{key if key in CHARACTERS else DEFAULT}"


def tile_pieces(chosen: str) -> list[dict]:
    """The tile set as the page's tray pieces: ``cell`` ``char:<index>`` (the page lays them out left of the board),
    ``character`` the key a tap chooses, ``lit`` the chosen one."""
    return [{"id": f"char:{key}", "cell": f"char:{i}", "mesh": mesh_of(key), "color": mesh_of(key), "character": key,
             "lit": key == chosen} for i, key in enumerate(CHARACTERS)]
