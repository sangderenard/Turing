"""The games with direct page addresses in the exported Meowdoku site."""

from pathlib import Path

GAMES = (
    "meowdoku", "chess", "checkers", "wordle", "chess_vs_net", "chess_arena",
    "lichess", "space_invaders", "bullet_hell", "platformer", "delve",
    "side_scroller", "tilequest", "boardwalk", "worldwalk", "fountain",
    "fountain3d", "marble_madness", "tipjar", "purrline", "mobroad",
)


def export_game_pages(out):
    """Give every existing game a stable entry page over the shared game runtime."""
    out = Path(out)
    play_shell = (out / "play.html").read_text(encoding="utf-8")
    if '<body data-game="meowdoku">' not in play_shell or '<title>Meowdoku</title>' not in play_shell:
        raise ValueError("game shell markers changed")
    for game in GAMES:
        page = play_shell.replace('<body data-game="meowdoku">',
                                  f'<body data-game="{game}" data-page-game="{game}">', 1)
        page = page.replace("<title>Meowdoku</title>", f"<title>{game.replace('_', ' ').title()} · Meowdoku</title>", 1)
        (out / f"play-{game}.html").write_text(page, encoding="utf-8")
    game_links = "\n".join(f'<li><a href="/play-{game}.html">{game.replace("_", " ").title()}</a></li>' for game in GAMES)
    (out / "games.html").write_text(
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Games · Meowdoku</title><style>body{font:16px/1.5 system-ui;background:#121822;color:#edf2fb;max-width:48rem;margin:2rem auto;padding:0 1rem}'
        'a{color:#a9cfff}li{margin:.5rem 0}</style></head><body><p><a href="/">← Pages</a></p><h1>Games</h1><ul>'
        + game_links + '</ul><script type="module" src="/site-nav.js" data-site-nav="allow"></script></body></html>',
        encoding="utf-8")
