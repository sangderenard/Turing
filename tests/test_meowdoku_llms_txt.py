"""[LLMS] /llms.txt directs agents to the MCP and REST commands, for anyone (case 7ab138b9813e4cceb687b9fc47149329)."""
from __future__ import annotations

import http.client
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples"), str(ROOT / "tests")]

from meowdoku import access as A  # noqa: E402
from test_meowdoku_access import tiered  # noqa: E402,F401  (the loopback server whose tier the test sets)

LLMS = ROOT / "examples" / "meowdoku" / "web" / "llms.txt"
ORIGIN = "https://stichka.duckdns.org:8766"


def test_it_follows_the_llmstxt_form_and_links_only_the_public_origin():
    text = LLMS.read_text(encoding="utf-8")
    lines = text.splitlines()
    assert lines[0] == "# SpeakToMe" and sum(1 for line in lines if line.startswith("# ")) == 1, "one H1, first"
    assert next(line for line in lines[1:] if line.strip()).startswith("> "), "then the blockquote summary"
    sections = [line[3:] for line in lines if line.startswith("## ")]
    assert sections[:2] == ["Connect", "REST"] and sections[-1] == "Optional"
    links = re.findall(r"\]\((https?://[^)]+)\)", text)
    assert links and all(url.startswith(ORIGIN + "/") or url == "https://llmstxt.org/" for url in links), links
    for name in ("/mcp)", "/api/index)", "/.well-known/oauth-protected-resource/mcp)"):
        assert name in text


def test_a_visitor_with_no_account_gets_the_file_as_text(tiered):
    port, set_tier, sessions = tiered
    assert "/llms.txt" in A.SHADER_FILES
    shutil.copyfile(LLMS, Path(sessions.web_dir) / "llms.txt")
    set_tier("shader")
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    conn.request("GET", "/llms.txt")
    response = conn.getresponse()
    body = response.read().decode("utf-8")
    conn.close()
    assert response.status == 200 and response.getheader("Content-Type", "").startswith("text/plain")
    assert body.startswith("# SpeakToMe"), "the file itself, not the shader page"


def test_it_is_exported_with_the_pages():
    assert '"llms.txt"' in (ROOT / "examples" / "meowdoku" / "web_export.py").read_text(encoding="utf-8")
