"""Every deploy script that writes a secedit policy template writes the signature secedit requires: "$CHICAGO$".

Start-HostedLlm.ps1, Admin-Control-Service.ps1 and Setup-LiaisonService.ps1 wrote "$CHICAGO" (2026-11-17). secedit
refused the template, the batch logon right was never granted, and the hosted Llama task never ran (0x80070569).
"""
import re
from pathlib import Path

DEPLOY = Path(__file__).resolve().parents[1] / "examples" / "meowdoku" / "deploy"


def _scripts():
    return [p for p in DEPLOY.glob("*.ps1") if "[Privilege Rights]" in p.read_text(encoding="utf-8", errors="replace")]


def test_every_template_signature_is_chicago_dollar():
    found = _scripts()
    assert {p.name for p in found} >= {"Start-HostedLlm.ps1", "Admin-Control-Service.ps1", "Setup-LiaisonService.ps1",
                                       "Setup-ServiceAccount.ps1", "Setup-DevAccount.ps1"}
    for script in found:
        text = script.read_text(encoding="utf-8", errors="replace")
        line = next(l for l in text.splitlines() if "signature=" in l and "CHICAGO" in l)
        assert re.search(r"CHICAGO`\$", line), (script.name, line.strip())

