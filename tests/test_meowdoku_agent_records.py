"""[RECORDS] A run record is alive only while the process that wrote it lives (case fd54c5e9404140238c7d5f1f56d73feb).

2026-11-17: a record left by a run that a reboot killed named pid 4772, which Windows had given to a svchost after the
boot, and Codex started no run from 20:10 to 21:12.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "examples" / "meowdoku" / "deploy" / "AgentAccounts.psm1"

SCRIPT = r"""
$ErrorActionPreference = 'Stop'
Import-Module '{module}' -Force
$days = '{days}'
$running = Join-Path $days 'running'
New-Item -ItemType Directory -Force -Path $running | Out-Null
$boot = (Get-CimInstance Win32_OperatingSystem).LastBootUpTime
$began = (Get-Process -Id $PID).StartTime
$taken = @(Get-Process | ForEach-Object Id)
$free = 4000001; while ($taken -contains $free) {{ $free += 4 }}
function Put([string]$name, [int]$id, [datetime]$at) {{
    @{{ kind = 'run'; account = $name; pid = $id; started = $at.ToString('o') }} | ConvertTo-Json -Compress |
        Set-Content -LiteralPath (Join-Path $running "run-$name-$id.json") -Encoding utf8
}}
Put 'live' $PID (Get-Date)
Put 'beforeboot' $PID $boot.AddMinutes(-5)
$reusable = $began.AddSeconds(-30) -gt $boot
if ($reusable) {{ Put 'reused' $PID $began.AddSeconds(-30) }}
Put 'gone' $free (Get-Date)
$kept = @(Get-AgentRecords $days | ForEach-Object account)
$left = @(Get-ChildItem $running -Filter *.json | ForEach-Object {{ ($_.BaseName -split '-')[1] }})
@{{ kept = $kept; left = $left; reusable = $reusable }} | ConvertTo-Json -Compress
"""


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="needs PowerShell 7")
def test_only_the_writing_process_keeps_a_record_alive(tmp_path):
    script = SCRIPT.format(module=str(MODULE).replace("'", "''"), days=str(tmp_path).replace("'", "''"))
    done = subprocess.run(["pwsh", "-NoProfile", "-NonInteractive", "-Command", script],
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    answer = json.loads(done.stdout.strip().splitlines()[-1])
    assert answer["kept"] == ["live"], "only the record of the live process that wrote it"
    assert answer["left"] == ["live"], "every dead record is deleted, as before"


def test_every_launcher_reads_records_through_the_one_rule():
    text = MODULE.read_text(encoding="utf-8")
    body = text[text.index("function Get-AgentRecords"):text.index("# A console as an agent account")]
    assert "Test-AgentRecordAlive $record $boot" in body and "LastBootUpTime" in body
    rule = text[text.index("function Test-AgentRecordAlive"):text.index("function Get-AgentRecords")]
    assert "$written -lt $Boot" in rule and "$began -gt $written.AddSeconds(1)" in rule
