from src.compiler import repo_world_refs_shell as S

TRACKED = frozenset({
    "tools/gate/run_gate.sh", "tools/gate/lib.sh", "tools/gate/check.py", "tools/run.bat",
    "tools/tool.py", "tools/helper.cmd", "tools/hp.ps1", "top.py", "scripts/a.sh",
})
SH = "tools/gate/run_gate.sh"
BAT = "tools/run.bat"


def sh(text, path=SH, tracked=TRACKED):
    return S.references(path, text, tracked)


def pairs(text, **kw):
    return [(r[0], r[1], r[3]) for r in sh(text, **kw)]


def test_contract_constants():
    assert S.LANGUAGE == "shell"
    assert S.SUFFIXES == (".sh", ".bash", ".bat", ".cmd")


def test_source_forms():
    assert pairs("source lib.sh\n. ./lib.sh\n. tools/gate/lib.sh\n") == [
        (1, "shell-source", "tools/gate/lib.sh"),
        (2, "shell-source", "tools/gate/lib.sh"),
        (3, "shell-source", "tools/gate/lib.sh"),
    ]


def test_call_forms():
    text = "bash lib.sh\nsh ./lib.sh\n./lib.sh arg\npython check.py\npython3 tools/tool.py -x\npython -u top.py\n"
    assert pairs(text) == [
        (1, "shell-call", "tools/gate/lib.sh"),
        (2, "shell-call", "tools/gate/lib.sh"),
        (3, "shell-call", "tools/gate/lib.sh"),
        (4, "shell-call", "tools/gate/check.py"),
        (5, "shell-call", "tools/tool.py"),
        (6, "shell-call", "top.py"),
    ]


def test_statement_positions_and_prefixes():
    text = "set -e; python check.py\nif true; then bash lib.sh; fi\nFOO=1 python3 check.py\nexec ./lib.sh\nx && sh lib.sh\n"
    assert [r[0] for r in sh(text)] == [1, 2, 3, 4, 5]


def test_resolution_dir_then_root():
    # dir wins when both exist
    both = TRACKED | {"top.py", "tools/gate/top.py"}
    assert pairs("python top.py", tracked=both) == [(1, "shell-call", "tools/gate/top.py")]
    # root fallback
    assert pairs("python3 top.py") == [(1, "shell-call", "top.py")]
    assert pairs("bash scripts/a.sh") == [(1, "shell-call", "scripts/a.sh")]


def test_unresolved():
    out = sh("python gone.py\nsource nope.sh\nbash ../../../../x.sh\n")
    assert [(r[3], r[4]) for r in out] == [
        (None, "not tracked"), (None, "not tracked"), (None, "outside repository")]


def test_non_script_unresolved_silent():
    assert sh("bash -c 'echo hi'\npython -m pkg\nsource ~/.profile\ncat x.txt\n") == []


def test_variables_and_substitutions_skipped():
    text = 'source "$DIR/lib.sh"\nbash $X/a.sh\npython "$(dirname "$0")/check.py"\n./`x`.sh\n. ${A}/b.sh\n'
    assert sh(text) == []


def test_absolute_skipped():
    assert sh("python /usr/bin/x.py\nsource /etc/profile\n/bin/sh\n") == []


def test_comments_skipped():
    text = "#!/bin/bash\n# python check.py\npython check.py # trailing\necho '#' ; bash lib.sh\n"
    assert [r[0] for r in sh(text)] == [3, 4]


def test_crlf():
    out = sh("a\r\nsource lib.sh\r\npython check.py\r\n")
    assert [(r[0], r[3]) for r in out] == [(2, "tools/gate/lib.sh"), (3, "tools/gate/check.py")]


def test_batch_forms():
    text = (
        "@echo off\r\n"
        "REM python tool.py\r\n"
        ":: call helper.cmd\r\n"
        "call helper.cmd\r\n"
        "python tool.py arg\r\n"
        "python \"%~dp0tool.py\"\r\n"
        "call \"%~dp0helper.cmd\"\r\n"
        "powershell -NoProfile -File hp.ps1\r\n"
        ".\\helper.cmd\r\n"
        "python ..\\top.py\r\n"
    )
    out = sh(text, path=BAT)
    assert [(r[0], r[1], r[3]) for r in out] == [
        (4, "batch-call", "tools/helper.cmd"),
        (5, "batch-call", "tools/tool.py"),
        (6, "batch-call", "tools/tool.py"),
        (7, "batch-call", "tools/helper.cmd"),
        (8, "batch-call", "tools/hp.ps1"),
        (9, "batch-call", "tools/helper.cmd"),
        (10, "batch-call", "top.py"),
    ]


def test_batch_dp0_anywhere_and_dir_only():
    out = sh('start "" "%~dp0tool.py"\r\n', path=BAT)
    assert [(r[1], r[3]) for r in out] == [("batch-call", "tools/tool.py")]
    # %~dp0 never falls back to the root
    out = sh("python %~dp0top.py\r\n", path=BAT)
    assert out[0][3] is None and out[0][4] == "not tracked"


def test_batch_other_percent_vars_skipped():
    assert sh("python %TOOLS%\\tool.py\r\ncall %X%\\a.bat\r\n", path=BAT) == []


def test_garbage_does_not_raise():
    for g in ("", "\x00", "'", '"', "python", "source", "call", "%~dp0", "$(", "`", "�" * 40):
        assert isinstance(sh(g), list)
        assert isinstance(sh(g, path=BAT), list)


def test_deterministic():
    t = "source lib.sh\npython check.py\nbash gone.sh\n"
    assert sh(t) == sh(t)
