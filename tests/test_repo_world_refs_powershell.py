from src.compiler import repo_world_refs_powershell as P

D = "examples/meowdoku/deploy"
SCRIPT = f"{D}/Migrate-State.ps1"
TRACKED = frozenset({
    f"{D}/Migrate-State.ps1", f"{D}/Start-Apache.ps1", f"{D}/helper.ps1", f"{D}/mod.psm1",
    "examples/meowdoku/authz.py", "examples/meowdoku/state_migrate.py",
    f"{D}/apache_supervisor.py", "tools/x.ps1", "top.ps1",
})

MIGRATE = r"""
$__repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$__args = @((Join-Path $__repo 'examples\meowdoku\authz.py'), 'script-auth', $MyInvocation.MyCommand.Name)
$__gate = & 'C:\Users\alber\AppData\Local\Programs\Python\Python311\python.exe' @__args 2>&1
$Tool = Join-Path $__repo 'examples\meowdoku\state_migrate.py'
& $Python @ToolArgs
"""
START = r"""
$Sup = Join-Path $Repo 'examples\meowdoku\deploy\apache_supervisor.py'
"""


def refs(text, path=SCRIPT, tracked=TRACKED):
    return P.references(path, text, tracked)


def kinds(text, **kw):
    return [(r[1], r[3]) for r in refs(text, **kw)]


def test_contract_constants():
    assert P.LANGUAGE == "powershell"
    assert P.SUFFIXES == (".ps1", ".psm1", ".psd1")


def test_real_migrate_excerpt_finds_python_tools():
    out = refs(MIGRATE)
    targets = {r[3] for r in out if r[1] == "powershell-path"}
    assert targets == {"examples/meowdoku/authz.py", "examples/meowdoku/state_migrate.py"}
    assert all(r[1] == "powershell-path" for r in out)
    assert next(r for r in out if r[3].endswith("authz.py"))[0] == 3


def test_real_start_apache_excerpt():
    assert kinds(START) == [("powershell-path", f"{D}/apache_supervisor.py")]


def test_dot_source_forms():
    text = ". .\\helper.ps1\n. \"$PSScriptRoot\\helper.ps1\"\n. (Join-Path $PSScriptRoot 'helper.ps1')\n"
    out = refs(text)
    assert [(r[0], r[1], r[3]) for r in out] == [
        (1, "powershell-dot-source", f"{D}/helper.ps1"),
        (2, "powershell-dot-source", f"{D}/helper.ps1"),
        (3, "powershell-dot-source", f"{D}/helper.ps1"),
    ]


def test_call_forms():
    text = "& .\\helper.ps1\n& \"$PSScriptRoot\\..\\..\\..\\top.ps1\"\n.\\Start-Apache.ps1 -Force\n"
    assert kinds(text) == [
        ("powershell-call", f"{D}/helper.ps1"),
        ("powershell-call", "top.ps1"),
        ("powershell-call", f"{D}/Start-Apache.ps1"),
    ]


def test_import_module():
    assert kinds("Import-Module .\\mod.psm1\n") == [("powershell-import-module", f"{D}/mod.psm1")]
    assert kinds("Import-Module (Join-Path $PSScriptRoot 'mod.psm1')\n") == [
        ("powershell-import-module", f"{D}/mod.psm1")]
    assert kinds("Import-Module ActiveDirectory\n") == []


def test_python_and_pwsh_file_literals():
    text = "python examples\\meowdoku\\authz.py --x\npwsh -File tools\\x.ps1\npowershell -NoProfile -File .\\helper.ps1\n"
    assert kinds(text) == [
        ("powershell-call", "examples/meowdoku/authz.py"),
        ("powershell-call", "tools/x.ps1"),
        ("powershell-call", f"{D}/helper.ps1"),
    ]


def test_variable_resolution_branches():
    # repo-root only
    assert kinds("Join-Path $r 'examples\\meowdoku\\authz.py'") == [
        ("powershell-path", "examples/meowdoku/authz.py")]
    # script-dir only
    assert kinds("Join-Path $r 'helper.ps1'") == [("powershell-path", f"{D}/helper.ps1")]
    # both tracked -> ambiguous, unresolved
    both = TRACKED | {"helper.ps1"}
    out = refs("Join-Path $r 'helper.ps1'", tracked=both)
    assert out[0][3] is None and out[0][4].startswith("ambiguous")
    # neither tracked, non-script file -> silent
    assert refs("Join-Path $r 'build\\x.db'") == []


def test_non_file_join_is_silent():
    assert refs("Join-Path $PSScriptRoot '..'") == []


def test_unresolved_script_literal():
    out = refs("Join-Path $PSScriptRoot 'gone.py'\n& .\\gone.ps1\n")
    assert [(r[1], r[3], r[4]) for r in out] == [
        ("powershell-path", None, "not tracked"), ("powershell-call", None, "not tracked")]


def test_escape_root_not_clamped():
    out = refs("& \"$PSScriptRoot\\..\\..\\..\\..\\top.ps1\"")
    assert out[0][3] is None and out[0][4] == "outside repository"


def test_other_variables_and_subexpressions_skipped():
    text = ". $x\\a.ps1\n& \"$env:HOME\\a.ps1\"\n. \"$PSScriptRoot\\$name.ps1\"\nJoin-Path $r \"a$b.py\"\n& \"$(Get-X)\\a.ps1\"\n"
    assert refs(text) == []


def test_absolute_skipped():
    assert refs("& 'C:\\x\\y.ps1'\npython C:\\x\\y.py\n") == []


def test_comments_skipped():
    text = "# . .\\helper.ps1\n<#\n& .\\helper.ps1\n#>\n<# x #> . .\\helper.ps1 # tail\n$s = '#not comment' ; . .\\helper.ps1\n"
    out = refs(text)
    assert [r[0] for r in out] == [5, 6]


def test_crlf_and_line_numbers():
    out = refs("a\r\n. .\\helper.ps1\r\n\r\n& .\\mod.psm1\r\n")
    assert [(r[0], r[3]) for r in out] == [(2, f"{D}/helper.ps1"), (4, f"{D}/mod.psm1")]


def test_psd1_manifest():
    out = refs("@{ RootModule = 'mod.psm1' }", path=f"{D}/m.psd1")
    assert [(r[1], r[3]) for r in out] == [("powershell-import-module", f"{D}/mod.psm1")]


def test_garbage_does_not_raise():
    for g in ("", "\x00\x01", "'", '"', "<#", "Join-Path", "& ", ". ", "`" * 5, "Join-Path $a '", "\ufffd" * 50):
        assert isinstance(refs(g), list)
    assert P.references("", "", frozenset()) == []


def test_deterministic():
    assert refs(MIGRATE + START) == refs(MIGRATE + START)
