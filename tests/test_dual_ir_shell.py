from src.common.tensors.accelerator_backends.aot_compile import AOTCompilation
from src.common.tensors.accelerator_backends.dual_ir_shell import (
    DualIRShell,
    compose_dual_ir_shell,
)
from src.common.tensors.accelerator_backends.profiled_c_shell import (
    CLaunchProfile,
    ShellLanguage,
)


def _aot(entrypoint="f", numeric="numeric-program", control=None):
    return AOTCompilation(
        entrypoint=entrypoint,
        outputs={},
        compiled_shell_program=numeric,
        shell_control_program=control,
        deployment=None,
        shell=DualIRShell(compiled_shell_program=numeric, shell_control_program=control),
    )


def test_shell_pairs_the_same_numeric_and_control_the_aot_already_returns():
    shell = DualIRShell(compiled_shell_program="numeric", shell_control_program="control")
    assert shell.compiled_shell_program == "numeric"
    assert shell.shell_control_program == "control"


def test_a_bare_shell_has_no_profile_no_log_no_children():
    shell = DualIRShell(compiled_shell_program="numeric")
    assert shell.shell_control_program is None
    assert shell.profile is None
    assert shell.log_messages == ()
    assert shell.children == ()


def test_compose_reads_the_aot_output_without_altering_it():
    aot = _aot(entrypoint="affine", numeric="fused", control="control")
    shell = compose_dual_ir_shell(aot)
    assert shell.compiled_shell_program == "fused"
    assert shell.shell_control_program == "control"
    assert shell.name == "affine"


def test_compose_accepts_an_explicit_name_profile_and_log():
    aot = _aot()
    profile = CLaunchProfile(shell_ns=10, device_ns=5, status=1, language=ShellLanguage.C)
    shell = compose_dual_ir_shell(
        aot, profile=profile, log_messages=("started",), name="renamed"
    )
    assert shell.name == "renamed"
    assert shell.profile is profile
    assert shell.log_messages == ("started",)


def test_rollup_profile_is_none_when_nothing_in_the_tree_has_one():
    root = DualIRShell(compiled_shell_program="numeric")
    assert root.rollup_profile() is None


def test_rollup_profile_sums_timings_across_the_whole_tree():
    left = DualIRShell(
        compiled_shell_program="numeric",
        profile=CLaunchProfile(shell_ns=100, device_ns=80, status=1, language=ShellLanguage.C),
    )
    right = DualIRShell(
        compiled_shell_program="numeric",
        profile=CLaunchProfile(shell_ns=50, device_ns=40, status=1, language=ShellLanguage.C),
    )
    root = DualIRShell(compiled_shell_program="numeric", children=(left, right))

    rolled = root.rollup_profile()

    assert rolled.shell_ns == 150
    assert rolled.device_ns == 120
    assert rolled.language == ShellLanguage.C


def test_rollup_profile_reports_the_worst_status_in_the_tree():
    ok = DualIRShell(
        compiled_shell_program="numeric",
        profile=CLaunchProfile(shell_ns=1, device_ns=1, status=1, language=ShellLanguage.C),
    )
    failed = DualIRShell(
        compiled_shell_program="numeric",
        profile=CLaunchProfile(shell_ns=1, device_ns=1, status=2, language=ShellLanguage.C),
    )
    root = DualIRShell(compiled_shell_program="numeric", children=(ok, failed))

    assert root.rollup_profile().status == 2


def test_rollup_profile_falls_back_to_unknown_language_on_disagreement():
    c_leaf = DualIRShell(
        compiled_shell_program="numeric",
        profile=CLaunchProfile(shell_ns=1, device_ns=1, status=1, language=ShellLanguage.C),
    )
    fortran_leaf = DualIRShell(
        compiled_shell_program="numeric",
        profile=CLaunchProfile(shell_ns=1, device_ns=1, status=1, language=ShellLanguage.FORTRAN),
    )
    root = DualIRShell(compiled_shell_program="numeric", children=(c_leaf, fortran_leaf))

    assert root.rollup_profile().language == ShellLanguage.UNKNOWN


def test_rollup_log_walks_the_whole_tree_depth_first_to_the_root():
    grandchild = DualIRShell(compiled_shell_program="numeric", log_messages=("grandchild",))
    child = DualIRShell(
        compiled_shell_program="numeric",
        log_messages=("child",),
        children=(grandchild,),
    )
    root = DualIRShell(
        compiled_shell_program="numeric", log_messages=("root",), children=(child,)
    )

    assert root.rollup_log() == ("root", "child", "grandchild")
