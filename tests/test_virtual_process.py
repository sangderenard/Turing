from src.compiler.virtual_process import (
    VirtualProgramRegistry,
    VirtualProgramResult,
    split_windows_command_line,
)


def test_windows_command_line_split_preserves_quotes_and_backslashes():
    assert split_windows_command_line(
        '"C:\\Program Files\\tool.exe" alpha "two words" tail\\'
    ) == (r"C:\Program Files\tool.exe", "alpha", "two words", "tail\\")


def test_virtual_program_registry_resolves_path_and_executes_only_registered_bundle():
    observed = []
    registry = VirtualProgramRegistry()
    registry.register(
        "/c/tools/reduce.exe",
        bundle_reference="bundle:math/reduce@sha256:abc",
        executor_reference="card-executor:v1",
        executor=lambda invocation: (
            observed.append(invocation)
            or VirtualProgramResult(7, b"reduced\r\n", execution_units=3)
        ),
    )

    deployment = registry.launch(
        "reduce",
        ("4", "8"),
        deployment_id=19,
        current_directory="/c/work",
        path_search=(r"C:\tools",),
        environment={"MODE": "exact"},
    )

    assert deployment is not None
    assert deployment.program.bundle_reference == "bundle:math/reduce@sha256:abc"
    assert deployment.result.standard_output == b"reduced\r\n"
    assert deployment.result.execution_units == 3
    assert observed[0].arguments == ("4", "8")
    assert registry.launch(
        "host-only.exe", (), deployment_id=20, current_directory="/c/work",
    ) is None
