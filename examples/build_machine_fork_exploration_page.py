from types import SimpleNamespace

from src.compiler import machine_fork_exploration as mfe
from src.compiler.amd64_machine_semantics import default_effect_handlers
from src.compiler.machine_execution import (
    MachineExecutionOrchestrator, MachineExecutionState, ReversibleMachineExecutor,
)
from src.compiler.machine_path_forest import MachinePathForest
from src.compiler.site_bundle import build_program_bundle

entry = 0x2000
program = SimpleNamespace(
    image=SimpleNamespace(image_base=entry, entrypoint_rva=0),
    functions=(SimpleNamespace(report=SimpleNamespace(instructions=())),),
)
orchestrator = MachineExecutionOrchestrator(program, effect_handlers=default_effect_handlers())
reversible = ReversibleMachineExecutor.create(orchestrator, MachineExecutionState(pc=entry))
forest = MachinePathForest(reversible, maximum_heads=64)

build_program_bundle(
    open(mfe.__file__, encoding="utf-8").read(), "C:/dev/Powershell",
    source_filename="machine_fork_exploration.py",
    entrypoint="explore_forking_paths",
    python_package="src.compiler",
    probes={"forest": forest, "root_head_id": {"literal": 0}},
)
