from src.compiler import reversible_forest_from_binary as rffb
from src.compiler.site_bundle import build_program_bundle

build_program_bundle(
    open(rffb.__file__, encoding="utf-8").read(), "C:/dev/Powershell",
    source_filename="reversible_forest_from_binary.py",
    entrypoint="open_decode_step_and_fork",
    python_package="src.compiler",
    probes={"root_head_id": {"literal": 0}},
    progress_sink=print,
)
