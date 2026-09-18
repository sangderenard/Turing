"""Show how a comprehension's nodes are partitioned into dispatch regions.

Patches the two decision points -- loop discovery and scheduled-region
reduction -- and prints what each produced for one repro case.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.compiler import glsl_deployment_strategy as strategy  # noqa: E402
from src.compiler import loop_composer  # noqa: E402
from src.compiler import precompile_to_ssa  # noqa: E402
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa  # noqa: E402

from repro_keyed_get import CASES  # noqa: E402

CONTRACT = (
    Path(__file__).resolve().parents[1]
    / "extraction_contracts"
    / "program_extraction.yaml"
)


def main() -> int:
    case = sys.argv[1]

    original_discover = loop_composer.LoopComposer.discover
    original_reduce = strategy.reduce_scheduled_shader_regions
    original_partition = strategy._control_partition_keys
    original_plan_region = precompile_to_ssa.plan_region_to_ssa_instrs

    def discover(self, graph, *args, **kwargs):
        plans = original_discover(self, graph, *args, **kwargs)
        for plan in plans:
            loop = plan.loop
            print(
                f"[loop] node={loop.node_id} strategy={plan.strategy} "
                f"source={loop.source_type} target={loop.target!r} "
                f"kind={loop.iterator_kind}"
            )
            print(f"       body_nodes={sorted(loop.body_nodes)}")
            print(f"       target_bindings={loop.target_bindings}")
            print(f"       carried={loop.carried_bindings}")
            print(f"       iterable_node={loop.iterable_node} "
                  f"start={loop.start} stop={loop.stop}")
            print(
                "       iteration_outputs="
                + str([
                    (int(o.value_id), int(o.result_value_id))
                    for o in loop.iteration_outputs
                ])
            )
        return plans

    def partition(graph, loop_plans, executable_nodes, *args, **kwargs):
        keys = original_partition(graph, loop_plans, executable_nodes, *args, **kwargs)
        interesting = {
            node_id: key for node_id, key in keys.items() if key
        }
        print(f"[partition] non-empty keys: {interesting}")
        missing = [
            node_id for node_id in executable_nodes if node_id not in keys
        ]
        print(f"[partition] executable={sorted(executable_nodes)}")
        print(f"[partition] without a key: {sorted(missing)}")
        return keys

    def reduce_regions(graph, executable_nodes, **kwargs):
        plan = original_reduce(graph, executable_nodes, **kwargs)
        for index, dispatch in enumerate(plan.dispatches):
            print(f"[region {index}] nodes={sorted(dispatch.node_ids)}")
        return plan

    def plan_region(region):
        lines = [
            (item.opcode, tuple(item.inputs), tuple(item.outputs))
            for item in region.items
            if hasattr(item, 'opcode')
        ]
        print(f'[plan {region.name}] captures={region.captures} lines={lines}')
        return original_plan_region(region)

    precompile_to_ssa.plan_region_to_ssa_instrs = plan_region

    original_build = precompile_to_ssa._ControlSSABuilder.__init__

    def build(self, *a, **k):
        result = original_build(self, *a, **k)
        print(f'[control] collection_bindings={self.program.collection_bindings}')
        print(f'[control] projected={self.program.projected_iterable_bindings}')
        print(f'[control] aliases={dict(self.value_aliases)}')
        return result

    precompile_to_ssa._ControlSSABuilder.__init__ = build
    loop_composer.LoopComposer.discover = discover
    strategy.reduce_scheduled_shader_regions = reduce_regions
    strategy._control_partition_keys = partition
    try:
        lower_ast_source_to_ssa(
            CASES[case], "root", name=case, extraction_contract=CONTRACT,
        )
    finally:
        loop_composer.LoopComposer.discover = original_discover
        strategy.reduce_scheduled_shader_regions = original_reduce
        strategy._control_partition_keys = original_partition
        precompile_to_ssa.plan_region_to_ssa_instrs = original_plan_region
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
