from src.common.dt_system.engine_api import create_identity_assembly
from src.common.dt_system.state_table import StateTable
from src.common.dt_system.classic_mechanics.engines import DemoState, GravityEngine, GroundCollisionEngine


def test_identity_assembly_attaches_multiple_engines():
    state = DemoState(
        pos=[(0.0, 0.0)],
        vel=[(0.0, 0.0)],
        acc=[(0.0, 0.0)],
        mass=[1.0],
        springs=[],
        rest_len={},
        k_spring={},
        pneu_damp={},
    )
    table = StateTable()
    items = range(len(state.pos))
    schema = lambda i: {"pos": state.pos[i], "mass": state.mass[i]}
    assembly = create_identity_assembly(table, schema, items, group_label="craft")
    grav = GravityEngine(state, assembly=assembly)
    ground = GroundCollisionEngine(state, assembly=assembly)
    ok_g, _, _ = grav.step(0.1, None, table)
    ok_c, _, _ = ground.step(0.1, None, table)
    assert ok_g and ok_c
    assert grav.uuids == ground.uuids == assembly.uuids
    assert table.get_group_vertices("craft") == set(assembly.uuids)
