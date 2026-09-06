from src.common.tensors.accelerator_backends.aot_compile import compile_ast_aot
from src.compiler.fortran_c_shell import _class_surface_ssa_program
from src.compiler.ssa_fortran_backend import emit_module


def _compile_source(source: str, entrypoint: str, feeds: dict[str, object], name: str):
    compilation = compile_ast_aot(
        source,
        entrypoint,
        feeds,
        precompile_only=True,
        bake_mode="whole_program",
        require_planned_shells=True,
        project_captured_hierarchy=False,
    )
    return _class_surface_ssa_program(compilation, name)


def _records(module):
    return tuple(
        record
        for records in module.call_table.values()
        for record in records
    )


def test_deep_authored_call_chain_reaches_static_call_fixed_point():
    module, outputs, exports = _compile_source(
        """
def leaf(value):
    return value + 1

def middle(value):
    return leaf(value)

def root(value):
    return middle(value)
""",
        "root",
        {"value": 2.0},
        "deep_call_chain",
    )

    records = _records(module)
    assert records
    unresolved = tuple(
        record for record in records if record.resolution == "unresolved"
    )
    assert not unresolved, {
        "records": records,
        "functions": {
            name: tuple(
                (
                    block_name,
                    instruction.op,
                    None if instruction.res is None else instruction.res.id,
                    tuple(value.id for value in instruction.args),
                    dict(instruction.attributes),
                )
                for block_name, block in function.blocks.items()
                for instruction in block.instrs
            )
            for name, function in module.functions.items()
        },
    }
    unresolved = tuple(
        record for record in _records(module)
        if record.resolution == "unresolved"
    )
    assert not unresolved, unresolved
    assert emit_module(
        module,
        name="deep_call_chain",
        outputs=outputs,
        extra_roots=exports,
    ).complete


def test_void_authored_call_is_a_static_call_not_omitted_execution():
    module, outputs, exports = _compile_source(
        """
def observe(value):
    checked = value

def root(value):
    observe(value)
    return value
""",
        "root",
        {"value": 2.0},
        "void_source_call",
    )

    set_records = tuple(
        record for record in _records(module)
        if record.callee_name == "observe"
    )
    assert set_records
    assert all(record.resolution == "native_call" for record in set_records), {
        "set_records": set_records,
        "run": tuple(
            (
                block_name,
                instruction.op,
                None if instruction.res is None else instruction.res.id,
                tuple(value.id for value in instruction.args),
                dict(instruction.attributes),
            )
            for name, function in module.functions.items()
            if name.endswith("__root")
            for block_name, block in function.blocks.items()
            for instruction in block.instrs
        ),
    }
    unresolved = tuple(
        record for record in _records(module)
        if record.resolution == "unresolved"
    )
    assert not unresolved, unresolved
    assert emit_module(
        module,
        name="void_source_call",
        outputs=outputs,
        extra_roots=exports,
    ).complete
