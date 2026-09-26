from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm
from src.transmogrifier.ssa import BasicBlock, Function, IRModule, Instr, SSAValue


def test_static_scalar_compiler_frame_storage_is_private_to_wrapper():
    authored = SSAValue(
        0,
        "float64",
        accounting={"program_abi_storage": "scalar"},
    )
    frame = SSAValue(
        1,
        "int64",
        accounting={"linked_call_frame_storage": "helper"},
    )
    output = SSAValue(2, "int64")
    function = Function("root", [authored, frame], {
        "entry": BasicBlock("entry", [
            Instr("Br", [], None, attributes={"target": "exit"}),
        ], successors=["exit"]),
        "exit": BasicBlock("exit", [
            Instr("Const", [], output, attributes={"constant": 7}),
            Instr("Ret", [output], None),
        ]),
    })
    module = IRModule({function.name: function})

    artifact = emit_ssa_function_to_llvm(module, function.name)

    assert artifact.shortfalls == ()
    assert artifact.buffer_order == (authored.id, output.id)
    assert "%root.frame.0 = alloca i64, i64 1" in artifact.llvm_ir
