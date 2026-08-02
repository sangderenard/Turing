from src.compiler.hierarchical_plan import PlanCall, PlanClosure, PlanLine
from src.compiler.multipart_graph_io import (
    BackpressurePolicy,
    IOMode,
    MultipartChannel,
    MultipartIOBuilder,
    MultipartPort,
    OutputWheel,
    PortDirection,
    PortSchema,
    multipart_io_from_hierarchy,
)
from src.compiler.shell_io import ShellIOManifest, ShellIORequest


def test_hierarchy_bindings_become_typed_cross_part_channels():
    child = PlanClosure(
        "child",
        (1,),
        (PlanLine.create("add", inputs=(1,), outputs=(2,)),),
        closure_id=1,
    )
    root = PlanClosure(
        "root",
        (10,),
        (PlanCall(
            7,
            child,
            argument_bindings=((10, 1),),
            result_bindings=((2, 11),),
        ),),
        closure_id=0,
    )

    io = multipart_io_from_hierarchy(
        root,
        root_output_value_ids=(11,),
        schemas={
            (0, 10): PortSchema("float64", (4,), 32),
            (1, 1): PortSchema("float64", (4,), 32),
            (1, 2): PortSchema("float64", (4,), 32),
            (0, 11): PortSchema("float64", (4,), 32),
        },
    )

    assert io.part_ids == (0, 1)
    assert [channel.channel_id for channel in io.channels] == [
        "call7.argument0",
        "call7.result0",
    ]
    assert {port.port_id for port in io.ports if port.external} == {
        "root.input0",
        "root.output0",
    }
    assert io.to_mapping()["schema"] == "turing-multipart-graph-io"


def test_io_validation_rejects_schema_mismatch():
    builder = MultipartIOBuilder()
    builder.add_port(MultipartPort(
        "out", 0, PortDirection.OUTPUT, 1, "result",
        PortSchema("float32"),
    ))
    builder.add_port(MultipartPort(
        "in", 1, PortDirection.INPUT, 2, "argument",
        PortSchema("int32"),
    ))
    builder.add_channel(MultipartChannel(
        "bad", "out", ("in",), IOMode.VALUE,
    ))

    try:
        builder.finish()
    except ValueError as error:
        assert "incompatible schemas" in str(error)
    else:
        raise AssertionError("schema mismatch should fail validation")


def test_hierarchy_io_carries_explicit_shell_requirements():
    root = PlanClosure("root", (), (), closure_id=0)
    manifest = ShellIOManifest((
        ShellIORequest.create("keyboard"),
        ShellIORequest.create("display_double_buffer", optional=True),
    ))

    mapping = multipart_io_from_hierarchy(root, shell_io=manifest).to_mapping()

    assert mapping["shell_io"]["requests"] == [
        {"capability": "keyboard", "optional": False, "attributes": {}},
        {
            "capability": "display_double_buffer",
            "optional": True,
            "attributes": {},
        },
    ]


def test_stream_channel_can_declare_a_fixed_backpressured_output_wheel():
    builder = MultipartIOBuilder()
    builder.add_port(MultipartPort(
        "out", 0, PortDirection.OUTPUT, 1, "irregular",
        PortSchema("float32", token_bytes=4),
    ))
    builder.add_port(MultipartPort(
        "in", 1, PortDirection.INPUT, 2, "irregular",
        PortSchema("float32", token_bytes=4),
    ))
    builder.add_channel(MultipartChannel(
        "events", "out", ("in",), IOMode.STREAM, capacity=64,
        output_wheel=OutputWheel(
            64, token_bytes=4, backpressure=BackpressurePolicy.YIELD,
            low_watermark=16, high_watermark=48,
        ),
    ))

    wheel = builder.finish().to_mapping()["channels"][0]["output_wheel"]

    assert wheel["capacity"] == 64
    assert wheel["backpressure"] == "yield"
    assert wheel["counter_fields"][:2] == [
        "read_sequence", "write_sequence",
    ]
