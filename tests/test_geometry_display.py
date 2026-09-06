import numpy as np
import pytest

from src.compiler.geometry_display import GeometryFrame, HEADER, decode_geometry_frame
from src.compiler.shell_io import (
    NATIVE_PROCESS_SHELL, ShellIOABI, ShellIOManifest, ShellIORequest,
    plan_shell_stack,
)


def test_geometry_packet_owns_frame_data_and_rejects_invalid_native_spans():
    vertices = np.array([
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 1],
    ], dtype=np.float32)
    frame = GeometryFrame(
        19, 7, 3, (800, 600), np.eye(4), vertices,
        np.array([0, 1, 2], dtype=np.uint32), vertices[:2],
        np.array([[0, 0, 0, 1, 1, 1, 1, 6]]), "attempt 7 · revision 3",
    )
    packet = frame.encode()
    vertices[:] = 0
    decoded = decode_geometry_frame(packet)
    assert (decoded.generation, decoded.attempt, decoded.revision) == (19, 7, 3)
    assert decoded.vertices[1, 0] == 1
    assert decoded.overlay == frame.overlay
    assert decoded.viewport == (800, 600)
    with pytest.raises(ValueError):
        decoded.vertices[0, 0] = 9
    with pytest.raises(ValueError):
        decoded.vertices.setflags(write=True)
    for invalid in (packet[:4], packet[:-1], packet + b"x"):
        with pytest.raises(ValueError, match="truncated|size"):
            decode_geometry_frame(invalid)
    header = list(HEADER.unpack_from(packet))
    header[6] = 0xFFFFFFFF  # Native descriptor claims impossible vertex span.
    with pytest.raises(ValueError, match="size"):
        decode_geometry_frame(HEADER.pack(*header) + packet[HEADER.size:])
    invalid = bytearray(packet)
    index_offset = HEADER.size + 64 + 3 * 28
    invalid[index_offset:index_offset + 4] = (3).to_bytes(4, "little")
    with pytest.raises(ValueError, match="index outside"):
        decode_geometry_frame(invalid)


def test_geometry_abi_is_explicit_and_pixel_shell_does_not_claim_support():
    assert "geometry_display" not in ShellIOABI().to_mapping()
    geometry = ShellIOABI(geometry_display=True).to_mapping()["geometry_display"]
    assert geometry["header_format"] == HEADER.format
    assert geometry["completion_identity"] == ["generation", "attempt", "revision"]
    manifest = ShellIOManifest((ShellIORequest.create("geometry_display"),))
    with pytest.raises(ValueError, match="no shell stack"):
        plan_shell_stack("native_library", manifest, (NATIVE_PROCESS_SHELL,))
