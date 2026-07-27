from io import BytesIO
import ctypes
import struct
import sys

import pytest

from src.common.tensors.abstraction import AbstractTensor as AT
from src.common.tensors.compression.containers.avi import (
    MJPEGAVIWriter,
    write_mjpeg_avi,
)
from src.common.tensors.compression.pcm import PCMFormat
from src.common.tensors.compression.jpeg.frame import encode_grayscale_jfif


def _frame(phase):
    samples = [
        [
            int((row * 11 + column * 7 + phase * 29) % 256)
            for column in range(16)
        ]
        for row in range(16)
    ]
    return encode_grayscale_jfif(AT.tensor(samples))


def test_video_only_mjpeg_avi_contains_indexed_readable_frames(tmp_path):
    image_module = pytest.importorskip("PIL.Image")
    with AT.use_backend("numpy"):
        frames = [_frame(phase) for phase in range(3)]
    path = write_mjpeg_avi(
        tmp_path / "tensor_frames.avi",
        frames,
        width=16,
        height=16,
        fps=12,
    )
    data = path.read_bytes()

    assert data[:4] == b"RIFF"
    assert data[8:12] == b"AVI "
    assert struct.unpack_from("<I", data, 4)[0] == len(data) - 8
    avih = data.index(b"avih") + 8
    strh = data.index(b"strh") + 8
    assert struct.unpack_from("<I", data, avih + 16)[0] == 3
    assert struct.unpack_from("<I", data, strh + 32)[0] == 3

    movi_type = data.index(b"movi")
    idx1 = data.index(b"idx1")
    index_size = struct.unpack_from("<I", data, idx1 + 4)[0]
    assert index_size == 3 * 16
    for frame_index, expected in enumerate(frames):
        entry = idx1 + 8 + frame_index * 16
        chunk_id, flags, offset, size = struct.unpack_from(
            "<4sIII", data, entry
        )
        assert chunk_id == b"00dc"
        assert flags == 0x10
        chunk = movi_type + offset
        assert data[chunk:chunk + 4] == b"00dc"
        assert size == len(expected)
        payload = data[chunk + 8:chunk + 8 + size]
        assert payload == expected
        image = image_module.open(BytesIO(payload))
        image.load()
        assert image.size == (16, 16)


def test_avi_writer_rejects_mismatched_frame_dimensions(tmp_path):
    with AT.use_backend("numpy"):
        frame = _frame(0)
    with pytest.raises(ValueError, match="dimensions"):
        write_mjpeg_avi(
            tmp_path / "wrong.avi",
            [frame],
            width=24,
            height=16,
            fps=10,
        )


def test_abstract_tensor_avi_redirect_encodes_a_frame_stack(tmp_path):
    image_module = pytest.importorskip("PIL.Image")
    samples = [
        [
            [
                int((row * 11 + column * 7 + phase * 29) % 256)
                for column in range(16)
            ]
            for row in range(16)
        ]
        for phase in range(3)
    ]
    with AT.use_backend("numpy"):
        destination = AT.tensor(samples).avi(
            path=tmp_path / "tensor_stack.avi",
            fps=15,
        )

    data = destination.read_bytes()
    avih = data.index(b"avih") + 8
    assert struct.unpack_from("<I", data, avih + 16)[0] == 3
    movi_type = data.index(b"movi")
    idx1 = data.index(b"idx1")
    for frame_index in range(3):
        entry = idx1 + 8 + frame_index * 16
        _, _, offset, size = struct.unpack_from("<4sIII", data, entry)
        chunk = movi_type + offset
        image = image_module.open(
            BytesIO(data[chunk + 8:chunk + 8 + size])
        )
        image.load()
        assert image.mode == "L"
        assert image.size == (16, 16)


def test_abstract_tensor_avi_redirect_encodes_rgb_frames(tmp_path):
    image_module = pytest.importorskip("PIL.Image")
    samples = [
        [
            [
                (
                    (row * 13 + phase * 31) % 256,
                    (column * 17 + phase * 19) % 256,
                    ((row + column) * 9 + phase * 23) % 256,
                )
                for column in range(16)
            ]
            for row in range(16)
        ]
        for phase in range(2)
    ]
    with AT.use_backend("numpy"):
        destination = AT.tensor(samples).avi(
            path=tmp_path / "tensor_rgb.avi",
            fps=12,
        )

    data = destination.read_bytes()
    movi_type = data.index(b"movi")
    idx1 = data.index(b"idx1")
    for frame_index in range(2):
        entry = idx1 + 8 + frame_index * 16
        _, _, offset, size = struct.unpack_from("<4sIII", data, entry)
        chunk = movi_type + offset
        image = image_module.open(
            BytesIO(data[chunk + 8:chunk + 8 + size])
        )
        image.load()
        assert image.mode == "RGB"
        assert image.size == (16, 16)


def test_tensor_avi_interleaves_drift_free_pcm_audio(tmp_path):
    video = [
        [
            [int((row * 9 + column * 5 + phase * 17) % 256) for column in range(16)]
            for row in range(16)
        ]
        for phase in range(3)
    ]
    audio = [
        [0.25 if index & 1 else -0.25, 0.5 if index & 1 else -0.5]
        for index in range(300)
    ]
    with AT.use_backend("numpy"):
        path = AT.tensor(video).avi(
            path=tmp_path / "audio_video.avi",
            fps=10,
            audio=AT.tensor(audio),
            sample_rate=1000,
            channels=2,
        )
    data = path.read_bytes()

    avih = data.index(b"avih") + 8
    assert struct.unpack_from("<I", data, avih + 24)[0] == 2
    audio_strh = data.index(b"auds")
    assert struct.unpack_from("<I", data, audio_strh + 20)[0] == 4
    assert struct.unpack_from("<I", data, audio_strh + 24)[0] == 4000
    assert struct.unpack_from("<I", data, audio_strh + 32)[0] == 300
    audio_strf = data.index(b"strf", audio_strh) + 8
    assert struct.unpack_from("<HHIIHH", data, audio_strf) == (
        1,
        2,
        1000,
        4000,
        4,
        16,
    )

    movi = data.index(b"movi")
    idx1 = data.index(b"idx1")
    index_size = struct.unpack_from("<I", data, idx1 + 4)[0]
    ids = [
        struct.unpack_from("<4s", data, position)[0]
        for position in range(idx1 + 8, idx1 + 8 + index_size, 16)
    ]
    assert ids == [b"00dc", b"01wb"] * 3
    audio_sizes = []
    for position in range(idx1 + 8, idx1 + 8 + index_size, 16):
        chunk_id, _, offset, size = struct.unpack_from("<4sIII", data, position)
        if chunk_id == b"01wb":
            chunk = movi + offset
            audio_sizes.append(size)
            assert data[chunk:chunk + 4] == b"01wb"
    assert audio_sizes == [400, 400, 400]


@pytest.mark.skipif(
    sys.platform != "win32",
    reason="independent Video for Windows reader is Windows-only",
)
def test_windows_avi_reader_seeks_all_video_and_pcm_samples(tmp_path):
    with AT.use_backend("numpy"):
        frames = [_frame(phase) for phase in range(3)]
    pcm_format = PCMFormat(sample_rate=1000, channels=2)
    path = tmp_path / "system_reader.avi"
    with MJPEGAVIWriter(
        path,
        width=16,
        height=16,
        fps=10,
        pcm_format=pcm_format,
    ) as writer:
        for frame in frames:
            writer.append_frame(frame)
            writer.append_audio(bytes(100 * pcm_format.block_align))

    library = ctypes.WinDLL("avifil32")
    library.AVIFileOpenW.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_wchar_p,
        ctypes.c_uint,
        ctypes.c_void_p,
    ]
    library.AVIFileOpenW.restype = ctypes.c_long
    library.AVIFileGetStream.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_uint,
        ctypes.c_long,
    ]
    library.AVIFileGetStream.restype = ctypes.c_long
    library.AVIStreamStart.argtypes = [ctypes.c_void_p]
    library.AVIStreamStart.restype = ctypes.c_long
    library.AVIStreamLength.argtypes = [ctypes.c_void_p]
    library.AVIStreamLength.restype = ctypes.c_long
    library.AVIStreamRead.argtypes = [
        ctypes.c_void_p,
        ctypes.c_long,
        ctypes.c_long,
        ctypes.c_void_p,
        ctypes.c_long,
        ctypes.POINTER(ctypes.c_long),
        ctypes.POINTER(ctypes.c_long),
    ]
    library.AVIStreamRead.restype = ctypes.c_long
    library.AVIStreamRelease.argtypes = [ctypes.c_void_p]
    library.AVIFileRelease.argtypes = [ctypes.c_void_p]

    def fourcc(value):
        return sum(ord(character) << (8 * index) for index, character in enumerate(value))

    library.AVIFileInit()
    avi_file = ctypes.c_void_p()
    try:
        assert library.AVIFileOpenW(
            ctypes.byref(avi_file), str(path), 0, None
        ) == 0
        for kind, stream_type, positions, expected_lengths in (
            ("video", "vids", (0, 1, 2), (3,)),
            ("audio", "auds", (0, 150, 299), (300,)),
        ):
            stream = ctypes.c_void_p()
            assert library.AVIFileGetStream(
                avi_file,
                ctypes.byref(stream),
                fourcc(stream_type),
                0,
            ) == 0
            try:
                assert library.AVIStreamStart(stream) == 0
                assert library.AVIStreamLength(stream) in expected_lengths
                for position in positions:
                    output = ctypes.create_string_buffer(65536)
                    byte_count = ctypes.c_long()
                    sample_count = ctypes.c_long()
                    assert library.AVIStreamRead(
                        stream,
                        position,
                        1,
                        output,
                        len(output),
                        ctypes.byref(byte_count),
                        ctypes.byref(sample_count),
                    ) == 0, (kind, position)
                    assert sample_count.value == 1
                    if kind == "video":
                        assert output.raw[:2] == b"\xFF\xD8"
                    else:
                        assert byte_count.value == pcm_format.block_align
            finally:
                library.AVIStreamRelease(stream)
    finally:
        if avi_file:
            library.AVIFileRelease(avi_file)
        library.AVIFileExit()


def _riff_sections(data):
    sections = []
    position = 0
    while position < len(data):
        assert data[position:position + 4] == b"RIFF"
        size = struct.unpack_from("<I", data, position + 4)[0]
        sections.append((position, data[position + 8:position + 12], size))
        position += size + 8
    assert position == len(data)
    return sections


def _verify_superindex(data, position, expected_chunk_id, expected_ix):
    payload = position + 8
    longs, subtype, index_type, count, chunk_id, _, _, _ = struct.unpack_from(
        "<HBBI4sIII", data, payload
    )
    assert (longs, subtype, index_type) == (4, 0, 0)
    assert chunk_id == expected_chunk_id
    assert count >= 2
    for entry in range(count):
        offset, size, duration = struct.unpack_from(
            "<QII", data, payload + 24 + entry * 16
        )
        assert data[offset:offset + 4] == expected_ix
        assert struct.unpack_from("<I", data, offset + 4)[0] + 8 == size
        standard = offset + 8
        (
            entry_longs,
            entry_subtype,
            entry_type,
            entry_count,
            indexed_chunk,
            base,
            _,
        ) = struct.unpack_from("<HBBI4sQI", data, standard)
        assert (entry_longs, entry_subtype, entry_type) == (2, 0, 1)
        assert indexed_chunk == expected_chunk_id
        assert duration > 0
        for item in range(entry_count):
            relative, indexed_size = struct.unpack_from(
                "<II", data, standard + 24 + item * 8
            )
            payload_position = base + relative
            payload_size = indexed_size & 0x7FFFFFFF
            if expected_chunk_id == b"00dc":
                assert indexed_size & 0x80000000 == 0
                assert data[payload_position:payload_position + 2] == b"\xFF\xD8"
            else:
                assert indexed_size & 0x80000000
            assert payload_size > 0


def test_opendml_segments_video_and_audio_with_two_level_indexes(tmp_path):
    pcm_format = PCMFormat(sample_rate=1000, channels=1)
    with AT.use_backend("numpy"):
        frames = [_frame(index) for index in range(5)]
    path = tmp_path / "segmented.avi"
    with MJPEGAVIWriter(
        path,
        width=16,
        height=16,
        fps=10,
        pcm_format=pcm_format,
        opendml=True,
        segment_bytes=500,
        superindex_capacity=32,
    ) as writer:
        for frame in frames:
            writer.append_frame(frame)
            writer.append_audio(bytes(200))

    data = path.read_bytes()
    sections = _riff_sections(data)
    assert sections[0][1] == b"AVI "
    assert all(form == b"AVIX" for _, form, _ in sections[1:])
    assert len(sections) >= 3
    dmlh = data.index(b"dmlh") + 8
    assert struct.unpack_from("<I", data, dmlh)[0] == 5
    video_indx = data.index(b"indx")
    audio_indx = data.index(b"indx", video_indx + 4)
    _verify_superindex(data, video_indx, b"00dc", b"ix00")
    _verify_superindex(data, audio_indx, b"01wb", b"ix01")
