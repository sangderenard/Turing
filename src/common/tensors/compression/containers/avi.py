"""Streaming AVI/OpenDML container for MJPEG video and optional PCM audio."""

from __future__ import annotations

from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path
import struct
from typing import BinaryIO, Iterable

import numpy as np

from ..pcm import PCMFormat, RationalAudioScheduler, encode_pcm


AVIIF_KEYFRAME = 0x10
AVIF_HASINDEX = 0x10
AVI_INDEX_OF_INDEXES = 0x00
AVI_INDEX_OF_CHUNKS = 0x01


def _chunk(fourcc: bytes, payload: bytes) -> bytes:
    if len(fourcc) != 4:
        raise ValueError("RIFF chunk identifiers must contain four bytes")
    padding = b"\x00" if len(payload) & 1 else b""
    return fourcc + struct.pack("<I", len(payload)) + payload + padding


def _list(list_type: bytes, payload: bytes) -> bytes:
    return _chunk(b"LIST", list_type + payload)


def _jpeg_dimensions(frame: bytes) -> tuple[int, int]:
    """Read dimensions from a standalone JPEG frame without decoding pixels."""
    if not frame.startswith(b"\xFF\xD8") or not frame.endswith(b"\xFF\xD9"):
        raise ValueError("MJPEG frames must be complete JPEG images")
    position = 2
    standalone = {0x01, *range(0xD0, 0xDA)}
    sof_markers = {
        0xC0, 0xC1, 0xC2, 0xC3,
        0xC5, 0xC6, 0xC7,
        0xC9, 0xCA, 0xCB,
        0xCD, 0xCE, 0xCF,
    }
    while position + 1 < len(frame):
        if frame[position] != 0xFF:
            position += 1
            continue
        while position < len(frame) and frame[position] == 0xFF:
            position += 1
        if position >= len(frame):
            break
        marker = frame[position]
        position += 1
        if marker in standalone or marker == 0xD9:
            continue
        if position + 2 > len(frame):
            break
        length = int.from_bytes(frame[position:position + 2], "big")
        if length < 2 or position + length > len(frame):
            raise ValueError("JPEG frame contains an invalid marker length")
        if marker in sof_markers:
            if length < 8:
                raise ValueError("JPEG frame header is too short")
            height = int.from_bytes(
                frame[position + 3:position + 5], "big"
            )
            width = int.from_bytes(
                frame[position + 5:position + 7], "big"
            )
            return width, height
        if marker == 0xDA:
            break
        position += length
    raise ValueError("JPEG frame has no supported start-of-frame marker")


def _superindex_placeholder(chunk_id: bytes, capacity: int) -> bytes:
    payload = struct.pack(
        "<HBBI4sIII",
        4,
        0,
        AVI_INDEX_OF_INDEXES,
        0,
        chunk_id,
        0,
        0,
        0,
    ) + bytes(capacity * 16)
    return _chunk(b"indx", payload)


def _avi_headers(
    width: int,
    height: int,
    *,
    rate: int,
    scale: int,
    pcm_format: PCMFormat | None,
    opendml: bool,
    superindex_capacity: int,
) -> tuple[bytes, dict[str, int]]:
    microseconds = round(1_000_000 * scale / rate)
    stream_count = 1 + int(pcm_format is not None)
    avih = struct.pack(
        "<14I",
        microseconds,
        0,
        0,
        AVIF_HASINDEX,
        0,
        0,
        stream_count,
        0,
        width,
        height,
        0, 0, 0, 0,
    )
    video_strh = struct.pack(
        "<4s4sIHHIIIIIIIIhhhh",
        b"vids",
        b"MJPG",
        0,
        0,
        0,
        0,
        scale,
        rate,
        0,
        0,
        0,
        0xFFFFFFFF,
        0,
        0,
        0,
        width,
        height,
    )
    video_strf = struct.pack(
        "<IiiHH4sIiiII",
        40,
        width,
        height,
        1,
        24,
        b"MJPG",
        0,
        0,
        0,
        0,
        0,
    )
    video_payload = (
        _chunk(b"strh", video_strh)
        + _chunk(b"strf", video_strf)
    )
    if opendml:
        video_payload += _superindex_placeholder(
            b"00dc", superindex_capacity
        )
    stream_payload = _list(b"strl", video_payload)

    if pcm_format is not None:
        audio_strh = struct.pack(
            "<4s4sIHHIIIIIIIIhhhh",
            b"auds",
            b"\x00\x00\x00\x00",
            0,
            0,
            0,
            0,
            pcm_format.block_align,
            pcm_format.bytes_per_second,
            0,
            0,
            0,
            0xFFFFFFFF,
            pcm_format.block_align,
            0,
            0,
            0,
            0,
        )
        audio_payload = (
            _chunk(b"strh", audio_strh)
            + _chunk(b"strf", pcm_format.wave_format_ex())
        )
        if opendml:
            audio_payload += _superindex_placeholder(
                b"01wb", superindex_capacity
            )
        stream_payload += _list(b"strl", audio_payload)

    if opendml:
        stream_payload += _list(b"odml", _chunk(b"dmlh", bytes(248)))
    headers = _list(
        b"hdrl",
        _chunk(b"avih", avih) + stream_payload,
    )

    video_strh_id = headers.index(b"strh")
    video_strf_id = headers.index(b"strf", video_strh_id + 4)
    locations = {
        "avih": headers.index(b"avih") + 8,
        "video_strh": video_strh_id + 8,
        "video_strf": video_strf_id + 8,
    }
    cursor = video_strf_id + 4
    if opendml:
        video_indx_id = headers.index(b"indx", cursor)
        locations["video_indx"] = video_indx_id + 8
        cursor = video_indx_id + 4
    if pcm_format is not None:
        audio_strh_id = headers.index(b"strh", cursor)
        audio_strf_id = headers.index(b"strf", audio_strh_id + 4)
        locations["audio_strh"] = audio_strh_id + 8
        cursor = audio_strf_id + 4
        if opendml:
            audio_indx_id = headers.index(b"indx", cursor)
            locations["audio_indx"] = audio_indx_id + 8
            cursor = audio_indx_id + 4
    if opendml:
        locations["dmlh"] = headers.index(b"dmlh", cursor) + 8
    return headers, locations


@dataclass(frozen=True)
class _IndexEntry:
    chunk_id: bytes
    chunk_position: int
    payload_position: int
    size: int
    flags: int
    duration: int
    stream: int


@dataclass
class _Segment:
    riff_start: int
    movi_list_start: int
    movi_type_position: int
    entries: list[_IndexEntry] = field(default_factory=list)
    data_bytes: int = 0


class MJPEGAVIWriter:
    """Append JPEG/PCM chunks and finalize AVI 1.0 or OpenDML indexes."""

    def __init__(
        self,
        path: str | Path,
        *,
        width: int,
        height: int,
        fps: int | float | Fraction = 30,
        pcm_format: PCMFormat | None = None,
        opendml: bool = False,
        segment_bytes: int = 1 << 30,
        superindex_capacity: int = 256,
    ) -> None:
        if width < 1 or height < 1 or width > 0x7FFF or height > 0x7FFF:
            raise ValueError("AVI dimensions must fit the stream rectangle")
        cadence = Fraction(fps).limit_denominator(1_000_000)
        if cadence <= 0:
            raise ValueError("AVI frame rate must be positive")
        if segment_bytes < 256:
            raise ValueError("OpenDML segment_bytes must be at least 256")
        if superindex_capacity < 1:
            raise ValueError("superindex_capacity must be positive")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.width = int(width)
        self.height = int(height)
        self.rate = int(cadence.numerator)
        self.scale = int(cadence.denominator)
        self.pcm_format = pcm_format
        self.opendml = bool(opendml)
        self.segment_bytes = int(segment_bytes)
        self.superindex_capacity = int(superindex_capacity)
        self._file: BinaryIO = self.path.open("w+b")
        self._closed = False
        self._segments: list[_Segment] = []
        self._superindexes: dict[int, list[tuple[int, int, int]]] = {
            0: [],
            1: [],
        }
        self._video_frames = 0
        self._audio_sample_frames = 0
        self._max_video_chunk = 0
        self._max_audio_chunk = 0

        headers, relative_locations = _avi_headers(
            self.width,
            self.height,
            rate=self.rate,
            scale=self.scale,
            pcm_format=self.pcm_format,
            opendml=self.opendml,
            superindex_capacity=self.superindex_capacity,
        )
        self._file.write(b"RIFF\x00\x00\x00\x00AVI ")
        self._header_start = self._file.tell()
        self._file.write(headers)
        self._locations = {
            name: self._header_start + offset
            for name, offset in relative_locations.items()
        }
        self._start_movi_segment(first=True)

    def _start_movi_segment(self, *, first: bool) -> None:
        if not first:
            riff_start = self._file.tell()
            self._file.write(b"RIFF\x00\x00\x00\x00AVIX")
        else:
            riff_start = 0
        movi_list_start = self._file.tell()
        self._file.write(b"LIST\x00\x00\x00\x00movi")
        self._segments.append(
            _Segment(
                riff_start=riff_start,
                movi_list_start=movi_list_start,
                movi_type_position=movi_list_start + 8,
            )
        )

    @property
    def _segment(self) -> _Segment:
        return self._segments[-1]

    def _patch_u32(self, position: int, value: int) -> None:
        if value < 0 or value > 0xFFFFFFFF:
            raise OverflowError("RIFF 32-bit field overflow")
        current = self._file.tell()
        self._file.seek(position)
        self._file.write(struct.pack("<I", value))
        self._file.seek(current)

    def _write_standard_index(
        self,
        segment: _Segment,
        *,
        stream: int,
        chunk_id: bytes,
    ) -> None:
        entries = [entry for entry in segment.entries if entry.stream == stream]
        if not entries:
            return
        base = segment.movi_type_position
        payload = struct.pack(
            "<HBBI4sQI",
            2,
            0,
            AVI_INDEX_OF_CHUNKS,
            len(entries),
            chunk_id,
            base,
            0,
        )
        payload += b"".join(
            struct.pack(
                "<II",
                entry.payload_position - base,
                (
                    entry.size
                    if entry.flags & AVIIF_KEYFRAME
                    else entry.size | 0x80000000
                ),
            )
            for entry in entries
        )
        index_position = self._file.tell()
        index_chunk = _chunk(f"ix{stream:02d}".encode("ascii"), payload)
        self._file.write(index_chunk)
        self._superindexes[stream].append(
            (
                index_position,
                len(index_chunk),
                sum(entry.duration for entry in entries),
            )
        )

    def _finish_segment(self, *, start_next: bool) -> None:
        segment = self._segment
        if self.opendml:
            # OpenDML partial indexes are themselves movi chunks. The
            # superindex in each stream header points at these ix## chunks.
            self._write_standard_index(
                segment, stream=0, chunk_id=b"00dc"
            )
            if self.pcm_format is not None:
                self._write_standard_index(
                    segment, stream=1, chunk_id=b"01wb"
                )
        movi_end = self._file.tell()
        self._patch_u32(
            segment.movi_list_start + 4,
            movi_end - (segment.movi_list_start + 8),
        )

        if segment is self._segments[0]:
            legacy_payload = b"".join(
                struct.pack(
                    "<4sIII",
                    entry.chunk_id,
                    entry.flags,
                    entry.chunk_position - segment.movi_type_position,
                    entry.size,
                )
                for entry in segment.entries
            )
            self._file.write(_chunk(b"idx1", legacy_payload))

        segment_end = self._file.tell()
        self._patch_u32(
            segment.riff_start + 4,
            segment_end - segment.riff_start - 8,
        )
        if start_next:
            if len(self._segments) >= self.superindex_capacity:
                raise OverflowError(
                    "OpenDML superindex capacity exhausted; increase "
                    "superindex_capacity"
                )
            self._start_movi_segment(first=False)

    def _append_chunk(
        self,
        chunk_id: bytes,
        payload: bytes,
        *,
        stream: int,
        flags: int,
        duration: int,
    ) -> None:
        if self._closed:
            raise ValueError("cannot append to a closed AVI writer")
        payload = bytes(payload)
        chunk_storage = 8 + len(payload) + (len(payload) & 1)
        if (
            self.opendml
            and self._segment.entries
            and self._segment.data_bytes + chunk_storage > self.segment_bytes
        ):
            self._finish_segment(start_next=True)
        chunk_position = self._file.tell()
        self._file.write(chunk_id)
        self._file.write(struct.pack("<I", len(payload)))
        payload_position = self._file.tell()
        self._file.write(payload)
        if len(payload) & 1:
            self._file.write(b"\x00")
        self._segment.entries.append(
            _IndexEntry(
                chunk_id=chunk_id,
                chunk_position=chunk_position,
                payload_position=payload_position,
                size=len(payload),
                flags=flags,
                duration=duration,
                stream=stream,
            )
        )
        self._segment.data_bytes += chunk_storage

    def append_frame(self, frame: bytes) -> None:
        frame = bytes(frame)
        if _jpeg_dimensions(frame) != (self.width, self.height):
            raise ValueError("JPEG frame dimensions do not match the AVI stream")
        self._append_chunk(
            b"00dc",
            frame,
            stream=0,
            flags=AVIIF_KEYFRAME,
            duration=1,
        )
        self._video_frames += 1
        self._max_video_chunk = max(self._max_video_chunk, len(frame))

    def append_audio(self, pcm: bytes) -> None:
        if self.pcm_format is None:
            raise ValueError("AVI writer has no PCM audio stream")
        pcm = bytes(pcm)
        if len(pcm) % self.pcm_format.block_align:
            raise ValueError("PCM chunk is not aligned to complete sample frames")
        sample_frames = len(pcm) // self.pcm_format.block_align
        if sample_frames == 0:
            return
        self._append_chunk(
            b"01wb",
            pcm,
            stream=1,
            flags=0,
            duration=sample_frames,
        )
        self._audio_sample_frames += sample_frames
        self._max_audio_chunk = max(self._max_audio_chunk, len(pcm))

    def append_audio_tensor(
        self,
        samples,
        *,
        gain: float = 1.0,
        clip: bool = True,
    ) -> None:
        if self.pcm_format is None:
            raise ValueError("AVI writer has no PCM audio stream")
        self.append_audio(
            encode_pcm(
                samples,
                pcm_format=self.pcm_format,
                gain=gain,
                clip=clip,
            )
        )

    def _patch_superindex(self, stream: int, location_name: str) -> None:
        entries = self._superindexes[stream]
        if len(entries) > self.superindex_capacity:
            raise OverflowError("OpenDML superindex capacity exceeded")
        data = self._locations[location_name]
        self._patch_u32(data + 4, len(entries))
        current = self._file.tell()
        self._file.seek(data + 24)
        for offset, size, duration in entries:
            self._file.write(struct.pack("<QII", offset, size, duration))
        self._file.seek(current)

    def close(self) -> Path:
        if self._closed:
            return self.path
        self._finish_segment(start_next=False)

        video_bytes_per_second = (
            self._max_video_chunk * self.rate + self.scale - 1
        ) // self.scale
        max_bytes_per_second = video_bytes_per_second
        if self.pcm_format is not None:
            max_bytes_per_second += self.pcm_format.bytes_per_second
        self._patch_u32(self._locations["avih"] + 4, max_bytes_per_second)
        first_riff_video_frames = sum(
            entry.duration
            for entry in self._segments[0].entries
            if entry.stream == 0
        )
        self._patch_u32(
            self._locations["avih"] + 16,
            (
                first_riff_video_frames
                if self.opendml
                else self._video_frames
            ),
        )
        self._patch_u32(
            self._locations["avih"] + 28,
            max(self._max_video_chunk, self._max_audio_chunk),
        )
        self._patch_u32(
            self._locations["video_strh"] + 32,
            self._video_frames,
        )
        self._patch_u32(
            self._locations["video_strh"] + 36,
            self._max_video_chunk,
        )
        self._patch_u32(
            self._locations["video_strf"] + 20,
            self._max_video_chunk,
        )
        if self.pcm_format is not None:
            self._patch_u32(
                self._locations["audio_strh"] + 32,
                self._audio_sample_frames,
            )
            self._patch_u32(
                self._locations["audio_strh"] + 36,
                self._max_audio_chunk,
            )
        if self.opendml:
            self._patch_u32(self._locations["dmlh"], self._video_frames)
            self._patch_superindex(0, "video_indx")
            if self.pcm_format is not None:
                self._patch_superindex(1, "audio_indx")

        self._file.flush()
        self._file.close()
        self._closed = True
        return self.path

    def __enter__(self) -> "MJPEGAVIWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is None:
            self.close()
        else:
            self._file.close()
            self._closed = True


def encode_tensor_mjpeg_frames(frames, *, jpeg_options=None) -> tuple[bytes, ...]:
    """Encode a tensor frame stack into shell-streamable JPEG packets."""

    from ...abstraction import AbstractTensor
    from ..jpeg.frame import encode_jfif

    if not isinstance(frames, AbstractTensor):
        raise TypeError("frames must be an AbstractTensor")
    options = {} if jpeg_options is None else dict(jpeg_options)
    if frames.ndims() == 2 or (
        frames.ndims() == 3 and frames.shape[-1] == 3
    ):
        source = (frames,)
    elif frames.ndims() == 3:
        source = (frames[index] for index in range(frames.shape[0]))
    elif frames.ndims() == 4 and frames.shape[-1] == 3:
        source = (frames[index] for index in range(frames.shape[0]))
    else:
        raise ValueError("MJPEG packets require one image or a frame stack")
    return tuple(encode_jfif(frame, **options) for frame in source)


class DoubleBufferedAVISink:
    """Shell-owned asynchronous AVI sink with two bounded host buffers."""

    def __init__(
        self,
        path,
        *,
        width,
        height,
        fps,
        pcm_format,
        audio,
        opendml=True,
        segment_bytes=1 << 30,
    ):
        self.writer = MJPEGAVIWriter(
            path,
            width=width,
            height=height,
            fps=fps,
            pcm_format=pcm_format,
            opendml=opendml,
            segment_bytes=segment_bytes,
        )
        self.audio = audio
        self.scheduler = RationalAudioScheduler(
            sample_rate=pcm_format.sample_rate,
            fps=fps,
        )
        self.audio_position = 0
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="avi-shell-sink",
        )
        self._buffers = [None, None]
        self._futures = [None, None]
        self._next_buffer = 0

    def _flush(self, payload):
        for packet_index, (video, audio) in enumerate(payload):
            try:
                self.writer.append_frame(video)
            except Exception as error:
                encoded = bytes(video)
                last_nonzero = max(
                    (
                        index
                        for index, value in enumerate(encoded)
                        if value
                    ),
                    default=-1,
                )
                raise ValueError(
                    "AVI sink received an invalid compiled video packet; "
                    f"buffer_packet={packet_index}, bytes={len(encoded)}, "
                    f"soi={encoded.find(bytes((0xFF, 0xD8)))}, "
                    f"eoi={encoded.rfind(bytes((0xFF, 0xD9)))}, "
                    f"last_nonzero={last_nonzero}, "
                    f"head={encoded[:16].hex()}, "
                    f"tail={encoded[-16:].hex()}"
                ) from error
            self.writer.append_audio(audio)

    def submit(self, video_packets):
        """Fill one buffer, then flush it while the shell fills the other."""

        from ...abstraction import AbstractTensor

        payload = []
        for video in video_packets:
            if isinstance(video, tuple) and len(video) == 2:
                octets, byte_count = video
                octets = (
                    octets.numpy()
                    if callable(getattr(octets, "numpy", None))
                    else np.asarray(octets)
                )
                byte_count = (
                    byte_count.numpy()
                    if callable(getattr(byte_count, "numpy", None))
                    else np.asarray(byte_count)
                )
                count = int(np.asarray(byte_count).reshape(-1)[0])
                flat_octets = np.asarray(octets).reshape(-1)
                if count < 0 or count > flat_octets.size:
                    raise ValueError(
                        "compiled resident packet byte count is outside "
                        "its octet allocation"
                    )
                video = bytes(
                    flat_octets[:count].astype(np.uint8, copy=False)
                )
            sample_count = self.scheduler.samples_for_next_frame()
            stop = min(self.audio.shape[0], self.audio_position + sample_count)
            samples = self.audio[self.audio_position:stop]
            self.audio_position = stop
            if samples.shape[0] < sample_count:
                missing = sample_count - samples.shape[0]
                samples = AbstractTensor.cat(
                    (
                        samples,
                        AbstractTensor.zeros(
                            (missing,),
                            cls=type(self.audio),
                        ),
                    ),
                    dim=0,
                )
            payload.append(
                (
                    bytes(video),
                    encode_pcm(samples, pcm_format=self.writer.pcm_format),
                )
            )

        index = self._next_buffer
        prior = self._futures[index]
        if prior is not None:
            prior.result()
        self._buffers[index] = tuple(payload)
        self._futures[index] = self._executor.submit(
            self._flush,
            self._buffers[index],
        )
        self._next_buffer = 1 - index

    def close(self):
        for future in self._futures:
            if future is not None:
                future.result()
        self._executor.shutdown(wait=True)
        return self.writer.close()

    def __enter__(self) -> "DoubleBufferedAVISink":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is None:
            self.close()
        else:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self.writer._file.close()
            self.writer._closed = True


def write_mjpeg_avi(
    path: str | Path,
    frames: Iterable[bytes],
    *,
    width: int,
    height: int,
    fps: int | float | Fraction = 30,
    pcm_format: PCMFormat | None = None,
    audio_chunks: Iterable[bytes] | None = None,
    opendml: bool = False,
    segment_bytes: int = 1 << 30,
) -> Path:
    """Write complete JPEG frames and optional already-encoded PCM chunks."""
    with MJPEGAVIWriter(
        path,
        width=width,
        height=height,
        fps=fps,
        pcm_format=pcm_format,
        opendml=opendml,
        segment_bytes=segment_bytes,
    ) as writer:
        if audio_chunks is None:
            for frame in frames:
                writer.append_frame(frame)
        else:
            audio_iterator = iter(audio_chunks)
            for frame in frames:
                writer.append_frame(frame)
                try:
                    writer.append_audio(next(audio_iterator))
                except StopIteration:
                    audio_iterator = iter(())
            for chunk in audio_iterator:
                writer.append_audio(chunk)
    return writer.path


def write_grayscale_mjpeg_avi(
    path: str | Path,
    frames,
    *,
    fps: int | float | Fraction = 30,
    jpeg_options: dict | None = None,
    **writer_options,
) -> Path:
    """Encode a grayscale tensor frame stack and write an MJPEG AVI."""
    from ...abstraction import AbstractTensor
    from ..jpeg.frame import encode_grayscale_jfif

    if not isinstance(frames, AbstractTensor):
        raise TypeError("frames must be an AbstractTensor")
    dimensions = frames.ndims()
    if dimensions == 2:
        height, width = frames.shape
        frame_source = (frames,)
    elif dimensions == 3:
        frame_count, height, width = frames.shape
        if frame_count < 1:
            raise ValueError("AVI frame stack must contain at least one frame")
        frame_source = (frames[index] for index in range(frame_count))
    else:
        raise ValueError(
            "grayscale AVI input must have shape (height, width) or "
            "(frames, height, width)"
        )
    options = {} if jpeg_options is None else dict(jpeg_options)
    return write_mjpeg_avi(
        path,
        (encode_grayscale_jfif(frame, **options) for frame in frame_source),
        width=int(width),
        height=int(height),
        fps=fps,
        **writer_options,
    )


def write_tensor_mjpeg_avi(
    path: str | Path | MJPEGAVIWriter,
    frames,
    *,
    fps: int | float | Fraction = 30,
    jpeg_options: dict | None = None,
    audio=None,
    sample_rate: int = 48_000,
    channels: int | None = None,
    pcm_dtype: str = "s16le",
    audio_gain: float = 1.0,
    clip: bool = True,
    pad_audio: bool = True,
    opendml: bool = False,
    segment_bytes: int = 1 << 30,
) -> Path:
    """Encode tensor video plus optional normalized tensor audio into AVI."""
    from ...abstraction import AbstractTensor
    from ..jpeg.frame import encode_jfif

    if not isinstance(frames, AbstractTensor):
        raise TypeError("frames must be an AbstractTensor")
    dimensions = frames.ndims()
    if dimensions == 2:
        height, width = frames.shape
        frame_source = (frames,)
        frame_count = 1
    elif dimensions == 3 and frames.shape[-1] == 3:
        height, width, _ = frames.shape
        frame_source = (frames,)
        frame_count = 1
    elif dimensions == 3:
        frame_count, height, width = frames.shape
        frame_source = (frames[index] for index in range(frame_count))
    elif dimensions == 4 and frames.shape[-1] == 3:
        frame_count, height, width, _ = frames.shape
        frame_source = (frames[index] for index in range(frame_count))
    else:
        raise ValueError(
            "AVI input must be one grayscale/RGB image or a stack of them"
        )
    if frame_count < 1:
        raise ValueError("AVI frame stack must contain at least one frame")

    pcm_format = None
    if audio is not None:
        if not isinstance(audio, AbstractTensor):
            raise TypeError("audio must be an AbstractTensor")
        inferred_channels = 1 if audio.ndims() == 1 else audio.shape[-1]
        pcm_format = PCMFormat(
            sample_rate=sample_rate,
            channels=int(channels or inferred_channels),
            sample_format=pcm_dtype,
        )
    options = {} if jpeg_options is None else dict(jpeg_options)
    supplied_writer = isinstance(path, MJPEGAVIWriter)
    writer = (
        path
        if supplied_writer
        else MJPEGAVIWriter(
            path,
            width=int(width),
            height=int(height),
            fps=fps,
            pcm_format=pcm_format,
            opendml=opendml,
            segment_bytes=segment_bytes,
        )
    )
    if supplied_writer:
        if (writer.width, writer.height) != (int(width), int(height)):
            raise ValueError("AVI batch dimensions changed between appends")
        if (writer.rate, writer.scale) != (
            Fraction(fps).limit_denominator(1_000_000).numerator,
            Fraction(fps).limit_denominator(1_000_000).denominator,
        ):
            raise ValueError("AVI batch frame rate changed between appends")

    scheduler = getattr(writer, "_tensor_audio_scheduler", None)
    if pcm_format is not None and scheduler is None:
        scheduler = RationalAudioScheduler(sample_rate=sample_rate, fps=fps)
        writer._tensor_audio_scheduler = scheduler
        writer._tensor_audio_position = 0
    audio_position = int(getattr(writer, "_tensor_audio_position", 0))

    try:
        for frame in frame_source:
            writer.append_frame(encode_jfif(frame, **options))
            if scheduler is None:
                continue
            sample_count = scheduler.samples_for_next_frame()
            available = max(0, min(sample_count, audio.shape[0] - audio_position))
            chunk = audio[audio_position:audio_position + available]
            audio_position += available
            if available < sample_count:
                if not pad_audio:
                    raise ValueError("audio ends before the video stream")
                missing = sample_count - available
                shape = (
                    (missing,)
                    if pcm_format.channels == 1
                    else (missing, pcm_format.channels)
                )
                silence = AbstractTensor.zeros(shape, cls=type(audio))
                chunk = (
                    silence
                    if available == 0
                    else AbstractTensor.cat((chunk, silence), dim=0)
                )
            writer.append_audio_tensor(
                chunk,
                gain=audio_gain,
                clip=clip,
            )
        writer._tensor_audio_position = audio_position
    finally:
        if not supplied_writer:
            writer.close()
    return writer.path


__all__ = [
    "MJPEGAVIWriter",
    "write_grayscale_mjpeg_avi",
    "write_mjpeg_avi",
    "write_tensor_mjpeg_avi",
]
