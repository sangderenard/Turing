"""Operator-owned FFmpeg row declaration and authenticated Radio HLS file reads."""
import json
import os
from pathlib import Path
import re


_SEGMENT = re.compile(r"radio_[0-9]{3,10}\.ts\Z")
_MIME = {"radio.m3u8": "application/vnd.apple.mpegurl"}


def output_dir(state):
    return state.content_dir / "radio" / "hls"


def clear_output(state):
    directory = output_dir(state)
    for path in directory.iterdir():
        if path.name == "radio.m3u8" or _SEGMENT.fullmatch(path.name):
            if path.is_file(): path.unlink()


def prepare(state, hls):
    """Called only by explicit jukebox resume; no GET or constructor starts an encoder."""
    ffmpeg = Path(hls["ffmpeg_executable"])
    roots = hls["core_roots"]
    if (not ffmpeg.is_absolute() or not ffmpeg.is_file() or not isinstance(roots, list) or
            not roots or any(not isinstance(root, str) or not Path(root).is_absolute() or
                             '\n' in root or '\r' in root or ';' in root for root in roots)):
        raise ValueError("invalid Radio HLS operator configuration")
    directory = output_dir(state)
    directory.mkdir(parents=True, exist_ok=True)
    arguments = ["-hide_banner", "-loglevel", "error", "-probesize", "32",
                 "-analyzeduration", "0", "-f", "s16le", "-ar", "48000", "-ac", "2",
                 "-i", "pipe:0", "-c:a", "aac", "-b:a", "96k", "-f", "hls",
                 "-muxdelay", "0", "-muxpreload", "0",
                 "-hls_time", "1", "-hls_list_size", "3", "-hls_flags", "delete_segments",
                 "-hls_segment_filename", str(directory / "radio_%03d.ts"), "-y",
                 str(directory / "radio.m3u8")]
    rows = state.run_dir / "nodus_radio_hls_rows.txt"
    temporary = rows.with_suffix(".tmp")
    lines = ["row exec.radio.hls", "name Radio AAC HLS", f"program {ffmpeg}",
             f"directory {directory}", "mode stream", "queue_bytes 1048576"]
    lines.extend(f"argument {value}" for value in arguments)
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, rows)
    return {"NODUS_EXEC_ROWS": str(rows), "NODUS_EXEC_CORE_ROOTS": ";".join(roots)}


def read(state, name):
    if name != "radio.m3u8" and not _SEGMENT.fullmatch(name):
        raise FileNotFoundError(name)
    config = json.loads(state.radio_json.read_text(encoding="utf-8"))
    try:
        enabled = config["jukebox"]["native"]["hls"]["enabled"] is True
    except (KeyError, TypeError):
        enabled = False
    if config.get("enabled") is not True or config.get("source") != "file" or not enabled:
        raise FileNotFoundError(name)
    path = output_dir(state) / name
    if not path.is_file() or path.resolve().parent != output_dir(state).resolve() or \
            path.stat().st_size > 4 * 1024 * 1024:
        raise FileNotFoundError(name)
    return path.read_bytes(), _MIME.get(name, "video/mp2t")
