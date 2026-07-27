"""Binary container adapters around encoded tensor media."""

from .avi import (
    MJPEGAVIWriter,
    write_grayscale_mjpeg_avi,
    write_mjpeg_avi,
    write_tensor_mjpeg_avi,
)

__all__ = [
    "MJPEGAVIWriter",
    "write_grayscale_mjpeg_avi",
    "write_mjpeg_avi",
    "write_tensor_mjpeg_avi",
]
