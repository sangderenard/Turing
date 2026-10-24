"""Reference-only application products shared by web and native shells.

An application product does not restate a compiled-program ABI, ShellIO ABI,
shader component ABI, or shader deployment record.  It names the files which
already own those contracts and says which execution and presentation products
a shell launches.  This keeps website publication, Android packaging, and iOS
packaging as consumers of one product instead of three application ABIs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


SCHEMA = "turing.application-product.v1"


def _relative_path(value: str) -> str:
    path = PurePosixPath(str(value).replace("\\", "/"))
    if path.is_absolute() or not path.parts or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise ValueError(f"application-product paths must be contained relative paths: {value!r}")
    return path.as_posix()


@dataclass(frozen=True, slots=True)
class ProductFile:
    """One immutable file already published by a compiler/product writer."""

    path: str
    sha256: str

    @classmethod
    def from_root(cls, root: str | Path, path: str | Path) -> "ProductFile":
        relative = _relative_path(Path(path).as_posix())
        data = (Path(root) / relative).read_bytes()
        return cls(relative, hashlib.sha256(data).hexdigest())

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _relative_path(self.path))
        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.sha256
        ):
            raise ValueError("application-product file hashes must be lowercase SHA-256")

    def to_mapping(self) -> dict[str, str]:
        return {"path": self.path, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class ExecutionProduct:
    """The artifact a shell invokes; topology remains in its own manifest."""

    kind: str
    artifact: ProductFile
    manifest: ProductFile | None = None

    def to_mapping(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "kind": self.kind,
            "artifact": self.artifact.to_mapping(),
        }
        if self.manifest is not None:
            result["manifest"] = self.manifest.to_mapping()
        return result


@dataclass(frozen=True, slots=True)
class PresentationProduct:
    """One logical shader component and its physical deployment product."""

    component_id: str
    deployment: ProductFile
    artifact: ProductFile

    def __post_init__(self) -> None:
        if not self.component_id:
            raise ValueError("an application presentation needs a component identity")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "deployment": self.deployment.to_mapping(),
            "artifact": self.artifact.to_mapping(),
        }


@dataclass(frozen=True, slots=True)
class TargetProduct:
    """One target realization of the common program and presentation."""

    execution: ExecutionProduct
    presentation: PresentationProduct | None = None
    runtime_dependencies: tuple[Mapping[str, Any], ...] = ()

    def to_mapping(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "execution": self.execution.to_mapping(),
            "runtime_dependencies": [dict(item) for item in self.runtime_dependencies],
        }
        if self.presentation is not None:
            result["presentation"] = self.presentation.to_mapping()
        return result


@dataclass(frozen=True, slots=True)
class ApplicationProduct:
    """A launchable program expressed entirely as references to owned ABIs."""

    identity: str
    program_api: ProductFile
    entrypoint: str
    shell_io_metadata_key: str = "shell_io"
    targets: Mapping[str, TargetProduct] = field(default_factory=dict)
    default_target: str | None = None

    def __post_init__(self) -> None:
        if not self.identity or not self.entrypoint:
            raise ValueError("application products require an identity and entrypoint")
        if not self.targets:
            raise ValueError("application products require at least one target")
        selected = self.default_target or next(iter(self.targets))
        if selected not in self.targets:
            raise ValueError(f"unknown default application target {selected!r}")
        object.__setattr__(self, "default_target", selected)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "identity": self.identity,
            "program": {
                "api": self.program_api.to_mapping(),
                "entrypoint": self.entrypoint,
            },
            "shell": {
                "program_api_metadata": self.shell_io_metadata_key,
            },
            "default_target": self.default_target,
            "targets": {
                name: product.to_mapping()
                for name, product in sorted(self.targets.items())
            },
        }

    def write(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_mapping(), indent=2) + "\n", encoding="utf-8",
        )
        return destination


def moltenvk_ios_dependency(*, version: str = "1.4.2") -> dict[str, Any]:
    """The public iOS Vulkan runtime dependency, without downloading it.

    The static XCFramework is acquired by the Apple build host and linked into
    the application.  Turing records the dependency; it does not vendor an
    opaque binary or pretend a Windows build can validate an iOS framework.
    """

    releases = {
        "1.4.2": {
            "url": (
                "https://github.com/KhronosGroup/MoltenVK/releases/download/"
                "v1.4.2/MoltenVK-ios.tar"
            ),
            "sha256": "b5d947b1660e6e9fed40b9cd2387e160aaab9e80b775c0cef7e14059405178c1",
            "minimum_ios": "15.0",
        },
    }
    try:
        release = releases[str(version)]
    except KeyError as error:
        raise ValueError(
            f"no pinned MoltenVK iOS dependency receipt for {version!r}"
        ) from error
    return {
        "name": "MoltenVK",
        "version": str(version),
        "kind": "static-xcframework",
        "platform": "ios",
        "repository": "https://github.com/KhronosGroup/MoltenVK",
        "archive": dict(release),
        "product": "MoltenVK/MoltenVK/static/MoltenVK.xcframework",
        "provides": "vulkan-over-metal",
        "shader_input": "spirv",
    }


def android_vulkan_dependency() -> dict[str, Any]:
    """The Android platform Vulkan loader supplied through the NDK."""

    return {
        "name": "vulkan",
        "kind": "platform-library",
        "platform": "android",
        "provider": "android-ndk",
        "link": "vulkan",
        "shader_input": "spirv",
    }


def publish_web_application_product(
    root: str | Path,
    *,
    identity: str,
    api: Any,
    coordinator: str,
    execution_manifest: Mapping[str, Any],
) -> ProductFile:
    """Publish the reusable execution product currently hosted by a web page.

    ``execution_manifest`` is the existing segmented/contiguous Wasm manifest;
    this function moves no execution decisions into the application record.
    It persists that record and the already-generated API, then writes the
    reference-only application product beside them.
    """

    destination = Path(root)
    if api.entry is None:
        raise ValueError("a web application product requires a compiled-program entrypoint")
    directory = destination / "application"
    directory.mkdir(parents=True, exist_ok=True)
    api_relative = "application/program.api.yaml"
    execution_relative = "application/web-execution.json"
    product_relative = "application/application.json"
    api.write(destination / api_relative)
    (destination / execution_relative).write_text(
        json.dumps(dict(execution_manifest), indent=2) + "\n", encoding="utf-8",
    )
    execution = ExecutionProduct(
        "wasm-coordinator",
        ProductFile.from_root(destination, _relative_path(coordinator)),
        ProductFile.from_root(destination, execution_relative),
    )
    product = ApplicationProduct(
        str(identity),
        ProductFile.from_root(destination, api_relative),
        str(api.entry),
        targets={"web": TargetProduct(execution)},
        default_target="web",
    )
    product.write(destination / product_relative)
    return ProductFile.from_root(destination, product_relative)


__all__ = [
    "ApplicationProduct", "ExecutionProduct", "PresentationProduct",
    "ProductFile", "SCHEMA", "TargetProduct",
    "android_vulkan_dependency", "moltenvk_ios_dependency",
    "publish_web_application_product",
]
