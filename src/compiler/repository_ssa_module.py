"""One repository-SSA program with several platform realizations.

The module is the compiler product.  A DLL, WebAssembly binary, WebGPU
compute shader, or graphics shader is only one realization of that product.
Every realization published here therefore carries the digest of the same
logical entry ABI, derived once from the repository-SSA root function.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..transmogrifier.ssa import IRModule, SSAValue


def _returned_values(function: Any) -> tuple[SSAValue, ...]:
    returns = tuple(
        instruction
        for block in function.blocks.values()
        for instruction in block.instrs
        if str(instruction.op).casefold() in {"ret", "return"}
    )
    if len(returns) != 1:
        raise ValueError(
            f"{function.name} must publish exactly one return; found {len(returns)}"
        )
    return tuple(returns[0].args)


def _logical_abi(
    module: IRModule, root_name: str, entry_name: str,
) -> dict[str, Any]:
    function = module.functions[root_name]
    inputs = tuple(function.metadata.get("argument_names", ()))
    if len(inputs) != len(function.args):
        inputs = tuple(f"arg{index}" for index in range(len(function.args)))
    returned = _returned_values(function)
    outputs = tuple(function.metadata.get("output_names", ()))
    if len(outputs) != len(returned):
        outputs = tuple(f"output{index}" for index in range(len(returned)))

    def value_record(name: str, value: SSAValue) -> dict[str, Any]:
        return {
            "name": name,
            "ssa_value_id": int(value.id),
            "dtype": str(value.dtype or "float64"),
            "shape": list(map(int, value.shape or ())),
        }

    return {
        "schema": "turing.repository-ssa-module-abi.v1",
        "entrypoint": entry_name,
        "source_function": root_name,
        "inputs": [
            value_record(name, value)
            for name, value in zip(inputs, function.args, strict=True)
        ],
        "outputs": [
            value_record(name, value)
            for name, value in zip(outputs, returned, strict=True)
        ],
    }


def _digest(record: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        record, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _shortfall_records(items: Sequence[Any]) -> list[dict[str, str]]:
    return [
        {
            "operation": str(getattr(item, "operation", "emission")),
            "reason": str(getattr(item, "reason", item)),
            **(
                {"function": str(item.function)}
                if getattr(item, "function", None) is not None else {}
            ),
        }
        for item in items
    ]


@dataclass(slots=True)
class RepositorySSAModuleAssembly:
    name: str
    root_name: str
    abi: Mapping[str, Any]
    abi_digest: str
    c_artifact: Any
    wasm_artifact: Any
    webgpu_artifact: Any
    graphics_shaders: tuple[Path, ...] = ()

    def compiled_program_api(self) -> dict[str, Any]:
        """Describe the shared logical ABI with this Wasm member's layout."""

        inputs = {
            int(item["ssa_value_id"]): dict(item) for item in self.abi["inputs"]
        }
        outputs = {
            int(item["ssa_value_id"]): dict(item) for item in self.abi["outputs"]
        }
        parameters = []
        for kind, value_id in self.wasm_artifact.parameters:
            value_id = int(value_id)
            logical = inputs.get(value_id) or outputs.get(value_id)
            if logical is None:
                raise ValueError(f"Wasm ABI value %{value_id} has no logical name")
            dtype = str(logical["dtype"])
            integer = dtype.casefold() in {
                "bool", "i1", "int", "int32", "i32", "int64", "i64", "long",
            }
            parameters.append({
                "name": str(logical["name"]),
                "source_name": str(logical["name"]),
                "role": (
                    "inout" if value_id in inputs and value_id in outputs
                    else "output" if value_id in outputs else "input"
                ),
                "dtype": dtype,
                "c_type": "int64_t" if integer else "double",
                "ctypes": "c_int64" if integer else "c_double",
                "passing": "reference" if kind == "buffer" else "value",
                **(
                    {"shape": list(logical["shape"])}
                    if logical.get("shape") else {}
                ),
                "ssa_value_id": value_id,
            })
        return {
            "schema": "turing-compiled-program-api-v1",
            "module": self.name,
            "language": "wasm",
            "entry": self.name,
            "metadata": {
                "logical_abi_digest": self.abi_digest,
                "source_function": self.root_name,
                "execution_model": "repository-ssa-module",
            },
            "entry_points": [{
                "name": self.name,
                "symbol": self.name,
                "kind": "control",
                "parameters": parameters,
            }],
        }

    def write(
        self,
        directory: str | Path,
        *,
        compile_native: bool = True,
        emit_diagnostic_shell: bool = False,
    ) -> Path:
        destination = Path(directory).resolve()
        destination.mkdir(parents=True, exist_ok=True)
        realizations: dict[str, Any] = {}

        c_path = destination / f"{self.name}.c"
        c_path.write_text(self.c_artifact.source, encoding="utf-8", newline="\n")
        native_files = [c_path.name]
        if compile_native and self.c_artifact.complete:
            self.c_artifact.compile(destination)
            if self.c_artifact.library_path is not None:
                native_files.append(self.c_artifact.library_path.name)
        realizations["windows-native"] = {
            "kind": "native-library",
            "complete": bool(self.c_artifact.complete),
            "files": native_files,
            "shortfalls": _shortfall_records(self.c_artifact.shortfalls),
            "abi_digest": self.abi_digest,
        }

        wasm_path = destination / f"{self.name}.wasm"
        wasm_path.write_bytes(self.wasm_artifact.binary)
        realizations["webassembly"] = {
            "kind": "webassembly-module",
            "complete": bool(self.wasm_artifact.complete),
            "files": [wasm_path.name],
            "shortfalls": _shortfall_records(self.wasm_artifact.shortfalls),
            "abi_digest": self.abi_digest,
        }

        wgsl_path = destination / f"{self.name}.compute.wgsl"
        wgsl_path.write_text(
            self.webgpu_artifact.source, encoding="utf-8", newline="\n",
        )
        realizations["webgpu-compute"] = {
            "kind": "webgpu-compute-shader",
            "complete": bool(self.webgpu_artifact.complete),
            "files": [wgsl_path.name],
            "shortfalls": _shortfall_records(self.webgpu_artifact.shortfalls),
            "physical_api": (
                self.webgpu_artifact.api.to_mapping()
                if self.webgpu_artifact.api is not None else None
            ),
            "abi_digest": self.abi_digest,
        }

        graphics = []
        for source in self.graphics_shaders:
            path = Path(source).resolve()
            member = destination / path.name
            member.write_bytes(path.read_bytes())
            graphics.append({
                "path": member.name,
                "source_path": str(path),
                "sha256": hashlib.sha256(member.read_bytes()).hexdigest(),
            })
        if graphics:
            realizations["graphics-shaders"] = {
                "kind": "graphics-shader-members",
                "complete": True,
                "files": graphics,
                "abi_digest": self.abi_digest,
            }

        manifest = {
            "schema": "turing.repository-ssa-module-assembly.v1",
            "name": self.name,
            "source_function": self.root_name,
            "abi": dict(self.abi),
            "abi_digest": self.abi_digest,
            "realizations": realizations,
        }
        manifest_path = destination / f"{self.name}.module.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8",
        )
        if emit_diagnostic_shell and self.wasm_artifact.complete:
            from .wasm_html_shell import emit_html_shell

            shell = emit_html_shell(
                self.compiled_program_api(),
                wasm_bytes=self.wasm_artifact.binary,
                name=f"{self.name}_diagnostic",
                backend_sources=[
                        {
                            "language": "c", "title": "Windows native C",
                            "source": self.c_artifact.source,
                            "available": bool(self.c_artifact.complete),
                            "reason": "; ".join(
                                item["reason"] for item in _shortfall_records(
                                    self.c_artifact.shortfalls
                                )
                            ),
                        },
                        {
                            "language": "wgsl", "title": "WebGPU compute",
                            "source": self.webgpu_artifact.source,
                            "available": bool(self.webgpu_artifact.complete),
                            "reason": "; ".join(
                                item["reason"] for item in _shortfall_records(
                                    self.webgpu_artifact.shortfalls
                                )
                            ),
                        },
                ],
                build_parameters={"logical_abi_digest": self.abi_digest},
            )
            shell.write(destination)
        return manifest_path


def assemble_repository_ssa_module(
    module: IRModule,
    root_name: str,
    *,
    entry_name: str,
    graphics_shaders: Sequence[str | Path] = (),
) -> RepositorySSAModuleAssembly:
    """Emit native, Wasm, and WebGPU members from one unchanged SSA module."""

    from .ssa_c_backend import emit_ssa_to_c
    from .ssa_wasm_backend import emit_ssa_module_to_wasm_core
    from .ssa_webgpu_backend import emit_module as emit_webgpu_module

    abi = _logical_abi(module, root_name, entry_name)
    abi_digest = _digest(abi)
    returned = _returned_values(module.functions[root_name])
    c_artifact = emit_ssa_to_c(
        module, root_name, entry_name=entry_name,
    )
    wasm_artifact = emit_ssa_module_to_wasm_core(
        module, root_name, entry_name=entry_name,
    )
    webgpu_artifact = emit_webgpu_module(
        module,
        name=entry_name,
        outputs={root_name: returned},
        packed_outputs=True,
    )

    logical_inputs = tuple(item["name"] for item in abi["inputs"])
    logical_outputs = tuple(item["name"] for item in abi["outputs"])
    expected_value_order = tuple(dict.fromkeys((
        *(int(value.id) for value in module.functions[root_name].args),
        *(int(value.id) for value in returned),
    )))
    if tuple(c_artifact.buffer_order) != expected_value_order:
        raise ValueError("native realization changed the logical value ABI")
    if tuple(value_id for _kind, value_id in wasm_artifact.parameters) != (
        expected_value_order
    ):
        raise ValueError("WebAssembly realization changed the logical value ABI")
    if tuple(wasm_artifact.output_order) != tuple(
        int(value.id) for value in returned
    ):
        raise ValueError("WebAssembly realization changed the logical output ABI")
    webgpu_api = webgpu_artifact.api.to_mapping()
    webgpu_metadata = dict(webgpu_api.get("metadata") or {})
    webgpu_inputs = webgpu_metadata.get("feed_span")
    if webgpu_inputs is None:
        webgpu_inputs = [
            item.get("value_id")
            for item in webgpu_metadata.get("feed_bindings") or ()
        ]
    webgpu_outputs = webgpu_metadata.get("output_span")
    if webgpu_outputs is None:
        webgpu_outputs = [
            item.get("value_id")
            for item in webgpu_metadata.get("outputs") or ()
        ]
    if tuple(map(int, webgpu_inputs)) != tuple(
        int(value.id) for value in module.functions[root_name].args
    ):
        raise ValueError("WebGPU realization changed the logical input ABI")
    if tuple(map(int, webgpu_outputs)) != tuple(
        int(value.id) for value in returned
    ):
        raise ValueError("WebGPU realization changed the logical output ABI")

    return RepositorySSAModuleAssembly(
        entry_name,
        root_name,
        abi,
        abi_digest,
        c_artifact,
        wasm_artifact,
        webgpu_artifact,
        tuple(Path(path) for path in graphics_shaders),
    )


__all__ = [
    "RepositorySSAModuleAssembly",
    "assemble_repository_ssa_module",
]
