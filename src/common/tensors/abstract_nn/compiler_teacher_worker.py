"""Execute registered compiler teachers into the training corpus."""
from __future__ import annotations

import hashlib
import inspect
import time
from typing import Any, Callable, Mapping

from .token_encoder import encode_identity_tokens
from .token_lexicon import structural_context_tokens
from .training_data_store import (
    CompilerCommandRequest,
    CompilerTrainingDatabase,
    _training_value,
)


Teacher = Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]]


def _ssa_payload(program: Mapping[str, Any], arguments: Mapping[str, Any]) -> dict:
    """Exact source -> repository-SSA teacher for the first dense route."""

    from ....compiler.fortran_c_shell import lower_ast_source_to_ssa

    entry = str(arguments.get("entry") or program["entrypoint"])
    module, outputs, exports = lower_ast_source_to_ssa(
        str(program["source_text"]), entry,
        name=f"teacher_{program['source_sha256'][:12]}",
    )
    functions = []
    for name, function in sorted(module.functions.items()):
        functions.append({
            "name": name,
            "arguments": [
                {"id": int(value.id), "dtype": str(value.dtype)}
                for value in function.args
            ],
            "blocks": [
                {
                    "label": label,
                    "instructions": [
                        {
                            "op": str(item.op),
                            "arguments": [int(value.id) for value in item.args],
                            "result": (
                                None if item.res is None else int(item.res.id)
                            ),
                            "attributes": _training_value(
                                dict(item.attributes or {})
                            ),
                        }
                        for item in block.instrs
                    ],
                }
                for label, block in function.blocks.items()
            ],
            "metadata": _training_value(dict(function.metadata or {})),
        })
    return {
        "schema": "turing-training-repository-ssa-v1",
        "entrypoint": entry,
        "functions": functions,
        "outputs": {
            str(name): [int(value.id) for value in values]
            for name, values in sorted(outputs.items())
        },
        "exports": list(map(str, exports or ())),
        "module_metadata": _training_value(dict(module.metadata or {})),
    }


DEFAULT_TEACHERS: Mapping[str, Teacher] = {
    "lower_repository_ssa": _ssa_payload,
}


class CompilerTeacherWorker:
    """Claim, execute, verify, link, and finish compiler commands."""

    def __init__(
        self, database: CompilerTrainingDatabase,
        teachers: Mapping[str, Teacher] | None = None,
    ) -> None:
        self.database = database
        self.teachers = dict(teachers or DEFAULT_TEACHERS)

    def run(self, request: CompilerCommandRequest) -> Any:
        claimed = self.database.claim_command(request.id)
        started = time.perf_counter()
        try:
            teacher = self.teachers[claimed.command_name]
            source_view = self.database.latest_view(
                claimed.program_id, claimed.source_form,
            )
            program = self.database.program_record(claimed.program_id)
            payload = dict(teacher(program, claimed.arguments))
            tokens = structural_context_tokens(payload)
            target = self.database.put_view(
                claimed.program_id, claimed.target_form, payload, tokens,
                token_ids=tuple(
                    encode_identity_tokens({"token": token}) for token in tokens
                ),
                context={
                    "teacher": claimed.command_name,
                    "source_sha256": program["source_sha256"],
                },
                generator=f"compiler_teacher:{claimed.command_name}",
                generator_version=hashlib.sha256(
                    inspect.getsource(teacher).encode("utf-8")
                ).hexdigest()[:16],
            )
            self.database.link_views(
                claimed.program_id, source_view.id, target.id,
                claimed.command_name,
                weight_key=f"{claimed.source_form}->{claimed.target_form}@compiler",
                compiler_command={
                    "command": claimed.command_name,
                    "arguments": dict(claimed.arguments),
                },
                metadata={
                    "verified": True,
                    "elapsed_seconds": time.perf_counter() - started,
                    "source_sha256": program["source_sha256"],
                },
            )
            self.database.complete_command(claimed.id, target.id)
            return target
        except Exception as error:
            self.database.fail_command(claimed.id, {
                "type": type(error).__name__, "message": str(error),
                "elapsed_seconds": time.perf_counter() - started,
            })
            raise

    def run_pending(self, *, limit: int | None = None) -> tuple[Any, ...]:
        pending = tuple(
            request for request in self.database.pending_commands()
            if request.command_name in self.teachers
        )
        if limit is not None:
            pending = pending[:max(0, int(limit))]
        return tuple(self.run(request) for request in pending)


__all__ = ["CompilerTeacherWorker", "DEFAULT_TEACHERS"]
