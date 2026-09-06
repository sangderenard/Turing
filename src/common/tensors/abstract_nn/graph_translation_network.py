"""Programmatic weight matrix and routing network for compiler graph views.

Each directed ``source_form -> target_form`` cell owns an independent weight
set.  The data/command architecture is operational; learned attention cells
remain explicitly unavailable until AbstractNN's transformer is implemented.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

from .training_data_store import (
    CompilerCommandRequest,
    CompilerTrainingDatabase,
)


class TransformerUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class TransformerCellSpec:
    source_form: str
    target_form: str
    vocabulary_size: int
    model_width: int = 256
    attention_heads: int = 8
    encoder_layers: int = 4
    decoder_layers: int = 4
    feed_forward_width: int = 1024
    maximum_tokens: int = 4096
    revision: int = 0

    @property
    def weight_key(self) -> str:
        return f"{self.source_form}->{self.target_form}@{self.revision}"


@dataclass(frozen=True)
class CompilerTransformationRoute:
    source_form: str
    target_form: str
    command_name: str
    static_arguments: Mapping[str, Any] | None = None


class TransformationCell(Protocol):
    spec: TransformerCellSpec

    def predict(self, token_ids: Sequence[int]) -> tuple[int, ...]: ...


@dataclass
class UnfinishedTransformerCell:
    """Honest placeholder for the not-yet-implemented attention network."""

    spec: TransformerCellSpec

    def predict(self, token_ids: Sequence[int]) -> tuple[int, ...]:
        del token_ids
        raise TransformerUnavailableError(
            f"transformer cell {self.spec.weight_key} has no implemented "
            "attention network or trained checkpoint"
        )


class TransformationWeightMatrix:
    """Directed matrix of independently trainable transformation cells."""

    def __init__(self, forms: Sequence[str]) -> None:
        self.forms = tuple(dict.fromkeys(map(str, forms)))
        self._cells: dict[tuple[str, str], TransformationCell] = {}

    def install(self, cell: TransformationCell) -> None:
        key = (cell.spec.source_form, cell.spec.target_form)
        if key[0] not in self.forms or key[1] not in self.forms:
            raise KeyError(f"weight cell outside declared matrix: {key!r}")
        self._cells[key] = cell

    def cell(self, source_form: str, target_form: str) -> TransformationCell:
        try:
            return self._cells[(str(source_form), str(target_form))]
        except KeyError as exc:
            raise KeyError(
                f"no transformation weight cell {source_form!r} -> {target_form!r}"
            ) from exc

    def populate_stubs(
        self,
        vocabulary_size: int,
        *,
        spec_factory: Callable[[str, str, int], TransformerCellSpec] | None = None,
    ) -> None:
        factory = spec_factory or (
            lambda source, target, size: TransformerCellSpec(source, target, size)
        )
        for source in self.forms:
            for target in self.forms:
                if source == target:
                    continue
                self.install(UnfinishedTransformerCell(
                    factory(source, target, int(vocabulary_size))
                ))

    def persist(self, database: CompilerTrainingDatabase) -> None:
        for cell in self._cells.values():
            database.put_weight_set(
                cell.spec.weight_key,
                cell.spec.source_form,
                cell.spec.target_form,
                asdict(cell.spec),
                revision=cell.spec.revision,
                status=(
                    "stub" if isinstance(cell, UnfinishedTransformerCell)
                    else "ready"
                ),
            )


class GraphTranslationNetwork:
    """Route known views through weights or request exact compiler teachers.

    A missing view is never hallucinated by an untrained cell.  The network
    records a structured compiler command, which an authorized compiler worker
    can fulfill and link back into the corpus as dense permutation data.
    """

    def __init__(
        self,
        database: CompilerTrainingDatabase,
        weights: TransformationWeightMatrix,
        routes: Sequence[CompilerTransformationRoute],
    ) -> None:
        self.database = database
        self.weights = weights
        self.routes = {
            (route.source_form, route.target_form): route for route in routes
        }

    def request_translation(
        self,
        program_id: int,
        source_form: str,
        target_form: str,
        *,
        arguments: Mapping[str, Any] | None = None,
    ) -> CompilerCommandRequest:
        route = self.routes.get((str(source_form), str(target_form)))
        if route is None:
            raise KeyError(
                f"no compiler teacher route {source_form!r} -> {target_form!r}"
            )
        merged = {
            **dict(route.static_arguments or {}),
            **dict(arguments or {}),
        }
        return self.database.request_compiler_view(
            int(program_id), route.source_form, route.target_form,
            route.command_name, merged,
        )

    def densify(
        self,
        program_id: int,
        *,
        arguments: Mapping[str, Any] | None = None,
    ) -> tuple[CompilerCommandRequest, ...]:
        available = set(self.database.forms(program_id))
        pending = {
            (request.source_form, request.target_form)
            for request in self.database.pending_commands()
            if request.program_id == int(program_id)
        }
        requests: list[CompilerCommandRequest] = []
        # Iterate to a fixed point over routes. A queued target becomes
        # available only after a compiler worker completes it, so this pass
        # requests every currently reachable missing permutation exactly once.
        for (source, target), _route in sorted(self.routes.items()):
            if (
                source in available
                and target not in available
                and (source, target) not in pending
            ):
                requests.append(self.request_translation(
                    program_id, source, target, arguments=arguments,
                ))
        return tuple(requests)

    def predict(
        self,
        source_form: str,
        target_form: str,
        token_ids: Sequence[int],
    ) -> tuple[int, ...]:
        return self.weights.cell(source_form, target_form).predict(token_ids)


__all__ = [
    "CompilerTransformationRoute", "GraphTranslationNetwork",
    "TransformationCell", "TransformationWeightMatrix",
    "TransformerCellSpec", "TransformerUnavailableError",
    "UnfinishedTransformerCell",
]
