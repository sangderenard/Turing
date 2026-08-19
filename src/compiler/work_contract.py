"""The work contract: what a compilation is for, stated once, as presets.

The pipeline has accumulated independent switches that all answer the same
question -- "how faithful must this artifact be, and to whom?" -- from
different corners:

* the identity policy (``ir_identities``: exact-only vs the bit-changing set),
* multiply-add contraction (``ssa_llvm_backend``: ``contract`` + host target),
* emission register reuse (the slot-keyed cache: evaporate redundant loads),
* the fusion level (``fusion_levels``: REGIONS vs FUSED honored today),
* the diagnostic channels (``watch``/``history``/``text_sink``: values that
  must stay observable in real storage).

Any one of these chosen alone is a local decision; together they are a
CONTRACT for the work, and the combinations that make sense are few. This
module names them. A preset is a complete, internally consistent answer, so a
caller states intent ("prove", "develop", "deploy", "fast") instead of
recalling which five switches cooperate.

The presets
-----------

``prove``    Conservative equality-proving form. Every value lives in and is
             re-read from its pool slot; no register reuse, no identity that
             changes bits, no contraction. This is the shape you diff two
             backends over, value by value.
``develop``  The default. In-place pool composition with same-block register
             reuse and the EXACT identity set only -- bit-identical to
             ``prove`` by construction, measured 6x faster on the fluid
             flagship. Diagnostics fully available (stores never evaporate).
``deploy``   ``develop`` plus the inexact identity set (sqrt family). Changes
             bits within documented bounds (the fluid's mass_err <= 1e-15
             gate held); still no contraction, so results are stable across
             hosts.
``fast``     Everything: inexact identities, multiply-add contraction, host
             target named. Bit-stability across machines is explicitly
             surrendered (fma availability differs by CPU). ~10x measured.

Diagnostics are compatible with every preset TODAY because the register
cache only evaporates loads -- every store still lands in its slot, so
``watch``/``history`` read truthful storage under all four. Any future mode
that evaporates STORES must consult the watch set here first; that is a
contract change, not a flag.

Resolution order: an explicit ``set_active_contract`` wins; else the
``TURING_WORK_CONTRACT`` environment variable names a preset; else
``develop``. The two legacy variables ``TURING_POW_INEXACT`` and
``TURING_FMA_CONTRACT`` remain honored as single-field overrides on top of
the resolved preset, so every measurement recorded against them still means
what it meant.
"""
from __future__ import annotations

import dataclasses
import os


@dataclasses.dataclass(frozen=True)
class WorkContract:
    """One complete answer to "how faithful, and to whom?"."""

    name: str
    # Emission keeps a same-block register for a slot's known content.
    register_reuse: bool
    # ir_identities may fire the bit-changing reductions (sqrt family).
    inexact_identities: bool
    # Multiply-add contraction: `contract` flags + host target named.
    contract_multiply_add: bool

    def describe(self) -> str:
        held = [
            f"register_reuse={'on' if self.register_reuse else 'off'}",
            f"identities={'inexact' if self.inexact_identities else 'exact-only'}",
            f"fma={'contract' if self.contract_multiply_add else 'none'}",
        ]
        return f"{self.name}: " + ", ".join(held)


PRESETS: dict[str, WorkContract] = {
    "prove": WorkContract(
        "prove", register_reuse=False, inexact_identities=False,
        contract_multiply_add=False,
    ),
    "develop": WorkContract(
        "develop", register_reuse=True, inexact_identities=False,
        contract_multiply_add=False,
    ),
    "deploy": WorkContract(
        "deploy", register_reuse=True, inexact_identities=True,
        contract_multiply_add=False,
    ),
    "fast": WorkContract(
        "fast", register_reuse=True, inexact_identities=True,
        contract_multiply_add=True,
    ),
}

_active: WorkContract | None = None


def set_active_contract(contract: WorkContract | str | None) -> None:
    """Pin the contract for this process; ``None`` returns to resolution."""

    global _active
    if isinstance(contract, str):
        contract = _named(contract)
    _active = contract


def _named(name: str) -> WorkContract:
    preset = PRESETS.get(str(name).strip().lower())
    if preset is None:
        # Refuse, never fall back: a caller who asked for a contract and
        # silently got another is the failure shape this module exists to
        # prevent (same doctrine as fusion_levels).
        raise ValueError(
            f"unknown work contract {name!r}; presets: {sorted(PRESETS)}"
        )
    return preset


def _flag(variable: str) -> bool | None:
    raw = os.environ.get(variable)
    if raw is None or raw == "":
        return None
    return raw not in ("0",)


def active_contract() -> WorkContract:
    """The contract in force: pinned, else named by environment, else develop.

    Legacy single-field overrides (``TURING_POW_INEXACT``,
    ``TURING_FMA_CONTRACT``) apply on top, so a measurement script that sets
    only one of them gets exactly the historical meaning.
    """

    contract = _active
    if contract is None:
        named = os.environ.get("TURING_WORK_CONTRACT")
        contract = _named(named) if named else PRESETS["develop"]

    inexact = _flag("TURING_POW_INEXACT")
    fma = _flag("TURING_FMA_CONTRACT")
    if inexact is None and fma is None:
        return contract
    return dataclasses.replace(
        contract,
        name=f"{contract.name}+overrides",
        inexact_identities=(
            contract.inexact_identities if inexact is None else inexact
        ),
        contract_multiply_add=(
            contract.contract_multiply_add if fma is None else fma
        ),
    )
