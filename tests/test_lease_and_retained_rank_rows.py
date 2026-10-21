"""A recorded rank of 0 is a recorded rank when the linker copies physical rows.

Program: ``fresh`` (a callee returning a record built from a formal) and
``keep`` (a callee returning its record formal).  Wrong record: the frame
physical reader dropped a callee row's physical fields with
``value not in (None, False)``; ``0 == False``, so a scalar's recorded
``rank=0`` read as "records none" and the result-storage lease and retained
result were posted with storage but no rank.  Rule: ``rank`` is recorded when
it is not None; ``arena`` alone is a flag recorded when True.
"""
from __future__ import annotations

from src.common.dt_system.dt_scaler import Metrics
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable
import tests.test_aggregate_call_identity as T

FRESH = """
def copy(metrics):
    return Metrics(metrics.max_vel + 1.0, 0.0, 0.0, 0.0)

def tick(metrics):
    other = copy(metrics)
    return other.max_vel + metrics.max_vel
"""

KEEP = """
def keep(metrics):
    metrics.max_vel = metrics.max_vel + 1.0
    return metrics

def tick(metrics):
    same = keep(metrics)
    return same.max_vel + metrics.max_vel
"""


def _module(source, name):
    return lower_ast_source_to_ssa(
        source, "tick", name=name, python_bindings={"Metrics": Metrics},
        extraction_contract=ExtractionContract(
            T.CONTRACTS / "program_extraction.yaml"
        ),
        runtime_closure_only=True,
    )[0]


def _quiet(module):
    kinds = {"ssa-physical-unposted", "ssa-physical-discordant"}
    found = [
        f for f in CorrelationTable.build(module).findings(module)
        if f.kind in kinds or (
            f.kind == "ssa-publication-type-disagreement"
            and "accounting" in f.detail
        )
    ]
    assert not found, [(f.kind, f.detail[:200]) for f in found]


def test_fresh_result_storage_lease_posts_rank_zero():
    _quiet(_module(FRESH, "p_lease_rank_fresh"))


def test_keep_retained_result_posts_rank_zero():
    _quiet(_module(KEEP, "p_lease_rank_keep"))

