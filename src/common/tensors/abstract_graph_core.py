"""
AbstractGraphCore: graph-first mixin for AbstractTensor.

This defines a stable, backend-agnostic API for graph & segment operations.
Public methods call backend hooks with trailing underscores (e.g., segment_sum_),
and wrap raw backend returns into the same wrapper type as `self`, exactly like
AbstractTensor does.

Conventions
----------
- edge_index: int tensor of shape (2, E) in COO (src=0, dst=1)
- edge_weight: shape (E,) or (E, D) aligned with edges
- node features x: shape (N, F)
- segment_ids: int tensor of shape (E,) or (N,), grouping by identical ids
- All indices are 0-based.
- Methods that return multiple tensors will wrap each into the same class as `self`
  unless noted (e.g., some index arrays may be returned raw if consistent with AbstractTensor.topk).

To use: have AbstractTensor inherit from AbstractGraphCore:
    class AbstractTensor(AbstractGraphCore):
        ...

Then implement backend hooks in your backends:
- segment_sum_(self, values, segment_ids, num_segments: int | None, dim: int) -> BACKEND_DATA
- ...
"""

from __future__ import annotations
from abc import ABC
from typing import Any, Callable, Iterable, Optional, Tuple, List, Union, Literal

SegmentReduce = Literal["sum", "mean", "max", "min"]
NormKind = Literal["sym", "rw", None]
SemiringAdd = Literal["sum", "max", "min", "logsumexp"]
SemiringMul = Literal["mul", "plus"]  # "mul": multiply, "plus": a+b (for tropical/max-plus)


class AbstractGraphCore(ABC):
    # ---------- helper ----------
    def _wrap(self, backend_data: Any):
        """Wrap backend-native 'backend_data' as the same AbstractTensor type as self."""
        out = type(self)(track_time=getattr(self, "track_time", False))
        out.data = backend_data
        return out

    # ---------- tier 1: segment / scatter / gather ----------
    def segment_sum(
        self,
        values: "AbstractGraphCore",
        segment_ids: "AbstractGraphCore",
        *,
        num_segments: Optional[int] = None,
        dim: int = 0,
    ) -> "AbstractGraphCore":
        """
        Sum 'values' per segment along 'dim' as grouped by 'segment_ids'.

        values: (..., E, ...), segment_ids: (E,)
        Returns tensor with segment axis size == num_segments (if given) or max(segment_ids)+1
        """
        values = self.ensure_tensor(values)
        segment_ids = self.ensure_tensor(segment_ids)
        return self._wrap(self.segment_sum_(values.data, segment_ids.data, num_segments, dim))

    def segment_mean(
        self,
        values: "AbstractGraphCore",
        segment_ids: "AbstractGraphCore",
        *,
        num_segments: Optional[int] = None,
        dim: int = 0,
        eps: float = 1e-12,
    ) -> "AbstractGraphCore":
        """Mean per segment; safe-divide with 'eps'."""
        values = self.ensure_tensor(values)
        segment_ids = self.ensure_tensor(segment_ids)
        return self._wrap(self.segment_mean_(values.data, segment_ids.data, num_segments, dim, eps))

    def segment_max(
        self,
        values: "AbstractGraphCore",
        segment_ids: "AbstractGraphCore",
        *,
        num_segments: Optional[int] = None,
        dim: int = 0,
        initial: Optional[float] = None,
    ) -> Tuple["AbstractGraphCore", Any]:
        """
        Max per segment. Returns (values, indices_within_segment).
        Indices may be backend-native if that matches your other APIs (like topk).
        """
        values = self.ensure_tensor(values)
        segment_ids = self.ensure_tensor(segment_ids)
        v, idx = self.segment_max_(values.data, segment_ids.data, num_segments, dim, initial)
        return self._wrap(v), idx

    def segment_min(
        self,
        values: "AbstractGraphCore",
        segment_ids: "AbstractGraphCore",
        *,
        num_segments: Optional[int] = None,
        dim: int = 0,
        initial: Optional[float] = None,
    ) -> Tuple["AbstractGraphCore", Any]:
        """Min per segment. Returns (values, indices_within_segment)."""
        values = self.ensure_tensor(values)
        segment_ids = self.ensure_tensor(segment_ids)
        v, idx = self.segment_min_(values.data, segment_ids.data, num_segments, dim, initial)
        return self._wrap(v), idx

    def gather(self, index: "AbstractGraphCore", *, dim: int = 0) -> "AbstractGraphCore":
        """
        Gather elements along 'dim' using integer 'index'. Mirrors torch.gather semantics on that axis.
        """
        index = self.ensure_tensor(index)
        return self._wrap(self.gather_(index.data, dim))

    def scatter_add(
        self,
        index: "AbstractGraphCore",
        *,
        dim: int = 0,
        dim_size: Optional[int] = None,
    ) -> "AbstractGraphCore":
        """
        Scatter-ADD this tensor into a result along 'dim' at positions in 'index'.
        'self' provides the values to add.
        """
        index = self.ensure_tensor(index)
        return self._wrap(self.scatter_add_(index.data, dim, dim_size))

    def edge_softmax(
        self,
        segment_ids: "AbstractGraphCore",
        *,
        temperature: float = 1.0,
        max_stabilize: bool = True,
    ) -> "AbstractGraphCore":
        """
        Softmax this tensor per-segment defined by 'segment_ids' (common for attention over incoming edges).
        Typically called on edge scores of shape (E, 1) or (E,).
        """
        segment_ids = self.ensure_tensor(segment_ids)
        return self._wrap(self.edge_softmax_(segment_ids.data, temperature, max_stabilize))

    def segment_softmax(
        self,
        segment_ids: "AbstractGraphCore",
        *,
        temperature: float = 1.0,
        max_stabilize: bool = True,
    ) -> "AbstractGraphCore":
        """Alias of edge_softmax."""
        return self.edge_softmax(segment_ids, temperature=temperature, max_stabilize=max_stabilize)

    def topk_per_segment(
        self,
        values: "AbstractGraphCore",
        segment_ids: "AbstractGraphCore",
        k: int,
        *,
        largest: bool = True,
    ) -> Tuple["AbstractGraphCore", Any]:
        """
        Top-k per segment. Returns (topk_values, topk_indices_within_segment).
        """
        values = self.ensure_tensor(values)
        segment_ids = self.ensure_tensor(segment_ids)
        v, idx = self.topk_per_segment_(values.data, segment_ids.data, k, largest)
        return self._wrap(v), idx

    # ---------- tier 2: sparse & masked ops ----------
    def coalesce_edges(
        self,
        edge_index: "AbstractGraphCore",
        edge_weight: Optional["AbstractGraphCore"] = None,
        *,
        num_nodes: Optional[int] = None,
        op: SegmentReduce = "sum",
    ) -> Tuple["AbstractGraphCore", Optional["AbstractGraphCore"]]:
        """
        Coalesce duplicate edges by (src,dst), reducing edge_weight with 'op'.
        """
        edge_index = self.ensure_tensor(edge_index)
        ew = None if edge_weight is None else self.ensure_tensor(edge_weight)
        ei, ew2 = self.coalesce_edges_(edge_index.data, None if ew is None else ew.data, num_nodes, op)
        return self._wrap(ei), (None if ew2 is None else self._wrap(ew2))

    def edge_index_to_csr(
        self,
        edge_index: "AbstractGraphCore",
        *,
        num_nodes: Optional[int] = None,
        sorted: bool = True,
    ) -> Tuple["AbstractGraphCore", "AbstractGraphCore"]:
        """
        Convert COO edge_index -> CSR (indptr, indices) for fast matvec/matmul.
        Returns (indptr: (N+1,), indices: (E,))
        """
        edge_index = self.ensure_tensor(edge_index)
        indptr, indices = self.edge_index_to_csr_(edge_index.data, num_nodes, sorted)
        return self._wrap(indptr), self._wrap(indices)

    def csr_matvec(
        self,
        indptr: "AbstractGraphCore",
        indices: "AbstractGraphCore",
        values: Optional["AbstractGraphCore"],
        x: "AbstractGraphCore",
    ) -> "AbstractGraphCore":
        """
        Sparse matvec y = A x with A in CSR (indptr, indices, values).
        If 'values' is None, treat unweighted (all ones).
        """
        indptr = self.ensure_tensor(indptr)
        indices = self.ensure_tensor(indices)
        x = self.ensure_tensor(x)
        vdata = None if values is None else self.ensure_tensor(values).data
        return self._wrap(self.csr_matvec_(indptr.data, indices.data, vdata, x.data))

    def masked_matmul(
        self,
        other: "AbstractGraphCore",
        mask: Optional["AbstractGraphCore"] = None,
        *,
        mask_mode: Literal["zero", "neg_inf"] = "zero",
    ) -> "AbstractGraphCore":
        """
        Dense matmul with optional mask. If 'neg_inf', use additive -inf before softmax attention.
        """
        other = self.ensure_tensor(other)
        m = None if mask is None else self.ensure_tensor(mask)
        return self._wrap(self.masked_matmul_(other.data, None if m is None else m.data, mask_mode))

    def semiring_matmul(
        self,
        other: "AbstractGraphCore",
        *,
        add: SemiringAdd = "sum",
        mul: SemiringMul = "mul",
        mask: Optional["AbstractGraphCore"] = None,
    ) -> "AbstractGraphCore":
        """
        Generalized matmul over a (add, mul) semiring.
        Presets: tropical (min, plus), max-plus (max, plus), log-sum-exp (logsumexp, plus).
        """
        other = self.ensure_tensor(other)
        m = None if mask is None else self.ensure_tensor(mask)
        return self._wrap(self.semiring_matmul_(other.data, add, mul, None if m is None else m.data))

    # ---------- tier 3: algebraic graph ops ----------
    def add_self_loops(
        self,
        edge_index: "AbstractGraphCore",
        edge_weight: Optional["AbstractGraphCore"] = None,
        *,
        fill_value: float = 1.0,
        num_nodes: Optional[int] = None,
    ) -> Tuple["AbstractGraphCore", Optional["AbstractGraphCore"]]:
        """Add self-loops (i,i) with weight fill_value for missing nodes."""
        edge_index = self.ensure_tensor(edge_index)
        ew = None if edge_weight is None else self.ensure_tensor(edge_weight)
        ei, ew2 = self.add_self_loops_(edge_index.data, None if ew is None else ew.data, fill_value, num_nodes)
        return self._wrap(ei), (None if ew2 is None else self._wrap(ew2))

    def remove_self_loops(
        self,
        edge_index: "AbstractGraphCore",
        edge_weight: Optional["AbstractGraphCore"] = None,
    ) -> Tuple["AbstractGraphCore", Optional["AbstractGraphCore"]]:
        """Remove edges (i,i)."""
        edge_index = self.ensure_tensor(edge_index)
        ew = None if edge_weight is None else self.ensure_tensor(edge_weight)
        ei, ew2 = self.remove_self_loops_(edge_index.data, None if ew is None else ew.data)
        return self._wrap(ei), (None if ew2 is None else self._wrap(ew2))

    def laplacian(
        self,
        edge_index: "AbstractGraphCore",
        edge_weight: Optional["AbstractGraphCore"] = None,
        *,
        num_nodes: Optional[int] = None,
        norm: NormKind = "sym",
        add_self_loops_if_absent: bool = True,
    ) -> Tuple["AbstractGraphCore", Optional["AbstractGraphCore"]]:
        """
        Return Laplacian in COO: (edge_index_L, edge_weight_L).
        norm: "sym" (I - D^{-1/2} A D^{-1/2}), "rw" (I - D^{-1} A), or None (D - A).
        """
        edge_index = self.ensure_tensor(edge_index)
        ew = None if edge_weight is None else self.ensure_tensor(edge_weight)
        ei, ew2 = self.laplacian_(edge_index.data, None if ew is None else ew.data, num_nodes, norm, add_self_loops_if_absent)
        return self._wrap(ei), (None if ew2 is None else self._wrap(ew2))

    def chebyshev_filter(
        self,
        x: "AbstractGraphCore",
        L: Tuple["AbstractGraphCore","AbstractGraphCore"] | Any,
        K: int,
        *,
        lambda_max: Optional[float] = None,
    ) -> "AbstractGraphCore":
        """
        Apply Chebyshev polynomial filter sum_{k=0}^{K} theta_k T_k(tilde{L}) x.
        L can be (edge_index_L, edge_weight_L) in COO or a backend-specific Laplacian handle.
        """
        x = self.ensure_tensor(x)
        if isinstance(L, tuple):
            L = (self.ensure_tensor(L[0]).data, None if L[1] is None else self.ensure_tensor(L[1]).data)
        return self._wrap(self.chebyshev_filter_(x.data, L, K, lambda_max))

    def random_walk(
        self,
        edge_index: "AbstractGraphCore",
        edge_weight: Optional["AbstractGraphCore"] = None,
        *,
        start: Optional["AbstractGraphCore"] = None,
        steps: int = 1,
        num_nodes: Optional[int] = None,
        return_distribution: bool = True,
    ) -> "AbstractGraphCore":
        """
        Discrete-time random walk. If 'start' is None, return the transition matrix^steps or stationary distribution
        depending on backend capability. Otherwise simulate from start nodes.
        """
        edge_index = self.ensure_tensor(edge_index)
        ew = None if edge_weight is None else self.ensure_tensor(edge_weight)
        st = None if start is None else self.ensure_tensor(start)
        return self._wrap(self.random_walk_(edge_index.data, None if ew is None else ew.data, None if st is None else st.data, steps, num_nodes, return_distribution))

    # ---------- tier 5: ragged batching & utilities ----------
    def pack_segments(self, lengths: "AbstractGraphCore") -> "AbstractGraphCore":
        """
        Return prefix-sum offsets given lengths (e.g., node-per-graph counts).
        """
        lengths = self.ensure_tensor(lengths)
        return self._wrap(self.pack_segments_(lengths.data))

    def unpack_segments(self, offsets: "AbstractGraphCore") -> "AbstractGraphCore":
        """
        Invert pack_segments for lengths (diff of consecutive offsets).
        """
        offsets = self.ensure_tensor(offsets)
        return self._wrap(self.unpack_segments_(offsets.data))

    def block_diag(self, blocks: List["AbstractGraphCore"]) -> "AbstractGraphCore":
        """
        Build a block-diagonal matrix from a list of dense/sparse adjacency blocks.
        For sparse COO inputs, return sparse COO.
        """
        blocks = [self.ensure_tensor(b) for b in blocks]
        return self._wrap(self.block_diag_([b.data for b in blocks]))

    def permute(self, perm: "AbstractGraphCore", *, dim: int = 0) -> "AbstractGraphCore":
        """Permute along 'dim' according to integer permutation 'perm'."""
        perm = self.ensure_tensor(perm)
        return self._wrap(self.permute_(perm.data, dim))

    def reindex_nodes(
        self,
        edge_index: "AbstractGraphCore",
        perm: "AbstractGraphCore",
    ) -> "AbstractGraphCore":
        """Apply permutation 'perm' to node ids in COO edge_index."""
        edge_index = self.ensure_tensor(edge_index)
        perm = self.ensure_tensor(perm)
        return self._wrap(self.reindex_nodes_(edge_index.data, perm.data))

    # ---------- backend hooks (to be implemented per backend) ----------
    # Segment ops
    def segment_sum_(self, values, segment_ids, num_segments, dim): raise NotImplementedError
    def segment_mean_(self, values, segment_ids, num_segments, dim, eps): raise NotImplementedError
    def segment_max_(self, values, segment_ids, num_segments, dim, initial): raise NotImplementedError
    def segment_min_(self, values, segment_ids, num_segments, dim, initial): raise NotImplementedError
    def gather_(self, index, dim): raise NotImplementedError
    def scatter_add_(self, index, dim, dim_size): raise NotImplementedError
    def edge_softmax_(self, segment_ids, temperature, max_stabilize): raise NotImplementedError
    def topk_per_segment_(self, values, segment_ids, k, largest): raise NotImplementedError

    # Sparse / masked
    def coalesce_edges_(self, edge_index, edge_weight, num_nodes, op): raise NotImplementedError
    def edge_index_to_csr_(self, edge_index, num_nodes, sorted): raise NotImplementedError
    def csr_matvec_(self, indptr, indices, values, x): raise NotImplementedError
    def masked_matmul_(self, other, mask, mask_mode): raise NotImplementedError
    def semiring_matmul_(self, other, add, mul, mask): raise NotImplementedError

    # Algebraic ops
    def add_self_loops_(self, edge_index, edge_weight, fill_value, num_nodes): raise NotImplementedError
    def remove_self_loops_(self, edge_index, edge_weight): raise NotImplementedError
    def laplacian_(self, edge_index, edge_weight, num_nodes, norm, add_self_loops_if_absent): raise NotImplementedError
    def chebyshev_filter_(self, x, L, K, lambda_max): raise NotImplementedError
    def random_walk_(self, edge_index, edge_weight, start, steps, num_nodes, return_distribution): raise NotImplementedError

    # Ragged / utils
    def pack_segments_(self, lengths): raise NotImplementedError
    def unpack_segments_(self, offsets): raise NotImplementedError
    def block_diag_(self, blocks): raise NotImplementedError
    def permute_(self, perm, dim): raise NotImplementedError
    def reindex_nodes_(self, edge_index, perm): raise NotImplementedError
