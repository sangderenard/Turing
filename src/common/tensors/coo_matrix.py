from src.common.tensors.abstraction import AbstractTensor

class COOMatrix:
    """
    Minimal backend-agnostic COO matrix for graph core.
    Stores edge_index (2, E), edge_weight (E,), and shape (N, N).
    Uses AbstractTensor for all storage and access.
    Changes to indices/values propagate to the network if attached.
    """
    def __init__(self, edge_index, edge_weight, shape, network=None):
        self.edge_index = AbstractTensor.get_tensor(edge_index)
        self.edge_weight = AbstractTensor.get_tensor(edge_weight)
        self.shape = shape
        self.network = network  # Optional: link to AbstractGraphCore for propagation

    @property
    def indices(self):
        return self.edge_index

    @property
    def values(self):
        return self.edge_weight

    @property
    def row(self):
        return self.edge_index[0]

    @property
    def col(self):
        return self.edge_index[1]

    @property
    def data(self):
        return self.edge_weight

    def to_dense(self):
        N = self.shape[0]
        dense = AbstractTensor.zeros((N, N))
        idx = self.edge_index
        vals = self.edge_weight
        for i in range(idx.shape[1]):
            dense[idx[0, i], idx[1, i]] = vals[i]
        return dense

    def toarray(self):
        dense = self.to_dense()
        return getattr(dense, "data", dense)

    def diagonal(self):
        N = self.shape[0]
        diag = AbstractTensor.zeros(N)
        idx = self.edge_index
        vals = self.edge_weight
        for i in range(idx.shape[1]):
            r = int(idx[0, i])
            c = int(idx[1, i])
            if r == c:
                diag[r] = vals[i]
        return diag

    def update(self, edge_index=None, edge_weight=None):
        if edge_index is not None:
            self.edge_index = AbstractTensor.get_tensor(edge_index)
        if edge_weight is not None:
            self.edge_weight = AbstractTensor.get_tensor(edge_weight)
        if self.network is not None:
            self.network.on_coo_update(self)

    def __repr__(self):
        return f"COOMatrix(shape={self.shape}, nnz={self.edge_weight.shape[0]})"
