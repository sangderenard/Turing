def root(values, indices):
    return AbstractTensor.scatter(
        AbstractTensor.zeros((3,), dtype="float64"),
        indices,
        values,
        dim=0,
        reduce="sum",
    )
