from typing import TYPE_CHECKING, List
if TYPE_CHECKING:
    import numpy as np
    _Vec = "np.ndarray"
else:
    try:
        import numpy as np
        _Vec = np.ndarray
    except ModuleNotFoundError:
        _Vec = List[float]
