import threading
_repr_local = threading.local()

def depth_guarded_repr(obj, fallback="<...>"):
    depth = getattr(_repr_local, 'depth', 0)
    if depth > 1:
        return fallback
    try:
        _repr_local.depth = depth + 1
        return obj.__repr__()
    finally:
        _repr_local.depth = depth
