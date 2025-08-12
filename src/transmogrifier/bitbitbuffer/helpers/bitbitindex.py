from typing import Any, Optional, Callable, Union, List
from .utils import depth_guarded_repr

class BitBitIndex:
    """
    A simple metadata struct holding an indexing request:
      - caller: the object being indexed (BitBitBuffer, BitBitItem, BitBitSlice)
      - index: raw Python index (int or slice)
      - mode: 'get' or 'set'
      - value: payload for writes
    """
    def __init__(self, caller: Any, index: Union[int, slice], mode: str = 'get', value: Any = None, inverted: bool = False, index_hook: Optional[Callable] = None):
        self.caller = caller
        self.index = index
        self.mode = mode
        self.value = value
        self.inverted = inverted
        self.index_hook = index_hook
        self.empty = False
        if isinstance(index, slice):
            if index.start is not None and index.stop is not None and index.start == index.stop:
                self.empty = True

    def __repr__(self):
        #print(self.empty)
        return depth_guarded_repr(self)
        

    def normalize(self) -> tuple[int,int,int]:
        """
        Normalize raw index into (start, stop, step) following native Python slice rules.
        """

        # Slice / Item views use their *local* length, not the parent buffer size
        from .bitbitslice import BitBitSlice
        from .bitbititem import BitBitItem

        if isinstance(self.caller, (BitBitSlice, BitBitItem)):
            mask_size = len(self.caller)
        else:
            buf = self.caller.buffer if hasattr(self.caller, 'buffer') else self.caller
            mask_size = buf.mask_size

        if isinstance(self.index, int):
            start, stop, step = self.index, self.index + 1, 1
        else:
            sl = self.index
            step = sl.step or 1
            # default start/stop based on step sign
            if step > 0:
                start = sl.start if sl.start is not None else 0
                stop  = sl.stop  if sl.stop  is not None else mask_size
            else:
                start = sl.start if sl.start is not None else mask_size - 1
                stop  = sl.stop  if sl.stop  is not None else -1
        assert start >= 0, "Index out of range"
        assert stop <= mask_size, "Index out of range"
        assert step != 0, "Slice step cannot be zero"
        if stop == start:
            self.empty = True
        return start, stop, step

    def indices(self) -> List[int]:
        """
        Compute the flat list of mask-bit positions to access.
        """
        start, stop, step = self.normalize()
        return list(range(start, stop, step))

    @property
    def plane(self) -> str:
        if hasattr(self.caller, 'plane'):
            return self.caller.plane
        return 'mask'


    @property
    def base_offset(self) -> int:
        """Bit offset of this view in the global buffer."""
        return getattr(self.caller, 'mask_index', 0)

    @property
    def bitsforbits(self) -> int:
        return self.caller.buffer.bitsforbits if hasattr(self.caller, 'buffer') else self.caller.bitsforbits

    @property
    def caster(self):
        return getattr(self.caller, 'cast', int)
