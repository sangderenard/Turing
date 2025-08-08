from .bitbitindex import BitBitIndex

class BitBitItem:
    def __init__(self, buffer=None, mask_index=None, length=None, cast=None, padding=0, padding_mask=None, reversed=False, plane='mask'):
        self._plane = plane
        self.id = id(self)
        self.buffer = buffer
        self.mask_index = mask_index
        self.padding = padding
        self.padding_mask = padding_mask or 0
        self.reversed = reversed
        self.padded_length = length
        self.useful_length = length - padding
        self.cast = cast or int
        if self.mask_index is not None:
            self.data_index = self.buffer.bitsforbits * self.mask_index
        else:
            self.data_index = None
    def __len__(self):
        return self.useful_length

    @property
    def data_or_mask(self):
        """Default plane when indexing with slice or int."""
        return self._plane


    def __bytes__(self):
        # build index-object for full‐slice get
        spec = BitBitIndex(self, slice(None), mode='get')
        return self.buffer.indexer.access(spec)

    def __int__(self):
        # build index-object for single-bit get
        spec = BitBitIndex(self, 0, mode='get')
        raw = self.buffer.indexer.access(spec)
        return (raw[0] >> 7) & 1

    def __getitem__(self, key):
            # Allow default plane indexing with int or slice
            second_key = None
            if isinstance(key, (slice, int)):
                second_key = key
                key = self.data_or_mask
            # Determine index for spec
            idx = second_key if second_key is not None else (self.mask_index if key == 'data' else 0)
            mode = 'view' if isinstance(idx, slice) else 'get'

            if key == 'mask':
                spec = BitBitIndex(self, idx, mode=mode)
                result = self.buffer.indexer.access(spec)

                # If the mode was 'get' (for an integer index), process and return the bit value.
                if mode == 'get':
                    return int(result[0] >> 7) if isinstance(result, (bytes, bytearray)) else int(result)

                # Otherwise (for 'view' mode), return the new slice object directly.
                return result

            if key == 'data':
                spec = BitBitIndex(self.buffer._data_access, idx, mode=mode)
                return self.buffer.indexer.access(spec)

            raise KeyError("Expected 'mask' or 'data'")


    def __setitem__(self, key, value):
        # Allow default plane setting with int or slice
        second_key = None
        if isinstance(key, (slice, int)):
            second_key = key
            key = self.data_or_mask
        # Determine index for spec
        idx = second_key if second_key is not None else (self.mask_index if key == 'data' else 0)

        # FIX: Explicitly set the mode based on the index type.
        mode = 'view' if isinstance(idx, slice) else 'get' # This line isn't strictly necessary for setitem,
                                                             # but the spec requires a mode. We'll use 'set'.

        if key == 'mask':
            # Always use 'set' mode for __setitem__
            spec = BitBitIndex(self, idx, mode='set', value=value)
            return self.buffer.indexer.access(spec)
        if key == 'data':
            # Always use 'set' mode for __setitem__
            spec = BitBitIndex(self.buffer._data_access, idx, mode='set', value=value)
            return self.buffer.indexer.access(spec)

        raise KeyError("Expected 'mask' or 'data'")

    def hex(self):
        spec = BitBitIndex(self, slice(None), mode='hex')
        return self.buffer.indexer.access(spec)

    def data_hex(self):
        spec = BitBitIndex(self.buffer._data_access, self.mask_index, mode='data_hex')
        return self.buffer.indexer.access(spec)

    def __repr__(self):
        spec = BitBitIndex(self, slice(None), mode='repr')
        return self.buffer.indexer.access(spec)

    @property
    def plane(self):
        """Which plane are we targeting?"""
        return self._plane

    def __iter__(self):
        """Yield this view's mask bits (0/1) in local order."""
        for i in range(self.useful_length):
            # __getitem__(i) already returns 0/1 for mask-plane 'get'
            yield int(self[i])

    def __reversed__(self):
        """Iterate bits in reverse local order (useful for reversed views)."""
        for i in range(self.useful_length - 1, -1, -1):
            yield int(self[i])
