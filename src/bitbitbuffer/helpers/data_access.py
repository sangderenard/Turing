class BitBitBufferDataAccess:
    def __init__(self, buffer, caster=int):
        self.buffer = buffer
        self.caster = caster

    @property
    def plane(self):
        # tell the indexer to target the data plane
        return 'data'

    def __getitem__(self, index):
        # data-plane get via index object
        from .bitbitindex import BitBitIndex
        spec = BitBitIndex(self, index, mode='get')
        return self.buffer.indexer.access(spec)

    def __setitem__(self, index, value):
        # data-plane set via index object
        from .bitbitindex import BitBitIndex
        spec = BitBitIndex(self, index, mode='set', value=value)
        return self.buffer.indexer.access(spec)
