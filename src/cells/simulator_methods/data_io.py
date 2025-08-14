def actual_data_hook(self, payload: bytes, dst_bits: int, length_bits: int):
    self.bitbuffer._data_access[dst_bits : dst_bits + length_bits] = payload
