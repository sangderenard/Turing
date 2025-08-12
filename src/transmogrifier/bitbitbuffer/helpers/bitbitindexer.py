import logging
from typing import Union, List, Any
from .bitbitindex import BitBitIndex
from .bitbitslice import BitBitSlice
from .bitbititem import BitBitItem

class BitBitIndexer:
    """
    Centralized entry point for all BitBitBuffer get/set accesses.
    """
    # toggleable logging config
    logging_enabled = False
    verbosity = 0

    @staticmethod
    def configure(enabled: bool = True, verbosity: int = 1):
        """
        Turn on/off detailed logging and set verbosity (1=basic, 2=normalize, 3=bit-level, …).
        """
        BitBitIndexer.logging_enabled = enabled
        BitBitIndexer.verbosity = verbosity
        level = logging.DEBUG if enabled and verbosity > 0 else logging.INFO
        logging.getLogger().setLevel(level)
        logging.debug(f"[configure] enabled={enabled}, verbosity={verbosity}")

    @staticmethod
    def _invert_bits(raw: bytes, bit_length: int) -> bytes:
        """
        Given a bytestring `raw` containing at least `bit_length` bits
        in big‑endian bit order, invert the first `bit_length` bits
        and return the resulting bytes.
        """
        # Extract the first bit_length bits into a list
        bits = [(raw[i//8] >> (7 - (i % 8))) & 1 for i in range(bit_length)]
        # Invert that list
        bits = [1 - b for b in bits]
        # Pack back into bytes
        out = bytearray((bit_length + 7) // 8)
        for pos, b in enumerate(bits):
            if b:
                out[pos // 8] |= 1 << (7 - (pos % 8))
        return bytes(out)

    @staticmethod
    def _reverse_bits(raw: bytes, bit_length: int) -> bytes:
        """
        Given a bytestring `raw` containing at least `bit_length` bits
        in big‑endian bit order, reverse the first `bit_length` bits
        and return the resulting bytes.
        """
        # Extract the first bit_length bits into a list
        bits = [(raw[i//8] >> (7 - (i % 8))) & 1 for i in range(bit_length)]
        # Reverse that list
        bits.reverse()
        # Pack back into bytes
        out = bytearray((bit_length + 7) // 8)
        for pos, b in enumerate(bits):
            if b:
                out[pos // 8] |= 1 << (7 - (pos % 8))
        return bytes(out)

    @staticmethod
    def access(spec: BitBitIndex) -> Any:
        #print(f"accessed {spec.index} with spec: {spec}")
        if spec.empty:
            buf = spec.caller.buffer if hasattr(spec.caller, 'buffer') else spec.caller
            if isinstance(spec.index, slice) and spec.mode == 'view':
                start, stop, step = spec.normalize()
                reversed_ = step < 0
                return BitBitSlice(buf, start, 0, reversed=reversed_, plane=spec.plane)
            if spec.mode in ('hex', 'data_hex'):
                return ''
            if spec.mode == 'get':
                return b''
            return None
        if spec.index_hook is not None:
            translated_index = spec.index_hook(spec.index)
            if translated_index is None:
                raise KeyError(f"Index hook failed to find a match for '{spec.index}'")
            spec.index = translated_index
            spec.index_hook = None

        if BitBitIndexer.logging_enabled:
            logging.debug("[access] ENTER: index=%r, mode=%r, value=%r, caller=<%s at 0x%x>",
              spec.index, spec.mode, spec.value,
              type(spec.caller).__name__, id(spec.caller))

            if BitBitIndexer.verbosity >= 2:
                start, stop, step = spec.normalize()
                logging.debug(f"[access] normalize -> start={start}, stop={stop}, step={step}, plane={spec.plane}, base_offset={spec.base_offset}")

        # ───────────────────────────────────────────────────────────────
        # 0 · INDEX HOOK   Translate exotic keys (PID, label, etc.) into a
        #                  plain Python index *before* any other handling.
        #                  The hook must return an int or slice understood
        #                  by the downstream logic.
        # ───────────────────────────────────────────────────────────────

        buf = spec.caller.buffer if hasattr(spec.caller, 'buffer') else spec.caller



        # ───────────────────────────────────────────────────────
        # 0. VIEW → return a BitBitItem / BitBitSlice, never mutate
        # ───────────────────────────────────────────────────────
        if spec.mode == 'view':
            idxs = spec.indices()
            # single‑bit view  → BitBitItem
            if isinstance(spec.index, int):
                return BitBitItem(
                    buffer = buf,
                    mask_index = idxs[0],
                    length = 1,
                    cast = spec.caster,
                    plane = spec.plane
                )
            # slice view → BitBitSlice (handle reverse step)
            step      = spec.index.step or 1
            reversed_ = step < 0
            start_bit = idxs[0] if step > 0 else idxs[-1]
            return BitBitSlice(
                buffer   = buf,
                start_bit= start_bit,
                length   = len(idxs),
                reversed = reversed_,
                plane    = spec.plane
            )
        # custom convenience modes
        if spec.mode == 'hex':
            # 1) pull raw mask-bits 
            result = BitBitIndexer._get_mask(buf, spec.base_offset, spec.indices())
            # 2) invert bits if requested
            if isinstance(spec.caller, (BitBitItem, BitBitSlice)) and getattr(spec.caller, 'inverted', False):
                result = BitBitIndexer._invert_bits(result, spec.caller.useful_length)
            # 3) reverse bit-order if a reversed slice
            if isinstance(spec.caller, BitBitSlice) and spec.caller.reversed:
                result = BitBitIndexer._reverse_bits(result, spec.caller.useful_length)
            # 4) hex-string it
            result = result.hex()
            if BitBitIndexer.logging_enabled:
                logging.debug(f"[access] hex result={result}")
            return result

        if spec.mode == 'data_hex':
            # 1) pull raw data-bytes
            raw = BitBitIndexer._get_data(buf,
                                          spec.base_offset,
                                          spec.indices(),
                                          spec.bitsforbits,
                                          spec.caster)
            # 2) reverse byte-order if reversed slice
            if isinstance(spec.caller, BitBitSlice) and spec.caller.reversed:
                raw = raw[::-1]
            # 3) invert bytes if requested
            if isinstance(spec.caller, (BitBitItem, BitBitSlice)) and getattr(spec.caller, 'inverted', False):
                raw = bytes((~b & 0xFF) for b in raw)
            # 4) hex-string it
            result = raw.hex()
            if BitBitIndexer.logging_enabled:
                logging.debug(f"[access] data_hex result={result}")
            return result

        if spec.mode == 'repr':
            if BitBitIndexer.logging_enabled:
                logging.debug("[access] mode=repr")
            # BitBitBuffer repr
            try:
                from ..bitbitbuffer import BitBitBuffer
            except Exception:
                print("Failed to import BitBitBuffer")
                BitBitBuffer = None
            if BitBitBuffer is not None and isinstance(spec.caller, BitBitBuffer):
                full_mask = BitBitIndexer._get_mask(buf, 0, list(range(buf.mask_size)))
                result = f"BitBitBuffer(mask={full_mask.hex()}, bitsforbits={buf.bitsforbits}, mask_size={buf.mask_size})"
                if BitBitIndexer.logging_enabled:
                    logging.debug(f"[access] repr result={result}")
                return result
            elif BitBitBuffer is None:
                print("BitBitBuffer is None (indexer access)")
                return f"Empty Buffer (indexer access)"
            # BitBitSlice repr
            if isinstance(spec.caller, BitBitSlice):
                mraw = BitBitIndexer._get_mask(buf, spec.base_offset, spec.indices())
                if spec.inverted:
                    mraw = BitBitIndexer._invert_bits(mraw, spec.caller.useful_length)
                if spec.caller.reversed:
                    mraw = BitBitIndexer._reverse_bits(mraw, spec.caller.useful_length)
                draw = BitBitIndexer._get_data(buf,
                                               spec.base_offset,
                                               spec.indices(),
                                               spec.bitsforbits,
                                               spec.caster)
                if spec.caller.reversed:
                    draw = draw[::-1]
                if spec.inverted:
                    draw = bytes((~b & 0xFF) for b in draw)
                result = f"BitBitSlice(mask={mraw.hex()}, data={draw.hex()})"
                if BitBitIndexer.logging_enabled:
                    logging.debug(f"[access] repr result={result}")
                return result
            # BitBitItem repr
            if isinstance(spec.caller, BitBitItem):
                mraw = BitBitIndexer._get_mask(buf, spec.base_offset, spec.indices())
                if spec.inverted:
                    mraw = BitBitIndexer._invert_bits(mraw, spec.caller.useful_length)
                dbyte = BitBitIndexer._get_data(buf,
                                                spec.caller.data_index,
                                                [0],
                                                buf.bitsforbits,
                                                spec.caster)
                if spec.inverted:
                    dbyte = bytes((~b & 0xFF) for b in dbyte)
                result = (f"BitBitItem(mask={mraw.hex()}, data={dbyte.hex()}, "
                          f"len={spec.caller.useful_length}, idx={spec.caller.mask_index})")
                if BitBitIndexer.logging_enabled:
                    logging.debug(f"[access] repr result={result}")
                return result

        # default get/set
        idxs = spec.indices()
        if BitBitIndexer.logging_enabled and BitBitIndexer.verbosity >= 3:
            logging.debug(f"[access] final indices={idxs}")

        if spec.mode == 'get':
            if spec.plane == 'mask':
                if BitBitIndexer.logging_enabled:
                    logging.debug("[access] get ↔ _get_mask")
                result = BitBitIndexer._get_mask(buf, spec.base_offset, idxs)
            else:
                if BitBitIndexer.logging_enabled:
                    logging.debug("[access] get ↔ _get_data")
                result = BitBitIndexer._get_data(buf, spec.base_offset, idxs, spec.bitsforbits, spec.caster)
            if BitBitIndexer.logging_enabled:
                logging.debug(f"[access] get result={result}")
            return result
        else:
            if spec.plane == 'mask':
                if isinstance(spec.value, (BitBitSlice, BitBitItem)):
                    # mirror the data plane from the source view first
                    v = spec.value
                    v_idxs = list(range(len(idxs)))
                    raw = BitBitIndexer._get_data(v.buffer, v.mask_index, v_idxs, v.buffer.bitsforbits, v.cast)
                    if getattr(v, 'reversed', False):
                        raw = raw[::-1]
                    BitBitIndexer._set_data(buf, spec.base_offset, idxs, spec.bitsforbits, raw)
                if BitBitIndexer.logging_enabled:
                    logging.debug(f"[access] set↔_set_mask value={spec.value}")
                BitBitIndexer._set_mask(buf, spec.base_offset, idxs, spec.value)
                if BitBitIndexer.logging_enabled:
                    logging.debug("[access] set completed (_set_mask)")
            else:
                if BitBitIndexer.logging_enabled:
                    logging.debug(f"[access] set↔_set_data value={spec.value}")
                BitBitIndexer._set_data(buf, spec.base_offset, idxs, spec.bitsforbits, spec.value)
                if BitBitIndexer.logging_enabled:
                    logging.debug("[access] set completed (_set_data)")

    @staticmethod
    def _get_mask(buf, base: int, idxs: List[int]) -> bytes:
        if BitBitIndexer.logging_enabled and BitBitIndexer.verbosity >= 2:
            logging.debug(f"[_get_mask] base={base}, idxs={idxs}")
        # extract individual bits and pack into bytes
        bits = []
        for i in idxs:
            bit_pos = base + i
            if BitBitIndexer.logging_enabled and BitBitIndexer.verbosity >= 4:
                logging.debug(f"[_get_mask] computing bit for bit_pos={bit_pos}")
                assert isinstance(bit_pos, int) and bit_pos >= 0
            byte_i = bit_pos // 8
            bit_off = 7 - (bit_pos % 8)
            if BitBitIndexer.logging_enabled and BitBitIndexer.verbosity >= 4:
                logging.debug(f"[_get_mask] byte_i={byte_i}, bit_off={bit_off}")
                assert 0 <= bit_off < 8
            bits.append((buf.mask[byte_i] >> bit_off) & 1)
            if BitBitIndexer.logging_enabled and BitBitIndexer.verbosity >= 4:
                last_bit = bits[-1]
                logging.debug(f"[_get_mask] extracted bit={last_bit}")
                assert last_bit in (0, 1)
        out = bytearray((len(bits) + 7) // 8)
        for pos, b in enumerate(bits):
            if BitBitIndexer.logging_enabled and BitBitIndexer.verbosity >= 4:
                logging.debug(f"[_get_mask] packing bit at pos={pos}, b={b}")
                assert b in (0, 1)
            if b:
                out[pos // 8] |= 1 << (7 - (pos % 8))
        if BitBitIndexer.logging_enabled and BitBitIndexer.verbosity >= 3:
            logging.debug(f"[_get_mask] out_bytes={out.hex()}")
        return bytes(out)

    @staticmethod
    def _get_data(buf, base: int, idxs: List[int], stride: int, caster) -> bytes:
        if BitBitIndexer.logging_enabled and BitBitIndexer.verbosity >= 2:
            logging.debug(f"[_get_data] base={base}, idxs={idxs}, stride={stride}")

        # Correctly handle byte-aligned strides.
        if stride % 8 == 0:
            byte_w = stride // 8
            chunks = []
            for i in idxs:
                # The data for mask index `i` starts at data byte `i * byte_w`.
                # The `base` is a bit offset for the view, which needs to be accounted for.
                start_byte = (base * byte_w) + (i * byte_w)
                chunks.append(buf.data[start_byte : start_byte + byte_w])
            return b''.join(chunks) # Return the correct result immediately.

        # Sub-byte stride: return 1 byte per element; MSB-aligned field shifted into low bits.
        out = bytearray(len(idxs))
        for k, i in enumerate(idxs):
            start_bit = (base + i) * stride
            seg = buf.extract_bit_region(buf.data, start_bit, stride)  # MSB-first in seg[0]
            out[k] = seg[0] >> (8 - stride)  # put the S-bit value in the low bits
        return bytes(out)

    @staticmethod
    def _set_mask(buf, base: int, idxs: List[int], value: Union[int, List[int]]) -> None:
        if BitBitIndexer.logging_enabled and BitBitIndexer.verbosity >= 2:
            logging.debug(f"[_set_mask] base={base}, idxs={idxs}, value={value}")
        # value as int (0/1) or list of bits
        if isinstance(value, int):
            bits = [value] * len(idxs)
        if isinstance(value, (bytes, bytearray)):
            total = len(idxs)
            need  = (total + 7) // 8
            raw   = bytes(value)
            if len(raw) != need:
                raise ValueError("length mismatch for mask bitstream")
            bits = [ (raw[i//8] >> (7 - (i % 8))) & 1 for i in range(total) ]
        elif not isinstance(value, int):
            bits = [1 if int(b) else 0 for b in value]
            if len(bits) != len(idxs):
                raise ValueError("length mismatch")

        for bit, i in zip(bits, idxs):
            if BitBitIndexer.logging_enabled and BitBitIndexer.verbosity >= 4:
                logging.debug(f"[_set_mask] setting bit={bit} at idx={i}")
                assert bit in (0, 1) and isinstance(i, int)
            bit_pos = base + i
            byte_i = bit_pos // 8
            assert 0 <= bit_pos < buf.mask_size, "Index out of range"
            bit_off = 7 - (bit_pos % 8)
            if bit:
                buf.mask[byte_i] |= (1 << bit_off)
            else:
                buf.mask[byte_i] &= ~(1 << bit_off)
        if BitBitIndexer.logging_enabled and BitBitIndexer.verbosity >= 3:
            # verify by re-reading
            post = BitBitIndexer._get_mask(buf, base, idxs)
            logging.debug(f"[_set_mask] verified bits now={post.hex()}")

    @staticmethod
    def _set_data(buf, base: int, idxs: list[int], stride: int, value) -> None:
        if BitBitIndexer.logging_enabled and BitBitIndexer.verbosity >= 2:
            logging.debug(f"[_set_data] base={base}, idxs={idxs}, stride={stride}, value={value}")
        """
        Write to the data‑plane for *any* stride.
        • byte‑multiple strides use fast slice assignment
        • sub‑byte strides delegate to BitBitBuffer.write_bit_region
        """
        total_bits = stride * len(idxs)

        # 1) normal whole‑byte case  (8,16,24 … bits per element)
        if stride % 8 == 0:
            byte_w = stride // 8
            # normalise *value* to contiguous bytes
            if isinstance(value, (bytes, bytearray)):
                raw = bytes(value)
                if len(raw) != byte_w * len(idxs):
                    raise ValueError("length mismatch")
            elif isinstance(value, int):
                raw = value.to_bytes(byte_w, 'big') * len(idxs)
            else:                       # iterable of ints / bytes
                chunks = []
                for v in value:
                    if isinstance(v, int):
                        chunks.append(v.to_bytes(byte_w, 'big'))
                    else:
                        chunks.append(bytes(v))
                raw = b''.join(chunks)
            # fast path only if indices are contiguous ascending
            is_contig = (idxs == list(range(idxs[0], idxs[0] + len(idxs))))
            if is_contig:
                # single splice into the data plane
                first = idxs[0]
                byte_off = (base + first) * byte_w
                buf.data[byte_off : byte_off + len(raw)] = raw
                return
            # non-contiguous or reversed: write per-index to preserve ordering
            for k, i in enumerate(idxs):
                byte_off = (base + i) * byte_w
                seg = raw[k*byte_w:(k+1)*byte_w]
                buf.data[byte_off:byte_off+byte_w] = seg
            return

        # 2) sub‑byte stride → fall back to generic bit writer
        #    convert *value* into one contiguous bytearray of `total_bits`
        if isinstance(value, int):
            v   = value & ((1 << stride) - 1)
            raw = bytearray((total_bits + 7) // 8)
            pos = 0
            for _ in idxs:
                for b in range(stride):
                    if (v >> (stride - 1 - b)) & 1:
                        raw[pos // 8] |= 1 << (7 - (pos % 8))
                    pos += 1
        elif isinstance(value, (bytes, bytearray)):
            raw = bytes(value)
            need = (total_bits + 7) // 8
            if len(raw) != need:
                raise ValueError("length mismatch for packed sub-byte payload")
        else:   # iterable of ints/bits
            bits = []
            for v in value:
                v &= (1 << stride) - 1
                bits.extend(((v >> (stride - 1 - b)) & 1) for b in range(stride))
            raw = bytearray((total_bits + 7) // 8)
            for i, bit in enumerate(bits):
                if bit:
                    raw[i//8] |= 1 << (7 - i%8)

        buf.write_bit_region(buf.data, (base + idxs[0]) * stride, raw, total_bits)
        objects = []

        referring_buffer = buf._origin if hasattr(buf, '_origin') else buf

        # PID tagging/clipping should be in mask-index units, not bit-length
        absolute_start = base + idxs[0]           # mask index units
        count = len(idxs)                         # number of elements written
        absolute_end = absolute_start + count     # still mask index units

        for pids in referring_buffer.pid_buffers.values():
            # Skip buffers whose domain is fully outside the modified range
            if absolute_end <= pids.domain_left or absolute_start >= pids.domain_right:
                continue

            # Clip to the intersection of write region and PID domain
            pid_start = max(absolute_start, pids.domain_left)
            pid_end   = min(absolute_end,   pids.domain_right)

            # Round up to nearest stride-aligned offset within domain
            rel_start = ((pid_start - pids.domain_left + pids.domain_stride - 1) //
                        pids.domain_stride) * pids.domain_stride + pids.domain_left

            for abs_loc in range(rel_start, pid_end, pids.domain_stride):
                pid = pids.create_id(abs_loc)
                objects.append((abs_loc, pid))

        referring_buffer.object_references.update(objects)

        BitBitIndexer._set_mask(buf, base, idxs, 1)
