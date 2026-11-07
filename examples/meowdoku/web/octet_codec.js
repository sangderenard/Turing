// JS decoder of src/common/tensors/compression/octet_codec.py (`compress_octets` with no codebook): LZ77 groups over canonical Huffman.
// A faithful port, tested bit-exact against the Python encoder (tests/test_meowdoku_loading_anim.py).  Decode only; the page never encodes.
//
//   packet   = LZ header (varints: output length n, group count, command octets) | varint coded-command length | Huffman(commands) | Huffman(literals)
//   Huffman  = version 1 | varint raw length | varint valid bits | 128 octets of code lengths (two per octet, high nibble first) | MSB-first stream
//   group    = (literal count, match length, match distance): literals from the literal stream, then `length` octets copied from `distance` back
// Canonical codes: first[1] = 0, first[w] = (first[w-1] + count[w-1]) * 2; equal widths take consecutive codes in symbol order.

const MAX_BITS = 12, ALPHABET = 256, VARINT_LIMIT = 8;

function readVarint(bytes, at, what) {
  let value = 0, scale = 1, shift = 0;
  for (;;) {
    if (at >= bytes.length || shift > 7 * VARINT_LIMIT) throw new Error(`octet codec: ${what} is truncated`);
    const octet = bytes[at++];
    value += (octet & 0x7f) * scale;
    scale *= 128; shift += 7;
    if (!(octet & 0x80)) return [value, at];
  }
}

export function huffmanDecode(packet) {
  if (!packet.length || packet[0] !== 1) throw new Error("octet Huffman packet has no valid header");
  let [count, at] = readVarint(packet, 1, "Huffman header");
  let bitCount; [bitCount, at] = readVarint(packet, at, "Huffman header");
  if (count === 0) {
    if (bitCount || packet.length !== at) throw new Error("empty octet Huffman packet carries a stream");
    return new Uint8Array(0);
  }
  const tableEnd = at + ALPHABET / 2;
  if (tableEnd > packet.length || bitCount > 8 * (packet.length - tableEnd) || packet.length - tableEnd !== Math.ceil(bitCount / 8)) {
    throw new Error("octet Huffman stream length disagrees with its header");
  }
  const lengths = new Uint8Array(ALPHABET);
  for (let i = 0; i < ALPHABET / 2; i++) { const b = packet[at + i]; lengths[2 * i] = b >> 4; lengths[2 * i + 1] = b & 15; }
  // canonical table: per width the first code, the count, and the symbols in order
  const counts = new Array(MAX_BITS + 2).fill(0);
  for (const l of lengths) { if (l > MAX_BITS) throw new Error("code length out of range"); if (l) counts[l]++; }
  const first = new Array(MAX_BITS + 2).fill(0), offset = new Array(MAX_BITS + 2).fill(0);
  let code = 0, seen = 0;
  for (let w = 1; w <= MAX_BITS; w++) {
    if (w > 1) code = (code + counts[w - 1]) * 2;
    first[w] = code; offset[w] = seen; seen += counts[w];
  }
  const symbols = new Uint16Array(seen);
  const fill = offset.slice();
  for (let s = 0; s < ALPHABET; s++) if (lengths[s]) symbols[fill[lengths[s]]++] = s;
  const out = new Uint8Array(count);
  let bit = 0;
  const payload = tableEnd;
  for (let n = 0; n < count; n++) {
    let acc = 0, w = 0, found = -1;
    while (w < MAX_BITS) {
      if (bit >= bitCount) throw new Error("octet Huffman stream is corrupt");
      acc = acc * 2 + ((packet[payload + (bit >> 3)] >> (7 - (bit & 7))) & 1);
      bit++; w++;
      if (counts[w] && acc >= first[w] && acc - first[w] < counts[w]) { found = symbols[offset[w] + acc - first[w]]; break; }
    }
    if (found < 0) throw new Error("octet Huffman stream is corrupt");
    out[n] = found;
  }
  if (bit !== bitCount) throw new Error("octet Huffman stream is corrupt");
  return out;
}

// the length in octets of the Huffman packet starting at `at` is not self-describing, so the container carries it (coded command length)
export function lzDecode(commands, literals, n, groups) {
  const out = new Uint8Array(n);
  let c = 0, l = 0, o = 0;
  const next = () => {
    let value = 0, scale = 1;
    for (let k = 0; ; k++) {
      if (c >= commands.length || k >= VARINT_LIMIT) throw new Error("LZ packet commands are not whole varints");
      const octet = commands[c++];
      value += (octet & 0x7f) * scale; scale *= 128;
      if (octet < 128) return value;
    }
  };
  for (let g = 0; g < groups; g++) {
    const lit = next(), len = next(), dist = next();
    if (o + lit + len > n || l + lit > literals.length) throw new Error("LZ packet groups disagree with its lengths");
    for (let i = 0; i < lit; i++) out[o++] = literals[l++];
    if (len) {
      if (dist < 1 || dist > o) throw new Error("LZ packet copies from before its output");
      for (let i = 0; i < len; i++, o++) out[o] = out[o - dist];
    }
  }
  if (o !== n || c !== commands.length || l !== literals.length) throw new Error("LZ packet groups disagree with its lengths");
  return out;
}

export function decompressOctets(packet) {
  packet = packet instanceof Uint8Array ? packet : new Uint8Array(packet);
  let at = 0;
  const head = [];
  for (let i = 0; i < 3; i++) { let v; [v, at] = readVarint(packet, at, "LZ header"); head.push(v); }
  const [n, groups, commandBytes] = head;
  let codedLength; [codedLength, at] = readVarint(packet, at, "codec header");
  if (at + codedLength > packet.length) throw new Error("octet codec packet is truncated");
  if (n === 0) {
    if (groups || commandBytes) throw new Error("empty LZ packet carries data");
    return new Uint8Array(0);
  }
  const commands = huffmanDecode(packet.subarray(at, at + codedLength));
  const literals = huffmanDecode(packet.subarray(at + codedLength));
  if (commands.length !== commandBytes) throw new Error("octet codec command stream disagrees with its header");
  return lzDecode(commands, literals, n, groups);
}
