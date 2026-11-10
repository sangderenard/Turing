// Minimal reader for NumPy's .npz format (a ZIP archive of .npy arrays),
// enough to load Spectral's saved linear HDR exposures client-side so the
// marble photo can be re-tonemapped live without another ray trace.
//
// Parses the ZIP *central directory* (authoritative compressed sizes and
// offsets) rather than trusting local file headers, since zipfile can write
// those with a zero size and a trailing data descriptor. Python's
// zipfile.ZIP_DEFLATED stores raw DEFLATE with no zlib/gzip wrapper, which
// is exactly what the browser's DecompressionStream('deflate-raw') expects.
const textDecoder = new TextDecoder("utf-8");

function findEndOfCentralDirectory(view, bytes) {
  const max = Math.min(bytes.length, 65557); // EOCD record + max comment length
  for (let i = bytes.length - 22; i >= bytes.length - max && i >= 0; i--) {
    if (view.getUint32(i, true) === 0x06054b50) return i;
  }
  throw new Error("not a zip file (no end-of-central-directory record)");
}

const NPY_DTYPES = {
  "<f4": Float32Array, "<f8": Float64Array,
  "<i4": Int32Array, "<i8": BigInt64Array,
  "<u4": Uint32Array, "<u8": BigUint64Array,
  "|u1": Uint8Array, "|i1": Int8Array,
};

function parseNpy(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (textDecoder.decode(bytes.subarray(1, 6)) !== "NUMPY") throw new Error("not an .npy array");
  const major = bytes[6];
  const headerStart = major >= 2 ? 12 : 10;
  const headerLen = major >= 2 ? view.getUint32(8, true) : view.getUint16(8, true);
  const header = textDecoder.decode(bytes.subarray(headerStart, headerStart + headerLen));
  const descr = /'descr':\s*'([^']+)'/.exec(header)?.[1];
  const fortran = /'fortran_order':\s*(True|False)/.exec(header)?.[1] === "True";
  const shapeText = /'shape':\s*\(([^)]*)\)/.exec(header)?.[1] ?? "";
  const shape = shapeText.split(",").map((s) => s.trim()).filter((s) => s.length).map(Number);
  if (fortran) throw new Error("fortran-ordered .npy arrays are not supported");
  const Ctor = NPY_DTYPES[descr];
  if (!Ctor) throw new Error(`unsupported .npy dtype ${descr}`);
  const dataStart = headerStart + headerLen;
  const raw = bytes.buffer.slice(bytes.byteOffset + dataStart, bytes.byteOffset + bytes.byteLength);
  return { data: new Ctor(raw), shape, dtype: descr };
}

async function inflateRaw(bytes) {
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("deflate-raw"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

// Returns a Map of array name -> { data, shape, dtype }.
export async function parseNpz(buffer, arrayNames = null) {
  // Read only requested numeric arrays when a current HDR also contains
  // Unicode JSON metadata. Unselected members remain byte-for-byte in the
  // original archive; they are not decoded as an invented numeric dtype.
  if (arrayNames !== null && (!Array.isArray(arrayNames) || arrayNames.some(name => typeof name !== "string")))
    throw new Error("NPZ array selection must be an array of names");
  const selected = arrayNames === null ? null : new Set(arrayNames);
  const bytes = new Uint8Array(buffer);
  const view = new DataView(buffer);
  const eocd = findEndOfCentralDirectory(view, bytes);
  const entryCount = view.getUint16(eocd + 10, true);
  let offset = view.getUint32(eocd + 16, true);
  const out = new Map();
  for (let i = 0; i < entryCount; i++) {
    if (view.getUint32(offset, true) !== 0x02014b50) throw new Error("malformed zip central directory");
    const method = view.getUint16(offset + 10, true);
    const compressedSize = view.getUint32(offset + 20, true);
    const nameLen = view.getUint16(offset + 28, true);
    const extraLen = view.getUint16(offset + 30, true);
    const commentLen = view.getUint16(offset + 32, true);
    const localHeaderOffset = view.getUint32(offset + 42, true);
    const name = textDecoder.decode(bytes.subarray(offset + 46, offset + 46 + nameLen));
    offset += 46 + nameLen + extraLen + commentLen;
    if (!name.endsWith(".npy") || (selected && !selected.has(name.slice(0, -4)))) continue;
    if (method !== 0 && method !== 8) throw new Error(`unsupported ZIP compression method ${method}`);
    const localNameLen = view.getUint16(localHeaderOffset + 26, true);
    const localExtraLen = view.getUint16(localHeaderOffset + 28, true);
    const dataStart = localHeaderOffset + 30 + localNameLen + localExtraLen;
    const compressed = bytes.subarray(dataStart, dataStart + compressedSize);
    const raw = method === 0 ? compressed : await inflateRaw(compressed);
    if (name.endsWith(".npy")) out.set(name.slice(0, -4), parseNpy(raw));
  }
  return out;
}
