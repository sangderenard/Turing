// Local exports consume the Studio's existing saved recipe and the accepted
// ray frame. They never save to the server or synthesize an image from Phong.
import { MARBLE_OBJECT_VERSION } from "./marble_object.js";

const PNG_SIGNATURE = [137, 80, 78, 71, 13, 10, 26, 10];

export function marbleDownloadFilename(name, {kind, timestamp = new Date(), digest = ""} = {}) {
  if(kind !== "object" && kind !== "live")throw new RangeError("Unknown Studio export kind.");
  const date = new Date(timestamp);
  if(!Number.isFinite(date.getTime()))throw new RangeError("Invalid Studio export timestamp.");
  if(digest && (typeof digest !== "string" || !/^[a-f0-9]{8,64}$/i.test(digest)))
    throw new RangeError("Invalid Studio export digest.");
  const stem = String(name || "object").normalize("NFKD").replace(/[\u0300-\u036f]/g, "")
    .replace(/[^a-z0-9_-]+/gi, "-").replace(/-+/g, "-").replace(/^[-_]+|[-_]+$/g, "")
    .slice(0, 64).replace(/[-_]+$/g, "") || "object";
  const stamp = date.toISOString().replace(/[-:.]/g, "");
  // The fixed prefix also keeps Windows device names (CON, NUL, etc.) harmless.
  return `marble-${stem}-${kind}-${stamp}${digest ? `-${digest.slice(0, 12).toLowerCase()}` : ""}.${kind === "object" ? "json" : "png"}`;
}

export function createMarbleObjectDownload(snapshot, options = {}) {
  if(!snapshot || snapshot.v !== MARBLE_OBJECT_VERSION || !["meowdoku-marble-object-v1","meowdoku-imported-object-v1"].includes(snapshot.generator))
    throw new RangeError("Export requires a versioned saved marble object.");
  // Preserve every field of the existing recipe. Refuse values JSON would
  // silently discard or change rather than emit a lossy, apparently valid file.
  const json = JSON.stringify(snapshot, (_key, value) => {
    if(["undefined", "function", "symbol", "bigint"].includes(typeof value)
      || (typeof value === "number" && !Number.isFinite(value)))
      throw new TypeError("The saved marble object contains a non-JSON value.");
    return value;
  }, 2);
  return {
    filename: marbleDownloadFilename(options.name ?? snapshot.name, {
      ...options, kind: "object", digest: options.digest ?? snapshot.geometry_sha256 ?? "",
    }),
    blob: new Blob([json, "\n"], {type: "application/json;charset=utf-8"}),
  };
}

export async function createLivePngDownload(blob, options = {}) {
  if(!(blob instanceof Blob) || blob.size === 0)
    throw new TypeError("No accepted live PNG frame is available yet.");
  if(blob.type.toLowerCase() !== "image/png")throw new TypeError("The accepted live frame is not a PNG.");
  const header = new Uint8Array(await blob.slice(0, PNG_SIGNATURE.length).arrayBuffer());
  if(header.length !== PNG_SIGNATURE.length || !PNG_SIGNATURE.every((byte, index) => header[index] === byte))
    throw new TypeError("The accepted live frame has an invalid PNG signature.");
  return {filename: marbleDownloadFilename(options.name, {...options, kind: "live"}), blob};
}

async function downloadArtifact({filename, blob}, {
  document: doc = globalThis.document,
  URL: urls = globalThis.URL,
  delay = ms => new Promise(resolve => setTimeout(resolve, ms)),
} = {}) {
  if(!doc?.body || typeof doc.createElement !== "function"
    || typeof urls?.createObjectURL !== "function" || typeof urls?.revokeObjectURL !== "function")
    throw new Error("File downloads are not available in this browser.");
  const url = urls.createObjectURL(blob);
  let anchor;
  try {
    anchor = doc.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    anchor.hidden = true;
    doc.body.appendChild(anchor);
    anchor.click();
    // Browsers may consume the Blob after the click returns. Keep it alive for
    // that handoff, then release this export's URL (not the retained live Blob).
    await delay(1000);
    return filename;
  } finally {
    try { anchor?.remove(); }
    finally { urls.revokeObjectURL(url); }
  }
}

// Rejections are intentionally passed to the caller's visible status handler.
// Success means a browser download was requested, not that disk delivery is known.
export async function downloadMarbleObject(snapshot, options = {}) {
  return downloadArtifact(createMarbleObjectDownload(snapshot, options), options);
}

export async function downloadLivePng(blob, options = {}) {
  return downloadArtifact(await createLivePngDownload(blob, options), options);
}
