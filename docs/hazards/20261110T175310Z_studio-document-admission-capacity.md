2026-11-10 Document admission capacity preflight (MEASURED, offline Windows)

Independent review found that the importer checked document-array mutability
but not capacity before canonical append. The focused reproduction prepared
and validated two ordinary new shelf cuboids at normal array size, then set
model.document_geometry.boxes.length to 0xfffffffe and called only the actual
admission function. The canonical object/part tables all grew by two before
document retention failed. This reproduced the exact partial-mutation boundary
without iterating or serializing the sparse expanded array.

The importer now rejects boxes.length + missing.length >= 0x100000000 before
constructing or appending any objects. Equality and overflow (sparse lengths
0xfffffffe and 0xffffffff with two parts) both reject with canonical map sizes,
world/order/specialization lengths and document keys unchanged. Tests restore
normal length in finally and inspect only bounded counts, known map keys and
Object.keys. No sparse-array iteration or stringify is used.

Offline gates passed: bridge syntax, full combined native bridge/canonical
admission test on corrected registry-game.html, both preserved-original
compatibility checks, and six patcher pytest cases in 1.47 seconds. Candidate
SHA256 remains 6d3016268e4069e9411f4c23819c716330e659fb731a65e6488cda5b21a1052b.

Frozen bridge SHA256:
a072ab655c14733d9ab0e2786d5828972a8c9ed3b329cf57e85f721f9479e514.
Frozen focused test SHA256:
6f21cf390776a279afc4d69d32915f6651f45b7fb1d41edd3bc38c74e648673e.
Only the importer guard and focused regression changed in this repair. No
canonical registry, identity/auth/exposure, worker, physics, server, live page,
publication, commit or push changed. Parent owns subsequent offline restaging.
