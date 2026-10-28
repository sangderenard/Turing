2026-10-24 LANE D, D2: OPTIONAL SCALAR PAYLOADS (MEASURED, cloud):
`_planned_scalar_dtypes_from` types an IfExp or Phi with a `None` literal arm
by its non-None arms, the payload class the linked-ABI rule already states
for an optional call result. Emitted-plan float64 fallbacks over 387
programs: 724 at main ef82644e -> 691 (c367 31 -> 11; the rest of c367's are
the None literals themselves). Gate at 3d08b09d vs main ef82644e: failure
names identical in all four parts (aa 77, ab 7, ac 26, rec 20). Typed corpus
387/387 on both trees. Fuzz seed 32, FUZZ_EXT=3: 30 PASS. Also WRONG on
main, unchanged here and sent to the silent-wrong lane: an optional whose
payload arms mix int and float (`None if i == 1 else (i if i % 2 == 0 else
i * 0.5)`) truncates the float arm.
