2026-10-24 PRECISION LANE READS TODAY'S LOWERING (MEASURED, cloud): the
precision-core GLSL lane compiles the lowered pack cores again. It follows
the bound through Casts and fuses a body that calls several planned regions
into one; unsupported shapes refuse by name. Both flavours match a Python
reference evaluation of the lowered module on llvmpipe (split bit-identical;
fma once- or twice-rounded per element). The two C-lane comparison tests fail
by name: the C lane publishes no buffer for the core's Precision formals, and
its run used to abort pytest on heap corruption. The lowering itself still
reads one double per element and stores only the leading limb, which is
wrong for limbed inputs (decided by Albert; it is the next PR). At this
branch on main 8fb85e7, the 12 GL test files are identical to main except
the two emission tests, which now pass. C corpus 379/380, with
c428_class_keyword_construct WRONG on main too. Fuzz seed 69 (40): 0 WRONG /
0 TYPE. Gate names identical to main 8fb85e7 in all four parts. See doc 38.
