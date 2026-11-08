# Fountain 3D: bingo hopper, 8 ball, and composite grains

Status: design note, 2026-11-07. None of the features below is implemented by this note. Keep the existing Fountain and Fountain3D laws, compiled products, and chamber controls as the starting point.

## Bingo hopper

Make a rotating 3D drum that contains numbered grains or balls. A catch pocket and gate should admit **one item per flip**. The admitted item leaves the moving population, remains visible in the pocket or output tray, and appears in a persistent draw list by its original particle `pid`. Do not reuse that ID during the run.

The capture must be a deterministic state transition in the simulation: test the gate crossing after case motion and contact for a step, choose at most one crossing by a documented step and ID order, and mark it captured. Capture must not be inferred from a screen position or remove an item only from the renderer. Log `{flip, step, pid}` once; replay and compact visual recording should show the same draw sequence. A manual flip and automatic low-energy flip should use the same gate rule. Define the drum, gate, pocket, and outlet in the same geometry source used for physical walls and displayed mesh, with a visible way to inspect the draw list.

Current seam: `fountain3d.py` has typed SDF fluid/solid slots and the 3D contact law; `fountain_game.py` chooses a chamber preset and ships its `chamber_mesh`; `web/compute.js` owns the compiled run and its replay inputs. The existing wheel body state describes a hinged rotating obstacle, not a one-item capture gate. This feature therefore needs explicit captured state and an event path, not just a new chamber label.

Acceptance: a fixed seed and flip sequence draws the same IDs in native and WebGPU runs; no ID is listed twice; each flip adds zero or one ID; the listed ID no longer participates in subsequent grain contacts; total active plus captured plus parked accounting remains consistent. Test a gate with two grains crossing in one step and a grain touching the lip without crossing.

## Magic 8 ball chamber

Offer a spherical chamber with one **flat side**. The flat face should be part of the containing surface and the visible mesh, so grains cannot pass through a merely decorative window. The existing SDF composition unions FLUID slots and subtracts SOLID slots; a sphere FLUID slot with a sufficiently wide box SOLID slot cutting off one cap is a candidate for a clipped sphere without adding a new SDF primitive. Verify the resulting analytic contact normal at the circular rim and the flat face, and derive the displayed zero-surface mesh from that same composed field. Orient the flat side with case motion and allow the page's material controls to distinguish shell and window.

If this is to behave as a traditional answer ball, define the answer object, its selectable faces, and the readout as a separate interaction after the containing chamber is correct. Do not substitute a cosmetic answer overlay for the simulated object.

Current seam: `fountain3d.py` already has sphere and box typed slots and `sdf_field`; `fountain_game.py:chamber_mesh` extracts the visible SDF surface. This is the smallest candidate feature, but changing the chamber preset and mesh requires a focused physical and browser check, and a changed law source can invalidate compiled product digests. It was not added during this low-credit pass.

Acceptance: the flat face and rim coincide in collision and rendering; a grain settles against the flat face under each relevant orientation; no escape through the seam; existing sphere behavior remains unchanged.

## Edge lists and composite objects in Fountain and Fountain3D

Add an optional graph over particle IDs to both simulators. Each edge connects two existing grains and carries rest geometry and a physical law (stiffness, damping, break rule, and optionally formation state). Compute an equal-and-opposite force for each edge and the corresponding torque at the contact offsets; preserve the existing mass, spin, friction, and contact accounting. An edge list is part of simulation state and replay, not a renderer-only set of lines. Keep the ordinary independent-grain path when no edges are present so large dust simulations do not pay for bonded-object work.

Connected components can form polygonal or polyhedral crystals from bound dust corners. Their collision shape must exclude other grains from the **interior of the convex hull mesh**, not only from its vertices or edge rods. Store an oriented hull face list or equivalent support representation for each component. A max-of-face-planes field gives a correct inside test for a convex polytope, but is not the exact exterior distance around edges and corners; use closest-feature or support-based contact there. Handle point-versus-hull and hull-versus-hull contacts, self-contact suppression within one bonded component, separation after bond breakage, and changes in mass distribution and moment of inertia. The displayed crystal mesh and the force hull must come from the same component geometry.

The current 3D law sorts cells and walks bounded neighbor runs (`K_RUN_MAX`), reports dropped contacts, and has angular velocity state. Bond edges need their own deterministic gather/reduction or a justified integration with that walk; a spatial neighbor cutoff cannot silently drop a permanent bond. The 2D law needs the matching graph contract with polygon interiors. Follow the compiler's existing graph and tensor lowering path and measure arena slots, dispatches, contact drops, and timestep stability before enabling bonds at scale. Do not replace the contact law with a visual aggregation shortcut.

First engineering slice: one fixed triangle of three grains in 2D and one tetrahedron of four grains in 3D, each in an otherwise empty chamber. Check force balance, angular momentum, moment of inertia, inside exclusion, face/edge/corner contact normals, and deterministic breakage against the unbonded baseline. Only then expose authored edge lists and crystal presets in Studio.

## Order when work resumes

1. Prototype and verify the clipped-sphere 8 ball using existing SDF slot composition and the current chamber mesh path.
2. Define the hopper's capture state and event/replay contract, then its gate geometry and UI.
3. Specify the common 2D/3D edge-list ABI and a tiny bonded-component law; add convex hull contact after the force and momentum checks pass.

WebGPU product builds can follow the existing fast lane. Keep Wasm fallback builds lazy, when a device actually needs one or the server is idle, as previously requested.

## Prompt history

> After can you make while i have you for the 3d fountain a bingo hopper, a drum that can catch single items and list their id as they are caught and removed each flip, and a magic 8 ball, a sphere with a flat side, and can you also introduce to fountain and fountain 3d edge lists that are for a aspiring force engine allowing for fountains to have composite objects so we can get polygon crystals out of bound dust corners and extend exclusion to the interior of convex hull meshes, put it down in an idea file that's probably too much for the credits we have left. If you can do anything of these super quick though maybe try
