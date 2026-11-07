# Marble liquify and draw-through — development note

**Status:** design note, not a claim that molten-glass flow is implemented. The current designer builds closed ribbon, chip, and shell meshes and lights their material IDs with the exported Pluck shader. This note describes the next construction stage for the interior.

## Intended result

Construct the dyed interior as material that was injected into, stretched through, and twisted inside a softened clear-glass gather. The visible artifacts should follow from the construction: narrowing and widening ribbons, feathered edges, dragged color streaks, curled tips, small folds, trapped bubbles, and regions where adjacent colors remain distinct or diffuse into one another. A preset should retain its editable construction parameters until the user bakes it; the game can then use a decimated mesh derived from that baked result.

## Construction path

1. **Material domains.** Start with clear glass and one or more dye stocks as disjoint, volume-bearing domains. Give each stock density and its own exported MaterialDatabase record. Track the domains separately from the outer coating or painted shell.
2. **Injection.** Define nozzle positions, timing, flow rate, cross section, and dye sequence along a gather path. Multiple nozzles make genuinely separate strands; a changing sequence makes color bands within a strand. A finite dye volume is assigned before any stretching.
3. **Liquify and draw.** Advect each cross section along a user-drawn centerline. Let draw speed and local viscosity set its longitudinal stretch; preserve volume by changing cross-sectional area with the stretch. Expose a bounded softening/cooling curve, twist rate, pull distance, and tip pinch. Record the deformation in the construction rather than painting a pattern onto a finished sphere.
4. **Artifact formation.** Split, fold, or feather a domain only through explicit deformation or controlled diffusion between neighboring dye domains. Keep the undiffused boundary as geometry so the existing Phong lights can produce a hard edge; use finer material domains where a soft color transition is wanted. Avoid adding a visual-only shader effect that disagrees with the interior mesh.
5. **Containment and bake.** Clip or reform the resulting domains within the clear sphere, close every surface with consistent winding, resolve domain intersections, then triangulate at designer resolution. Preserve phase (placement within the process) separately from the final three-axis orientation of the inclusion. Bake material IDs and a source-parameter receipt with the high-detail mesh; derive smaller representations from that bake.

## Mechanical and rendering contracts

- The final closed, nonoverlapping domains determine displaced clear-glass volume. Integrate their triangles for volume, center of mass, and inertia tensor with documented densities. Inertial roll is enabled only when those integrals describe the actual baked geometry.
- The existing exported Pluck PBR/Phong/enamel tables remain the material source. The shader's transmission is an approximation, so a checkerboard scene can reveal its actual behavior without labeling the result ray tracing.
- Outer porcelain paint, confetti rolling, and partial enamel are separate construction layers. They may combine with a drawn interior, but their surface coverage and added mass must be accounted for after the interior bake.

## First acceptance experiment

Build one two-color injection with a curved pull and one full twist. Compare its cross-sectional dye area before and after drawing, inspect a cutaway and an opaque-shell view, then render it over the designer's patterned floor and wall. Changing pull speed should narrow the ribbon and lengthen it without changing its dye volume. Changing phase should rearrange the dyed domains without rotating the whole inclusion; changing orientation should rotate both geometry and its calculated inertia tensor. A decimated game mesh should retain the same material regions and computed mechanical properties within a stated error bound.
