2026-11-09 TITLE (MEASURED, where): 
# Stage-sized gentle course geometry, not an activated physics product

New source-only marble_studio_courses.js and its focused Node suite author
gentle-S and broad-return trough routes from explicit stage width/length/baseY
in the existing Studio unit frame. Radius never sizes the stage or course.
Raised rails/end stops are actual geometry; running grades are <=5%, measured
at 1.8% / 2.2% for the 16x10 QA stage. Fit feedback is advisory, not an automatic
marble resize. Spawn/finish metadata is a surface point, not a collision-safe
body-center placement.

Global vertices and exact indexed triangles are authoritative. Every convex
construction prism is closed/outward-wound and retains the shared compound
bodyId. Exact internal triangles pair oppositely using index identity, never
tolerance welding. Exterior Studio faces link back to their actual part and
triangle. Authored surface certificates distinguish planar nonconvex S ribbons
from convex planar return straights; folded route groups are not certified as
convex or activated. Native union ownership remains necessary across parts.

Default 48-segment routes each have 576 construction parts and 1368 exterior
triangles. No capacity/performance claim. The module explicitly returns
collisionReady:false and is not imported by Studio or added to production
export lists while the native compound manifold/product remains unfinished.
No replacement integrator, contact law, physics material mapping, body toggle,
server endpoint, live worker change, browser action, restart, export, commit
or push occurred.

Passed node tests/meowdoku_marble_studio_courses_check.mjs: ten constructed
stage/aspect/resolution cases, all-part topology and convexity, exact internal
pairing/adjacency, full exterior topology, part/union volume equality, actual
render-to-part identity, authored plane validation, footprint/grade, material
identity, invalid input and radius-independent geometry. JS syntax passes.
Inspected CPU rendering of exact exported triangles at
build/dot-studio-course-20261109/courses.png. No GPU or native dynamic course
execution was tested. Existing keyboard/input suite still passes independently.
