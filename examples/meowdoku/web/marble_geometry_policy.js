// Exposure policy changes neither the saved mesh nor its geometry receipt.
// Both tracer directions and nested media have passed the exact-sphere GPU gate.
// New-page selection is explicit in HTML; omitted legacy options remain mesh.
export const EXACT_SPHERES_READY = true;
export const DEFAULT_TRIANGLE_BUDGET = 250000;

export function geometryPolicy({triangleBudget = DEFAULT_TRIANGLE_BUDGET, sphereCoverage = "mesh"} = {}) {
  if (!Number.isInteger(triangleBudget) || triangleBudget < 1000 || triangleBudget > 1000000)
    throw new RangeError("Triangle budget must be a whole number from 1,000 to 1,000,000.");
  if (sphereCoverage !== "mesh" && sphereCoverage !== "exact")
    throw new RangeError("Choose Saved triangle mesh or Exact analytic spheres.");
  return {triangleBudget, sphereCoverage};
}

export function photoGeometryPolicy(doc = document) {
  const input = doc.getElementById("photo-triangle-budget");
  const policy = geometryPolicy({
    triangleBudget: input ? Number(input.value) : DEFAULT_TRIANGLE_BUDGET,
    sphereCoverage: doc.getElementById("photo-sphere-coverage")?.value ?? "mesh",
  });
  if (policy.sphereCoverage === "exact" && !EXACT_SPHERES_READY)
    throw new RangeError("Exact analytic spheres are unavailable until renderer validation is complete. Choose Saved triangle mesh.");
  return policy;
}

export function createGeometryPolicyControls(doc = document) {
  const budget = doc.getElementById("photo-triangle-budget"), coverage = doc.getElementById("photo-sphere-coverage");
  const exact = doc.getElementById("photo-sphere-exact");
  if (exact) exact.disabled = !EXACT_SPHERES_READY;
  const sync = () => {
    budget?.setCustomValidity?.(""); coverage?.setCustomValidity?.("");
    try { photoGeometryPolicy(doc); }
    catch (error) {
      const input = error.message.startsWith("Triangle budget") ? budget : coverage;
      input?.setCustomValidity?.(error.message);
    }
  };
  for (const input of [budget, coverage]) {
    input?.addEventListener("input", sync);
    input?.addEventListener("change", sync);
  }
  sync();
}

export function showGeometryDiagnostic(doc, value) {
  const output = doc.getElementById("photo-geometry-diagnostic");
  if (!output) return;
  if (!value) { output.textContent = "Geometry counts will be reported by the next photo or live scene."; return; }
  const stats = value.geometry_stats;
  if (!stats) {
    output.textContent = "Geometry counts were not recorded for this exposure.";
    return;
  }
  if (!["mesh", "exact"].includes(stats.sphere_coverage) ||
      ![stats.triangle_budget, stats.input_triangles, stats.effective_triangles, stats.analytic_spheres]
        .every(number => Number.isSafeInteger(number) && number >= 0)) {
    output.textContent = "Geometry counts are unavailable: the exposure diagnostic is invalid.";
    return;
  }
  const count = number => number.toLocaleString("en-US");
  output.textContent = `${stats.sphere_coverage === "exact" ? "Exact analytic spheres" : "Saved triangle mesh"}: ` +
    `${count(stats.effective_triangles)} scene triangles / ${count(stats.triangle_budget)} budget; ` +
    `${count(stats.input_triangles)} input triangles; ${count(stats.analytic_spheres)} analytic spheres. ` +
    "Analytic spheres do not consume the triangle budget.";
}
