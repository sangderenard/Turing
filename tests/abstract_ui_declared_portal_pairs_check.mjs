// Behaviour of the living map's portal graph with producer-declared pairs, run on the incumbent functions
// extracted from the browser template in src/compiler/abstract_ui_div_map.py.
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";

const source = readFileSync(new URL("../src/compiler/abstract_ui_div_map.py", import.meta.url), "utf8").replaceAll("\r\n", "\n");
const fn = name => {
  const match = source.match(new RegExp(`^function ${name}\\([^\\n]*\\) \\{\\n[\\s\\S]*?^\\}`, "m"));
  assert.ok(match, `incumbent function ${name}`);
  return match[0];
};
const build = new Function("portalRuntime", [fn("portalCurvePoint"), fn("portalApertureClass"),
  fn("portalDeclaredPair"), fn("rebuildPortalGraph"), "return rebuildPortalGraph;"].join("\n"));
const splat = (identity, port_role, x, extra = {}) => ({identity, port_role, center: [x, 0, 0], normal: [0, 0, 1], ...extra});
const graphOf = splats => {
  const runtime = {splats, graph: {identity: "g", sigma: 4.0, nodes: [], edges: []}};
  return build(runtime)();
};
const links = graph => graph.edges.map(edge => `${edge.source}>${edge.target}`).sort();

// Unchanged: splats without a pair form the derived all-in x all-out graph, weights normalised per source.
{
  const graph = graphOf([splat("g/nodes/in-1", "in", 0), splat("g/nodes/in-2", "in", 1),
    splat("g/nodes/out-3", "out", 2), splat("g/nodes/out-4", "out", 9)]);
  assert.deepEqual(links(graph), ["g/nodes/in-1>g/nodes/out-3", "g/nodes/in-1>g/nodes/out-4",
    "g/nodes/in-2>g/nodes/out-3", "g/nodes/in-2>g/nodes/out-4"]);
  for (const source of ["g/nodes/in-1", "g/nodes/in-2"]) {
    const total = graph.edges.filter(edge => edge.source === source).reduce((sum, edge) => sum + edge.weight, 0);
    assert.ok(Math.abs(total - 1) < 1e-12);
  }
  assert.equal(graph.edges[0].identity, "g/edges/in-1-to-out-3");
  assert.ok(graph.edges.every(edge => !("pair" in edge)));
}

// A declared pair links only its own two ends, with certainty, however far apart and whatever else is nearby.
{
  const graph = graphOf([
    splat("w/a/end:in", "in", 0, {pair: "P1"}), splat("w/b/end:out", "out", 500, {pair: "P1"}),
    splat("w/c/end:in", "in", 0.1, {pair: "P2"}), splat("w/d/end:out", "out", 0.2, {pair: "P2"}),
    splat("g/nodes/in-9", "in", 0.05), splat("g/nodes/out-10", "out", 0.15)]);
  assert.deepEqual(links(graph), ["g/nodes/in-9>g/nodes/out-10", "w/a/end:in>w/b/end:out", "w/c/end:in>w/d/end:out"]);
  const declared = graph.edges.filter(edge => edge.pair);
  assert.deepEqual(declared.map(edge => edge.pair).sort(), ["P1", "P2"]);
  assert.ok(declared.every(edge => edge.weight === 1));
  assert.equal(new Set(graph.edges.map(edge => edge.identity)).size, graph.edges.length);
  assert.ok(graph.edges.every(edge => Number.isFinite(edge.length) && edge.length > 0));
}

// Two declared ends whose identities share a last path segment still get distinct edge identities.
{
  const graph = graphOf([splat("x/1/end:in", "in", 0, {pair: "A"}), splat("x/2/end:out", "out", 3, {pair: "A"}),
    splat("y/1/end:in", "in", 0, {pair: "B"}), splat("y/2/end:out", "out", 3, {pair: "B"})]);
  assert.equal(new Set(graph.edges.map(edge => edge.identity)).size, 2);
}

// A declared end with no partner yet has no edge (a doorway to a place that is not built).
assert.equal(graphOf([splat("w/a/end:in", "in", 0, {pair: "P1"}), splat("g/nodes/out-1", "out", 1)]).edges.length, 0);

// The opening-to-splat conversion carries the declaration.
const migrate = fn("migrateLegacyPortalOpenings");
for (const text of ["...(opening.pair ? {pair: opening.pair} : {})", "opening.aperture_class", "opening.reference"])
  assert.ok(migrate.includes(text), text);
console.log("declared portal pairs ok");
