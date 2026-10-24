// const bindings, unary minus (-0 survives) and the exact Math.sqrt.
/** @param {number} a @param {number} b */
function root(a, b) { const d = a - b; const n = -d; if (d > 0) { return Math.sqrt(d) / 2; } return n * 0.5; }
// INPUTS [[9, 1], [1, 9], [2, 2], [NaN, 1], [Infinity, 1], [0, 1e-300]]
