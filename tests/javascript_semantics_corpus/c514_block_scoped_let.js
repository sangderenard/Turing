// A block's let is scoped to the block; the outer parameter keeps its new version.
/** @param {number} x */
function root(x) { x = x + 1; { let t = x * 2; x = t; } return x; }
// INPUTS [[1], [-1], [NaN]]
