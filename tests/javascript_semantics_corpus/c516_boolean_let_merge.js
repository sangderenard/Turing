// A boolean let merged across an if keeps the boolean kind.
/** @param {number} x */
function root(x) { let k = true; if (x === 0) { k = false; } return k; }
// INPUTS [[0], [-0], [1], [NaN]]
