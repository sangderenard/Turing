// -0 && b is -0, and 1 / that is -Infinity.
/** @param {number} a @param {number} b */
function root(a, b) { return 1 / (a && b); }
// INPUTS [[5, 3], [0, 3], [-0, 3], [NaN, 1], [1, NaN], [0, -0], [-0, 0], [NaN, 0], [Infinity, 0], [-1, -0]]
