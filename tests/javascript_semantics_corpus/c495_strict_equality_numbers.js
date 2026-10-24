// === on Numbers: 0 === -0, NaN !== NaN.
/** @param {number} a @param {number} b */
function root(a, b) { return a === b; }
// INPUTS [[0, -0], [NaN, NaN], [1, 1], [1, 2], [Infinity, Infinity]]
