// A Number condition in ?: takes JavaScript truthiness, NaN included.
/** @param {number} x @param {number} y */
function root(x, y) { return x ? x : y; }
// INPUTS [[0, 5], [NaN, 5], [3, 5], [-0, 7], [-2, 1]]
