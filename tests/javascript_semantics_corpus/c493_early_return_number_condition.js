// Early-return if on a Number condition, both paths returning Numbers.
/** @param {number} x */
function root(x) { if (x) { return 1; } return 2; }
// INPUTS [[0], [NaN], [3], [-0], [Infinity]]
