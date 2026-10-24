// Nested if/else merges inner versions before the outer merge.
/** @param {number} x */
function root(x) { let a = x; let b = 0; if (a > 0) { if (a > 10) { b = 2; } else { b = 1; } } else { b = -1; } return b; }
// INPUTS [[-5], [5], [50], [NaN], [10]]
