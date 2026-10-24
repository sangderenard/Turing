// An arm that returns takes the rest of the function with the other arm only.
/** @param {number} x */
function root(x) { let r = 1; if (x < 0) { return -1; } else { r = x; } r = r * 3; if (r > 6) { return r; } return 0; }
// INPUTS [[-1], [1], [5], [NaN], [Infinity]]
