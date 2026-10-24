// while reassigning a parameter; the exit value is the header Phi.
/** @param {number} x */
function root(x) { let k = 0; while (x > 1) { x = x / 2; k++; } return k + x; }
// INPUTS [[1], [8], [1000], [0.5], [NaN], [1e300]]
