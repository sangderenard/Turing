// Nested loops: the inner loop's Phis sit inside the outer body.
/** @param {number} n */
function root(n) { let s = 0; for (let i = 0; i < n; i++) { for (let j = 0; j < i; j++) { s += j; } } return s; }
// INPUTS [[0], [3], [6], [2.5]]
