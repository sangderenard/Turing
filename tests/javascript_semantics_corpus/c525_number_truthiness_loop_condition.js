// A Number loop condition (i) takes JavaScript truthiness.
/** @param {number} n */
function root(n) { let s = 0; for (let i = n; i; i--) { s += i; } return s; }
// INPUTS [[0], [4], [5]]
