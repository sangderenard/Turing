// An if/else merge and a body-scoped binding inside a loop body.
/** @param {number} n @param {boolean} p */
function root(n, p) { let a = 1; let b = 0; for (let i = 0; i < n; i++) { if (p) { a = a + b; } else { b = b + a; } let t = a; a = b + 1; b = t; } return a - b; }
// INPUTS [[0, true], [3, true], [4, false], [7, true]]
