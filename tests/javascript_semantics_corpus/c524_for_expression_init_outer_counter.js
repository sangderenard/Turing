// A for-head expression init on an outer let; the counter is read after the loop.
/** @param {number} n */
function root(n) { let i = 0; let s = 0; for (i = 1; i <= n; i += 2) { s = s + i; } return s * 1000 + i; }
// INPUTS [[0], [1], [6], [7]]
