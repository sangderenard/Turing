/** @param {number} n @param {boolean} p */
function root(n, p) { if (p) { let t = n + 1; if (t > 2) return t; } let t = 5 * n; { let u = t; if (u > 9) { return -u; } } let u = 1; return t + u; }
// INPUTS [[0, true], [3, true], [1, false], [2, false], [NaN, true], [-Infinity, false]]
