/** @param {number} n @param {boolean} p @param {boolean} q */
function root(n, p, q) { let x = 0; let y = n; if (p) { let t = n * 2; { if (q) { if (t > 4) return t; y = t; } y += 1; } x = y; } else { if (q) return -n; } let z = x + y; if (z > 10) { if (q) return z; z = 10; } return z - x; }
// INPUTS [[0, true, true], [3, true, true], [1, true, false], [7, true, false], [2, false, true], [2, false, false], [NaN, true, true], [-Infinity, true, false], [1e308, false, false]]
