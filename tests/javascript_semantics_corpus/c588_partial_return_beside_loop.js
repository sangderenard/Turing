/** @param {number} n @param {boolean} p */
function root(n, p) { let s = 0; for (let i = 0; i < 4; i++) { s += i * n; } if (p) { if (s > 20) return s; s = -s; } let k = 0; while (k < 3) { s += k; k++; } if (!p) { if (s < 0) { return s * 2; } } return s + 0.5; }
// INPUTS [[0, true], [3, true], [1, true], [-2, false], [2, false], [NaN, true], [Infinity, false], [-Infinity, true]]
