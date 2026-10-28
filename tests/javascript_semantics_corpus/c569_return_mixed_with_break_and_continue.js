// return, break and continue in one loop; a loop-local let is returned.
/** @param {number} n @param {number} b */
function root(n, b) { let s = 0; for (let i = 0; i < n && i < 20; i++) { if (i === 2) continue; let t = s * b; if (t > 30) break; s += i; if (t > 9) return t; } let i = 0; while (true) { i++; if (i > n || i > 5) return s + i; } }
// INPUTS [[0, 1], [5, 2], [10, 3], [NaN, 1], [3, -1], [7, 0.5], [Infinity, 2]]
