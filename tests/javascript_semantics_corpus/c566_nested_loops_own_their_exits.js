// An inner break or continue leaves only the inner loop.
/** @param {number} n @param {number} b */
function root(n, b) { let s = 0; for (let i = 0; i < 4; i++) { for (let j = 0; j < 4; j++) { if (j > i) break; if (j === 1) continue; s += j * b; } if (s > n) break; } return s; }
// INPUTS [[0, 1], [5, 2], [10, 3], [NaN, 1], [3, -1], [7, 0.5], [Infinity, 2]]
