// return inside a loop (and inside a nested loop) leaves the function with
// the value at that point; otherwise the code after the loop runs.
/** @param {number} n @param {number} b */
function root(n, b) { let s = 0; for (let i = 0; i < 4; i++) { for (let j = 0; j < 4; j++) { if (i * j * b > n) return i * 10 + j; } s += b; if (s > 6) return -s; } return 99 + s; }
// INPUTS [[0, 1], [5, 2], [10, 3], [NaN, 1], [3, -1], [7, 0.5], [Infinity, 2]]
