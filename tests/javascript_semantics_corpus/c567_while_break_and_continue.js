// A while loop with both exits; continue before the counter's last use.
/** @param {number} n @param {number} b */
function root(n, b) { let s = 0; let i = 0; while (i < n) { i++; if (i === 3) continue; if (s > 10) break; s += i * b; } return s + i; }
// INPUTS [[0, 1], [5, 2], [10, 3], [NaN, 1], [3, -1], [7, 0.5], [Infinity, 2]]
