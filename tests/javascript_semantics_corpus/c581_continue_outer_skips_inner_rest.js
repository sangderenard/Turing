// continue outer ends the inner loop and the outer iteration; the outer
// update still runs.
/** @param {number} n @param {number} b */
function root(n, b) { let s = 0; outer: for (let i = 0; i < 5; i++) { for (let j = 0; j < 5; j++) { if (j > i) continue outer; s += j * b; } s += 100; } return s + n; }
// INPUTS [[0, 1], [5, 2], [10, 3], [NaN, 1], [3, -1], [7, 0.5], [Infinity, 2]]
