// break outer leaves both loops with the values at the break.
/** @param {number} n @param {number} b */
function root(n, b) { let s = 0; outer: for (let i = 0; i < 5; i++) { for (let j = 0; j < 5; j++) { if (i * j * b > n) break outer; s += j; } s += 100; } return s; }
// INPUTS [[0, 1], [5, 2], [10, 3], [NaN, 1], [3, -1], [7, 0.5], [Infinity, 2]]
