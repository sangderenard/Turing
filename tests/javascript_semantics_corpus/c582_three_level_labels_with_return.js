// Exits to three different loops beside a return, and a labeled while.
/** @param {number} n @param {number} b */
function root(n, b) { let s = 0; let c = 0; a: for (let i = 0; i < 4; i++) { m: for (let j = 0; j < 4; j++) { for (let k = 0; k < 4; k++) { if (k > j) continue m; if (i + j + k > n) break a; if (s * b > 60) return -s; s += 1; } s += 10; } s += 100; } lp: while (c < 30) { c++; s += b; if (s > n * 50) break lp; if (s > 3) continue lp; s += 1; } return s * 100 + c; }
// INPUTS [[0, 1], [5, 2], [10, 3], [NaN, 1], [3, -1], [7, 0.5], [Infinity, 2]]
