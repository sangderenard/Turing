/** @param {number} n @param {boolean} p */
function root(n, p) { let s = 0; let i = 0; do { s += i * n; i++; if (s > 50) break; if (p && i === 2) continue; s += 1; } while (i < 5 && n > 0); let k = 10; d: do { k -= 3; for (let j = 0; j < 3; j++) { if (k < 0) break d; if (j > k) continue d; s += j; } } while (k > 0); do { if (n > 100) return -1; } while (false); return s * 100 + k; }
// INPUTS [[0, true], [1, false], [3, true], [-2, false], [NaN, true], [7, false], [1000, true], [Infinity, false]]
