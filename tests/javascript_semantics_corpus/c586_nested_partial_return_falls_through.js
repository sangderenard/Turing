/** @param {number} n @param {boolean} p */
function root(n, p) { let x = n; if (p) { if (n > 1) return 1; x = 2; } else { if (n < -1) { return -1; } else { x -= 3; } x *= 2; } return x + 10; }
// INPUTS [[0, true], [5, true], [-5, true], [0, false], [5, false], [-5, false], [NaN, true], [NaN, false], [Infinity, false], [-Infinity, true]]
