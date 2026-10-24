// A loop inside one if arm; its exit Phi meets the other arm in a Select.
/** @param {number} n @param {boolean} p */
function root(n, p) { let s = 1; if (p) { for (let i = 0; i < n; i++) { s *= 2; } } else { s = -1; } return s; }
// INPUTS [[3, true], [3, false], [0, true], [NaN, true]]
