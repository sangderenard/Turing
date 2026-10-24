// A for loop carries s and i as Phis; the trip count follows a Number (NaN runs zero times).
/** @param {number} n */
function root(n) { let s = 0; for (let i = 0; i < n; i++) { s += i * i; } return s; }
// INPUTS [[0], [1], [5], [10.5], [-3], [NaN]]
