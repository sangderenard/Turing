// %= rebinds a let; a loop of remainders stays exact.
/** @param {number} a @param {number} b */
function root(a, b) { let x = a; for (let i = 0; i < 3; i++) { x %= b; x += i; } return x; }
// INPUTS [[5, 3], [-5, 3], [5, -3], [-0, 3], [0, -3], [5, 0], [NaN, 1], [1, NaN], [Infinity, 2], [2, Infinity], [-2, -Infinity], [-5.5, 2], [1e308, 1e-308], [0.1, 0.03], [-0, -0], [0, -0], [-0, 0], [-1e-308, 7]]
