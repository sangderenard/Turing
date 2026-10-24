// Math.sign returns +-0 and NaN unchanged.
/** @param {number} a @param {number} b */
function root(a, b) { return 1 / Math.sign(a) + Math.sign(b); }
// INPUTS [[5, 3], [-5, 3], [5, -3], [-0, 3], [0, -3], [5, 0], [NaN, 1], [1, NaN], [Infinity, 2], [2, Infinity], [-2, -Infinity], [-5.5, 2], [1e308, 1e-308], [0.1, 0.03], [-0, -0], [0, -0], [-0, 0], [-1e-308, 7]]
