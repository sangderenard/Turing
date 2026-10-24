// ECMAScript % truncates and keeps the dividend's sign (-0 included);
// the repository Mod is floored, so this is not a plain Mod.
/** @param {number} a @param {number} b */
function root(a, b) { return a % b; }
// INPUTS [[5, 3], [-5, 3], [5, -3], [-0, 3], [0, -3], [5, 0], [NaN, 1], [1, NaN], [Infinity, 2], [2, Infinity], [-2, -Infinity], [-5.5, 2], [1e308, 1e-308], [0.1, 0.03], [-0, -0], [0, -0], [-0, 0], [-1e-308, 7]]
