// Math.max/min: any NaN gives NaN, max(+0, -0) is +0 and min is -0;
// C's fmax/fmin promise neither.
/** @param {number} a @param {number} b */
function root(a, b) { return Math.max(a, b) / (Math.min(b, a) === 0 ? Math.min(a, b) : 1); }
// INPUTS [[5, 3], [-5, 3], [5, -3], [-0, 3], [0, -3], [5, 0], [NaN, 1], [1, NaN], [Infinity, 2], [2, Infinity], [-2, -Infinity], [-5.5, 2], [1e308, 1e-308], [0.1, 0.03], [-0, -0], [0, -0], [-0, 0], [-1e-308, 7]]
