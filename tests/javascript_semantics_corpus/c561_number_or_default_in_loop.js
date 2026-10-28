// The "x || default" idiom on a carried Number; -0 and NaN take the default.
/** @param {number} a @param {number} b */
function root(a, b) { let x = a; for (let i = 0; i < 3; i++) { x = (x - i) || b; } return x; }
// INPUTS [[5, 3], [0, 3], [-0, 3], [NaN, 1], [1, NaN], [0, -0], [-0, 0], [NaN, 0], [Infinity, 0], [-1, -0]]
