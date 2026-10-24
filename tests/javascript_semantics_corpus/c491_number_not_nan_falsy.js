// !x on a Number: 0, -0 and NaN are falsy.  A bare LNot reads NaN as true.
/** @param {number} x */
function root(x) { return !x; }
// INPUTS [[0], [-0], [NaN], [1], [-Infinity], [5e-324]]
