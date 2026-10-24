// Number.isNaN on a computed value; 0/0 and Infinity/Infinity are NaN.
/** @param {number} x */
function root(x) { return Number.isNaN(x / x); }
// INPUTS [[0], [1], [Infinity], [-0], [NaN]]
