// Every ordered comparison with NaN is false, so both tests fail for NaN.
/** @param {number} a @param {number} b */
function root(a, b) { return a < b ? 1 : (a >= b ? 2 : 3); }
// INPUTS [[1, 2], [2, 1], [NaN, 1], [1, NaN], [-0, 0]]
