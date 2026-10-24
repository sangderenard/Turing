// Math.floor/ceil/trunc/abs are exact (not implementation-approximated).
/** @param {number} x */
function root(x) { return Math.floor(x) + Math.ceil(x) * 10 + Math.trunc(x) * 100 + Math.abs(x) * 1000; }
// INPUTS [[-2.5], [2.5], [-0.5], [NaN], [1e300], [-0], [0.49999999999999994]]
