// A parameter is reassignable; a Number condition (y) takes JS truthiness.
/** @param {number} x @param {boolean} p */
function root(x, p) { let y = x; if (p) { y = y + 10; } if (y) { y--; } return y; }
// INPUTS [[0, true], [0, false], [NaN, true], [-10, true], [1, false]]
