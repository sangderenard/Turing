// **= goes through ECMAScript exponentiate (approximated: ulp tier).
/** @param {number} x */
function root(x) { let y = x; y **= 2; y /= 4; return y; }
// INPUTS [[2], [NaN], [-3], [1e200]]
