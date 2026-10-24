// Compound assignment and ++ rebind a let; each version is a new value.
/** @param {number} a @param {number} b */
function root(a, b) { let s = a; s += b; s *= s; s -= a / b; s++; return s; }
// INPUTS [[1, 2], [3, 0], [-0, 5], [NaN, 1], [0.1, 0.2]]
