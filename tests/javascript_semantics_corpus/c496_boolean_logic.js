// && and || on proved booleans return booleans.
/** @param {boolean} p @param {boolean} q */
function root(p, q) { return !p || (q && p); }
// INPUTS [[true, true], [true, false], [false, true], [false, false]]
