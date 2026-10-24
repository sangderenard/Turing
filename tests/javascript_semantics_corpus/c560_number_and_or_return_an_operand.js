// && and || on Numbers return one of the operands, chosen by the left
// operand's truthiness (0, -0 and NaN are falsy): never a boolean.
/** @param {number} a @param {number} b */
function root(a, b) { return (a && b) + 2 * (a || b); }
// INPUTS [[5, 3], [0, 3], [-0, 3], [NaN, 1], [1, NaN], [0, -0], [-0, 0], [NaN, 0], [Infinity, 0], [-1, -0]]
