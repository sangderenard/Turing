// if/else assigning one let: the two versions meet in a Select on the condition.
/** @param {number} x */
function root(x) { let y = 0; if (x > 1) { y = x * 2; } else { y = -x; } return y + 1; }
// INPUTS [[0], [2], [NaN], [-0], [1]]
