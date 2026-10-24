// A carried boolean keeps its kind around the back edge.
/** @param {number} n */
function root(n) { let f = true; let c = 0; while (c < n) { f = !f; c += 1; } return f; }
// INPUTS [[0], [1], [2], [5]]
