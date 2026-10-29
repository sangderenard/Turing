/** @param {number} n @param {boolean} p */
function root(n, p) { var x = n; if (p) { var x = x * 2; { var x = x + 1; } } else { x -= 3; } var k = 0; while (k < 3) { var k = k + 1; x += k; } return x; }
// INPUTS [[1, true], [1, false], [NaN, true], [-0, false], [Infinity, true], [0.1, false]]
