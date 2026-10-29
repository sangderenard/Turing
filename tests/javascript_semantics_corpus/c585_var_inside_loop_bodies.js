/** @param {number} n @param {number} b */
function root(n, b) { var s = 0, c = 1; for (var i = 0; i < n; i++) { var sq = i * i; if (sq > b) { var over = sq - b; s += over; continue; } s += sq; } var j = 0; while (j < 4) { var step = j % 2 ? b : -b; c = c * 2 + step; j++; } return s * 1000 + c; }
// INPUTS [[0, 1], [5, 3], [10, 20], [NaN, 1], [3, -1], [7, 0.5], [4, Infinity]]
