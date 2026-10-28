/** @param {number} a @param {number} b */
function root(a, b) { var s = 0; for (var i = 0; i < 5; i++) { var t = a * i; s += t; var s = s + b; } var a = a + i; return s + a; }
// INPUTS [[1, 2], [0, 0], [-3, 0.5], [NaN, 1], [Infinity, -Infinity], [1e308, 1e308], [-0, -0]]
