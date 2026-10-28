/** @param {number} n @param {number} m */
function root(n, m) { let s = 0; switch (n) { case 1: s += 1; case 2: s += 2; break; default: s += 100; case 3: s += 3; if (m > 0) break; s += 30; case 4: { s += 4; } } for (let i = 0; i < 4; i++) { switch (i % 3) { case 0: if (m < 0) continue; s += 10; break; case m: s += 1000; break; default: if (s > 5000) return -s; } s += 0.5; } switch (m) { } switch (m) { default: s *= 2; } return s; }
// INPUTS [[1, 1], [2, -1], [3, 1], [3, -1], [4, 0], [5, 2], [NaN, NaN], [-0, 0], [0, 1], [Infinity, 2]]
