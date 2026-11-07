// Node check of web/tipjar.js's pure parts: the QR code's Reed-Solomon codewords, format bits and structure, and the amount formatting.
import { rsEcc, formatBits, qrMatrix, btc, withCommas, hsl } from "../examples/meowdoku/web/tipjar.js";

const fail = (m) => { console.error("FAIL " + m); process.exit(1); };
const eq = (a, b, m) => { if (JSON.stringify(a) !== JSON.stringify(b)) fail(`${m}: ${JSON.stringify(a)} != ${JSON.stringify(b)}`); };

// the ISO 18004 annex: "01234567" (version 1-M) data codewords 16 32 12 86 97 85 ... ; the published RS example: data 32 91 11 120 209 114 220 77 67 64 236 17 236 17 236 17
// with 10 ecc codewords is 196 35 39 119 235 215 231 226 93 23 (the HELLO WORLD 1-M example of the standard's tutorials)
eq(rsEcc([32, 91, 11, 120, 209, 114, 220, 77, 67, 64, 236, 17, 236, 17, 236, 17], 10), [196, 35, 39, 119, 235, 215, 231, 226, 93, 23], "RS ecc of HELLO WORLD");
// format information for ECC L, masks 0..7 (the standard's table)
const L = ["111011111000100", "111001011110011", "111110110101010", "111100010011101", "110011000101111", "110001100011000", "110110001000001", "110100101110110"];
L.forEach((bits, mask) => eq(formatBits(mask).toString(2).padStart(15, "0"), bits, "format bits mask " + mask));

const m = qrMatrix("bitcoin:bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4");
eq(m.length, 29, "a 50-byte payload is version 3");
for (let i = 8; i < 21; i++) { eq(m[6][i], i % 2 === 0, "timing row " + i); eq(m[i][6], i % 2 === 0, "timing col " + i); }
for (const [x, y] of [[0, 0], [22, 0], [0, 22]]) { eq(m[y + 3][x + 3], true, "finder centre"); eq(m[y + 1][x + 1], false, "finder ring"); eq(m[y][x], true, "finder corner"); }
// [QRWALL] every finder is the full 7 x 7 ring (the timing lines are drawn UNDER the finders; they once cut a gap in each finder's edge and no scanner could read the code),
// with its light separator
const finder = ["#######", "#.....#", "#.###.#", "#.###.#", "#.###.#", "#.....#", "#######"];
for (const [fx, fy] of [[0, 0], [22, 0], [0, 22]]) for (let dy = 0; dy < 7; dy++) eq(m[fy + dy].slice(fx, fx + 7).map((v) => (v ? "#" : ".")).join(""), finder[dy], `finder row ${dy} at ${fx},${fy}`);
for (let i = 0; i < 8; i++) { eq(m[7][i], false, "separator below the top-left finder"); eq(m[i][7], false, "separator right of the top-left finder"); }
eq(m[29 - 8][8], true, "dark module");
eq(m[22][22], true, "alignment centre");
eq(qrMatrix("a").length, 21, "version 1");
eq(btc(16515729), "0.16515729", "btc");
eq(withCommas(16515729), "16,515,729", "commas");
eq(hsl(0, 1, 0.5), [255, 0, 0], "hsl red");
console.log("ok");
