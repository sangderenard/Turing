// Reads JavaScript source on stdin, prints its acorn ESTree AST as JSON on stdout.
// Vendored alongside acorn.js so Dream JavaScript sections can be parsed into
// a real dependency graph without a network/package-manager dependency.
"use strict";
const acorn = require("./acorn.js");

let source = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (chunk) => { source += chunk; });
process.stdin.on("end", () => {
    try {
        const ast = acorn.parse(source, {
            ecmaVersion: "latest",
            sourceType: "module",
            locations: false,
        });
        process.stdout.write(JSON.stringify(ast, (key, value) => (
            typeof value === "bigint" ? value.toString() : value
        )));
    } catch (error) {
        process.stderr.write(String(error && error.message || error));
        process.exit(1);
    }
});
