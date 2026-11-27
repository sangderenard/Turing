// applyPalette on a page that already wears a full design: a later candidate whose secondary, bevel or shadow is
// omitted or invalid removes that inline custom property, so the page falls back to the stylesheet defaults instead
// of still wearing the previous design's value. Six transitions: each property x (valid -> omitted, valid -> invalid).
import assert from "node:assert/strict";

function fakeDocument() {
  const properties = new Map();
  const style = {
    setProperty: (name, value) => properties.set(name, String(value)),
    removeProperty: name => { properties.delete(name); },
    getPropertyValue: name => properties.get(name) ?? "",
  };
  const elements = new Map();
  return {
    documentElement: {style},
    getElementById: id => elements.get(id) ?? null,
    createElement: () => ({}),
    head: {append: element => elements.set(element.id, element)},
  };
}

globalThis.document = fakeDocument();
const {applyPalette} = await import(new URL("../examples/meowdoku/web/palette-theme.js", import.meta.url));

const FULL = {color: "#adff2f", secondary: "#f0fff0", bevel: "inset 0 1px 0 rgba(255,255,255,0.4)",
              shadow: "0 2px 6px rgba(0,0,0,0.5)"};
const FIELDS = {secondary: "--palette-secondary", bevel: "--palette-bevel", shadow: "--palette-shadow"};
const INVALID = {secondary: "green", bevel: "url(x);", shadow: "expression{1}"};
const root = () => document.documentElement.style;

for (const [field, property] of Object.entries(FIELDS)) {
  for (const [kind, replacement] of [["omitted", undefined], ["invalid", INVALID[field]]]) {
    globalThis.document = fakeDocument();
    applyPalette(FULL);
    for (const p of Object.values(FIELDS)) assert.notEqual(root().getPropertyValue(p), "", `${p} installed`);
    const next = {...FULL, color: "#336699"};
    if (replacement === undefined) delete next[field]; else next[field] = replacement;
    applyPalette(next);
    assert.equal(root().getPropertyValue(property), "", `${field} ${kind}: ${property} removed`);
    for (const [other, p] of Object.entries(FIELDS)) {
      if (other !== field) assert.equal(root().getPropertyValue(p), String(FULL[other]), `${field} ${kind}: ${p} kept`);
    }
    assert.equal(root().getPropertyValue("--palette-accent"), "#336699");
  }
}
console.log("palette stale values: 6 transitions ok");
