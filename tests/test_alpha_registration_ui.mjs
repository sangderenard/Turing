import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const web = new URL("../examples/meowdoku/web/", import.meta.url);

test("successful alpha signup keeps confirmation and profile link visible", async () => {
  const html = readFileSync(new URL("register.html", web), "utf8");
  const script = readFileSync(new URL("register.js", web), "utf8");
  const formStart = html.indexOf('<form id="alpha-form"');
  const formEnd = html.indexOf("</form>", formStart);
  const statusAt = html.indexOf('id="status"');
  assert.ok(formStart >= 0 && formEnd > formStart && statusAt >= 0);

  const main = { hidden: false };
  const alpha = { hidden: false, dataset: { termsVersion: "alpha-2026-11-15" },
    addEventListener(name, handler) { if (name === "submit") this.submit = handler; } };
  const codeForm = { addEventListener() {} };
  const status = { textContent: "", className: "",
    parentElement: statusAt > formStart && statusAt < formEnd ? alpha : main,
    after(node) { node.parentElement = this.parentElement; this.next = node; } };
  const elements = {
    "code-form": codeForm, code: { value: "" }, "alpha-form": alpha,
    "alpha-submit": { disabled: false }, status, username: { value: "tester" },
    password: { value: "password123" }, password2: { value: "password123" },
    "agree-mit": { checked: true }, "agree-observation": { checked: true },
    "agree-research": { checked: true },
  };
  const document = {
    getElementById(id) { return elements[id]; },
    createElement(tagName) { return { tagName, children: [],
      appendChild(child) { child.parentElement = this; this.children.push(child); } }; },
  };
  const fetch = async () => ({ ok: true, json: async () => ({}) });
  vm.runInNewContext(script, { document, fetch, location: { origin: "http://localhost" },
    URL, URLSearchParams, encodeURIComponent });
  alpha.submit({ preventDefault() {} });
  await new Promise(resolve => setImmediate(resolve));

  const visible = element => {
    for (let current = element; current; current = current.parentElement)
      if (current.hidden) return false;
    return true;
  };
  assert.equal(alpha.hidden, true);
  assert.match(status.textContent, /account is ready/);
  assert.equal(visible(status), true);
  const link = status.next?.children[0];
  assert.equal(link?.href, "/profile.html");
  assert.equal(visible(link), true);
});

test("a visitor can start a preview and keep it with a chosen password", async () => {
  const script = readFileSync(new URL("register.js", web), "utf8");
  const calls = [];
  const alpha = { hidden: false, dataset: { termsVersion: "alpha-2026-11-15" },
    addEventListener(name, handler) { if (name === "submit") this.submit = handler; } };
  const previewStart = { hidden: false, disabled: false,
    addEventListener(name, handler) { if (name === "click") this.click = handler; } };
  const status = { textContent: "", className: "", after() {} };
  const elements = {
    "code-form": { addEventListener() {} }, code: { value: "" }, "alpha-form": alpha,
    "alpha-submit": { disabled: false, textContent: "" }, status,
    "preview-panel": { hidden: false }, "preview-start": previewStart,
    "preview-expiry": { hidden: true, textContent: "" },
    "alpha-heading": { textContent: "" }, username: { value: "keeper" },
    password: { value: "unique-password-123" }, password2: { value: "unique-password-123" },
    "agree-mit": { checked: true }, "agree-observation": { checked: true },
    "agree-research": { checked: true },
  };
  const document = {
    getElementById(id) { return elements[id]; },
    createElement(tagName) { return { tagName, appendChild() {} }; },
  };
  const fetch = async (path, options) => {
    calls.push({ path, options });
    const body = path === "/api/account/me" ? { account: null }
      : path === "/api/account/preview" ? { account: { temporary: true, expires_at: 1795400000 } }
      : { account: { username: "keeper" } };
    return { ok: true, json: async () => body };
  };
  vm.runInNewContext(script, { document, fetch, location: { origin: "http://localhost" },
    URL, URLSearchParams, encodeURIComponent, Date });
  await new Promise(resolve => setImmediate(resolve));
  previewStart.click();
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(calls[1].path, "/api/account/preview");
  assert.equal(calls[1].options.headers["X-Meowdoku-Preview"], "1");
  assert.equal(elements["preview-expiry"].hidden, false);
  assert.match(elements["alpha-heading"].textContent, /Keep/);
  alpha.submit({ preventDefault() {} });
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(calls[2].path, "/api/account/keep");
  assert.equal(calls[2].options.headers["X-Meowdoku-Preview"], "1");
  assert.equal(JSON.parse(calls[2].options.body).password, "unique-password-123");
  assert.equal(alpha.hidden, true);
});
