const $ = id => document.getElementById(id);
const account = () => encodeURIComponent($("agent").value.trim());
const path = suffix => `/api/hosted-agents/${account()}/${suffix}`;
async function api(url, body) {
  const response = await fetch(url, {method: body ? "POST" : "GET", credentials: "same-origin", cache: "no-store",
    headers: body ? {"Content-Type": "application/json"} : {}, body: body ? JSON.stringify(body) : undefined});
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || `HTTP ${response.status}`);
  return value;
}
async function action(work) {
  try { await work(); $("status").textContent = "Saved or loaded successfully."; }
  catch (error) { $("status").textContent = error.message; }
}
function scope() {
  $("key").disabled = $("scope").value === "global";
  $("mode").querySelector('[value="inherit"]').disabled = $("scope").value === "global";
  if ($("scope").value === "global") { $("key").value = ""; if ($("mode").value === "inherit") $("mode").value = "off"; }
}
function modePath() {
  return path("last-word") + "?" + new URLSearchParams({scope:$("scope").value, key:$("key").value.trim()});
}
$("load").onclick = () => action(async () => {
  const value = await api(path("controls")); $("enabled").checked = value.enabled; $("goal").value = value.goal;
});
$("save").onclick = () => action(async () => {
  await api(path("controls"), {enabled:$("enabled").checked, goal:$("goal").value});
});
$("scope").onchange = scope;
$("load-mode").onclick = () => action(async () => {
  const value = await api(modePath()); $("mode").value = value.mode === "inherit" && $("scope").value === "global" ? "off" : value.mode;
  $("effective").textContent = `Effective: ${value.effective ? "on" : "off"} (server-wide: ${value.global_mode})`;
});
$("save-mode").onclick = () => action(async () => {
  const value = await api(path("last-word"), {scope:$("scope").value, key:$("key").value.trim(), mode:$("mode").value});
  $("effective").textContent = `Effective: ${value.effective ? "on" : "off"} (server-wide: ${value.global_mode})`;
});
scope(); $("load").click();
