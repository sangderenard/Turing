// Reuse the account and Keys status routes. No provider credentials are read.
const $ = (id) => document.getElementById(id);
let pending = false;
async function json(path) {
  const response = await fetch(path, { credentials: "same-origin", cache: "no-store", redirect: "error" });
  if (!response.ok) throw new Error("status unavailable");
  return response.json();
}
async function refreshAccount() {
  if (pending) return;
  pending = true;
  $("refresh").disabled = true;
  resetSuno();
  $("sign-in").hidden = true;
  $("paper-actions").hidden = true;
  $("paper-status").textContent = "Sign in to check the credential stored for your account.";
  $("identity").textContent = "Checking your site account.";
  try {
    const who = await json("/api/whoami");
    if (who.signed_in !== true || typeof who.username !== "string") {
      $("identity").textContent = "You are not signed in to this site.";
      $("sign-in").hidden = false;
      return;
    }
    $("identity").textContent = "Site account: " + who.username + ".";
    $("paper-actions").hidden = false;
    sunoSignedIn = true;
    void refreshSuno();
    try {
      const paper = await json("/api/user-secrets/alpaca-paper");
      if (typeof paper.configured !== "boolean") throw new Error("invalid status");
      $("paper-status").textContent = paper.configured ? "Paper credential stored for this account." : "No paper credential stored for this account.";
    } catch {
      $("paper-status").textContent = "Unable to check the paper connection. No connection was changed.";
    }
  } catch {
    $("identity").textContent = "Unable to check your site account. No connection was changed.";
  } finally {
    pending = false;
    $("refresh").disabled = false;
  }
}



// Suno is a real session-owned backend connection, not a browser token store.
let sunoSignedIn = false, sunoBusy = false, sunoEpoch = 0, sunoState = "disconnected";
let sunoTools = [], sunoTimer;
const sunoBase = "/api/connections/suno/";
const workingStates = ["connecting", "authorization_required", "finishing"];
function resetSuno() {
  sunoEpoch++; sunoSignedIn = false; sunoBusy = false;
  clearTimeout(sunoTimer);
  sunoTools = []; $("suno-tools").hidden = true;
  $("suno-authorize").hidden = true; $("suno-authorize").removeAttribute("href");
  $("suno-schema").textContent = ""; $("suno-result").textContent = "";
  $("suno-arguments").value = "{}";
  $("suno-status").textContent = "Sign in to manage Suno.";
  sunoState = "disconnected"; updateSunoButtons();
}
function updateSunoButtons() {
  const locked = !sunoSignedIn || sunoBusy;
  $("suno-connect").disabled = locked || workingStates.includes(sunoState) || sunoState === "connected";
  $("suno-disconnect").disabled = locked || sunoState === "disconnected";
  $("suno-refresh").disabled = locked;
  $("suno-tools-refresh").disabled = locked || !["connected", "authorized"].includes(sunoState);
  $("suno-call").disabled = locked || !sunoTools.length || !["connected", "authorized"].includes(sunoState);
}
async function sunoRequest(action, body) {
  const options = {credentials:"same-origin", cache:"no-store", redirect:"error"};
  if (body !== undefined) {
    options.method = "POST";
    options.headers = {"Content-Type":"application/json", "X-Meowdoku-Secret":"1"};
    options.body = JSON.stringify(body);
  }
  const response = await fetch(sunoBase + action, options);
  if (response.status === 401) { resetSuno(); throw Error("Sign in to this site again."); }
  if (!response.ok) throw Error("Suno request failed (HTTP " + response.status + "). Refresh status before retrying.");
  return response.json();
}
function showSunoStatus(data) {
  const messages = {
    disconnected:"Suno is not connected.",
    authorized:"Saved Suno authorization. Load tools to verify access.",
    connecting:"Connecting to Suno…",
    authorization_required:"Waiting for your approval at Suno.",
    finishing:"Completing Suno authorization…",
    connected:"Connected to Suno.",
    failed:"Suno connection failed. You can retry connecting."
  };
  if (!data || !Object.hasOwn(messages, data.status)) throw Error("Invalid Suno connection status.");
  sunoState = data.status;
  $("suno-status").textContent = messages[sunoState] + (data.error ? " " + String(data.error) : "");
  $("suno-authorize").hidden = true; $("suno-authorize").removeAttribute("href");
  if (data.authorization_url && sunoState === "authorization_required") {
    const url = new URL(data.authorization_url);
    if (url.protocol !== "https:" || url.username || url.password) throw Error("Invalid authorization destination.");
    // The backend validates discovered OAuth origins; never synthesize a login URL here.
    $("suno-authorize").href = url.href;
    $("suno-authorize").hidden = false;
  }
  if (!["connected", "authorized"].includes(sunoState)) {
    sunoTools = []; $("suno-tools").hidden = true;
    $("suno-schema").textContent = ""; $("suno-result").textContent = "";
  }
  clearTimeout(sunoTimer);
  if (workingStates.includes(sunoState)) sunoTimer = setTimeout(refreshSuno, 2000);
}
async function sunoAction(action, body, consume = showSunoStatus) {
  if (!sunoSignedIn || sunoBusy) return;
  const epoch = sunoEpoch;
  sunoBusy = true; updateSunoButtons();
  try {
    const data = await sunoRequest(action, body);
    if (epoch !== sunoEpoch) return;
    consume(data);
  } catch (error) {
    if (epoch === sunoEpoch) $("suno-status").textContent = error.message;
  } finally {
    if (epoch === sunoEpoch) { sunoBusy = false; updateSunoButtons(); }
  }
}
function refreshSuno() { return sunoAction("status"); }
function showTool() {
  const tool = sunoTools[Number($("suno-tool").value)];
  $("suno-schema").textContent = tool ? JSON.stringify(tool, null, 2) : "";
}
function showTools(data) {
  if (!data || !Array.isArray(data.tools) || data.tools.some(t => !t || typeof t.name !== "string"))
    throw Error("Invalid Suno tool catalogue.");
  sunoTools = data.tools;
  $("suno-tool").replaceChildren();
  sunoTools.forEach((tool, index) => {
    const option = document.createElement("option");
    option.value = String(index); option.textContent = tool.title || tool.name;
    $("suno-tool").appendChild(option);
  });
  $("suno-tool").value = "0"; $("suno-tools").hidden = false;
  $("suno-status").textContent = sunoTools.length ? "Suno tools loaded. Select a tool to inspect its full schema." : "Suno returned no tools.";
  showTool();
}
$("suno-connect").addEventListener("click", () => sunoAction("connect", {}));
$("suno-disconnect").addEventListener("click", () => sunoAction("disconnect", {}));
$("suno-refresh").addEventListener("click", refreshSuno);
$("suno-tools-refresh").addEventListener("click", () => sunoAction("tools", undefined, showTools));
$("suno-tool").addEventListener("change", () => { showTool(); $("suno-arguments").value = "{}"; $("suno-result").textContent = ""; });
$("suno-call").addEventListener("click", () => {
  const tool = sunoTools[Number($("suno-tool").value)];
  if (!tool || sunoBusy) return;
  let args;
  try {
    args = JSON.parse($("suno-arguments").value);
    if (!args || typeof args !== "object" || Array.isArray(args)) throw Error();
  } catch { $("suno-result").textContent = "Enter a JSON object for the arguments. Nothing was sent."; return; }
  return sunoAction("call", {name:tool.name, arguments:args}, data => {
    $("suno-result").textContent = JSON.stringify(data, null, 2);
  });
});
$("refresh").addEventListener("click", refreshAccount);
void refreshAccount();
