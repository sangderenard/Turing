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
$("refresh").addEventListener("click", refreshAccount);
void refreshAccount();
