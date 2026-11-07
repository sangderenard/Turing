// The game and log pages share the same saved-setting reset. Session identity,
// recovery copy, and client diagnostics are retained for testing.
import { groupKey } from "./log_view.js";

export function settingsKeys(storage = localStorage) {
  return Object.keys(storage).filter((key) =>
    groupKey(key) !== "other" && !["player (identity)", "session copy", "local log"].includes(groupKey(key)));
}

export function resetSettings(storage = localStorage) {
  const keys = settingsKeys(storage);
  for (const key of keys) storage.removeItem(key);
  return keys.length;
}

const button = document.getElementById("settings-reset");
if (button) button.addEventListener("click", () => {
  resetSettings();
  location.reload();
});
