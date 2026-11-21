const byId = id => document.getElementById(id);
const add = (parent, tag, value, cls) => { const node = document.createElement(tag); node.textContent = value; if (cls) node.className = cls; parent.append(node); return node; };
const date = value => value ? new Date(value * 1000).toLocaleString() : "—";

async function load() {
  const queue = byId("queue"); queue.replaceChildren();
  try {
    const response = await fetch("/api/account/admin/look-queue", { credentials: "same-origin", cache: "no-store" });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    byId("status").textContent = `${data.previews.length} temporary previews. Full Apache access log: ${data.access_log}`;
    for (const preview of data.previews) {
      const card = add(queue, "section", "", "card");
      add(card, "h2", `${preview.username} · ${preview.state}`);
      add(card, "p", `ID ${preview.id} · created ${date(preview.created_at)} · preview through ${date(preview.expires_at)} · frozen ${date(preview.frozen_at)}`, "muted");
      const entries = (data.recent_access || {})[preview.username] || [];
      if (entries.length) {
        const list = add(card, "ul", "");
        for (const item of entries.slice(-10).reverse()) add(list, "li", `${date(item.ts)} · ${item.event} · ${item.ip}`);
      } else add(card, "p", "No recent account-linked IP entries in the bounded log.", "muted");
      if (preview.state === "frozen") {
        const button = add(card, "button", "Wipe frozen preview"); button.type = "button";
        button.onclick = async () => {
          const name = prompt(`Type ${preview.username} to confirm the wipe:`);
          if (name !== preview.username) return;
          const password = prompt("Enter your admin account password to confirm:");
          if (!password) return;
          button.disabled = true;
          try {
            const result = await fetch("/api/account/admin/look-wipe", { method: "POST", credentials: "same-origin", cache: "no-store",
              headers: { "Content-Type": "application/json", "X-Meowdoku-Look-Wipe": "1" },
              body: JSON.stringify({ user_id: preview.id, username: name, password }) });
            const body = await result.json();
            if (!result.ok) throw new Error(body.error || `HTTP ${result.status}`);
            await load();
          } catch (error) { byId("status").textContent = error.message; button.disabled = false; }
        };
      }
    }
  } catch (error) { byId("status").textContent = error.message; }
}
byId("refresh").onclick = load;
load();
