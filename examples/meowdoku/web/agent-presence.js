// Observed agent activity. A square means a recorded signal; an empty square never claims offline.
const node = (tag, text, cls) => { const n = document.createElement(tag); if (text != null) n.textContent = text; if (cls) n.className = cls; return n; };
export const CHANNELS = [["api_in", "API in"], ["mcp_in", "MCP in"], ["api_out", "API out"], ["mcp_out", "MCP out"], ["local", "Local"]];
export const ago = at => at == null ? "time unknown" : new Date(at * 1000).toLocaleString();
export const remaining = (due, now) => {
  if (due == null) return "schedule unknown";
  const seconds = Math.max(0, Math.ceil(due - now));
  if (!seconds) return "due, waiting for the scheduler";
  return `in ${Math.floor(seconds / 3600)}h ${Math.floor(seconds % 3600 / 60)}m ${seconds % 60}s`;
};

export function renderAgentPresence(target, data, { locations = false, onLocation = null } = {}) {
  target.replaceChildren();
  if (!data || !Array.isArray(data.agents)) { target.append(node("p", "Agent status unavailable.")); return; }
  const places = new Map((data.locations || []).map(place => [place.username, place]));
  for (const agent of data.agents) {
    const row = node("div", null, "agent-presence"), head = node("div", null, "agent-head");
    head.append(node("strong", agent.username));
    const runs = agent.runs || [];
    head.append(node("span", runs.length ? runs.map(run => run.kind === "seat" ? `at ${run.room} table` : `${run.kind} run`).join(" + ") : "no validated run observed", "agent-run"));
    row.append(head);
    const channels = node("div", null, "agent-channels");
    for (const [key, label] of CHANNELS) {
      const state = agent.channels?.[key]?.state || "unknown";
      const chip = node("span", `${state === "active" || state === "recent" ? "■" : "□"} ${label}: ${state}`, `agent-channel ${state}`);
      chip.title = state === "recent" ? `Last observed ${ago(agent.channels[key].at)}` : state === "unobserved" ? "No current observation; this does not mean offline." : state === "unknown" ? "No shared observation source." : `Since ${ago(agent.channels[key].since)}`;
      channels.append(chip);
    }
    row.append(channels);
    if (agent.schedule) {
      const schedule = agent.schedule;
      const line = node("small", null, "agent-schedule");
      line.append(document.createTextNode("Routine check-in "));
      const countdown = node("span", remaining(schedule.next_due, data.as_of));
      countdown.dataset.due = String(schedule.next_due); countdown.dataset.asOf = String(data.as_of);
      countdown.dataset.readAt = String(performance.now());
      line.append(countdown);
      line.append(document.createTextNode(schedule.bell_off_until ? ` · bell paused until ${ago(schedule.bell_off_until)}` : " · no bell pause recorded"));
      line.append(document.createTextNode(" · provider cooldown unknown"));
      row.append(line);
    } else row.append(node("small", "Routine schedule unknown", "agent-schedule"));
    if (locations) {
      const place = places.get(agent.username);
      if (place) {
        const label = `#${(data.locations || []).indexOf(place) + 1} · Last checkout file: ${place.repository}/${place.path} (${ago(place.at)})`;
        if (onLocation) { const button = node("button", label, "agent-location"); button.type = "button"; button.onclick = () => onLocation(place); row.append(button); }
        else row.append(node("small", label, "agent-schedule"));
      } else row.append(node("small", "Repository location unknown", "agent-schedule"));
    }
    target.append(row);
  }
  target.append(node("p", data.note || "Squares show observed signals only.", "agent-note"));
}

export function tickAgentCountdowns(target) {
  for (const span of target.querySelectorAll("[data-due]")) {
    const now = Number(span.dataset.asOf) + (performance.now() - Number(span.dataset.readAt)) / 1000;
    span.textContent = remaining(Number(span.dataset.due), now);
  }
}
