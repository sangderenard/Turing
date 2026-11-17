const $ = id => document.getElementById(id);
const node = (tag, text) => { const e = document.createElement(tag); if (text !== undefined) e.textContent = String(text); return e; };
const status = message => $("status").textContent = message;
async function request(path, body) {
  const response = await fetch(path, {credentials:"same-origin",cache:"no-store",...(body === undefined ? {} :
    {method:"POST",headers:{"Content-Type":"application/json","X-Meowdoku-Arena":"1"},body:JSON.stringify(body)})});
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || `Request failed (${response.status})`);
  return value;
}
function team(name = "") {
  const box = node("div"); box.className = "team";
  const heading = node("h3", "Team");
  const title = node("label", "Team name"); const input = node("input"); input.maxLength = 120; input.value = name; title.append(input);
  const first = node("label", "Prompt crafter"); const a = node("select"); for (const agent of ["codex","claude"]) a.append(new Option(agent,agent)); first.append(a);
  const second = node("label", "Communicator"); const b = node("select"); for (const agent of ["same agent","codex","claude"]) b.append(new Option(agent,agent === "same agent" ? "" : agent)); second.append(b);
  const state = node("label"); const check = node("input"); check.type = "checkbox"; check.checked = true; state.append(check," Stateful instances");
  const remove = node("button","Remove team"); remove.type = "button"; remove.onclick = () => { if ($("teams").children.length > 2) box.remove(); else status("Keep at least two teams."); };
  box.append(heading,title,first,second,state,remove); $("teams").append(box);
  return {box,input,a,b,check};
}
const teams = [team("Team 1"),team("Team 2")];
let nextTeam = 3;
$("add-team").onclick = () => teams.push(team(`Team ${nextTeam++}`));
function specification() {
  let parameters; try { parameters = JSON.parse($("parameters").value); } catch { throw new Error("Runtime parameters must be JSON."); }
  if (!parameters || Array.isArray(parameters) || typeof parameters !== "object") throw new Error("Runtime parameters must be a JSON object.");
  const selected = teams.filter(t => t.box.isConnected).map(t => ({name:t.input.value.trim(),agents:[t.a.value,...(t.b.value ? [t.b.value] : [])],stateful:t.check.checked}));
  return {source:{kind:$("source-kind").value,id:$("source-id").value.trim()},goal:$("goal").value.trim(),
    target:{agent:$("target").value,stateful:$("target-stateful").checked},teams:selected,judge:$("judge").value,
    training_mode:$("training-mode").checked,runtime:{turn_seconds:Number($("seconds").value),model:$("model").value,parameters}};
}
$("submit").onclick = async () => { try { $("submit").disabled = true; const run = await request("/api/agent-training-arena",specification()); status(`Queued run ${run.id}.`); await listing(); }
  catch(error){status(error.message)} finally {$("submit").disabled = false} };
async function sources() {
  const kind = $("source-kind").value; $("source-select").replaceChildren(new Option("Choose a real source",""));
  try {
    const data = kind === "ticket" ? await request("/api/issues?scope=mine&status=all&offset=0") : await request("/api/research/cases");
    for (const row of kind === "ticket" ? data.tickets : data.cases) $("source-select").append(new Option(`${row.title} · ${row.id}`,row.id));
  } catch(error) { status(error.message); }
}
$("source-kind").onchange = () => { $("source-id").value = ""; sources(); };
$("source-select").onchange = () => { $("source-id").value = $("source-select").value; };
function showText(parent,label,value) { const details = node("details"); details.append(node("summary",label),node("pre",typeof value === "string" ? value : JSON.stringify(value,null,2))); parent.append(details); }
async function detail(id) {
  try {
    const run = await request(`/api/agent-training-arena/${id}`); const box = $("detail"); box.replaceChildren();
    box.append(node("h2",`Run ${id}`),node("p",`${run.status} · ${run.specification.source.kind} ${run.specification.source.id} · ${run.specification.training_mode ? "training mode" : "evaluation mode"}`));
    if (run.error) box.append(node("p",run.error));
    showText(box,"Shared goal and runtime",{goal:run.specification.goal,runtime:run.specification.runtime});
    if (run.result?.teams?.length) {
      const board = node("ol");
      for (const row of [...run.result.teams].sort((a,b) => Object.values(b.scores).reduce((x,y)=>x+y,0) - Object.values(a.scores).reduce((x,y)=>x+y,0))) {
        board.append(node("li",`${row.name}: ${Object.values(row.scores).reduce((x,y)=>x+y,0)}/20`));
      }
      box.append(node("h3","Team scores"),board);
    }
    for (const row of run.result?.teams || []) {
      const section = node("section"); section.className = "receipt"; section.append(node("h3",row.name));
      const scores = node("div"); scores.className = "scores";
      for (const [axis,score] of Object.entries(row.scores || {})) scores.append(node("span",`${axis.replaceAll("_"," ")}: ${score}/4`));
      section.append(scores); showText(section,"Prompt override",row.system_prompt); showText(section,"First message",row.first_message);
      showText(section,"Target result",row.response); showText(section,"Scores for each piece",row.pieces || row.reasons); showText(section,"Demonstrated tests",row.test_evidence);
      box.append(section);
    }
  } catch(error) { status(error.message); }
}
async function listing() { try { const data = await request("/api/agent-training-arena"); $("runs").replaceChildren();
    for (const row of data.runs) { const button = node("button",`${row.status} · ${new Date(row.created*1000).toLocaleString()} · ${row.id}`); button.type="button"; button.onclick=()=>detail(row.id); $("runs").append(button); }
    if (!data.runs.length) $("runs").append(node("p","No arena runs yet."));
  } catch(error) {status(error.message)} }
$("refresh").onclick = listing;
sources(); listing();
