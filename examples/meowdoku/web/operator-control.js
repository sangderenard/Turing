const byId = id => document.getElementById(id);
let state;
function show(message, bad=false) {
  const box=byId("notice"); box.textContent=message; box.style.color=bad?"#b00020":"";
}
function node(tag, text, attrs={}) {
  const el=document.createElement(tag); el.textContent=text;
  for(const [key,value] of Object.entries(attrs)) el.setAttribute(key,value);
  return el;
}
function selected(id) { return Number(byId(id).value); }
async function refresh() {
  const res=await fetch("/api/operator-control",{credentials:"same-origin",cache:"no-store"});
  const body=await res.json().catch(()=>({error:"Could not read the control service."}));
  if(!res.ok) throw Error(body.error||"HTTP "+res.status);
  state=body;
  render();
}
async function act(data) {
  const password=byId("password").value;
  byId("password").value="";
  if(!password) { show("Enter your account password for this request.",true); return; }
  try {
    const res=await fetch("/api/operator-control",{method:"POST",credentials:"same-origin",
      headers:{"Content-Type":"application/json","X-Control-CSRF":state.csrf},
      body:JSON.stringify({...data,password})});
    const body=await res.json().catch(()=>({error:"No response from control service."}));
    if(!res.ok) throw Error(body.error||"HTTP "+res.status);
    await refresh();
    show(data.action==="run"?"Queued run "+body.job_id:"Saved "+data.action+".");
  } catch(e) { show(e.message,true); }
}
function render() {
  byId("create-section").hidden=!state.admin;
  byId("grant-section").hidden=!state.admin;
  const scripts=byId("script"); scripts.replaceChildren();
  for(const name of state.scripts) scripts.append(node("option",name,{value:name}));
  const grants=byId("grant-preset"); grants.replaceChildren();
  const presets=byId("presets"); presets.replaceChildren();
  for(const item of state.presets) {
    const card=node("div","",{class:"card"});
    card.append(node("strong",item.label+" "),node("small",item.script+" · timeout "+item.timeout+"s"));
    const row=node("div","",{class:"row"});
    const run=node("button","Run",{type:"button"});
    run.addEventListener("click",()=>act({action:"run",preset_id:item.id}));
    row.append(run);
    if(state.admin) {
      grants.append(node("option",item.label+" ("+item.script+")",{value:String(item.id)}));
      const disable=node("button","Disable",{type:"button"});
      disable.addEventListener("click",()=>{if(confirm("Disable this action and revoke its grants?")) act({action:"disable",preset_id:item.id});});
      row.append(disable);
      const args=node("small","Arguments: "+JSON.stringify(item.argv)); card.append(args);
    }
    card.append(row); presets.append(card);
  }
  if(!state.presets.length) presets.append(node("p","No actions available."));
  const grantBox=byId("grants"); grantBox.replaceChildren();
  if(state.admin) for(const item of state.grants) {
    const row=node("div","Grant "+item.id+" · account "+item.user_id+" · action "+item.preset_id+
      " · expires "+new Date(item.expires*1000).toLocaleString()+(item.revoked?" · revoked":""),{class:"card"});
    if(!item.revoked && item.expires>Date.now()/1000) {
      const revoke=node("button","Revoke",{type:"button"});
      revoke.addEventListener("click",()=>act({action:"revoke",grant_id:item.id}));
      row.append(revoke);
    }
    grantBox.append(row);
  }
  const jobs=byId("jobs"); jobs.replaceChildren();
  for(const item of state.jobs) jobs.append(node("div",
    "Run "+item.id+" · action "+item.preset_id+" · "+item.status+
    (item.exit_code===null?"":" · exit "+item.exit_code)+
    (item.note?" · "+item.note:""),{class:"card"}));
  if(!state.jobs.length) jobs.append(node("p","No runs yet."));
}
byId("create").addEventListener("click",()=>{
  let argv;
  try { argv=JSON.parse(byId("argv").value); } catch { show("Arguments must be a JSON array.",true); return; }
  act({action:"preset",label:byId("label").value,script:byId("script").value,argv,timeout:selected("timeout")});
});
byId("grant").addEventListener("click",()=>act({action:"grant",username:byId("operator").value,
  preset_id:selected("grant-preset"),hours:Number(byId("hours").value)}));
refresh().catch(e=>show(e.message,true));
setInterval(()=>refresh().catch(()=>{}),15000);
