const byId = id => document.getElementById(id);
let state;
let selectedInboundRequest=null;
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
    show(data.action==="run"?"Queued run "+body.job_id:data.action==="agent_declare"?
      "Declared "+body.agent.name+". Install it over SSH: "+body.agent.install:"Saved "+data.action+".");
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
  renderAgents();
  renderInbound();
  const jobs=byId("jobs"); jobs.replaceChildren();
  for(const item of state.jobs) jobs.append(node("div",
    "Run "+item.id+" · action "+item.preset_id+" · "+item.status+
    (item.exit_code===null?"":" · exit "+item.exit_code)+
    (item.note?" · "+item.note:""),{class:"card"}));
  if(!state.jobs.length) jobs.append(node("p","No runs yet."));
}
const BASES={openai:"https://api.openai.com/v1",anthropic:"https://api.anthropic.com/v1",gemini:""};
const BASE_HELP={openai:"Any OpenAI-compatible Chat Completions server: OpenAI, Mistral, Groq, DeepSeek, xAI, OpenRouter, or http://localhost for Ollama or vLLM.",
  anthropic:"Anthropic Messages API.",gemini:"Gemini's base URL is fixed; leave this blank."};
function providerChanged() {
  const p=byId("agent-provider").value;
  byId("agent-base").placeholder=BASES[p]||""; byId("agent-base").disabled=p==="gemini";
  byId("agent-base-help").textContent=BASE_HELP[p]||"";
  byId("agent-api-row").hidden=p!=="openai"; if(p!=="openai") byId("agent-api").value="chat";
}
const API_HELP={chat:"Chat Completions: any OpenAI-compatible server.",
  responses:"OpenAI's Responses API: reasoning models (gpt-5.6-sol and later) use tools while reasoning only there."};
function renderAgents() {
  byId("agents-section").hidden=!state.admin;
  if(!state.admin) return;
  const providers=byId("agent-provider");
  if(!providers.options.length) {
    for(const name of state.providers) providers.append(node("option",name,{value:name}));
    for(const name of state.apis) byId("agent-api").append(node("option",name+" · "+(API_HELP[name]||""),{value:name}));
    byId("agent-tools").value=state.agent_tools.join(", ");
    providerChanged();
  }
  const box=byId("agents"); box.replaceChildren();
  for(const item of state.agents) {
    const card=node("div","",{class:"card"});
    card.append(node("strong",item.name+" "),node("small",item.provider+(item.provider==="openai"?" ("+item.api+")":"")+" · "+item.model+" · "+item.base_url+
      " · :"+String(item.minute).padStart(2,"0")+" · "+item.account+" · "+item.status+(item.note?" · "+item.note:"")));
    if(item.status!=="installed"&&item.status!=="withdrawn") {
      const cmd=node("div","",{class:"row"}); cmd.append(node("small","Install over SSH: "),node("code",item.install));
      card.append(cmd);
    }
    const row=node("div","",{class:"row"});
    if(item.status!=="installed") {
      const edit=node("button","Edit",{type:"button"});
      edit.addEventListener("click",()=>{
        byId("agent-name").value=item.name; byId("agent-provider").value=item.provider; providerChanged(); byId("agent-api").value=item.api||"chat";
        byId("agent-base").value=item.provider==="gemini"?"":item.base_url; byId("agent-model").value=item.model;
        byId("agent-tools").value=item.tools.join(", "); byId("agent-minute").value=item.minute; byId("agent-brief").value=item.brief;
      });
      row.append(edit);
    }
    if(item.status!=="withdrawn"&&item.provider==="openai") {   // the API style alone, installed or not
      const other=item.api==="responses"?"chat":"responses";
      const style=node("button","Use "+other+" API",{type:"button"});
      style.addEventListener("click",()=>{if(confirm(item.name+": speak the "+other+" API from its next run?")) act({action:"agent_api",name:item.name,api:other});});
      row.append(style);
    }
    if(item.status!=="withdrawn") {
      const withdraw=node("button","Withdraw",{type:"button"});
      withdraw.addEventListener("click",()=>{if(confirm("Stop scheduling "+item.name+"? Its accounts and key stay until removed by hand.")) act({action:"agent_withdraw",name:item.name});});
      row.append(withdraw);
    }
    card.append(row); box.append(card);
  }
  if(!state.agents.length) box.append(node("p","No external agents declared."));
}
function renderInbound() {
  byId("inbound-section").hidden=!state.admin;
  if(!state.admin) return;
  const agents=state.inbound_agents||[], pending=state.inbound_pending||[], sessions=state.inbound_sessions||[];
  const people=byId("inbound-agents"); people.replaceChildren();
  const identities=byId("inbound-identity"), prior=identities.value; identities.replaceChildren();
  for(const item of agents) {
    people.append(node("div",item.username+" · account "+item.id,{class:"card"}));
    identities.append(node("option",item.username,{value:String(item.id)}));
  }
  if(agents.some(item=>String(item.id)===prior)) identities.value=prior;
  if(!agents.length) people.append(node("p","No inbound identities yet."));
  const lookRequests=byId("inbound-look-requests"); lookRequests.replaceChildren();
  for(const item of state.inbound_requests||[]) {
    const card=node("div","Look request "+item.id+" · "+item.username+" · "+new Date(item.created*1000).toLocaleString(),{class:"card"});
    card.append(node("p",item.untrusted?.purpose||""));
    lookRequests.append(card);
  }
  if(!(state.inbound_requests||[]).length) lookRequests.append(node("p","No recent look requests."));
  const requests=byId("inbound-pending"); requests.replaceChildren();
  if(!pending.some(item=>item.nonce===selectedInboundRequest)) selectedInboundRequest=null;
  for(const item of pending) {
    const card=node("div","",{class:"card"});
    const choose=node("input","",{type:"radio",name:"inbound-request",value:item.nonce});
    choose.checked=item.nonce===selectedInboundRequest;
    choose.addEventListener("change",()=>{selectedInboundRequest=item.nonce;});
    card.append(choose,node("strong",item.client_name+" "),node("small",
      "client "+item.client_id+" · redirect "+item.redirect_uri+" · resource "+item.resource+
      " · challenge SHA-256 "+item.challenge_sha256+
      " · expires "+new Date(item.expires*1000).toLocaleTimeString()));
    requests.append(card);
  }
  if(!pending.length) requests.append(node("p","No pending MCP connections."));
  const history=byId("inbound-sessions"); history.replaceChildren();
  for(const item of sessions) {
    const card=node("div",item.username+" · client "+item.client_id+" · tools "+item.tools.join(", ")+
      " · expires "+new Date(item.expires*1000).toLocaleString()+
      (item.revoked?" · revoked":""),{class:"card"});
    if(!item.revoked&&item.expires>Date.now()/1000) {
      const revoke=node("button","Revoke now",{type:"button"});
      revoke.addEventListener("click",()=>{if(confirm("Revoke this agent connection now?")) act({action:"inbound_revoke",delegation_id:item.id});});
      card.append(revoke);
    }
    history.append(card);
  }
  if(!sessions.length) history.append(node("p","No authorized sessions."));
}
byId("inbound-create").addEventListener("click",()=>act({action:"inbound_create",username:byId("inbound-username").value.trim()}));
byId("inbound-authorize").addEventListener("click",()=>{
  if(!selectedInboundRequest) { show("Select the exact pending MCP request.",true); return; }
  const minutes=Number(byId("inbound-minutes").value);
  act({action:"inbound_authorize",nonce:selectedInboundRequest,user_id:Number(byId("inbound-identity").value),
    tools:byId("inbound-tools").value.split(",").map(t=>t.trim()).filter(Boolean),seconds:minutes*60});
});
byId("agent-provider").addEventListener("change",providerChanged);
byId("agent-declare").addEventListener("click",()=>{
  const minute=byId("agent-minute").value;
  act({action:"agent_declare",agent:{name:byId("agent-name").value.trim(),provider:byId("agent-provider").value,api:byId("agent-api").value,
    base_url:byId("agent-base").value.trim(),model:byId("agent-model").value.trim(),
    tools:byId("agent-tools").value.split(",").map(t=>t.trim()).filter(Boolean),
    minute:minute===""?null:Number(minute),brief:byId("agent-brief").value}});
});
byId("create").addEventListener("click",()=>{
  let argv;
  try { argv=JSON.parse(byId("argv").value); } catch { show("Arguments must be a JSON array.",true); return; }
  act({action:"preset",label:byId("label").value,script:byId("script").value,argv,timeout:selected("timeout")});
});
byId("grant").addEventListener("click",()=>act({action:"grant",username:byId("operator").value,
  preset_id:selected("grant-preset"),hours:Number(byId("hours").value)}));
refresh().catch(e=>show(e.message,true));
setInterval(()=>refresh().catch(()=>{}),15000);
