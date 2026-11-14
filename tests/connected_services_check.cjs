const vm = require("node:vm"), fs = require("node:fs"), assert = require("node:assert/strict");
const source = fs.readFileSync(process.argv[2], "utf8");
const settle = () => new Promise(resolve => setImmediate(resolve));
function harness(who={signed_in:true,username:"test"}, paper={configured:true}) {
  const elements = new Proxy({}, {get:(target,id)=>target[id] ||= {
    hidden:false,disabled:false,textContent:"",value:"",children:[],
    addEventListener(event,fn){this[event]=fn;},removeAttribute(name){delete this[name];},
    replaceChildren(){this.children=[];},appendChild(el){this.children.push(el);}
  }});
  let handler = async path => path === "/api/whoami" ? who : path.includes("alpaca") ? paper : {status:"disconnected"};
  const calls=[];
  const context=vm.createContext({
    document:{getElementById:id=>elements[id],createElement:()=>({})},URL,
    setTimeout:()=>1,clearTimeout:()=>{},
    fetch:async(path,options)=>{
      calls.push({path,options});assert.equal(options.credentials,"same-origin");
      assert.equal(options.cache,"no-store");assert.equal(options.redirect,"error");
      const data=await handler(path,options);
      return data?.http || {ok:true,status:200,json:async()=>data};
    }
  });
  vm.runInContext(source,context);
  return {elements,calls,setHandler:fn=>handler=fn};
}
(async()=>{
  for(const who of [{signed_in:false},{signed_in:true,username:"<img onerror=x>"}]){
    const h=harness(who);await settle();
    assert.equal(h.elements["sign-in"].hidden,who.signed_in);
    assert.equal(h.elements.refresh.disabled,false);
    if(!who.signed_in) assert.equal(h.calls.length,1);
    else assert.equal(h.elements.identity.textContent,"Site account: <img onerror=x>.");
  }
  const h=harness();await settle();
  const {elements:e,calls}=h;
  assert.equal(e["suno-connect"].disabled,false);
  let release;
  h.setHandler(()=>new Promise(resolve=>release=resolve));
  const p=e["suno-connect"].click();e["suno-connect"].click();
  assert.equal(calls.filter(c=>c.path.endsWith("/connect")).length,1);
  assert.equal(calls.at(-1).options.headers["X-Meowdoku-Secret"],"1");
  release({status:"authorization_required",authorization_url:"https://suno.com/authorize?state=test"});
  await p;
  assert.equal(e["suno-authorize"].hidden,false);
  assert.equal(e["suno-connect"].disabled,true);
  h.setHandler(()=>({status:"connected"}));await e["suno-refresh"].click();
  assert.equal(e["suno-authorize"].hidden,true);
  const tool={name:"real_remote_tool",inputSchema:{type:"object",properties:{nested:{anyOf:[{type:"null"},{type:"array",items:{type:"number"}}]}}},annotations:{readOnlyHint:false},description:"<script>bad</script>"};
  h.setHandler(()=>({tools:[tool]}));await e["suno-tools-refresh"].click();
  assert.deepEqual(JSON.parse(e["suno-schema"].textContent),tool);
  e["suno-arguments"].value="[]";const count=calls.length;await e["suno-call"].click();
  assert.equal(calls.length,count);
  e["suno-arguments"].value='{"nested":[1,2]}';
  h.setHandler((path,options)=>{
    assert.equal(path,"/api/connections/suno/call");
    assert.deepEqual(JSON.parse(options.body),{name:"real_remote_tool",arguments:{nested:[1,2]}});
    return {isError:true,content:[{type:"text",text:"<img onerror=x>"}]};
  });
  await e["suno-call"].click();assert.match(e["suno-result"].textContent,/isError/);
  h.setHandler(()=>({status:"disconnected"}));await e["suno-disconnect"].click();
  assert.equal(e["suno-tools"].hidden,true);assert.equal(e["suno-call"].disabled,true);
  h.setHandler(()=>({status:"authorization_required",authorization_url:"javascript:bad()"}));
  await e["suno-connect"].click();assert.equal(e["suno-authorize"].hidden,true);
  assert.match(e["suno-status"].textContent,/Invalid authorization/);
  // Logout clears private catalogue/results; late old-account requests cannot restore it.
  h.setHandler(()=>new Promise(resolve=>release=resolve));
  const old=e["suno-refresh"].click();
  h.setHandler(()=>({signed_in:false}));await e.refresh.click();
  release({status:"connected"});await old;
  assert.equal(e["suno-connect"].disabled,true);assert.equal(e["suno-tools"].hidden,true);
  assert.equal(e["suno-result"].textContent,"");
  const offline=harness();offline.setHandler(()=>{throw Error("offline");});await settle();
  assert.equal(offline.elements.refresh.disabled,false);
  console.log("PASS: account isolation, real auth handoff, repeated clicks, schema preservation, explicit calls, logout races, offline");
})().catch(error=>{console.error(error);process.exitCode=1;});
