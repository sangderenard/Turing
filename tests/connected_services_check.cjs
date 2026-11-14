const vm = require("node:vm");
const fs = require("node:fs");
const assert = require("node:assert/strict");
const source = fs.readFileSync(process.argv[2], "utf8");
const settle = () => new Promise(resolve => setImmediate(resolve));
async function scenario(who, paper, fail = false) {
  const elements = Object.fromEntries(["identity", "sign-in", "paper-status", "paper-actions", "refresh"].map(id =>
    [id, {hidden:false,disabled:false,textContent:"", addEventListener(_,fn){this.click=fn;}}]));
  const calls = [];
  let block, release;
  const context = vm.createContext({document:{getElementById:id=>elements[id]}, fetch:async(path, options)=>{
    calls.push(path);
    assert.equal(options.credentials,"same-origin");assert.equal(options.redirect,"error");assert.equal(options.cache,"no-store");
    assert.equal(options.method,undefined);
    if (block) await block;
    if (fail && path === "/api/whoami") throw Error("offline");
    return {ok: true, json:async()=>path === "/api/whoami" ? who : paper};
  }});
  vm.runInContext(source,context);await settle();
  if(fail) assert.match(elements.identity.textContent,/Unable/);
  else if(!who.signed_in) {assert.equal(calls.length,1);assert.equal(elements["sign-in"].hidden,false);}
  else {
    assert.equal(elements.identity.textContent,"Site account: " + who.username + ".");
    assert.equal(elements["paper-actions"].hidden,false);
    assert.match(elements["paper-status"].textContent, typeof paper.configured === "boolean" ?
      (paper.configured ? /^Paper credential stored/ : /^No paper credential/) : /^Unable/);
  }
  assert.equal(elements.refresh.disabled,false);
  block=new Promise(resolve=>release=resolve);
  const before=calls.length;
  elements.refresh.click();elements.refresh.click();
  assert.equal(calls.length,before+1);assert.equal(elements.refresh.disabled,true);
  release();block=null;await settle();
  assert.equal(elements.refresh.disabled,false);
  assert.ok(calls.every(path=>["/api/whoami","/api/user-secrets/alpaca-paper"].includes(path)));
}
(async()=>{
  await scenario({signed_in:false},{});
  await scenario({signed_in:true,username:"<img onerror=x>"},{configured:true});
  await scenario({signed_in:true,username:"test"},{configured:false});
  await scenario({signed_in:true,username:"test"},{});
  await scenario({}, {}, true);
  console.log("PASS: guest, stored, absent, invalid, offline, repeated/interrupted refresh");
})().catch(error=>{console.error(error);process.exitCode=1;});
