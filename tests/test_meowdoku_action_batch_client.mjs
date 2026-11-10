import assert from 'node:assert/strict';
import fs from 'node:fs';
const source=fs.readFileSync('examples/meowdoku/web/action_batch_client.js','utf8');
const {ActionBatchClient}=await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
let submitted=null,loss=true;
const client=new ActionBatchClient(async(url,options)=>{
 assert.equal(url,'/api/action/batch');assert.equal(options.credentials,'same-origin');
 const b=JSON.parse(options.body);
 if(b.op)return {ok:true,json:async()=>({table_epoch:'e',next_sequence:0})};
 if(!submitted)submitted=b;else assert.deepEqual(b,submitted);
 if(loss){loss=false;throw new Error('response lost');}
 return {ok:true,json:async()=>({table_epoch:'e',sequence:0,next_sequence:1,status:'stopped',completed:1,failed_action_may_have_mutated:true})};
});
await client.describe();
await assert.rejects(client.run([{type:'cell'},{type:'unknown'}]),/response lost/);
await assert.rejects(client.describe(),/pending/);
await assert.rejects(client.run([{type:'cell'}]),/pending/);
assert.equal((await client.recover()).status,'stopped');
assert.equal(client.pending,null);assert.equal(client.meta.next_sequence,1);
console.log('ActionBatchClient lost response / explicit recovery passed');
