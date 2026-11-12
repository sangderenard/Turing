import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
const source=readFileSync(new URL('../examples/meowdoku/web/marble_calibration.js',import.meta.url),'utf8');
const start=source.indexOf('async function exposeCase('),end=source.indexOf('\nfunction gainReport(',start);
const body=new ArrayBuffer(16),header=new DataView(body);
header.setUint32(4,2097153,true);header.setUint32(8,6500,true);
const expose=Function('packMarblePhoto','fetch',`${source.slice(start,end)};return exposeCase;`)(()=>body,
  async()=>({ok:false,status:400,json:async()=>({error:'invalid MRP1 header'})}));
await assert.rejects(expose({},'x',{}, {},{}),/invalid MRP1 header \(config 2097153 bytes; limit 2097152; triangles 6500; reserved 0; payload 16 bytes\)/);
console.log('Calibration header error diagnostics passed with mocked rejection; no network/render');
