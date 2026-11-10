const fs=require('fs'),vm=require('vm'),assert=require('assert');
const source=fs.readFileSync('src/compiler/abstract_ui_div_map.py','utf8');
const exact=n=>source.match(new RegExp('^function '+n+'\\b[^]*?^}','m'))[0];
const ctx={model:{world:{object_order:[]}},shaderViewer:{revision:0},turingWorld:{partRuntimeId:()=>1,objectRuntimeId:()=>1},Math,Map,Set,Float32Array,Uint32Array};vm.createContext(ctx);
vm.runInContext(['colorVector','boxYawDegrees','normalized3','cross3','buildExtrudedBoxMesh'].map(exact).join('\n'),ctx);
let cases=0;
for(const rotation of [[0,0,0],[0,45,0],[25,0,0],[0,0,20],[15,33,-19]]){
 const box={identity:'plate',kind:'static-solid',geometry_mode:'solid',center:[3,-2],half_extent:[2,1],height:.6,floor_height:.08,placement:{elevation:3.4,rotation},physics:{kind:'static',role:'solid-contact-surface'}};
 const result=ctx.buildExtrudedBoxMesh([box],{wall:'#777777',floor:'#777777',line:'#777777',roof:'#777777'});
 assert.equal(result.colliders.length,1);
 const c=result.colliders[0],s=c.solid_shape;assert(s);const dot=(a,b)=>a.reduce((t,x,i)=>t+x*b[i],0);
 for(let i=0;i<3;i++)for(let j=0;j<3;j++)assert(Math.abs(dot(s.axes[i],s.axes[j])-(i===j?1:0))<1e-12);
 const vertices=result.mesh;
 for(let i=0;i<vertices.length;i+=9){const p=Array.from(vertices.slice(i,i+3));const local=s.axes.map(a=>dot(a,p.map((v,k)=>v-s.origin[k])));
  for(let k=0;k<3;k++)assert(p[k]>=c.minimum[k]-1e-5&&p[k]<=c.maximum[k]+1e-5);
  assert(local.some((v,k)=>Math.min(Math.abs(v-s.minimum[k]),Math.abs(v-s.maximum[k]))<1e-5));
 }
 cases++;
}
assert.equal(ctx.boxYawDegrees({placement:{yaw_degrees:12}}),12);
assert.equal(ctx.boxYawDegrees({placement:{rotation:[0,0,0],yaw_degrees:12}}),0);
console.log(JSON.stringify({result:'PASS',cases}));

