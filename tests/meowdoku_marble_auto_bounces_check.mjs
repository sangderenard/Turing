// Auto/manual modes are separate from the preserved numeric manual limit.
import assert from 'node:assert/strict';
import {photoBounceMode, createPhotoBounceControls, packMarblePhoto, showBounceDiagnostic}
  from '../examples/meowdoku/web/marble_photo.js';
class Element extends EventTarget {
  constructor(id,value=''){super();this.id=id;this.value=value;this.textContent='';}
  setAttribute(){} setCustomValidity(value){this.validationMessage=value;}
}
const elements=new Map(['photo-bounces','photo-bounces-slider','photo-bounces-out','photo-bounce-mode','photo-bounce-diagnostic'].map(id=>[id,new Element(id)]));
elements.get('photo-bounces').value='37';elements.get('photo-bounce-mode').value='auto';
const doc={getElementById:id=>elements.get(id),defaultView:{Event}};
createPhotoBounceControls(doc);
assert.equal(photoBounceMode(doc),'auto');
assert.equal(elements.get('photo-bounces').disabled,true);
assert.equal(elements.get('photo-bounces').value,'37');
assert.equal(elements.get('photo-bounces-slider').disabled,true);
for(const mode of ['manual','auto','manual']){
 elements.get('photo-bounce-mode').value=mode;
 elements.get('photo-bounce-mode').dispatchEvent(new Event('change'));
 assert.equal(elements.get('photo-bounces').value,'37');
 assert.equal(elements.get('photo-bounces').disabled,mode==='auto');
}
const unpack=bytes=>JSON.parse(new TextDecoder().decode(new Uint8Array(bytes,16,new DataView(bytes).getUint32(4,true))));
const geo={interior:[],glass:[]},digest='a'.repeat(64);
assert.equal(unpack(packMarblePhoto(geo,digest)).bounce_mode,'auto');
assert.equal(unpack(packMarblePhoto(geo,digest,{maxBounces:37})).bounce_mode,'manual');
const auto=unpack(packMarblePhoto(geo,digest,{maxBounces:37,bounceMode:'auto'}));
assert.equal(auto.max_bounces,37);assert.equal(auto.bounce_mode,'auto');
showBounceDiagnostic(doc,{bounce_diagnostic:{mode:'auto',chosen_depth:200,paths_at_render_cap:12,paths_at_probe_limit:3,
 tail_fraction:.03,tail_fraction_95_batch_upper:.06,coverage_hits:[50,49,51,44],confidence:'limited sparse pilot',warning:'Mirror-room warning'}});
const text=elements.get('photo-bounce-diagnostic').textContent;
for(const pattern of [/rendering 200/,/Probe depth 1000/,/3\/256 still carry energy after 1000/,/not an image-error guarantee/,/Mirror-room warning/]) assert.match(text,pattern);
showBounceDiagnostic(doc,{bounce_diagnostic:{mode:'auto',state:'probing',batch:2,batches:4}});
assert.match(elements.get('photo-bounce-diagnostic').textContent,/batch 3\/4/);
showBounceDiagnostic(doc,{bounce_diagnostic:{mode:'manual',chosen_depth:37}});
assert.equal(elements.get('photo-bounce-diagnostic').textContent,'Manual: 37 bounces.');
console.log('Auto bounce controls: default, manual preservation, payload, probe limits and uncertainty PASS');
