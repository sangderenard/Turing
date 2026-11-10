// A failed native ownership sample invalidates the image, not just a progress
// statistic. Keep this warning independent of routine status messages.
export function createRodBundleWarning(doc,id) {
  const node=doc.getElementById(id);
  let invalidKey=null;
  const count=status=>status?.rod_bundle_diagnostic?.unresolved_samples;
  function observe(status,key) {
    const n=count(status);
    if(!Number.isSafeInteger(n)||n<=0)return false;
    invalidKey=key;
    if(node) {
      node.hidden=false;
      const detail=status.rod_bundle_diagnostic.warning;
      node.textContent=`INVALID IMAGE: ${n.toLocaleString('en-US')} unresolved rod-bundle sample${n===1?'':'s'}. This image is not a valid converged result.${typeof detail==='string'&&detail?` ${detail}`:''}`;
    }
    return true;
  }
  function displayed(status,key) {
    if(observe(status,key))return;
    // Missing status, an error, a repeated image or a zero from the same
    // invalid image cannot erase the warning. Only a newly displayed image
    // with an explicit zero diagnostic is affirmative evidence.
    if(invalidKey!==null&&key!==invalidKey&&count(status)===0) {
      invalidKey=null;if(node){node.hidden=true;node.textContent='';}
    }
  }
  return {observe,displayed,get invalid(){return invalidKey!==null;}};
}
