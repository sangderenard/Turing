// Logical participation bursts stream server packets; this is not a buffer size.
export const MAX_PARTICIPATION_BURST=10000000;
export const PARTICIPATION_KERNEL_CHUNK=4096;
export function participationSettings(value={}) {
 const backend=value.backend??'webgpu';
 const batch=Number(value.batch??1000000),hz=Number(value.hz??10),duty=Number(value.duty??50);
 if(!['webgpu','wasm'].includes(backend)||!Number.isSafeInteger(batch)||batch<16||batch>MAX_PARTICIPATION_BURST||!Number.isFinite(hz)||hz<.1||hz>10||!Number.isFinite(duty)||duty<1||duty>100)
  throw Error('Use 16–10,000,000 rays per streamed burst, 0.1–10 idle polls/s, and 1–100% duty.');
 return {batch,hz,duty,backend};
}
export function participationRest(settings,workMs) {
 const s=participationSettings(settings);
 return Math.max(0,Number(workMs)||0)*(100/s.duty-1);
}
