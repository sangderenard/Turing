// Isolated Studio observation only. Does not replace any renderer or API result.
const originalGetContext = HTMLCanvasElement.prototype.getContext;
const originalRaf = window.requestAnimationFrame;
let timingMode = false, timingSamples = [], gpuSamples = [], timerExt = null, pendingQueries = [], lastTimestamp = null;
function timingSummary(values) {
  const sorted = values.slice().sort((a,b)=>a-b);
  return {count:sorted.length, median:sorted[Math.floor(sorted.length*.5)] ?? null,
    p95:sorted[Math.floor(sorted.length*.95)] ?? null};
}
let frame = 0, draws = [], gains = {}, errors = [], viewGL = null, baselinePixels = null, uploads = [], uploadedMask = [], drawnMaterials = {}, mappings = {}, localPbrRows = {};
window.requestAnimationFrame = callback => originalRaf.call(window, timestamp => {
  if (timingMode && viewGL) {
    const gl = viewGL;
    if (timerExt && !gl.getParameter(timerExt.GPU_DISJOINT_EXT)) {
      pendingQueries = pendingQueries.filter(q => {
        if (!gl.getQueryParameter(q,gl.QUERY_RESULT_AVAILABLE)) return true;
        gpuSamples.push(gl.getQueryParameter(q,gl.QUERY_RESULT)/1e6); gl.deleteQuery(q); return false;
      });
    } else if (timerExt) {
      pendingQueries.forEach(q => gl.deleteQuery(q)); pendingQueries=[]; gpuSamples=[];
    }
    const q = timerExt && pendingQueries.length < 8 ? gl.createQuery() : null;
    if(q) gl.beginQuery(timerExt.TIME_ELAPSED_EXT,q);
    const start = performance.now();
    callback(timestamp);
    const cpuMs = performance.now()-start;
    if(q) {gl.endQuery(timerExt.TIME_ELAPSED_EXT); pendingQueries.push(q);}
    timingSamples.push({cpuMs,frameMs:lastTimestamp===null?0:timestamp-lastTimestamp}); lastTimestamp=timestamp;
    if(timingSamples.length>240) timingSamples.shift();
    if(gpuSamples.length>240) gpuSamples.shift();
    document.getElementById('secondary-observation').textContent=JSON.stringify({timing:true,
      cpuSubmitMs:timingSummary(timingSamples.slice(20).map(x=>x.cpuMs)),
      frameIntervalMs:timingSummary(timingSamples.slice(20).map(x=>x.frameMs)),
      gpuMs:timingSummary(gpuSamples.slice(20)),gpuTimerSupported:!!timerExt,
      size:[gl.drawingBufferWidth,gl.drawingBufferHeight]});
    return;
  }
  frame++;
  draws = [];
  gains = {};
  callback(timestamp);
  if (!draws.length) return;
  // Read synchronously before the browser discards the default framebuffer.
  const pixels = new Uint8Array(8 * 8 * 4);
  viewGL.readPixels(Math.floor(viewGL.drawingBufferWidth / 2),
    Math.floor(viewGL.drawingBufferHeight / 2), 8, 8,
    viewGL.RGBA, viewGL.UNSIGNED_BYTE, pixels);
  const allPixels = new Uint8Array(viewGL.drawingBufferWidth * viewGL.drawingBufferHeight * 4);
  viewGL.readPixels(0, 0, viewGL.drawingBufferWidth, viewGL.drawingBufferHeight,
    viewGL.RGBA, viewGL.UNSIGNED_BYTE, allPixels);
  if (document.getElementById('qa-zero-gain')?.checked) baselinePixels = allPixels;
  let changedChannels = 0, absoluteDifference = 0, maximumDifference = 0;
  if (baselinePixels?.length === allPixels.length) {
    for (let i = 0; i < allPixels.length; i++) {
      const delta = Math.abs(allPixels[i] - baselinePixels[i]);
      if (delta) changedChannels++;
      absoluteDifference += delta; maximumDifference = Math.max(maximumDifference, delta);
    }
  }
  const observation = {
    changedChannels, absoluteDifference, maximumDifference, uploads, uploadedMask, drawnMaterials, mappings, localPbrRows,
    wholeFrameHash: allPixels.reduce((hash, value) => Math.imul(hash ^ value, 16777619) >>> 0, 2166136261),
    frame, passes: [...new Set(draws)], gains, errors: errors.slice(-10),
    pixelHash: pixels.reduce((hash, value) => Math.imul(hash ^ value, 16777619) >>> 0, 2166136261),
    size: [viewGL.drawingBufferWidth, viewGL.drawingBufferHeight],
  };
  let output = document.getElementById('secondary-observation');
  if (!output) {
    output = document.createElement('output');
    output.id = 'secondary-observation';
    output.style = 'position:fixed;bottom:0;left:0;z-index:99999;background:black;color:white;max-height:65px;max-width:100%;overflow:auto;font:10px monospace';
    document.body.append(output);
  }
  output.textContent = JSON.stringify(observation);
});
HTMLCanvasElement.prototype.getContext = function (...args) {
  const gl = originalGetContext.apply(this, args);
  if (this.id === 'view' && args[0] === 'webgl2' && gl && !gl.__secondaryProbe) {
    gl.__secondaryProbe = true;
    viewGL = gl;
    timerExt = gl.getExtension('EXT_disjoint_timer_query_webgl2');
    const timingLabel=document.createElement('label'), timingCheckbox=document.createElement('input');
    timingCheckbox.type='checkbox'; timingCheckbox.id='qa-frame-timing';
    timingLabel.style='position:fixed;top:100px;left:10px;z-index:99999;background:black;color:white;display:flex;align-items:center;width:260px;height:30px';
    timingCheckbox.style='width:20px;margin:0 6px;';
    timingLabel.append(timingCheckbox,'Measure frame cost (QA)'); document.body.append(timingLabel);
    timingCheckbox.addEventListener('change',()=>{timingMode=timingCheckbox.checked;timingSamples=[];gpuSamples=[];lastTimestamp=null;});
    const buffers = new WeakMap(), uploadBuffer = gl.bufferData.bind(gl);
    gl.bufferData = (target, data, ...rest) => {
      if (target === gl.ARRAY_BUFFER && (data instanceof Int32Array || data instanceof Uint32Array))
        buffers.set(gl.getParameter(gl.ARRAY_BUFFER_BINDING), [...new Set(data)]);
      return uploadBuffer(target, data, ...rest);
    };
    const upload3 = gl.texImage3D.bind(gl), upload2 = gl.texImage2D.bind(gl);
    gl.texImage3D = (...args) => {
      const values = args[9];
      if (values?.length) {
        let nonzero = 0, maximum = 0;
        for (const value of values) {if (value) nonzero++; maximum = Math.max(maximum, Math.abs(value));}
        uploads.push({dims:args.slice(3,6), nonzero, maximum});
        if (uploads.length > 64) uploads.shift();
      }
      return upload3(...args);
    };
    gl.texImage2D = (...args) => {
      if (args[2] === gl.R32F && gl.getParameter(gl.ACTIVE_TEXTURE) === gl.TEXTURE0 && args[8]?.length)
        localPbrRows = Object.fromEntries([14,15,41,42].map(id => [id, Array.from(args[8].slice(id*16,id*16+16))]));
      if (args[2] === gl.R32UI && args[8]?.length)
        uploadedMask = Array.from(args[8], (mask,id) => ({id,mask})).filter(row => row.mask);
      return upload2(...args);
    };
    const locations = new WeakMap(), getLocation = gl.getUniformLocation.bind(gl);
    gl.getUniformLocation = (program, name) => {
      const location = getLocation(program, name);
      if (location) locations.set(location, name);
      return location;
    };
    const setFloat = gl.uniform1f.bind(gl);
    gl.uniform1f = (location, value) => setFloat(location,
      locations.get(location) === 'uFieldGain' && document.getElementById('qa-zero-gain')?.checked ? 0 : value);
    const label = document.createElement('label'), checkbox = document.createElement('input');
    checkbox.type = 'checkbox'; checkbox.id = 'qa-zero-gain';
    label.style = 'position:fixed;top:55px;left:10px;z-index:99999;background:black;color:white;display:flex;align-items:center;width:280px;height:30px';
    checkbox.style='width:20px;margin:0 6px;';
    label.append(checkbox, 'Suppress field contribution (QA)'); document.body.append(label);

    // Transparent Studio geometry uses indexed draws; capture both draw paths.
    for (const method of ['drawArrays', 'drawElements']) {
      const draw = gl[method].bind(gl);
      gl[method] = function (...drawArgs) {
        if(timingMode) return draw(...drawArgs);
        const program = gl.getParameter(gl.CURRENT_PROGRAM);
        const location = gl.getUniformLocation(program, 'uSecondaryPass');
        draws.push(location === null ? 'primary' : gl.getUniform(program, location));
        const gain = gl.getUniformLocation(program, 'uFieldGain');
        if (location && gain) gains[gl.getUniform(program, location)] = gl.getUniform(program, gain);
        if (location && gl.getUniform(program, location) < 3) {
          const pass = gl.getUniform(program, location), matrix = name => {
            const loc = gl.getUniformLocation(program, name);
            return loc ? Array.from(gl.getUniform(program, loc)) : null;
          };
          mappings[pass] = {fieldFromView:matrix('uFieldFromView'),modelView:matrix('uMV'),modelViewProjection:matrix('uMVP')};
        }
        if (location) {
          const buffer = gl.getVertexAttrib(2, gl.VERTEX_ATTRIB_ARRAY_BUFFER_BINDING);
          const ids = buffer && buffers.get(buffer);
          if (ids) drawnMaterials[gl.getUniform(program, location)] =
            [...new Set([...(drawnMaterials[gl.getUniform(program, location)] || []), ...ids])];
        }
        draw(...drawArgs);
        const error = gl.getError();
        if (error) errors.push({frame, pass: draws.at(-1), error});
      };
    }
  }
  return gl;
};

