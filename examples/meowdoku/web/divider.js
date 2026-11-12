// Source divider page: submit a .py/.zip (or record a repo link), list the caller's jobs, show one job's pieces.
// Plain DOM (createElement / textContent only).  Refreshes every 5 s while a job is active and the page is visible.
const REFRESH_MS = 5000;
const $ = (id) => document.getElementById(id);

let selected = null;
let timer = null;
let jobs = [];

function cell(text, className) {
  const td = document.createElement('td');
  td.textContent = text == null ? '' : String(text);
  if (className) td.className = className;
  return td;
}

function stateClass(state) {
  if (state === 'complete' || state === 'done') return 'ok';
  if (['refused', 'error', 'blocked', 'needs-annotation', 'cancelled'].includes(state)) return 'bad';
  return '';
}

function setStatus(text, bad) {
  const el = $('status');
  el.textContent = text;
  el.className = bad ? 'bad' : 'mut';
}

async function api(path, options = {}) {
  const response = await fetch(path, {cache: 'no-store', credentials: 'same-origin', ...options});
  let body = null;
  try { body = await response.json(); } catch (e) { body = null; }
  if (!response.ok) {
    const error = new Error((body && body.error) || `HTTP ${response.status}`);
    error.status = response.status;
    throw error;
  }
  return body;
}

export function attemptSummary(counts) {
  return Object.keys(counts || {}).sort().map((k) => `${k} ${counts[k]}`).join(', ');
}

function renderJobs() {
  const body = $('jobs').querySelector('tbody');
  body.replaceChildren();
  for (const job of jobs) {
    const tr = document.createElement('tr');
    tr.className = 'pick' + (job.job_id === selected ? ' sel' : '');
    tr.append(cell(job.job_id), cell(job.label || job.origin), cell(job.state, stateClass(job.state)), cell(job.pieces),
              cell(attemptSummary(job.attempts)), cell(new Date(job.created * 1000).toLocaleString()));
    tr.addEventListener('click', () => { selected = job.job_id; renderJobs(); loadDetail(); });
    body.append(tr);
  }
}

function renderDetail(job) {
  $('detail').hidden = false;
  $('dtitle').textContent = `Job ${job.job_id}: ${job.label || job.origin}`;
  $('dstate').textContent = `${job.state}; ${job.pieces} piece(s); backends ${job.backends.join(', ')}`;
  $('cancel').hidden = !job.active;
  const note = $('dnote');
  note.hidden = !job.note;
  note.textContent = job.note || '';
  const storage = $('dstorage');
  const runs = job.storage || [];
  storage.hidden = !runs.length;
  storage.textContent = runs.map((run) => {
    const measured = run.measured_at == null ? 'not measured yet' : `${run.measured_bytes.toLocaleString()} bytes measured`;
    const review = run.review_due ? 'retention review due' : `review after ${new Date(run.review_after * 1000).toLocaleString()}`;
    return `Compile ${run.contract_id}: ${measured}, ${run.charged_bytes.toLocaleString()} bytes reserved/charged; ${run.status}; ${review}`
      + (run.measurement_error ? '; measurement needs attention' : '');
  }).join('. ') + (runs.length ? '. Files belong to this account. Review dates never delete files automatically.' : '');
  const body = $('pieces').querySelector('tbody');
  body.replaceChildren();
  for (const piece of job.pieces_detail) {
    const tr = document.createElement('tr');
    const attempts = piece.attempts.map((a) => {
      const bits = [`${a.backend}: ${a.state}`];
      if (a.attempts) bits.push(`x${a.attempts}`);
      if (a.error_type) bits.push(`${a.error_type}${a.error ? ` (${a.error})` : ''}`);
      return bits.join(' ');
    }).join('; ');
    const state = piece.annotated ? piece.state : `${piece.state} (${job.unstable_allow_inference ? 'unannotated; inference permitted' : 'needs annotation'})`;
    tr.append(cell(piece.file), cell(piece.names.join(', ')), cell(state, stateClass(piece.state)), cell(attempts));
    body.append(tr);
  }
}

async function loadDetail() {
  if (!selected) { $('detail').hidden = true; return; }
  try {
    renderDetail(await api(`/api/divider/jobs/${encodeURIComponent(selected)}`));
  } catch (error) {
    $('detail').hidden = true;
    setStatus(error.message, true);
  }
}

async function refresh() {
  try {
    jobs = (await api('/api/divider/jobs')).jobs;
    $('gate').hidden = true;
  } catch (error) {
    if (error.status === 403) {
      $('gate').hidden = false;
      $('gate').textContent = 'Sign in with a member account to use the divider.';
    } else {
      setStatus(error.message, true);
    }
    return;
  }
  renderJobs();
  await loadDetail();
  schedule();
}

function schedule() {
  clearTimeout(timer);
  timer = null;
  if (document.visibilityState === 'visible' && jobs.some((j) => j.active)) timer = setTimeout(refresh, REFRESH_MS);
}

async function submit(path, options, label) {
  setStatus(`${label}...`);
  try {
    const job = await api(path, options);
    selected = job.job_id;
    setStatus(`job ${job.job_id} ${job.state}`);
  } catch (error) {
    setStatus(error.message, true);
    return;
  }
  await refresh();
}

function init() {
  $('fileform').addEventListener('submit', async (event) => {
    event.preventDefault();
    const file = $('file').files[0];
    if (!file) { setStatus('choose a .py or .zip file first', true); return; }
    if ($('inference').checked && !$('unstable').checked) {
      setStatus('Type inference requires unstable source-only compilation.', true); return;
    }
    const headers = {'Content-Type': 'application/octet-stream', 'X-Filename': file.name};
    if ($('unstable').checked) headers['X-Compile-Unstable'] = 'source-only';
    if ($('inference').checked) headers['X-Compile-Inference'] = 'allow';
    await submit('/api/divider/jobs', {method: 'POST', headers,
                                       body: file}, 'uploading');
  });
  $('repoform').addEventListener('submit', async (event) => {
    event.preventDefault();
    const repo = $('repo').value.trim();
    const ref = $('ref').value.trim();
    await submit('/api/divider/jobs', {method: 'POST', headers: {'Content-Type': 'application/json'},
                                       body: JSON.stringify(ref ? {repo, ref} : {repo})}, 'recording');
  });
  $('cancel').addEventListener('click', async () => {
    if (!selected) return;
    try {
      await api(`/api/divider/jobs/${encodeURIComponent(selected)}/cancel`, {method: 'POST'});
      setStatus('cancelled');
    } catch (error) {
      setStatus(error.message, true);
    }
    await refresh();
  });
  document.addEventListener('visibilitychange', () => { if (document.visibilityState === 'visible') refresh(); else schedule(); });
  refresh();
}

if (typeof document !== 'undefined' && document.getElementById('jobs')) init();
