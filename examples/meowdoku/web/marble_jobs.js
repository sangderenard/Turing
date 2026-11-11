import {mountJoinedPhotoParticipation} from './marble_photo_participation.js';
const ACTIVE = new Set(['queued','preparing','exposing','assembling','encoding','live']);
export function collationLabel(collation) {
  if (!collation) return '';
  const kind = collation.kind === 'tiles' ? 'tiles' : collation.source === 'orbit' ? 'orbit frames' : 'camera poses';
  return `${collation.completed}/${collation.total} ${kind}`;
}
export function jobLabel(job) {
  const queue = job.state === 'queued' && job.queue_position ? ` #${job.queue_position}` : '';
  const collection = job.collation ? ` · ${collationLabel(job.collation)}` : '';
  return `${job.kind === 'live' ? 'Live' : 'Photo'} · ${job.state || 'unknown'}${queue}${collection} · ${String(job.digest).slice(0, 10)}`;
}
export function jobFileUrl(job, type = 'image') {
  const name = type === 'image' ? (job.retouched_image || job.image) : job.hdr;
  if (!/^[a-f0-9]{64}$/.test(job.digest || '') || !name || !['image','hdr'].includes(type)) return null;
  return `/api/marble_photo/${type}?digest=${job.digest}&name=${encodeURIComponent(name)}`;
}
export function mountJobs(root, {doc = document, fetcher = (...args) => globalThis.fetch(...args), interval = 5000, mountParticipant = mountJoinedPhotoParticipation} = {}) {
  const el = (tag, text) => { const n = doc.createElement(tag); if (text) n.textContent = text; return n; };
  const label = el('label','My jobs');
  const select = el('select'); select.setAttribute('aria-label','My render jobs'); label.append(select);
  const refreshButton = el('button','Refresh'); refreshButton.type = 'button';
  const library = el('a','Open jobs library'); library.href = 'marble_jobs.html';
  const status = el('p'); status.setAttribute('role','status');
  const detail = el('div'), participation = el('div');
  root.append(label, refreshButton, library, status, detail, participation);
  let entries = [], selected = new URLSearchParams(globalThis.location?.search || '').get('job'), disposed = false, loading = false, timer, cancelling = false, joining = false, participant = null, participantDigest = null;
  const expanded = new Set();
  const show = () => {
    detail.replaceChildren();
    const job = entries.find(j => j.digest === select.value); if (!job) return;
    const description = el('p', [jobLabel(job), job.created_at ? new Date(job.created_at * 1000).toLocaleString() : '', job.queue_reason || '', job.error || ''].filter(Boolean).join(' · '));
    detail.append(description);
    const url = jobFileUrl(job);
    if (url) {
      const link = el('a','View image'); link.href = url; link.target = '_blank'; link.rel = 'noopener'; detail.append(link);
      if (root.dataset.preview === 'true') { const img = el('img'); img.src = url; img.alt = jobLabel(job); img.style.cssText = 'display:block;max-width:100%;max-height:70vh;margin-top:1rem'; detail.append(img); }
    }
    const hdr = jobFileUrl(job,'hdr');
    if (hdr) { const link = el('a','Download linear HDR'); link.href = hdr; link.style.marginLeft = '1em'; detail.append(link); }
    if (job.collation) {
      const collection = job.collation;
      const group = el('details');
      group.open = expanded.has(job.digest);
      group.addEventListener('toggle', () => { if (group.open) expanded.add(job.digest); else expanded.delete(job.digest); });
      group.append(el('summary', `${collationLabel(collection)} · ${collection.state}`));
      group.append(el('p', collection.kind === 'tiles'
        ? 'Tiles are sensor windows of this photograph. Completed contributions are combined into one final image; individual tile files are not retained.'
        : 'Each completed camera frame stays available below. Canceling stops the remaining sequence and keeps completed frames.'));
      const list = el('ol');
      for (const child of collection.children || []) {
        const window = child.window;
        const description = child.kind === 'tile'
          ? `Tile ${child.index + 1}${window ? ` (${window.x}, ${window.y}; ${window.width} × ${window.height})` : ''}`
          : `Frame ${child.index + 1}`;
        const row = el('li', `${description}: ${child.state}`);
        for (const type of ['image', 'hdr']) {
          const childUrl = jobFileUrl({...child, digest: job.digest}, type);
          if (!childUrl) continue;
          const link = el('a', type === 'image' ? 'View frame' : 'Download HDR');
          link.href = childUrl; link.target = '_blank'; link.rel = 'noopener'; link.style.marginLeft = '1em'; row.append(link);
        }
        list.append(row);
      }
      group.append(list); detail.append(group);
    }
    if (ACTIVE.has(job.state) && job.can_cancel !== false && job.client_intersections === true) {
      const joinButton = el('button', participantDigest === job.digest && participant?.loop.enabled ? 'Participating on this device' : 'Participate / resume on this device');
      joinButton.type = 'button'; joinButton.disabled = joining || (participantDigest === job.digest && participant?.loop.enabled);
      joinButton.addEventListener('click', async () => {
        if (joining || disposed) return; joining = true; joinButton.disabled = true;
        try {
          const response = await fetcher('/api/client_rays/join', {method:'POST',credentials:'same-origin',headers:{'Content-Type':'application/json','X-Meowdoku-Rays':'1'},body:JSON.stringify({job:job.digest})});
          const join = await response.json(); if (!response.ok) throw Error(join.error || `HTTP ${response.status}`);
          if (disposed) return;
          if (join.job !== job.digest) throw Error('Participation connection belongs to a different job.');
          participant?.destroy(); participant = null; participantDigest = null;
          participant = mountParticipant({doc,parent:participation,join,digest:job.digest}); participantDigest = job.digest;
          status.textContent = 'Device participation enabled. Keep this tab visible; pause it below at any time.';
        } catch (error) { if (!disposed) status.textContent = `Could not join job: ${error.message}`; }
        finally { joining = false; if (!disposed) show(); }
      }); detail.append(joinButton);
    }
    if (job.can_cancel === true || (ACTIVE.has(job.state) && job.can_cancel !== false)) {
      const cancel = el('button', job.kind === 'live' ? 'Stop this live job' : 'Cancel this job'); cancel.type = 'button'; cancel.disabled = cancelling; cancel.style.marginLeft = '1em';
      cancel.addEventListener('click', async () => {
        if (cancelling) return; cancelling = true; cancel.disabled = true;
        try {
          const response = await fetcher('/api/marble_photo/cancel',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({digest:job.digest})});
          const value = await response.json(); if (!response.ok) throw new Error(value.error || `HTTP ${response.status}`);
          status.textContent = 'Cancellation requested.'; await refresh();
        } catch (error) { status.textContent = `Could not cancel job: ${error.message}`; }
        finally { cancelling = false; if (!disposed) show(); }
      }); detail.append(cancel);
    }
  };
  const refresh = async () => {
    if (disposed || loading) return; loading = true; refreshButton.disabled = true;
    try {
      const response = await fetcher('/api/marble_photo/jobs',{cache:'no-store',credentials:'same-origin'});
      const value = await response.json(); if (!response.ok) throw new Error(value.error || `HTTP ${response.status}`);
      if (!Array.isArray(value.entries)) throw new Error('Invalid jobs response');
      if (disposed) return;
      selected = select.value || selected; entries = value.entries;
      if (participant && !entries.some(job => job.digest === participantDigest && ACTIVE.has(job.state) && job.can_cancel !== false)) { participant.destroy(); participant = null; participantDigest = null; }
      select.replaceChildren();
      for (const job of entries) { const option = el('option',jobLabel(job)); option.value = job.digest; select.append(option); }
      if (entries.some(j => j.digest === selected)) select.value = selected;
      select.disabled = !entries.length;
      const scheduling = value.queue_available === false ? 'One active GPU job at a time. New jobs cannot queue yet.' : '';
      status.textContent = `${entries.length ? `${entries.length} job(s).` : 'No render jobs for this profile yet.'} ${scheduling}`.trim();
      show();
    } catch (error) { if (!disposed) status.textContent = `Jobs unavailable: ${error.message}`; }
    finally { loading = false; if (!disposed) refreshButton.disabled = false; }
  };
  select.addEventListener('change',show); refreshButton.addEventListener('click',refresh);
  const tick = () => { if (disposed) return; if (!doc.hidden) refresh(); timer = setTimeout(tick, interval); };
  const visible = () => { if (!doc.hidden) refresh(); }; doc.addEventListener('visibilitychange',visible);
  refresh(); timer = setTimeout(tick,interval);
  return {refresh, dispose() { disposed = true; participant?.destroy(); participant = null; clearTimeout(timer); doc.removeEventListener('visibilitychange',visible); }};
}
if (typeof document !== 'undefined') {
  const root = document.getElementById('marble-jobs');
  if (root) { const jobs = mountJobs(root); globalThis.addEventListener('pagehide', event => { if (!event.persisted) jobs.dispose(); });
    globalThis.addEventListener('pageshow', event => { if (event.persisted) jobs.refresh(); }); }
}
