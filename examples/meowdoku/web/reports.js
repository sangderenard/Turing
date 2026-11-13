/* Report content is untrusted text. No HTML execution or rendering work. */
'use strict';
(() => {
  const $ = id => document.getElementById(id);
  let account = false, current = null, nextOffset = null, listVersion = 0, detailVersion = 0, busy = false;
  const status = value => { $('status').textContent = value; };
  async function call(path, body) {
    const response = await fetch(path, {credentials: 'same-origin', cache: 'no-store',
      ...(body === undefined ? {} : {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)})});
    const value = await response.json();
    if (!response.ok) throw new Error(value.error || `Request failed (${response.status})`);
    return value;
  }
  function restoreSharing() {
    if (!current) return;
    $('visibility').value = current.access.visibility;
    $('group').value = current.access.group_id || '';
    $('signoff').checked = false;
    changeScope();
  }
  function changeScope() {
    $('group-label').hidden = $('visibility').value !== 'group';
    $('signoff-label').hidden = $('visibility').value !== 'public';
    if ($('visibility').value !== 'public') $('signoff').checked = false;
  }
  async function detail(id) {
    const version = ++detailVersion;
    current = null; $('detail').hidden = true;
    if (!/^[0-9a-f]{32}$/.test(id)) return;
    try {
      const value = await call(account ? `/api/reports/${id}` : `/api/reports/public/${id}`);
      if (version !== detailVersion) return;
      current = value.report;
      $('title').textContent = current.title; $('body').textContent = current.body;
      $('prompts').textContent = current.prompt_history.join('\n\n');
      $('access').textContent = `Access: ${current.access.visibility}`;
      $('steps').replaceChildren(...current.next_steps.map(step => {
        const li = document.createElement('li'); li.textContent = step.text; return li;
      }));
      $('sharing').hidden = !current.can_share;
      if (current.can_share) {
        const groups = await call('/api/account/groups');
        if (version !== detailVersion) return;
        $('group').replaceChildren(...groups.groups.map(group => {
          const option = document.createElement('option'); option.value = group.id; option.textContent = group.name; return option;
        }));
        restoreSharing();
      }
      $('detail').hidden = false;
    } catch (error) { if (version === detailVersion) status(error.message); }
  }
  async function listing(offset = 0) {
    const version = ++listVersion, scope = $('scope').value;
    nextOffset = null; $('next').disabled = true; $('list').replaceChildren();
    status('Loading reports...');
    try {
      const path = scope === 'public' ? `/api/reports/public?offset=${offset}` : `/api/reports?scope=${scope}&offset=${offset}`;
      const value = await call(path);
      if (version !== listVersion) return;
      $('list').replaceChildren(...value.reports.map(report => {
        const li = document.createElement('li'), link = document.createElement('a');
        link.href = '#' + report.id; link.textContent = `${report.category}: ${report.title} (${report.access.visibility})`;
        li.append(link); return li;
      }));
      nextOffset = value.next_offset; $('next').disabled = nextOffset === null;
      status(`${value.reports.length} reports on this page.`);
    } catch (error) { if (version === listVersion) status(error.message); }
  }
  $('scope').addEventListener('change', () => {
    ++detailVersion; current = null; $('detail').hidden = true; listing();
  });
  $('refresh').addEventListener('click', () => { listing(); if (current) detail(current.id); });
  $('next').addEventListener('click', () => { if (nextOffset !== null) listing(nextOffset); });
  $('visibility').addEventListener('change', changeScope);
  $('cancel').addEventListener('click', restoreSharing);
  $('sharing').addEventListener('submit', async event => {
    event.preventDefault(); if (!current || busy || !current.can_share) return;
    const id = current.id, visibility = $('visibility').value, viewVersion = detailVersion, expected = current.access.revision;
    if (visibility === 'public' && !$('signoff').checked) return status('Check the owner sign-off before publishing.');
    if (visibility === 'group' && !$('group').value) return status('Choose a group you belong to.');
    busy = true; $('save').disabled = true; $('cancel').disabled = true;
    try {
      await call(`/api/reports/${id}/sharing`, {visibility, group_id: visibility === 'group' ? $('group').value : null,
        expected_revision: expected, sign_off: visibility === 'public'});
      // Confirm the fresh record rather than treating a send as proof of state.
      const verified = await call(`/api/reports/${id}`);
      if (verified.report.access.visibility !== visibility || verified.report.access.revision !== expected + 1)
        throw new Error('Sharing changed again before readback');
      if (detailVersion === viewVersion && current && current.id === id) {
        await detail(id); await listing(); status('Sharing saved and read back.');
      }
    } catch (error) { status(error.message + ' Refresh before retrying; no automatic write retry.'); }
    finally { busy = false; $('save').disabled = false; $('cancel').disabled = false; }
  });
  window.addEventListener('hashchange', () => detail(location.hash.slice(1)));
  (async () => {
    try {
      const who = await call('/api/account/me'); account = Boolean(who.account);
      const identity = await call('/api/whoami');
      $('scope').querySelector('[value="all"]').hidden = identity.role !== 'admin';
    } catch (error) { status(error.message); }
    await listing(); if (location.hash) await detail(location.hash.slice(1));
  })();
})();
