// Small adapter for the pinned llama.cpp Markdown renderer: local generated video
// references use the same Markdown image syntax, then become native video elements.
(() => {
  const toolExpansion = new Map();
  function upgradeTools() {
    const groups = new Map();
    for (const code of document.querySelectorAll('code.language-stack-tool')) {
      let row;try {row = JSON.parse(code.textContent);} catch {continue;}
      if (!row.id || !row.name || !row.status) continue;
      const host = code.closest('.code-block-wrapper') || code.closest('pre');
      if (!host) continue;
      host.classList.add('sl-tool-card');
      const group = groups.get(row.id);
      if (group) {group.row = row;host.hidden = true;}
      else groups.set(row.id, {host, row});
    }
    for (const [id, {host, row}] of groups) {
      host.hidden = false;host.classList.add('sl-tool-card');
      let details = host.querySelector('details.sl-tool-details');
      if (!details) {
        details = document.createElement('details');details.className = 'sl-tool-details';
        const summary = document.createElement('summary');
        summary.innerHTML = '<span class="sl-tool-name"></span><span class="sl-tool-state"></span>';
        const pre = document.createElement('pre');pre.className = 'sl-tool-payload';
        details.append(summary,pre);host.append(details);
        details.open = toolExpansion.get(id) || false;
        details.addEventListener('toggle', () => toolExpansion.set(id,details.open));
      }
      const payload = JSON.stringify(row);
      if (details.dataset.payload === payload) continue;
      details.dataset.payload = payload;
      details.querySelector('.sl-tool-name').textContent = row.name.replaceAll('_',' ');
      details.querySelector('.sl-tool-state').textContent = ({running:'Running…',complete:'Complete',failed:'Failed'})[row.status] || row.status;
      details.querySelector('.sl-tool-payload').textContent = JSON.stringify({arguments:row.arguments,...(row.result === undefined ? {} : {result:row.result})},null,2);
    }
  }
  const upgrade = () => {
    upgradeTools();
    for (const img of document.querySelectorAll('img[src]')) {
      let url;
      try { url = new URL(img.getAttribute('src'), location.href); } catch { continue; }
      if (url.origin !== location.origin || !/^\/outputs\/.+\.(mp4|webm)$/i.test(url.pathname)) continue;
      const video = document.createElement('video');
      video.src = url.href;
      video.controls = true;
      video.playsInline = true;
      video.preload = 'metadata';
      video.style.maxWidth = '100%';
      video.setAttribute('aria-label', img.alt || 'Generated video');
      img.replaceWith(video);
    }
  };
  new MutationObserver(upgrade).observe(document.body, {childList: true, subtree: true});
  upgrade();
  const status = document.createElement('div');
  status.setAttribute('role', 'status');
  status.setAttribute('aria-live', 'polite');
  Object.assign(status.style, {position: 'fixed', bottom: '0.5rem', left: '0.5rem',
    zIndex: '100', padding: '0.4rem 0.7rem', borderRadius: '0.4rem',
    background: 'Canvas', color: 'CanvasText', border: '1px solid GrayText', maxWidth: '80vw'});
  status.hidden = true;
  document.body.appendChild(status);
  const style = document.createElement('style');
  style.textContent = `
    .sl-tool-card{margin:10px 0!important;padding:0!important;border:1px solid var(--border)!important;border-radius:var(--radius-lg,10px)!important;background:var(--background)!important;color:var(--foreground)!important;overflow:hidden}
    .sl-tool-card[hidden]{display:none!important}
    .sl-tool-card>.code-block-header,.sl-tool-card>pre:not(.sl-tool-payload),pre.sl-tool-card>code{display:none!important}
    .sl-tool-details summary{display:flex;align-items:center;gap:10px;cursor:pointer;padding:10px 12px;font:inherit;font-size:12px;list-style:none}
    .sl-tool-details summary::before{content:'›';font-size:16px;line-height:1;transition:transform .15s}
    .sl-tool-details[open] summary::before{transform:rotate(90deg)}
    .sl-tool-state{margin-left:auto;color:var(--muted-foreground);font-size:11px}
    .sl-tool-payload{margin:0!important;padding:12px!important;max-height:320px;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere;border-top:1px solid var(--border);background:var(--muted)!important;color:var(--foreground)!important;font:11px/1.5 ui-monospace,monospace}
    #sl-media-progress{box-sizing:border-box;margin:0 0 10px;padding:12px 14px;border:1px solid var(--border);border-radius:var(--radius-lg,10px);background:var(--background);color:var(--foreground);font:inherit;font-size:12px}
    #sl-media-progress[hidden]{display:none}
    #sl-media-progress.sl-media-floating{position:fixed;bottom:24px;left:50%;transform:translateX(-50%);width:min(460px,calc(100vw - 32px));z-index:90;box-shadow:0 4px 20px #0001}
    .sl-media-heading,.sl-media-metrics{display:flex;align-items:center;justify-content:space-between;gap:12px}
    .sl-media-heading{font-weight:500;margin-bottom:8px}
    .sl-media-phase,.sl-media-metrics{color:var(--muted-foreground);font-weight:400}
    .sl-media-metrics{margin-top:7px;font-variant-numeric:tabular-nums}
    #sl-media-progress progress{display:block;width:100%;height:4px;appearance:none;border:0;border-radius:4px;background:var(--muted);accent-color:var(--foreground);overflow:hidden}
    #sl-media-progress progress::-webkit-progress-bar{background:var(--muted);border-radius:4px}
    #sl-media-progress progress::-webkit-progress-value{background:var(--foreground);border-radius:4px;transition:width .2s}
    #sl-media-progress progress::-moz-progress-bar{background:var(--foreground);border-radius:4px}
  `;
  document.head.append(style);
  const card = document.createElement('section');card.id = 'sl-media-progress';card.hidden = true;
  card.setAttribute('aria-label', 'Media generation progress');
  card.innerHTML = '<div class="sl-media-heading"><span class="sl-media-title"></span><span class="sl-media-phase" role="status" aria-live="polite"></span></div><progress aria-label="Generation steps"></progress><div class="sl-media-metrics"><span class="sl-media-steps"></span><span class="sl-media-rate"></span></div>';
  document.body.append(card);
  function mountProgress() {
    const form = document.querySelector('[data-slot="chat-form"]');
    if (form?.parentElement) {
      if (card.nextElementSibling !== form) form.before(card);
      card.classList.remove('sl-media-floating');
    } else {
      if (card.parentElement !== document.body) document.body.append(card);
      card.classList.add('sl-media-floating');
    }
  }
  new MutationObserver(mountProgress).observe(document.body, {childList:true, subtree:true});mountProgress();
  function paintProgress(item) {
    card.hidden = !item || !item.active && Date.now()/1000 - item.updated > 8;
    if (card.hidden) return;
    const labels = {preparing:'Preparing…', denoising:'Generating…', decoding:'Decoding…', encoding:'Encoding…', upscaling:'Upscaling…', restoring:'Restoring conversation…', complete:'Complete', failed:'Failed'};
    card.querySelector('.sl-media-title').textContent = item.kind === 'video' ? 'Video generation' : 'Image generation';
    card.querySelector('.sl-media-phase').textContent = labels[item.phase] || 'Preparing…';
    const bar = card.querySelector('progress');bar.max = item.total || 1;
    if (item.step > 0 || !item.active) bar.value = item.step || 0;else bar.removeAttribute('value');
    card.querySelector('.sl-media-steps').textContent = `Step ${item.step || 0}/${item.total || '…'}`;
    card.querySelector('.sl-media-rate').textContent = Number.isFinite(item.s_iter) && item.s_iter > 0 ? `${item.s_iter.toFixed(2)} s/it` : '— s/it';
  }
  let polling = false;
  async function updateStatus() {
    if (polling || document.hidden) return;
    polling = true;
    try {
      const response = await fetch('/slstack/status', {cache: 'no-store'});
      if (response.ok) {
        const data = await response.json();
        const event = data.events?.at(-1);
        paintProgress(data.media_progress);
        status.textContent = event ? event.message : '';
        status.hidden = !card.hidden || !event || Date.now() / 1000 - event.time > 15;
      }
    } catch {} finally { polling = false; }
  }
  setInterval(updateStatus, 500);
  updateStatus();
})();
