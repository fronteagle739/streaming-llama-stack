// Small adapter for the pinned llama.cpp Markdown renderer: local generated video
// references use the same Markdown image syntax, then become native video elements.
(() => {
  const upgrade = () => {
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
  let polling = false;
  async function updateStatus() {
    if (polling || document.hidden) return;
    polling = true;
    try {
      const response = await fetch('/slstack/status', {cache: 'no-store'});
      if (response.ok) {
        const data = await response.json();
        const event = data.events?.at(-1);
        status.textContent = event ? event.message : '';
        status.hidden = !event || Date.now() / 1000 - event.time > 15;
      }
    } catch {} finally { polling = false; }
  }
  setInterval(updateStatus, 1000);
  updateStatus();
})();
