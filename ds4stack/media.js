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
})();
