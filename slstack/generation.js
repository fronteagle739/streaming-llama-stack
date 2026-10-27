// Studio's quality/speed contract presented using the pinned WebUI's neutral
// toolbar, Lucide strokes and theme tokens. Injected by the stack router so the
// upstream generated UI bundle need not be patched or rebuilt.
(() => {
  const nativeControls = window.slstackNativeGeneration !== false;
  const phases = {reasoning: 'Reasoning', answer: 'Answer', tool_call: 'Tools', code: 'Code'};
  const points = [['Max quality', 0], ['Balanced', .35], ['Fast', .65], ['Faster', .9], ['Max speed', 1.36125]];
  let settings, applied, selected = 'reasoning', dirty = false, saving = false, editing = false, request = 0;
  let revision = 0, pending = {}, timer, error = '';
  const originalFetch = window.fetch.bind(window);
  // The toolbar owns sampling for WebUI turns. Remove the stock UI's always-
  // present defaults, which otherwise override explicit per-phase settings.
  // Direct API callers retain the OpenAI request-level override semantics.
  window.fetch = function(input, init) {
    let url; try { url = new URL(typeof input === 'string' ? input : input.url, location.href); } catch {}
    if (url?.origin === location.origin && /\/chat\/completions$/.test(url.pathname) && typeof init?.body === 'string') {
      try {
        const body = JSON.parse(init.body);
        if (nativeControls) for (const key of ['temperature', 'top_p', 'top_k', 'min_p']) delete body[key];
        if (body.seed == null || body.seed < 0) delete body.seed;
        init = {...init, body: JSON.stringify(body)};
      } catch {}
    }
    return originalFetch(input, init);
  };
  if (!nativeControls) return;
  const style = document.createElement('style');
  style.textContent = `
    #sl-generation-toolbar{display:flex;align-items:center;gap:8px;pointer-events:auto;font:inherit;color:var(--foreground);min-height:32px}
    #sl-generation-toolbar[hidden]{display:none}
    #sl-generation-toolbar label{display:flex;align-items:center;gap:8px;font-size:12px;color:var(--muted-foreground)}
    #sl-generation-toolbar input[type=range]{width:106px;margin:0;accent-color:var(--foreground);cursor:pointer}
    .sl-generation-button{display:inline-flex;align-items:center;justify-content:center;gap:6px;min-height:32px;padding:6px 9px;border:0;border-radius:var(--radius-md,6px);background:transparent;color:var(--foreground);font:inherit;font-size:12px;cursor:pointer}
    .sl-generation-button:hover,.sl-generation-button[aria-expanded=true]{background:var(--accent);color:var(--accent-foreground)}
    .sl-generation-button:focus-visible,#sl-generation-panel input:focus-visible{outline:2px solid var(--ring);outline-offset:2px}
    #sl-generation-toolbar svg{width:16px;height:16px;flex-shrink:0}
    #sl-generation-mode{min-width:64px;color:var(--foreground)}
    #sl-generation-panel{position:fixed;inset:64px 16px auto auto;margin:0;z-index:1000;width:min(380px,calc(100vw - 24px));max-height:calc(100dvh - 88px);overflow:auto;padding:18px;border:1px solid var(--border);border-radius:var(--radius-lg,10px);background:var(--popover,var(--background));color:var(--popover-foreground,var(--foreground));box-shadow:0 8px 30px #0002;font:inherit;font-size:13px}
    #sl-generation-panel::backdrop{background:transparent}
    .sl-generation-row{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:14px}
    #sl-generation-panel h2{font-size:14px;font-weight:600;margin:0}
    #sl-generation-panel small{color:var(--muted-foreground);font-size:12px;line-height:1.5;display:block}
    #sl-generation-panel input[type=checkbox]{accent-color:var(--foreground)}
    .sl-generation-tabs{display:flex;gap:3px;padding:3px;border-radius:var(--radius-md,6px);background:var(--muted);margin:14px 0}
    .sl-generation-tabs button{flex:1;border:0;padding:6px 2px;border-radius:4px;font:inherit;font-size:12px;color:var(--muted-foreground);background:transparent;cursor:pointer}
    .sl-generation-tabs button[aria-selected=true]{background:var(--background);color:var(--foreground);box-shadow:0 1px 3px #0001}
    .sl-generation-field{display:grid;grid-template-columns:1fr 84px;align-items:center;gap:10px;margin:12px 0}
    .sl-generation-field input{box-sizing:border-box;min-width:0;width:100%;height:32px;padding:5px 8px;border:1px solid var(--input,var(--border));border-radius:var(--radius-md,6px);background:transparent;color:inherit;font:inherit;font-size:12px}
    .sl-generation-field input[type=range]{padding:0;border:0;accent-color:var(--foreground)}
    #sl-generation-status{font-size:11px;min-height:16px;margin-top:12px;color:var(--muted-foreground)}
    #sl-generation-status[role=alert]{color:var(--destructive)}
    @media(max-width:600px){#sl-generation-toolbar{gap:4px}#sl-generation-toolbar input[type=range]{width:76px}#sl-generation-mode{display:none}.sl-generation-button{padding:6px}#sl-generation-panel{top:60px;right:12px}}
  `;
  document.head.append(style);
  const toolbar = document.createElement('div'); toolbar.id = 'sl-generation-toolbar'; toolbar.hidden = true;
  toolbar.innerHTML = `<label title="Quality vs speed. Changing this slider sets every phase; applies at the next token."><span id="sl-generation-mode">Balanced</span><input aria-label="Quality vs speed" type="range" min="0" max="1.36125" step="0.00125" value="0.35"></label><button class="sl-generation-button" type="button" aria-label="Generation phase settings" aria-haspopup="dialog" aria-expanded="false" aria-controls="sl-generation-panel" title="Bias and sampling by response phase"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 4h-7M10 4H3M21 12h-9M8 12H3M21 20h-5M12 20H3M14 2v4M8 10v4M16 18v4"/></svg><span>Phases</span></button>`;
  const range = toolbar.querySelector('input'), trigger = toolbar.querySelector('button');
  const panel = document.createElement('dialog'); panel.id = 'sl-generation-panel'; panel.setAttribute('aria-label', 'Generation settings');
  panel.innerHTML = `<div class="sl-generation-row"><h2>Generation settings</h2><button type="button" class="sl-generation-button" aria-label="Close generation settings">✕</button></div><small>Quality and sampling for this session. Changes apply at the next token.</small><label class="sl-generation-row" style="margin-top:16px"><span>Settings by phase</span><input id="sl-generation-enabled" type="checkbox"></label><div class="sl-generation-tabs" role="tablist" aria-label="Response phase">${Object.entries(phases).map(([id, title]) => `<button type="button" role="tab" data-phase="${id}" aria-selected="false">${title}</button>`).join('')}</div><div id="sl-generation-fields"></div><button id="sl-generation-preset" type="button" class="sl-generation-button" style="border:1px solid var(--border);width:100%;margin-top:8px">Think fast, answer carefully</button><small style="margin-top:12px">Higher bias favors speed and may reduce answer quality. Tools and code use careful settings.</small><div id="sl-generation-status" role="status" aria-live="polite"></div>`;
  document.body.append(panel);
  const modeName = v => points.find(([, x]) => Math.abs(x - v) < .0001)?.[0] || 'Custom';
  function mount() {
    if (toolbar.isConnected) return;
    const header = document.querySelector('header');
    if (!header) return;
    const deleteIcon = header.querySelector('svg.lucide-trash-2, svg.lucide-trash2, svg.lucide-trash');
    const anchor = deleteIcon?.closest('button');
    const host = anchor?.parentElement || header.querySelector('.pointer-events-auto') || header;
    host.insertBefore(toolbar, anchor || host.firstChild);
  }
  new MutationObserver(mount).observe(document.body, {childList: true, subtree: true}); mount();
  function close() { panel.close(); trigger.setAttribute('aria-expanded', 'false'); trigger.focus(); }
  trigger.addEventListener('click', () => { if (panel.open) close(); else {panel.show();trigger.setAttribute('aria-expanded', 'true');renderFields();panel.querySelector('button').focus();} });
  panel.querySelector('[aria-label="Close generation settings"]').onclick = close;
  panel.addEventListener('cancel', () => { trigger.setAttribute('aria-expanded', 'false');trigger.focus(); });
  document.addEventListener('pointerdown', e => { if (panel.open && !panel.contains(e.target) && !toolbar.contains(e.target)) close(); });
  document.addEventListener('keydown', e => { if (e.key === 'Escape' && panel.open) {e.preventDefault();close();} });
  function paint() {
    if (!settings) return;
    toolbar.hidden = false;
    if (!editing) range.value = settings.per_phase ? settings.phases[applied?.phase || 'answer']?.bias ?? settings.value : settings.value;
    const label = settings.per_phase ? 'Per phase' : modeName(Number(range.value));
    toolbar.querySelector('#sl-generation-mode').textContent = label;
    range.setAttribute('aria-valuetext', `${label}, bias ${range.value}`);
    range.title = `Bias ${Number(range.value).toFixed(3)} · next token`;
    panel.querySelector('#sl-generation-enabled').checked = settings.per_phase;
    panel.querySelectorAll('[role=tab]').forEach(b => {b.setAttribute('aria-selected', String(b.dataset.phase === selected));b.title = applied?.phase === b.dataset.phase ? 'Current phase' : phases[b.dataset.phase];});
    const status = panel.querySelector('#sl-generation-status');
    status.setAttribute('role', error ? 'alert' : 'status');
    status.textContent = error || (saving ? 'Saving…' : applied?.revision === settings.revision ? `Applied · ${phases[applied.phase]} · token ${applied.position}` : 'Saved · applies at the next token');
  }
  function renderFields() {
    if (!settings) return;
    const phase = settings.per_phase, knobs = phase ? settings.phases[selected] : {...settings.sampler, bias: settings.value};
    const fields = [['bias', 'Speed bias', 0, 1.36125, .00125], ['temperature', 'Temperature', 0, 2, .05], ['top_p', 'Top p', .001, 1, .01], ['top_k', 'Top k', 0, 1024, 1], ['min_p', 'Min p', 0, 1, .01]];
    panel.querySelector('#sl-generation-fields').innerHTML = fields.map(([key, title, min, max, step]) => `<label class="sl-generation-field"><span>${title}</span><input aria-label="${phase ? phases[selected] : 'All phases'} ${title}" data-knob="${key}" type="number" min="${min}" max="${max}" step="${step}" value="${knobs[key]}"></label>`).join('');
    panel.querySelector('.sl-generation-tabs').style.opacity = phase ? '1' : '.5';
    panel.querySelectorAll('[data-knob]').forEach(input => input.addEventListener('input', () => {
      if (input.value === '' || !input.checkValidity()) return;
      const value = Number(input.value), key = input.dataset.knob;
      if (settings.per_phase) {settings.phases[selected][key] = value;queue({phases:{[selected]:{[key]:value}}});}
      else if (key === 'bias') {settings.value = value;queue({value});}
      else {settings.sampler[key] = value;queue({sampler:{[key]:value}});}
      paint();
    }));paint();
  }
  function merge(target, patch) {for (const [key, value] of Object.entries(patch)) {if (value && typeof value === 'object') merge(target[key] ||= {}, value);else target[key] = value;}return target;}
  function queue(patch) {merge(pending, patch);dirty = true;clearTimeout(timer);timer = setTimeout(flush, 35);}
  async function flush() {
    if (saving || !dirty) return;
    const patch = pending;pending = {};dirty = false;saving = true;error = '';paint();
    try {
      const response = await originalFetch('/slstack/generation', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(patch)});
      const data = await response.json();if (!response.ok) throw new Error(data.error?.message || 'Settings refused');
      revision = data.settings.revision;
      settings = merge(data.settings, pending);
    } catch (e) {error = `Couldn’t apply settings: ${e.message}`;}
    finally {saving = false;paint();if (dirty) void flush();}
  }
  range.addEventListener('input', () => {editing = true;settings.per_phase = false;settings.value = Number(range.value);queue({value:settings.value,per_phase:false});paint();});
  range.addEventListener('change', () => {editing = false;void flush();if(panel.open)renderFields();});
  range.addEventListener('blur', () => {editing = false;});
  panel.querySelector('#sl-generation-enabled').addEventListener('change', e => {settings.per_phase=e.target.checked;queue({per_phase:settings.per_phase});renderFields();});
  panel.querySelectorAll('[data-phase]').forEach(b => b.onclick = () => {selected=b.dataset.phase;renderFields();});
  panel.querySelector('#sl-generation-preset').onclick = () => {
    const map = {reasoning:{bias:1.36125,temperature:1},answer:{bias:.35,temperature:.6},tool_call:{bias:0,temperature:.1},code:{bias:0,temperature:.2}};
    settings.per_phase=true;merge(settings.phases,map);queue({per_phase:true,phases:map});renderFields();
  };
  async function poll() {
    if (document.hidden || request) return;request++;
    try {
      const response=await originalFetch('/slstack/generation',{cache:'no-store'});if(!response.ok)throw new Error('Generation controls unavailable');
      const data=await response.json();if(!data.supported){toolbar.hidden=true;return;}
      applied=data.applied;
      if(!saving&&!dirty&&!editing&&data.settings.revision>=revision){const first=!settings;settings=data.settings;revision=settings.revision;if(first||panel.open&&!panel.contains(document.activeElement))renderFields();}
      paint();
    } catch(e) {error=e.message;paint();} finally{request--;}
  }
  setInterval(poll,500);void poll();
})();
