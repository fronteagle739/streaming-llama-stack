// Extend the pinned WebUI's actual settings navigation, using its existing
// button classes and theme tokens. Paths are browsed on the serving machine;
// model weights are selected in place and are never uploaded to the browser.
(() => {
  const css=document.createElement('style');css.textContent=`
    .sl-stack-section{flex:1;min-width:0;overflow:auto;padding:24px;color:var(--foreground);font:inherit;font-size:13px}
    .sl-stack-section[hidden]{display:none}.sl-stack-section h3{margin:0 0 20px;font-size:18px;font-weight:600;border-bottom:1px solid var(--border);padding-bottom:20px}
    .sl-stack-section h4{font-size:14px;font-weight:600;margin:24px 0 12px}.sl-stack-section small{display:block;font-size:12px;line-height:1.5;color:var(--muted-foreground)}
    .sl-stack-section input:not([type=checkbox]),.sl-stack-section select{width:100%;min-width:0;height:34px;padding:6px 9px;border:1px solid var(--input,var(--border));border-radius:var(--radius-md,6px);background:var(--background);color:var(--foreground);font:inherit;font-size:12px}
    .sl-stack-section input[type=checkbox]{accent-color:var(--foreground);width:15px;height:15px;flex-shrink:0}
    .sl-stack-section button{border:1px solid var(--border);border-radius:var(--radius-md,6px);padding:6px 10px;background:transparent;color:inherit;font:inherit;font-size:12px;cursor:pointer}
    .sl-stack-section button:hover{background:var(--accent)}.sl-stack-section button:focus-visible,.sl-stack-section input:focus-visible{outline:2px solid var(--ring);outline-offset:2px}
    .sl-stack-setting{display:flex;align-items:center;justify-content:space-between;gap:16px;padding:10px 0}.sl-stack-setting span{min-width:0}
    .sl-stack-path{display:flex;align-items:center;gap:8px;margin:8px 0}.sl-stack-path input{flex:1}.sl-stack-path button{flex-shrink:0;height:34px}
    .sl-stack-models{max-height:180px;overflow:auto;margin:10px 0;border:1px solid var(--border);border-radius:var(--radius-md,6px)}
    .sl-stack-model{padding:9px 12px;border-bottom:1px solid var(--border);overflow-wrap:anywhere;display:flex;align-items:center;gap:8px}.sl-stack-model>span{flex:1;min-width:0}.sl-stack-model:last-child{border-bottom:0}
    .sl-stack-model button[aria-pressed=true]{background:var(--accent)}
    .sl-stack-load{display:flex;align-items:center;gap:6px;flex-shrink:0;font-size:12px;cursor:pointer}.sl-stack-load input{margin:0}
    .sl-stack-files{display:grid;gap:4px;max-height:280px;overflow:auto;margin:12px 0}.sl-stack-files button{text-align:left;overflow-wrap:anywhere;border:0;padding:8px}
    .sl-stack-error{color:var(--destructive);font-size:12px;margin:12px 0}.sl-stack-status{font-size:12px;color:var(--muted-foreground);margin-top:16px}
    @media(max-width:767px){.sl-stack-section{padding:16px}.sl-stack-section h3{font-size:16px}.sl-stack-path{flex-wrap:wrap}.sl-stack-path input{flex-basis:200px}}
  `;document.head.append(css);
  const icon='<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 3 9 5-9 5-9-5 9-5zM3 12l9 5 9-5M3 16l9 5 9-5"/></svg>';
  const titles={set_quality_speed:'Change quality / speed',get_quality_speed:'Read quality / speed',list_media_models:'List supported media models',list_loras:'List compatible LoRAs',generate_image:'Generate images',generate_video:'Generate videos'};
  async function api(path,body){const r=await fetch('/slstack/'+path,body===undefined?{cache:'no-store'}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const data=await r.json();if(!r.ok)throw new Error(data.error?.message||'Settings unavailable');return data;}
  function attach(dialog){
    if(dialog.querySelector('[data-sl-stack]'))return;
    const nav=dialog.querySelector('nav'),layout=nav?.parentElement?.parentElement;
    if(!nav||!layout||!nav.textContent.includes('Sampling'))return;
    const nativeContent=layout.querySelector('[data-slot="scroll-area"]');if(!nativeContent)return;
    const section=document.createElement('section');section.className='sl-stack-section';section.dataset.slStack='';section.hidden=true;section.setAttribute('aria-label','Stack settings');layout.append(section);
    let state,busy=false,modelStamp='';
    const buttons=[];
    const footer=Array.from(dialog.children).find(e=>e!==layout&&e.textContent.includes('Save'));
    const activeClasses=new Map();
    function show(value){section.hidden=!value;nativeContent.hidden=value;if(footer)footer.hidden=value;
      buttons.forEach(b=>{b.classList.toggle('bg-accent',value);b.classList.toggle('text-accent-foreground',value);b.setAttribute('aria-selected',String(value));});
      if(value){for(const b of nav.querySelectorAll('button:not([data-sl-stack-nav])')){activeClasses.set(b,b.className);b.classList.remove('bg-accent','text-accent-foreground');}void load();}
      else{for(const [b,c] of activeClasses)b.className=c;activeClasses.clear();}}
    const mobile=dialog.querySelector('.min-w-max');
    for(const host of [nav,mobile].filter(Boolean)){
      const b=document.createElement('button');b.type='button';b.dataset.slStackNav='';b.className=host.querySelector('button')?.className||'';b.classList.remove('bg-accent','text-accent-foreground');b.innerHTML=icon+'<span>Stack</span>';b.setAttribute('aria-label','Stack settings');b.onclick=()=>show(true);host.append(b);buttons.push(b);
      for(const native of host.querySelectorAll('button:not([data-sl-stack-nav])'))native.addEventListener('click',()=>show(false));
    }
    function status(message,failed=false){let e=section.querySelector('[data-status]');if(!e){e=document.createElement('div');e.dataset.status='';section.append(e);}e.className=failed?'sl-stack-error':'sl-stack-status';e.setAttribute('role',failed?'alert':'status');e.textContent=message;}
    async function save(patch){if(busy)return;busy=true;const scroll=section.scrollTop,modelsScroll=section.querySelector('[data-model-list]')?.scrollTop||0;status('Saving…');try{state=await api('settings',patch);render();section.scrollTop=scroll;const list=section.querySelector('[data-model-list]');if(list)list.scrollTop=modelsScroll;status('Saved · changes apply to the next tool call');}catch(e){status(e.message,true);}finally{busy=false;}}
    function pathField(key,label,kind){const box=document.createElement('div');const title=document.createElement('label');title.textContent=label;const row=document.createElement('div');row.className='sl-stack-path';const input=document.createElement('input');input.type='text';input.readOnly=true;input.value=state.settings[key]||'';input.placeholder='Not selected';input.setAttribute('aria-label',label);const button=document.createElement('button');button.type='button';button.textContent='Browse…';button.setAttribute('aria-label','Browse '+label);button.onclick=()=>browseFor(key,label,kind,input.value);row.append(input,button);box.append(title,row);return box;}
    function toggle(label,description,value,change){const row=document.createElement('label');row.className='sl-stack-setting';const span=document.createElement('span');span.textContent=label;const hint=document.createElement('small');hint.textContent=description;span.append(hint);const input=document.createElement('input');input.type='checkbox';input.checked=value;input.setAttribute('aria-label',label);input.onchange=()=>change(input.checked);row.append(span,input);return row;}
    function heading(text){const h=document.createElement('h4');h.textContent=text;section.append(h);}
    function renderModels(){
      const list=section.querySelector('[data-model-list]'),progress=section.querySelector('[data-scan-progress]');if(!list)return;
      const catalog=state.catalog,stamp=JSON.stringify([catalog.models,catalog.loras,state.settings.image_model,state.settings.video_model,state.media_load_modes]);
      if(progress)progress.textContent=catalog.error||`${catalog.scanning?'Scanning… · ':''}${catalog.models.length} supported models · ${(catalog.loras||[]).length} LoRAs · ${(catalog.files_seen||0).toLocaleString()} files checked${catalog.unreadable_directories?' · '+catalog.unreadable_directories+' folders unavailable':''}`;
      if(stamp===modelStamp)return;modelStamp=stamp;const scroll=list.scrollTop;list.replaceChildren();
      for(const model of catalog.models){
        const row=document.createElement('div');row.className='sl-stack-model';const name=document.createElement('span');name.textContent=model.name;
        const detail=document.createElement('small');detail.textContent=model.arch;detail.title=model.path;name.append(detail);row.append(name);
        for(const kind of model.kinds){const button=document.createElement('button');button.type='button';button.textContent=kind==='image'?'Image':'Video';button.setAttribute('aria-label','Use '+model.name+' for '+kind);button.setAttribute('aria-pressed',String(state.settings[kind+'_model']===model.path));button.onclick=()=>save({[kind+'_model']:model.path,[kind+'_arch']:model.arch});row.append(button);}list.append(row);
        const loading=document.createElement('label');loading.className='sl-stack-load';loading.title='Stream weights from storage. Turn off to load them into GPU memory.';
        const stream=document.createElement('input');stream.type='checkbox';stream.checked=(state.media_load_modes?.[model.id]||'streaming')==='streaming';stream.setAttribute('aria-label','Stream weights for '+model.name);
        const mode=document.createElement('span');mode.textContent=stream.checked?'Streamed':'Resident';loading.append(stream,mode);row.append(loading);
        stream.onchange=()=>save({media_load_modes:{[model.path]:stream.checked?'streaming':'resident'}});
      }
      if(!catalog.models.length){const e=document.createElement('small');e.style.padding='12px';e.textContent=catalog.scanning?'Looking for supported models…':'No supported media models found yet.';list.append(e);}list.scrollTop=scroll;
    }
    function render(){
      section.replaceChildren();const h=document.createElement('h3');h.textContent='Stack';section.append(h);
      heading('Model tools');const help=document.createElement('small');help.textContent='Choose which tools the chat model and MCP clients can use. Disabled tools are removed from the tool list.';section.append(help);
      for(const name of state.tools)section.append(toggle(titles[name]||name,name,state.settings.enabled_tools[name],value=>save({enabled_tools:{[name]:value}})));
      heading('Media models');section.append(pathField('comfyui_models','ComfyUI models folder','folder'));
      const note=document.createElement('small');note.textContent='Choose a ComfyUI folder, any models folder, or an entire drive. Click Image or Video beside a model to use it. Each model remembers its Streamed / Resident setting for future generations.';section.append(note);
      for(const root of state.settings.model_search_roots||[]){const row=document.createElement('div');row.className='sl-stack-path';const input=document.createElement('input');input.readOnly=true;input.value=root;input.setAttribute('aria-label','Additional search folder');const remove=document.createElement('button');remove.type='button';remove.textContent='Remove';remove.setAttribute('aria-label','Remove search folder '+root);remove.onclick=()=>save({model_search_roots:state.settings.model_search_roots.filter(p=>p!==root)});row.append(input,remove);section.append(row);}
      const add=document.createElement('button');add.type='button';add.textContent='Add drive or folder…';add.onclick=()=>browseFor('model_search_roots','additional search folder','folder','');section.append(add);
      section.append(pathField('image_model','Default image model','model'),pathField('video_model','Default video model','model'));
      const models=document.createElement('div');models.className='sl-stack-models';models.dataset.modelList='';const progress=document.createElement('small');progress.dataset.scanProgress='';section.append(progress,models);modelStamp='';renderModels();
      const refresh=document.createElement('button');refresh.type='button';refresh.textContent='Rescan models and LoRAs';refresh.onclick=()=>save({});section.append(refresh);
      const loraNote=document.createElement('small');loraNote.textContent='LoRAs are discovered in all search locations. Ask chat to list compatible LoRAs, then name the ones and strengths to use for a generation. Nothing is applied automatically.';section.append(loraNote);
      heading('Conversation state');section.append(toggle('Save / restore KV for media generation','Automatically includes YOCO state when the preset uses YOCO.',state.settings.cache_media,value=>save({cache_media:value})));
      section.append(toggle('Save / restore KV when switching conversations','Resume a saved conversation without prefilling its full history again.',state.settings.cache_conversations,value=>save({cache_conversations:value})));
    }
    async function load(){section.textContent='Loading stack settings…';try{state=await api('settings');render();}catch(e){status(e.message,true);}}
    async function browseFor(key,title,kind,start){
      const back=document.createElement('button');back.type='button';back.textContent='← Back to Stack';back.onclick=render;
      const name=document.createElement('h3');name.textContent='Select '+title.toLowerCase();
      const path=document.createElement('input');path.type='text';path.setAttribute('aria-label','Folder path');path.value=start;
      const go=document.createElement('button');go.type='button';go.textContent='Go';const nav=document.createElement('div');nav.className='sl-stack-path';nav.append(path,go);
      const list=document.createElement('div');list.className='sl-stack-files';const use=document.createElement('button');use.type='button';use.textContent=kind==='folder'?'Use this folder':'Use this model folder';
      const architecture=document.createElement('select');architecture.setAttribute('aria-label','Model architecture');const automatic=document.createElement('option');automatic.value='';automatic.textContent='Detect architecture automatically';architecture.append(automatic);
      const mediaKind=key==='video_model'?'video':'image';
      for(const [arch,kinds] of Object.entries(state.architectures||{})){if(!kinds.includes(mediaKind))continue;const option=document.createElement('option');option.value=arch;option.textContent=arch;architecture.append(option);}
      const choose=value=>save({[key]:key==='model_search_roots'?[...new Set([...(state.settings.model_search_roots||[]),value])]:value,...(kind==='model'&&architecture.value?{[mediaKind+'_arch']:architecture.value}:{})});
      section.replaceChildren(back,name,nav,...(kind==='model'?[architecture]:[]),list,use);
      async function visit(value){list.textContent='Loading…';try{const row=await api('browse?path='+encodeURIComponent(value));path.value=row.path;use.disabled=!row.path||kind==='vision';list.replaceChildren();
        if(row.parent!==null){const up=document.createElement('button');up.type='button';up.textContent='↑ Parent folder';up.onclick=()=>visit(row.parent);list.append(up);}
        for(const entry of row.entries){if(!entry.directory&&kind==='folder')continue;if(!entry.directory&&kind==='vision'&&!entry.name.toLowerCase().endsWith('.gguf'))continue;const b=document.createElement('button');b.type='button';b.textContent=(entry.directory?'▸ ':'')+entry.name;b.onclick=()=>entry.directory?visit(entry.path):void choose(entry.path);list.append(b);}
      }catch(e){list.textContent=e.message;}}
      go.onclick=()=>visit(path.value);path.onkeydown=e=>{if(e.key==='Enter'){e.preventDefault();void visit(path.value);}};use.onclick=()=>choose(path.value);
      let initial=start;if(initial&&!/[\\/]$/.test(initial)&&/\.(safetensors|gguf)$/i.test(initial))initial=initial.replace(/[\\/][^\\/]+$/,'');await visit(initial||'');
    }
    const poll=setInterval(async()=>{if(!dialog.isConnected){clearInterval(poll);return;}if(section.hidden||busy||!section.querySelector('[data-model-list]'))return;try{state=await api('settings');renderModels();}catch(e){status(e.message,true);}},1500);
  }
  const update=()=>document.querySelectorAll('[role="dialog"][data-slot="dialog-content"]').forEach(attach);
  new MutationObserver(update).observe(document.body,{childList:true,subtree:true});update();
})();
