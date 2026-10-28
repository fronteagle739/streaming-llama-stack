// Actual stack WebUI gate. No interception, synthetic response or engine launch.
// Usage: node cone_webui_gate.cjs BASE PROMPT_FILE OUT_DIR [REFERENCE_PROOF_JSON]
// Own/guard Chromium separately, then provide CONE_CDP and PLAYWRIGHT_MODULE.
const {chromium}=require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const [base,promptFile,out,referenceFile]=process.argv.slice(2);
assert(base&&promptFile&&out&&process.env.CONE_CDP,'base, prompt, output and guarded CONE_CDP required');
fs.mkdirSync(out,{recursive:true});
const prompt=fs.readFileSync(promptFile,'utf8'),promptSha=crypto.createHash('sha256').update(prompt).digest('hex');
(async()=>{
 const browser=await chromium.connectOverCDP(process.env.CONE_CDP),errors=[],turns=[];
 let page;
 try{
  page=await browser.newPage({viewport:{width:1440,height:1000}});
  page.on('pageerror',error=>errors.push(error.message));
  await page.addInitScript(()=>{
   localStorage.setItem('LlamaCppWebui.config',JSON.stringify({max_tokens:64,keepStatsVisible:true,
    custom:JSON.stringify({reasoning_effort:'none',seed:123456,tool_choice:'none'})}));
   localStorage.setItem('LlamaCppWebui.userOverrides',JSON.stringify(['max_tokens','keepStatsVisible','custom']));
   window.__coneWire=[];const fetch=window.fetch;
   window.fetch=async(...args)=>{
    const response=await fetch(...args);
    if(String(args[0]).includes('/chat/completions')){
     const row={start:performance.now(),events:[]};window.__coneWire.push(row);
     (async()=>{const reader=response.clone().body.getReader(),decoder=new TextDecoder();let tail='';
      for(;;){const {done,value}=await reader.read();if(done){row.end=performance.now();break;}
       tail+=decoder.decode(value,{stream:true});const lines=tail.split('\n');tail=lines.pop();
       for(const line of lines)if(line.startsWith('data: ')){let data=line.slice(6);try{data=JSON.parse(data)}catch{}
        row.events.push({at:performance.now(),data});}
      }
     })().catch(error=>row.error=String(error));
    }
    return response;
   };
  });
  await page.goto(base,{waitUntil:'domcontentloaded',timeout:120000});
  await page.locator('#sl-generation-toolbar').waitFor({state:'visible'});
  await page.getByRole('button',{name:'Generation phase settings',exact:true}).click();
  await page.locator('#sl-generation-enabled').uncheck();
  for(const [name,value] of [['Temperature','0'],['Top p','1'],['Top k','1'],['Min p','0'],['Speed bias','.22']])
   await page.getByRole('spinbutton',{name:'All phases '+name,exact:true}).fill(value);
  await page.screenshot({path:path.join(out,'samplers.png')});
  await page.getByRole('button',{name:'Close generation settings',exact:true}).click();
  async function send(text,label){
   const before=await page.evaluate(()=>window.__coneWire.length),samples=[];let response,request,lastBucket=-1,lastMark='',lastActivity=Date.now();
   const listener=async r=>{if(r.url().endsWith('/chat/completions')){
    request=r.request().postDataJSON();response={status:r.status(),text:await r.text()};
   }};
   page.on('response',listener);
   try{
    await page.locator('textarea').fill(text);await page.locator('textarea').press('Enter');
    while(!response){
     const sample=await page.evaluate(()=>({at:performance.now(),stats:document.querySelector('.chat-processing-info-content')?.innerText||'',
      bars:[...document.querySelectorAll('[role=progressbar],progress')].map(e=>({value:e.getAttribute('aria-valuenow'),max:e.getAttribute('aria-valuemax'),style:e.getAttribute('style')})),
      wire:window.__coneWire.at(-1)?.events.length||0,progress:window.__coneWire.at(-1)?.events.map(e=>e.data?.prompt_progress).filter(Boolean).at(-1)}));
     samples.push(sample);const mark=JSON.stringify([sample.stats,sample.bars,sample.wire]);
     if(mark!==lastMark){lastMark=mark;lastActivity=Date.now();}
     if(Date.now()-lastActivity>600000)throw Error('WebUI made no progress for600s');
     if(sample.progress?.total){const bucket=Math.floor(4*sample.progress.processed/sample.progress.total);
      if(bucket!==lastBucket){lastBucket=bucket;await page.screenshot({path:path.join(out,label+'-progress-'+bucket+'.png')});}}
     await page.waitForTimeout(100);
    }
   }finally{page.off('response',listener);}
   assert.equal(response.status,200);await page.waitForTimeout(200);
   const wire=await page.evaluate(i=>window.__coneWire[i],before);
   const events=response.text.split('\n').filter(line=>line.startsWith('data: ')&&line!=='data: [DONE]').map(line=>JSON.parse(line.slice(6)));
   assert(!events.some(event=>event.error),'server error in actual SSE');assert(!wire?.error,'SSE reader failed');
   const answer=events.flatMap(event=>event.choices||[]).map(choice=>choice.delta?.content||'').join('').trim();
   const timings=events.filter(event=>event.timings).at(-1)?.timings;assert(timings,'real final timing fields required');
   const content=wire.events.filter(event=>event.data?.choices?.some(choice=>choice.delta?.content));
   if(label==='recall')assert(content.length>1&&content.at(-1).at>content[0].at,'incremental initial reply required');
   const progress=wire.events.map(event=>event.data?.prompt_progress).filter(Boolean);
   if(label==='recall')assert(new Set(progress.map(event=>event.processed)).size>1,'advancing real prefill progress required');
   const stats=await page.locator('.chat-processing-info-content').innerText();
   assert(/Context: \d+\/\d+/.test(stats)&&!stats.includes('?'),'numeric context/timing display required');
   assert((timings.predicted_per_second||timings.predicted_n*1000/timings.predicted_ms)>0,'numeric decode throughput required');
   const row={label,answer,timings,stats,content_events:content.length,stream_span_ms:content.length?content.at(-1).at-content[0].at:0};
   fs.writeFileSync(path.join(out,label+'-sse.json'),JSON.stringify({request,response,wire,samples},null,2));
   await page.screenshot({path:path.join(out,label+'-final.png')});turns.push(row);console.log(JSON.stringify(row));return answer;
  }
  assert.match(await send(prompt,'recall'),/AMBER[\s_]*OTTER[\s_]*731/i);
  assert.match(await send('Repeat the secret access phrase for station Cedarlight, with no other text.','followup'),/AMBER[\s_]*OTTER[\s_]*731/i);
  assert.match(await send('What is the three-digit number in that phrase? Answer only the number.','number'),/731/);
  assert.deepEqual(errors,[]);
  let exactReference=null;
  if(referenceFile){const reference=JSON.parse(fs.readFileSync(referenceFile));
   assert.equal(reference.prompt_sha256,promptSha,'same actual prompt bytes required');
   assert.deepEqual(turns.map(row=>row.answer),reference.turns.map(row=>row.answer),'single/paired recall answers must match exactly');
   assert.equal(turns[0].timings.prompt_n,reference.turns[0].timings.prompt_n,'same native token count required');exactReference=true;
  }
  const proof={status:'PASS',scope:'actual stack WebUI exact-cone recall',prompt_sha256:promptSha,
   deterministic_samplers:{temperature:0,top_p:1,top_k:1,min_p:0,quality_speed:.22,seed:123456,max_tokens:64},
   turns,exact_reference_answers:exactReference,errors};
  fs.writeFileSync(path.join(out,'proof.json'),JSON.stringify(proof,null,2));console.log(JSON.stringify(proof));
 }finally{if(page)await page.close();await browser.close();}
})().catch(error=>{fs.writeFileSync(path.join(out,'failure.txt'),String(error.stack));console.error(error);process.exitCode=1;});
