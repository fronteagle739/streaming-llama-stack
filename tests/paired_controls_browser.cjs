// A headless real-browser CPU fixture; responses are explicitly synthetic.
const {chromium}=require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
(async()=>{
 const [base,out]=process.argv.slice(2),errors=[],requests=[];
 const browser=await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_EXECUTABLE || undefined,
   args:['--disk-cache-dir='+path.join(out,'browser-cache')]});
 try{
   const page=await browser.newPage({viewport:{width:1280,height:900}});
   page.on('pageerror',e=>errors.push(e.message));
   await page.route('**/chat/completions',async route=>{
     const body=route.request().postDataJSON();
     const reply=await page.request.post(base+'/test/relay',{data:body});assert(reply.ok());
     requests.push(await reply.json());
     const head={id:'cpu-only-fixture',object:'chat.completion.chunk',created:1,model:'DSV4.1-1017'};
     const rows=[{...head,choices:[{index:0,delta:{role:'assistant',content:'CPU fixture response.'},finish_reason:null}]},
                 {...head,choices:[{index:0,delta:{},finish_reason:'stop'}]}];
     await route.fulfill({contentType:'text/event-stream',body:rows.map(x=>'data: '+JSON.stringify(x)+'\n\n').join('')+'data: [DONE]\n\n'});
   });
   await page.goto(base,{waitUntil:'networkidle'});
   await page.locator('#sl-generation-toolbar').waitFor({state:'visible'});
   await page.getByRole('button',{name:'Generation phase settings',exact:true}).click();
   await page.locator('#sl-generation-enabled').uncheck();
   for(const [name,value] of [['Temperature','.83'],['Top p','.87'],['Top k','37'],['Min p','.06'],['Speed bias','.22']])
     await page.getByRole('spinbutton',{name:'All phases '+name,exact:true}).fill(value);
   let saved=false;
   for(let attempt=0;attempt<100;attempt++){
     const r=await (await page.request.get(base+'/slstack/generation')).json();
     saved=!r.settings.per_phase&&r.settings.value===.22&&r.settings.sampler.temperature===.83&&r.settings.sampler.top_p===.87&&r.settings.sampler.top_k===37&&r.settings.sampler.min_p===.06;
     if(saved)break;await page.waitForTimeout(50);
   }
   assert(saved,'typed sampler values must be saved, including fractional values outside the old step lattice');
   await page.screenshot({path:path.join(out,'paired-controls.png')});
   await page.getByRole('button',{name:'Close generation settings',exact:true}).click();
   await page.locator('textarea').fill('CPU fixture only: retain these sampler settings.');
   await page.locator('textarea').press('Enter');
   await page.waitForFunction(()=>document.body.innerText.includes('CPU fixture response.'));
   const row=requests[0];assert(row);
   for(const [key,value] of Object.entries({temperature:.83,top_p:.87,top_k:37,min_p:.06,quality_speed:.22,max_tokens:-1}))assert.equal(row[key],value,key);
   assert.deepEqual(row.samplers,['top_k','typ_p','top_p','min_p','temperature']);
   // Force a toolbar save to remain in flight while Send is pressed. The
   // old fetch wrapper sent the previous revision in this exact race.
   await page.route('**/slstack/generation',async route=>{
     if(route.request().method()==='POST')await new Promise(resolve=>setTimeout(resolve,300));
     await route.continue();
   });
   await page.getByRole('button',{name:'Generation phase settings',exact:true}).click();
   await page.getByRole('spinbutton',{name:'All phases Temperature',exact:true}).fill('.73');
   await page.getByRole('button',{name:'Close generation settings',exact:true}).click();
   await page.locator('textarea').fill('CPU fixture only: send while the toolbar save is pending.');
   await page.locator('textarea').press('Enter');
   for(let i=0;i<100&&requests.length<2;i++)await page.waitForTimeout(50);
   assert.equal(requests[1]?.temperature,.73,'Send must await the pending sampler save');
   assert.deepEqual(errors,[]);
   const proof={status:'PASS',scope:'headless CPU WebUI fixture',model_opened:false,native_decode:false,
      sampler_values:{temperature:row.temperature,top_p:row.top_p,top_k:row.top_k,min_p:row.min_p,quality_speed:row.quality_speed},
      max_tokens:row.max_tokens,samplers:row.samplers,pending_save_before_send:true,errors};
   fs.writeFileSync(path.join(out,'proof.json'),JSON.stringify(proof,null,2));console.log(JSON.stringify(proof));
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
