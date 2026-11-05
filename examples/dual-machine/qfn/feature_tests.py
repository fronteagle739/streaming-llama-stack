"""Explicit real-API qualification after a native handover; no GUI dependency."""
import base64
import io
import json
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request


def chat(url,body,events_path,cancel_chunks=None):
    body={**body,'stream':True}
    request=urllib.request.Request(url+'/v1/chat/completions',data=json.dumps(body).encode(),
        headers={'Content-Type':'application/json'})
    content=[];reasoning=[];calls={};finish=None;count=0;timings=None;mtp=None;image_prefill=None
    with urllib.request.urlopen(request,timeout=300) as response,events_path.open('x',encoding='utf8') as log:
        for raw in response:
            if not raw.startswith(b'data:'):continue
            payload=raw[5:].strip()
            if payload==b'[DONE]':break
            event=json.loads(payload);log.write(json.dumps(event,ensure_ascii=False)+'\n');log.flush()
            if 'error' in event:raise RuntimeError(str(event['error']))
            timings=event.get('timings',timings);mtp=event.get('qfn_mtp',mtp)
            image_prefill=event.get('qfn_image_prefill',image_prefill)
            for choice in event.get('choices',[]):
                delta=choice.get('delta',{})
                if delta.get('content'):content.append(delta['content']);count+=1
                thought=delta.get('reasoning_content',delta.get('reasoning'))
                if thought:reasoning.append(thought);count+=1
                for tool in delta.get('tool_calls',[]):
                    index=tool.get('index',0);entry=calls.setdefault(index,dict(id='',type='function',function=dict(name='',arguments='')))
                    if tool.get('id'):entry['id']=tool['id']
                    for field in ('name','arguments'):
                        entry['function'][field]+=tool.get('function',{}).get(field,'')
                finish=choice.get('finish_reason') or finish
            if cancel_chunks and count>=cancel_chunks:break
    message=dict(role='assistant',content=''.join(content))
    if reasoning:message['reasoning_content']=''.join(reasoning)
    if calls:message['tool_calls']=[calls[i] for i in sorted(calls)]
    return dict(message=message,finish_reason=finish,chunks=count,timings=timings,mtp=mtp,
        image_prefill=image_prefill,cancelled_by_client=bool(cancel_chunks and count>=cancel_chunks))


def run(url,output,producer_pack,model,phrase,token_fixture,proof_program=None,python=None,prefix_tokens=100000,rtx_images=False,image_fixture=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    # The complete source tokenizer remains available without a model load.
    from scripts.slpack import ModelFile
    metadata=json.loads(ModelFile(producer_pack).read('meta'))
    vocab=metadata['gguf_kv']['tokenizer.ggml.tokens']
    byte_values=list(range(33,127))+list(range(161,173))+list(range(174,256))
    codepoints=byte_values[:];extra=0
    for value in range(256):
        if value not in byte_values:byte_values.append(value);codepoints.append(256+extra);extra+=1
    decoder={chr(c):b for b,c in zip(byte_values,codepoints)}
    fixture=Path(token_fixture)
    # Use a long real-text prefix and preserve full chat history between cases.
    import struct
    raw=fixture.read_bytes() if fixture.exists() else None
    if raw is None:raise ValueError('Explicit task prompt fixture is required for feature qualification')
    tokens=struct.unpack('<'+'i'*(len(raw)//4),raw)
    if not 1<=prefix_tokens<=len(tokens):raise ValueError('Feature source-prefix token count exceeds its explicit fixture')
    encoded=bytearray()
    for token in tokens[:prefix_tokens]:
        for character in vocab[token]:
            if character in decoder:encoded.append(decoder[character])
            else:encoded.extend(character.encode('utf8'))
    text=encoded.decode('utf8',errors='strict')
    if '<|im_start|>user\n' in text:text=text.split('<|im_start|>user\n',1)[1]
    text=text.replace('<|im_end|>','').replace('<|im_start|>','')
    long_user='Remember the password '+phrase+'. The following is source text for later questions.\n'+text+'\nReply with the single word READY.'
    messages=[dict(role='user',content=long_user)]
    results=[];previous='native handover benchmark'
    def case(name,body,check,append=True,cancel_chunks=None):
        nonlocal previous
        # Preserve the exact explicit test request for a failed-turn reproducer.
        (output/(name+'.request.json')).write_text(json.dumps(body,ensure_ascii=False),encoding='utf8')
        settings={k:v for k,v in body.items() if k not in ('messages',)}
        record=dict(case=name,model=model,temperature=body.get('temperature',1),settings=settings,
            previous=previous,status='FAIL',actual='',source_prefix_tokens=prefix_tokens,
            image_prefill='RTX' if rtx_images else 'Mac-local')
        try:
            result=chat(url,body,output/(name+'.events.jsonl'),cancel_chunks)
            record.update(result=result,actual=result['message'].get('content',''))
            check(result)
            if rtx_images and name in ('vision-after-handover','vision-inspection-card'):
                if result.get('image_prefill')!='rtx':raise AssertionError('Image answer lacks actual RTX image-prefill telemetry')
            record['status']='PASS'
            if append:messages.append(result['message'])
        except Exception as error:
            record['error']=str(error)
            if isinstance(error,urllib.error.HTTPError):
                record['error_body']=error.read().decode('utf8',errors='replace')
        target=output/(name+'.json');target.write_text(json.dumps(record,indent=2,ensure_ascii=False),encoding='utf8')
        results.append(record)
        if proof_program:
            subprocess.run([python,'-B',str(proof_program),'--kind','feature','--result',str(target),
                '--name','feature-'+output.parent.name+'-'+name],check=True)
        previous=name+' '+record['status']
        return record
    def require(condition,why):
        if not condition:raise AssertionError(why)
    def body(**kwargs):
        return dict(messages=list(messages),temperature=1,top_k=20,top_p=.95,min_p=0,seed=1,
            max_tokens=128,thinking=False,**kwargs)
    case('chat-streaming',body(),lambda r:require(r['chunks']>0 and bool(r['message']['content']),'No real streaming content'))
    messages.append(dict(role='user',content='What is the password? Return only the password.'))
    case('short-followup',body(),lambda r:require(phrase in r['message']['content'],'Recall failed after native warm append'))
    messages.append(dict(role='user',content='Retain the source context and this additional passage:\n'+text[:30000]+
        '\nWhat is the password? Return only the password.'))
    def mid_append_check(r):
        timing=r.get('timings') or {}
        require(phrase in r['message']['content'],'Recall failed after medium local append')
        require(timing.get('cache_n',0)>=100000 and 4096<timing.get('prompt_n',0)<=11776,
            'Medium follow-up did not reuse the live Mac frontier across the old 4096-row threshold')
    case('mid-followup',body(),mid_append_check)
    messages.append(dict(role='user',content='Describe the mood of the source passage in one short sentence.'))
    sampled=body()
    sampled.update(top_k=40,top_p=.9,min_p=.02,seed=17,typical_p=.97,repeat_penalty=1.05,
        repeat_last_n=-1,presence_penalty=.1,frequency_penalty=.1,
        dynatemp_range=.1,dynatemp_exponent=1.1,xtc_probability=.03,xtc_threshold=.1,
        dry_multiplier=.2,dry_base=1.75,dry_allowed_length=2,dry_penalty_last_n=128,
        top_n_sigma=2,logit_bias={'200000':1.0,'200001':-1.0},
        samplers=['penalties','dry','top_k','typ_p','top_p','min_p','temperature','xtc','top_n_sigma'])
    case('all-samplers',sampled,lambda r:require(bool(r['message']['content']),'Sampler-chain request returned no content'))
    messages.append(dict(role='user',content='Reply with a short greeting.'))
    large_k=body();large_k.update(top_k=2048,top_p=1,max_tokens=16)
    case('large-top-k',large_k,lambda r:require(bool(r['message']['content']),'Large top-k request returned no content'))
    messages.append(dict(role='user',content='Think through whether 19 times 23 is greater than 400, then answer briefly.'))
    thinking=body();thinking.update(thinking=True,max_tokens=256)
    case('thinking',thinking,lambda r:require(bool(r['message'].get('reasoning_content')),'No reasoning stream was emitted'))
    messages.append(dict(role='user',content='Call get_temperature with city London, then report the measured temperature from its tool response.'))
    tools=body();tools.update(max_tokens=256,tools=[dict(type='function',function=dict(name='get_temperature',
        description='Return the measured temperature for a city.',parameters=dict(type='object',
        properties=dict(city=dict(type='string')),required=['city'])))],tool_choice='auto')
    def tool_check(r):
        calls=r['message'].get('tool_calls',[])
        require(bool(calls),'The model did not emit an actual tool call')
        require(calls[0]['function']['name']=='get_temperature','Tool name differs')
        args=json.loads(calls[0]['function']['arguments'])
        require(str(args.get('city','')).lower()=='london','Tool arguments differ')
    tool=case('tool-call',tools,tool_check)
    if tool['status']=='PASS':
        call=tool['result']['message']['tool_calls'][0]
        messages.append(dict(role='tool',tool_call_id=call['id'],content='{"city":"London","temperature_c":12}'))
        follow=body();follow.update(tools=tools['tools'],max_tokens=128)
        case('tool-result',follow,lambda r:require('12' in r['message']['content'],'Tool result was not used'))
    messages.append(dict(role='user',content=('Continue to retain all previous context. Here is extra source text: '+text[:50000]+'\n')+
        'What is the password? Return only the password.'))
    case('long-followup',body(),lambda r:require(phrase in r['message']['content'],'Long follow-up recall failed'))
    messages.append(dict(role='user',content='Write a very long detailed account of the source passage and continue for many pages.'))
    cancelled=body();cancelled.update(max_tokens=2048)
    case('cancel',cancelled,lambda r:require(r['cancelled_by_client'],'No streaming output was cancelled'),append=False,cancel_chunks=8)
    # Keep the last completed transcript, without the intentionally cancelled turn.
    messages.pop();messages.append(dict(role='user',content='Reply with READY.'))
    time.sleep(1)
    case('cancel-recovery',body(),lambda r:require(bool(r['message']['content']),'API did not recover after cancellation'))
    from PIL import Image
    image=io.BytesIO();Image.new('RGB',(224,224),(255,0,0)).save(image,format='PNG')
    url_image='data:image/png;base64,'+base64.b64encode(image.getvalue()).decode()
    messages.append(dict(role='user',content=[dict(type='text',text='What single color is this square? Answer with the color only.'),
        dict(type='image_url',image_url=dict(url=url_image))]))
    case('vision-after-handover',body(),lambda r:require('red' in r['message']['content'].lower(),'The real image was not correctly identified'))
    messages.append(dict(role='user',content='What was the color of the square in the previous image? Reply with the color only.'))
    case('vision-followup',body(),lambda r:require('red' in r['message']['content'].lower(),'Image follow-up lost the original visual context'))
    if image_fixture:
        image_url='data:image/png;base64,'+base64.b64encode(Path(image_fixture).read_bytes()).decode()
        messages.append(dict(role='user',content=[dict(type='text',text='Read the inspection number and the code printed in this image; answer with both only.'),
            dict(type='image_url',image_url=dict(url=image_url))]))
        case('vision-inspection-card',body(),lambda r:require('731' in r['message']['content'] and 'otter' in r['message']['content'].lower(),'Image text was not read correctly'))
    messages.append(dict(role='user',content='Output exactly ALPHA<STOP>OMEGA with no explanation.'))
    stopping=body();stopping.update(stop=['<STOP>'],max_tokens=64)
    case('stop-sequence',stopping,lambda r:require('ALPHA' in r['message']['content'] and
        'OMEGA' not in r['message']['content'] and r['finish_reason']=='stop','Stop sequence was not demonstrated'))
    messages.append(dict(role='user',content='Repeat the password remembered at the beginning, with no explanation.'))
    case('stop-followup',body(),lambda r:require(phrase in r['message']['content'],'Stop repair lost the retained prompt or answer state'))
    # A different short conversation must retire the paired identity while
    # keeping the full ordinary local API, then retain its own new context.
    fresh=dict(messages=[dict(role='user',content='Remember the code violet-829. Reply READY.')],
        temperature=1,top_k=20,top_p=.95,seed=1,max_tokens=32,thinking=False)
    first=case('new-short-conversation',fresh,lambda r:require(bool(r['message']['content']),
        'New short conversation failed after the paired frontier'),append=False)
    if first['status']=='PASS':
        follow={**fresh,'messages':fresh['messages']+[first['result']['message'],
            dict(role='user',content='Repeat the code only.')]}
        case('new-short-followup',follow,lambda r:require('violet-829' in r['message']['content'],
            'New short conversation lost its own follow-up state'),append=False)
    summary=dict(model=model,cases=results,passed=all(r['status']=='PASS' for r in results),
        ordinary_sampler_chain_can_select_serial_native_decode=True,source_prefix_tokens=prefix_tokens)
    (output/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf8')
    if not summary['passed']:raise ValueError('Native API feature failures: '+', '.join(r['case'] for r in results if r['status']!='PASS'))
    return summary
