"""GPU proof: upload pixels mid-chat, release encoder, restore KV, append image."""
import argparse
import base64
import json
import os
from pathlib import Path
import struct
import sys
import zlib

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from slstack.config import load
from slstack.combo_single import SingleEngine


def png(color):
    width=height=384
    def chunk(kind,data):return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)
    raw=b''.join(b'\0'+bytes(color)*width for _ in range(height))
    return b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>2I5B',width,height,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(raw))+chunk(b'IEND',b'')


def turn(engine,body):
    final={};metrics={}
    def emit(kind,*values):
        if kind=='final':final.update(values[0])
    engine.run(body,emit,metrics)
    return dict(content=final['parsed']['content'],reasoning=final['parsed']['reasoning'],metrics=metrics)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();cfg=load(args.config);cfg['serving_mode']='single_engine'
    cfg.setdefault('env_overrides',{})['LLAMADECODE_DS4_DECODE']='1'
    os.environ['CUDA_VISIBLE_DEVICES']=cfg['gpu_uuid'];os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    engine=SingleEngine(cfg);result={}
    try:
        body=dict(messages=[dict(role='user',content='Reply only READY.')],reasoning_effort='none',temperature=0,max_tokens=8)
        one=turn(engine,body);result['before_image']=one
        pixels=png((240,10,10));uri='data:image/png;base64,'+base64.b64encode(pixels).decode()
        following=dict(body,max_tokens=16,messages=body['messages']+[dict(role='assistant',content=one['content']),
            dict(role='user',content=[dict(type='image_url',image_url=dict(url=uri)),dict(type='text',text='What is the dominant color of this image? Answer with only the color name.')])])
        two=turn(engine,following);result['after_image']=two
        logs=list((Path(cfg['runtime_root'])/'vision').glob('*.log'));text='\n'.join(p.read_text(errors='replace') for p in logs)
        result['directstorage_gpu_fetch']='backend=directstorage destination=gpu' in text
        result['encoder_weights_released']='encoder_weights_released=1' in text
        result['resumed_prefix']=two['metrics'].get('prefix_source')=='checkpoint' and two['metrics'].get('reused_prefix_tokens',0)>0
        result['pixel_check']='red' in (two['content'] or '').lower()
        result['status']='PASS' if all(result[k] for k in ('directstorage_gpu_fetch','encoder_weights_released','resumed_prefix','pixel_check')) else 'FAIL'
    except BaseException as exc:result.update(status='FAIL',error=repr(exc));raise
    finally:
        engine.close();Path(args.out).write_text(json.dumps(result,indent=2,default=str),encoding='utf-8')
        print(json.dumps({k:v for k,v in result.items() if k not in ('before_image','after_image')}),flush=True)
    return result['status']=='PASS'


if __name__=='__main__':raise SystemExit(0 if main() else 1)
