"""One isolated call into the original ds4 V4.1 vision encoder; no LLM load."""
from contextlib import ExitStack
import ctypes
import json
import os
from pathlib import Path
import struct
import subprocess
import sys


def run(row):
    if os.name!='nt':raise ValueError('This vision bridge requires the CUDA single engine')
    if row.get('family')=='qfn':
        # The native reference decodes PNG/JPEG. Retain the existing image API's
        # other codecs through Pillow, without converting those reference bytes.
        image=Path(row['image']);signature=image.read_bytes()[:8]
        if not (signature.startswith(b'\x89PNG\r\n\x1a\n') or signature.startswith(b'\xff\xd8') or signature.startswith(b'P6')):
            from PIL import Image,ImageOps
            normalized=image.with_suffix('.normalized.png')
            with Image.open(image) as source:
                source=ImageOps.exif_transpose(source)
                if source.mode in ('RGBA','LA') or 'transparency' in source.info:
                    rgba=source.convert('RGBA');background=Image.new('RGBA',rgba.size,'white');background.alpha_composite(rgba);source=background.convert('RGB')
                else:source=source.convert('RGB')
                with normalized.open('xb') as target:source.save(target,format='PNG')
            image=normalized
        args=[row['library'],'--vision-encode','--vision-model',row['encoder'],
            '--vision-image',str(image),'--vision-output',row['output'],
            '--vision-backend',row['backend'],'--vision-min-tokens',str(row['min_tokens']),
            '--vision-max-tokens',str(row['max_tokens']),'--vision-threads',str(row['threads']),
            '--vision-flash-attn',str(row['flash_attention']),
            '--host-budget-gib',str(row['host_budget_gib']),'--protect-host-working-set','1']
        result=subprocess.run(args,capture_output=True,text=True,encoding='utf8',errors='replace',
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        sys.stderr.write(result.stderr)
        lines=[json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
        if result.returncode or len(lines)!=1 or lines[0].get('event')!='qfn_vision_encoded':
            raise RuntimeError('Isolated QFN vision worker failed: '+result.stdout[-1000:])
        native=lines[0]
        with Path(row['output']).open('rb') as f:
            magic,version,count,width=struct.unpack('<4I',f.read(16));fingerprint=f.read(32).hex()
        if (magic,version,width)!=(0x49564c53,1,2560) or count!=native['count'] or Path(row['output']).stat().st_size!=48+count*width*4:
            raise ValueError('QFN encoder returned an invalid projected image block')
        return dict(native,count=count,width=width,fingerprint=fingerprint,worker_pid=os.getpid())
    with ExitStack() as handles:
        for directory in row['libraries']:
            if Path(directory).is_dir():handles.enter_context(os.add_dll_directory(directory))
        library=ctypes.CDLL(row['library']);encode=library.ds4x_vision_file
        encode.argtypes=[ctypes.c_char_p]*4+[ctypes.c_char_p,ctypes.c_size_t];encode.restype=ctypes.c_int
        error=ctypes.create_string_buffer(2048)
        arguments=[row[k].encode('utf-8') for k in ('model','encoder','image','output')]
        if encode(*arguments,error,len(error))!=1:raise RuntimeError(error.value.decode('utf-8','replace') or 'Native image encoding failed')
    with Path(row['output']).open('rb') as f:
        magic,version,count,width=struct.unpack('<4I',f.read(16));fingerprint=f.read(32).hex()
    if (magic,version,width)!=(0x49564c53,1,5120) or not 0<count<=32768 or Path(row['output']).stat().st_size!=48+count*width*4:
        raise ValueError('Native encoder returned an invalid projected image block')
    return dict(count=count,width=width,fingerprint=fingerprint,worker_pid=os.getpid())


if __name__=='__main__':
    try:print('SLSTACK_VISION '+json.dumps(run(json.loads(sys.stdin.readline()))),flush=True)
    except BaseException as exc:
        print('SLSTACK_VISION '+json.dumps(dict(error=str(exc))),flush=True);raise
