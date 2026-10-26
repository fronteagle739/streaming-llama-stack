"""One isolated call into the original ds4 V4.1 vision encoder; no LLM load."""
from contextlib import ExitStack
import ctypes
import json
import os
from pathlib import Path
import struct
import sys


def run(row):
    if os.name!='nt':raise ValueError('This vision bridge requires the CUDA single engine')
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
