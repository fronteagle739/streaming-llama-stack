import ctypes as c,json,os
from pathlib import Path
import argparse
parser=argparse.ArgumentParser(description='GLM external CUDA state ownership checks; no decode claim')
parser.add_argument('--model',type=Path,required=True)
parser.add_argument('--library',type=Path,required=True)
parser.add_argument('--cuda-runtime',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
# Invoke through OwnedProcess with explicit host/commit budgets and a GPU lease.
os.add_dll_directory(str(args.cuda_runtime.resolve().parent))
os.add_dll_directory(str(args.library.resolve().parent))
d=c.CDLL(str(args.library.resolve()));cu=c.CDLL(str(args.cuda_runtime.resolve()))
P=c.c_void_p;U=c.c_uint32;Q=c.c_uint64
class Span(c.Structure):_fields_=[('data',P),('bytes',Q)]
class Layer(c.Structure):_fields_=[(x,Span) for x in ('conv','recurrent','latent','pooled','tail_key','tail_gate')]
class State(c.Structure):_fields_=[('abi',U),('layers',U),('context',U),('position',U),('layer',c.POINTER(Layer))]
d.ds4glm_open.argtypes=[c.c_char_p,U,Q,c.c_char_p,c.c_size_t];d.ds4glm_open.restype=c.c_int
d.ds4glm_bind_state.argtypes=[c.POINTER(State),c.c_char_p,c.c_size_t];d.ds4glm_bind_state.restype=c.c_int
d.ds4glm_decode.argtypes=[c.c_int32,U,c.POINTER(c.c_float)];d.ds4glm_decode.restype=c.c_int
d.ds4glm_close.argtypes=[];d.ds4glm_close.restype=None
cu.cudaMalloc.argtypes=[c.POINTER(P),c.c_size_t];cu.cudaMalloc.restype=c.c_int
cu.cudaMemset.argtypes=[P,c.c_int,c.c_size_t];cu.cudaMemset.restype=c.c_int
cu.cudaMemcpy.argtypes=[P,P,c.c_size_t,c.c_int];cu.cudaMemcpy.restype=c.c_int
cu.cudaFree.argtypes=[P];cu.cudaFree.restype=c.c_int
path=os.fsencode(args.model.resolve())
error=c.create_string_buffer(1024);cases=[];owners=[];opened=False
try:
    assert not d.ds4glm_open(path,0,256*2**20,error,len(error));cases.append('invalid context refused')
    assert not d.ds4glm_open(path,128,1,error,len(error));cases.append('one-byte scratch budget refused')
    assert d.ds4glm_open(path,128,256*2**20,error,len(error)),error.value;opened=True
    bank=(Layer*45)();state=State(1,45,128,0,bank)
    assert not d.ds4glm_bind_state(c.byref(state),error,len(error));cases.append('missing owners refused')
    for il,layer in enumerate(bank):
        sizes=(3*3*64*128*4,64*128*128*4,0,0,0,0) if il%4!=3 else (0,0,128*512*2,32*128*2,4*128*4,4*128*4)
        for field,n in zip(('conv','recurrent','latent','pooled','tail_key','tail_gate'),sizes):
            if n:
                ptr=P();assert cu.cudaMalloc(c.byref(ptr),n)==0
                assert cu.cudaMemset(ptr,0,n)==0;owners.append(ptr)
                setattr(layer,field,Span(ptr,n))
    marker=c.c_float(43.0);assert cu.cudaMemcpy(bank[0].conv.data,c.byref(marker),4,1)==0
    assert d.ds4glm_bind_state(c.byref(state),error,len(error)),error.value
    cases.append('45-layer native state bound')
    bank[0].conv.bytes-=1
    assert not d.ds4glm_bind_state(c.byref(state),error,len(error));bank[0].conv.bytes+=1
    cases.append('undersized owner refused before mutation')
    state.position=128
    assert not d.ds4glm_bind_state(c.byref(state),error,len(error));state.position=0
    cases.append('out-of-context frontier refused')
    logits=(c.c_float*154880)()
    assert not d.ds4glm_decode(1,1,logits);cases.append('non-monotonic decode refused')
    d.ds4glm_close();opened=False
    after=c.c_float();assert cu.cudaMemcpy(c.byref(after),bank[0].conv.data,4,2)==0
    assert after.value==43.0;cases.append('close preserves borrowed allocation and contents')
    result=dict(status='PASS',cases=cases,owner_bytes=sum(x.bytes for layer in bank for x in [layer.conv,layer.recurrent,layer.latent,layer.pooled,layer.tail_key,layer.tail_gate]),
                trained_model_decode=False,weight_registration=False)
    args.output.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
finally:
    if opened:d.ds4glm_close()
    for ptr in reversed(owners):assert cu.cudaFree(ptr)==0
