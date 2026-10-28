"""Explicit GPU byte oracle for the versioned ds4 host-cache bridge.

Run only inside an exclusive GPU lease and an admitted memory budget. This
fixture never opens a model. It checks LRU wraps, smaller mixed-quant extents,
padding, growth/shrink teardown and exact device/host/device bytes.
"""
import ctypes as C
import os
from pathlib import Path


def check(dll_path, cuda_directory):
    handles=[os.add_dll_directory(str(Path(dll_path).resolve().parent)),
             os.add_dll_directory(str(Path(cuda_directory).resolve()))]
    cuda=C.CDLL(str(Path(cuda_directory)/'cudart64_13.dll'))
    dll=C.CDLL(str(Path(dll_path).resolve()))
    ptr=C.c_void_p; u32=C.c_uint32; u64=C.c_uint64
    resize=C.CFUNCTYPE(u32,u64,u64,u64)
    find=C.CFUNCTYPE(C.c_int,u32,u32)
    transfer=C.CFUNCTYPE(C.c_int,u32,C.POINTER(ptr),C.POINTER(u64))
    store=C.CFUNCTYPE(C.c_int,u32,u32,C.POINTER(ptr),C.POINTER(u64))
    clear=C.CFUNCTYPE(None)
    class API(C.Structure):
        _fields_=[('resize',resize),('find',find),('upload',transfer),('store',store),('clear',clear)]
    dll.ds4x_host_cache_v1.argtypes=[C.POINTER(API)];dll.ds4x_host_cache_v1.restype=C.c_int
    cuda.cudaMalloc.argtypes=[C.POINTER(ptr),C.c_size_t]
    cuda.cudaMemcpy.argtypes=[ptr,ptr,C.c_size_t,C.c_int]
    cuda.cudaFree.argtypes=[ptr]
    api=API();assert dll.ds4x_host_cache_v1(C.byref(api))==1
    device=[ptr() for _ in range(3)];comparisons=0
    def ok(code):assert code==0,code
    def put(key,sizes):
        for k,n in enumerate(sizes):
            data=bytes((key*31+k*7+i)%256 for i in range(n))
            buf=C.create_string_buffer(data);ok(cuda.cudaMemcpy(device[k],buf,n,1))
        assert api.store(key//384,key%384,(ptr*3)(*[x.value for x in device]),(u64*3)(*sizes))==1
    def get(key,sizes):
        nonlocal comparisons
        slot=api.find(key//384,key%384);assert slot>=0
        # Clear complete destination; the upload must not write past its extent.
        zero=C.create_string_buffer(1024)
        for d in device:ok(cuda.cudaMemcpy(d,zero,1024,1))
        assert api.upload(slot,(ptr*3)(*[x.value for x in device]),(u64*3)(*sizes))==1
        for k,n in enumerate(sizes):
            result=C.create_string_buffer(1024);ok(cuda.cudaMemcpy(result,device[k],1024,2))
            expected=bytes((key*31+k*7+i)%256 for i in range(n))+bytes(1024-n)
            assert result.raw==expected,(key,k,n);comparisons+=1
    try:
        for d in device:ok(cuda.cudaMalloc(C.byref(d),1024))
        assert api.resize(3*(256*2+512),256,512)==3
        for key in (0,1,2):put(key,(256,256,512))
        get(0,(256,256,512));put(3,(256,256,512))
        assert api.find(0,1)==-1 and api.find(0,0)>=0
        put(384,(64,64,128));get(384,(64,64,128))
        for key in range(385,405):put(key,(64,64,128));get(key,(64,64,128))
        assert api.resize(1024,256,512)==1
        assert api.find(1,20)==-1
        put(700,(256,256,512));get(700,(256,256,512))
        assert api.resize(0,256,512)==0 and api.find(1,316)==-1
        return dict(status='PASS',byte_comparisons=comparisons,model_opened=False)
    finally:
        api.clear()
        for d in device:
            if d.value:ok(cuda.cudaFree(d))
        for h in handles:h.close()
