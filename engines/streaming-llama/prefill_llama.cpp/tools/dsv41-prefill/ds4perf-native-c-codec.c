#include "dsv41-native-c-artifact.h"
/* Compile the adapter as C, including on Windows, while the oracle is C++. */
int ds4perf_c_plan(uint32_t c,uint32_t b,uint32_t t,dsv41_c_plan*p){return dsv41_c_delta(c,b,t,p);}
int ds4perf_c_header(const dsv41_c_plan*p,uint64_t s,uint64_t m,uint8_t*out){return dsv41_c_artifact_header(p,s,m,out);}
int ds4perf_c_write(const dsv41_c_plan*p,uint64_t s,uint64_t m,const int32_t*t,const float*l,
    dsv41_c_read_span r,void*state,dsv41_c_io w,void*stream){return dsv41_c_write_delta(p,s,m,t,l,r,state,w,stream);}
int ds4perf_c_read(uint32_t c,uint32_t b,uint32_t cap,uint64_t n,uint64_t s,uint64_t m,dsv41_c_io r,
    void*stream,int32_t*t,float*l,dsv41_c_plan*p){return dsv41_c_read_delta_prefix(c,b,cap,n,s,m,r,stream,t,l,p);}
