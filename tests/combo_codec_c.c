/* Compile the production adapter as C; compare against the independent C++
 * producer planner, including all owner kinds and wrapped ring slices. */
#include "dsv41-native-c-artifact.h"
int combo_plan(uint32_t mode,uint32_t ctx,uint32_t base,uint32_t target,dsv41_c_plan*out){
    return dsv41_c_delta_mode(mode,ctx,base,target,out);
}
int combo_header(const dsv41_c_plan*p,uint64_t session,uint64_t model,uint8_t*out){
    if(!dsv41_c_artifact_header(p,session,model,out))return 0;
    return (int)dsv41_c_native_header_mode(p->mode,p->context,p->target,out+204);
}
int combo_write(const dsv41_c_plan*p,const int32_t*t,const float*l,dsv41_c_read_span r,void*state,dsv41_c_io w,void*stream){
    return dsv41_c_write_delta(p,17,23,t,l,r,state,w,stream);
}
int combo_read(uint32_t mode,uint32_t c,uint32_t b,uint64_t bytes,dsv41_c_io read,void*stream,int32_t*t,float*l,dsv41_c_plan*out){
    return dsv41_c_read_delta_prefix_mode(mode,c,b,49152,bytes,17,23,read,stream,t,l,out);
}
