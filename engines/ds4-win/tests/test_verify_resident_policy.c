#include "../ds4_cache_affinity.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
static void run_case(float *p,float *b,uint8_t *r,int applied,int protected_top,int insufficient) {
    int32_t selected[3]={0,1,2};float weights[3]={.6f,.3f,.1f};
    ds4_affinity_resident_decision d;
    assert(ds4_cache_affinity_select_resident_confident(selected,weights,p,b,r,8,3,1,&d));
    assert(d.applied==applied && d.protected_top==protected_top && d.insufficient_resident==insufficient);
    if(applied) {
        for(int j=0;j<3;j++)assert(r[selected[j]]);
        float sum=0;for(int j=0;j<3;j++)sum+=p[selected[j]];sum=fmaxf(sum,6.103515625e-5f);
        for(int j=0;j<3;j++)assert(weights[j]==p[selected[j]]/sum);
    } else {assert(selected[0]==0 && selected[1]==1 && selected[2]==2 && weights[0]==.6f && weights[1]==.3f && weights[2]==.1f);}
}
int main(void) {
    float p[8]={.51f,.509f,.50f,.499f,.498f,.09f,.08f,.07f};uint8_t r[8]={0,1,0,1,1};
    run_case(p,NULL,r,1,0,0);
    p[0]=.9f;p[1]=.1f;p[2]=.095f;p[3]=.094f;p[4]=.093f;p[5]=.092f;r[1]=0;r[5]=1;run_case(p,NULL,r,0,1,0);
    r[3]=r[4]=0;run_case(p,NULL,r,0,0,1);
    p[0]=.51f;p[1]=.50f;p[2]=.499f;r[0]=r[1]=r[2]=1;run_case(p,NULL,r,1,0,0);
    float bias[8]={.3f,.09f,.09f,.079f,.078f};
    p[0]=.3f;p[1]=.5f;p[2]=.49f;p[3]=p[4]=.5f;r[0]=r[2]=r[5]=0;r[1]=r[3]=r[4]=1;
    run_case(p,bias,r,1,0,0);
    uint32_t random=12345;
    for(int attempt=0;attempt<10000;attempt++) {
        float prob[32],weight[6];uint8_t resident[32];int32_t selected[6];
        for(int i=0;i<32;i++) {
            random=random*1664525u+1013904223u;prob[i]=(float)(random>>8)/16777216.0f;
            resident[i]=(random>>3)&1;
        }
        uint8_t all[32];memset(all,1,sizeof all);
        ds4_cache_affinity_select(selected,weight,prob,NULL,all,0,32,6,1);
        int32_t original[6];float original_weight[6];memcpy(original,selected,sizeof original);memcpy(original_weight,weight,sizeof weight);
        ds4_affinity_resident_decision d;
        assert(ds4_cache_affinity_select_resident_confident(selected,weight,prob,NULL,resident,32,6,1,&d));
        if(d.applied) {
            for(int j=0;j<6;j++)assert(resident[selected[j]]);
            float sum=0;for(int j=0;j<6;j++)sum+=prob[selected[j]];
            sum=fmaxf(sum,6.103515625e-5f);
            for(int j=0;j<6;j++)assert(weight[j]==prob[selected[j]]/sum);
        } else {
            assert(!memcmp(original,selected,sizeof original) && !memcmp(original_weight,weight,sizeof weight));
            assert(d.protected_top || d.insufficient_resident);
        }
    }
    puts("VERIFY_HARD_RESIDENT_PROTECTION_NORMALIZATION_AND10000_INVARIANTS_PASS");
    return 0;
}
