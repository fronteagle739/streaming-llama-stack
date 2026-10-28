#include "../engines/streaming-llama/prefill_llama.cpp/src/llama-dsv41-singlepass.h"
#include "../engines/streaming-llama/prefill_llama.cpp/src/llama-dsv41-prefill-plan.h"
#include <cstdio>
#include <cstdlib>

int main(){
    using llama_dsv41_singlepass::hidden_budget;
    if(hidden_budget(100,300,50,320)!=320||hidden_budget(100,300,250,320)!=150||
       hidden_budget(100,300,301,320)!=100||hidden_budget(100,300,0,90)!=90||
       hidden_budget(UINT64_MAX-10,UINT64_MAX,0,UINT64_MAX)!=UINT64_MAX)return 3;
    for(uint64_t full=1;full<1000;full+=13)for(uint64_t active=0;active<=full;active+=7){
        const auto hidden=hidden_budget(100,full,active,500);
        if(hidden+active>100+full||hidden>500||hidden<100)return 4;
    }
    using llama_dsv41_singlepass::frontier;
    // Compare independently against the actual existing N=1 fold schedule,
    // including both sides of every fold/group boundary and odd tails.
    for(uint32_t n=1;n<=410000;n+=37){
        const auto reference=llama_dsv41_prefill::plan_sweeps(n,61440,49152,61440);
        const auto grouped=llama_dsv41_prefill::plan_sweeps(n,110592,98304,110592);
        for(const auto&g:grouped)for(uint32_t start=g.first;start<g.first+g.count;start+=4096){
            uint32_t expected=0;
            for(const auto&r:reference)if(start>=r.first&&start<r.first+r.count){expected=r.first+r.count;break;}
            if(frontier(start,g.first+g.count)!=expected){std::fprintf(stderr,"frontier mismatch n=%u start=%u\n",n,start);return 1;}
        }
    }
    if(frontier(10,10)!=0||frontier(11,10)!=0)return 2;
    std::puts("canonical index frontiers match legacy N=1 across grouped boundaries");
}
