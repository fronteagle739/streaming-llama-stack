#include "llama-dsv41-host-plan.h"
#include <cassert>
#include <cstdio>
#include <limits>
int main() {
 using llama_dsv41_host_plan::plan;
 constexpr uint64_t m=1ull<<20;
 auto warm=plan(8455716864ull,6254253576ull,4ull<<30);
 assert(warm.ok&&warm.row_cache_bytes==2200961024ull);
 auto paired=plan(8455716864ull,6388471304ull,warm.row_cache_bytes);
 assert(paired.ok&&paired.row_cache_bytes==2066743296ull);
 assert(paired.planned_host==8455214600ull&&paired.planned_host<=8455716864ull);
 assert(plan(100,100,UINT64_MAX).ok&&plan(100,100,UINT64_MAX).row_cache_bytes==0);
 auto bad=plan(99,100,0);assert(!bad.ok&&bad.reason.find("non-elastic host minimum=100")!=std::string::npos);
 assert(plan(32*m,0,32*m).row_cache_bytes==32*m);
 assert(plan(16*m-1,0,32*m).row_cache_bytes==0);
 assert(plan(16*m,0,32*m).row_cache_bytes==16*m);
 assert(plan(100*m,0,0).row_cache_bytes==0);
 for(uint64_t minimum=0;minimum<128*m;minimum+=999983)
  for(uint64_t headroom=0;headroom<128*m;headroom+=999983){
   auto p=plan(minimum+headroom,minimum,UINT64_MAX);
   assert(p.ok&&p.planned_host<=minimum+headroom&&p.row_cache_bytes<=headroom);
  }
 auto huge=plan(UINT64_MAX,UINT64_MAX-32*m,UINT64_MAX);
 assert(huge.ok&&huge.planned_host==UINT64_MAX);
 std::puts("PASS host planner: Round4 warmup->paired regression, caps, zero, exact/minimum refusal, quantization, 18K boundaries and overflow");
}
