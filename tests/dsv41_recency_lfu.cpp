#include "../engines/streaming-llama/prefill_llama.cpp/src/llama-dsv41-recency-lfu.h"
#include <cassert>
#include <iostream>
int main(){
    using llama_dsv41_recency_lfu::scores;
    scores policy;
    assert(policy.admit(1,-1));
    policy.touch(1);policy.touch(2);
    assert(!policy.admit(1,2)); // equal candidates must not thrash
    for(int n=0;n<3;++n)policy.touch(1);
    assert(policy.admit(1,2));
    assert(!policy.admit(2,1));
    policy.decay();
    assert(policy.get(1)==512&&policy.get(2)==128);
    for(int n=0;n<4;++n)policy.touch(2);
    assert(policy.admit(2,1)); // adapt to a changed working set
    bool refused=false;try{policy.touch(40*384);}catch(const std::out_of_range&){refused=true;}
    assert(refused);
    scores longer;
    longer.tick(16);longer.touch(3);
    for(int n=0;n<15;++n)longer.tick(16);
    assert(longer.get(3)==256);
    longer.tick(16);assert(longer.get(3)==128);
    bool invalid=false;try{longer.tick(0);}catch(const std::invalid_argument&){invalid=true;}
    assert(invalid);
    std::cout<<"recency LFU admission, decay and phase change PASS\n";
}
