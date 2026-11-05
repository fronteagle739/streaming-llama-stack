"""Model-free fail-closed coverage for the actual producer capture plan."""
from pathlib import Path
import os
import shutil
import subprocess

import pytest


def test_native_capture_requires_complete_unique_sites(tmp_path):
    compiler = shutil.which('clang++') or shutil.which('c++')
    if not compiler:
        pytest.skip('C++ compiler unavailable')
    root = Path(__file__).resolve().parents[1]
    producer = root/'engines/streaming-llama/prefill_llama.cpp'
    source = tmp_path/'capture.cpp'
    source.write_text(r'''
#include "llama-dsv41-writer-capture.h"
#include "llama-dsv41-writer-memory.h"
#include "ds4_direction_slots.h"
#include "ds4_direction_cache_memory.h"
#include <cassert>
using dsv41_writer_capture::plan;
template<class F> void refused(F work) {
    bool failed=false;try {work();}catch(const std::runtime_error&){failed=true;}
    assert(failed);
}
int main() {
    static_assert(ds4_direction_rank_slot_active(0),"GPU slot must keep its delta");
    static_assert(ds4_direction_rank_slot_active(-2),"CPU slot must also receive its delta");
    static_assert(!ds4_direction_rank_slot_active(-1),"Absent slot must not receive a delta");
    static_assert(!ds4_direction_rank_slot_active(-3),"Unknown sentinel must not receive a delta");
    static_assert(ds4_direction_packed_scratch_bytes()<=(8ull<<20),"Packed CPU target staging must remain inside the charged scratch reserve");
    static_assert(!ds4_direction_cache_memory::fits(0,160ull<<20,4ull<<30,0,256ull<<20),"Zero measured memory must refuse");
    static_assert(ds4_direction_cache_memory::fits(0,160ull<<20,4ull<<30,8ull<<30,256ull<<20),"Physical imported-buffer headroom must admit the incident writer");
    static_assert(!ds4_direction_cache_memory::fits(4ull<<30,160ull<<20,4ull<<30,8ull<<30,256ull<<20),"Cache cap remains enforced");
    static_assert(!ds4_direction_cache_memory::fits(0,160ull<<20,4ull<<30,415ull<<20,256ull<<20),"The free-memory floor must remain after allocation");
    static_assert(!ds4_direction_cache_memory::fits(UINT64_MAX,1,4ull<<30,UINT64_MAX,256ull<<20),"Budget arithmetic must not wrap");
    const uint64_t expected[] = {/* MEMORY_EXPECTED */};
    const uint64_t prefill[] = {/* PREFILL_EXPECTED */};
    // Incident: all80writers exceed4GiB, so a4GiB dense LRU must re-fuse each
    // token. The admitted working set must fit every writer without eviction.
    const auto dense=ds4_direction_cache_memory::dense_capacity_bytes();
    assert(dense==40ull*5120ull*(8192ull+2304ull)*sizeof(float));
    uint64_t resident=0;
    for(unsigned layer=0;layer<40;++layer)for(auto input:{8192u,2304u}){
        const auto bytes=uint64_t(input)*5120ull*sizeof(float);
        assert(ds4_direction_cache_memory::fits(resident,bytes,dense,dense-resident+(256ull<<20),256ull<<20));
        resident+=bytes;
    }
    assert(resident==dense);
    for(uint32_t n=0;n<=8;++n){
        assert(dsv41_writer_memory::resident_device_bytes(n)==expected[n]);
        assert(dsv41_writer_memory::pending_device_bytes(n,0)==expected[n]);
        assert(dsv41_writer_memory::pending_device_bytes(n,expected[n])==0);
        if(n)assert(dsv41_writer_memory::pending_device_bytes(n,expected[n]-1)==1);
        refused([&]{dsv41_writer_memory::pending_device_bytes(n,expected[n]+1);});
    }
    for(uint32_t n=0;n<=8;++n)assert(dsv41_writer_memory::prefill_device_bytes(n)==prefill[n]);
    refused([]{dsv41_writer_memory::resident_device_bytes(9);});
    refused([]{dsv41_writer_memory::prefill_device_bytes(9);});
    refused([]{plan p;p.configure(5,1,{1,1},{0,0});});
    refused([]{plan p;p.configure(5,2,{1},{0});});
    refused([]{plan p;p.configure(5,1,{5},{0});});
    plan partial;partial.configure(5,1,{1,3},{0,0});
    refused([&]{partial.enable_evaluate();});
    refused([&]{partial.finish();});
    plan complete;complete.configure(5,2,{1,3},{0,1});
    assert(complete.counts[0]==1&&complete.counts[1]==1);
    complete.enable_evaluate();
    std::fill(complete.seen.begin(),complete.seen.end(),1);
    refused([&]{complete.finish();});
    std::fill(complete.final_seen.begin(),complete.final_seen.end(),1);
    complete.finish();assert(!complete.active);
    plan duplicate;duplicate.configure(5,1,{1},{0});
    std::fill(duplicate.seen.begin(),duplicate.seen.end(),1);
    duplicate.seen[0]=2;refused([&]{duplicate.finish();});
}
'''.replace('/* MEMORY_EXPECTED */', ','.join(str(__import__('slstack.writer_directions',fromlist=['device_reserve']).device_reserve(n))+'ull' for n in range(9))).replace('/* PREFILL_EXPECTED */', ','.join(str(__import__('slstack.writer_directions',fromlist=['prefill_device_reserve']).prefill_device_reserve(n))+'ull' for n in range(9))))
    output = Path(os.environ.get('TMPDIR',str(tmp_path)))/'directions-writer-capture-test'
    subprocess.run([compiler,'-std=c++17','-O1','-I'+str(producer/'src'),'-I'+str(producer/'ggml/include'),'-I'+str(root/'engines/ds4-win'),str(source),'-o',str(output)],check=True,timeout=60)
    subprocess.run([str(output)],check=True,timeout=10)


def test_real_writer_mixed_wire_contract_compiles():
    compiler=shutil.which('clang')or shutil.which('cc')
    assert compiler, 'C compiler is required for the mixed writer wire contract'
    source=Path(__file__).resolve().parents[1]/'engines/ds4-win/tests/test_direction_fusion_cuda.c'
    subprocess.run([compiler,'-std=c11','-fsyntax-only',str(source)],check=True,timeout=30)
