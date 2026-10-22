#pragma once
// Pure host bookkeeping. No allocator, environment, CUDA API, sync or readback.
#include "ggml-ds4-native-half-stats.h"
#include <atomic>
namespace ds4split_native_half_witness {
struct counters {
 std::atomic<uint64_t> queued[3]{},completed[3]{},graphs{0};std::atomic<bool> active{false};
 void snapshot(ggml_ds4_native_half_stats & out)const{
  for(int i=0;i<3;++i){out.queued[i]=queued[i].load(std::memory_order_acquire);out.completed[i]=completed[i].load(std::memory_order_acquire);}
  out.completed_graphs=graphs.load(std::memory_order_acquire);out.graph_active=active.load(std::memory_order_acquire)?1:0;
 }
 bool begin(ggml_ds4_native_half_stats & out){
  bool expected=false;if(!active.compare_exchange_strong(expected,true,std::memory_order_acq_rel))return false;
  snapshot(out);for(int i=0;i<3;++i)if(out.queued[i]!=out.completed[i])return false;
  return true;
 }
 bool queue(int slot,bool & first){
  if(slot<1||slot>3||!active.load(std::memory_order_acquire))return false;
  auto & value=queued[slot-1];auto previous=value.load(std::memory_order_relaxed);
  do{if(previous==UINT64_MAX)return false;}while(!value.compare_exchange_weak(previous,previous+1,std::memory_order_acq_rel));
  first=previous==0;return true;
 }
 bool complete(const ggml_ds4_native_half_stats & before,ggml_ds4_native_half_stats & after){
  if(!active.load(std::memory_order_acquire)||before.graph_active!=1||before.completed_graphs==UINT64_MAX)return false;
  snapshot(after);if(after.completed_graphs!=before.completed_graphs)return false;
  for(int i=0;i<3;++i)if(before.queued[i]!=before.completed[i]||before.queued[i]==UINT64_MAX||after.queued[i]!=before.queued[i]+1||after.completed[i]!=before.completed[i])return false;
  for(int i=0;i<3;++i){completed[i].store(after.queued[i],std::memory_order_release);after.completed[i]=after.queued[i];}
  after.completed_graphs=before.completed_graphs+1;graphs.store(after.completed_graphs,std::memory_order_release);after.graph_active=0;
  active.store(false,std::memory_order_release);return true;
 }
};
static_assert(sizeof(counters)<=128,"bounded host BSS counters; no heap/device storage");
}
