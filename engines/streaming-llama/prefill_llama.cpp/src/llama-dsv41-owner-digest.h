#pragma once

#include "llama-dsv41-wire.h"
#include "boundary-capture.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

// LLAMADECODE incremental owner verification (producer-local; the wire
// owner_sha256 values are unchanged).
//
// Each owner digest is, as before, the plain SHA-256 of that span's canonical
// F32 bytes. compressed/index spans are append-only: row i is final once the
// frontier passes it and only the rows [old_live, new_live) are ever added
// (make_delta_plan writes exactly that tail). The memo keeps, per
// compressed/index owner, the SHA-256 running state after its first L bytes.
// Extending that state over the appended bytes [L, bytes) and finalizing a
// copy is exactly SHA-256 of the whole span: the same value a full re-hash and
// the Mac receiver compute, so integrity of everything hashed is unchanged and
// the cost is O(appended rows). Ring/pool/DSpark owners (bounded, <=128 rows,
// rewritten in place) are re-hashed in full every time.
//
// Trust boundary: rows below L are not re-read. They are covered by the
// digest verified when they were hashed; any path that can rewrite them
// (rewind, restore/replace, clear, suspend, failure) must invalidate().
// LLAMADECODE_OWNER_VERIFY=full keeps the previous full re-hash everywhere.
namespace llama_dsv41 {

inline bool owner_verify_incremental(){
    static const bool on=[]{const char*e=std::getenv("LLAMADECODE_OWNER_VERIFY");
        return !(e&&std::strcmp(e,"full")==0);}();
    return on;
}

class owner_digest_memo {
public:
    using reader=std::function<bool(const state_span&,uint64_t,void*,size_t)>;
    void invalidate(){valid=false;for(auto&e:entries)e.have=false;}
    bool ready()const{return valid;}
    uint32_t position()const{return memo_position;}
    // Digests in plan.spans order. incremental=false re-hashes every byte
    // (and refreshes the memo). hashed_bytes reports the bytes actually read.
    std::vector<std::string> digests(const wire_plan&plan,const reader&read,bool incremental,uint64_t*hashed_bytes=nullptr){
        const auto&f=plan.frontier;
        const bool use=incremental&&valid&&f.context==memo_context&&f.mode==memo_mode&&f.position>=memo_position;
        valid=false; // only a fully completed pass re-arms the memo
        std::vector<std::string> out;out.reserve(plan.spans.size());uint64_t hashed=0;
        std::vector<unsigned char> page(chunk);
        auto feed=[&](dsv41_boundary::sha256_state&st,const state_span&s,uint64_t from){
            for(uint64_t off=from;off<s.bytes;){
                const size_t n=size_t(std::min<uint64_t>(chunk,s.bytes-off));
                if(!read(s,off,page.data(),n))throw std::runtime_error("Owner digest span read failed");
                dsv41_boundary::sha256_update(st,page.data(),n);off+=n;hashed+=n;
            }
        };
        for(const auto&s:plan.spans){
            auto st=dsv41_boundary::sha256_begin();
            const int k=slot(s);
            if(k>=0){
                auto&e=entries[size_t(k)];uint64_t from=0;
                if(use&&e.have&&e.bytes<=s.bytes){st=e.state;from=e.bytes;}
                feed(st,s,from);e.have=true;e.bytes=s.bytes;e.state=st;
            }else feed(st,s,0);
            out.push_back(dsv41_boundary::sha256_hex(st));
        }
        memo_context=f.context;memo_mode=f.mode;memo_position=f.position;valid=true;
        if(hashed_bytes)*hashed_bytes=hashed;
        return out;
    }
private:
    static constexpr size_t chunk=1u<<20; // read_native_span's bound
    static int slot(const state_span&s){
        if(s.index>=4)return -1;
        if(s.kind==state_kind::compressed)return int(s.index);
        if(s.kind==state_kind::index)return 4+int(s.index);
        return -1;
    }
    struct entry{bool have=false;uint64_t bytes=0;dsv41_boundary::sha256_state state{};};
    std::array<entry,8> entries{};
    bool valid=false;
    uint32_t memo_context=0,memo_mode=0,memo_position=0;
};

} // namespace llama_dsv41
