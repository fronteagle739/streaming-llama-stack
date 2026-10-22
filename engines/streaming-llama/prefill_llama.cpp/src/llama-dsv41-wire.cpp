#include "llama-dsv41-wire.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <utility>

namespace llama_dsv41 {
namespace {
constexpr uint32_t vocab_size = 129280;
constexpr uint32_t payload_magic = 0x34565344;
constexpr uint32_t payload_version = 2;
constexpr uint32_t architecture = 0x413431;
bool fail(std::string & error, const char * message) {
    error = message;
    return false;
}
bool write_u32(const wire_write & write, uint32_t value) {
    const uint8_t le[] = {uint8_t(value), uint8_t(value >> 8), uint8_t(value >> 16), uint8_t(value >> 24)};
    return write(le, sizeof(le));
}
std::array<uint32_t,13> payload_header(const wire_frontier & f){
    return {{payload_magic,payload_version,f.context,f.mode,128u,128u,f.context+1u,f.position,40u,512u,128u,vocab_size,architecture}};
}
bool validate_payload(const wire_frontier & f,const int32_t * tokens,size_t token_count,
        const float * logits,size_t logits_count,const wire_read_span & read_span,bool has_writer,
        wire_plan & plan,std::string & error){
    if(!make_wire_plan(f,plan,error))return false;
    if(!tokens || !logits || !read_span || !has_writer || token_count!=f.position || logits_count!=vocab_size)
        return fail(error,"invalid V4.1 native payload inputs");
    for(size_t i=0;i<token_count;++i)if(tokens[i]<0 || uint32_t(tokens[i])>=vocab_size)return fail(error,"invalid V4.1 native payload token");
    // Release builds can inherit -ffast-math. Inspect the IEEE exponent rather
    // than a floating predicate the compiler may assume is always finite.
    for(size_t i=0;i<logits_count;++i){
        uint32_t bits;std::memcpy(&bits,logits+i,sizeof(bits));
        if((bits&0x7f800000u)==0x7f800000u)return fail(error,"nonfinite V4.1 payload logits");
    }
    const uint32_t one=1;
    if(*reinterpret_cast<const uint8_t*>(&one)!=1 || sizeof(float)!=4 || !std::numeric_limits<float>::is_iec559)
        return fail(error,"V4.1 payload requires little-endian IEEE F32");
    return true;
}
}

bool make_wire_plan(const wire_frontier & f, wire_plan & result, std::string & error) {
    // Native ds41_load_payload requires room for the next token in both the
    // producing and receiving contexts; an empty frontier is not restorable.
    if (!f.position || f.position >= f.context || f.context > 1048576u ||
        f.mode < 1u || f.mode > 3u ||
        (f.mode == 3u && (f.capture_end != f.position ||
                         f.capture_length > std::min(f.position, 128u)))) {
        return fail(error, "invalid V4.1 native payload frontier");
    }
    wire_plan plan;
    plan.frontier = f;
    plan.bytes = 13u * 4u + (uint64_t(f.position) + vocab_size) * 4u;
    if (f.mode == 3u) plan.bytes += 8u;
    const uint32_t raw = std::min(f.position, 128u);
    auto span = [&](state_kind kind, uint32_t index, uint64_t bytes) {
        plan.spans.push_back({kind, index, bytes});
        plan.bytes += bytes;
    };
    if (f.mode != 1u) span(state_kind::dspark_hidden, 0, uint64_t(raw) * 3u * 5120u * 4u);
    for (uint32_t layer = 0; layer < 40u; ++layer) {
        span(state_kind::raw_window, layer, uint64_t(raw) * 512u * 4u);
    }
    for (uint32_t owner = 0; owner < 4u; ++owner) {
        const uint32_t live = f.position / (owner < 3u ? 2u : 1u);
        span(state_kind::compressed, owner, uint64_t(live) * 512u * 4u);
        span(state_kind::index, owner, uint64_t(live) * 128u * 4u);
        if (owner < 3u && (f.position & 1u)) {
            span(state_kind::pool_kv, owner, 512u * 4u);
            span(state_kind::pool_score, owner, 512u * 4u);
        }
    }
    result = std::move(plan);
    error.clear();
    return true;
}

bool write_native_payload(const wire_frontier & f,
                           const int32_t * tokens, size_t token_count,
                           const float * logits, size_t logits_count,
                           const wire_read_span & read_span, const wire_write & write,
                           std::string & error) {
    wire_plan plan;
    if(!validate_payload(f,tokens,token_count,logits,logits_count,read_span,bool(write),plan,error))return false;
    const auto header=payload_header(f);
    for (uint32_t field : header) {
        if (!write_u32(write, field)) return fail(error, "V4.1 payload header write failed");
    }
    if (f.mode == 3u && (!write_u32(write, f.capture_end) || !write_u32(write, f.capture_length))) {
        return fail(error, "V4.1 payload capture frontier write failed");
    }
    // Tokens are streamed in blocks too: a 1M-token frontier must not turn
    // into one giant transport write or a second full-payload allocation.
    constexpr size_t chunk_bytes = 1u << 20;
    const auto * token_bytes = reinterpret_cast<const uint8_t *>(tokens);
    const size_t tokens_bytes = token_count * sizeof(int32_t);
    for (size_t off = 0; off < tokens_bytes; off += chunk_bytes) {
        if (!write(token_bytes + off, std::min(chunk_bytes, tokens_bytes - off))) {
            return fail(error, "V4.1 payload token write failed");
        }
    }
    if (!write(logits, logits_count * sizeof(float))) return fail(error, "V4.1 payload logits write failed");
    std::vector<uint8_t> scratch(chunk_bytes);
    for (const state_span & span : plan.spans) {
        for (uint64_t off = 0; off < span.bytes; off += chunk_bytes) {
            const size_t size = size_t(std::min<uint64_t>(chunk_bytes, span.bytes - off));
            if (!read_span(span, off, scratch.data(), size)) return fail(error, "V4.1 payload state read failed");
            if (!write(scratch.data(), size)) return fail(error, "V4.1 payload state write failed");
        }
    }
    error.clear();
    return true;
}

bool wire_payload_reader::init(const wire_frontier & f,const int32_t * token_data,size_t token_count,
        const float * logit_data,size_t logits_count,const wire_read_span & reader,std::string & error){
    std::lock_guard<std::mutex> lock(mutex);ready=false;
    if(!validate_payload(f,token_data,token_count,logit_data,logits_count,reader,true,plan,error))return false;
    header_bytes=0;
    auto append=[&](uint32_t value){for(int b=0;b<4;++b)header[header_bytes++]=uint8_t(value>>(b*8));};
    for(uint32_t word:payload_header(f))append(word);
    if(f.mode==3){append(f.capture_end);append(f.capture_length);}
    tokens=reinterpret_cast<const uint8_t*>(token_data);token_bytes=uint64_t(token_count)*4;
    logits=reinterpret_cast<const uint8_t*>(logit_data);logit_bytes=uint64_t(logits_count)*4;
    read_span=reader;ready=true;error.clear();return true;
}
uint64_t wire_payload_reader::bytes() const{
    std::lock_guard<std::mutex> lock(mutex);return ready?plan.bytes:0;
}
bool wire_payload_reader::read(uint64_t offset,void * dst,size_t count,std::string & error){
    std::lock_guard<std::mutex> lock(mutex);
    if(!ready || (!dst && count) || count>16u*1024u*1024u || offset>plan.bytes || count>plan.bytes-offset)
        return fail(error,"invalid or unbounded V4.1 transport pull");
    if(!count){error.clear();return true;}
    auto * output=static_cast<uint8_t*>(dst);size_t remaining=count;
    auto copy_region=[&](const uint8_t * source,uint64_t first,uint64_t length){
        if(offset<first || offset-first>=length)return;
        const size_t take=size_t(std::min<uint64_t>(remaining,length-(offset-first)));
        memcpy(output,source+(offset-first),take);output+=take;offset+=take;remaining-=take;
    };
    copy_region(header.data(),0,header_bytes);
    copy_region(tokens,header_bytes,token_bytes);
    copy_region(logits,header_bytes+token_bytes,logit_bytes);
    uint64_t first=header_bytes+token_bytes+logit_bytes;
    for(const auto & span:plan.spans){
        if(!remaining)break;
        while(remaining && offset>=first && offset-first<span.bytes){
            const uint64_t within=offset-first;
            const size_t take=size_t(std::min<uint64_t>(1024u*1024u,std::min<uint64_t>(remaining,span.bytes-within)));
            if(!read_span(span,within,output,take)){ready=false;return fail(error,"V4.1 transport state read failed");}
            output+=take;offset+=take;remaining-=take;
        }
        first+=span.bytes;
    }
    if(remaining){ready=false;return fail(error,"V4.1 transport pull was incomplete");}
    error.clear();return true;
}
} // namespace llama_dsv41
