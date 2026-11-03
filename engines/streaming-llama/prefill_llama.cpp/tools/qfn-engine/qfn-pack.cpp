// Streaming Llama Pack reader (header, sections, meta JSON, expert layout, profile).
#include "qfn-pack.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <cmath>
#include <algorithm>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <bcrypt.h>
#endif
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace {

using json = nlohmann::json;

bool read_at(std::ifstream & f, uint64_t off, void * dst, uint64_t n) {
    f.seekg((std::streamoff) off);
    f.read((char *) dst, (std::streamsize) n);
    return (uint64_t) f.gcount() == n;
}

std::wstring widen(const std::string & s) {
    std::wstring w;
    w.reserve(s.size());
    // the paths this engine accepts are ASCII (drive letters and directories it creates); reject anything else
    for (unsigned char c : s) {
        if (c >= 0x80) throw std::runtime_error("non-ASCII model path is not supported: " + s);
        w.push_back((wchar_t) c);
    }
    return w;
}

// SHA only covers the small auxiliary sections. The engine already links bcrypt.lib.
std::string sha256(const void * data, size_t size) {
#ifdef _WIN32
    struct resources {
        std::vector<UCHAR> object;
        BCRYPT_ALG_HANDLE algorithm = nullptr;
        BCRYPT_HASH_HANDLE hash = nullptr;
        ~resources() { if (hash) BCryptDestroyHash(hash); if (algorithm) BCryptCloseAlgorithmProvider(algorithm, 0); }
    } handles;
    if (BCryptOpenAlgorithmProvider(&handles.algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0) < 0)
        throw std::runtime_error("checkpoint SHA256 provider unavailable");
    DWORD object_bytes = 0, result_bytes = 0;
    if (BCryptGetProperty(handles.algorithm, BCRYPT_OBJECT_LENGTH,
            reinterpret_cast<PUCHAR>(&object_bytes), sizeof object_bytes, &result_bytes, 0) < 0 ||
        result_bytes != sizeof object_bytes)
        throw std::runtime_error("checkpoint SHA256 object query failed");
    handles.object.resize(object_bytes);
    if (BCryptCreateHash(handles.algorithm, &handles.hash, handles.object.data(), object_bytes, nullptr, 0, 0) < 0)
        throw std::runtime_error("checkpoint SHA256 creation failed");
    const auto * source = static_cast<const UCHAR *>(data);
    while (size) {
        const ULONG chunk = ULONG(std::min<size_t>(size, 1u << 20));
        if (BCryptHashData(handles.hash, const_cast<PUCHAR>(source), chunk, 0) < 0)
            throw std::runtime_error("checkpoint SHA256 update failed");
        source += chunk; size -= chunk;
    }
    UCHAR digest[32];
    if (BCryptFinishHash(handles.hash, digest, sizeof digest, 0) < 0)
        throw std::runtime_error("checkpoint SHA256 finish failed");
    const char * digits = "0123456789abcdef";
    std::string result; result.reserve(64);
    for (const UCHAR byte : digest) { result += digits[byte >> 4]; result += digits[byte & 15]; }
    return result;
#else
    (void) data; (void) size;
    throw std::runtime_error("checkpoint SHA256 validation requires the native Windows backend");
#endif
}

uint32_t u32(const json & kv, const char * key) {
    if (!kv.contains(key)) throw std::runtime_error(std::string("pack metadata lacks ") + key);
    return kv.at(key).get<uint32_t>();
}

// Basalt tools/convert/model_sections.py:nvfp4_row_bytes and kernels/cpu/native_expert.hpp
// (MIT): type 40 has 36-byte blocks of 64 values and a 16-byte per-expert FP32 tail.
// Do not cast type 40 into this tree's reserved ggml enum or alias it to another precision.
uint64_t expert_row_bytes(uint32_t type, uint32_t values) {
    if (!values) throw std::runtime_error("expert row has no values");
    if (type == 40) {
        if (values % 64) throw std::runtime_error("NVFP4 expert row is not whole 64-value blocks");
        return (uint64_t) values / 64 * 36;
    }
    if (type >= GGML_TYPE_COUNT || ggml_blck_size((ggml_type) type) <= 0 ||
        values % ggml_blck_size((ggml_type) type))
        throw std::runtime_error("unsupported expert type or row geometry: " + std::to_string(type));
    return ggml_row_size((ggml_type) type, values);
}

qfn_expert_layer expert_format(uint32_t gt, uint32_t dt, uint32_t embd, uint32_t ff) {
    qfn_expert_layer e;
    e.gu_type_id = gt;
    e.d_type_id = dt;
    e.gu_type = gt == 40 ? GGML_TYPE_COUNT : (ggml_type) gt;
    e.d_type = dt == 40 ? GGML_TYPE_COUNT : (ggml_type) dt;
    // The NVIDIA converter emits NVFP4 for all three matrices; a mixed NVFP4 blob has no defined tail policy.
    if ((gt == 40) != (dt == 40)) throw std::runtime_error("mixed NVFP4 expert types have no supported scale-tail layout");
    const uint64_t gate_row = expert_row_bytes(gt, embd), down_row = expert_row_bytes(dt, ff);
    if (gate_row > std::numeric_limits<uint64_t>::max() / ff ||
        down_row > std::numeric_limits<uint64_t>::max() / embd)
        throw std::runtime_error("expert matrix size overflows");
    e.gate_bytes = gate_row * ff;
    e.down_bytes = down_row * embd;
    if (e.gate_bytes > (std::numeric_limits<uint64_t>::max() - e.down_bytes - 16) / 2)
        throw std::runtime_error("expert blob size overflows");
    e.tail_offset = 2 * e.gate_bytes + e.down_bytes;
    e.tail_bytes = gt == 40 ? 16 : 0;
    e.blob = e.tail_offset + e.tail_bytes;
    return e;
}

}  // namespace

const qfn_tensor * qfn_pack::find(const std::string & name) const {
    auto it = tensors.find(name);
    return it == tensors.end() ? nullptr : &it->second;
}

const qfn_tensor & qfn_pack::get(const std::string & name) const {
    const qfn_tensor * t = find(name);
    if (!t) throw std::runtime_error("pack has no tensor " + name);
    return *t;
}

std::pair<uint64_t, uint64_t> qfn_pack::section(const std::string & name) const {
    auto it = sections.find(name);
    if (it == sections.end()) throw std::runtime_error("pack has no section " + name);
    return it->second;
}

void qfn_pack::configure_generation(const std::string & raw, const std::string & gguf_defaults, const std::string & base_defaults) {
    const json source = raw.empty() ? json::object() : json::parse(raw);
    const json gguf=json::parse(gguf_defaults), base=json::parse(base_defaults);
    json defaults=json::object();
    if (!source.is_object() || !gguf.is_object() || !base.is_object())
        throw std::runtime_error("generation defaults must be JSON objects");
    auto numeric = [](const json & value, const std::string & key, double lo, double hi, bool integer) {
        if (!value.is_number()) throw std::runtime_error(key + " must be numeric");
        const double n = value.get<double>();
        if (!std::isfinite(n) || n < lo || n > hi || (integer && std::floor(n) != n))
            throw std::runtime_error("invalid generation " + key);
    };
    auto canonical=[](const std::string & key) {
        if(key=="temp") return std::string("temperature");
        if(key=="repetition_penalty" || key=="rep_pen" || key=="penalty_repeat") return std::string("repeat_penalty");
        if(key=="penalty_last_n") return std::string("repeat_last_n");
        if(key=="penalty_present") return std::string("presence_penalty");
        if(key=="penalty_freq") return std::string("frequency_penalty");
        return key;
    };
    auto sample_value = [&](const std::string & key, const json & value) {
        const double max=std::numeric_limits<float>::max();
        if (key == "temperature" || key == "presence_penalty" || key == "frequency_penalty" || key == "bias") numeric(value,key,-max,max,false);
        else if (key == "top_p" || key == "min_p" || key == "typical_p") numeric(value,key,0,1,false);
        else if (key == "top_k") numeric(value,key,INT32_MIN,INT32_MAX,true);
        else if (key == "repeat_penalty") numeric(value,key,std::numeric_limits<float>::min(),max,false);
        else if (key == "repeat_last_n") numeric(value,key,-1,INT32_MAX,true);
        else if (key == "seed") {
            if(!value.is_number_integer() || (!value.is_number_unsigned() && value.get<int64_t>() < -1))
                throw std::runtime_error("generation seed must be integer >=-1");
        }
        else if (key == "logit_bias") {
            auto entry=[&](const json & id,const json & amount,bool object_key) {
                if(!id.is_string() && (!id.is_number_integer() || id.get<double>()<0 || id.get<double>()>=hp.n_vocab))
                    throw std::runtime_error("generation logit_bias key must be text or an in-vocabulary integer");
                if(object_key && id.is_string()) {
                    const auto text=id.get<std::string>();
                    if(!text.empty() && text.find_first_not_of("0123456789")==std::string::npos &&
                       std::stoull(text)>=hp.n_vocab)
                        throw std::runtime_error("generation logit_bias token outside vocabulary");
                }
                if(amount!=false) numeric(amount,key,-max,max,false);
            };
            if(value.is_object()) for(const auto & it:value.items()) entry(it.key(),it.value(),true);
            else if(value.is_array()) for(const auto & it:value) {
                if(!it.is_array() || it.size()!=2) throw std::runtime_error("generation logit_bias requires pairs");
                entry(it[0],it[1],false);
            }
            else throw std::runtime_error("generation logit_bias requires an object or pair array");
        }
        else if (key == "do_sample") {
            if (!value.is_boolean()) throw std::runtime_error("generation do_sample must be boolean");
        } else throw std::runtime_error("BLOCKED-OWNER: unsupported generation default " + key);
    };
    auto token = [&](const json & value, const std::string & key) {
        if (value.is_null()) return int32_t(-1); // HF None is an explicit absent role, not fallback.
        if (!value.is_number_integer() || value.get<double>() < 0 ||
            value.get<double>() >= hp.n_vocab || value.get<double>() > INT32_MAX)
            throw std::runtime_error("generation " + key + " must be an in-vocabulary integer or null");
        return value.get<int32_t>();
    };
    bool eos_explicit=false,bos_explicit=false,pad_explicit=false;
    int32_t bos=-1,pad=-1;
    std::vector<int32_t> eos;
    for (const auto & it : source.items()) {
        const auto & key=it.key(); const auto & value=it.value();
        if (key == "bos_token_id") { bos_explicit=true; bos=token(value,key); }
        else if (key == "pad_token_id") { pad_explicit=true; pad=token(value,key); }
        else if (key == "eos_token_id") {
            eos_explicit=true;
            if (value.is_array()) {
                for (const auto & v : value) {
                    const int32_t id=token(v,key);
                    if (id < 0) throw std::runtime_error("generation EOS list cannot contain null");
                    if (std::find(eos.begin(),eos.end(),id)==eos.end()) eos.push_back(id);
                }
            } else { const int32_t id=token(value,key); if (id>=0) eos.push_back(id); }
        } else if (key == "_from_model_config") {
            if (!value.is_boolean()) throw std::runtime_error("_from_model_config must be boolean");
        } else if (key == "transformers_version") {
            if (!value.is_string()) throw std::runtime_error("transformers_version must be a string");
        } else {
            const auto field=canonical(key); sample_value(field,value);
            if(field!=key && source.contains(field) && !source.at(field).is_null()) continue;
            if(key=="rep_pen" && source.contains("repetition_penalty") && !source.at("repetition_penalty").is_null()) continue;
            defaults[field]=value;
        }
    }
    // Converter policy: checkpoint sampling is fallback; canonical GGUF then explicit base metadata win.
    // Token role/EOS source fields above remain authoritative and independent from sampler defaults.
    for(const auto * layer:{&gguf,&base}) for(const auto & it:layer->items()) {
        if(it.value().is_null()) continue; // inactive defaults; raw metadata retained
        const auto key=canonical(it.key()); sample_value(key,it.value());
        if(key!=it.key() && layer->contains(key) && !layer->at(key).is_null()) continue;
        if(it.key()=="rep_pen" && layer->contains("repetition_penalty") && !layer->at("repetition_penalty").is_null()) continue;
        defaults[key]=it.value();
    }
    generation_config_json=raw;
    generation_defaults_json=defaults.dump();
    generation_eos_explicit=eos_explicit; generation_eos=std::move(eos);
    generation_bos_explicit=bos_explicit; generation_bos_id=bos;
    generation_pad_explicit=pad_explicit; generation_pad_id=pad;
}

std::vector<int32_t> qfn_pack::eos_tokens() const {
    if (generation_eos_explicit) return generation_eos;
    // Exact preexisting fallback for legacy packs with no source EOS declaration.
    std::vector<int32_t> out{int32_t(hp.tok_eos),int32_t(hp.tok_pad)};
    for (const char * s : {"<|endoftext|>","<|im_end|>","<|fim_pad|>","<|repo_name|>","<|file_sep|>"})
        for (size_t i=0;i<vocab.size();++i) if (vocab[i]==s) { out.push_back(int32_t(i)); break; }
    return out;
}

bool qfn_pack::generation_self_test(std::string & err) {
    try {
        int checks=0;
        auto expect=[&](bool pass,const char * why) { ++checks; if(!pass) throw std::runtime_error(why); };
        qfn_pack p; p.hp.n_vocab=248320; p.hp.tok_bos=248044; p.hp.tok_eos=248046; p.hp.tok_pad=248044;
        const std::string actual=R"({"bos_token_id":248044,"do_sample":true,"eos_token_id":[248046,248044],"pad_token_id":248044,"temperature":1.0,"top_k":20,"top_p":0.95})";
        p.configure_generation(actual,R"({"temperature":0.4,"top_k":9,"top_p":0.6})");
        expect(p.generation_config_json==actual,"source generation raw bytes changed");
        expect(p.eos_tokens()==std::vector<int32_t>({248046,248044}),"source EOS exact set lost");
        for(int32_t id:{248063,248064,248065}) {
            const auto eos=p.eos_tokens(); expect(std::find(eos.begin(),eos.end(),id)==eos.end(),"extra special became source EOS");
        }
        const auto d=json::parse(p.generation_defaults_json);
        expect(d==json{{"do_sample",true},{"temperature",0.4},{"top_k",9},{"top_p",0.6}},"checkpoint fallback overrode canonical GGUF sampling");
        expect(p.generation_bos_explicit && p.generation_bos_id==248044 && p.generation_pad_explicit && p.generation_pad_id==248044,"source BOS/PAD roles lost");
        p.configure_generation(R"({"eos_token_id":null,"bos_token_id":null,"pad_token_id":null})");
        expect(p.eos_tokens().empty() && p.generation_bos_explicit && p.generation_bos_id==-1,"explicit null token role silently fell back");
        p.configure_generation(R"({"eos_token_id":[]})"); expect(p.eos_tokens().empty(),"empty EOS policy changed");
        p.configure_generation(R"({"eos_token_id":248046})"); expect(p.eos_tokens()==std::vector<int32_t>({248046}),"single source EOS lost");
        p.configure_generation("",R"({"temperature":0.4})");
        expect(!p.generation_eos_explicit && p.eos_tokens()==std::vector<int32_t>({248046,248044}),"legacy source-absent EOS changed");
        expect(json::parse(p.generation_defaults_json).at("temperature")==0.4,"GGUF fallback lost");
        p.hp.tok_eos=0; p.hp.tok_pad=1; p.vocab={"<|endoftext|>","<|im_end|>","<|fim_pad|>","<|repo_name|>","<|file_sep|>"};
        expect(p.eos_tokens()==std::vector<int32_t>({0,1,0,1,2,3,4}),"legacy named-special fallback changed");
        const std::string base=R"({"temperature":0.7,"top_p":0.8,"top_k":30,"min_p":0.05,"presence_penalty":0.4})";
        p.configure_generation(R"({"top_k":20})",R"({"temperature":0.9,"min_p":0.1})",base);
        const auto preserved=json::parse(p.generation_defaults_json);
        expect(preserved.at("temperature")==0.7 && preserved.at("min_p")==0.05 && preserved.at("top_k")==30 && preserved.at("top_p")==0.8 && preserved.at("presence_penalty")==0.4,"explicit base sampling did not override canonical GGUF/checkpoint fallback");
        const std::string all=R"({"temp":-1,"top_p":0.8,"top_k":-1,"min_p":0.1,"typical_p":0.9,"penalty_repeat":1.2,"penalty_last_n":-1,"penalty_present":0.3,"penalty_freq":0.4,"seed":-1,"bias":1.36125,"logit_bias":{"3":-2},"do_sample":false})";
        p.configure_generation("",all);
        const auto capable=json::parse(p.generation_defaults_json);
        expect(capable.size()==13 && capable.at("repeat_penalty")==1.2 && capable.at("repeat_last_n")==-1 && capable.at("frequency_penalty")==0.4 && capable.at("temperature")==-1,"supported GGUF sampler field/alias dropped");
        for(const char * bad:{"[]",R"({"min_p":2})",R"({"penalty_repeat":0})",R"({"penalty_last_n":-2})",R"({"seed":-2})",R"({"seed":1.5})",R"({"logit_bias":{"248320":1}})",R"({"new_sampler":0.5})"}) {
            bool refused=false;try { p.configure_generation("",bad); } catch(const std::exception &) { refused=true; }
            expect(refused,"invalid/unsupported GGUF/default sampling field accepted");
        }
        for(const char * bad:{"null","[]","{",R"({"do_sample":1})",R"({"temperature":true})",R"({"temperature":1e100})",R"({"temperature":"1"})",R"({"top_k":1.5})",R"({"top_p":2})",R"({"eos_token_id":true})",R"({"eos_token_id":[null]})",R"({"eos_token_id":248320})",R"({"bos_token_id":-1})",R"({"pad_token_id":1.5})",R"({"num_beams":2})",R"({"unknown_active":false})"}) {
            bool refused=false;try { p.configure_generation(bad); } catch(const std::exception &) { refused=true; }
            expect(refused,"malformed/unsupported generation configuration accepted");
        }
        std::fprintf(stderr,"generation metadata self-test PASS: %d strict source/default/EOS/legacy checks\n",checks);
        return true;
    } catch(const std::exception & e) { err=e.what(); return false; }
}

bool qfn_pack::open(const std::string & p, std::string & err) {
    try {
        // Reopening a descriptor must not retain sections, profiles or drafter fields from the previous pack.
        const std::string requested_path = p; // p may alias this->path on a reopen
        *this = qfn_pack{};
        err.clear();
        path = requested_path;
        wpath = widen(path);
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("cannot open " + path);
        f.seekg(0, std::ios::end);
        file_bytes = (uint64_t) f.tellg();

        // ---- header + section table (first 64 KiB)
        std::vector<uint8_t> hdr(64 << 10);
        if (!read_at(f, 0, hdr.data(), hdr.size())) throw std::runtime_error("short header");
        const bool slpack = std::memcmp(hdr.data(), "SLPACK\0\0", 8) == 0;
        // Explicit compatibility path for historical containers; all payload capabilities remain available.
        const uint8_t legacy_magic[8] = {0x42,0x41,0x53,0x41,0x4c,0x54,0x4d,0x46};
        const bool legacy_container = std::memcmp(hdr.data(), legacy_magic, 8) == 0;
        if (!slpack && !legacy_container) throw std::runtime_error("not a Streaming Llama Pack");
        uint32_t n_sections = 0;
        uint64_t declared = 0;
        std::memcpy(&version, hdr.data() + 8, 4);
        std::memcpy(&n_sections, hdr.data() + 12, 4);
        std::memcpy(&declared, hdr.data() + 16, 8);
        if ((slpack && version != 1) || (legacy_container && version != 1 && version != 2))
            throw std::runtime_error("unsupported pack envelope version " + std::to_string(version));
        if (slpack && std::any_of(hdr.begin()+24, hdr.begin()+64, [](uint8_t b) { return b != 0; }))
            throw std::runtime_error("unknown active Streaming Llama Pack header fields");
        if (declared != file_bytes) throw std::runtime_error("pack size differs from its header (truncated?)");
        if (n_sections == 0 || 64 + 128ull * n_sections > hdr.size()) throw std::runtime_error("bad section count");
        uint64_t previous_end = 64 << 10;
        for (uint32_t i = 0; i < n_sections; ++i) {
            const uint8_t * e = hdr.data() + 64 + 128ull * i;
            char name[65] = {0};
            std::memcpy(name, e, 64);
            uint64_t off = 0, bytes = 0;
            std::memcpy(&off, e + 64, 8);
            std::memcpy(&bytes, e + 72, 8);
            if (off > file_bytes || bytes > file_bytes - off) throw std::runtime_error(std::string("section out of bounds: ") + name);
            if (slpack) {
                const auto nul = std::find(e, e + 64, uint8_t(0));
                if (nul == e || nul == e + 64 || std::any_of(nul, e + 64, [](uint8_t b) { return b != 0; }) ||
                    std::any_of(e + 112, e + 128, [](uint8_t b) { return b != 0; }) ||
                    sections.count(name) || off % (2 << 20) || off < previous_end)
                    throw std::runtime_error("invalid Streaming Llama Pack section descriptor");
            }
            sections[name] = { off, bytes };
            previous_end = off + bytes;
        }
        if (slpack && (previous_end != file_bytes ||
            std::any_of(hdr.begin()+64+128ull*n_sections, hdr.end(), [](uint8_t b) { return b != 0; })))
            throw std::runtime_error("unknown Streaming Llama Pack table/file tail");

        // ---- meta JSON
        const auto ms = section("meta");
        std::string meta_text(ms.second, '\0');
        if (!read_at(f, ms.first, meta_text.data(), ms.second)) throw std::runtime_error("short meta");
        const json meta = json::parse(meta_text);
        const json & kv = meta.at("gguf_kv");
        if (kv.at("general.architecture").get<std::string>() != "qwen4exp") throw std::runtime_error("pack is not qwen4exp");

        // ---- tensors of the gguf section
        const uint64_t gsec = section("gguf").first;
        for (auto it = meta.at("tensors").begin(); it != meta.at("tensors").end(); ++it) {
            qfn_tensor t;
            t.name = it.key();
            const json & v = it.value();
            t.type = (ggml_type) v.at("type_id").get<int>();
            const auto & shape = v.at("shape");
            t.n_dims = (int) shape.size();
            if (t.n_dims < 1 || t.n_dims > 4) throw std::runtime_error("bad tensor rank: " + t.name);
            for (int d = 0; d < t.n_dims; ++d) t.ne[d] = shape[d].get<int64_t>();
            t.offset = gsec + v.at("offset").get<uint64_t>();
            t.bytes  = v.at("bytes").get<uint64_t>();
            if (t.type >= GGML_TYPE_COUNT || ggml_blck_size(t.type) == 0) throw std::runtime_error("unsupported tensor type: " + t.name);
            const uint64_t expect = ggml_row_size(t.type, t.ne[0]) * (uint64_t) (t.ne[1] * t.ne[2] * t.ne[3]);
            if (expect != t.bytes) throw std::runtime_error("tensor size mismatch: " + t.name);
            if (t.offset + t.bytes > file_bytes) throw std::runtime_error("tensor beyond file: " + t.name);
            tensors[t.name] = t;
        }

        // ---- hyperparameters
        hp.n_layer       = u32(kv, "qwen4exp.block_count");
        hp.n_embd        = u32(kv, "qwen4exp.embedding_length");
        hp.n_ctx_train   = u32(kv, "qwen4exp.context_length");
        hp.n_head        = u32(kv, "qwen4exp.attention.head_count");
        hp.n_head_kv     = u32(kv, "qwen4exp.attention.head_count_kv");
        hp.head_dim      = u32(kv, "qwen4exp.attention.key_length");
        hp.n_rot         = u32(kv, "qwen4exp.rope.dimension_count");
        {
            const auto & s = kv.at("qwen4exp.rope.dimension_sections");
            for (int i = 0; i < 4 && i < (int) s.size(); ++i) hp.rope_sections[i] = s[i].get<int>();
        }
        hp.rope_freq_base = kv.at("qwen4exp.rope.freq_base").get<float>();
        hp.rms_eps        = kv.at("qwen4exp.attention.layer_norm_rms_epsilon").get<float>();
        hp.n_expert       = u32(kv, "qwen4exp.expert_count");
        hp.n_expert_used  = u32(kv, "qwen4exp.expert_used_count");
        hp.n_ff_exp       = u32(kv, "qwen4exp.expert_feed_forward_length");
        hp.n_ff_shexp     = u32(kv, "qwen4exp.expert_shared_feed_forward_length");
        hp.ssm_conv       = u32(kv, "qwen4exp.ssm.conv_kernel");
        hp.ssm_state      = u32(kv, "qwen4exp.ssm.state_size");
        hp.ssm_group      = u32(kv, "qwen4exp.ssm.group_count");
        hp.ssm_dt_rank    = u32(kv, "qwen4exp.ssm.time_step_rank");
        hp.ssm_inner      = u32(kv, "qwen4exp.ssm.inner_size");
        hp.full_attn_interval = u32(kv, "qwen4exp.full_attention_interval");
        hp.hc             = u32(kv, "qwen4exp.hyper_connection.count");
        hp.hc_low_rank    = u32(kv, "qwen4exp.hyper_connection.low_rank");
        hp.idx_heads      = u32(kv, "qwen4exp.attention.indexer.head_count");
        hp.idx_dim        = u32(kv, "qwen4exp.attention.indexer.key_length");
        hp.idx_top_k      = u32(kv, "qwen4exp.attention.indexer.top_k");
        for (const auto & r : kv.at("qwen4exp.attention.compress_ratios")) hp.compress_ratios.push_back(r.get<uint32_t>());
        for (uint32_t il = 0; il < hp.n_layer; ++il) {
            if (il < hp.compress_ratios.size() && hp.compress_ratios[il]) {
                if (hp.kpool && hp.kpool != hp.compress_ratios[il]) throw std::runtime_error("QSA layers disagree on the pool size");
                hp.kpool = hp.compress_ratios[il];
            }
        }
        for (const auto & l : kv.at("qwen4exp.ple.layers")) hp.ple_layers.push_back(l.get<uint32_t>());
        hp.ple_ngram           = u32(kv, "qwen4exp.ple.ngram_size");
        hp.ple_heads_per_ngram = u32(kv, "qwen4exp.ple.heads_per_ngram");
        hp.ple_conv            = u32(kv, "qwen4exp.ple.conv_kernel");
        hp.ple_eos             = u32(kv, "qwen4exp.ple.eos_token_id");
        hp.ple_image           = kv.contains("qwen4exp.ple.image_token_id") ? u32(kv, "qwen4exp.ple.image_token_id") : 0;
        hp.ple_dim             = u32(kv, "qwen4exp.embedding_length_per_layer_input");
        hp.ple_n_heads         = (hp.ple_ngram - 1) * hp.ple_heads_per_ngram;
        for (const auto & v : kv.at("qwen4exp.ple.layer_multipliers")) hp.ple_mult.push_back(v.get<uint64_t>());
        for (const auto & v : kv.at("qwen4exp.ple.head_offsets"))      hp.ple_head_off.push_back(v.get<uint64_t>());
        for (const auto & v : kv.at("qwen4exp.ple.head_vocab_sizes"))  hp.ple_head_vocab.push_back(v.get<uint64_t>());
        if (hp.ple_mult.size() < hp.ple_ngram || hp.ple_head_off.size() < hp.ple_n_heads || hp.ple_head_vocab.size() < hp.ple_n_heads)
            throw std::runtime_error("PLE hash constants are incomplete");
        hp.tok_bos = u32(kv, "tokenizer.ggml.bos_token_id");
        hp.tok_eos = u32(kv, "tokenizer.ggml.eos_token_id");
        hp.tok_pad = u32(kv, "tokenizer.ggml.padding_token_id");
        hp.n_vocab = (uint32_t) kv.at("tokenizer.ggml.tokens").size();
        vocab = kv.at("tokenizer.ggml.tokens").get<std::vector<std::string>>();
        merges = kv.at("tokenizer.ggml.merges").get<std::vector<std::string>>();
        token_type = kv.at("tokenizer.ggml.token_type").get<std::vector<int32_t>>();
        tok_model = kv.at("tokenizer.ggml.model").get<std::string>();
        tok_pre = kv.at("tokenizer.ggml.pre").get<std::string>();
        if (kv.contains("tokenizer.ggml.normalizer_json")) {
            if (!kv.at("tokenizer.ggml.normalizer_json").is_string())
                throw std::runtime_error("tokenizer.ggml.normalizer_json must be a JSON string");
            tok_normalizer_json = kv.at("tokenizer.ggml.normalizer_json").get<std::string>();
            const auto normalizer = json::parse(tok_normalizer_json);
            if (!normalizer.is_null()) {
                if (normalizer != json{{"type", "NFC"}})
                    throw std::runtime_error("unsupported active tokenizer normalizer; require exactly NFC or null");
                if (tok_model != "gpt2") throw std::runtime_error("NFC normalizer requires GPT-2 BPE tokenizer");
                tok_normalize_nfc = true;
            }
        }
        add_bos = kv.contains("tokenizer.ggml.add_bos_token") && kv.at("tokenizer.ggml.add_bos_token").get<bool>();
        if (kv.contains("tokenizer.chat_template")) chat_template = kv.at("tokenizer.chat_template").get<std::string>();
        if (hp.n_embd * hp.hc == 0 || hp.ssm_dt_rank * hp.ssm_state != hp.ssm_inner || hp.kpool < 2 || hp.idx_top_k % hp.kpool)
            throw std::runtime_error("inconsistent qwen4exp hyperparameters");

        // Checkpoint sampling fallback < canonical GGUF < explicit base sampling metadata.
        json generation_defaults=json::object();
        for(const auto & item:kv.items()) {
            if(item.key().compare(0,17,"general.sampling.")!=0) continue;
            const std::string key=item.key().substr(17);
            generation_defaults[key]=item.value(); // strict canonicalization/validation below
        }
        const json base_sampling=meta.contains("sampling") && !meta.at("sampling").is_null() ? meta.at("sampling") : json::object();
        configure_generation("",generation_defaults.dump(),base_sampling.dump());

        // ---- opaque NVIDIA checkpoint calibration (never an ordinary ggml/GGUF tensor)
        if (meta.contains("checkpoint")) checkpoint_json = meta.at("checkpoint").dump();
        const bool has_descriptor = meta.contains("checkpoint") && meta.at("checkpoint").contains("experts") &&
            meta.at("checkpoint").at("experts").contains("input_scale");
        const bool has_payload = sections.count("experts.input_scale") != 0;
        if (has_descriptor != has_payload || (sections.count("experts.input_scale.index") != 0) != has_payload)
            throw std::runtime_error("NVFP4 calibration descriptor and experts.input_scale section must both be present");
        if (has_descriptor) {
            const auto & descriptor = meta.at("checkpoint").at("experts").at("input_scale");
            const json shape = {hp.n_layer, hp.n_expert, 3};
            const json roles = {"gate_proj", "up_proj", "down_proj"};
            if ((legacy_container && version != 2) || descriptor.at("section") != "experts.input_scale" ||
                descriptor.at("dtype") != "F32" || descriptor.at("shape") != shape ||
                descriptor.at("byte_order") != "little" || descriptor.at("role_order") != roles ||
                descriptor.at("scale_semantics") != "amax/(6*448)" ||
                descriptor.at("index_section") != "experts.input_scale.index")
                throw std::runtime_error("invalid NVIDIA experts.input_scale descriptor");
            const uint64_t elements = uint64_t(hp.n_layer) * hp.n_expert * 3;
            const auto payload = section("experts.input_scale");
            if (!elements || elements > SIZE_MAX / sizeof(float) ||
                payload.second != elements * sizeof(float))
                throw std::runtime_error("NVIDIA experts.input_scale byte count differs from [layer,expert,3]");
            expert_input_scale_offset = payload.first;
            expert_input_scale.resize(size_t(elements));
            if (!read_at(f, payload.first, expert_input_scale.data(), payload.second))
                throw std::runtime_error("short NVIDIA experts.input_scale");
            // Decode LE bits explicitly; retain exported scalars without reciprocal/requantization.
            auto * bytes = reinterpret_cast<const uint8_t *>(expert_input_scale.data());
            expert_input_scale_sha256 = sha256(bytes, size_t(payload.second));
            if (descriptor.at("sha256").get<std::string>() != expert_input_scale_sha256)
                throw std::runtime_error("NVIDIA experts.input_scale SHA256 mismatch");
            for (size_t i = 0; i < expert_input_scale.size(); ++i) {
                const uint32_t bits = uint32_t(bytes[4*i]) | (uint32_t(bytes[4*i+1]) << 8) |
                    (uint32_t(bytes[4*i+2]) << 16) | (uint32_t(bytes[4*i+3]) << 24);
                float value; std::memcpy(&value, &bits, sizeof value);
                if (!std::isfinite(value) || value <= 0)
                    throw std::runtime_error("NVIDIA experts.input_scale must contain finite positive scalars");
                expert_input_scale[i] = value;
            }
            const auto index = section("experts.input_scale.index");
            if (index.second > SIZE_MAX) throw std::runtime_error("NVIDIA calibration index exceeds native address space");
            expert_input_scale_index_json.resize(size_t(index.second));
            if (!read_at(f, index.first, expert_input_scale_index_json.data(), index.second))
                throw std::runtime_error("short NVIDIA experts.input_scale.index");
            (void) json::parse(expert_input_scale_index_json); // preserve every original provenance field verbatim
        }
        if (meta.contains("checkpoint") && meta.at("checkpoint").contains("source_metadata")) {
            const auto & metadata = meta.at("checkpoint").at("source_metadata");
            if (!metadata.is_object()) throw std::runtime_error("checkpoint source_metadata must be a filename map");
            for (auto it = metadata.begin(); it != metadata.end(); ++it) {
                const auto & descriptor = it.value();
                const std::string name = descriptor.at("section").get<std::string>();
                if (name != "checkpoint/" + it.key()) throw std::runtime_error("checkpoint source metadata section mismatch");
                const auto source = section(name);
                if (source.second != descriptor.at("bytes").get<uint64_t>() || source.second > SIZE_MAX)
                    throw std::runtime_error("checkpoint source metadata byte count mismatch");
                std::vector<uint8_t> raw(size_t(source.second));
                if (!read_at(f, source.first, raw.data(), source.second))
                    throw std::runtime_error("short checkpoint source metadata section");
                if (sha256(raw.data(), raw.size()) != descriptor.at("sha256").get<std::string>())
                    throw std::runtime_error("checkpoint source metadata SHA256 mismatch");
                // Interpret only after authentication; the original section remains byte-for-byte preserved.
                if(it.key()=="generation_config.json")
                    configure_generation(std::string(raw.begin(),raw.end()),generation_defaults_json);
            }
        }

        // ---- experts.layout
        const auto es = section("experts");
        const auto ls = section("experts.layout");
        std::string layout(ls.second, '\0');
        if (!read_at(f, ls.first, layout.data(), ls.second)) throw std::runtime_error("short experts.layout");
        std::istringstream lin(layout);
        std::string line;
        std::getline(lin, line);
        int header_end = 0;
        if (std::sscanf(line.c_str(), "# experts v%u%n", &expert_layout_version, &header_end) != 1 ||
            (expert_layout_version != 5 && expert_layout_version != 6 && expert_layout_version != 7) ||
            (line.size() > (size_t) header_end && line[header_end] != ':' && line[header_end] != ' '))
            throw std::runtime_error("unsupported experts.layout: " + line);
        native_nvfp4_packed = expert_layout_version == 7;
        if (native_nvfp4_packed) {
            const std::string expected = "flashinfer-sm120-nvfp4-e16-expert-major-v1";
            if (line != "# experts v7 " + expected || !meta.contains("qfn_native_nvfp4"))
                throw std::runtime_error("native NVFP4 requires its exact v7 layout and metadata descriptor");
            const auto & native = meta.at("qfn_native_nvfp4");
            if (native.at("layout").get<std::string>() != expected || !native.at("lossless").get<bool>() ||
                native.at("expert_stride").get<uint64_t>() != 2764816 ||
                native.at("tail_offset").get<uint64_t>() != 2764800 || native.at("tail_bytes").get<unsigned>() != 16)
                throw std::runtime_error("native NVFP4 metadata layout/stride/tail mismatch");
        } else if (meta.contains("qfn_native_nvfp4")) {
            throw std::runtime_error("native NVFP4 metadata requires experts.layout v7");
        }
        if (!hp.n_layer || !hp.n_expert) throw std::runtime_error("empty expert geometry");
        experts.assign(hp.n_layer, {});
        std::vector<bool> layer_seen(hp.n_layer, false);
        uint32_t seen = 0;
        while (std::getline(lin, line)) {
            if (line.empty()) continue;
            unsigned l = 0, gt = 0, dt = 0;
            unsigned long long off = 0, blob = 0;
            if (std::sscanf(line.c_str(), "%u %u %u %llu %llu", &l, &gt, &dt, &off, &blob) != 5 || l >= hp.n_layer)
                throw std::runtime_error("bad experts.layout line: " + line);
            if (layer_seen[l]) throw std::runtime_error("duplicate experts.layout layer " + std::to_string(l));
            layer_seen[l] = true;
            auto & e = experts[l];
            e = expert_format(gt, dt, hp.n_embd, hp.n_ff_exp);
            if (e.is_nvfp4() && expert_layout_version != 6 && expert_layout_version != 7)
                throw std::runtime_error("NVFP4 scale tails require experts.layout v6/v7");
            if (native_nvfp4_packed && !e.is_nvfp4())
                throw std::runtime_error("native NVFP4 v7 requires NVFP4 main experts in every layer");
            if (blob != e.blob) throw std::runtime_error("expert blob differs from gate|up|down and its declared scale tail");
            if (off > es.second || blob > (es.second - off) / hp.n_expert)
                throw std::runtime_error("experts beyond their section");
            e.offset  = es.first + off;
            ++seen;
        }
        if (seen != hp.n_layer) throw std::runtime_error("experts.layout does not cover every layer");

        // ---- profile: 5 x u32 header, pairs, rank table
        if (sections.count("profile")) {
            const auto ps = section("profile");
            std::vector<uint8_t> prof(ps.second);
            if (!read_at(f, ps.first, prof.data(), prof.size())) throw std::runtime_error("short profile");
            uint32_t h[5];
            std::memcpy(h, prof.data(), 20);
            if (h[0] != 1 || h[1] != hp.n_layer || h[2] != hp.n_expert) throw std::runtime_error("profile geometry differs");
            const uint64_t n_pairs = h[4];
            if (prof.size() != 20 + 4 * n_pairs + 4ull * hp.n_layer * hp.n_expert) throw std::runtime_error("profile size");
            profile_pairs.resize(n_pairs);
            for (uint64_t i = 0; i < n_pairs; ++i) {
                uint16_t le[2];
                std::memcpy(le, prof.data() + 20 + 4 * i, 4);
                profile_pairs[i] = { le[0], le[1] };
            }
            profile_rank.resize((size_t) hp.n_layer * hp.n_expert);
            std::memcpy(profile_rank.data(), prof.data() + 20 + 4 * n_pairs, profile_rank.size() * 4);
        }

        // ---- MTP: "name type rows cols offset bytes" per line (HF [out, in] order), offsets within mtp/dense
        if (sections.count("mtp/dense") && sections.count("mtp/dense.index") && sections.count("mtp/experts")) {
            const auto ds = section("mtp/dense");
            const auto is = section("mtp/dense.index");
            std::string text(is.second, '\0');
            if (!read_at(f, is.first, text.data(), is.second)) throw std::runtime_error("short mtp/dense.index");
            std::istringstream in(text);
            std::string ln;
            while (std::getline(in, ln)) {
                if (ln.empty() || ln[0] == '#') continue;
                char name[256], type[32];
                unsigned long long rows = 0, cols = 0, off = 0, bytes = 0;
                if (std::sscanf(ln.c_str(), "%255s %31s %llu %llu %llu %llu", name, type, &rows, &cols, &off, &bytes) != 6)
                    throw std::runtime_error("bad mtp/dense.index line: " + ln);
                qfn_tensor t;
                t.name = name;
                const std::string ty = type;
                t.type = ty == "q8_0" ? GGML_TYPE_Q8_0 : ty == "bf16" ? GGML_TYPE_BF16 : ty == "f32" ? GGML_TYPE_F32 :
                         ty == "f16" ? GGML_TYPE_F16 : GGML_TYPE_COUNT;
                if (t.type == GGML_TYPE_COUNT) throw std::runtime_error("unsupported MTP tensor type " + ty);
                t.n_dims = rows == 1 ? 1 : 2;
                t.ne[0] = (int64_t) (rows == 1 ? cols : cols);
                t.ne[1] = (int64_t) (rows == 1 ? 1 : rows);
                t.offset = ds.first + off;
                t.bytes = bytes;
                if (off + bytes > ds.second || ggml_row_size(t.type, t.ne[0]) * (uint64_t) t.ne[1] != bytes)
                    throw std::runtime_error("MTP tensor geometry differs: " + t.name);
                mtp_tensors[t.name] = t;
            }
            const auto xs = section("mtp/experts");
            mtp_experts_offset = xs.first;
            mtp_expert_blob = xs.second / hp.n_expert;
            mtp_expert_layout = experts[0];
            // Basalt checkpoint conversion copies the original Q2_0 MTP sections byte-for-byte;
            // its drafter format must not be inferred from the newly converted NVFP4 main model.
            if (experts[0].is_nvfp4() && mtp_expert_blob != experts[0].blob)
                mtp_expert_layout = expert_format(GGML_TYPE_Q2_0, GGML_TYPE_Q2_0, hp.n_embd, hp.n_ff_exp);
            mtp_expert_layout.offset = xs.first;
            if (mtp_expert_blob * hp.n_expert != xs.second || mtp_expert_blob != mtp_expert_layout.blob)
                throw std::runtime_error("mtp/experts geometry differs");
            if (sections.count("mtp/draft_vocab")) {
                const auto vs = section("mtp/draft_vocab");
                if (vs.second % 4) throw std::runtime_error("mtp/draft_vocab size");
                draft_vocab.resize(vs.second / 4);
                if (!read_at(f, vs.first, draft_vocab.data(), vs.second)) throw std::runtime_error("short mtp/draft_vocab");
                for (int32_t v : draft_vocab) if (v < 0 || (uint32_t) v >= hp.n_vocab) throw std::runtime_error("draft vocab id out of range");
            }
        }
        return true;
    } catch (const std::exception & e) {
        err = e.what();
        return false;
    }
}

bool qfn_pack::runtime_supported(std::string & err) const {
    for (size_t l = 0; l < experts.size(); ++l) {
        if (experts[l].is_nvfp4()) {
            err = "QFN_NVFP4_BRIDGE_REQUIRED: parsed NVFP4 expert layer " + std::to_string(l) +
                  " including its 16-byte scale tail; execution requires the native bridge DLL/ABI/format admission";
            return false;
        }
    }
    if (mtp_expert_layout.is_nvfp4()) {
        err = "QFN_NVFP4_MTP_FORMAT_UNSUPPORTED: NVFP4 MTP experts require their own relayout implementation";
        return false;
    }
    err.clear();
    return true;
}

bool qfn_pack::expert_scale_tail(uint32_t layer, uint32_t expert, std::array<float, 4> & scales, std::string & err) const {
    try {
        if (layer >= experts.size() || expert >= hp.n_expert) throw std::runtime_error("expert scale-tail index out of range");
        const auto & e = experts[layer];
        if (e.tail_bytes != sizeof(scales)) throw std::runtime_error("expert has no NVFP4 scale tail");
        std::ifstream f(path, std::ios::binary);
        if (!f || !read_at(f, e.offset + e.blob * expert + e.tail_offset, scales.data(), sizeof(scales)))
            throw std::runtime_error("short NVFP4 expert scale tail");
        err.clear();
        return true;
    } catch (const std::exception & e) {
        err = e.what();
        return false;
    }
}
