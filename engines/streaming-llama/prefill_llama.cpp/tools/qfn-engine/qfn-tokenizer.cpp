// QFN: tokenizer via llama_vocab (see qfn-tokenizer.h).
#include "qfn-tokenizer.h"

#include "gguf.h"
#include "llama-arch.h"
#include "llama-model-loader.h"
#include "llama-vocab.h"
#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstring>
#include <climits>
#define NOMINMAX
#include <windows.h>
#include <winnls.h>
#include <filesystem>
#include <stdexcept>


namespace {
void validate_vocab_cache(const std::string & path, const qfn_pack & pack) {
    gguf_context * g = gguf_init_from_file(path.c_str(), {true, nullptr});
    if (!g) throw std::runtime_error("cannot inspect tokenizer cache " + path);
    auto key = [&](const char * name, gguf_type type) {
        const int64_t k = gguf_find_key(g, name);
        return k >= 0 && gguf_get_kv_type(g, k) == type ? k : int64_t(-1);
    };
    auto text = [&](const char * name, const std::string & expected) {
        const auto k=key(name,GGUF_TYPE_STRING); return k>=0 && expected==gguf_get_val_str(g,k);
    };
    auto strings = [&](const char * name, const std::vector<std::string> & expected) {
        const auto k=key(name,GGUF_TYPE_ARRAY);
        if (k<0 || gguf_get_arr_type(g,k)!=GGUF_TYPE_STRING || gguf_get_arr_n(g,k)!=expected.size()) return false;
        for (size_t i=0;i<expected.size();++i) if (expected[i]!=gguf_get_arr_str(g,k,i)) return false;
        return true;
    };
    auto number = [&](const char * name, uint32_t expected) {
        const auto k=key(name,GGUF_TYPE_UINT32); return k>=0 && gguf_get_val_u32(g,k)==expected;
    };
    const auto types=key("tokenizer.ggml.token_type",GGUF_TYPE_ARRAY), bos=key("tokenizer.ggml.add_bos_token",GGUF_TYPE_BOOL);
    const bool matching = text("tokenizer.ggml.model",pack.tok_model) && text("tokenizer.ggml.pre",pack.tok_pre) &&
        strings("tokenizer.ggml.tokens",pack.vocab) && strings("tokenizer.ggml.merges",pack.merges) &&
        types>=0 && gguf_get_arr_type(g,types)==GGUF_TYPE_INT32 && gguf_get_arr_n(g,types)==pack.token_type.size() &&
        (pack.token_type.empty() || std::memcmp(gguf_get_arr_data(g,types),pack.token_type.data(),pack.token_type.size()*sizeof(int32_t))==0) &&
        number("tokenizer.ggml.bos_token_id",pack.hp.tok_bos) && number("tokenizer.ggml.eos_token_id",pack.hp.tok_eos) &&
        number("tokenizer.ggml.padding_token_id",pack.hp.tok_pad) && bos>=0 && gguf_get_val_bool(g,bos)==pack.add_bos &&
        (pack.tok_normalizer_json.empty() ? gguf_find_key(g,"tokenizer.ggml.normalizer_json")<0
            : text("tokenizer.ggml.normalizer_json",pack.tok_normalizer_json));
    gguf_free(g);
    if (!matching) throw std::runtime_error("vocab GGUF tokenizer metadata differs from pack; use a fresh explicit vocab cache path: " + path);
}

std::wstring strict_utf8(const std::string & text) {
    if (text.empty()) return {};
    if (text.size() > size_t(INT_MAX)) throw std::runtime_error("NFC input exceeds Windows UTF-8 length limit");
    const int size = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.data(), int(text.size()), nullptr, 0);
    if (!size) throw std::runtime_error("NFC tokenizer requires valid UTF-8");
    std::wstring wide(size, L'\0');
    if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.data(), int(text.size()), wide.data(), size) != size)
        throw std::runtime_error("NFC UTF-8 conversion failed");
    return wide;
}

std::string normalize_nfc(const std::string & text) {
    const auto wide = strict_utf8(text);
    if (wide.empty()) return {};
    int size = NormalizeString(NormalizationC, wide.data(), int(wide.size()), nullptr, 0);
    if (size <= 0) throw std::runtime_error("NFC normalization size query failed");
    std::wstring composed(size, L'\0');
    for (;;) {
        const int count = NormalizeString(NormalizationC, wide.data(), int(wide.size()), composed.data(), int(composed.size()));
        if (count > 0) { composed.resize(count); break; }
        if (count == INT_MIN || count >= 0 || size_t(-count) <= composed.size() || GetLastError() != ERROR_INSUFFICIENT_BUFFER)
            throw std::runtime_error("NFC normalization failed");
        composed.resize(size_t(-count));
    }
    const int bytes = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, composed.data(), int(composed.size()), nullptr, 0, nullptr, nullptr);
    if (bytes <= 0) throw std::runtime_error("NFC UTF-8 output size query failed");
    std::string result(bytes, '\0');
    if (WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, composed.data(), int(composed.size()), result.data(), bytes, nullptr, nullptr) != bytes)
        throw std::runtime_error("NFC UTF-8 output conversion failed");
    return result;
}
}

struct qfn_tokenizer::impl {
    llama_vocab vocab;
    bool normalize_nfc = false;
};

qfn_tokenizer::qfn_tokenizer() : p(new impl) {}
qfn_tokenizer::~qfn_tokenizer() = default;

bool qfn_tokenizer::open(const qfn_pack & pack, const std::string & path, std::string & err) {
    try {
        p->normalize_nfc = pack.tok_normalize_nfc;
        if (!std::filesystem::exists(path)) {
            gguf_context * g = gguf_init_empty();
            // a known architecture keeps the loader generic; only the tokenizer.* keys are read
            gguf_set_val_str(g, "general.architecture", "qwen2");
            gguf_set_val_str(g, "tokenizer.ggml.model", pack.tok_model.c_str());
            gguf_set_val_str(g, "tokenizer.ggml.pre", pack.tok_pre.c_str());
            std::vector<const char *> toks, merges;
            for (const auto & t : pack.vocab) toks.push_back(t.c_str());
            for (const auto & m : pack.merges) merges.push_back(m.c_str());
            gguf_set_arr_str(g, "tokenizer.ggml.tokens", toks.data(), toks.size());
            gguf_set_arr_str(g, "tokenizer.ggml.merges", merges.data(), merges.size());
            gguf_set_arr_data(g, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, pack.token_type.data(), pack.token_type.size());
            gguf_set_val_u32(g, "tokenizer.ggml.bos_token_id", pack.hp.tok_bos);
            gguf_set_val_u32(g, "tokenizer.ggml.eos_token_id", pack.hp.tok_eos);
            gguf_set_val_u32(g, "tokenizer.ggml.padding_token_id", pack.hp.tok_pad);
            gguf_set_val_bool(g, "tokenizer.ggml.add_bos_token", pack.add_bos);
            if (!pack.tok_normalizer_json.empty())
                gguf_set_val_str(g, "tokenizer.ggml.normalizer_json", pack.tok_normalizer_json.c_str());
            const std::string tmp = path + ".partial";
            const bool ok = gguf_write_to_file(g, tmp.c_str(), false);
            gguf_free(g);
            if (!ok) throw std::runtime_error("cannot write the vocab GGUF " + tmp);
            std::filesystem::rename(tmp, path);
        }
        validate_vocab_cache(path, pack);
        std::vector<std::string> splits;
        llama_model_loader ml(path, splits, false, false, false, true, nullptr, nullptr);
        p->vocab.load(ml, LLM_KV(LLM_ARCH_QWEN2));
        if ((size_t) p->vocab.n_tokens() != pack.vocab.size()) throw std::runtime_error("vocab size differs from the pack");
        return true;
    } catch (const std::exception & e) {
        err = e.what();
        return false;
    }
}

std::vector<int32_t> qfn_tokenizer::tokenize(const std::string & text, bool parse_special) const {
    if (!p->normalize_nfc) return p->vocab.tokenize(text, false, parse_special);
    // Validate the entire input before partitioning; no malformed byte may hide in a special token.
    (void) strict_utf8(text);
    return p->vocab.tokenize(text, false, parse_special, ::normalize_nfc);
}

std::string qfn_tokenizer::piece(int32_t token) const {
    char buf[512];
    const int32_t n = p->vocab.token_to_piece(token, buf, sizeof(buf), 0, true);
    return n > 0 ? std::string(buf, (size_t) n) : std::string();
}

bool qfn_tokenizer::self_test(const qfn_pack & pack, const std::string & path,
                              const std::string & cases_json, std::string & err) {
    try {
        int checks = 0;
        auto expect = [&](bool ok, const char * what) { if (!ok) throw std::runtime_error(what); ++checks; };
        qfn_pack normalized_pack = pack; normalized_pack.tok_normalize_nfc = true;
        normalized_pack.tok_normalizer_json = R"({"type":"NFC"})";
        qfn_tokenizer normalized;
        const auto nfc_path = path + ".nfc.gguf";
        if (!normalized.open(normalized_pack, nfc_path, err)) return false;
        const auto fixture = nlohmann::json::parse(cases_json);
        for (const auto & test : fixture.at("cases")) {
            expect(normalized.tokenize(test.at("text").get<std::string>(), test.at("parse_special").get<bool>()) ==
                   test.at("expected_ids").get<std::vector<int32_t>>(), "actual HF NFC token IDs differ");
        }
        qfn_pack legacy_pack = pack; legacy_pack.tok_normalize_nfc = false; legacy_pack.tok_normalizer_json.clear();
        qfn_tokenizer legacy;
        if (!legacy.open(legacy_pack, path, err)) return false;
        qfn_tokenizer cache_hit;
        expect(cache_hit.open(normalized_pack,nfc_path,err), "matching NFC cache refused");
        qfn_tokenizer cache_miss;
        expect(!cache_miss.open(normalized_pack,path,err) && err.find("fresh explicit vocab cache")!=std::string::npos,
               "legacy cache silently omitted NFC metadata");
        expect(!cache_miss.open(legacy_pack,nfc_path,err), "NFC cache silently accepted an absent-normalizer pack");
        for (int field=0;field<10;++field) {
            auto changed=normalized_pack;
            switch (field) {
                case 0: changed.tok_model += "-different"; break;
                case 1: changed.tok_pre += "-different"; break;
                case 2: changed.vocab[0] += "-different"; break;
                case 3: if (changed.merges.empty()) changed.merges.push_back("different merge"); else changed.merges[0]+="-different"; break;
                case 4: changed.token_type[0] ^= 1; break;
                case 5: ++changed.hp.tok_bos; break;
                case 6: ++changed.hp.tok_eos; break;
                case 7: ++changed.hp.tok_pad; break;
                case 8: changed.add_bos = !changed.add_bos; break;
                case 9: changed.tok_normalizer_json="null"; changed.tok_normalize_nfc=false; break;
            }
            expect(!cache_miss.open(changed,nfc_path,err) && err.find("fresh explicit vocab cache")!=std::string::npos,
                   "same-size wrong tokenizer cache passed full metadata comparison");
        }
        const auto bad_cache = path + ".nfc.bad-normalizer.gguf";
        if (!std::filesystem::exists(bad_cache)) {
            auto * raw=gguf_init_from_file(nfc_path.c_str(),{true,nullptr});
            if (!raw) throw std::runtime_error("cannot read generated NFC cache fixture");
            gguf_set_val_u32(raw,"tokenizer.ggml.normalizer_json",7);
            const bool written=gguf_write_to_file(raw,bad_cache.c_str(),false); gguf_free(raw);
            expect(written,"cannot create wrong-type raw normalizer cache fixture");
        }
        expect(!cache_miss.open(normalized_pack,bad_cache,err),"nonstring raw normalizer cache silently accepted");
        for (const std::string text : {std::string("ASCII unchanged 123\n<|im_end|>"), std::string("A\0B", 3), std::string()})
            for (bool special : {false, true})
                expect(normalized.tokenize(text, special) == legacy.tokenize(text, special), "NFC changed ASCII/NUL/empty input");
        expect(normalized.tokenize("cafe\xcc\x81", true) != legacy.tokenize("cafe\xcc\x81", true),
               "legacy absent normalizer silently acquired NFC");
        for (const std::string bad : {std::string("\x80"), std::string("\xc0\xaf"), std::string("\xe2\x82"),
                                     std::string("\xed\xa0\x80"), std::string("\xf4\x90\x80\x80")}) {
            bool refused = false;
            try { (void) normalized.tokenize(bad, true); } catch (const std::runtime_error &) { refused = true; }
            expect(refused, "NFC silently accepted malformed UTF-8");
        }
        // Exercise the REAL upstream special partition with Unicode literals that
        // full-string normalization would destroy, including USER_DEFINED with parse_special=false.
        qfn_pack unicode_pack = normalized_pack;
        const int32_t control = int32_t(unicode_pack.vocab.size()), user = control + 1;
        unicode_pack.vocab.push_back("e\xcc\x81"); unicode_pack.token_type.push_back(3);
        unicode_pack.vocab.push_back("A\xcc\x8a"); unicode_pack.token_type.push_back(4);
        unicode_pack.hp.n_vocab = uint32_t(unicode_pack.vocab.size());
        qfn_tokenizer unicode;
        if (!unicode.open(unicode_pack, path + ".nfc.unicode-specials.gguf", err)) return false;
        auto expected = normalized.tokenize("x", true); expected.push_back(control);
        const auto tail = normalized.tokenize("y", true); expected.insert(expected.end(), tail.begin(), tail.end());
        expect(unicode.tokenize("xe\xcc\x81y", true) == expected, "NFC changed a Unicode CONTROL literal");
        expect(unicode.tokenize("xe\xcc\x81y", false) == normalized.tokenize("xe\xcc\x81y", false),
               "parse_special=false did not normalize ordinary CONTROL spelling");
        expected = normalized.tokenize("x", false); expected.push_back(user);
        expected.insert(expected.end(), tail.begin(), tail.end());
        expect(unicode.tokenize("xA\xcc\x8ay", false) == expected && unicode.tokenize("xA\xcc\x8ay", true) == expected,
               "NFC changed Unicode USER_DEFINED parsing");
        std::printf("Tokenizer NFC PASS: %d checks, actual HF cases=%zu, strict UTF-8 and upstream Unicode special partition; no CUDA\n",
                    checks, fixture.at("cases").size());
        return true;
    } catch (const std::exception & e) { err = e.what(); return false; }
}
