#pragma once
// QFN: the model's tokenizer (GPT-2 byte-level BPE, the qwen35 pre-tokenizer) through this tree's llama_vocab.
// The pack carries the source GGUF's tokenizer metadata; it is written once into a vocab-only GGUF beside the
// runtime state and loaded by llama_vocab, so tokenization is the same code path upstream llama.cpp runs.
#include "qfn-pack.h"

#include <memory>
#include <string>
#include <vector>

class qfn_tokenizer {
public:
    qfn_tokenizer();
    ~qfn_tokenizer();
    // vocab_gguf: where to write/read the vocab-only GGUF (reused when it already matches)
    bool open(const qfn_pack & pack, const std::string & vocab_gguf, std::string & err);
    std::vector<int32_t> tokenize(const std::string & text, bool parse_special) const;
    std::string piece(int32_t token) const;       // the token's text (bytes; special tokens as their text)
    static bool self_test(const qfn_pack & pack, const std::string & vocab_path,
                          const std::string & cases_json, std::string & err);
private:
    struct impl;
    std::unique_ptr<impl> p;
};
