// QFN: qfn-engine command line.
//   qfn-engine --model PACK.basalt --tokens FILE.le32 --out PREFIX [--out-start N] [--top1-all] [--ctx N] [--tile N]
// Sampling: --temp, --top-k, --top-p, --min-p, --typical-p, --repeat-penalty,
// --repeat-last-n (-1 whole history, 0 off), --presence-penalty, --frequency-penalty,
// --seed, --logit-bias JSON (token ids or text), --stop TEXT, --max-tokens N.
// --mtp 0/1 controls speculative decoding. With --serve these sampling flags
// provide defaults, individually overridden by each chat/generate request.
// --pack-check --model PACK validates/describes the container without initializing CUDA or executing weights.
// Prefills the token file layer-major and writes, like the upstream reference tool (llama-qfn-logits):
//   PREFIX.f32   float32 logits rows for positions out-start..n-1
//   PREFIX.top1  int32 argmax per position, PREFIX.lse float32 logsumexp per position (with --top1-all)
//   PREFIX.json  a small header
#include "qfn-engine.h"
#include "qfn-state.h"
#include "qfn-tokenizer.h"
#include "qfn-vision.h"
#include "ggml-qfn-nvfp4.h"
#include "ggml-qfn-qsa-score.h"
#include "qfn-qsa-node-test.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <stdexcept>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <string>
#include <vector>

#include <csignal>
#include <exception>
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

// Every abnormal end says why on stderr: a silent exit code cannot tell a crash from an outside termination.
static LONG WINAPI qfn_unhandled(EXCEPTION_POINTERS * e) {
    std::fprintf(stderr, "[qfn] FATAL: unhandled exception 0x%08lx at %p\n", e->ExceptionRecord->ExceptionCode,
                 e->ExceptionRecord->ExceptionAddress);
    std::fflush(stderr);
    return EXCEPTION_CONTINUE_SEARCH;
}
static void qfn_on_abort(int) { std::fputs("[qfn] FATAL: abort()\n", stderr); std::fflush(stderr); }
static void qfn_on_terminate() {
    std::fputs("[qfn] FATAL: std::terminate (uncaught C++ exception)\n", stderr);
    std::fflush(stderr);
    std::abort();
}
static void qfn_on_exit() { std::fputs("[qfn] process exit\n", stderr); std::fflush(stderr); }

// GPT-2 byte-level BPE: token strings map bytes to printable code points; invert that map for display.
static std::string detok(const std::string & s) {
    static int byte_of[0x200];
    static bool init = false;
    if (!init) {
        for (int i = 0; i < 0x200; ++i) byte_of[i] = -1;
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            const bool printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
            if (printable) byte_of[b] = b;
            else byte_of[256 + n++] = b;
        }
        init = true;
    }
    if (s.size() > 4 && s.rfind("<|", 0) == 0) return s;  // special tokens are shown as they are
    std::string out;
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = (unsigned char) s[i];
        uint32_t cp; int len;
        if (c < 0x80) { cp = c; len = 1; }
        else if ((c >> 5) == 6) { cp = c & 0x1F; len = 2; }
        else if ((c >> 4) == 14) { cp = c & 0x0F; len = 3; }
        else { cp = c & 0x07; len = 4; }
        for (int k = 1; k < len && i + k < s.size(); ++k) cp = (cp << 6) | ((unsigned char) s[i + k] & 0x3F);
        i += len;
        if (cp < 0x200 && byte_of[cp] >= 0) out.push_back((char) byte_of[cp]);
        else out += "?";
    }
    return out;
}

static std::vector<int32_t> read_tokens(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    std::vector<int32_t> t;
    if (!f) return t;
    f.seekg(0, std::ios::end);
    const size_t n = (size_t) f.tellg() / 4;
    f.seekg(0);
    t.resize(n);
    f.read((char *) t.data(), (std::streamsize) (n * 4));
    return t;
}

int qfn_serve(qfn_options opt, const std::string & vocab_gguf, int port, const std::string & defaults);

int main(int argc, char ** argv) {
    SetUnhandledExceptionFilter(qfn_unhandled);
    std::signal(SIGABRT, qfn_on_abort);
    std::set_terminate(qfn_on_terminate);
    std::atexit(qfn_on_exit);
    if(argc==2&&std::string(argv[1])=="--paired-layout-self-test") {
        try {
            const qfn_native_shape shape={4,4,2,2,2,1,16,2,2,6,3,2,4,6,16,1};
            qfn_native_plan plan{};
            auto check=[](bool value,const char *why){if(!value)throw std::runtime_error(why);};
            check(qfn_native_geometry(&shape,4,5,4,5,0,&plan)&&plan.bytes==580&&plan.count==17,
                "independent one-row frontier accounting differs");
            bool parent=false,pool=false;
            for(uint32_t i=0;i<plan.count;i++) {
                const auto &r=plan.slices[i];
                if(r.owner==QFN_KEY&&r.layer==4)parent=r.offset==12&&r.bytes==8;
                if(r.owner==QFN_POOL&&r.layer==4)pool=r.offset==0&&r.bytes==4;
            }
            check(parent&&pool,"provisional predictor row / pool boundary omitted");
            check(qfn_native_geometry(&shape,4,5,4,5,1,&plan)&&plan.bytes==596&&plan.count==18,
                "reverse target carry omitted");
            check(!qfn_native_geometry(&shape,5,5,5,5,0,&plan),"empty delta accepted");
            check(!qfn_native_geometry(&shape,6,5,5,5,0,&plan),"backward delta accepted");
            check(!qfn_native_geometry(&shape,4,17,4,5,0,&plan),"context overflow accepted");
            check(!qfn_native_geometry(&shape,4,5,5,5,0,&plan),"base predictor exceeds base");
            check(!qfn_native_geometry(&shape,4,5,4,6,0,&plan),"target predictor exceeds target");
            auto no_mtp=shape;no_mtp.mtp=0;
            check(qfn_native_geometry(&no_mtp,4,5,0,0,1,&plan)&&plan.bytes==560&&plan.count==14,
                "MTP-off family layout differs");
            check(!qfn_native_geometry(&no_mtp,4,5,1,1,0,&plan),"MTP owners silently accepted when absent");
            auto invalid=shape;invalid.index_dim=UINT32_MAX;
            check(!qfn_native_geometry(&invalid,4,5,4,5,0,&plan),"oversized geometry accepted");
            check(qfn_native_anchor_geometry(&shape,5,5,&plan)&&plan.bytes==464&&plan.count==13,
                "mutable-only anchor accounting differs");
            for(uint32_t i=0;i<plan.count;i++)check(plan.slices[i].owner!=QFN_POS&&
                (plan.slices[i].owner<QFN_KEY||plan.slices[i].owner>QFN_POOL||plan.slices[i].layer==shape.layers),
                "anchor copied immutable target prefix caches");
            check(!qfn_native_anchor_geometry(&shape,0,0,&plan),"empty anchor accepted");
            std::puts("paired layout self-test PASS: canonical owner extents, provisional MTP row, pool crossing, reverse carry, absent-MTP and malformed-frontier refusals");return 0;
        }catch(const std::exception &e){std::fprintf(stderr,"paired layout self-test failed: %s\n",e.what());return 16;}
    }
    if ((argc == 3 || argc == 5 || argc == 7) && std::string(argv[1]) == "--state-codec-fixture") {
        try {
            qfn_state::shape s{16,4,5,4,2,2,2,1,16,4,2,2,6,3,2,4,6,0,true};
            bool narrow=false;
            for(int i=3;i<argc;i+=2) {
                const std::string flag=argv[i],value=argv[i+1];
                if(flag=="--state-codec-mtp-rows" && value=="0")s.mtp_rows=0;
                else if(flag=="--state-codec-narrow-index" && value=="1")narrow=true;
                else throw std::runtime_error("unknown codec fixture setting");
            }
            FILE *fp = nullptr;
            if (fopen_s(&fp, argv[2], "wbx") || !fp) throw std::runtime_error("fixture output must not exist");
            try {
                auto read = [](const std::string &owner, uint32_t layer, uint64_t off, void *dst, size_t n) {
                    auto *out = static_cast<uint8_t *>(dst);
                    const bool half = owner == "key" || owner == "value" || owner == "index" || owner == "pool";
                    const uint32_t width = half ? 2 : 4;
                    qfn_state::require(off % width == 0 && n % width == 0, "unaligned synthetic owner read");
                    for (size_t j = 0; j < n / width; ++j) {
                        if (half) { uint16_t x = uint16_t(0x3c00 + layer * 16 + off / 2 + j); std::memcpy(out + j * 2, &x, 2); }
                        else { float x = float(layer * 1000 + off / 4 + j); std::memcpy(out + j * 4, &x, 4); }
                    }
                };
                qfn_state::write_to([&](const void *p,size_t n){if(fwrite(p,1,n,fp)!=n)throw std::runtime_error("fixture write failed");},
                    s,{1,2,3,4,5},std::vector<float>(16,0.25f),qfn_state::payload_bytes(s,narrow),
                    read,[](uint16_t x){return ggml_fp16_to_fp32(x);},narrow);
            } catch (...) { fclose(fp); throw; }
            if (fclose(fp)) throw std::runtime_error("fixture close failed");
            std::printf("QFN complete-state codec fixture: %llu bytes\n", (unsigned long long)qfn_state::payload_bytes(s,narrow));
            return 0;
        } catch (const std::exception &e) { std::fprintf(stderr, "%s\n", e.what()); return 2; }
    }
    qfn_options opt;
    std::string tokens_path, out_prefix;
    int64_t out_start = -1;
    bool top1_all = false;
    int serve_port = 0;
    int64_t feed_from = -1;   // decode-path check: prefill toks[0..N), then feed the rest through the decode path
    int feed_chunk = 1;
    int n_gen = 0;
    qfn_sampling smp;
    nlohmann::ordered_json sampling_body = nlohmann::ordered_json::object();
    std::vector<std::string> stops;
    int max_tokens = INT32_MAX;
    bool max_tokens_set = false;
    std::string gen_out, vocab_gguf, tokenizer_cases;
    bool validate_options = false;
    bool serve = false, sampling_test = false, cache_policy_test = false, nvfp4_node_test = false, qsa_node_test = false, pack_check = false, pack_check_other_option = false;
    bool protect_host_working_set = false;
    bool vision_encode=false,vision_flash_attention=false;
    bool vision_args_seen=false;
    std::string vision_model,vision_image,vision_output,vision_backend="CUDA0";
    uint32_t vision_min_tokens=64,vision_max_tokens=1024;int vision_threads=8;
    try {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if(a.rfind("--vision-",0)==0)vision_args_seen=true;
        if (a != "--pack-check" && a != "--model") pack_check_other_option = true;
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        auto number = [&]() {
            const auto raw = next(); size_t used = 0;
            const double value = std::stod(raw, &used);
            if (!std::isfinite(value) || raw.find_first_not_of(" \t\r\n", used) != std::string::npos)
                throw std::runtime_error("invalid numeric value for " + a);
            return value;
        };
        auto integer = [&]() -> nlohmann::ordered_json {
            const auto raw = next(); size_t used = 0;
            nlohmann::ordered_json value;
            if (!raw.empty() && raw[0] == '-') value = std::stoll(raw, &used);
            else value = std::stoull(raw, &used);
            if (raw.find_first_not_of(" \t\r\n", used) != std::string::npos)
                throw std::runtime_error("invalid integer value for " + a);
            return value;
        };
        if (a == "--sampling-self-test") sampling_test = true;
        else if (a == "--cache-policy-self-test") cache_policy_test = true;
        else if (a == "--nvfp4-node-self-test") nvfp4_node_test = true;
        else if (a == "--tokenizer-check") tokenizer_cases = next();
        else if (a == "--qsa-node-self-test") qsa_node_test = true;
        else if (a == "--qsa-indexer") {
            const auto mode = next();
            if (mode != "auto" && mode != "scalar" && mode != "basalt-tc-refine-prefill" && mode != "basalt-tc-refine" && mode != "basalt-tc" && mode != "basalt-tc-prefill" && mode != "dsv-fp4")
                throw std::runtime_error("--qsa-indexer must be auto, scalar, basalt-tc-refine-prefill, basalt-tc-refine, basalt-tc, dsv-fp4 or basalt-tc-prefill (TC for >=64 queries, scalar below64)");
            opt.qsa_indexer = mode;
        }
        else if (a == "--nvfp4-mode") {
            const auto mode = next();
            if (mode != "auto" && mode != "w4a4" && mode != "native-q8" && mode != "mmq" && mode != "calibrated-w4a4" && mode != "dsv-native")
                throw std::runtime_error("--nvfp4-mode must be auto, w4a4, native-q8, mmq, calibrated-w4a4 or dsv-native (prefill only)");
            opt.nvfp4_mode = mode == "auto" ? 0 : mode == "w4a4" ? 1 : mode == "mmq" ? 3 : mode == "calibrated-w4a4" ? 4 : mode == "dsv-native" ? 6 : 2;
            opt.nvfp4_mode_explicit = true;
        }
        else if(a=="--nvfp4-decode-mode") {
            const auto mode=next();
            if(mode!="native-q8" && mode!="calibrated-w4a4" && mode!="dsv-native")
                throw std::runtime_error("--nvfp4-decode-mode must be native-q8, calibrated-w4a4 or dsv-native");
            opt.nvfp4_decode_mode=mode=="dsv-native" ? 6 : mode=="calibrated-w4a4" ? 5 : 2;
            opt.nvfp4_decode_mode_explicit=true;
        }
        else if(a=="--reference-prefill-max-tokens") {
            const auto value=integer().get<int>();
            if(value<0||value>1024)throw std::runtime_error("Reference prefill token bound must be0..1024");
            opt.reference_prefill_max_tokens=uint32_t(value);
        }
        else if (a == "--handover-prefill") opt.handover_prefill = true;
        else if(a=="--ple-submit-low-water") {auto v=integer().get<int>();if(v<0||v>256)throw std::runtime_error("PLE submit low watermark must be 0..256");opt.ple_submit_low_water=uint32_t(v);}
        else if(a=="--ple-thread-priority") {const auto v=next();if(v!="normal"&&v!="above-normal")throw std::runtime_error("PLE thread priority must be normal or above-normal");opt.ple_thread_above_normal=v=="above-normal";}
        else if(a=="--ple-slot-threads") {auto v=integer().get<int>();if(v<1||v>32)throw std::runtime_error("PLE slot threads must be 1..32");opt.ple_slot_threads=uint32_t(v);}
        else if(a=="--ple-io-workers") {auto v=integer().get<int>();if(v<1||v>8)throw std::runtime_error("PLE IO workers must be 1..8");opt.ple_io_workers=uint32_t(v);}
        else if(a=="--ple-poll-us") {auto v=integer().get<int>();if(v<0||v>5000)throw std::runtime_error("PLE completion polling must be 0..5000 microseconds");opt.ple_poll_us=uint32_t(v);}
        else if(a=="--process-qos") {const auto v=next();if(v!="inherit"&&v!="high")throw std::runtime_error("process QoS must be inherit or high");opt.process_qos_high=v=="high";}
        else if(a=="--lock-host-resident") {auto v=integer().get<int>();if(v<0||v>1)throw std::runtime_error("host residency lock must be 0 or 1");opt.lock_host_resident=v!=0;}
        else if(a=="--trim-host-heap-cache") {const auto v=integer();if(v!=0&&v!=1)throw std::runtime_error("host heap cache trim must be 0 or 1");opt.trim_host_heap_cache=v==1;}
        else if(a=="--populate-host-heaps") {const auto v=integer();if(v!=0&&v!=1)throw std::runtime_error("host heap population must be 0 or 1");opt.populate_host_heaps=v==1;}
        else if(a=="--vision-encode")vision_encode=true;
        else if(a=="--vision-model")vision_model=next();
        else if(a=="--vision-image")vision_image=next();
        else if(a=="--vision-output")vision_output=next();
        else if(a=="--vision-backend")vision_backend=next();
        else if(a=="--vision-flash-attn") {auto v=integer().get<int>();if(v<0||v>1)throw std::runtime_error("vision flash attention must be 0 or 1");vision_flash_attention=v!=0;}
        else if(a=="--vision-min-tokens")vision_min_tokens=uint32_t(std::stoul(next()));
        else if(a=="--vision-max-tokens")vision_max_tokens=uint32_t(std::stoul(next()));
        else if(a=="--vision-threads")vision_threads=std::stoi(next());
        else if (a == "--protect-host-working-set") {
            const auto value=next();
            if(value!="0"&&value!="1")throw std::runtime_error("--protect-host-working-set needs 0 or 1");
            protect_host_working_set=value=="1";
        }
        else if (a == "--pack-check") pack_check = true;
        else if (a == "--model") opt.model = next();
        else if (a == "--qsa-attention") {
            const auto policy=next();
            if (policy!="native" && policy!="basalt") throw std::runtime_error("--qsa-attention must be native or basalt");
            opt.qsa_attention=policy=="basalt"?1:0;
        }
        else if (a == "--experts-file") opt.experts_path = next();
        else if (a == "--ple-row-dump") opt.ple_row_dump = next();
        else if (a == "--warmup-prompt") opt.warmup_prompt = next();
        else if (a == "--pair-token-file") opt.pair_token_file = next();
        else if (a == "--pair-model-sha256") opt.pair_model_sha256 = next();
        else if (a == "--pair-compatibility-sha256") opt.pair_compatibility_sha256 = next();
        else if (a == "--tokens") tokens_path = next();
        else if (a == "--out") out_prefix = next();
        else if (a == "--out-start") out_start = std::atoll(next().c_str());
        else if (a == "--top1-all") top1_all = true;
        else if (a == "--ctx") opt.n_ctx = (uint32_t) std::atoll(next().c_str());
        else if (a == "--tile") opt.tile = (uint32_t) std::atoll(next().c_str());
        else if (a == "--host-budget-gib") opt.host_budget = (uint64_t) (std::atof(next().c_str()) * 1073741824.0);
        else if (a == "--generate") n_gen = std::atoi(next().c_str());
        else if (a == "--gen-out") gen_out = next();
        else if(a=="--do-sample") {
            const auto value=integer();
            if(value!=0 && value!=1) throw std::runtime_error("--do-sample requires0 or1");
            sampling_body["do_sample"]=(value!=0);
        }
        else if (a == "--temp" || a == "--temperature") sampling_body["temperature"] = number();
        else if (a == "--top-p") sampling_body["top_p"] = number();
        else if (a == "--top-k") sampling_body["top_k"] = integer();
        else if (a == "--min-p") sampling_body["min_p"] = number();
        else if (a == "--typical-p" || a == "--typical") sampling_body["typical_p"] = number();
        else if (a == "--repeat-penalty") sampling_body["repeat_penalty"] = number();
        else if (a == "--repeat-last-n") sampling_body["repeat_last_n"] = integer();
        else if (a == "--presence-penalty") sampling_body["presence_penalty"] = number();
        else if (a == "--frequency-penalty") sampling_body["frequency_penalty"] = number();
        else if (a == "--logit-bias") sampling_body["logit_bias"] = nlohmann::ordered_json::parse(next());
        else if (a == "--stop") { const auto stop = next(); if (stop.empty()) throw std::runtime_error("empty stop"); stops.push_back(stop); }
        else if (a == "--max-tokens" || a == "--max-completion-tokens") {
            const auto n = std::stoll(next());
            if (n < -1 || n > INT32_MAX) throw std::runtime_error("invalid max_tokens");
            max_tokens = n == -1 ? INT32_MAX : (int) n; max_tokens_set = true;
        }
        else if (a == "--seed") sampling_body["seed"] = integer();
        else if (a == "--bias") sampling_body["bias"] = number();
        else if (a == "--cache-slots") opt.cache_slots = (uint32_t) std::atoll(next().c_str());
        else if (a == "--runtime-dir") {
            const auto directory = next();
            if (directory.empty()) throw std::runtime_error("--runtime-dir must be nonempty");
            std::filesystem::create_directories(directory);
            for (const char * key : {"TEMP", "TMP", "TMPDIR", "CUDA_CACHE_PATH"})
                if (_putenv_s(key, directory.c_str()) != 0) throw std::runtime_error("cannot set runtime directory");
        }
        else if (a == "--mtp") {
            const auto value = next();
            if (value != "0" && value != "1") throw std::runtime_error("--mtp must be 0 or 1");
            _putenv_s("QFN_MTP", value.c_str());
        }
        else if (a == "--validate-options") validate_options = true;
        else if (a == "--serve") serve = true;
        else if (a == "--port") serve_port = std::atoi(next().c_str());
        else if (a == "--vocab-gguf") vocab_gguf = next();
        else if (a == "--quiet") opt.verbose = false;
        else if (a == "--feed-from") feed_from = std::atoll(next().c_str());
        else if (a == "--feed-chunk") feed_chunk = std::max(1, std::atoi(next().c_str()));
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    } catch (const std::exception & e) { std::fprintf(stderr, "invalid option: %s\n", e.what()); return 2; }
    if((opt.trim_host_heap_cache||opt.populate_host_heaps)&&!serve){std::fputs("Host heap cache/population controls require --serve\n",stderr);return 2;}
    if(vision_args_seen&&!vision_encode){std::fputs("Vision worker options require --vision-encode\n",stderr);return 2;}
    if(vision_encode) {
        const std::vector<std::string> values={"--vision-model","--vision-image","--vision-output","--vision-backend",
            "--vision-min-tokens","--vision-max-tokens","--vision-threads","--vision-flash-attn","--host-budget-gib","--protect-host-working-set","--process-qos"};
        for(int i=1;i<argc;i++) {
            const std::string arg=argv[i];if(arg=="--vision-encode"||arg=="--validate-options")continue;
            if(std::find(values.begin(),values.end(),arg)==values.end()) {std::fprintf(stderr,"Unrelated option in isolated vision worker: %s\n",arg.c_str());return 2;}
            i++;
        }
    }
    if(validate_options) {
        if((vision_encode?(vision_model.empty()||vision_image.empty()||vision_output.empty()):opt.model.empty()) ||
            opt.n_ctx<1 || opt.tile<1 || opt.host_budget<1) {std::fputs("invalid model/context/tile/budget\n",stderr);return 2;}
        std::puts("qfn-engine option parser PASS; no model or listener started");return 0;
    }
    if (!tokenizer_cases.empty()) {
        if (argc != 7 || opt.model.empty() || vocab_gguf.empty()) {
            std::fprintf(stderr, "--tokenizer-check CASES accepts only --model PACK --vocab-gguf FILE\n"); return 2;
        }
        std::string error;
        qfn_pack pack;
        std::ifstream input(tokenizer_cases, std::ios::binary);
        const std::string cases((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
        if (!input || !pack.open(opt.model, error) || !qfn_tokenizer::self_test(pack, vocab_gguf, cases, error)) {
            std::fprintf(stderr, "tokenizer check failed: %s\n", error.empty() ? "cannot read cases" : error.c_str()); return 15;
        }
        return 0;
    }
    if (cache_policy_test) {
        if (argc != 2) {
            std::fprintf(stderr, "--cache-policy-self-test accepts no other options; execution/model/sampling fields are not ignored\n");
            return 2;
        }
        std::string error;
        if (!qfn_engine::cache_policy_self_test(error)) { std::fprintf(stderr, "MTP cache policy self-test failed: %s\n", error.c_str()); return 14; }
        return 0;
    }
    if (qsa_node_test) {
        if (argc != 2) {
            std::fprintf(stderr, "--qsa-node-self-test accepts no other options; execution fields are not ignored\n"); return 2;
        }
        ggml_context * ctx = ggml_init({ 256 * ggml_tensor_overhead(), nullptr, true });
        try {
            if (!ctx) throw std::runtime_error("QSA metadata context allocation failed");
            int checks = 0;
            auto expect = [&](bool pass, const char * reason) { if (!pass) throw std::runtime_error(reason); ++checks; };
            for (auto kt : { GGML_TYPE_F16, GGML_TYPE_F32 }) for (int ns : { 1, 2 }) for (int nt : { 1, 17, 2048 }) {
                auto * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, 4, nt, ns);
                auto * k = ggml_new_tensor_4d(ctx, kt, 128, 1, 100000, ns);
                auto * w = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 4, nt, 1, ns);
                auto * t = ggml_qfn_qsa_score(ctx, q, k, w, 0.08838834764831845f, -0.75f);
                expect(ggml_qfn_qsa_score_valid(t) && t->ne[0] == 100000 && t->ne[1] == nt && t->ne[3] == ns,
                       "full score shape or key type rejected");
                q->nb[1] += 16; q->nb[2] += 128; q->nb[3] += 256; w->nb[3] += 128;
                expect(ggml_qfn_qsa_score_valid(t) && t->src[0] == q && t->src[2]->nb[3] == w->nb[3],
                       "supplied outer/stream strides were lost");
                auto saved = q->nb[2]; q->nb[2] = SIZE_MAX;
                expect(nt == 1 || !ggml_qfn_qsa_score_valid(t), "overflowing input span accepted"); q->nb[2] = saved;
                t->op_params[1] = 2; expect(ggml_qfn_qsa_score_valid(t), "explicit FP4 policy rejected");
                t->op_params[1] = 3; expect(!ggml_qfn_qsa_score_valid(t), "future ABI accepted"); t->op_params[1] = 1;
                k->type = GGML_TYPE_I32; expect(!ggml_qfn_qsa_score_valid(t), "unsupported key precision accepted"); k->type = kt;
                q->ne[0] = 64; expect(!ggml_qfn_qsa_score_valid(t), "unsupported geometry accepted"); q->ne[0] = 128;
                float infinity = std::numeric_limits<float>::infinity(); std::memcpy(t->op_params + 2, &infinity, sizeof infinity);
                expect(!ggml_qfn_qsa_score_valid(t), "nonfinite actual scale accepted");
            }
            qfn_attention_node_contract(ctx,expect);
            ggml_free(ctx); std::printf("QSA custom-node contract PASS: %d checks, metadata only, no CUDA initialization\n", checks); return 0;
        } catch (const std::exception & e) {
            if (ctx) ggml_free(ctx); std::fprintf(stderr, "QSA node self-test failed: %s\n", e.what()); return 13;
        }
    }
    if (nvfp4_node_test) {
        ggml_init_params ip = {256 * ggml_tensor_overhead(), nullptr, true};
        ggml_context * ctx = ggml_init(ip);
        try {
            if (!ctx) throw std::runtime_error("node test context allocation failed");
            int checks = 0;
            auto expect = [&](bool ok, const char * reason) {
                if (!ok) throw std::runtime_error(reason);
                ++checks;
            };
            auto * work = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, 1024);
            for (int tokens : {1, 4, 1023, 1024, 16384, 100000}) {
                auto * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2560, tokens);
                auto * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 10, tokens);
                auto * bank = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, 2764816, 512);
                auto * slab = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, 2764816, 24576);
                auto * t = ggml_qfn_nvfp4_experts(ctx, x, ids, bank, work, 0);
                expect(ggml_qfn_nvfp4_valid(t) && t->src[2] == bank && t->src[3] == work,
                       "valid raw-bank node lost payload/workspace dependencies");
                auto * cached = ggml_qfn_nvfp4_experts(ctx, x, ids, slab, work, 2);
                expect(ggml_qfn_nvfp4_valid(cached), "native-q8 physical slot domain was capped at 512/128");
                cached->op_params[2] = 1;
                expect(!ggml_qfn_nvfp4_valid(cached), "W4A4 incorrectly accepted physical cache slots");
                cached->op_params[2] = 2;
                bank->ne[0] -= 16;
                expect(!ggml_qfn_nvfp4_valid(t), "missing NVFP4 scale-tail bytes were accepted");
                bank->ne[0] += 16;
                ids->type = GGML_TYPE_F32;
                expect(!ggml_qfn_nvfp4_valid(t), "noninteger expert IDs were accepted");
                ids->type = GGML_TYPE_I32;
                t->op_params[1] = 2;
                expect(!ggml_qfn_nvfp4_valid(t), "unknown node ABI was accepted");
            }
            // Explicit calibrated dependency/layout contract at both short and 16K widths.
            for (int tokens : {1, 4, 1024, 16384}) {
                auto * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2560, tokens);
                auto * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 10, tokens);
                auto * bank = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, 2764816, 512);
                auto * scales = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 512);
                auto * mmq = ggml_qfn_nvfp4_experts(ctx, x, ids, bank, work, 3);
                expect(ggml_qfn_nvfp4_valid(mmq), "explicit MMQ bank node was rejected");
                bank->ne[1] = 24576;
                expect(!ggml_qfn_nvfp4_valid(mmq), "MMQ incorrectly accepted physical decode slots");
                bank->ne[1] = 512;
                auto * node = ggml_qfn_nvfp4_experts(ctx, x, ids, bank, work, 4, scales);
                expect(ggml_qfn_nvfp4_valid(node) && node->src[4] == scales, "calibrated scalar dependency lost");
                node->src[4] = nullptr;
                expect(!ggml_qfn_nvfp4_valid(node), "calibrated mode accepted absent scalars");
                node->src[4] = scales; scales->type = GGML_TYPE_F16;
                expect(!ggml_qfn_nvfp4_valid(node), "calibration F32 precision silently reduced");
                scales->type = GGML_TYPE_F32; scales->ne[0] = 512; scales->ne[1] = 3;
                expect(!ggml_qfn_nvfp4_valid(node), "calibration role/expert dimensions transposed");
                scales->ne[0] = 3; scales->ne[1] = 512; scales->nb[1] += 4;
                expect(!ggml_qfn_nvfp4_valid(node), "strided calibration table accepted");
                scales->nb[1] -= 4; bank->ne[1] = 24576;
                expect(!ggml_qfn_nvfp4_valid(node), "calibrated logical table accepted physical slab");
                bank->ne[1] = 512; node->op_params[2] = 2;
                expect(!ggml_qfn_nvfp4_valid(node), "unused calibration pointer silently ignored");
            }
            // Mode5 preserves independent logical calibration IDs and arbitrary physical cache slots.
            for(int nt:{1,2,3,4}) {
                auto * x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,2560,nt);
                auto * physical=ggml_new_tensor_2d(ctx,GGML_TYPE_I32,10,nt);
                auto * logical=ggml_new_tensor_2d(ctx,GGML_TYPE_I32,10,nt);
                auto * slab=ggml_new_tensor_2d(ctx,GGML_TYPE_I8,2764816,24576);
                auto * aux=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,3,512);
                auto * node=ggml_qfn_nvfp4_experts(ctx,x,physical,slab,work,5,aux,logical);
                expect(ggml_qfn_nvfp4_valid(node) && node->src[1]==physical && node->src[5]==logical && node->src[4]==aux,"calibrated decode lost independent physical/logical dependencies");
                node->src[5]=nullptr; expect(!ggml_qfn_nvfp4_valid(node),"calibrated decode accepted absent logical IDs"); node->src[5]=logical;
                logical->type=GGML_TYPE_F32; expect(!ggml_qfn_nvfp4_valid(node),"calibrated decode accepted noninteger logical IDs"); logical->type=GGML_TYPE_I32;
                logical->ne[0]=9; expect(!ggml_qfn_nvfp4_valid(node),"calibrated decode accepted wrong logical topK"); logical->ne[0]=10;
                logical->ne[1]=nt+1; expect(!ggml_qfn_nvfp4_valid(node),"calibrated decode accepted mismatched logical rows"); logical->ne[1]=nt;
                logical->nb[1]+=4; expect(nt==1 || !ggml_qfn_nvfp4_valid(node),"calibrated decode accepted strided logical IDs"); logical->nb[1]-=4;
                node->src[4]=nullptr; expect(!ggml_qfn_nvfp4_valid(node),"calibrated decode accepted absent calibration"); node->src[4]=aux;
                node->op_params[2]=2; expect(!ggml_qfn_nvfp4_valid(node),"old native mode silently ignored logical/scalar fields");
                node->op_params[2]=7; expect(!ggml_qfn_nvfp4_valid(node),"unknown future arithmetic mode accepted");
            }
            for (int nt : {1,4,1696,2048}) {
                auto * x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,2560,nt);
                auto * ids=ggml_new_tensor_2d(ctx,GGML_TYPE_I32,10,nt);
                auto * bank=ggml_new_tensor_2d(ctx,GGML_TYPE_I8,2764816,512);
                auto * aux=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,3,512);
                auto * node=ggml_qfn_nvfp4_experts(ctx,x,ids,bank,work,6,aux,nullptr,true);
                expect(ggml_qfn_nvfp4_valid(node)&&node->src[4]==aux,"native packed mode lost its calibration/layout dependency");
                node->op_params[3]=0;expect(!ggml_qfn_nvfp4_valid(node),"native packed arithmetic accepted raw weight bytes");node->op_params[3]=1;
                node->src[4]=nullptr;expect(!ggml_qfn_nvfp4_valid(node),"native packed arithmetic accepted absent calibration");node->src[4]=aux;
                bank->ne[1]=24576;expect(!ggml_qfn_nvfp4_valid(node),"native grouped bank accepted arbitrary decode slots");bank->ne[1]=512;
                node->op_params[3]=2;expect(!ggml_qfn_nvfp4_valid(node),"unknown native packed layout accepted");
            }
            for (int nt : {1,2,3,4}) for (int slots : {512,520,24576}) {
                auto * x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,2560,nt);
                auto * physical=ggml_new_tensor_2d(ctx,GGML_TYPE_I32,10,nt);
                auto * logical=ggml_new_tensor_2d(ctx,GGML_TYPE_I32,10,nt);
                auto * bank=ggml_new_tensor_2d(ctx,GGML_TYPE_I8,2764816,slots);
                auto * aux=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,3,512);
                auto * node=ggml_qfn_nvfp4_experts(ctx,x,physical,bank,work,6,aux,logical,true);
                expect(ggml_qfn_nvfp4_valid(node) && node->src[5]==logical,
                       "packed decode lost arbitrary physical slots or canonical calibration ids");
                logical->ne[0]=9; expect(!ggml_qfn_nvfp4_valid(node),"packed decode accepted wrong logical topK"); logical->ne[0]=10;
                node->op_params[3]=0; expect(!ggml_qfn_nvfp4_valid(node),"packed decode accepted raw source layout");
            }
            auto * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2560, 3);
            auto * all = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 512, 3);
            auto * ids = ggml_view_2d(ctx, all, 10, 3, all->nb[1], 0);
            auto * bank = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, 2764816, 512);
            auto * t = ggml_qfn_nvfp4_experts(ctx, x, ids, bank, work, 1);
            expect(t->src[1]->op == GGML_OP_CONT && t->src[1]->src[0] == ids && ggml_qfn_nvfp4_valid(t),
                   "strided top-k router IDs were not copied without changing order");
            ggml_free(ctx);
            std::printf("NVFP4 custom-node contract PASS: %d checks, metadata only, no CUDA initialization\n", checks);
            return 0;
        } catch (const std::exception & e) {
            if (ctx) ggml_free(ctx);
            std::fprintf(stderr, "NVFP4 node self-test failed: %s\n", e.what());
            return 12;
        }
    }
    if (pack_check) {
        try {
            if (opt.model.empty() || pack_check_other_option)
                throw std::runtime_error("--pack-check accepts only --model; execution/sampling options are not ignored");
            qfn_pack pack;
            std::string error;
            if (!pack.open(opt.model, error)) throw std::runtime_error(error);
            nlohmann::ordered_json report;
            report["version"] = pack.version;
            report["expert_layout_version"] = pack.expert_layout_version;
            report["file_bytes"] = pack.file_bytes;
            report["runtime_supported"] = pack.runtime_supported(error);
            report["runtime_error"] = error;
            report["sections"] = nlohmann::ordered_json::object();
            for (const auto & s : pack.sections) report["sections"][s.first] = {s.second.first, s.second.second};
            report["generation"]={{"raw_source",pack.generation_config_json},
                {"sampling_defaults",nlohmann::ordered_json::parse(pack.generation_defaults_json)},
                {"eos_explicit",pack.generation_eos_explicit},{"eos_ids",pack.eos_tokens()},
                {"bos_explicit",pack.generation_bos_explicit},{"bos_id",pack.generation_bos_id},
                {"pad_explicit",pack.generation_pad_explicit},{"pad_id",pack.generation_pad_id},
                {"add_bos_token",pack.add_bos}};
            report["checkpoint"] = pack.checkpoint_json.empty() ? nlohmann::ordered_json(nullptr)
                : nlohmann::ordered_json::parse(pack.checkpoint_json);
            report["expert_input_scale"] = {{"offset", pack.expert_input_scale_offset},
                {"scalars", pack.expert_input_scale.size()}, {"sha256", pack.expert_input_scale_sha256},
                {"index_bytes", pack.expert_input_scale_index_json.size()}};
            if (!pack.expert_input_scale.empty()) {
                std::array<uint32_t, 3> first, last;
                std::memcpy(first.data(), pack.expert_input_scale.data(), sizeof first);
                std::memcpy(last.data(), pack.expert_input_scale.data() + pack.expert_input_scale.size() - 3, sizeof last);
                report["expert_input_scale"]["first_expert_bits"] = first;
                report["expert_input_scale"]["last_expert_bits"] = last;
            }
            report["experts"] = nlohmann::ordered_json::array();
            for (uint32_t l = 0; l < pack.experts.size(); ++l) {
                const auto & e = pack.experts[l];
                nlohmann::ordered_json item = {{"layer", l}, {"gu_type_id", e.gu_type_id}, {"d_type_id", e.d_type_id},
                    {"offset", e.offset}, {"blob", e.blob}, {"gate_bytes", e.gate_bytes}, {"down_bytes", e.down_bytes},
                    {"tail_offset", e.tail_offset}, {"tail_bytes", e.tail_bytes}};
                if (e.tail_bytes) {
                    std::array<float, 4> tail;
                    if (!pack.expert_scale_tail(l, 0, tail, error)) throw std::runtime_error(error);
                    std::array<uint32_t, 4> bits;
                    std::memcpy(bits.data(), tail.data(), sizeof(bits));
                    item["first_expert_tail_bits"] = bits; // exact payload, including the reserved fourth FP32 word
                }
                report["experts"].push_back(std::move(item));
            }
            report["mtp"] = {{"offset", pack.mtp_experts_offset}, {"blob", pack.mtp_expert_blob},
                {"gu_type_id", pack.mtp_expert_layout.gu_type_id}, {"d_type_id", pack.mtp_expert_layout.d_type_id},
                {"tail_bytes", pack.mtp_expert_layout.tail_bytes}, {"dense_tensors", pack.mtp_tensors.size()},
                {"draft_vocab", pack.draft_vocab}};
            std::puts(report.dump(2).c_str());
            return 0;
        } catch (const std::exception & e) { std::fprintf(stderr, "pack check failed: %s\n", e.what()); return 11; }
    }
    if (sampling_test) {
        try {
            auto tokenize = [](const std::string &) { return std::vector<int32_t>{1, 2}; };
            (void) qfn_parse_sampling(sampling_body.dump(), tokenize);
            const auto defaults = qfn_parse_sampling("{}", tokenize);
            if (defaults.temperature != 1 || defaults.top_p != .95f || defaults.top_k != 20 || defaults.min_p != 0 || defaults.presence_penalty != 0)
                throw std::runtime_error("pack sampling defaults mismatch");
            const auto parsed = qfn_parse_sampling(R"({"seed":0,"repeat_penalty":2,"repeat_last_n":-1,"presence_penalty":3,"frequency_penalty":4,"typical_p":0.5,"logit_bias":{"hello":2,"3":-100}})", tokenize);
            if (parsed.seed != 0 || parsed.repeat_penalty != 2 || parsed.repeat_last_n != -1 || parsed.presence_penalty != 3 || parsed.frequency_penalty != 4 || parsed.typical_p != .5f || parsed.logit_bias.at(1) != 2 || parsed.logit_bias.at(2) != 2 || parsed.logit_bias.at(3) != -100)
                throw std::runtime_error("sampler parser lost a setting");
            bool refused = false;
            try { qfn_parse_sampling(R"({"mirostat":2})", tokenize); }
            catch (const std::runtime_error &) { refused = true; }
            if (!refused) throw std::runtime_error("unsupported advanced sampler silently accepted");
            qfn_pack model; model.hp.n_vocab=248320;
            model.configure_generation(R"({"bos_token_id":248044,"do_sample":true,"eos_token_id":[248046,248044],"pad_token_id":248044,"temperature":1.0,"top_k":20,"top_p":0.95})");
            const auto cli=qfn_merge_sampling_defaults(model.generation_defaults_json,R"({"temperature":0.7,"top_k":8,"do_sample":false,"repeat_penalty":1.3})");
            const auto merged=qfn_merge_sampling_defaults(cli,R"({"top_k":3,"stop":"<|repo_name|>","repetition_penalty":1.7})");
            const auto inherited=qfn_parse_sampling(merged,tokenize);
            if(inherited.temperature!=0.7f || inherited.top_k!=3 || inherited.top_p!=0.95f || !inherited.greedy() || inherited.do_sample || model.eos_tokens()!=std::vector<int32_t>({248046,248044}))
                throw std::runtime_error("source/CLI/request precedence, do_sample or independent stop policy failed");
            if(inherited.repeat_penalty!=1.7f) throw std::runtime_error("explicit request repetition alias lost to model/CLI default");
            const auto null_legacy=qfn_parse_sampling(qfn_merge_sampling_defaults("{}",R"({"temperature":null})"),tokenize);
            if(null_legacy.temperature!=1.0f) throw std::runtime_error("legacy null numeric fallback changed");
            for(const char * bad:{R"({"do_sample":0})",R"({"do_sample":null})",R"({"do_sample":"false"})",R"({"eos_token_id":[248046]})"}) {
                bool blocked=false;try { qfn_parse_sampling(bad,tokenize); } catch(const std::runtime_error &) { blocked=true; }
                if(!blocked) throw std::runtime_error("unsupported/malformed generation request accepted");
            }
            const auto pairs = qfn_parse_sampling(R"({"logit_bias":[[0,false],["hello",-3]]})", tokenize);
            if (pairs.logit_bias.at(0) != -INFINITY || pairs.logit_bias.at(2) != -3) throw std::runtime_error("pair logit_bias mismatch");
        } catch (const std::exception & e) { std::fprintf(stderr, "sampling parser self-test failed: %s\n", e.what()); return 10; }
        std::string error;
        if (!qfn_pack::generation_self_test(error)) { std::fprintf(stderr, "generation self-test failed: %s\n", error.c_str()); return 10; }
        if (!qfn_engine::sampling_self_test(error)) { std::fprintf(stderr, "sampling self-test failed: %s\n", error.c_str()); return 10; }
        std::puts("sampling self-test PASS: normalization, sample/prob/exclusion agreement, filters, penalties, speculative history");
        return 0;
    }
    n_gen = std::min(n_gen, max_tokens);
    if(opt.process_qos_high&&!serve&&!vision_encode&&tokens_path.empty()) {
        std::fputs("Explicit process QoS requires an execution mode\n",stderr);return 2;
    }
    if(serve||vision_encode||!tokens_path.empty()) {
        PROCESS_POWER_THROTTLING_STATE qos{};qos.Version=PROCESS_POWER_THROTTLING_CURRENT_VERSION;
        if(opt.process_qos_high) {
            qos.ControlMask=PROCESS_POWER_THROTTLING_EXECUTION_SPEED;qos.StateMask=0;
            if(!SetProcessInformation(GetCurrentProcess(),ProcessPowerThrottling,&qos,sizeof(qos))) {
                std::fprintf(stderr,"Owned process HighQoS selection failed: %lu\n",GetLastError());return 2;
            }
        }
        const BOOL available=GetProcessInformation(GetCurrentProcess(),ProcessPowerThrottling,&qos,sizeof(qos));
        if(opt.process_qos_high&&(!available||!(qos.ControlMask&PROCESS_POWER_THROTTLING_EXECUTION_SPEED)||
            (qos.StateMask&PROCESS_POWER_THROTTLING_EXECUTION_SPEED))) {
            std::fputs("Owned process HighQoS readback differs\n",stderr);return 2;
        }
        std::fprintf(stderr,"QFN_PROCESS_QOS policy=%s available=%d control_mask=%lu state_mask=%lu system_settings_unchanged=1\n",
            opt.process_qos_high?"high":"inherit",int(available),qos.ControlMask,qos.StateMask);
    }
    if(protect_host_working_set) {
        // Same admitted hard minimum/advisory maximum policy as
        // slstack/working_set.py. Per-process memory policy only; the existing
        // physical-floor, RSS, commit and zero-paging guardians still apply.
        const SIZE_T minimum=SIZE_T(opt.host_budget),maximum=minimum+(64ull<<20);
        if(!SetProcessWorkingSetSizeEx(GetCurrentProcess(),minimum,maximum,
                                      QUOTA_LIMITS_HARDWS_MIN_ENABLE|QUOTA_LIMITS_HARDWS_MAX_DISABLE)) {
            std::fprintf(stderr,"QFN host working-set admission failed: Win32 %lu\n",GetLastError());return 3;
        }
        SIZE_T actual_min=0,actual_max=0;DWORD flags=0;
        if(!GetProcessWorkingSetSizeEx(GetCurrentProcess(),&actual_min,&actual_max,&flags)||
           actual_min!=minimum||actual_max<minimum||!(flags&QUOTA_LIMITS_HARDWS_MIN_ENABLE)||
           (flags&QUOTA_LIMITS_HARDWS_MAX_ENABLE)) {
            std::fputs("QFN host working-set protection was not retained\n",stderr);return 3;
        }
        std::fprintf(stderr,"QFN_HOST_RESIDENCY minimum=%llu maximum=%llu flags=%lu paging_limit_unchanged=1\n",
            (unsigned long long)actual_min,(unsigned long long)actual_max,flags);
    }
    if(vision_encode) {
        if(serve||vision_model.empty()||vision_image.empty()||vision_output.empty()) {
            std::fputs("Isolated vision encoding requires model/image/output and cannot also serve language requests\n",stderr);return 2;
        }
        qfn_vision_result result;std::string error;
        if(!qfn_vision_encode_file(vision_model,vision_image,vision_min_tokens,vision_max_tokens,
                                  vision_threads,vision_backend,vision_flash_attention,result,error)) {
            std::fprintf(stderr,"QFN vision encoding failed: %s\n",error.c_str());return 3;
        }
        FILE *output=nullptr;
        if(_wfopen_s(&output,std::filesystem::u8path(vision_output).wstring().c_str(),L"wbx")||!output) {
            std::fputs("Vision output must be a new writable file\n",stderr);return 3;
        }
        const uint32_t header[4]={0x49564c53,1,result.count,result.width};
        const bool written=fwrite(header,sizeof(header),1,output)==1&&
            fwrite(result.fingerprint.data(),1,32,output)==32&&
            fwrite(result.values.data(),sizeof(float),result.values.size(),output)==result.values.size();
        const int closed=fclose(output);if(!written||closed){std::fputs("Vision embedding output failed\n",stderr);return 3;}
        nlohmann::ordered_json row={{"event","qfn_vision_encoded"},{"count",result.count},{"width",result.width},
            {"grid_height",result.grid_height},{"grid_width",result.grid_width},{"backend",vision_backend},
            {"image_width",result.image_width},{"image_height",result.image_height},{"flash_attention",vision_flash_attention},
            {"preprocess_seconds",result.preprocess_seconds},{"load_seconds",result.load_seconds},
            {"encode_seconds",result.encode_seconds},{"unload_seconds",result.unload_seconds},
            {"free_vram_before",result.free_before},{"free_vram_after_load",result.free_after_load},
            {"free_vram_after_unload",result.free_after_unload},{"language_model_loaded",false}};
        std::printf("%s\n",row.dump().c_str());return 0;
    }
    if (serve) {
        if (opt.model.empty() || vocab_gguf.empty()) { std::fprintf(stderr, "--serve needs --model and --vocab-gguf\n"); return 2; }
        if (!stops.empty()) sampling_body["stop"] = stops;
        if (max_tokens_set) sampling_body["max_tokens"] = max_tokens;
        return qfn_serve(opt, vocab_gguf, serve_port, sampling_body.dump());
    }
    if (opt.model.empty() || tokens_path.empty() || out_prefix.empty()) {
        std::fprintf(stderr, "usage: qfn-engine --model PACK --tokens FILE.le32 --out PREFIX [--out-start N] [--top1-all] [--ctx N] [--tile N] [--qsa-attention native|basalt]\n");
        std::fprintf(stderr, "  --ple-submit-low-water 0..256: PLE refill batching (0 retains the shared policy)\n");
        std::fprintf(stderr, "  --ple-thread-priority normal|above-normal --ple-slot-threads 1..32 --ple-io-workers 1..8\n");
        return 2;
    }
    const auto toks = read_tokens(tokens_path);
    if (toks.empty()) { std::fprintf(stderr, "cannot read tokens %s\n", tokens_path.c_str()); return 2; }
    if (opt.n_ctx < toks.size() + (size_t) n_gen) opt.n_ctx = (uint32_t) (toks.size() + n_gen);
    if (out_start < 0) out_start = toks.size() > 64 ? (int64_t) toks.size() - 64 : 0;

    const auto t0 = std::chrono::steady_clock::now(); // charge authenticated default admission to load time
    qfn_tokenizer tokenizer;
    qfn_pack sampling_pack;
    bool tokenizer_ready = false;
    try {
        std::string error;
        if(!sampling_pack.open(opt.model,error)) throw std::runtime_error(error);
        const auto merged=qfn_merge_sampling_defaults(sampling_pack.generation_defaults_json,sampling_body.dump());
        smp = qfn_parse_sampling(merged, [&](const std::string & text) {
            if (!tokenizer_ready) {
                const auto path = vocab_gguf.empty() ? out_prefix + ".vocab.gguf" : vocab_gguf;
                if (!tokenizer.open(sampling_pack, path, error)) throw std::runtime_error(error);
                tokenizer_ready = true;
            }
            return tokenizer.tokenize(text, true);
        });
    } catch (const std::exception & e) { std::fprintf(stderr, "sampling: %s\n", e.what()); return 2; }

    qfn_engine eng;
    std::string err;
    const std::vector<int32_t> cold_prompt(toks.begin(),
        feed_from > 0 && feed_from < int64_t(toks.size()) ? toks.begin() + feed_from : toks.end());
    if (!eng.open(opt, err, cold_prompt)) { std::fprintf(stderr, "open failed: %s\n", err.c_str()); return 3; }
    const auto t1 = std::chrono::steady_clock::now();

    if (feed_from > 0 && feed_from < (int64_t) toks.size()) {
        // decode-path check: logits after each fed chunk -> PREFIX.feed.f32 rows, their positions -> PREFIX.feed.pos
        std::vector<int32_t> head(toks.begin(), toks.begin() + feed_from);
        qfn_logits_request r0;
        if (!eng.prefill(head, r0, err)) { std::fprintf(stderr, "prefill failed: %s\n", err.c_str()); return 4; }
        std::ofstream ff(out_prefix + ".feed.f32", std::ios::binary), fp(out_prefix + ".feed.pos", std::ios::binary);
        qfn_decode_stats st;
        std::vector<float> lg;
        for (int64_t i = feed_from; i < (int64_t) toks.size(); i += feed_chunk) {
            const int64_t m = std::min<int64_t>(feed_chunk, (int64_t) toks.size() - i);
            std::vector<int32_t> part(toks.begin() + i, toks.begin() + i + m);
            if (!eng.feed(part, lg, st, err, smp.residency_bias)) { std::fprintf(stderr, "feed failed: %s\n", err.c_str()); return 5; }
            const int32_t pos = (int32_t) (i + m - 1);
            ff.write((const char *) lg.data(), (std::streamsize) lg.size() * 4);
            fp.write((const char *) &pos, 4);
        }
        std::fprintf(stderr, "qfn-engine: fed %lld tokens in chunks of %d: %.2f s (%.2f tok/s), experts hit %llu miss %llu\n",
                     (long long) (toks.size() - feed_from), feed_chunk, st.seconds, st.tokens / std::max(1e-9, st.seconds),
                     (unsigned long long) st.expert_hits, (unsigned long long) st.expert_misses);
        std::fprintf(stderr, "qfn-engine: replay stats: attempts %llu, replays %llu, misses %llu, replays/token %.6f\n",
                     (unsigned long long) st.attempts, (unsigned long long) st.replays, (unsigned long long) st.expert_misses,
                     st.replays / std::max<double>(1, st.tokens));
        std::fprintf(stderr, "qfn-engine: decode ledger ms: graph build %.0f, input staging %.0f, GPU %.0f, routing+misses %.0f (misses %.0f ms, %.1f MiB)\n",
                     st.build_ms, st.stage_ms, st.gpu_ms, st.route_ms, st.miss_ms, st.miss_bytes / 1048576.0);
        return 0;
    }
    std::ofstream fl(out_prefix + ".f32", std::ios::binary), ft, fs;
    if (top1_all) { ft.open(out_prefix + ".top1", std::ios::binary); fs.open(out_prefix + ".lse", std::ios::binary); }
    std::vector<float> last_logits;
    qfn_logits_request req;
    req.full_from = n_gen > 0 ? std::min<int64_t>(out_start, (int64_t) toks.size() - 1) : out_start;
    req.all_top1 = top1_all;
    req.on_full = [&](int64_t pos, const float * l, int64_t V) {
        if (pos >= out_start) fl.write((const char *) l, (std::streamsize) V * 4);
        if (pos == (int64_t) toks.size() - 1) last_logits.assign(l, l + V);
    };
    req.on_top1 = [&](int64_t, int32_t best, float lse) { ft.write((const char *) &best, 4); fs.write((const char *) &lse, 4); };
    if (!eng.prefill(toks, req, err)) { std::fprintf(stderr, "prefill failed: %s\n", err.c_str()); return 4; }
    const auto t2 = std::chrono::steady_clock::now();
    if (n_gen > 0) {
        const auto & vocab = eng.vocab();
        std::ofstream go;
        if (!gen_out.empty()) go.open(gen_out, std::ios::binary);
        qfn_decode_stats st;
        eng.begin_sampling(smp);
        const int32_t first = eng.sample(last_logits.data(), smp);
        std::string text;
        std::vector<int32_t> gen;
        std::string finish = eng.is_eog(first) ? "eog" : "length";
        size_t printed = 0;
        size_t hold = 0;
        for (const auto & stop : stops) hold = std::max(hold, stop.size() - 1);
        auto print_to = [&](size_t end) {
            if (end > printed) std::fwrite(text.data() + printed, 1, end - printed, stdout);
            printed = end; std::fflush(stdout);
        };
        const bool ok = eng.is_eog(first) || eng.decode(first, n_gen, smp, [&](int32_t t, const float * logits) {
            if (!logits) { finish = "eog"; return false; }
            gen.push_back(t);
            const std::string piece = detok(vocab[(size_t) t]);
            text += piece;
            if (go) go.write((const char *) &t, 4);
            size_t at = std::string::npos;
            for (const auto & stop : stops) at = std::min(at, text.find(stop));
            if (at != std::string::npos) { text.resize(at); print_to(at); finish = "stop"; return false; }
            print_to(text.size() > hold ? text.size() - hold : 0);
            return true;
        }, st, err);
        print_to(text.size());
        std::printf("\n");
        std::ofstream generation(out_prefix + ".generation.json");
        generation << nlohmann::ordered_json({{"tokens", gen}, {"text", text},
            {"text_bytes", std::vector<uint8_t>(text.begin(), text.end())}, {"seed", smp.seed},
            {"finish_reason", finish}, {"stop", stops}, {"max_tokens", n_gen},
            {"do_sample", smp.do_sample}, {"temperature", smp.temperature}, {"top_p", smp.top_p}, {"top_k", smp.top_k},
            {"min_p", smp.min_p}, {"typical_p", smp.typical_p}, {"repeat_penalty", smp.repeat_penalty},
            {"repeat_last_n", smp.repeat_last_n}, {"presence_penalty", smp.presence_penalty},
            {"frequency_penalty", smp.frequency_penalty}, {"mtp_drafted", st.drafted},
            {"mtp_policy_batches", st.policy_batches}, {"mtp_policy_conflicts", st.policy_conflicts},
            {"mtp_policy_replays", st.policy_replays}, {"mtp_policy_restore_bytes", st.policy_restore_bytes},
            {"mtp_policy_restore_slots", st.policy_restore_slots}, {"mtp_policy_restore_ms", st.policy_restore_ms},
            {"ram_tier_hits", st.ram_tier_hits}, {"mtp_policy_restore_ram_slots", st.policy_restore_ram_slots},
            {"mtp_policy_restore_ram_bytes", st.policy_restore_ram_bytes},
            {"mtp_accepted", st.accepted}, {"decode_tokens", st.tokens}, {"seconds", st.seconds}}).dump(2, ' ', false, nlohmann::ordered_json::error_handler_t::replace) << "\n";
        if (!ok) { std::fprintf(stderr, "decode failed: %s\n", err.c_str()); return 5; }
        std::fprintf(stderr, "qfn-engine: decoded %llu tokens in %.2f s = %.2f tok/s; experts hit %llu miss %llu (%.1f%% hit), "
                     "miss reads %.1f MiB in %.0f ms\n", (unsigned long long) st.tokens, st.seconds, st.tokens / std::max(1e-9, st.seconds),
                     (unsigned long long) st.expert_hits, (unsigned long long) st.expert_misses,
                     100.0 * st.expert_hits / std::max<double>(1, st.expert_hits + st.expert_misses), st.miss_bytes / 1048576.0, st.miss_ms);
        std::fprintf(stderr, "qfn-engine: replay stats: attempts %llu, replays %llu, misses %llu, replays/token %.6f\n",
                     (unsigned long long) st.attempts, (unsigned long long) st.replays, (unsigned long long) st.expert_misses,
                     st.replays / std::max<double>(1, st.tokens));
        std::fprintf(stderr, "qfn-engine: decode ledger ms: graph build %.0f, input staging %.0f, GPU %.0f, routing+misses %.0f, MTP drafts %.0f\n",
                     st.build_ms, st.stage_ms, st.gpu_ms, st.route_ms, st.mtp_ms);
        if (st.drafted) std::fprintf(stderr, "qfn-engine: MTP drafts %llu, accepted %llu (%.1f%%)\n", (unsigned long long) st.drafted,
                                     (unsigned long long) st.accepted, 100.0 * st.accepted / (double) st.drafted);
        if (st.policy_batches) std::fprintf(stderr, "qfn-engine: MTP policy batches %llu, conflicts %llu, full-batch repairs %llu, "
            "canonical payload restore %llu slots / %llu bytes in %.3f ms\n",
            (unsigned long long) st.policy_batches, (unsigned long long) st.policy_conflicts, (unsigned long long) st.policy_replays,
            (unsigned long long) st.policy_restore_slots, (unsigned long long) st.policy_restore_bytes, st.policy_restore_ms);
        std::fprintf(stderr, "qfn-engine: RAM tier actual decode-interval hits %llu; canonical RAM payload restores %llu slots / %llu bytes "
            "(subset of total policy restores)\n", (unsigned long long) st.ram_tier_hits,
            (unsigned long long) st.policy_restore_ram_slots, (unsigned long long) st.policy_restore_ram_bytes);
    }
    const double load_s = std::chrono::duration<double>(t1 - t0).count(), run_s = std::chrono::duration<double>(t2 - t1).count();
    std::ofstream fj(out_prefix + ".json");
    fj << "{\"n_tokens\":" << toks.size() << ",\"n_vocab\":" << eng.hparams().n_vocab << ",\"out_start\":" << out_start
       << ",\"top1_all\":" << (top1_all ? "true" : "false") << ",\"tile\":" << opt.tile << ",\"load_seconds\":" << load_s
       << ",\"seconds\":" << run_s << "}\n";
    std::fprintf(stderr, "qfn-engine: done (load %.1f s, prefill+logits %.1f s)\n", load_s, run_s);
    return 0;
}
