// QFNSTREAM: engine serve mode. One JSON object per line on stdin, events as JSON lines on stdout. The slstack
// service owns HTTP, authentication and the GPU lease; the engine owns everything model-specific. Ops:
//   {"op":"chat","body":{OpenAI chat request}}  -> {"delta":{...}} ... {"done":true,"message":{...},"finish_reason":...}
//       rendered with the pack's own chat template and parsed (reasoning, Qwen3-Coder XML tool calls) by this tree's
//       common/chat (the llama-server path); deltas are OpenAI chat-completion chunk deltas.
//   {"op":"cancel"}                                   -> stops a running chat/generate at its next token (no reply)
//   {"op":"tokenize","text":S,"special":true}         -> {"tokens":[...]}
//   {"op":"detokenize","tokens":[...]}                -> {"text":S}
//   {"op":"prefill","tokens":[...]}                   -> {"n_past":N,"reused":R,"route":...,"seconds":S}
//       tokens is the whole conversation; when it extends the live sequence only the new suffix is processed
//       (short suffixes through the decode path, longer ones layer-major), otherwise a fresh prefill
//   {"op":"generate","max_tokens":N,"temperature":T,"top_p":P,"top_k":K,"min_p":M,"seed":S,"bias":B}
//                                                     -> {"token":id,"piece":S} ... {"done":true,"reason":...}
//   {"op":"stats"} -> engine figures; {"op":"quit"}
// Every reply carries "ok"; an error reply carries "error" and the engine stays usable when it can.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <winsock2.h>
#include <ws2tcpip.h>

#include "qfn-engine.h"
#include "qfn-tokenizer.h"

#include "chat.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <algorithm>
#include <stdexcept>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

using json = nlohmann::ordered_json;

std::string qfn_merge_sampling_defaults(const std::string & lower, const std::string & upper) {
    auto out=json::parse(lower); auto override=json::parse(upper);
    if(!out.is_object() || !override.is_object()) throw std::runtime_error("sampling defaults/overrides must be objects");
    // Preserve the parser's within-request precedence, while an explicit alias still overrides a lower default.
    for(const char * key:{"repeat_penalty","repetition_penalty","rep_pen"})
        if(override.contains(key) && !override[key].is_null()) { override["repeat_penalty"]=override[key]; break; }
    out.update(override);
    return out.dump();
}

qfn_sampling qfn_parse_sampling(const std::string & body,
    const std::function<std::vector<int32_t>(const std::string &)> & tokenize) {
    const auto b = json::parse(body);
    if(!b.is_object()) throw std::runtime_error("sampling request must be an object");
    qfn_sampling s;
    if(b.contains("do_sample")) {
        if(!b["do_sample"].is_boolean()) throw std::runtime_error("do_sample must be boolean");
        s.do_sample=b["do_sample"].get<bool>();
    }
    // Model token roles are global authenticated policy, not per-request sampler no-ops.
    for(const char * key:{"eos_token_id","bos_token_id","pad_token_id","generation_config"})
        if(b.contains(key)) throw std::runtime_error(std::string("BLOCKED-OWNER: unsupported per-request generation override ")+key);
    // These are exposed by Studio's other backends, but not its DSV4.1 combo
    // contract. Refuse an explicit request rather than accepting a silent no-op.
    for (const char * key : {"mirostat", "mirostat_tau", "mirostat_eta", "min_keep",
            "xtc_probability", "xtc_threshold", "dynatemp_range", "dynatemp_exponent",
            "top_n_sigma", "dry_multiplier", "dry_base", "dry_allowed_length",
            "dry_penalty_last_n", "dry_sequence_breakers", "adaptive_target", "adaptive_decay",
            "samplers", "presence_context_size", "frequency_context_size"}) {
        if (b.contains(key) && !b[key].is_null())
            throw std::runtime_error(std::string("BLOCKED-OWNER: QFN does not implement sampler ") + key);
    }
    auto number = [&](const char * key, double fallback, double lo, double hi, bool integer = false) {
        if (!b.contains(key) || b[key].is_null()) return fallback;
        if (!b[key].is_number()) throw std::runtime_error(std::string(key) + " must be numeric");
        const double v = b[key].get<double>();
        if (!std::isfinite(v) || v < lo || v > hi || (integer && std::floor(v) != v))
            throw std::runtime_error(std::string("invalid ") + key);
        return v;
    };
    s.temperature = (float) number("temperature", s.temperature, -std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    s.top_p = (float) number("top_p", s.top_p, 0, 1);
    s.min_p = (float) number("min_p", s.min_p, 0, 1);
    s.typical_p = (float) number("typical_p", s.typical_p, 0, 1);
    s.top_k = (int) number("top_k", s.top_k, INT32_MIN, INT32_MAX, true);
    s.repeat_penalty = (float) number("repeat_penalty", number("repetition_penalty", number("rep_pen", 1, std::numeric_limits<float>::min(), std::numeric_limits<float>::max()), std::numeric_limits<float>::min(), std::numeric_limits<float>::max()), std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
    s.repeat_last_n = (int) number("repeat_last_n", s.repeat_last_n, -1, INT32_MAX, true);
    s.presence_penalty = (float) number("presence_penalty", 0, -std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    s.frequency_penalty = (float) number("frequency_penalty", 0, -std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    s.residency_bias = (float) number("bias", 0, -std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    s.seed = (uint64_t) std::random_device{}();
    if (b.contains("seed") && !b["seed"].is_null()) {
        if (!b["seed"].is_number_integer()) throw std::runtime_error("seed must be an integer");
        if (b["seed"].is_number_unsigned() || b["seed"].get<int64_t>() >= 0) s.seed = b["seed"].get<uint64_t>();
        else if (b["seed"].get<int64_t>() != -1) throw std::runtime_error("seed must be nonnegative or -1 for automatic");
    }
    auto bias = [&](const json & key, const json & value, bool object_key = false) {
        float amount;
        if (value == false) amount = -INFINITY;
        else if (value.is_number() && std::isfinite(value.get<double>()) && std::abs(value.get<double>()) <= std::numeric_limits<float>::max()) amount = value.get<float>();
        else throw std::runtime_error("logit_bias values must be finite numbers or false");
        if (key.is_number_integer()) {
            if (key.get<int64_t>() < 0 || key.get<int64_t>() > INT32_MAX) throw std::runtime_error("logit_bias token id out of range");
            s.logit_bias[key.get<int32_t>()] = amount;
        }
        else if (key.is_string()) {
            const std::string text = key.get<std::string>();
            const bool numeric = object_key && !text.empty() && text.find_first_not_of("0123456789") == std::string::npos;
            if (numeric) s.logit_bias[(int32_t) std::stol(text)] = amount;
            else {
                const auto ids = tokenize(text);
                if (ids.empty()) throw std::runtime_error("logit_bias string has no tokens");
                for (int32_t id : ids) s.logit_bias[id] = amount;
            }
        } else throw std::runtime_error("logit_bias key must be a token id or string");
    };
    if (b.contains("logit_bias") && !b["logit_bias"].is_null()) {
        const auto & lb = b["logit_bias"];
        if (lb.is_object()) for (const auto & it : lb.items()) bias(it.key(), it.value(), true);
        else if (lb.is_array()) for (const auto & it : lb) {
            if (!it.is_array() || it.size() != 2) throw std::runtime_error("logit_bias requires [token, bias] pairs");
            bias(it[0], it[1]);
        }
        else throw std::runtime_error("logit_bias must be an object or pair array");
    }
    return s;
}

static std::vector<std::string> request_stops(const json & body) {
    std::vector<std::string> stops;
    if (!body.contains("stop") || body["stop"].is_null()) return stops;
    auto add = [&](const json & v) {
        if (!v.is_string() || v.get<std::string>().empty()) throw std::runtime_error("stop must contain nonempty strings");
        stops.push_back(v.get<std::string>());
    };
    if (body["stop"].is_array()) for (const auto & v : body["stop"]) add(v);
    else add(body["stop"]);
    return stops;
}

static int request_max_tokens(const json & body, int fallback) {
    for (const char * key : {"max_completion_tokens", "max_tokens"}) {
        if (!body.contains(key) || body[key].is_null()) continue;
        if (!body[key].is_number_integer()) throw std::runtime_error(std::string(key) + " must be an integer");
        const auto n = body[key].get<int64_t>();
        if (n < -1 || n > INT32_MAX) throw std::runtime_error(std::string("invalid ") + key);
        return n == -1 ? INT32_MAX : (int) n;
    }
    return fallback;
}

// transport: stdio, or one TCP client at a time on 127.0.0.1 (the engine stays loaded across reconnects)
namespace {
struct transport {
    SOCKET listener = INVALID_SOCKET, client = INVALID_SOCKET;
    std::string inbuf;
    std::mutex out_mu;
    bool sock = false;

    bool open_port(int port) {
        WSADATA w;
        if (WSAStartup(MAKEWORD(2, 2), &w) != 0) return false;
        listener = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (listener == INVALID_SOCKET) return false;
        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_port = htons((u_short) port);
        a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        if (bind(listener, (sockaddr *) &a, sizeof(a)) != 0 || listen(listener, 1) != 0) return false;
        sock = true;
        return true;
    }
    bool accept_client() {
        client = accept(listener, nullptr, nullptr);
        inbuf.clear();
        return client != INVALID_SOCKET;
    }
    // one line (without the newline); false when the client closed (the caller then accepts the next one)
    bool getline(std::string & line) {
        if (!sock) return (bool) std::getline(std::cin, line);
        for (;;) {
            const size_t nl = inbuf.find('\n');
            if (nl != std::string::npos) {
                line = inbuf.substr(0, nl);
                if (!line.empty() && line.back() == '\r') line.pop_back();
                inbuf.erase(0, nl + 1);
                return true;
            }
            char buf[65536];
            const int got = recv(client, buf, sizeof(buf), 0);
            if (got <= 0) { closesocket(client); client = INVALID_SOCKET; return false; }
            inbuf.append(buf, (size_t) got);
        }
    }
    void put(const std::string & s) {
        std::lock_guard<std::mutex> lk(out_mu);
        if (!sock) {
            std::fwrite(s.data(), 1, s.size(), stdout);
            std::fputc('\n', stdout);
            std::fflush(stdout);
            return;
        }
        if (client == INVALID_SOCKET) return;
        std::string m = s + "\n";
        size_t off = 0;
        while (off < m.size()) {
            const int sent = send(client, m.data() + off, (int) (m.size() - off), 0);
            if (sent <= 0) return;   // the client went away; the reader notices
            off += (size_t) sent;
        }
    }
};
transport g_io;
}  // namespace

static void emit(const json & j) {
    g_io.put(j.dump(-1, ' ', false, json::error_handler_t::replace));
}

static double secs_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

// bytes at the end of s that start an incomplete UTF-8 sequence
static size_t utf8_incomplete_tail(const std::string & s) {
    const size_t n = s.size();
    for (size_t back = 1; back <= 4 && back <= n; ++back) {
        const unsigned char c = (unsigned char) s[n - back];
        if ((c & 0xC0) == 0x80) continue;          // continuation byte: keep looking for the lead byte
        size_t need = 1;
        if ((c & 0xE0) == 0xC0) need = 2; else if ((c & 0xF0) == 0xE0) need = 3; else if ((c & 0xF8) == 0xF0) need = 4;
        return need > back ? back : 0;
    }
    return 0;
}

// length of the longest suffix of text that is a proper prefix of some stop string (held back while streaming)
static size_t stop_prefix_tail(const std::string & text, const std::vector<std::string> & stops) {
    size_t best = 0;
    for (const auto & s : stops) {
        for (size_t k = std::min(s.size() - 1, text.size()); k > best; --k) {
            if (text.compare(text.size() - k, k, s, 0, k) == 0) { best = k; break; }
        }
    }
    return best;
}

namespace {

struct server {
    qfn_engine & eng;
    qfn_tokenizer & tok;
    common_chat_templates_ptr tmpls;
    size_t append_decode_max = 32;
    std::vector<int32_t> live;          // tokens the engine state holds
    std::vector<float> last_logits;     // logits after the last live token
    qfn_decode_stats total;
    std::atomic<bool> cancel{false};
    json defaults = json::object();

    server(qfn_engine & e, qfn_tokenizer & t) : eng(e), tok(t) {}

    // make the engine state hold exactly `want`, reusing the live prefix where possible
    json bring_to(const std::vector<int32_t> & want) {
        if (want.empty()) throw std::runtime_error("empty prompt");
        size_t common = 0;
        while (common < live.size() && common < want.size() && live[common] == want[common]) ++common;
        const auto t1 = std::chrono::steady_clock::now();
        std::string route, err;
        bool ok = true;
        qfn_logits_request lr;
        lr.full_from = (int64_t) want.size() - 1;
        lr.on_full = [&](int64_t, const float * l, int64_t V) { last_logits.assign(l, l + V); };
        if (common == live.size() && common == want.size() && !last_logits.empty()) {
            route = "none";
        } else if (common == live.size() && common > 0 && want.size() - common <= append_decode_max) {
            route = "decode";
            std::vector<int32_t> add(want.begin() + (ptrdiff_t) common, want.end());
            qfn_decode_stats st;
            ok = eng.feed(add, last_logits, st, err);
        } else if (common == live.size() && common > 0) {
            route = "layer-major-append";
            std::vector<int32_t> add(want.begin() + (ptrdiff_t) common, want.end());
            ok = eng.append(add, lr, err);
        } else {
            route = "layer-major";
            common = 0;
            ok = eng.prefill(want, lr, err);
        }
        if (!ok) { live.clear(); last_logits.clear(); throw std::runtime_error(err); }
        live = want;
        return { {"n_past", eng.n_past()}, {"reused", common}, {"new_tokens", want.size() - common}, {"route", route},
                 {"seconds", secs_since(t1)} };
    }

    void add_stats(const qfn_decode_stats & st) {
        total.tokens += st.tokens; total.expert_hits += st.expert_hits; total.expert_misses += st.expert_misses;
        total.miss_bytes += st.miss_bytes; total.seconds += st.seconds; total.miss_ms += st.miss_ms;
        total.drafted += st.drafted; total.accepted += st.accepted;
        total.policy_batches += st.policy_batches; total.policy_conflicts += st.policy_conflicts; total.policy_replays += st.policy_replays;
        total.policy_restore_bytes += st.policy_restore_bytes; total.policy_restore_slots += st.policy_restore_slots;
        total.policy_restore_ms += st.policy_restore_ms;
        total.ram_tier_hits += st.ram_tier_hits; total.policy_restore_ram_slots += st.policy_restore_ram_slots;
        total.policy_restore_ram_bytes += st.policy_restore_ram_bytes;
        total.attempts += st.attempts; total.replays += st.replays;
    }

    static json stats_json(const qfn_decode_stats & st) {
        return { {"decode_tokens", st.tokens}, {"decode_seconds", st.seconds},
                 {"tok_per_s", st.tokens / std::max(1e-9, st.seconds)}, {"expert_hits", st.expert_hits},
                 {"expert_misses", st.expert_misses}, {"miss_mib", st.miss_bytes / 1048576.0}, {"miss_ms", st.miss_ms},
                 {"mtp_drafted", st.drafted}, {"mtp_accepted", st.accepted},
                 {"mtp_policy_batches", st.policy_batches}, {"mtp_policy_conflicts", st.policy_conflicts},
                 {"mtp_policy_replays", st.policy_replays}, {"mtp_policy_restore_bytes", st.policy_restore_bytes},
                 {"mtp_policy_restore_slots", st.policy_restore_slots}, {"mtp_policy_restore_ms", st.policy_restore_ms},
                 {"ram_tier_hits", st.ram_tier_hits}, {"mtp_policy_restore_ram_slots", st.policy_restore_ram_slots},
                 {"mtp_policy_restore_ram_bytes", st.policy_restore_ram_bytes},
                 {"attempts", st.attempts}, {"replays", st.replays},
                 {"ledger_ms", { {"build", st.build_ms}, {"stage", st.stage_ms}, {"gpu", st.gpu_ms}, {"route", st.route_ms}, {"mtp", st.mtp_ms} }} };
    }

    qfn_sampling sampling_from(const json & b) {
        return qfn_parse_sampling(b.dump(), [&](const std::string & text) { return tok.tokenize(text, true); });
    }

    // ------------------------------------------------------------------------------------------- chat
    void chat(json body) {
        body=json::parse(qfn_merge_sampling_defaults(defaults.dump(),body.dump()));
        const auto t0 = std::chrono::steady_clock::now();
        if (!body.contains("messages") || !body["messages"].is_array()) throw std::runtime_error("messages must be an array");
        common_chat_templates_inputs in;
        in.messages = common_chat_msgs_parse_oaicompat(body["messages"]);
        if (body.contains("tools") && !body["tools"].is_null()) in.tools = common_chat_tools_parse_oaicompat(body["tools"]);
        if (body.contains("tool_choice") && body["tool_choice"].is_string()) {
            in.tool_choice = common_chat_tool_choice_parse_oaicompat(body["tool_choice"].get<std::string>());
        } else if (body.contains("tool_choice") && body["tool_choice"].is_object()) {
            in.tool_choice = COMMON_CHAT_TOOL_CHOICE_REQUIRED;
        }
        in.parallel_tool_calls = body.value("parallel_tool_calls", false);
        in.add_generation_prompt = true;
        in.use_jinja = true;
        in.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
        if (body.contains("chat_template_kwargs") && body["chat_template_kwargs"].is_object()) {
            for (const auto & it : body["chat_template_kwargs"].items()) in.chat_template_kwargs[it.key()] = it.value().dump();
        }
        if (body.contains("reasoning_effort") && body["reasoning_effort"].is_string() && !in.chat_template_kwargs.count("reasoning_effort")) {
            in.chat_template_kwargs["reasoning_effort"] = body["reasoning_effort"].dump();
        }
        auto et = in.chat_template_kwargs.find("enable_thinking");
        if (et != in.chat_template_kwargs.end()) {
            if (et->second == "true") in.enable_thinking = true;
            else if (et->second == "false") in.enable_thinking = false;
            else throw std::runtime_error("enable_thinking must be a boolean");
        }
        const common_chat_params cp = common_chat_templates_apply(tmpls.get(), in);
        common_chat_parser_params syn(cp);
        syn.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
        const std::string open_think = "<think>\n";
        syn.thinking_forced_open = cp.thinking_forced_open ||
            (cp.prompt.size() >= open_think.size() && cp.prompt.compare(cp.prompt.size() - open_think.size(), open_think.size(), open_think) == 0);
        syn.parse_tool_calls = !in.tools.empty() && in.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE;
        if (!cp.parser.empty()) syn.parser.load(cp.parser);

        std::vector<std::string> stops = cp.additional_stops;
        const auto requested_stops = request_stops(body);
        stops.insert(stops.end(), requested_stops.begin(), requested_stops.end());
        stops.erase(std::remove_if(stops.begin(), stops.end(), [](const std::string & s) { return s.empty(); }), stops.end());
        const int max_tokens = request_max_tokens(body, 32768);
        const qfn_sampling s = sampling_from(body);
        const std::vector<int32_t> prompt = tok.tokenize(cp.prompt, true);
        const json pf = bring_to(prompt);
        emit({ {"ok", true}, {"event", "prefilled"}, {"prompt_tokens", prompt.size()}, {"prefill", pf},
               {"format", common_chat_format_name(cp.format)}, {"thinking_forced_open", syn.thinking_forced_open} });

        std::string text;
        common_chat_msg prev;
        std::vector<std::string> ids;
        std::mt19937_64 id_rng(s.seed ^ 0x9E3779B97F4A7C15ull);
        auto gen_id = [&]() {
            static const char * a = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
            std::string id = "call_";
            for (int i = 0; i < 24; ++i) id += a[id_rng() % 62];
            return id;
        };
        auto publish = [&](const common_chat_msg & msg) {
            for (const auto & d : common_chat_msg_diff::compute_diffs(prev, msg)) {
                emit({ {"ok", true}, {"delta", common_chat_msg_diff_to_json_oaicompat(d)} });
            }
            prev = msg;
        };
        auto parse_partial = [&]() {
            const std::string vis = text.substr(0, text.size() - std::max(utf8_incomplete_tail(text), stop_prefix_tail(text, stops)));
            try {
                common_chat_msg msg = common_chat_parse(vis, true, syn);
                msg.set_tool_call_ids(ids, gen_id);
                publish(msg);
            } catch (const std::exception &) {
                // an incomplete construct the parser cannot represent yet; the next token retries
            }
        };
        std::string finish = "length";
        int n_gen = 0;
        qfn_decode_stats st;
        std::string err;
        cancel = false;
        const auto td = std::chrono::steady_clock::now();
        eng.begin_sampling(s);
        const int32_t first = max_tokens > 0 ? eng.sample(last_logits.data(), s) : -1;
        bool ok = true;
        if (eng.is_eog(first)) {
            finish = "stop";
        } else if (max_tokens > 0) {
            ok = eng.decode(first, max_tokens, s, [&](int32_t t, const float * l) {
                if (!l) { finish = "stop"; return false; }          // the end-of-generation token
                live.push_back(t);
                last_logits.assign(l, l + eng.hparams().n_vocab);
                ++n_gen;
                const std::string piece = tok.piece(t);
                text += piece;
                for (const auto & stop : stops) {
                    const size_t from = text.size() > piece.size() + stop.size() ? text.size() - piece.size() - stop.size() : 0;
                    const size_t at = text.find(stop, from);
                    if (at != std::string::npos) { text.resize(at); finish = "stop"; return false; }
                }
                parse_partial();
                if (cancel) { finish = "cancelled"; return false; }
                return true;
            }, st, err);
        }
        add_stats(st);
        if (!ok) {
            if (eng.reset_required()) { live.clear(); last_logits.clear(); }
            throw std::runtime_error(err);
        }
        common_chat_msg msg;
        try {
            msg = common_chat_parse(text, false, syn);
        } catch (const std::exception &) {
            msg = common_chat_msg();
            msg.role = "assistant";
            msg.content = text;
        }
        msg.set_tool_call_ids(ids, gen_id);
        publish(msg);
        if (!msg.tool_calls.empty() && finish == "stop") finish = "tool_calls";
        const double dsec = secs_since(td);
        json timings = { {"prompt_n", pf["new_tokens"]}, {"cache_n", pf["reused"]}, {"prompt_ms", pf["seconds"].get<double>() * 1000.0},
                         {"predicted_n", n_gen}, {"predicted_ms", dsec * 1000.0},
                         {"predicted_per_second", n_gen / std::max(1e-9, dsec)} };
        emit({ {"ok", true}, {"done", true}, {"finish_reason", finish}, {"message", msg.to_json_oaicompat()},
               {"usage", { {"prompt_tokens", prompt.size()}, {"completion_tokens", n_gen}, {"total_tokens", prompt.size() + n_gen} }},
               {"timings", timings}, {"stats", stats_json(st)}, {"seconds", secs_since(t0)}, {"n_past", eng.n_past()} });
    }

    // --------------------------------------------------------------------------------------- generate
    void generate(json req) {
        req=json::parse(qfn_merge_sampling_defaults(defaults.dump(),req.dump()));
        if (last_logits.empty()) throw std::runtime_error("generate needs a prefill first");
        const qfn_sampling s = sampling_from(req);
        const int n_max = request_max_tokens(req, 256);
        const auto stops = request_stops(req);
        std::string text;
        qfn_decode_stats st;
        std::string err;
        cancel = false;
        const auto t1 = std::chrono::steady_clock::now();
        eng.begin_sampling(s);
        const int32_t first = n_max > 0 ? eng.sample(last_logits.data(), s) : -1;
        std::string reason = "length";
        bool ok = true;
        if (eng.is_eog(first)) {
            emit({ {"ok", true}, {"token", first}, {"piece", tok.piece(first)}, {"eog", true} });
            reason = "eog";
        } else if (n_max > 0) {
            ok = eng.decode(first, n_max, s, [&](int32_t t, const float * l) {
                if (!l) {
                    emit({ {"ok", true}, {"token", t}, {"piece", tok.piece(t)}, {"eog", true} });
                    reason = "eog";
                    return false;
                }
                live.push_back(t);
                last_logits.assign(l, l + eng.hparams().n_vocab);
                const auto piece = tok.piece(t);
                text += piece;
                bool stopped = false;
                for (const auto & stop : stops) if (text.find(stop) != std::string::npos) { stopped = true; break; }
                emit({ {"ok", true}, {"token", t}, {"piece", piece} });
                if (stopped) { reason = "stop"; return false; }
                if (cancel) { reason = "cancelled"; return false; }
                return true;
            }, st, err);
        }
        add_stats(st);
        if (!ok) {
            if (eng.reset_required()) { live.clear(); last_logits.clear(); }
            throw std::runtime_error(err);
        }
        json done = stats_json(st);
        done["ok"] = true; done["done"] = true; done["reason"] = reason; done["tokens"] = st.tokens;
        done["seconds"] = secs_since(t1); done["n_past"] = eng.n_past();
        emit(done);
    }
};

}  // namespace

int qfn_serve(qfn_options opt, const std::string & vocab_gguf, int port, const std::string & defaults) {
    if (port > 0) {
        if (!g_io.open_port(port)) { std::fprintf(stderr, "qfn-engine: cannot listen on 127.0.0.1:%d\n", port); return 3; }
        std::fprintf(stderr, "qfn-engine: serving on 127.0.0.1:%d\n", port);
        if (!g_io.accept_client()) return 3;   // the front end connects at once and waits for ready
    }
    qfn_engine eng;
    std::string err;
    const auto t0 = std::chrono::steady_clock::now();
    if (!eng.open(opt, err)) { emit({ {"ok", false}, {"error", "open failed: " + err} }); return 3; }
    qfn_tokenizer tok;
    std::string chat_template, bos_text, eos_text;
    json model_defaults=json::object();
    {
        // the tokenizer and template come from the same pack metadata the engine already parsed
        qfn_pack pack;
        if (!pack.open(opt.model, err) || !tok.open(pack, vocab_gguf, err)) {
            emit({ {"ok", false}, {"error", "tokenizer: " + err} });
            return 3;
        }
        chat_template = pack.chat_template;
        model_defaults=json::parse(pack.generation_defaults_json);
        const int64_t bos=pack.generation_bos_explicit ? pack.generation_bos_id : int64_t(pack.hp.tok_bos);
        if(bos>=0 && uint64_t(bos)<pack.vocab.size()) bos_text=pack.vocab[size_t(bos)];
        if (pack.hp.tok_eos < pack.vocab.size()) eos_text = pack.vocab[pack.hp.tok_eos];
    }
    if (chat_template.empty()) { emit({ {"ok", false}, {"error", "the pack has no chat template"} }); return 3; }
    server srv(eng, tok);
    srv.defaults = json::parse(qfn_merge_sampling_defaults(model_defaults.dump(),defaults));
    srv.tmpls = common_chat_templates_init(nullptr, chat_template, bos_text, eos_text);
    if (const char * am = std::getenv("QFN_APPEND_DECODE_MAX"); am && *am) srv.append_decode_max = (size_t) std::atoll(am);
    emit({ {"ok", true}, {"event", "ready"}, {"n_ctx", opt.n_ctx}, {"n_vocab", eng.hparams().n_vocab},
           {"load_seconds", secs_since(t0)} });

    // stdin reader: "cancel" acts at once (the running op checks it per token); everything else queues
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> queue;
    bool eof = false;
    std::thread reader([&]() {
        std::string line;
        for (;;) {
            if (!g_io.getline(line)) {
                if (!g_io.sock) break;
                srv.cancel = true;                       // a lost client stops any running op at its next token
                if (!g_io.accept_client()) break;
                continue;
            }
            if (line.find("\"cancel\"") != std::string::npos) {
                try { if (json::parse(line).value("op", "") == "cancel") { srv.cancel = true; continue; } } catch (...) {}
            }
            std::lock_guard<std::mutex> lk(mu);
            queue.push_back(line);
            cv.notify_one();
        }
        std::lock_guard<std::mutex> lk(mu);
        eof = true;
        cv.notify_one();
    });
    reader.detach();

    for (;;) {
        std::string line;
        {
            std::unique_lock<std::mutex> lk(mu);
            cv.wait(lk, [&] { return !queue.empty() || eof; });
            if (queue.empty()) break;
            line = std::move(queue.front());
            queue.pop_front();
        }
        if (line.empty()) continue;
        json req;
        try { req = json::parse(line); } catch (...) { emit({ {"ok", false}, {"error", "bad json"} }); continue; }
        const std::string op = req.value("op", "");
        try {
            if (op == "quit") { emit({ {"ok", true}, {"event", "bye"} }); break; }
            if (op == "chat") {
                srv.chat(req.at("body"));
            } else if (op == "tokenize") {
                emit({ {"ok", true}, {"tokens", tok.tokenize(req.at("text").get<std::string>(), req.value("special", true))} });
            } else if (op == "detokenize") {
                std::string s;
                for (int32_t t : req.at("tokens").get<std::vector<int32_t>>()) s += tok.piece(t);
                emit({ {"ok", true}, {"text", s} });
            } else if (op == "prefill") {
                json r = srv.bring_to(req.at("tokens").get<std::vector<int32_t>>());
                r["ok"] = true;
                emit(r);
            } else if (op == "generate") {
                srv.generate(req);
            } else if (op == "stats") {
                json r = server::stats_json(srv.total);
                r["ok"] = true; r["n_past"] = eng.n_past();
                emit(r);
            } else {
                emit({ {"ok", false}, {"error", "unknown op " + op} });
            }
        } catch (const std::exception & e) {
            emit({ {"ok", false}, {"error", e.what()} });
        }
    }
    return 0;
}
