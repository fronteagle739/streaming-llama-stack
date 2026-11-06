// llama-diffusion-server — a small, dedicated, PERSISTENT HTTP server for DiffusionGemma-style block
// diffusion GGUFs. Loads the model ONCE (RAM-safe: --no-mmap -dio -ngl 99), keeps it resident, and serves
// OpenAI-compatible requests by running the proven diffusion_generate_entropy_bound() per request.
// Every entropy-bound knob is overridable per request; GET /props exposes the schema for GUI rendering.
// One generation in flight at a time (single shared context) with a tiny bounded queue, else 503.

#include "arg.h"
#include "chat.h"
#include "common.h"
#include "diffusion-core.h"
#include "llama.h"
#include "log.h"

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>
#include <cpp-httplib/httplib.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <mutex>
#include <random>
#include <string>
#include <utility>
#include <vector>

using json = nlohmann::ordered_json;

static int32_t g_max_steps = 256;

// ------------------------------------------------------------------ helpers
static json format_error(const std::string & msg, int code, const char * type = "invalid_request_error") {
    return json{ { "error", { { "message", msg }, { "type", type }, { "code", code } } } };
}

static std::string gen_id(const char * prefix) {
    static const char alnum[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    static std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> d(0, (int) sizeof(alnum) - 2);
    std::string s = prefix;
    for (int i = 0; i < 32; i++) s += alnum[d(rng)];
    return s;
}

// shared concurrency gate: one gen in flight; up to max_queue waiters block, the rest get 503.
struct gen_gate {
    std::mutex run_m;            // serializes the actual gen
    std::mutex count_m;
    int  inflight  = 0;
    int  waiting   = 0;
    int  max_queue = 1;          // waiters allowed; 0 => 503 immediately when busy
    bool try_enter() {           // false => caller should 503
        std::lock_guard<std::mutex> lk(count_m);
        if (inflight + waiting > max_queue) return false;
        waiting++;
        return true;
    }
    void begin() { std::lock_guard<std::mutex> lk(count_m); waiting--; inflight++; }
    void end()   { std::lock_guard<std::mutex> lk(count_m); inflight--; }
};

struct server_state {
    llama_model *       model         = nullptr;
    llama_context *     ctx           = nullptr;
    const llama_vocab * vocab         = nullptr;
    int32_t             canvas_length = 0;
    int32_t             n_ubatch      = 0;
    bool                sc_available  = false;
    bool                clamp_prompt  = false;
    std::string         alias;
    gen_gate            gate;
};

// the canonical knob schema -> /props + the request clamps (one source of truth)
static json knob_schema_json() {
    diffusion_eb_params d{};   // struct defaults = proven values
    return json::array({
        json{ {"name","steps"},                {"type","int"},   {"default",d.max_denoising_steps}, {"min",1}, {"max",g_max_steps},
              {"desc","max denoising passes (adaptive stop may finish earlier)"} },
        json{ {"name","t_min"},                {"type","float"}, {"default",d.t_min}, {"min",0.001}, {"max",4.0},
              {"desc","temperature at the LAST step (sharper)"} },
        json{ {"name","t_max"},                {"type","float"}, {"default",d.t_max}, {"min",0.001}, {"max",4.0},
              {"desc","temperature at the FIRST step (more exploration)"} },
        json{ {"name","entropy_bound"},        {"type","float"}, {"default",d.entropy_bound}, {"min",0.0},
              {"desc","accept lowest-entropy positions within this mutual-information bound"} },
        json{ {"name","stability_threshold"},  {"type","int"},   {"default",d.stability_threshold}, {"min",0},
              {"desc","steps the argmax canvas must hold to count as stable (adaptive stop)"} },
        json{ {"name","confidence_threshold"}, {"type","float"}, {"default",d.confidence_threshold}, {"min",0.0},
              {"desc","stop once mean canvas entropy drops below this"} },
        json{ {"name","seed"},                 {"type","int"},   {"default",d.seed},
              {"desc","-1 = random per request"} },
        json{ {"name","self_conditioning"},    {"type","bool"},  {"default",true},
              {"desc","feed prev-step canvas logits forward (DiffusionGemma); off-only if ctx built without SC"} },
        json{ {"name","temperature"},          {"type","float"}, {"default",0.0},
              {"desc","convenience alias: sets a FLAT t_min=t_max schedule when t_min/t_max omitted"} },
    });
}

static json props_json(const server_state & st) {
    diffusion_eb_params d{};
    return json{
        { "model",                        st.alias },
        { "canvas_length",                st.canvas_length },
        { "n_ubatch",                     st.n_ubatch },
        { "max_prompt_tokens",            st.n_ubatch - st.canvas_length },
        { "self_conditioning_available",  st.sc_available },
        { "default_generation_settings", json{
            { "steps", d.max_denoising_steps }, { "t_min", d.t_min }, { "t_max", d.t_max },
            { "entropy_bound", d.entropy_bound }, { "stability_threshold", d.stability_threshold },
            { "confidence_threshold", d.confidence_threshold }, { "seed", d.seed },
            { "self_conditioning", st.sc_available }, { "temperature", 0.0 } } },
        { "knobs", knob_schema_json() },
    };
}

static std::string apply_chat_template(llama_model * model, const json & messages) {
    auto tmpls = common_chat_templates_init(model, "");
    common_chat_templates_inputs in;
    for (const auto & m : messages) {
        common_chat_msg msg;
        msg.role    = m.at("role").get<std::string>();
        msg.content = m.value("content", std::string());
        in.messages.push_back(std::move(msg));
    }
    in.add_generation_prompt = true;
    return common_chat_templates_apply(tmpls.get(), in).prompt;
}

// build the eb_params + input tokens from a request body. false + fills `err` on a 4xx problem.
static bool build_request(server_state & st, const json & body, bool is_chat,
                          std::vector<llama_token> & input_tokens, diffusion_eb_params & eb, json & err) {
    std::string formatted;
    if (is_chat) {
        if (!body.contains("messages") || !body.at("messages").is_array()) {
            err = format_error("missing 'messages' array", 400); return false;
        }
        formatted = apply_chat_template(st.model, body.at("messages"));
    } else {
        formatted = body.value("prompt", std::string());
    }
    input_tokens = common_tokenize(st.vocab, formatted, /*add_special*/ true, /*parse_special*/ true);
    int32_t n_input = (int32_t) input_tokens.size();
    if (n_input <= 0) { err = format_error("empty prompt", 400); return false; }

    const int32_t max_prompt = st.n_ubatch - st.canvas_length;
    if (n_input > max_prompt) {
        if (st.clamp_prompt) {
            input_tokens.erase(input_tokens.begin(), input_tokens.end() - max_prompt); // keep the most recent
            n_input = (int32_t) input_tokens.size();
        } else {
            err = format_error(string_format("prompt too long: %d tokens > max_prompt_tokens %d "
                                             "(raise -ub or shorten the prompt)", n_input, max_prompt), 400);
            return false;
        }
    }

    eb = diffusion_eb_params{};                       // proven defaults
    eb.max_length        = n_input + st.canvas_length; // ALWAYS server-computed
    eb.self_conditioning = st.sc_available;

    eb.max_denoising_steps  = std::clamp(body.value("steps", eb.max_denoising_steps), 1, g_max_steps);
    eb.entropy_bound        = std::max(0.0f, body.value("entropy_bound", eb.entropy_bound));
    eb.stability_threshold  = std::max(0,    body.value("stability_threshold", eb.stability_threshold));
    eb.confidence_threshold = std::max(0.0f, body.value("confidence_threshold", eb.confidence_threshold));

    const bool has_tmin = body.contains("t_min"), has_tmax = body.contains("t_max");
    eb.t_min = body.value("t_min", eb.t_min);
    eb.t_max = body.value("t_max", eb.t_max);
    if (body.contains("temperature") && !has_tmin && !has_tmax) {   // temperature alias = flat schedule
        float tp = body.value("temperature", eb.t_max);
        eb.t_min = eb.t_max = std::max(1e-3f, tp);
    }
    if (eb.t_min > eb.t_max) std::swap(eb.t_min, eb.t_max);
    eb.t_min = std::max(1e-3f, eb.t_min);
    eb.t_max = std::max(eb.t_min, eb.t_max);

    int seed_in = body.value("seed", 0);
    eb.seed = (seed_in < 0) ? (int32_t) std::random_device{}() : seed_in;

    if (body.contains("self_conditioning")) {
        eb.self_conditioning = st.sc_available && body.value("self_conditioning", st.sc_available);
    }
    eb.step_callback = nullptr;
    return true;
}

static json effective_knobs(const diffusion_eb_params & eb) {
    return json{
        { "steps", eb.max_denoising_steps }, { "t_min", eb.t_min }, { "t_max", eb.t_max },
        { "entropy_bound", eb.entropy_bound }, { "stability_threshold", eb.stability_threshold },
        { "confidence_threshold", eb.confidence_threshold }, { "seed", eb.seed },
        { "self_conditioning", eb.self_conditioning },
    };
}

static void setup_cors(httplib::Server & s) {
    s.set_default_headers({ { "Server", "llama-diffusion" } });
    s.set_pre_routing_handler([](const httplib::Request & req, httplib::Response & res) {
        res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));
        if (req.method == "OPTIONS") {
            res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
            res.set_header("Access-Control-Allow-Headers", "*");
            res.set_content("", "text/plain");
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });
}

static void handle_gen(server_state & st, const httplib::Request & rq, httplib::Response & rs, bool is_chat) {
    json body;
    try { body = json::parse(rq.body.empty() ? "{}" : rq.body); }
    catch (...) { rs.status = 400; rs.set_content(format_error("invalid JSON", 400).dump(), "application/json"); return; }

    // The diffusion gen is non-incremental (the whole canvas materializes), but the Studio chat path
    // expects an OpenAI SSE stream — so on stream:true we run the full gen, then emit it as a SINGLE SSE
    // chunk + [DONE] (drop-in for any streaming OpenAI client). Non-stream returns the usual JSON.
    const bool stream = body.value("stream", false);

    std::vector<llama_token> input; diffusion_eb_params eb; json err;
    if (!build_request(st, body, is_chat, input, eb, err)) {
        rs.status = err["error"]["code"].get<int>();
        rs.set_content(err.dump(), "application/json");
        return;
    }

    if (!st.gate.try_enter()) {
        rs.status = 503; rs.set_header("Retry-After", "1");
        rs.set_content(format_error("diffusion busy", 503, "unavailable_error").dump(), "application/json");
        return;
    }

    std::string text; int32_t n_gen = 0; const int32_t n_input = (int32_t) input.size();
    {
        std::lock_guard<std::mutex> run(st.gate.run_m);  // single in-flight
        st.gate.begin();
        std::vector<llama_token> out(eb.max_length, 0);
        diffusion_generate_entropy_bound(st.ctx, input.data(), out.data(), n_input, eb, n_gen);
        st.gate.end();
        if (n_gen > 0) {
            out.erase(out.begin(), out.begin() + n_input);
            text = common_detokenize(st.vocab, out, /*special*/ false);
        }
    }
    if (n_gen <= 0) {
        rs.status = 500; rs.set_content(format_error("diffusion generation failed", 500, "server_error").dump(), "application/json");
        return;
    }

    // streaming chat: emit the whole result as ONE OpenAI SSE chunk + [DONE] (the gen is non-incremental,
    // but this is a drop-in for the streaming chat path so the daemon needs no diffusion-specific handling).
    if (stream && is_chat) {
        const std::string id = gen_id("chatcmpl-");
        const int64_t created = (int64_t) time(nullptr);
        json delta = json{ { "id", id }, { "object", "chat.completion.chunk" }, { "created", created },
            { "model", st.alias }, { "choices", json::array({ json{
                { "index", 0 }, { "delta", json{ { "role", "assistant" }, { "content", text } } },
                { "finish_reason", nullptr } } }) } };
        json done = json{ { "id", id }, { "object", "chat.completion.chunk" }, { "created", created },
            { "model", st.alias }, { "choices", json::array({ json{
                { "index", 0 }, { "delta", json::object() }, { "finish_reason", "stop" } } }) } };
        std::string sse = "data: " + delta.dump() + "\n\ndata: " + done.dump() + "\n\ndata: [DONE]\n\n";
        rs.set_header("Cache-Control", "no-cache");
        rs.set_content(sse, "text/event-stream");
        return;
    }

    const int32_t completion_tokens = st.canvas_length;
    json out_json;
    if (is_chat) {
        out_json = json{
            { "id",      gen_id("chatcmpl-") },
            { "object",  "chat.completion" },
            { "created", (int64_t) time(nullptr) },
            { "model",   st.alias },
            { "choices", json::array({ json{
                { "index", 0 },
                { "message", json{ { "role", "assistant" }, { "content", text } } },
                { "finish_reason", "stop" } } }) },
            { "usage", json{
                { "prompt_tokens", n_input }, { "completion_tokens", completion_tokens },
                { "total_tokens", n_input + completion_tokens } } },
            { "diffusion", effective_knobs(eb) },
        };
    } else {
        out_json = json{
            { "content", text },
            { "tokens_predicted", completion_tokens },
            { "tokens_evaluated", n_input },
            { "seed", eb.seed },
            { "stop", true },
            { "diffusion", effective_knobs(eb) },
        };
    }
    rs.set_content(out_json.dump(), "application/json");
}

int main(int argc, char ** argv) {
    ggml_time_init();
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_DIFFUSION)) {
        return 1;
    }
    common_init();
    llama_backend_init();

    // ---- load model RAM-safe (matches diffusion-cli.cpp) ----
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers  = params.n_gpu_layers;
    mp.devices       = params.devices.data();
    mp.use_mmap      = params.use_mmap;        // pass --no-mmap
    mp.use_direct_io = params.use_direct_io;   // pass -dio
    mp.use_mlock     = params.use_mlock;
    mp.check_tensors = params.check_tensors;

    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mp);
    if (!model) {
        LOG_ERR("error: failed to load model '%s'\n", params.model.path.c_str());
        return 1;
    }
    if (!llama_model_is_diffusion(model)) {
        LOG_ERR("error: '%s' is not a diffusion model\n", params.model.path.c_str());
        llama_model_free(model);
        return 1;
    }

    // ---- canvas_length + SC reserve BEFORE context (matches diffusion-cli.cpp) ----
    char buf[32]; int32_t canvas_length = 0;
    if (llama_model_meta_val_str(model, "diffusion.canvas_length", buf, sizeof(buf)) >= 0) {
        canvas_length = atoi(buf);
    }
    if (canvas_length <= 0) {
        LOG_ERR("error: model has no positive diffusion.canvas_length (not a canvas/DiffusionGemma model)\n");
        llama_model_free(model);
        return 1;
    }
    const bool dg_use_sc = (getenv("LLAMA_DG_NO_SC") == nullptr);
    if (dg_use_sc) {
        llama_diffusion_set_sc(model, nullptr, 0.0f, 1.0f, true);   // reserve the SC subgraph
    }

    // ---- context ----
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = params.n_ctx;
    cp.n_batch         = params.n_batch;
    cp.n_ubatch        = params.n_ubatch;
    cp.flash_attn_type = params.flash_attn_type;
    cp.no_perf         = params.no_perf;
    cp.type_k          = params.cache_type_k;
    cp.type_v          = params.cache_type_v;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        LOG_ERR("error: failed to create context\n");
        llama_model_free(model);
        return 1;
    }
    llama_set_n_threads(ctx, params.cpuparams.n_threads, params.cpuparams_batch.n_threads);

    if (params.n_ubatch <= canvas_length) {
        LOG_ERR("error: -ub (%d) must be > canvas_length (%d); raise -ub to leave room for prompts\n",
                params.n_ubatch, canvas_length);
        llama_free(ctx); llama_model_free(model);
        return 1;
    }

    g_max_steps = params.diffusion.server_max_steps;

    server_state st;
    st.model         = model;
    st.ctx           = ctx;
    st.vocab         = llama_model_get_vocab(model);
    st.canvas_length = canvas_length;
    st.n_ubatch      = params.n_ubatch;
    st.sc_available  = dg_use_sc;
    st.clamp_prompt  = params.diffusion.server_clamp_prompt;
    st.gate.max_queue = params.diffusion.server_max_queue;
    {
        // model name = --alias or the GGUF basename
        std::string p = params.model.path;
        size_t slash = p.find_last_of("/\\");
        st.alias = params.model_alias.empty() ? (slash == std::string::npos ? p : p.substr(slash + 1))
                                              : params.model_alias;
    }

    httplib::Server srv;
    setup_cors(srv);
    srv.set_exception_handler([](const httplib::Request &, httplib::Response & res, std::exception_ptr ep) {
        std::string msg = "internal error";
        try { std::rethrow_exception(ep); } catch (std::exception & e) { msg = e.what(); } catch (...) {}
        res.status = 500;
        res.set_content(format_error(msg, 500, "server_error").dump(), "application/json");
    });
    srv.set_error_handler([](const httplib::Request &, httplib::Response & res) {
        if (res.status >= 400 && res.body.empty()) {
            res.set_content(format_error("not found", res.status).dump(), "application/json");
        }
    });

    const std::string & pfx = params.api_prefix;
    srv.Post(pfx + "/v1/chat/completions", [&](const httplib::Request & rq, httplib::Response & rs) { handle_gen(st, rq, rs, true); });
    srv.Post(pfx + "/completion",          [&](const httplib::Request & rq, httplib::Response & rs) { handle_gen(st, rq, rs, false); });
    srv.Get (pfx + "/props",  [&](const httplib::Request &, httplib::Response & rs) { rs.set_content(props_json(st).dump(), "application/json"); });
    srv.Get (pfx + "/health", [&](const httplib::Request &, httplib::Response & rs) { rs.set_content(R"({"status":"ok"})", "application/json"); });
    srv.Get (pfx + "/v1/models", [&](const httplib::Request &, httplib::Response & rs) {
        rs.set_content(json{ { "object", "list" }, { "data", json::array({ json{ { "id", st.alias }, { "object", "model" } } }) } }.dump(), "application/json");
    });

    LOG_INF("llama-diffusion-server: model '%s' ready (canvas=%d, sc=%s); listening on http://%s:%d\n",
            st.alias.c_str(), canvas_length, dg_use_sc ? "on" : "off", params.hostname.c_str(), params.port);
    if (!srv.listen(params.hostname.c_str(), params.port)) {
        LOG_ERR("error: failed to bind %s:%d\n", params.hostname.c_str(), params.port);
        llama_free(ctx); llama_model_free(model); llama_backend_free();
        return 1;
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
