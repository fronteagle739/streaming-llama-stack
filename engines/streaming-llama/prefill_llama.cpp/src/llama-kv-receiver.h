#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct llama_kv_receiver_config {
    std::string bind_host      = "0.0.0.0";
    int32_t     port           = 9100;
    int32_t     accept_timeout_ms = 120000;  // 2 min
    int32_t     recv_timeout_ms   = 300000;  // 5 min
    bool        verify_crc     = true;
};

struct llama_kv_receiver_result {
    uint64_t    bytes_received  = 0;
    uint32_t    chunks_received = 0;
    double      receive_ms      = 0.0;
    double      throughput_gbps = 0.0;
    std::string session_info;
    bool        is_prompt_mode  = false;  // true if sender sent raw tokens (mode=prompt)
};

// Listen on bind_host:port, accept one sender connection, receive a complete
// KV artifact transferred via TBP1 framed protocol (as sent by
// send_artifact_over_tcp in llama-tb-transport.cpp).
//
// On success, artifact_out contains the full artifact file contents (header +
// payload) suitable for llama_kv_artifact_read_mem().
//
// Blocks until transfer completes, error, or timeout.
bool llama_kv_receiver_accept_artifact(
    const llama_kv_receiver_config & config,
    std::vector<uint8_t> &           artifact_out,
    llama_kv_receiver_result *       result,
    std::string *                    error);
