#include "tail_tokens.h"

#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static bool read_file_bytes(const std::string & path, std::vector<uint8_t> & bytes_out, std::string & error) {
    FILE * fp = std::fopen(path.c_str(), "rb");
    if (!fp) {
        error = "failed to open sidecar";
        return false;
    }

    if (std::fseek(fp, 0, SEEK_END) != 0) {
        std::fclose(fp);
        error = "failed to seek sidecar";
        return false;
    }

    const long size = std::ftell(fp);
    if (size < 0) {
        std::fclose(fp);
        error = "failed to stat sidecar";
        return false;
    }
    if (std::fseek(fp, 0, SEEK_SET) != 0) {
        std::fclose(fp);
        error = "failed to rewind sidecar";
        return false;
    }

    bytes_out.resize(static_cast<size_t>(size));
    if (!bytes_out.empty()) {
        const size_t nread = std::fread(bytes_out.data(), 1, bytes_out.size(), fp);
        if (nread != bytes_out.size()) {
            std::fclose(fp);
            error = "failed to read sidecar";
            return false;
        }
    }

    std::fclose(fp);
    return true;
}

static size_t skip_utf8_bom(const std::vector<uint8_t> & bytes) {
    return bytes.size() >= 3 && bytes[0] == 0xEF && bytes[1] == 0xBB && bytes[2] == 0xBF ? 3 : 0;
}

static size_t skip_json_ws(const std::string & text, size_t pos) {
    while (pos < text.size()) {
        const char ch = text[pos];
        if (ch != ' ' && ch != '\t' && ch != '\r' && ch != '\n') {
            break;
        }
        ++pos;
    }
    return pos;
}

static bool parse_tail_tokens_json(const std::string & text, std::vector<llama_token> & tokens_out, std::string & error) {
    size_t pos = skip_json_ws(text, 0);
    if (pos == text.size()) {
        return true;
    }
    if (text[pos] != '[') {
        error = "JSON sidecar must be an array of token IDs";
        return false;
    }
    pos = skip_json_ws(text, pos + 1);
    if (pos < text.size() && text[pos] == ']') {
        pos = skip_json_ws(text, pos + 1);
        if (pos != text.size()) {
            error = "unexpected trailing content after JSON token array";
            return false;
        }
        return true;
    }

    while (pos < text.size()) {
        char * end = nullptr;
        errno = 0;
        const long long value = std::strtoll(text.c_str() + pos, &end, 10);
        if (end == text.c_str() + pos) {
            error = "expected integer token ID in JSON sidecar";
            return false;
        }
        if (errno == ERANGE || value < INT32_MIN || value > INT32_MAX) {
            error = "token ID out of int32 range in JSON sidecar";
            return false;
        }

        tokens_out.push_back(static_cast<llama_token>(value));
        pos = skip_json_ws(text, static_cast<size_t>(end - text.c_str()));
        if (pos >= text.size()) {
            error = "unterminated JSON token array";
            return false;
        }
        if (text[pos] == ',') {
            pos = skip_json_ws(text, pos + 1);
            continue;
        }
        if (text[pos] == ']') {
            pos = skip_json_ws(text, pos + 1);
            if (pos != text.size()) {
                error = "unexpected trailing content after JSON token array";
                return false;
            }
            return true;
        }

        error = "expected ',' or ']' in JSON token array";
        return false;
    }

    error = "unterminated JSON token array";
    return false;
}

static bool parse_tail_tokens_binary(const std::vector<uint8_t> & bytes, std::vector<llama_token> & tokens_out, std::string & error) {
    if (bytes.size() % sizeof(int32_t) != 0) {
        error = "binary sidecar size must be a multiple of 4 bytes";
        return false;
    }

    tokens_out.reserve(bytes.size() / sizeof(int32_t));
    for (size_t i = 0; i < bytes.size(); i += sizeof(int32_t)) {
        const uint32_t value =
            static_cast<uint32_t>(bytes[i + 0]) |
            (static_cast<uint32_t>(bytes[i + 1]) << 8) |
            (static_cast<uint32_t>(bytes[i + 2]) << 16) |
            (static_cast<uint32_t>(bytes[i + 3]) << 24);
        tokens_out.push_back(static_cast<llama_token>(static_cast<int32_t>(value)));
    }

    return true;
}

bool load_tail_tokens_file(const std::string & path, std::vector<llama_token> & tokens_out, std::string & error) {
    tokens_out.clear();

    std::vector<uint8_t> bytes;
    if (!read_file_bytes(path, bytes, error)) {
        return false;
    }
    if (bytes.empty()) {
        return true;
    }

    const size_t bom = skip_utf8_bom(bytes);
    size_t first = bom;
    while (first < bytes.size()) {
        const char ch = static_cast<char>(bytes[first]);
        if (ch != ' ' && ch != '\t' && ch != '\r' && ch != '\n') {
            break;
        }
        ++first;
    }

    if (first < bytes.size() && bytes[first] == '[') {
        return parse_tail_tokens_json(std::string(reinterpret_cast<const char *>(bytes.data() + bom), bytes.size() - bom), tokens_out, error);
    }

    return parse_tail_tokens_binary(bytes, tokens_out, error);
}
