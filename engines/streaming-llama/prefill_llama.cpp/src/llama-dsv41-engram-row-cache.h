#pragma once
// LLAMADECODE_ENGRAM_CACHE_MB: a host-RAM LRU of 264-byte Engram rows keyed by
// (table, row id), owned by the weight provider and kept across turns and
// requests in one engine process. The early loader copies cached rows, reads
// only the missing ones through its existing IoRing extent path, then inserts
// them. Every byte is allocated at construction (the admitted charge is the
// real footprint); nothing grows later. Not thread-safe: callers hold `mutex`.
#include <cstdint>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <vector>

class llama_dsv41_engram_row_cache {
public:
    static constexpr uint32_t row_bytes = 264;
    static constexpr uint32_t none = 0xffffffffu;
    // Per row: data, key, LRU links and up to four 4-byte index cells (the
    // power-of-two index is 2-4x the capacity, load factor <= 0.5).
    static constexpr uint64_t bytes_per_row = row_bytes + 8 + 8 + 16;
    static constexpr uint64_t fixed_bytes = 4096;
    static uint64_t capacity_rows(uint64_t bytes) {
        if (bytes <= fixed_bytes) return 0;
        const uint64_t n = (bytes - fixed_bytes) / bytes_per_row;
        return n > 0x7fffffffull ? 0x7fffffffull : n;
    }
    std::mutex mutex;

    explicit llama_dsv41_engram_row_cache(uint64_t bytes) {
        capacity = uint32_t(capacity_rows(bytes));
        if (!capacity) throw std::invalid_argument("Engram row cache below one row");
        index_mask = 1;
        while (index_mask < uint64_t(capacity) * 2) index_mask <<= 1;
        data.assign(size_t(capacity) * row_bytes, 0);
        keys.assign(capacity, 0);
        prev.assign(capacity, none);
        next.assign(capacity, none);
        index.assign(size_t(index_mask), none);
        --index_mask;
        charged = bytes;
    }
    // The admitted charge (the requested size; the footprint never exceeds it).
    uint64_t charge() const { return charged; }
    uint64_t footprint() const {
        return fixed_bytes + data.size() + keys.size() * 8 + prev.size() * 4 + next.size() * 4 + (index_mask + 1) * 4;
    }
    uint32_t rows() const { return used; }
    uint32_t capacity_row_count() const { return capacity; }
    uint64_t evictions() const { return evicted; }

    bool contains(uint32_t table, uint32_t id) const { return find(key_of(table, id)) != none; }
    // Copies the row into dst and makes it most recently used.
    bool lookup(uint32_t table, uint32_t id, uint8_t *dst) {
        const uint32_t s = find(key_of(table, id));
        if (s == none) return false;
        std::memcpy(dst, data.data() + size_t(s) * row_bytes, row_bytes);
        touch(s);
        return true;
    }
    // Rows are immutable file content: a present key is only touched.
    void insert(uint32_t table, uint32_t id, const uint8_t *src) {
        const uint64_t k = key_of(table, id);
        uint32_t s = find(k);
        if (s != none) { touch(s); return; }
        if (used < capacity) s = used++;
        else { s = tail; unlink(s); erase_index(keys[s]); ++evicted; }
        keys[s] = k;
        std::memcpy(data.data() + size_t(s) * row_bytes, src, row_bytes);
        push_front(s);
        uint64_t h = hash(k) & index_mask;
        while (index[size_t(h)] != none) h = (h + 1) & index_mask;
        index[size_t(h)] = s;
    }
    // Least to most recently used keys (tests).
    std::vector<uint64_t> order() const {
        std::vector<uint64_t> out;
        for (uint32_t s = tail; s != none; s = prev[s]) out.push_back(keys[s]);
        return out;
    }
    static uint64_t key_of(uint32_t table, uint32_t id) {
        if (table > 1) throw std::invalid_argument("Engram row cache table");
        return (uint64_t(table) << 32) | id;
    }

private:
    uint32_t capacity = 0, used = 0, head = none, tail = none;
    uint64_t index_mask = 0, charged = 0, evicted = 0;
    std::vector<uint8_t> data;
    std::vector<uint64_t> keys;
    std::vector<uint32_t> prev, next, index;

    static uint64_t hash(uint64_t x) {
        x ^= x >> 33; x *= 0xff51afd7ed558ccdull; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ull; x ^= x >> 33;
        return x;
    }
    uint32_t find(uint64_t k) const {
        for (uint64_t h = hash(k) & index_mask;; h = (h + 1) & index_mask) {
            const uint32_t s = index[size_t(h)];
            if (s == none) return none;
            if (keys[s] == k) return s;
        }
    }
    // Backward-shift deletion keeps every probe chain intact.
    void erase_index(uint64_t k) {
        uint64_t h = hash(k) & index_mask;
        while (index[size_t(h)] == none || keys[index[size_t(h)]] != k) {
            if (index[size_t(h)] == none) throw std::logic_error("Engram row cache index lost a key");
            h = (h + 1) & index_mask;
        }
        uint64_t hole = h;
        for (uint64_t j = (hole + 1) & index_mask; index[size_t(j)] != none; j = (j + 1) & index_mask) {
            const uint64_t home = hash(keys[index[size_t(j)]]) & index_mask;
            // Move j into the hole when its home is not cyclically in (hole, j].
            const bool in_range = hole <= j ? (home > hole && home <= j) : (home > hole || home <= j);
            if (!in_range) { index[size_t(hole)] = index[size_t(j)]; hole = j; }
        }
        index[size_t(hole)] = none;
    }
    void unlink(uint32_t s) {
        if (prev[s] != none) next[prev[s]] = next[s]; else head = next[s];
        if (next[s] != none) prev[next[s]] = prev[s]; else tail = prev[s];
        prev[s] = next[s] = none;
    }
    void push_front(uint32_t s) {
        prev[s] = none; next[s] = head;
        if (head != none) prev[head] = s;
        head = s;
        if (tail == none) tail = s;
    }
    void touch(uint32_t s) { if (head != s) { unlink(s); push_front(s); } }
};
