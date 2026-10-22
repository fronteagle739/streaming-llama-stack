#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <list>
#include <unordered_map>
#include <vector>

struct expert_lru_key {
    int layer  = -1;
    int expert = -1;

    bool operator==(const expert_lru_key & other) const {
        return layer == other.layer && expert == other.expert;
    }
};

enum class expert_lru_classification {
    hit,
    miss,
};

struct expert_lru_stats {
    uint64_t accesses  = 0;
    uint64_t hits      = 0;
    uint64_t misses    = 0;
    uint64_t evictions = 0;
};

struct expert_lru_access {
    expert_lru_classification classification = expert_lru_classification::miss;
    int                       slot           = -1;
    bool                      evicted        = false;
    expert_lru_key            evicted_key;
};

class expert_lru_cache {
public:
    static constexpr size_t ROUTED_EXPERTS = 6;
    using layer_result = std::array<expert_lru_access, ROUTED_EXPERTS>;

    explicit expert_lru_cache(size_t capacity_slabs = 0);
    expert_lru_cache(const expert_lru_cache &) = delete;
    expert_lru_cache & operator=(const expert_lru_cache &) = delete;

    void reset(size_t capacity_slabs);

    // All six routes are classified against the pre-call cache snapshot. Existing
    // hits are touched first, then misses are inserted in route order.
    layer_result lookup(int layer, const std::array<int, ROUTED_EXPERTS> & expert_ids);

    size_t capacity() const;
    size_t size() const;
    const expert_lru_stats & stats() const;

private:
    struct key_hash {
        size_t operator()(const expert_lru_key & key) const;
    };

    struct resident_entry {
        int slot;
        std::list<expert_lru_key>::iterator recency;
    };

    using resident_map = std::unordered_map<expert_lru_key, resident_entry, key_hash>;

    void touch(resident_map::iterator entry);
    void insert(const expert_lru_key & key, expert_lru_access & access);

    size_t                    capacity_ = 0;
    expert_lru_stats          stats_;
    std::list<expert_lru_key> lru_; // oldest at front, newest at back
    resident_map              resident_;
    std::vector<int>          free_slots_;
};
