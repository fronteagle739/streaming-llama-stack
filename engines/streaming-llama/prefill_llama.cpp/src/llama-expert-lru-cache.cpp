#include "llama-expert-lru-cache.h"

#include <functional>
#include <iterator>
#include <limits>
#include <stdexcept>

expert_lru_cache::expert_lru_cache(size_t capacity_slabs) {
    reset(capacity_slabs);
}

void expert_lru_cache::reset(size_t capacity_slabs) {
    if (capacity_slabs > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("expert LRU capacity exceeds abstract int slot range");
    }

    capacity_ = capacity_slabs;
    stats_ = {};
    lru_.clear();
    resident_.clear();
    resident_.reserve(capacity_slabs);

    free_slots_.clear();
    free_slots_.reserve(capacity_slabs);
    for (size_t slot = capacity_slabs; slot > 0; --slot) {
        free_slots_.push_back(static_cast<int>(slot - 1));
    }
}

expert_lru_cache::layer_result expert_lru_cache::lookup(
        int layer, const std::array<int, ROUTED_EXPERTS> & expert_ids) {
    layer_result result;
    std::array<expert_lru_key, ROUTED_EXPERTS> keys;

    // Match sim_lru.py P1: classify the complete layer against one snapshot.
    for (size_t i = 0; i < ROUTED_EXPERTS; ++i) {
        keys[i] = { layer, expert_ids[i] };
        const auto found = resident_.find(keys[i]);
        if (found != resident_.end()) {
            result[i].classification = expert_lru_classification::hit;
            result[i].slot = found->second.slot;
        }
    }

    stats_.accesses += ROUTED_EXPERTS;

    // P1 applies all hits before any insertion. Repeated hits are touched in
    // route order, exactly like OrderedDict.move_to_end().
    for (size_t i = 0; i < ROUTED_EXPERTS; ++i) {
        if (result[i].classification != expert_lru_classification::hit) {
            continue;
        }
        ++stats_.hits;
        touch(resident_.find(keys[i]));
    }

    // Every snapshot miss counts as a miss, including duplicate keys. The
    // insertion helper itself is idempotent and only a new resident can evict.
    for (size_t i = 0; i < ROUTED_EXPERTS; ++i) {
        if (result[i].classification != expert_lru_classification::miss) {
            continue;
        }
        ++stats_.misses;
        insert(keys[i], result[i]);
    }

    return result;
}

size_t expert_lru_cache::capacity() const {
    return capacity_;
}

size_t expert_lru_cache::size() const {
    return resident_.size();
}

const expert_lru_stats & expert_lru_cache::stats() const {
    return stats_;
}

size_t expert_lru_cache::key_hash::operator()(const expert_lru_key & key) const {
    const size_t layer_hash = std::hash<int>{}(key.layer);
    const size_t expert_hash = std::hash<int>{}(key.expert);
    return layer_hash ^ (expert_hash + size_t{0x9e3779b9} + (layer_hash << 6) + (layer_hash >> 2));
}

void expert_lru_cache::touch(resident_map::iterator entry) {
    lru_.splice(lru_.end(), lru_, entry->second.recency);
}

void expert_lru_cache::insert(const expert_lru_key & key, expert_lru_access & access) {
    if (capacity_ == 0) {
        return;
    }

    auto existing = resident_.find(key);
    if (existing != resident_.end()) {
        touch(existing);
        access.slot = existing->second.slot;
        return;
    }

    int slot = -1;
    if (resident_.size() == capacity_) {
        const expert_lru_key evicted_key = lru_.front();
        const auto evicted = resident_.find(evicted_key);
        slot = evicted->second.slot;
        resident_.erase(evicted);
        lru_.pop_front();

        ++stats_.evictions;
        access.evicted = true;
        access.evicted_key = evicted_key;
    } else {
        slot = free_slots_.back();
        free_slots_.pop_back();
    }

    lru_.push_back(key);
    resident_.emplace(key, resident_entry{ slot, std::prev(lru_.end()) });
    access.slot = slot;
}
