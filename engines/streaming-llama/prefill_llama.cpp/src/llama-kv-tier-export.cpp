#include "llama-kv-tier-export.h"

#include <algorithm>
#include <cstring>

void llama_kv_tier_export::push_bytes(const void * p, size_t n) {
    if (n == 0) {
        return;
    }
    // coalesce with a preceding byte segment
    if (!segs_.empty() && segs_.back().kind == SEG_BYTES) {
        auto & s = segs_.back();
        s.bytes.insert(s.bytes.end(), (const uint8_t *) p, (const uint8_t *) p + n);
        s.size += n;
        return;
    }
    segment s;
    s.kind = SEG_BYTES;
    s.bytes.assign((const uint8_t *) p, (const uint8_t *) p + n);
    s.size = n;
    segs_.push_back(std::move(s));
}

void llama_kv_tier_export::push_meta() {
    segment s;
    s.kind = SEG_META;
    s.size = (uint64_t) (n_cells_ - base_) * 12;  // pos i32, n_seq_id u32 (=1), seq_id i32
    segs_.push_back(std::move(s));
}

void llama_kv_tier_export::push_tier(int il, int kind, uint64_t row_bytes, uint64_t n_rows) {
    if (n_rows == 0 || row_bytes == 0) {
        return;
    }
    segment s;
    s.kind = SEG_TIER;
    s.il = il;
    s.tier_kind = kind;
    s.row_bytes = row_bytes;
    s.n_rows = n_rows;
    s.size = row_bytes * n_rows;
    segs_.push_back(std::move(s));
}

void llama_kv_tier_export::build_segments() {
    segs_.clear();
    if (!header_.empty()) {
        push_bytes(header_.data(), header_.size());
    }
    // llama_kv_cache::state_write: n_stream, then per stream cell_count (+ meta + data when non-zero)
    const uint32_t n_stream = n_stream_;
    push_bytes(&n_stream, sizeof(n_stream));
    for (uint32_t s = 0; s < n_stream_; ++s) {
        const uint32_t cell_count = s == strm_ ? (n_cells_ - base_) : 0u;
        push_bytes(&cell_count, sizeof(cell_count));
        if (cell_count == 0) {
            continue;
        }
        // state_write_meta: per cell pos, n_seq_id, seq_id (no cell ext: n_pos_per_embd == 1)
        push_meta();
        // state_write_data
        const uint32_t v_trans = v_trans_;
        push_bytes(&v_trans, sizeof(v_trans));
        uint32_t n_layer_word = (uint32_t) layers_.size();
        const uint32_t state_ext_flags = has_k_idx_ ? 0x00000001u : 0u;
        if (has_k_idx_) {
            n_layer_word |= 0x80000000u;
        }
        push_bytes(&n_layer_word, sizeof(n_layer_word));
        if (state_ext_flags) {
            push_bytes(&state_ext_flags, sizeof(state_ext_flags));
        }
        for (const auto & L : layers_) {
            push_bytes(&L.k_type, sizeof(L.k_type));
            push_bytes(&L.k_size_row, sizeof(L.k_size_row));
            push_tier((int) L.il, llama_kv_tier::K, L.k_size_row, n_cells_ - base_);
        }
        if (has_k_idx_) {
            for (const auto & L : layers_) {
                const uint32_t present = L.k_idx_present ? 1u : 0u;
                push_bytes(&present, sizeof(present));
                if (!present) {
                    continue;
                }
                push_bytes(&L.k_idx_type, sizeof(L.k_idx_type));
                push_bytes(&L.k_idx_size_row, sizeof(L.k_idx_size_row));
                push_tier((int) L.il, llama_kv_tier::KIDX, L.k_idx_size_row, n_cells_ - base_);
            }
        }
        if (v_trans_ == 0) {
            for (const auto & L : layers_) {
                if (!L.v_present) {
                    continue;
                }
                push_bytes(&L.v_type, sizeof(L.v_type));
                push_bytes(&L.v_size_row, sizeof(L.v_size_row));
                push_tier((int) L.il, llama_kv_tier::V, L.v_size_row, n_cells_ - base_);
            }
        }
    }
    kv_state_bytes_ = 0;
    for (const auto & s : segs_) {
        kv_state_bytes_ += s.size;
    }
    kv_state_bytes_ -= header_.size();
    if (!tail_.empty()) {
        push_bytes(tail_.data(), tail_.size());
    }
    seg_i_ = 0;
    seg_off_ = 0;
    pulled_ = 0;
}

bool llama_kv_tier_export::plan(std::shared_ptr<llama_kv_tier> tier, uint32_t n_cells, uint32_t base_prefix, uint32_t n_stream, uint32_t strm,
                                int32_t seq_id, uint32_t v_trans, bool has_k_idx_state,
                                const std::vector<llama_kv_tier_export_layer> & layers, std::string & err) {
    if (base_prefix >= n_cells && n_cells > 0) {
        err = "base prefix " + std::to_string(base_prefix) + " leaves no cell to ship of " + std::to_string(n_cells);
        return false;
    }
    if (!tier || !tier->is_open()) {
        err = "tier not open";
        return false;
    }
    if (tier->failed()) {
        err = "tier failed earlier: " + tier->last_error();
        return false;
    }
    if (n_cells == 0) {
        err = "no cells to export";
        return false;
    }
    if (strm >= n_stream) {
        err = "stream index out of range";
        return false;
    }
    for (const auto & L : layers) {
        const int il = (int) L.il;
        if (il >= tier->n_layer()) {
            err = "layer " + std::to_string(il) + " beyond the tier's " + std::to_string(tier->n_layer()) + " layers";
            return false;
        }
        if ((uint64_t) tier->row_bytes(il, llama_kv_tier::K) != L.k_size_row) {
            err = "layer " + std::to_string(il) + " K row size " + std::to_string(tier->row_bytes(il, llama_kv_tier::K)) +
                  " in the tier vs " + std::to_string(L.k_size_row) + " in the cache";
            return false;
        }
        if (tier->rows(il, llama_kv_tier::K) < (int64_t) n_cells) {
            err = "layer " + std::to_string(il) + " holds " + std::to_string(tier->rows(il, llama_kv_tier::K)) +
                  " K rows, export needs " + std::to_string(n_cells);
            return false;
        }
        if (has_k_idx_state && L.k_idx_present) {
            if ((uint64_t) tier->row_bytes(il, llama_kv_tier::KIDX) != L.k_idx_size_row) {
                err = "layer " + std::to_string(il) + " k_idx row size mismatch";
                return false;
            }
            if (tier->rows(il, llama_kv_tier::KIDX) < (int64_t) n_cells) {
                err = "layer " + std::to_string(il) + " holds too few k_idx rows";
                return false;
            }
        }
        if (L.v_present) {
            if (v_trans != 0) {
                err = "transposed V is not streamable from cell-major row files";
                return false;
            }
            if ((uint64_t) tier->row_bytes(il, llama_kv_tier::V) != L.v_size_row) {
                err = "layer " + std::to_string(il) + " V row size mismatch";
                return false;
            }
            if (tier->rows(il, llama_kv_tier::V) < (int64_t) n_cells) {
                err = "layer " + std::to_string(il) + " holds too few V rows";
                return false;
            }
        }
    }
    tier_      = std::move(tier);
    n_cells_   = n_cells;
    base_      = base_prefix;
    n_stream_  = n_stream;
    strm_      = strm;
    seq_id_    = seq_id;
    v_trans_   = v_trans;
    has_k_idx_ = has_k_idx_state;
    layers_    = layers;
    build_segments();
    return true;
}

void llama_kv_tier_export::set_tail(std::vector<uint8_t> tail) {
    tail_ = std::move(tail);
    build_segments();
}

void llama_kv_tier_export::set_header(std::vector<uint8_t> header) {
    header_ = std::move(header);
    build_segments();
}

void llama_kv_tier_export::reset() {
    seg_i_ = 0;
    seg_off_ = 0;
    pulled_ = 0;
}

bool llama_kv_tier_export::pull(uint8_t * dst, size_t cap, size_t & n_out, std::string & err) {
    n_out = 0;
    while (n_out < cap && seg_i_ < segs_.size()) {
        segment & s = segs_[seg_i_];
        const uint64_t left_in_seg = s.size - seg_off_;
        if (left_in_seg == 0) {
            ++seg_i_;
            seg_off_ = 0;
            continue;
        }
        const size_t room = cap - n_out;
        if (s.kind == SEG_BYTES) {
            const size_t n = (size_t) std::min<uint64_t>(left_in_seg, room);
            std::memcpy(dst + n_out, s.bytes.data() + seg_off_, n);
            n_out += n;
            seg_off_ += n;
        } else if (s.kind == SEG_META) {
            // 12-byte records: pos (= cell index), n_seq_id = 1, seq_id
            const size_t n = (size_t) std::min<uint64_t>(left_in_seg, room);
            size_t done = 0;
            while (done < n) {
                const uint64_t off  = seg_off_ + done;
                const uint32_t cell = (uint32_t) (off / 12);
                const size_t   in_rec = (size_t) (off % 12);
                uint8_t rec[12];
                const int32_t  pos = (int32_t) (base_ + cell);
                const uint32_t n_seq_id = 1;
                std::memcpy(rec + 0, &pos, 4);
                std::memcpy(rec + 4, &n_seq_id, 4);
                std::memcpy(rec + 8, &seq_id_, 4);
                const size_t take = std::min<size_t>(12 - in_rec, n - done);
                std::memcpy(dst + n_out + done, rec + in_rec, take);
                done += take;
            }
            n_out += n;
            seg_off_ += n;
        } else {
            // SEG_TIER: whole rows only; seg_off_ is always on a row boundary here
            const uint64_t row0 = seg_off_ / s.row_bytes;   // within the segment
            const uint64_t rows_left = s.n_rows - row0;
            const uint64_t rows_fit  = room / s.row_bytes;
            if (rows_fit == 0) {
                if (n_out == 0) {
                    err = "chunk of " + std::to_string(cap) + " bytes is smaller than one row of " +
                          std::to_string(s.row_bytes) + " bytes";
                    return false;
                }
                break;  // return what we have; the next pull starts this row
            }
            const uint64_t n_rows = std::min(rows_left, rows_fit);
            if (!tier_->read_rows(s.il, (llama_kv_tier::kind) s.tier_kind, (int64_t) (base_ + row0), (int64_t) n_rows, dst + n_out, err)) {
                return false;
            }
            const size_t n = (size_t) (n_rows * s.row_bytes);
            n_out += n;
            seg_off_ += n;
        }
    }
    pulled_ += n_out;
    return true;
}
