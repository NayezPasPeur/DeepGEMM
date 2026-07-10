#pragma once

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/utils.cuh>

// SM100 prefill-paged scheduler: round-robin Q-block assignment (no metadata kernel)
// with paged KV loading via 3D TMA and per-token cu_seq_len_k masking.
// Supports multi-sequence batched prefill via seq_indices mapping.

namespace deep_gemm::sched {

template <uint32_t BLOCK_Q, uint32_t SPLIT_KV, uint32_t PAGE_KV, uint32_t kNumSMs,
          bool kForceContiguous = false, bool kSeparatedStorage = false>
struct SM100PrefillPagedMQALogitsScheduler {
    // kForceContiguous: treat entire KV as contiguous (single 2D TMA per split, no page lookup)
    // kSeparatedStorage: KV data and scales stored separately; use 2D TMA per-page with row offset
    //                    (pages can be non-contiguous via block_table)
    static constexpr bool kIsPaged = !kForceContiguous;
    static constexpr bool kHasPartialBlock = false;
    static constexpr uint32_t kPageKV = (kForceContiguous && !kSeparatedStorage) ? 0 : PAGE_KV;
    static constexpr uint32_t kNumPagesPerSplit = (kForceContiguous && !kSeparatedStorage) ? 0 : (SPLIT_KV / PAGE_KV);
    // When true, paged path uses 2D TMA with page_coord*PAGE_KV as row offset instead of 3D batch index
    static constexpr bool kUse2DPagedTMA = kSeparatedStorage;

    uint32_t current_q_block_idx;
    uint32_t num_q_blocks;
    uint32_t num_q_tokens;

    const uint32_t* cu_seq_len_k_start;
    const uint32_t* cu_seq_len_k_end;
    uint32_t* seq_k_start;
    uint32_t* seq_k_end;

    const uint32_t* block_table;
    uint32_t block_table_stride;
    const uint32_t* seq_indices;
    uint32_t num_kv_pages;

    uint32_t cur_block_table_row;

    CUTLASS_DEVICE SM100PrefillPagedMQALogitsScheduler(
            const uint32_t& sm_idx,
            const uint32_t& num_q_tokens,
            const uint32_t* cu_seq_len_k_start,
            const uint32_t* cu_seq_len_k_end,
            uint32_t* seq_k_start,
            uint32_t* seq_k_end,
            const uint32_t* block_table,
            const uint32_t& block_table_stride,
            const uint32_t* seq_indices,
            const uint32_t& num_kv_pages):
        current_q_block_idx(sm_idx),
        num_q_blocks(math::ceil_div(num_q_tokens, BLOCK_Q)),
        num_q_tokens(num_q_tokens),
        cu_seq_len_k_start(cu_seq_len_k_start),
        cu_seq_len_k_end(cu_seq_len_k_end),
        seq_k_start(seq_k_start),
        seq_k_end(seq_k_end),
        block_table(block_table),
        block_table_stride(block_table_stride),
        seq_indices(seq_indices),
        num_kv_pages(num_kv_pages),
        cur_block_table_row(0) {}

    CUTLASS_DEVICE bool next_q_block(uint32_t& q_block_idx, uint32_t& kv_base, uint32_t& num_kv_splits) {
        if (current_q_block_idx >= num_q_blocks)
            return false;

        q_block_idx = current_q_block_idx;
        current_q_block_idx += kNumSMs;

        uint32_t start = cute::numeric_limits<uint32_t>::max();
        uint32_t end = cute::numeric_limits<uint32_t>::min();
        #pragma unroll
        for (uint32_t token_idx = 0; token_idx < BLOCK_Q; ++ token_idx) {
            const auto row_idx = cute::min(q_block_idx * BLOCK_Q + token_idx, num_q_tokens - 1);
            seq_k_start[token_idx] = cu_seq_len_k_start[row_idx];
            seq_k_end[token_idx] = cu_seq_len_k_end[row_idx];
            start = cute::min(start, seq_k_start[token_idx]);
            end = cute::max(end, seq_k_end[token_idx]);
        }

        cur_block_table_row = seq_indices[cute::min(q_block_idx * BLOCK_Q, num_q_tokens - 1)];

        kv_base = start / SPLIT_KV;
        num_kv_splits = math::ceil_div(end - kv_base * SPLIT_KV, SPLIT_KV);
        return true;
    }

    CUTLASS_DEVICE uint32_t get_q_tma_token_base(const uint32_t& q_block_idx) const {
        return q_block_idx * BLOCK_Q;
    }

    CUTLASS_DEVICE static uint32_t get_kv_tma_offset(const uint32_t& kv_base, const uint32_t& kv_split_idx) {
        return (kv_base + kv_split_idx) * SPLIT_KV;
    }

    CUTLASS_DEVICE uint32_t get_kv_page_coord_by_page_offset(const uint32_t& page_offset) const {
        if (page_offset >= num_kv_pages)
            return 0;
        const auto block_table_offset = cur_block_table_row * static_cast<uint64_t>(block_table_stride);
        return block_table[block_table_offset + page_offset];
    }

    CUTLASS_DEVICE static uint32_t get_logits_row(const uint32_t& q_block_idx, const uint32_t& token_idx) {
        return q_block_idx * BLOCK_Q + token_idx;
    }

    CUTLASS_DEVICE static uint32_t get_logits_col(const uint32_t& kv_base,
                                                  const uint32_t& kv_split_idx,
                                                  const uint32_t& math_thread_idx) {
        return (kv_base + kv_split_idx) * SPLIT_KV + math_thread_idx;
    }
};

} // namespace deep_gemm::sched
