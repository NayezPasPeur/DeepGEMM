"""
Prefill Paged MQA Logits Performance Comparison (SM100)

Converts prefill contiguous KV data into paged format, then calls both
  - fp8_fp4_mqa_logits (contiguous)
  - fp8_fp4_paged_mqa_logits (paged)
on the same quantized data. Validates correctness and reports performance delta.
"""

import os
import random
import torch
from typing import Tuple

import deep_gemm
from deep_gemm.testing import (
    bench_kineto,
    assert_bitwise_equal, calc_diff, count_bytes,
    get_arch_major,
)
from deep_gemm.utils import (
    ceil_div, align,
    per_custom_dims_cast_to_fp8,
    per_token_cast_to_fp4, cast_back_from_fp4,
)


def contiguous_fp8_kv_to_paged(kv_fp8: torch.Tensor, kv_sf: torch.Tensor,
                                block_kv: int) -> torch.Tensor:
    """
    Convert contiguous FP8 KV [seq_len_kv, head_dim] + sf [seq_len_kv] (float per-token scale)
    into paged fused format [num_blocks, block_kv, 1, head_dim + 4].

    Memory layout per block (matching C++ from_blob expectations):
      [FP8 data: block_kv * head_dim bytes] [Scales: block_kv * sizeof(float) bytes]
    The 4D view is then applied for API shape/stride checks.
    """
    seq_len_kv, head_dim = kv_fp8.shape
    num_blocks = ceil_div(seq_len_kv, block_kv)
    pad_len = num_blocks * block_kv - seq_len_kv

    if pad_len > 0:
        kv_fp8_padded = torch.zeros(num_blocks * block_kv, head_dim, device=kv_fp8.device, dtype=kv_fp8.dtype)
        kv_fp8_padded[:seq_len_kv] = kv_fp8
        kv_sf_padded = torch.ones(num_blocks * block_kv, device=kv_sf.device, dtype=kv_sf.dtype)
        kv_sf_padded[:seq_len_kv] = kv_sf
    else:
        kv_fp8_padded = kv_fp8
        kv_sf_padded = kv_sf

    # Flat layout per block: [fp8_data (block_kv * head_dim bytes)] [scales (block_kv * 4 bytes)]
    stride_bytes = block_kv * (head_dim + 4)
    fused_flat = torch.empty(num_blocks, stride_bytes, device=kv_fp8.device, dtype=torch.uint8)
    fused_flat[:, :block_kv * head_dim] = kv_fp8_padded.view(num_blocks, block_kv * head_dim).view(torch.uint8)
    fused_flat[:, block_kv * head_dim:] = kv_sf_padded.view(num_blocks, block_kv).view(torch.uint8).view(num_blocks, block_kv * 4)
    return fused_flat.view(num_blocks, block_kv, 1, head_dim + 4)


def contiguous_fp4_kv_to_paged(kv_fp4: torch.Tensor, kv_sf: torch.Tensor,
                                head_dim: int, block_kv: int) -> torch.Tensor:
    """
    Convert contiguous FP4 KV [seq_len_kv, head_dim//2] (packed) + sf [seq_len_kv] (int32)
    into paged fused format [num_blocks, block_kv, 1, head_dim//2 + 4].

    Memory layout per block (matching C++ from_blob expectations):
      [FP4 data: block_kv * half_dim bytes] [Scales: block_kv * sizeof(int32) bytes]
    """
    seq_len_kv = kv_fp4.shape[0]
    half_dim = head_dim // 2
    num_blocks = ceil_div(seq_len_kv, block_kv)
    pad_len = num_blocks * block_kv - seq_len_kv

    if pad_len > 0:
        kv_fp4_padded = torch.zeros(num_blocks * block_kv, half_dim, device=kv_fp4.device, dtype=kv_fp4.dtype)
        kv_fp4_padded[:seq_len_kv] = kv_fp4
        # ue8m0 of 1.0 = exponent 127 → each byte = 0x7f; packed int32 = 0x7f7f7f7f
        kv_sf_padded = torch.full((num_blocks * block_kv,), 0x7f7f7f7f, device=kv_sf.device, dtype=kv_sf.dtype)
        kv_sf_padded[:seq_len_kv] = kv_sf
    else:
        kv_fp4_padded = kv_fp4
        kv_sf_padded = kv_sf

    # Flat layout per block: [fp4_data (block_kv * half_dim bytes)] [scales (block_kv * 4 bytes)]
    stride_bytes = block_kv * (half_dim + 4)
    fused_flat = torch.empty(num_blocks, stride_bytes, device=kv_fp4.device, dtype=torch.uint8)
    fused_flat[:, :block_kv * half_dim] = kv_fp4_padded.view(num_blocks, block_kv * half_dim).view(torch.uint8)
    fused_flat[:, block_kv * half_dim:] = kv_sf_padded.view(num_blocks, block_kv).view(torch.uint8).view(num_blocks, block_kv * 4)
    return fused_flat.view(num_blocks, block_kv, 1, half_dim + 4)


def test_prefill_paged_mqa_logits():
    """
    For each configuration, run the same quantized prefill data through both
    contiguous and paged APIs, verifying correctness and comparing latency.
    """
    arch_major = get_arch_major()
    if arch_major != 10:
        print(f'Skipping: SM100 required, got SM{arch_major}0')
        return

    num_sms = deep_gemm.get_num_sms()

    def enumerate_cases():
        for is_fp4 in (True, False):
            for logits_dtype in (torch.bfloat16, torch.float):
                for weights_dtype in (torch.float, torch.bfloat16):
                    if weights_dtype == torch.bfloat16 and logits_dtype == torch.float:
                        continue
                    for seq_len in (2048, 4096, 8192):
                        for seq_len_kv in (8192, 32768, 65536):
                            for num_heads in (8, 16, 32, 64):
                                head_dims = (64, 128) if is_fp4 else (32, 64, 128)
                                for head_dim in head_dims:
                                    for block_kv in (32, 64):
                                        yield (is_fp4, logits_dtype, weights_dtype,
                                               seq_len, seq_len_kv, num_heads, head_dim, block_kv)

    cases = list(enumerate_cases())
    num_cases_env = os.getenv('DG_MQA_NUM_CASES')
    if num_cases_env is not None:
        rng = random.Random(42)
        cases = rng.sample(cases, min(int(num_cases_env), len(cases)))
    print(f'Testing Prefill Paged vs Contiguous MQA Logits ({len(cases)} cases):')

    for is_fp4, logits_dtype, weights_dtype, seq_len, seq_len_kv, num_heads, head_dim, block_kv in cases:

        # --- Generate random inputs (same as prefill test, disable_cp=True) ---
        q_bf16 = torch.randn(seq_len, num_heads, head_dim, device='cuda', dtype=torch.bfloat16)
        kv_bf16 = torch.randn(seq_len_kv, head_dim, device='cuda', dtype=torch.bfloat16)
        weights = torch.randn(seq_len, num_heads, device='cuda', dtype=torch.float32)
        kernel_weights = weights.to(weights_dtype)

        # Simple causal-like mask: ks=0, ke=i+offset (disable_cp)
        ks = torch.zeros(seq_len, dtype=torch.int, device='cuda')
        ke = torch.arange(seq_len, dtype=torch.int, device='cuda') + (seq_len_kv - seq_len)

        # --- Quantize Q and KV ---
        if is_fp4:
            q_fp4 = per_token_cast_to_fp4(q_bf16.view(-1, head_dim), use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
            q_cont = (q_fp4[0].view(seq_len, num_heads, head_dim // 2), q_fp4[1].view(seq_len, num_heads))

            kv_fp4 = per_token_cast_to_fp4(kv_bf16.view(-1, head_dim), use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
            kv_cont = (kv_fp4[0].view(seq_len_kv, head_dim // 2), kv_fp4[1].view(seq_len_kv))
        else:
            q_cont = (q_bf16.to(torch.float8_e4m3fn), None)
            kv_cont = per_custom_dims_cast_to_fp8(kv_bf16, (0,), False)

        # ============================================================
        # Part 1: Run contiguous (prefill) kernel
        # ============================================================
        cont_kwargs = dict(
            q=q_cont, kv=kv_cont, weights=kernel_weights,
            cu_seq_len_k_start=ks, cu_seq_len_k_end=ke,
            clean_logits=False, max_seqlen_k=0,
            logits_dtype=logits_dtype
        )
        logits_cont = deep_gemm.fp8_fp4_mqa_logits(**cont_kwargs)

        # ============================================================
        # Part 2: Convert to paged format and run paged kernel
        # ============================================================

        # Convert KV to paged fused format
        if is_fp4:
            fused_kv_cache = contiguous_fp4_kv_to_paged(kv_cont[0], kv_cont[1], head_dim, block_kv)
        else:
            fused_kv_cache = contiguous_fp8_kv_to_paged(kv_cont[0], kv_cont[1], block_kv)

        num_blocks = fused_kv_cache.shape[0]

        # Q for paged: [batch_size=1, next_n=seq_len, num_heads, head_dim]
        if is_fp4:
            q_paged = (q_cont[0].unsqueeze(0), q_cont[1].unsqueeze(0))
        else:
            q_paged = (q_cont[0].unsqueeze(0), None)

        # Weights stay [seq_len, num_heads] (batch_size * next_n = 1 * seq_len = seq_len)
        # Context lens: [batch_size=1, next_n=seq_len] — each token sees up to ke[i] tokens
        context_lens = ke.view(1, seq_len)

        # Block table: [batch_size=1, max_num_blocks] — sequential mapping
        block_table = torch.arange(num_blocks, dtype=torch.int, device='cuda').unsqueeze(0)

        # Schedule metadata
        schedule_meta = deep_gemm.get_paged_mqa_logits_metadata(
            context_lens, block_kv, num_sms
        )

        # Max context len (rounded to block_kv boundary for paged)
        max_context_len = num_blocks * block_kv

        paged_kwargs = dict(
            q=q_paged, kv_cache=fused_kv_cache, weights=kernel_weights,
            context_lens=context_lens, block_table=block_table,
            schedule_meta=schedule_meta,
            max_context_len=max_context_len,
            clean_logits=False,
            logits_dtype=logits_dtype,
        )
        logits_paged = deep_gemm.fp8_fp4_paged_mqa_logits(**paged_kwargs)

        # ============================================================
        # Part 3: Correctness validation
        # ============================================================

        # Build valid mask: for each Q token i, valid KV range is [0, ke[i])
        # Contiguous logits shape: [seq_len, seq_len_kv]
        # Paged logits shape: [seq_len, max_context_len] (may be wider)
        positions_cont = torch.arange(logits_cont.shape[1], device='cuda').unsqueeze(0)
        valid_mask_cont = (positions_cont >= ks.unsqueeze(1)) & (positions_cont < ke.unsqueeze(1))

        positions_paged = torch.arange(logits_paged.shape[1], device='cuda').unsqueeze(0)
        valid_mask_paged = positions_paged < ke.unsqueeze(1)

        # Compare only the overlapping valid region
        min_cols = min(logits_cont.shape[1], logits_paged.shape[1])
        overlap_mask = valid_mask_cont[:, :min_cols] & valid_mask_paged[:, :min_cols]

        logits_cont_masked = logits_cont[:, :min_cols].float().masked_fill(~overlap_mask, 0)
        logits_paged_masked = logits_paged[:, :min_cols].float().masked_fill(~overlap_mask, 0)

        diff = calc_diff(logits_cont_masked, logits_paged_masked)
        tol = 0.02 if is_fp4 else 1e-3
        assert diff < tol, (
            f"Contiguous vs Paged diff too large: {diff:.6e} "
            f"(is_fp4={is_fp4}, H={num_heads}, D={head_dim}, "
            f"SQ={seq_len}, SK={seq_len_kv}, PAGE={block_kv})"
        )

        # Self-consistency of paged kernel
        logits_paged_again = deep_gemm.fp8_fp4_paged_mqa_logits(**paged_kwargs)
        paged_self_mask = valid_mask_paged
        assert_bitwise_equal(
            logits_paged_again.masked_fill(~paged_self_mask, 0),
            logits_paged.masked_fill(~paged_self_mask, 0),
            'prefill paged self-consistency'
        )

        # ============================================================
        # Part 4: Performance comparison
        # ============================================================

        # Contiguous kernel time
        t_cont = bench_kineto(
            lambda: deep_gemm.fp8_fp4_mqa_logits(**cont_kwargs),
            'mqa_logits', suppress_kineto_output=True
        )

        # Paged kernel time (includes the main kernel only; metadata is precomputed)
        t_paged = bench_kineto(
            lambda: deep_gemm.fp8_fp4_paged_mqa_logits(**paged_kwargs),
            'paged_mqa_logits', suppress_kineto_output=True
        )

        # Compute TFLOPS reference
        # Cost = sum of valid KV tokens across all Q tokens
        cost = ke.long().sum().item()
        tflops = 2 * cost * num_heads * head_dim / 1e12

        slowdown = t_paged / t_cont if t_cont > 0 else float('inf')

        dtype_tag = 'BF16' if logits_dtype == torch.bfloat16 else 'FP32'
        w_tag = 'BF16' if weights_dtype == torch.bfloat16 else 'FP32'
        print(f' > FP4={int(is_fp4):1d}, Logits={dtype_tag:4}, W={w_tag:4}, '
              f'H={num_heads:2}, D={head_dim:3}, SQ={seq_len:5}, SK={seq_len_kv:5}, PAGE={block_kv:2}: '
              f'Cont={t_cont*1e6:5.0f}us ({tflops/t_cont:4.0f} TFLOPS) | '
              f'Paged={t_paged*1e6:5.0f}us ({tflops/t_paged:4.0f} TFLOPS) | '
              f'Slowdown={slowdown:.2f}x | Diff={diff:.2e}')

        # Cleanup
        del logits_cont, logits_paged, logits_paged_again
        del fused_kv_cache, q_paged, context_lens, block_table, schedule_meta
        torch.cuda.empty_cache()

    print()


def test_prefill_paged_new_kernel():
    """
    3-way comparison: contiguous vs old paged vs NEW prefill_paged kernel.
    The new kernel uses round-robin scheduling (no metadata kernel) + paged 3D TMA.
    """
    arch_major = get_arch_major()
    if arch_major != 10:
        print(f'Skipping: SM100 required, got SM{arch_major}0')
        return

    num_sms = deep_gemm.get_num_sms()

    def enumerate_cases():
        for is_fp4 in (True, False):
            for logits_dtype in (torch.bfloat16, torch.float):
                for weights_dtype in (torch.float, torch.bfloat16):
                    if weights_dtype == torch.bfloat16 and logits_dtype == torch.float:
                        continue
                    for seq_len in (2048, 4096, 8192):
                        for seq_len_kv in (8192, 32768, 65536):
                            for num_heads in (8, 16, 32, 64):
                                head_dims = (64, 128) if is_fp4 else (64, 128)
                                for head_dim in head_dims:
                                    for block_kv in (32, 64):
                                        yield (is_fp4, logits_dtype, weights_dtype,
                                               seq_len, seq_len_kv, num_heads, head_dim, block_kv)

    cases = list(enumerate_cases())
    num_cases_env = os.getenv('DG_MQA_NUM_CASES')
    if num_cases_env is not None:
        rng = random.Random(99)
        cases = rng.sample(cases, min(int(num_cases_env), len(cases)))
    print(f'Testing NEW Prefill Paged Kernel (3-way comparison, {len(cases)} cases):')

    for is_fp4, logits_dtype, weights_dtype, seq_len, seq_len_kv, num_heads, head_dim, block_kv in cases:

        q_bf16 = torch.randn(seq_len, num_heads, head_dim, device='cuda', dtype=torch.bfloat16)
        kv_bf16 = torch.randn(seq_len_kv, head_dim, device='cuda', dtype=torch.bfloat16)
        weights = torch.randn(seq_len, num_heads, device='cuda', dtype=torch.float32)
        kernel_weights = weights.to(weights_dtype)

        # Simple causal mask: ks=0, ke varies per token
        ks = torch.zeros(seq_len, dtype=torch.int, device='cuda')
        ke = torch.arange(seq_len, dtype=torch.int, device='cuda') + (seq_len_kv - seq_len)

        # Quantize
        if is_fp4:
            q_fp4 = per_token_cast_to_fp4(q_bf16.view(-1, head_dim), use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
            q_cont = (q_fp4[0].view(seq_len, num_heads, head_dim // 2), q_fp4[1].view(seq_len, num_heads))
            kv_fp4 = per_token_cast_to_fp4(kv_bf16.view(-1, head_dim), use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
            kv_cont = (kv_fp4[0].view(seq_len_kv, head_dim // 2), kv_fp4[1].view(seq_len_kv))
        else:
            q_cont = (q_bf16.to(torch.float8_e4m3fn), None)
            kv_cont = per_custom_dims_cast_to_fp8(kv_bf16, (0,), False)

        # --- Contiguous kernel ---
        cont_kwargs = dict(
            q=q_cont, kv=kv_cont, weights=kernel_weights,
            cu_seq_len_k_start=ks, cu_seq_len_k_end=ke,
            clean_logits=False, max_seqlen_k=0,
            logits_dtype=logits_dtype
        )
        logits_cont = deep_gemm.fp8_fp4_mqa_logits(**cont_kwargs)

        # --- Convert KV to paged format ---
        if is_fp4:
            fused_kv_cache = contiguous_fp4_kv_to_paged(kv_cont[0], kv_cont[1], head_dim, block_kv)
        else:
            fused_kv_cache = contiguous_fp8_kv_to_paged(kv_cont[0], kv_cont[1], block_kv)

        num_blocks = fused_kv_cache.shape[0]

        # block_table: [1, num_blocks] sequential
        block_table = torch.arange(num_blocks, dtype=torch.int, device='cuda').unsqueeze(0)

        # seq_indices: all tokens belong to sequence 0
        seq_indices = torch.zeros(seq_len, dtype=torch.int, device='cuda')

        # --- NEW prefill paged kernel ---
        new_kwargs = dict(
            q=q_cont, kv_cache=fused_kv_cache, weights=kernel_weights,
            cu_seq_len_k_start=ks, cu_seq_len_k_end=ke,
            block_table=block_table, seq_indices=seq_indices,
            clean_logits=False, max_seqlen_k=0,
            logits_dtype=logits_dtype
        )
        logits_new = deep_gemm.fp8_fp4_prefill_paged_mqa_logits(**new_kwargs)

        # --- Correctness: compare new vs contiguous ---
        min_cols = min(logits_cont.shape[1], logits_new.shape[1])
        positions = torch.arange(min_cols, device='cuda').unsqueeze(0)
        valid_mask = (positions >= ks.unsqueeze(1)) & (positions < ke.unsqueeze(1))

        cont_masked = logits_cont[:, :min_cols].float().masked_fill(~valid_mask, 0)
        new_masked = logits_new[:, :min_cols].float().masked_fill(~valid_mask, 0)

        diff = calc_diff(cont_masked, new_masked)
        tol = 0.02 if is_fp4 else 1e-3
        assert diff < tol, (
            f"New prefill_paged vs Contiguous diff too large: {diff:.6e} "
            f"(is_fp4={is_fp4}, H={num_heads}, D={head_dim}, "
            f"SQ={seq_len}, SK={seq_len_kv}, PAGE={block_kv})"
        )

        # Self-consistency
        logits_new_again = deep_gemm.fp8_fp4_prefill_paged_mqa_logits(**new_kwargs)
        valid_mask_new = valid_mask[:, :logits_new.shape[1]] if valid_mask.shape[1] > logits_new.shape[1] else valid_mask
        assert_bitwise_equal(
            logits_new_again.masked_fill(~valid_mask_new, 0),
            logits_new.masked_fill(~valid_mask_new, 0),
            'new prefill paged self-consistency'
        )

        # --- Performance: 3-way comparison ---
        t_cont = bench_kineto(
            lambda: deep_gemm.fp8_fp4_mqa_logits(**cont_kwargs),
            'mqa_logits', suppress_kineto_output=True
        )
        t_new = bench_kineto(
            lambda: deep_gemm.fp8_fp4_prefill_paged_mqa_logits(**new_kwargs),
            'prefill_paged_mqa_logits', suppress_kineto_output=True
        )

        cost = ke.long().sum().item()
        tflops = 2 * cost * num_heads * head_dim / 1e12
        slowdown_new = t_new / t_cont if t_cont > 0 else float('inf')

        dtype_tag = 'BF16' if logits_dtype == torch.bfloat16 else 'FP32'
        w_tag = 'BF16' if weights_dtype == torch.bfloat16 else 'FP32'
        print(f' > FP4={int(is_fp4):1d}, Logits={dtype_tag:4}, W={w_tag:4}, '
              f'H={num_heads:2}, D={head_dim:3}, SQ={seq_len:5}, SK={seq_len_kv:5}, PAGE={block_kv:2}: '
              f'Cont={t_cont*1e6:5.0f}us ({tflops/t_cont:4.0f} TFLOPS) | '
              f'NewPaged={t_new*1e6:5.0f}us ({tflops/t_new:4.0f} TFLOPS) | '
              f'Slowdown={slowdown_new:.2f}x | Diff={diff:.2e}')

        del logits_cont, logits_new, logits_new_again
        del fused_kv_cache, block_table, seq_indices
        torch.cuda.empty_cache()

    print()


if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)
    test_prefill_paged_mqa_logits()
    test_prefill_paged_new_kernel()
