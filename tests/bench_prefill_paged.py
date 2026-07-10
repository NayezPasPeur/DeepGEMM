"""
Comprehensive Prefill Paged MQA Logits Benchmark

Tests:
1. Page size impact: block_kv ∈ {32, 64, 128} (affects #TMA copies per SPLIT_KV)
2. 3D TMA (paged) vs 2D TMA (contiguous) — the "cost of paging"
3. Various chunk-prefill (seq_len_q, seq_len_kv) combinations including non-power-of-2

Fixed config: FP8, num_heads=16, head_dim=128, logits_dtype=bf16, weights=fp32
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import deep_gemm
from deep_gemm.testing import bench_kineto, get_arch_major
from deep_gemm.utils import ceil_div, per_custom_dims_cast_to_fp8


TEST_CASES = [
    # Standard power-of-2
    (4096, 4096),
    (4096, 8192),
    (4096, 16384),
    (4096, 32768),
    (4096, 65536),
    (8192, 8192),
    (8192, 16384),
    (8192, 32768),
    (8192, 65536),
    (16384, 16384),
    (16384, 32768),
    (16384, 65536),
    (32768, 32768),
    (32768, 65536),
    (65536, 65536),
    # Non-power-of-2
    (5000, 5000),
    (5000, 10000),
    (5000, 20000),
    (10000, 10000),
    (10000, 20000),
    (10000, 50000),
    (15000, 20000),
    (15000, 50000),
    (20000, 20000),
    (20000, 50000),
    (20000, 60000),
    (25000, 50000),
    (25000, 60000),
    (35000, 50000),
    (35000, 60000),
    (40000, 60000),
    (45000, 60000),
    (50000, 60000),
    (55000, 60000),
    (60000, 60000),
    # Fixed large KV
    (1000, 64000),
    (2000, 64000),
    (3000, 64000),
    (4000, 64000),
    (5000, 64000),
    (6000, 64000),
    (7000, 64000),
    (8000, 64000),
]


def contiguous_fp8_kv_to_paged(kv_fp8, kv_sf, block_kv):
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

    stride_bytes = block_kv * (head_dim + 4)
    fused_flat = torch.empty(num_blocks, stride_bytes, device=kv_fp8.device, dtype=torch.uint8)
    fused_flat[:, :block_kv * head_dim] = kv_fp8_padded.view(num_blocks, block_kv * head_dim).view(torch.uint8)
    fused_flat[:, block_kv * head_dim:] = kv_sf_padded.view(num_blocks, block_kv).view(torch.uint8).view(num_blocks, block_kv * 4)
    return fused_flat.view(num_blocks, block_kv, 1, head_dim + 4)


def run_benchmark():
    assert get_arch_major() == 10, "SM100 required"
    num_sms = deep_gemm.get_num_sms()

    num_heads = 16
    head_dim = 128
    logits_dtype = torch.bfloat16
    page_sizes = [32, 64, 128]

    print(f"{'='*140}")
    print(f"Prefill Paged MQA Logits Benchmark — B200 SM100, H={num_heads}, D={head_dim}, FP8, Logits=BF16")
    print(f"{'='*140}")
    print(f"{'SQ':>6} {'SK':>6} | {'Contiguous':>10} | {'2D-TMA(P64)':>11} {'vs Cont':>7} | {'3D-P32':>10} {'vs Cont':>7} | {'3D-P64':>10} {'vs Cont':>7} | {'3D-P128':>10} {'vs Cont':>7} |")
    print(f"{'-'*140}")

    for seq_len, seq_len_kv in TEST_CASES:
        # Generate data
        q_bf16 = torch.randn(seq_len, num_heads, head_dim, device='cuda', dtype=torch.bfloat16)
        kv_bf16 = torch.randn(seq_len_kv, head_dim, device='cuda', dtype=torch.bfloat16)
        weights = torch.randn(seq_len, num_heads, device='cuda', dtype=torch.float32)

        # True causal mask for chunk prefill: token i sees KV [0, (SK-SQ)+i+1)
        ks = torch.zeros(seq_len, dtype=torch.int, device='cuda')
        ke = torch.arange(seq_len, dtype=torch.int, device='cuda') + (seq_len_kv - seq_len) + 1
        ke = ke.clamp(max=seq_len_kv)

        # Quantize
        q_fp8 = q_bf16.to(torch.float8_e4m3fn)
        kv_fp8, kv_sf = per_custom_dims_cast_to_fp8(kv_bf16, (0,), False)

        # --- Contiguous (2D TMA baseline) ---
        cont_kwargs = dict(
            q=(q_fp8, None), kv=(kv_fp8, kv_sf), weights=weights,
            cu_seq_len_k_start=ks, cu_seq_len_k_end=ke,
            clean_logits=False, max_seqlen_k=0, logits_dtype=logits_dtype
        )
        try:
            deep_gemm.fp8_fp4_mqa_logits(**cont_kwargs)
            t_cont = bench_kineto(lambda: deep_gemm.fp8_fp4_mqa_logits(**cont_kwargs),
                                  'mqa_logits', suppress_kineto_output=True)
        except Exception as e:
            t_cont = None

        # --- 2D TMA path (force_contiguous=True, same scheduler, uses page_kv=64 for layout) ---
        t_2d = None
        try:
            fused_kv_64 = contiguous_fp8_kv_to_paged(kv_fp8, kv_sf, 64)
            num_blocks_64 = fused_kv_64.shape[0]
            block_table_64 = torch.arange(num_blocks_64, dtype=torch.int, device='cuda').unsqueeze(0)
            seq_indices = torch.zeros(seq_len, dtype=torch.int, device='cuda')

            kwargs_2d = dict(
                q=(q_fp8, None), kv_cache=fused_kv_64, weights=weights,
                cu_seq_len_k_start=ks, cu_seq_len_k_end=ke,
                block_table=block_table_64, seq_indices=seq_indices,
                clean_logits=False, max_seqlen_k=0, logits_dtype=logits_dtype,
                force_contiguous=True
            )
            deep_gemm.fp8_fp4_prefill_paged_mqa_logits(**kwargs_2d)
            t_2d = bench_kineto(lambda: deep_gemm.fp8_fp4_prefill_paged_mqa_logits(**kwargs_2d),
                                'prefill_paged', suppress_kineto_output=True)
            del fused_kv_64, block_table_64
        except Exception as e:
            pass

        # --- Paged (3D TMA) with different page sizes ---
        page_results = {}
        for block_kv in page_sizes:
            try:
                fused_kv = contiguous_fp8_kv_to_paged(kv_fp8, kv_sf, block_kv)
                num_blocks = fused_kv.shape[0]
                block_table = torch.arange(num_blocks, dtype=torch.int, device='cuda').unsqueeze(0)
                seq_indices = torch.zeros(seq_len, dtype=torch.int, device='cuda')

                paged_kwargs = dict(
                    q=(q_fp8, None), kv_cache=fused_kv, weights=weights,
                    cu_seq_len_k_start=ks, cu_seq_len_k_end=ke,
                    block_table=block_table, seq_indices=seq_indices,
                    clean_logits=False, max_seqlen_k=0, logits_dtype=logits_dtype
                )
                deep_gemm.fp8_fp4_prefill_paged_mqa_logits(**paged_kwargs)
                t_paged = bench_kineto(lambda: deep_gemm.fp8_fp4_prefill_paged_mqa_logits(**paged_kwargs),
                                       'prefill_paged', suppress_kineto_output=True)
                page_results[block_kv] = t_paged
                del fused_kv, block_table, seq_indices
            except Exception as e:
                page_results[block_kv] = None
                if block_kv == page_sizes[0]:
                    import traceback
                    traceback.print_exc()

        # Format output
        def fmt_time(t):
            if t is None:
                return "ERR"
            return f"{t*1e6:.0f}us"

        def fmt_ratio(t, baseline):
            if t is None or baseline is None:
                return "N/A"
            return f"{t/baseline:.2f}x"

        cont_str = fmt_time(t_cont)
        d2_str = fmt_time(t_2d)
        p32_str = fmt_time(page_results.get(32))
        p64_str = fmt_time(page_results.get(64))
        p128_str = fmt_time(page_results.get(128))
        r2d = fmt_ratio(t_2d, t_cont)
        r32 = fmt_ratio(page_results.get(32), t_cont)
        r64 = fmt_ratio(page_results.get(64), t_cont)
        r128 = fmt_ratio(page_results.get(128), t_cont)

        print(f"{seq_len:>6} {seq_len_kv:>6} | {cont_str:>10} | {d2_str:>11} {r2d:>7} | {p32_str:>10} {r32:>7} | {p64_str:>10} {r64:>7} | {p128_str:>10} {r128:>7} |")

        del q_bf16, kv_bf16, q_fp8, kv_fp8, kv_sf, weights
        torch.cuda.empty_cache()

    print(f"{'='*140}")
    print()
    print("Legend:")
    print("  Contiguous  = original contiguous kernel (2D TMA, separate Q/KV tensors)")
    print("  2D-TMA(P64) = new kernel with force_contiguous=True (2D TMA on paged layout, sequential block_table)")
    print("  3D-P32/64/128 = new kernel with 3D TMA, page_size=32/64/128")
    print("  vs Cont = slowdown relative to contiguous baseline (1.00x = same speed)")
    print()
    print("Key:")
    print("  2D-TMA vs Contiguous = overhead of new scheduler + paged memory layout (no TMA difference)")
    print("  3D-Pxx vs 2D-TMA    = pure 3D TMA overhead (same scheduler, same layout)")
    print("  3D-Pxx vs Contiguous = total paged overhead (scheduler + layout + TMA)")


if __name__ == '__main__':
    torch.manual_seed(0)
    run_benchmark()
