"""
Comprehensive Prefill Paged MQA Logits Benchmark v2

Tests:
1. Page size: block_kv ∈ {32, 64, 128}
2. 2D TMA (force_contiguous) vs 3D TMA — separated overhead
3. Non-contiguous (shuffled) block_table — real-world fragmentation
4. Multiple num_heads configs: H=8, H=16, H=32, H=64
5. Full TEST_CASES matrix with causal mask

Output: machine-readable CSV + human-readable table
"""

import os
import sys
import csv
import random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import deep_gemm
from deep_gemm.testing import bench_kineto, get_arch_major
from deep_gemm.utils import ceil_div, per_custom_dims_cast_to_fp8


TEST_CASES = [
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


def shuffle_paged_kv(fused_kv_cache, block_table):
    """
    Physically shuffle KV pages in memory and update block_table accordingly.
    This simulates real-world fragmented paged KV where pages are non-contiguous.
    """
    num_blocks = fused_kv_cache.shape[0]
    perm = torch.randperm(num_blocks, device=fused_kv_cache.device)
    shuffled_kv = fused_kv_cache[perm]
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(num_blocks, device=perm.device)
    shuffled_block_table = inv_perm[block_table.long()].int()
    return shuffled_kv, shuffled_block_table


def bench_one(fn, kernel_name):
    try:
        fn()
        return bench_kineto(fn, kernel_name, suppress_kineto_output=True)
    except Exception:
        return None


def run_benchmark():
    assert get_arch_major() == 10, "SM100 required"

    head_dim = 128
    logits_dtype = torch.bfloat16
    page_sizes = [32, 64, 128]
    heads_list = [16, 32, 64]

    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bench_results.csv')
    csv_file = open(csv_path, 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(['num_heads', 'head_dim', 'seq_len_q', 'seq_len_kv', 'page_size',
                     'contiguous_us', '2d_tma_seq_us', '3d_tma_seq_us', '3d_tma_shuf_us',
                     'ratio_2d_vs_cont', 'ratio_3d_seq_vs_cont', 'ratio_3d_shuf_vs_cont',
                     'ratio_3d_vs_2d', 'ratio_shuf_vs_seq'])

    for num_heads in heads_list:
        block_q = 128 // num_heads
        print(f"\n{'='*150}")
        print(f"  H={num_heads}, D={head_dim}, BLOCK_Q={block_q}, FP8, Logits=BF16, Causal Mask")
        print(f"{'='*150}")
        print(f"{'SQ':>6} {'SK':>6} {'PG':>3} | {'Contig':>8} | {'2D-seq':>8} {'r':>5} | {'3D-seq':>8} {'r':>5} | {'3D-shuf':>8} {'r':>5} | {'3D/2D':>6} | {'shuf/seq':>8} |")
        print(f"{'-'*150}")

        for seq_len, seq_len_kv in TEST_CASES:
            q_bf16 = torch.randn(seq_len, num_heads, head_dim, device='cuda', dtype=torch.bfloat16)
            kv_bf16 = torch.randn(seq_len_kv, head_dim, device='cuda', dtype=torch.bfloat16)
            weights = torch.randn(seq_len, num_heads, device='cuda', dtype=torch.float32)

            ks = torch.zeros(seq_len, dtype=torch.int, device='cuda')
            ke = (torch.arange(seq_len, dtype=torch.int, device='cuda') + (seq_len_kv - seq_len) + 1).clamp(max=seq_len_kv)

            q_fp8 = q_bf16.to(torch.float8_e4m3fn)
            kv_fp8, kv_sf = per_custom_dims_cast_to_fp8(kv_bf16, (0,), False)

            cont_kwargs = dict(
                q=(q_fp8, None), kv=(kv_fp8, kv_sf), weights=weights,
                cu_seq_len_k_start=ks, cu_seq_len_k_end=ke,
                clean_logits=False, max_seqlen_k=0, logits_dtype=logits_dtype
            )
            t_cont = bench_one(lambda: deep_gemm.fp8_fp4_mqa_logits(**cont_kwargs), 'mqa_logits')

            for block_kv in page_sizes:
                fused_kv = contiguous_fp8_kv_to_paged(kv_fp8, kv_sf, block_kv)
                num_blocks = fused_kv.shape[0]
                block_table_seq = torch.arange(num_blocks, dtype=torch.int, device='cuda').unsqueeze(0)
                seq_indices = torch.zeros(seq_len, dtype=torch.int, device='cuda')

                # 2D TMA with sequential block_table (force_contiguous=True)
                kwargs_2d = dict(
                    q=(q_fp8, None), kv_cache=fused_kv, weights=weights,
                    cu_seq_len_k_start=ks, cu_seq_len_k_end=ke,
                    block_table=block_table_seq, seq_indices=seq_indices,
                    clean_logits=False, max_seqlen_k=0, logits_dtype=logits_dtype,
                    force_contiguous=True
                )
                t_2d_seq = bench_one(lambda: deep_gemm.fp8_fp4_prefill_paged_mqa_logits(**kwargs_2d), 'prefill_paged')

                # 3D TMA with sequential block_table
                kwargs_3d_seq = dict(
                    q=(q_fp8, None), kv_cache=fused_kv, weights=weights,
                    cu_seq_len_k_start=ks, cu_seq_len_k_end=ke,
                    block_table=block_table_seq, seq_indices=seq_indices,
                    clean_logits=False, max_seqlen_k=0, logits_dtype=logits_dtype,
                    force_contiguous=False
                )
                t_3d_seq = bench_one(lambda: deep_gemm.fp8_fp4_prefill_paged_mqa_logits(**kwargs_3d_seq), 'prefill_paged')

                # 3D TMA with shuffled (non-contiguous) block_table
                shuffled_kv, shuffled_bt = shuffle_paged_kv(fused_kv, block_table_seq)
                kwargs_3d_shuf = dict(
                    q=(q_fp8, None), kv_cache=shuffled_kv, weights=weights,
                    cu_seq_len_k_start=ks, cu_seq_len_k_end=ke,
                    block_table=shuffled_bt, seq_indices=seq_indices,
                    clean_logits=False, max_seqlen_k=0, logits_dtype=logits_dtype,
                    force_contiguous=False
                )
                t_3d_shuf = bench_one(lambda: deep_gemm.fp8_fp4_prefill_paged_mqa_logits(**kwargs_3d_shuf), 'prefill_paged')

                def ratio(a, b):
                    if a is None or b is None or b == 0:
                        return None
                    return a / b

                r_2d = ratio(t_2d_seq, t_cont)
                r_3d_seq = ratio(t_3d_seq, t_cont)
                r_3d_shuf = ratio(t_3d_shuf, t_cont)
                r_3d_vs_2d = ratio(t_3d_seq, t_2d_seq)
                r_shuf_vs_seq = ratio(t_3d_shuf, t_3d_seq)

                def f_us(t):
                    return f"{t*1e6:.0f}us" if t else "ERR"
                def f_r(r):
                    return f"{r:.2f}x" if r else "N/A"

                print(f"{seq_len:>6} {seq_len_kv:>6} {block_kv:>3} | {f_us(t_cont):>8} | "
                      f"{f_us(t_2d_seq):>8} {f_r(r_2d):>5} | "
                      f"{f_us(t_3d_seq):>8} {f_r(r_3d_seq):>5} | "
                      f"{f_us(t_3d_shuf):>8} {f_r(r_3d_shuf):>5} | "
                      f"{f_r(r_3d_vs_2d):>6} | {f_r(r_shuf_vs_seq):>8} |")

                writer.writerow([
                    num_heads, head_dim, seq_len, seq_len_kv, block_kv,
                    f"{t_cont*1e6:.1f}" if t_cont else "",
                    f"{t_2d_seq*1e6:.1f}" if t_2d_seq else "",
                    f"{t_3d_seq*1e6:.1f}" if t_3d_seq else "",
                    f"{t_3d_shuf*1e6:.1f}" if t_3d_shuf else "",
                    f"{r_2d:.4f}" if r_2d else "",
                    f"{r_3d_seq:.4f}" if r_3d_seq else "",
                    f"{r_3d_shuf:.4f}" if r_3d_shuf else "",
                    f"{r_3d_vs_2d:.4f}" if r_3d_vs_2d else "",
                    f"{r_shuf_vs_seq:.4f}" if r_shuf_vs_seq else "",
                ])

                del fused_kv, shuffled_kv, shuffled_bt, block_table_seq
                torch.cuda.empty_cache()

            del q_bf16, kv_bf16, q_fp8, kv_fp8, kv_sf, weights
            torch.cuda.empty_cache()

    csv_file.close()
    print(f"\n\nCSV results saved to: {csv_path}")
    print("\nColumn definitions:")
    print("  Contig     = original contiguous kernel (2D TMA, separate tensors)")
    print("  2D-seq     = new kernel, force_contiguous=True, sequential block_table")
    print("  3D-seq     = new kernel, 3D TMA, sequential block_table (pages physically contiguous)")
    print("  3D-shuf    = new kernel, 3D TMA, shuffled block_table (pages physically scattered)")
    print("  3D/2D      = pure 3D TMA overhead (same scheduler, same data)")
    print("  shuf/seq   = fragmentation overhead (same TMA type, scattered vs contiguous pages)")


if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)
    run_benchmark()
