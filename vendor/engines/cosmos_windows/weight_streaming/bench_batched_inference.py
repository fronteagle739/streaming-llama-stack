# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/bench_batched_inference.py
# Machine fallback paths cleared; no model data included.
import sys, time, torch
sys.path.insert(0, '')
from cosmos_windows.weight_streaming.build_streamed_transformer import build_streamed_backward_transformer
from cosmos_windows.weight_streaming.batched_backward import batched_block_loop
DEV = 'cuda'

def _imgs(tf, N, und_len, gen_len, seed=5):
    cfg = tf.config
    H = cfg.hidden_size
    head_dim = getattr(cfg, 'head_dim', getattr(cfg, 'attention_head_dim', 128))
    dt = next((p for p in tf.parameters() if p.device.type == 'cuda')).dtype
    g = torch.Generator(device=DEV).manual_seed(seed)
    out = []
    for _ in range(N):
        und = torch.randn(und_len, H, generator=g, device=DEV, dtype=dt)
        gen = torch.randn(gen_len, H, generator=g, device=DEV, dtype=dt)
        rot = (torch.ones(und_len, head_dim, device=DEV, dtype=dt), torch.zeros(und_len, head_dim, device=DEV, dtype=dt), torch.ones(gen_len, head_dim, device=DEV, dtype=dt), torch.zeros(gen_len, head_dim, device=DEV, dtype=dt))
        out.append((und, gen, rot))
    return out

def _bench(tf, st, imgs, reps=3):
    with torch.no_grad():
        st.new_pass()
        batched_block_loop(tf, st, imgs)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            st.new_pass()
            batched_block_loop(tf, st, imgs)
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps

def main():
    model = sys.argv[1] if len(sys.argv) > 1 else ''
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    und_len = int(sys.argv[3]) if len(sys.argv) > 3 else 55
    gen_len = int(sys.argv[4]) if len(sys.argv) > 4 else 64
    tf, st = build_streamed_backward_transformer(model, device=DEV, n_gpu_buffers=1)
    tf.gradient_checkpointing = False
    print(f'[bench] {model.split(chr(92))[-1]} {len(tf.layers)} layers, und={und_len} gen={gen_len}', flush=True)
    one = _imgs(tf, 1, und_len, gen_len)
    many = _imgs(tf, N, und_len, gen_len)
    t1 = _bench(tf, st, one)
    tN = _bench(tf, st, many)
    print(f'[bench] single-image step : {t1 * 1000:7.1f} ms = {t1:.3f}s/img', flush=True)
    print(f'[bench] batched N={N} step : {tN * 1000:7.1f} ms = {tN / N:.3f}s/img  (speedup {t1 / (tN / N):.2f}x)', flush=True)
    C = max((tN - t1) / (N - 1), 1e-06)
    S = max(t1 - C, 1e-06)
    K = S / C
    free = max(1, round(K))
    print(f'[bench] split: stream/step ~= {S * 1000:.0f} ms, compute/img ~= {C * 1000:.0f} ms  => streaming:compute ~= {K:.1f}:1', flush=True)
    print(f"[bench] => ~{free} images hide under the stream (near-free); beyond that it's compute-bound. At N={N}: {t1 / (tN / N):.2f}x throughput.", flush=True)
    print('RESULT DONE', flush=True)
if __name__ == '__main__':
    main()
