"""Sweep launch configs for the flash Cl(1,3) kernels and print the winning exports.

WHY (docs/cgenn-compile.md, round-trip #8): the flash-compiled profile put
_fcgp_bwd_kernel at 61.8% of the WHOLE training step (104 ms/call, 3.4x its fwd
twin) on launch configs hand-picked for correctness and never swept. With ~50
accumulators per thread, BLOCK/num_warps decide register pressure, and a bad
choice spills to local memory. A 2x on the bwd kernel alone is ~+45% model-level
-- the single largest lever left in the program.

WHAT IT DOES: times the fwd and bwd kernels standalone (CUDA events, warmup
excluded) over a (BLOCK, num_warps, num_stages) grid at a representative shape,
checks each candidate's outputs against the shipped config at fp64 before timing it
(a config that changes the math beyond reassociation is rejected, not raced),
and prints the two `export CGENN_FLASH_*_CFG=...` lines for the winners plus the
measured speedups. It changes nothing by itself: the exports feed the env hook
in flash_kernels_p1m3.py, and a tuned config ships only after the usual gates
(gw is TOL across bwd BLOCK changes -- the partial-buffer count changes -- while
gx/gy stay bit-equal; see _launch_cfg's docstring).

USAGE (GPU node):
    python utils/flash_tune.py                # default shape = the racing posture
    python utils/flash_tune.py --edges 250000 --n 20 --m 27
    python utils/flash_tune.py --grid-small   # quick 3x2x2 sanity sweep

The default shape mirrors tag_cgenn at the adopted posture: E edge rows at the
+5sd-probe scale for bs512, message channels N=20 (3*x + e), M=27. Override from
a profile if the campaign shape moves.
"""

import argparse
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--edges", type=int, default=3_500_000,
                    help="edge rows B for the probe tensors (default ~bs512 FC scale)")
    ap.add_argument("--n", type=int, default=20, help="input channels")
    ap.add_argument("--m", type=int, default=27, help="output channels")
    ap.add_argument("--iters", type=int, default=20, help="timed launches per config")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--grid-small", action="store_true",
                    help="3x2x2 sanity grid instead of the full sweep")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("flash_tune needs CUDA -- the kernels only launch there.")

    import triton
    from experiments.baselines.cgenn import flash_kernels_p1m3 as fk

    if args.grid_small:
        blocks, warps, stages = [32, 64, 128], [4, 8], [2, 3]
    else:
        blocks, warps, stages = [16, 32, 64, 128, 256], [2, 4, 8], [1, 2, 3, 4]

    torch.manual_seed(args.seed)
    dev = torch.device("cuda")
    B, N, M = args.edges, args.n, args.m
    x = torch.randn(B, N, fk.NB, device=dev)
    y = torch.randn(B, N, fk.NB, device=dev)
    w = torch.randn(M, N, fk.NP, device=dev)
    go = torch.randn(B, M, fk.NB, device=dev)
    print(f"[flash_tune] shape B={B} N={N} M={M}; grid "
          f"{len(blocks)}x{len(warps)}x{len(stages)} per kernel", flush=True)

    def time_launch(fn):
        for _ in range(args.warmup):
            fn()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(args.iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / args.iters

    # references from the SHIPPED configs (whatever env the process was launched
    # with -- normally unset =the defaults), fp64 for the parity check
    ref_out = fk.fcgp(x, y, w)
    ref_gx, ref_gy, ref_gw = fk.fcgp_bwd(x, y, w, go)
    ref64 = (ref_out.double(), ref_gx.double(), ref_gy.double(), ref_gw.double())

    def sweep(kernel_name, launch, check):
        rows = []
        for blk, wp, st in itertools.product(blocks, warps, stages):
            try:
                outs = launch(blk, wp, st)          # correctness first
            except Exception as e:                   # illegal config (e.g. OOR warps)
                rows.append((None, blk, wp, st, f"launch failed: {type(e).__name__}"))
                continue
            err = check(outs)
            if err > 1e-4:                           # reassociation-scale only
                rows.append((None, blk, wp, st, f"PARITY FAIL {err:.1e}"))
                continue
            ms = time_launch(lambda: launch(blk, wp, st))
            rows.append((ms, blk, wp, st, f"{ms:8.2f} ms  (parity {err:.1e})"))
        rows.sort(key=lambda r: (r[0] is None, r[0]))
        print(f"\n[flash_tune] {kernel_name}:")
        for ms, blk, wp, st, msg in rows[:8]:
            print(f"  BLOCK={blk:<4} warps={wp} stages={st}  {msg}")
        best = rows[0]
        return best if best[0] is not None else None

    def launch_fwd(blk, wp, st):
        out = x.new_empty(B, M, fk.NB)
        grid = (triton.cdiv(B, blk), M)
        fk._fcgp_fwd_kernel[grid](x, y, w, out, B, N=N, M=M, BLOCK=blk,
                                  num_warps=wp, num_stages=st)
        return (out,)

    def check_fwd(outs):
        return (outs[0].double() - ref64[0]).abs().max().item() / (
            1 + ref64[0].abs().max().item())

    def launch_bwd(blk, wp, st):
        gx, gy = torch.empty_like(x), torch.empty_like(y)
        nblk = triton.cdiv(B, blk)
        partial = x.new_empty(nblk, M, N, fk.NP)
        grid = (nblk, N)
        fk._fcgp_bwd_kernel[grid](x, y, w, go, gx, gy, partial, B, N=N, M=M,
                                  BLOCK=blk, num_warps=wp, num_stages=st)
        return gx, gy, partial.sum(dim=0)

    def check_bwd(outs):
        worst = 0.0
        for got, ref in zip(outs, ref64[1:]):
            worst = max(worst, (got.double() - ref).abs().max().item()
                        / (1 + ref.abs().max().item()))
        return worst

    base_fwd = time_launch(lambda: fk.fcgp(x, y, w))
    base_bwd = time_launch(lambda: fk.fcgp_bwd(x, y, w, go))
    print(f"[flash_tune] shipped configs: fwd {base_fwd:.2f} ms, bwd {base_bwd:.2f} ms")

    best_fwd = sweep("_fcgp_fwd_kernel", launch_fwd, check_fwd)
    best_bwd = sweep("_fcgp_bwd_kernel", launch_bwd, check_bwd)

    print("\n[flash_tune] TRANSCRIBE (then re-run the gate battery under these):")
    for name, best, base in (("CGENN_FLASH_FWD_CFG", best_fwd, base_fwd),
                             ("CGENN_FLASH_BWD_CFG", best_bwd, base_bwd)):
        if best is None:
            print(f"  {name}: no valid config beat the sweep -- keep unset")
            continue
        ms, blk, wp, st, _ = best
        print(f"  export {name}={blk},{wp},{st}   # {ms:.2f} ms vs shipped "
              f"{base:.2f} ms = {base / ms:.2f}x")


if __name__ == "__main__":
    main()
