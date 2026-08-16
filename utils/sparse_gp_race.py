"""Race the sparse-GP fc-contraction respellings on this device -- the LAST
measurement-gated decision of the CGENN program's in-campaign scope.

Run inside the container on the campaign card (torch-only, no repo imports):

    python utils/sparse_gp_race.py

Context (docs/cgenn-compile.md, gate-day rounds 2-3): after 2.2a+2.2b the H100
profile's remaining CGENN-side cost is the sparse-GP pair clone + reduce
(~17-18% of CUDA on tag_cgenn and CGENN-GPS) -- the einsum "bnij,mnij->bmj" in
sparse_gp_expression. Two respellings are TOL-verified (~1e-15 fp64,
scratchpad lab 2026-08-15) but were NOT adopted because, unlike 2.2b, they
carry no determinism win -- speed is the only motive, so the shipped GP only
changes if a measured race on the campaign device says so. This file is that
race. All three forms get identical inputs; timing is CUDA-event based,
fwd+bwd, after warmup.

Verdict rule printed at the end: adopt a challenger only if it beats the
einsum by >10% at BOTH shape points (GPS-kNN-sized and tag_cgenn-FC-sized);
otherwise the einsum stays and this question CLOSES for the campaign.
"""
import torch

# Campaign posture: the repo trains with matmul precision "highest" (no TF32), so
# the race must be run under the same setting -- an unpinned probe on NGC images
# (which default this to a TF32-enabled mode) hands the GEMM-shaped challengers a
# TF32 speedup the shipped model would never see, and the ~3e-04 rel-vs-einsum
# error of such a run is the tell. The 2026-08-15 first run had exactly that tell.
torch.set_float32_matmul_precision("highest")

nb = 16
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"torch {torch.__version__} | device {dev} | "
      f"float32_matmul_precision={torch.get_float32_matmul_precision()}"
      + ("" if dev == "cuda" else "  (CPU: numbers are NOT the decision input)"))


def cur3(pair, w):
    return torch.einsum("bnij,mnij->bmj", pair, w)


def c3a(pair, w):  # explicit permute+contiguous+bmm
    E, n = pair.shape[0], pair.shape[1]
    m = w.shape[0]
    pj = pair.permute(3, 0, 1, 2).reshape(nb, E, n * nb).contiguous()
    wj = w.permute(3, 1, 2, 0).reshape(nb, n * nb, m).contiguous()
    return torch.bmm(pj, wj).permute(1, 2, 0)


def c3b(pair, w):  # block-diagonal flat GEMM (16x FLOP padding, zero activation copies)
    E, n = pair.shape[0], pair.shape[1]
    m = w.shape[0]
    wperm = w.permute(1, 2, 3, 0)
    wbd = torch.diag_embed(wperm.movedim(2, -1))
    wbd = wbd.permute(0, 1, 3, 2, 4).reshape(n * nb * nb, m * nb)
    return (pair.reshape(E, n * nb * nb) @ wbd).view(E, m, nb)


def bench(fn, pair, w, reps=30):
    if dev == "cpu":
        reps = 2
    for _ in range(2 if dev == "cpu" else 5):
        out = fn(pair, w)
        out.sum().backward()
        pair.grad = w.grad = None
    if dev == "cuda":
        torch.cuda.synchronize()
        t0, t1 = torch.cuda.Event(True), torch.cuda.Event(True)
        t0.record()
    else:
        import time
        s = time.perf_counter()
    for _ in range(reps):
        out = fn(pair, w)
        out.sum().backward()
        pair.grad = w.grad = None
    if dev == "cuda":
        t1.record()
        torch.cuda.synchronize()
        return t0.elapsed_time(t1) / reps
    return (time.perf_counter() - s) / reps * 1e3


torch.manual_seed(0)
results = {}
# (E, n, m): GPS kNN edges at bs64 ~13k; tag_cgenn fully-connected at bs64 ~204k.
# n=11/19 are the production concat widths (census: 176=11*16, 304=19*16).
# On CPU shrink E and reps: the run is only a does-it-run + correctness smoke there.
SHAPES = (("GPS-kNN   E=13k n=11", 13000, 11, 11),
          ("cgenn-FC  E=200k n=11", 200000, 11, 11),
          ("cgenn-FC  E=200k n=19", 200000, 19, 11))
if dev == "cpu":
    SHAPES = tuple((lbl, E // 40, n, m) for lbl, E, n, m in SHAPES)
for label, E, n, m in SHAPES:
    pair = torch.randn(E, n, nb, nb, device=dev, requires_grad=True)
    w = torch.randn(m, n, nb, nb, device=dev, requires_grad=True)
    ref = cur3(pair.detach(), w.detach())
    row = {}
    for name, fn in (("einsum", cur3), ("ctg-bmm", c3a), ("blockdiag", c3b)):
        got = fn(pair.detach(), w.detach())
        rel = ((got - ref).abs().max() / ref.abs().max()).item()
        row[name] = bench(fn, pair, w)
        print(f"  {label}  {name:9s}: {row[name]:8.2f} ms/call  (rel vs einsum {rel:.1e})")
    results[label] = row
    del pair, w

print("\nVERDICT (adopt only if a challenger beats einsum by >10% at EVERY shape):")
for name in ("ctg-bmm", "blockdiag"):
    wins = all(r[name] < 0.9 * r["einsum"] for r in results.values())
    ratios = ", ".join(f"{r[name] / r['einsum']:.2f}x" for r in results.values())
    print(f"  {name:9s}: {'ADOPT-candidate' if wins else 'stays un-adopted'}  ({ratios} of einsum)")
