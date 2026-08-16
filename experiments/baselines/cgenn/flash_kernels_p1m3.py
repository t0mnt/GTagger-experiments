"""FLASH PLAN v2, Step 3: Cl(1,3) fully-connected weighted-GP as a Triton custom op.

`cgenn_flash::fcgp(x, y, weight)`: x, y (B, N, 16) blade-minor contiguous, weight
(M, N, 35) in the repo's compact-path order -> out (B, M, 16). This is the fc-layer
contraction (the shipped hot path: phi_x/theta_x are FCGP in the `fc` layer type);
the gpmlp-only dim-2 layer is out of scope, stated.

Structure follows the two references by name:
- flash-kingdon's central trick (README): the GENERATED flat-arithmetic functions are
  valid Triton device functions as-is -- `triton.jit(flash_ref_p1m3._wgp_fwd)` makes
  the committed, 3e-16-gated reference THE kernel body. Zero transcription, so kernel
  math cannot drift from the gated math.
- flash-clifford's fc kernel shape (ops/fc_p3m0.py): one program per (row-block, m)
  for the forward with an n-loop accumulating in registers; explicit per-blade
  loads/stores. DELIBERATE DEPARTURE from that reference: their dL/dweight uses
  `tl.atomic_add` (nondeterministic float accumulation); here every (block, m, n)
  writes its own partial slot and a torch `.sum(0)` finishes the reduction in a fixed
  order -- determinism is a ship requirement (same bar SortedGather and 2.2b met).

The op registers a CPU composite (the reference wrappers) so every piece of wiring is
testable without a GPU, a fake for compile, and an autograd that routes each device to
its implementation. CUDA parity/determinism/benchmark gates live in
tests/experiments/test_flash_kernels_cuda.py (GPU gate day); CPU wiring gates in
tests/internal/test_flash_kernels_cpu.py.
"""

import torch

from experiments.baselines.cgenn import flash_ref_p1m3 as _ref

try:  # triton ships with CUDA torch builds; keep CPU-only environments importable
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover - exercised only on triton-less installs
    _HAS_TRITON = False

NB, NP = 16, 35

if _HAS_TRITON:
    _fwd_body = triton.jit(_ref._wgp_fwd)
    _grad_body = triton.jit(_ref._wgp_grad)

    @triton.jit
    def _fcgp_fwd_kernel(xp, yp, wp, op, B, N: tl.constexpr, M: tl.constexpr,
                         BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        m = tl.program_id(1)
        rows = pid * BLOCK + tl.arange(0, BLOCK)
        mask = rows < B
        o0 = tl.zeros([BLOCK], dtype=tl.float32)
        o1 = tl.zeros([BLOCK], dtype=tl.float32)
        o2 = tl.zeros([BLOCK], dtype=tl.float32)
        o3 = tl.zeros([BLOCK], dtype=tl.float32)
        o4 = tl.zeros([BLOCK], dtype=tl.float32)
        o5 = tl.zeros([BLOCK], dtype=tl.float32)
        o6 = tl.zeros([BLOCK], dtype=tl.float32)
        o7 = tl.zeros([BLOCK], dtype=tl.float32)
        o8 = tl.zeros([BLOCK], dtype=tl.float32)
        o9 = tl.zeros([BLOCK], dtype=tl.float32)
        o10 = tl.zeros([BLOCK], dtype=tl.float32)
        o11 = tl.zeros([BLOCK], dtype=tl.float32)
        o12 = tl.zeros([BLOCK], dtype=tl.float32)
        o13 = tl.zeros([BLOCK], dtype=tl.float32)
        o14 = tl.zeros([BLOCK], dtype=tl.float32)
        o15 = tl.zeros([BLOCK], dtype=tl.float32)
        for n in range(N):
            xb = xp + rows * (N * NB) + n * NB
            yb = yp + rows * (N * NB) + n * NB
            wb = wp + m * (N * NP) + n * NP
            x = tuple(tl.load(xb + i, mask=mask, other=0.0) for i in range(NB))
            y = tuple(tl.load(yb + i, mask=mask, other=0.0) for i in range(NB))
            w = tuple(tl.load(wb + i) for i in range(NP))
            o = _fwd_body(*x, *y, *w)
            o0 += o[0]; o1 += o[1]; o2 += o[2]; o3 += o[3]
            o4 += o[4]; o5 += o[5]; o6 += o[6]; o7 += o[7]
            o8 += o[8]; o9 += o[9]; o10 += o[10]; o11 += o[11]
            o12 += o[12]; o13 += o[13]; o14 += o[14]; o15 += o[15]
        ob = op + rows * (M * NB) + m * NB
        outs = (o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15)
        for j in tl.static_range(NB):
            tl.store(ob + j, outs[j], mask=mask)

    @triton.jit
    def _fcgp_bwd_kernel(xp, yp, wp, gp, gxp, gyp, pwp, B,
                         N: tl.constexpr, M: tl.constexpr, BLOCK: tl.constexpr):
        # one program per (row-block, n): accumulates gx/gy over m in registers and
        # writes this block's dL/dw partial to its OWN (block, m, n) slot -- no atomics.
        pid = tl.program_id(0)
        n = tl.program_id(1)
        rows = pid * BLOCK + tl.arange(0, BLOCK)
        mask = rows < B
        xb = xp + rows * (N * NB) + n * NB
        yb = yp + rows * (N * NB) + n * NB
        x = tuple(tl.load(xb + i, mask=mask, other=0.0) for i in range(NB))
        y = tuple(tl.load(yb + i, mask=mask, other=0.0) for i in range(NB))
        gx = tuple(tl.zeros([BLOCK], dtype=tl.float32) for _ in range(NB))
        gy = tuple(tl.zeros([BLOCK], dtype=tl.float32) for _ in range(NB))
        for m in range(M):
            wb = wp + m * (N * NP) + n * NP
            gb = gp + rows * (M * NB) + m * NB
            w = tuple(tl.load(wb + i) for i in range(NP))
            g = tuple(tl.load(gb + i, mask=mask, other=0.0) for i in range(NB))
            outs = _grad_body(*x, *y, *w, *g)
            gx = tuple(gx[i] + outs[i] for i in range(NB))
            gy = tuple(gy[i] + outs[NB + i] for i in range(NB))
            pb = pwp + pid * (M * N * NP) + m * (N * NP) + n * NP
            for p in tl.static_range(NP):
                tl.store(pb + p, tl.sum(outs[2 * NB + p], axis=0))
        gxb = gxp + rows * (N * NB) + n * NB
        gyb = gyp + rows * (N * NB) + n * NB
        for i in tl.static_range(NB):
            tl.store(gxb + i, gx[i], mask=mask)
            tl.store(gyb + i, gy[i], mask=mask)


def _reference_forward(x, y, weight):
    B, N = x.shape[0], x.shape[1]
    M = weight.shape[0]
    out = _ref.wgp(x[:, None], y[:, None], weight[None].expand(B, M, N, NP))
    return out.sum(dim=2)


def _reference_backward(x, y, weight, go):
    B, N = x.shape[0], x.shape[1]
    M = weight.shape[0]
    gx, gy, gw = _ref.wgp_grads(
        x[:, None].expand(B, M, N, NB),
        y[:, None].expand(B, M, N, NB),
        weight[None].expand(B, M, N, NP),
        go[:, :, None].expand(B, M, N, NB),
    )
    return gx.sum(dim=1), gy.sum(dim=1), gw.sum(dim=0)


@torch.library.custom_op("cgenn_flash::fcgp", mutates_args=())
def fcgp(x: torch.Tensor, y: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """out[b, m, j] = sum_{n, i} w[m, n, path(i, j)] * s(i, j) * x[b, n, i] * y[b, n, k(i, j)]"""
    B, N = x.shape[0], x.shape[1]
    M = weight.shape[0]
    if not x.is_cuda:
        return _reference_forward(x, y, weight)
    x, y, weight = x.contiguous(), y.contiguous(), weight.contiguous()
    out = x.new_empty(B, M, NB)
    BLOCK = 64
    grid = (triton.cdiv(B, BLOCK), M)
    _fcgp_fwd_kernel[grid](x, y, weight, out, B, N=N, M=M, BLOCK=BLOCK)
    return out


@fcgp.register_fake
def _(x, y, weight):
    return x.new_empty(x.shape[0], weight.shape[0], NB)


def _backward(ctx, go):
    x, y, weight = ctx.saved_tensors
    if not x.is_cuda:
        gx, gy, gw = _reference_backward(x, y, weight, go)
        return gx, gy, gw
    B, N = x.shape[0], x.shape[1]
    M = weight.shape[0]
    x, y, weight, go = (t.contiguous() for t in (x, y, weight, go))
    gx = torch.empty_like(x)
    gy = torch.empty_like(y)
    BLOCK = 32
    nblk = triton.cdiv(B, BLOCK)
    partial = x.new_empty(nblk, M, N, NP)
    grid = (nblk, N)
    _fcgp_bwd_kernel[grid](x, y, weight, go, gx, gy, partial, B, N=N, M=M, BLOCK=BLOCK)
    return gx, gy, partial.sum(dim=0)  # fixed-order stage-2: deterministic


def _setup_context(ctx, inputs, output):
    x, y, weight = inputs
    ctx.save_for_backward(x, y, weight)


fcgp.register_autograd(_backward, setup_context=_setup_context)
