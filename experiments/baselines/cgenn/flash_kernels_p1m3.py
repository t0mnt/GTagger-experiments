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
        # fully explicit loads/stores (flash-clifford ops/fc_p3m0.py style): this
        # Triton frontend has no tuple()/comprehension support inside @jit (GPU
        # round-trip #2 finding); only the tuple RETURN from the generated body is
        # relied on (flash-kingdon's proven pattern).
        pid = tl.program_id(0)
        m = tl.program_id(1)
        rows = pid * BLOCK + tl.arange(0, BLOCK)
        mask = rows < B
        o0 = tl.zeros([BLOCK], dtype=xp.dtype.element_ty)
        o1 = tl.zeros([BLOCK], dtype=xp.dtype.element_ty)
        o2 = tl.zeros([BLOCK], dtype=xp.dtype.element_ty)
        o3 = tl.zeros([BLOCK], dtype=xp.dtype.element_ty)
        o4 = tl.zeros([BLOCK], dtype=xp.dtype.element_ty)
        o5 = tl.zeros([BLOCK], dtype=xp.dtype.element_ty)
        o6 = tl.zeros([BLOCK], dtype=xp.dtype.element_ty)
        o7 = tl.zeros([BLOCK], dtype=xp.dtype.element_ty)
        o8 = tl.zeros([BLOCK], dtype=xp.dtype.element_ty)
        o9 = tl.zeros([BLOCK], dtype=xp.dtype.element_ty)
        o10 = tl.zeros([BLOCK], dtype=xp.dtype.element_ty)
        o11 = tl.zeros([BLOCK], dtype=xp.dtype.element_ty)
        o12 = tl.zeros([BLOCK], dtype=xp.dtype.element_ty)
        o13 = tl.zeros([BLOCK], dtype=xp.dtype.element_ty)
        o14 = tl.zeros([BLOCK], dtype=xp.dtype.element_ty)
        o15 = tl.zeros([BLOCK], dtype=xp.dtype.element_ty)
        for n in range(N):
            xb = xp + rows * (N * 16) + n * 16
            yb = yp + rows * (N * 16) + n * 16
            wb = wp + m * (N * 35) + n * 35
            x0 = tl.load(xb + 0, mask=mask, other=0.0)
            x1 = tl.load(xb + 1, mask=mask, other=0.0)
            x2 = tl.load(xb + 2, mask=mask, other=0.0)
            x3 = tl.load(xb + 3, mask=mask, other=0.0)
            x4 = tl.load(xb + 4, mask=mask, other=0.0)
            x5 = tl.load(xb + 5, mask=mask, other=0.0)
            x6 = tl.load(xb + 6, mask=mask, other=0.0)
            x7 = tl.load(xb + 7, mask=mask, other=0.0)
            x8 = tl.load(xb + 8, mask=mask, other=0.0)
            x9 = tl.load(xb + 9, mask=mask, other=0.0)
            x10 = tl.load(xb + 10, mask=mask, other=0.0)
            x11 = tl.load(xb + 11, mask=mask, other=0.0)
            x12 = tl.load(xb + 12, mask=mask, other=0.0)
            x13 = tl.load(xb + 13, mask=mask, other=0.0)
            x14 = tl.load(xb + 14, mask=mask, other=0.0)
            x15 = tl.load(xb + 15, mask=mask, other=0.0)
            y0 = tl.load(yb + 0, mask=mask, other=0.0)
            y1 = tl.load(yb + 1, mask=mask, other=0.0)
            y2 = tl.load(yb + 2, mask=mask, other=0.0)
            y3 = tl.load(yb + 3, mask=mask, other=0.0)
            y4 = tl.load(yb + 4, mask=mask, other=0.0)
            y5 = tl.load(yb + 5, mask=mask, other=0.0)
            y6 = tl.load(yb + 6, mask=mask, other=0.0)
            y7 = tl.load(yb + 7, mask=mask, other=0.0)
            y8 = tl.load(yb + 8, mask=mask, other=0.0)
            y9 = tl.load(yb + 9, mask=mask, other=0.0)
            y10 = tl.load(yb + 10, mask=mask, other=0.0)
            y11 = tl.load(yb + 11, mask=mask, other=0.0)
            y12 = tl.load(yb + 12, mask=mask, other=0.0)
            y13 = tl.load(yb + 13, mask=mask, other=0.0)
            y14 = tl.load(yb + 14, mask=mask, other=0.0)
            y15 = tl.load(yb + 15, mask=mask, other=0.0)
            w0 = tl.load(wb + 0)
            w1 = tl.load(wb + 1)
            w2 = tl.load(wb + 2)
            w3 = tl.load(wb + 3)
            w4 = tl.load(wb + 4)
            w5 = tl.load(wb + 5)
            w6 = tl.load(wb + 6)
            w7 = tl.load(wb + 7)
            w8 = tl.load(wb + 8)
            w9 = tl.load(wb + 9)
            w10 = tl.load(wb + 10)
            w11 = tl.load(wb + 11)
            w12 = tl.load(wb + 12)
            w13 = tl.load(wb + 13)
            w14 = tl.load(wb + 14)
            w15 = tl.load(wb + 15)
            w16 = tl.load(wb + 16)
            w17 = tl.load(wb + 17)
            w18 = tl.load(wb + 18)
            w19 = tl.load(wb + 19)
            w20 = tl.load(wb + 20)
            w21 = tl.load(wb + 21)
            w22 = tl.load(wb + 22)
            w23 = tl.load(wb + 23)
            w24 = tl.load(wb + 24)
            w25 = tl.load(wb + 25)
            w26 = tl.load(wb + 26)
            w27 = tl.load(wb + 27)
            w28 = tl.load(wb + 28)
            w29 = tl.load(wb + 29)
            w30 = tl.load(wb + 30)
            w31 = tl.load(wb + 31)
            w32 = tl.load(wb + 32)
            w33 = tl.load(wb + 33)
            w34 = tl.load(wb + 34)
            o = _fwd_body(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, w16, w17, w18, w19, w20, w21, w22, w23, w24, w25, w26, w27, w28, w29, w30, w31, w32, w33, w34)
            o0 += o[0]
            o1 += o[1]
            o2 += o[2]
            o3 += o[3]
            o4 += o[4]
            o5 += o[5]
            o6 += o[6]
            o7 += o[7]
            o8 += o[8]
            o9 += o[9]
            o10 += o[10]
            o11 += o[11]
            o12 += o[12]
            o13 += o[13]
            o14 += o[14]
            o15 += o[15]
        ob = op + rows * (M * 16) + m * 16
        tl.store(ob + 0, o0, mask=mask)
        tl.store(ob + 1, o1, mask=mask)
        tl.store(ob + 2, o2, mask=mask)
        tl.store(ob + 3, o3, mask=mask)
        tl.store(ob + 4, o4, mask=mask)
        tl.store(ob + 5, o5, mask=mask)
        tl.store(ob + 6, o6, mask=mask)
        tl.store(ob + 7, o7, mask=mask)
        tl.store(ob + 8, o8, mask=mask)
        tl.store(ob + 9, o9, mask=mask)
        tl.store(ob + 10, o10, mask=mask)
        tl.store(ob + 11, o11, mask=mask)
        tl.store(ob + 12, o12, mask=mask)
        tl.store(ob + 13, o13, mask=mask)
        tl.store(ob + 14, o14, mask=mask)
        tl.store(ob + 15, o15, mask=mask)

    @triton.jit
    def _fcgp_bwd_kernel(xp, yp, wp, gp, gxp, gyp, pwp, B,
                         N: tl.constexpr, M: tl.constexpr, BLOCK: tl.constexpr):
        # one program per (row-block, n): accumulates gx/gy over m in registers and
        # writes this block's dL/dw partial to its OWN (block, m, n) slot -- no atomics.
        pid = tl.program_id(0)
        n = tl.program_id(1)
        rows = pid * BLOCK + tl.arange(0, BLOCK)
        mask = rows < B
        xb = xp + rows * (N * 16) + n * 16
        yb = yp + rows * (N * 16) + n * 16
        x0 = tl.load(xb + 0, mask=mask, other=0.0)
        x1 = tl.load(xb + 1, mask=mask, other=0.0)
        x2 = tl.load(xb + 2, mask=mask, other=0.0)
        x3 = tl.load(xb + 3, mask=mask, other=0.0)
        x4 = tl.load(xb + 4, mask=mask, other=0.0)
        x5 = tl.load(xb + 5, mask=mask, other=0.0)
        x6 = tl.load(xb + 6, mask=mask, other=0.0)
        x7 = tl.load(xb + 7, mask=mask, other=0.0)
        x8 = tl.load(xb + 8, mask=mask, other=0.0)
        x9 = tl.load(xb + 9, mask=mask, other=0.0)
        x10 = tl.load(xb + 10, mask=mask, other=0.0)
        x11 = tl.load(xb + 11, mask=mask, other=0.0)
        x12 = tl.load(xb + 12, mask=mask, other=0.0)
        x13 = tl.load(xb + 13, mask=mask, other=0.0)
        x14 = tl.load(xb + 14, mask=mask, other=0.0)
        x15 = tl.load(xb + 15, mask=mask, other=0.0)
        y0 = tl.load(yb + 0, mask=mask, other=0.0)
        y1 = tl.load(yb + 1, mask=mask, other=0.0)
        y2 = tl.load(yb + 2, mask=mask, other=0.0)
        y3 = tl.load(yb + 3, mask=mask, other=0.0)
        y4 = tl.load(yb + 4, mask=mask, other=0.0)
        y5 = tl.load(yb + 5, mask=mask, other=0.0)
        y6 = tl.load(yb + 6, mask=mask, other=0.0)
        y7 = tl.load(yb + 7, mask=mask, other=0.0)
        y8 = tl.load(yb + 8, mask=mask, other=0.0)
        y9 = tl.load(yb + 9, mask=mask, other=0.0)
        y10 = tl.load(yb + 10, mask=mask, other=0.0)
        y11 = tl.load(yb + 11, mask=mask, other=0.0)
        y12 = tl.load(yb + 12, mask=mask, other=0.0)
        y13 = tl.load(yb + 13, mask=mask, other=0.0)
        y14 = tl.load(yb + 14, mask=mask, other=0.0)
        y15 = tl.load(yb + 15, mask=mask, other=0.0)
        gx0 = tl.zeros([BLOCK], dtype=xp.dtype.element_ty)
        gx1 = tl.zeros([BLOCK], dtype=xp.dtype.element_ty)
        gx2 = tl.zeros([BLOCK], dtype=xp.dtype.element_ty)
        gx3 = tl.zeros([BLOCK], dtype=xp.dtype.element_ty)
        gx4 = tl.zeros([BLOCK], dtype=xp.dtype.element_ty)
        gx5 = tl.zeros([BLOCK], dtype=xp.dtype.element_ty)
        gx6 = tl.zeros([BLOCK], dtype=xp.dtype.element_ty)
        gx7 = tl.zeros([BLOCK], dtype=xp.dtype.element_ty)
        gx8 = tl.zeros([BLOCK], dtype=xp.dtype.element_ty)
        gx9 = tl.zeros([BLOCK], dtype=xp.dtype.element_ty)
        gx10 = tl.zeros([BLOCK], dtype=xp.dtype.element_ty)
        gx11 = tl.zeros([BLOCK], dtype=xp.dtype.element_ty)
        gx12 = tl.zeros([BLOCK], dtype=xp.dtype.element_ty)
        gx13 = tl.zeros([BLOCK], dtype=xp.dtype.element_ty)
        gx14 = tl.zeros([BLOCK], dtype=xp.dtype.element_ty)
        gx15 = tl.zeros([BLOCK], dtype=xp.dtype.element_ty)
        gy0 = tl.zeros([BLOCK], dtype=xp.dtype.element_ty)
        gy1 = tl.zeros([BLOCK], dtype=xp.dtype.element_ty)
        gy2 = tl.zeros([BLOCK], dtype=xp.dtype.element_ty)
        gy3 = tl.zeros([BLOCK], dtype=xp.dtype.element_ty)
        gy4 = tl.zeros([BLOCK], dtype=xp.dtype.element_ty)
        gy5 = tl.zeros([BLOCK], dtype=xp.dtype.element_ty)
        gy6 = tl.zeros([BLOCK], dtype=xp.dtype.element_ty)
        gy7 = tl.zeros([BLOCK], dtype=xp.dtype.element_ty)
        gy8 = tl.zeros([BLOCK], dtype=xp.dtype.element_ty)
        gy9 = tl.zeros([BLOCK], dtype=xp.dtype.element_ty)
        gy10 = tl.zeros([BLOCK], dtype=xp.dtype.element_ty)
        gy11 = tl.zeros([BLOCK], dtype=xp.dtype.element_ty)
        gy12 = tl.zeros([BLOCK], dtype=xp.dtype.element_ty)
        gy13 = tl.zeros([BLOCK], dtype=xp.dtype.element_ty)
        gy14 = tl.zeros([BLOCK], dtype=xp.dtype.element_ty)
        gy15 = tl.zeros([BLOCK], dtype=xp.dtype.element_ty)
        for m in range(M):
            wb = wp + m * (N * 35) + n * 35
            gb = gp + rows * (M * 16) + m * 16
            w0 = tl.load(wb + 0)
            w1 = tl.load(wb + 1)
            w2 = tl.load(wb + 2)
            w3 = tl.load(wb + 3)
            w4 = tl.load(wb + 4)
            w5 = tl.load(wb + 5)
            w6 = tl.load(wb + 6)
            w7 = tl.load(wb + 7)
            w8 = tl.load(wb + 8)
            w9 = tl.load(wb + 9)
            w10 = tl.load(wb + 10)
            w11 = tl.load(wb + 11)
            w12 = tl.load(wb + 12)
            w13 = tl.load(wb + 13)
            w14 = tl.load(wb + 14)
            w15 = tl.load(wb + 15)
            w16 = tl.load(wb + 16)
            w17 = tl.load(wb + 17)
            w18 = tl.load(wb + 18)
            w19 = tl.load(wb + 19)
            w20 = tl.load(wb + 20)
            w21 = tl.load(wb + 21)
            w22 = tl.load(wb + 22)
            w23 = tl.load(wb + 23)
            w24 = tl.load(wb + 24)
            w25 = tl.load(wb + 25)
            w26 = tl.load(wb + 26)
            w27 = tl.load(wb + 27)
            w28 = tl.load(wb + 28)
            w29 = tl.load(wb + 29)
            w30 = tl.load(wb + 30)
            w31 = tl.load(wb + 31)
            w32 = tl.load(wb + 32)
            w33 = tl.load(wb + 33)
            w34 = tl.load(wb + 34)
            g0 = tl.load(gb + 0, mask=mask, other=0.0)
            g1 = tl.load(gb + 1, mask=mask, other=0.0)
            g2 = tl.load(gb + 2, mask=mask, other=0.0)
            g3 = tl.load(gb + 3, mask=mask, other=0.0)
            g4 = tl.load(gb + 4, mask=mask, other=0.0)
            g5 = tl.load(gb + 5, mask=mask, other=0.0)
            g6 = tl.load(gb + 6, mask=mask, other=0.0)
            g7 = tl.load(gb + 7, mask=mask, other=0.0)
            g8 = tl.load(gb + 8, mask=mask, other=0.0)
            g9 = tl.load(gb + 9, mask=mask, other=0.0)
            g10 = tl.load(gb + 10, mask=mask, other=0.0)
            g11 = tl.load(gb + 11, mask=mask, other=0.0)
            g12 = tl.load(gb + 12, mask=mask, other=0.0)
            g13 = tl.load(gb + 13, mask=mask, other=0.0)
            g14 = tl.load(gb + 14, mask=mask, other=0.0)
            g15 = tl.load(gb + 15, mask=mask, other=0.0)
            outs = _grad_body(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, w16, w17, w18, w19, w20, w21, w22, w23, w24, w25, w26, w27, w28, w29, w30, w31, w32, w33, w34, g0, g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11, g12, g13, g14, g15)
            gx0 += outs[0]
            gx1 += outs[1]
            gx2 += outs[2]
            gx3 += outs[3]
            gx4 += outs[4]
            gx5 += outs[5]
            gx6 += outs[6]
            gx7 += outs[7]
            gx8 += outs[8]
            gx9 += outs[9]
            gx10 += outs[10]
            gx11 += outs[11]
            gx12 += outs[12]
            gx13 += outs[13]
            gx14 += outs[14]
            gx15 += outs[15]
            gy0 += outs[16]
            gy1 += outs[17]
            gy2 += outs[18]
            gy3 += outs[19]
            gy4 += outs[20]
            gy5 += outs[21]
            gy6 += outs[22]
            gy7 += outs[23]
            gy8 += outs[24]
            gy9 += outs[25]
            gy10 += outs[26]
            gy11 += outs[27]
            gy12 += outs[28]
            gy13 += outs[29]
            gy14 += outs[30]
            gy15 += outs[31]
            pb = pwp + pid * (M * N * 35) + m * (N * 35) + n * 35
            tl.store(pb + 0, tl.sum(outs[32], axis=0))
            tl.store(pb + 1, tl.sum(outs[33], axis=0))
            tl.store(pb + 2, tl.sum(outs[34], axis=0))
            tl.store(pb + 3, tl.sum(outs[35], axis=0))
            tl.store(pb + 4, tl.sum(outs[36], axis=0))
            tl.store(pb + 5, tl.sum(outs[37], axis=0))
            tl.store(pb + 6, tl.sum(outs[38], axis=0))
            tl.store(pb + 7, tl.sum(outs[39], axis=0))
            tl.store(pb + 8, tl.sum(outs[40], axis=0))
            tl.store(pb + 9, tl.sum(outs[41], axis=0))
            tl.store(pb + 10, tl.sum(outs[42], axis=0))
            tl.store(pb + 11, tl.sum(outs[43], axis=0))
            tl.store(pb + 12, tl.sum(outs[44], axis=0))
            tl.store(pb + 13, tl.sum(outs[45], axis=0))
            tl.store(pb + 14, tl.sum(outs[46], axis=0))
            tl.store(pb + 15, tl.sum(outs[47], axis=0))
            tl.store(pb + 16, tl.sum(outs[48], axis=0))
            tl.store(pb + 17, tl.sum(outs[49], axis=0))
            tl.store(pb + 18, tl.sum(outs[50], axis=0))
            tl.store(pb + 19, tl.sum(outs[51], axis=0))
            tl.store(pb + 20, tl.sum(outs[52], axis=0))
            tl.store(pb + 21, tl.sum(outs[53], axis=0))
            tl.store(pb + 22, tl.sum(outs[54], axis=0))
            tl.store(pb + 23, tl.sum(outs[55], axis=0))
            tl.store(pb + 24, tl.sum(outs[56], axis=0))
            tl.store(pb + 25, tl.sum(outs[57], axis=0))
            tl.store(pb + 26, tl.sum(outs[58], axis=0))
            tl.store(pb + 27, tl.sum(outs[59], axis=0))
            tl.store(pb + 28, tl.sum(outs[60], axis=0))
            tl.store(pb + 29, tl.sum(outs[61], axis=0))
            tl.store(pb + 30, tl.sum(outs[62], axis=0))
            tl.store(pb + 31, tl.sum(outs[63], axis=0))
            tl.store(pb + 32, tl.sum(outs[64], axis=0))
            tl.store(pb + 33, tl.sum(outs[65], axis=0))
            tl.store(pb + 34, tl.sum(outs[66], axis=0))
        gxb = gxp + rows * (N * 16) + n * 16
        gyb = gyp + rows * (N * 16) + n * 16
        tl.store(gxb + 0, gx0, mask=mask)
        tl.store(gxb + 1, gx1, mask=mask)
        tl.store(gxb + 2, gx2, mask=mask)
        tl.store(gxb + 3, gx3, mask=mask)
        tl.store(gxb + 4, gx4, mask=mask)
        tl.store(gxb + 5, gx5, mask=mask)
        tl.store(gxb + 6, gx6, mask=mask)
        tl.store(gxb + 7, gx7, mask=mask)
        tl.store(gxb + 8, gx8, mask=mask)
        tl.store(gxb + 9, gx9, mask=mask)
        tl.store(gxb + 10, gx10, mask=mask)
        tl.store(gxb + 11, gx11, mask=mask)
        tl.store(gxb + 12, gx12, mask=mask)
        tl.store(gxb + 13, gx13, mask=mask)
        tl.store(gxb + 14, gx14, mask=mask)
        tl.store(gxb + 15, gx15, mask=mask)
        tl.store(gyb + 0, gy0, mask=mask)
        tl.store(gyb + 1, gy1, mask=mask)
        tl.store(gyb + 2, gy2, mask=mask)
        tl.store(gyb + 3, gy3, mask=mask)
        tl.store(gyb + 4, gy4, mask=mask)
        tl.store(gyb + 5, gy5, mask=mask)
        tl.store(gyb + 6, gy6, mask=mask)
        tl.store(gyb + 7, gy7, mask=mask)
        tl.store(gyb + 8, gy8, mask=mask)
        tl.store(gyb + 9, gy9, mask=mask)
        tl.store(gyb + 10, gy10, mask=mask)
        tl.store(gyb + 11, gy11, mask=mask)
        tl.store(gyb + 12, gy12, mask=mask)
        tl.store(gyb + 13, gy13, mask=mask)
        tl.store(gyb + 14, gy14, mask=mask)
        tl.store(gyb + 15, gy15, mask=mask)


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


@torch.library.custom_op("cgenn_flash::fcgp_bwd", mutates_args=())
def fcgp_bwd(x: torch.Tensor, y: torch.Tensor, weight: torch.Tensor,
             go: torch.Tensor) -> list[torch.Tensor]:
    """(gx, gy, gw) for fcgp. AN OPAQUE OP IN ITS OWN RIGHT, not a plain backward fn
    (GPU round-trip #4 finding): AOT's joint-graph trace runs the registered backward
    with fake/functional tensors, and a raw Triton launch there dies on data_ptr().
    Wrapping the launch in a custom op makes the backward trace as one opaque node --
    exactly what the PyTorch custom-op guidance prescribes -- while eager behavior is
    unchanged. CPU keeps the gated reference composition, so the whole autograd wiring
    stays testable without a GPU."""
    if not x.is_cuda:
        gx, gy, gw = _reference_backward(x, y, weight, go)
        return [gx, gy, gw]
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
    return [gx, gy, partial.sum(dim=0)]  # fixed-order stage-2: deterministic


@fcgp_bwd.register_fake
def _(x, y, weight, go):
    # contiguous_format: the real branches always return contiguous tensors, and a
    # fake that inherits exotic input strides would misdescribe them to AOT (audit)
    return [torch.empty_like(x, memory_format=torch.contiguous_format),
            torch.empty_like(y, memory_format=torch.contiguous_format),
            torch.empty_like(weight, memory_format=torch.contiguous_format)]


def _backward(ctx, go):
    x, y, weight = ctx.saved_tensors
    gx, gy, gw = fcgp_bwd(x, y, weight, go)
    return gx, gy, gw


def _setup_context(ctx, inputs, output):
    x, y, weight = inputs
    ctx.save_for_backward(x, y, weight)


fcgp.register_autograd(_backward, setup_context=_setup_context)
