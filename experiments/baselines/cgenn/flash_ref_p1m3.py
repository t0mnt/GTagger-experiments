"""Cl(1,3) weighted geometric product -- GENERATED, DO NOT EDIT.

Generator: utils/flash_gen.py (FLASH PLAN v2 step 2), terms sourced from kingdon
2.1.1 (`Algebra(1, 3)` blade products; MIT, arXiv:2503.10451) and asserted against
this repo's `CliffordAlgebra` cayley + `sparse_gp_tables` at generation time; the
expression assembly, differentiation and CSE run in sympy 1.14.0 (a stated deviation
from kingdon's own compile() printer pipeline -- sourcing terms from the product
algebra keeps kingdon the mathematical authority while the repo controls weight
order and emission style). Weight order = the repo's 35-entry compact-path order:
checkpoint-compatible with every sparse-GP weight tensor. Flat arithmetic bodies
(flash-clifford kernel style) so step 3 wraps them with `triton.jit` directly
(flash-kingdon's trick); the torch wrappers below are the CPU reference / parity
twin. Gates: tests/internal/test_flash_ref_p1m3.py.
"""
# generated-with: kingdon=2.1.1 sympy=1.14.0

import torch


def _wgp_fwd(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, w16, w17, w18, w19, w20, w21, w22, w23, w24, w25, w26, w27, w28, w29, w30, w31, w32, w33, w34):
    _f0 = w14*x5
    _f1 = w14*y3
    _f2 = w14*y4
    _f3 = w15*x10
    _f4 = w15*y11
    _f5 = w15*y12
    _f6 = w23*x11
    _f7 = w23*x12
    _f8 = w23*y10
    _f9 = w24*y15
    _f10 = w1*x0
    _f11 = w14*y2
    _f12 = w15*y14
    _f13 = w23*x14
    _f14 = w6*y0
    _f15 = w7*x1
    _f16 = w14*y1
    _f17 = w15*y13
    _f18 = w23*x13
    _f19 = w31*x15
    _f20 = w7*x2
    _f21 = w17*x8
    _f22 = w17*y7
    _f23 = w18*y15
    _f24 = w25*x11
    _f25 = w25*y4
    _f26 = w26*y14
    _f27 = w32*x15
    _f28 = w8*y1
    _f29 = w9*y11
    _f30 = w9*x4
    _f31 = w16*y0
    _f32 = w17*x7
    _f33 = w2*x0
    _f34 = w8*x1
    _f35 = w17*y8
    _f36 = w26*x14
    _f37 = w17*x10
    _f38 = w17*y5
    _f39 = w25*x12
    _f40 = w25*y3
    _f41 = w9*y12
    _f42 = w9*x3
    _f43 = w17*x5
    _f44 = w17*y10
    _f45 = w26*y13
    _f46 = w8*x2
    _f47 = w26*x13
    _f48 = w8*y2
    _f49 = w10*x1
    _f50 = w10*y5
    _f51 = w11*y15
    _f52 = w19*x5
    _f53 = w19*y1
    _f54 = w20*y13
    _f55 = w27*y0
    _f56 = w28*y10
    _f57 = w28*x14
    _f58 = w3*x0
    _f59 = w10*x2
    _f60 = w19*y2
    _f61 = w20*x10
    _f62 = w20*y14
    _f63 = w28*x13
    _f64 = w33*x15
    _f65 = w10*x4
    _f66 = w19*y4
    _f67 = w20*y12
    _f68 = w28*x11
    _f69 = w10*x3
    _f70 = w19*y3
    _f71 = w20*y11
    _f72 = w28*x12
    return (
        w0*x0*y0 - w13*x10*y10 + w13*x5*y5 + w13*x6*y6 + w13*x7*y7 - w13*x8*y8 - w13*x9*y9 - w22*x11*y11 - w22*x12*y12 - w22*x13*y13 + w22*x14*y14 - w30*x15*y15 + w5*x1*y1 - w5*x2*y2 - w5*x3*y3 - w5*x4*y4,
        -_f0*y2 - _f1*x6 - _f2*x7 - _f3*y13 - _f4*x8 - _f5*x9 - _f6*y8 - _f7*y9 - _f8*x13 - _f9*x14 + w1*x0*y1 + w31*x15*y14 + w6*x1*y0 + w7*x2*y5 + w7*x3*y6 + w7*x4*y7,
        -_f0*y1 - _f1*x8 - _f2*x9 - _f3*y14 - _f4*x6 - _f5*x7 - _f6*y6 - _f7*y7 - _f8*x14 - _f9*x13 + w1*x0*y2 + w31*x15*y13 + w6*x2*y0 + w7*x1*y5 + w7*x3*y8 + w7*x4*y9,
        _f10*y3 + _f11*x8 + _f12*x9 + _f13*y9 + _f14*x3 + _f15*y6 - _f16*x6 - _f17*x7 - _f18*y7 - _f19*y12 - _f2*x10 - _f20*y8 + _f4*x5 + _f6*y5 + _f9*x12 + w7*x4*y10,
        _f1*x10 + _f10*y4 + _f11*x9 - _f12*x8 - _f13*y8 + _f14*x4 + _f15*y7 - _f16*x7 + _f17*x6 + _f18*y6 + _f19*y11 - _f20*y9 + _f5*x5 + _f7*y5 - _f9*x11 - w7*x3*y10,
        -_f21*y6 - _f22*x9 - _f23*x10 - _f24*y3 - _f25*x12 - _f26*x13 - _f27*y10 - _f28*x2 - _f29*x3 - _f30*y12 + w16*x5*y0 + w17*x6*y8 + w17*x7*y9 + w2*x0*y5 + w26*x14*y13 + w8*x1*y2,
        _f21*y5 - _f22*x10 + _f23*x9 + _f24*y2 - _f25*x13 + _f26*x12 + _f27*y9 - _f28*x3 + _f29*x2 - _f30*y13 + _f31*x6 + _f32*y10 + _f33*y6 + _f34*y3 - _f35*x5 - _f36*y12,
        -_f23*x8 - _f26*x11 - _f27*y8 - _f28*x4 + _f31*x7 + _f33*y7 + _f34*y4 + _f36*y11 + _f37*y6 + _f38*x9 + _f39*y2 + _f40*x13 + _f41*x2 + _f42*y13 - _f43*y9 - _f44*x6,
        _f23*x7 + _f24*y1 - _f25*x14 + _f27*y7 + _f29*x1 - _f30*y14 + _f31*x8 + _f33*y8 - _f37*y9 + _f38*x6 - _f43*y6 + _f44*x9 + _f45*x12 + _f46*y3 - _f47*y12 - _f48*x3,
        -_f21*y10 - _f22*x5 - _f23*x6 - _f27*y6 + _f31*x9 + _f32*y5 + _f33*y9 + _f35*x10 + _f39*y1 + _f40*x14 + _f41*x1 + _f42*y14 - _f45*x11 + _f46*y4 + _f47*y11 - _f48*x4,
        _f21*y9 - _f22*x6 + _f23*x5 + _f27*y5 + _f31*x10 + _f32*y6 + _f33*y10 - _f35*x9 + w25*x13*y1 - w25*x14*y2 + w26*x11*y12 - w26*x12*y11 + w8*x3*y4 - w8*x4*y3 + w9*x1*y13 - w9*x2*y14,
        _f49*y8 + _f50*x3 + _f51*x4 + _f52*y3 + _f53*x8 + _f54*x9 + _f55*x11 + _f56*x12 + _f57*y7 + _f58*y11 - _f59*y6 - _f60*x6 - _f61*y12 - _f62*x7 - _f63*y9 - _f64*y4,
        _f49*y9 + _f50*x4 - _f51*x3 + _f52*y4 + _f53*x9 - _f54*x8 + _f55*x12 - _f56*x11 - _f57*y6 + _f58*y12 - _f59*y7 - _f60*x7 + _f61*y11 + _f62*x6 + _f63*y8 + _f64*y3,
        _f49*y10 + _f51*x2 + _f53*x10 + _f55*x13 + _f57*y5 + _f58*y13 - _f62*x5 - _f64*y2 + _f65*y6 + _f66*x6 + _f67*x8 + _f68*y9 - _f69*y7 - _f70*x7 - _f71*x9 - _f72*y8,
        _f51*x1 - _f54*x5 + _f55*x14 + _f58*y14 + _f59*y10 + _f60*x10 + _f63*y5 - _f64*y1 + _f65*y8 + _f66*x8 + _f67*x6 + _f68*y7 - _f69*y9 - _f70*x9 - _f71*x7 - _f72*y6,
        w12*x1*y14 - w12*x2*y13 + w12*x3*y12 - w12*x4*y11 + w21*x10*y5 + w21*x5*y10 - w21*x6*y9 + w21*x7*y8 + w21*x8*y7 - w21*x9*y6 + w29*x11*y4 - w29*x12*y3 + w29*x13*y2 - w29*x14*y1 + w34*x15*y0 + w4*x0*y15,
    )

def _wgp_grad(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, w16, w17, w18, w19, w20, w21, w22, w23, w24, w25, w26, w27, w28, w29, w30, w31, w32, w33, w34, g0, g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11, g12, g13, g14, g15):
    _b0 = g0*w0
    _b1 = g1*w1
    _b2 = g14*w3
    _b3 = g5*w2
    _b4 = g6*w2
    _b5 = g7*w2
    _b6 = g0*w5
    _b7 = g10*w9
    _b8 = g11*w10
    _b9 = g12*w10
    _b10 = w11*y15
    _b11 = g15*w12
    _b12 = g3*w7
    _b13 = g4*w7
    _b14 = g5*w8
    _b15 = w8*y3
    _b16 = w9*y11
    _b17 = w9*y12
    _b18 = g13*w10
    _b19 = g14*w10
    _b20 = w8*y1
    _b21 = w8*y2
    _b22 = g0*w13
    _b23 = w18*y15
    _b24 = g11*w19
    _b25 = w19*y4
    _b26 = g15*w21
    _b27 = w15*y11
    _b28 = g4*w15
    _b29 = w16*y0
    _b30 = g1*w14
    _b31 = w20*y14
    _b32 = g14*w20
    _b33 = w14*y1
    _b34 = w17*y8
    _b35 = g7*w17
    _b36 = g8*w17
    _b37 = w17*y7
    _b38 = g10*w17
    _b39 = g5*w17
    _b40 = g6*w17
    _b41 = g9*w17
    _b42 = g12*w19
    _b43 = w19*y3
    _b44 = w15*y12
    _b45 = g3*w15
    _b46 = g2*w14
    _b47 = g13*w20
    _b48 = g1*w15
    _b49 = g11*w20
    _b50 = g2*w15
    _b51 = g3*w14
    _b52 = g0*w22
    _b53 = g1*w23
    _b54 = g2*w23
    _b55 = w24*y15
    _b56 = g5*w25
    _b57 = w26*y14
    _b58 = w26*y13
    _b59 = g10*w26
    _b60 = g13*w28
    _b61 = g14*w28
    _b62 = g15*w29
    _b63 = g11*w28
    _b64 = g3*w23
    _b65 = w25*y4
    _b66 = w26*y12
    _b67 = g12*w28
    _b68 = g4*w23
    _b69 = g0*w30
    _b70 = g11*w33
    _b71 = g13*w33
    _b72 = g14*w33
    _b73 = g3*w31
    _b74 = g5*w32
    _b75 = g7*w32
    _b76 = g9*w32
    _b77 = g1*x1
    _b78 = g10*x10
    _b79 = g11*x11
    _b80 = g12*x12
    _b81 = g13*x13
    _b82 = g14*x14
    _b83 = g2*x2
    _b84 = g3*x3
    _b85 = g4*x4
    _b86 = g5*x5
    _b87 = g6*x6
    _b88 = g7*x7
    _b89 = g8*x8
    _b90 = g9*x9
    _b91 = g10*w25
    _b92 = g13*w19
    _b93 = w8*x3
    _b94 = w8*x4
    _b95 = g1*w7
    _b96 = g5*w9
    _b97 = w26*x13
    _b98 = g12*w20
    _b99 = w9*x4
    _b100 = w26*x11
    _b101 = g1*x14
    _b102 = g12*x3
    _b103 = g2*x13
    _b104 = g4*x11
    _b105 = g5*x10
    _b106 = g7*x8
    _b107 = g9*x6
    return (
        _b0*y0 + _b1*y1 + _b2*y14 + _b3*y5 + _b4*y6 + _b5*y7 + g10*w2*y10 + g11*w3*y11 + g12*w3*y12 + g13*w3*y13 + g15*w4*y15 + g2*w1*y2 + g3*w1*y3 + g4*w1*y4 + g8*w2*y8 + g9*w2*y9,
        _b10*g14 + _b11*y14 + _b12*y6 + _b13*y7 + _b14*y2 + _b15*g6 + _b16*g8 + _b17*g9 + _b6*y1 + _b7*y13 + _b8*y8 + _b9*y9 + g1*w6*y0 + g13*w10*y10 + g2*w7*y5 + g7*w8*y4,
        -_b11*y13 - _b12*y8 - _b13*y9 - _b14*y1 - _b6*y2 - _b7*y14 - _b8*y6 - _b9*y7 + g1*w7*y5 + g13*w11*y15 + g14*w10*y10 + g2*w6*y0 + g6*w9*y11 + g7*w9*y12 + g8*w8*y3 + g9*w8*y4,
        -_b10*g12 - _b13*y10 - _b16*g5 - _b18*y7 - _b19*y9 - _b20*g6 - _b21*g8 - _b6*y3 + g1*w7*y6 + g10*w8*y4 + g11*w10*y5 + g15*w12*y12 + g2*w7*y8 + g3*w6*y0 + g7*w9*y13 + g9*w9*y14,
        -_b11*y11 - _b15*g10 - _b17*g5 - _b20*g7 - _b21*g9 - _b6*y4 + g1*w7*y7 + g11*w11*y15 + g12*w10*y5 + g13*w10*y6 + g14*w10*y8 + g2*w7*y9 + g3*w7*y10 + g4*w6*y0 - g6*w9*y13 - g8*w9*y14,
        _b22*y5 + _b23*g10 + _b24*y3 + _b25*g12 + _b26*y10 + _b27*g3 + _b28*y12 + _b29*g5 - _b30*y2 - _b31*g13 - _b32*y13 - _b33*g2 - _b34*g6 - _b35*y9 - _b36*y6 - _b37*g9,
        _b22*y6 - _b23*g9 - _b24*y2 + _b25*g13 - _b26*y9 - _b27*g2 + _b28*y13 + _b29*g6 - _b30*y3 + _b31*g12 + _b32*y12 - _b33*g3 + _b34*g5 - _b35*y10 + _b36*y5 - _b37*g10,
        _b22*y7 + _b23*g8 + _b26*y8 + _b29*g7 - _b30*y4 - _b31*g11 - _b32*y11 - _b33*g4 + _b38*y6 + _b39*y9 + _b40*y10 + _b41*y5 - _b42*y2 - _b43*g13 - _b44*g2 - _b45*y13,
        -_b22*y8 - _b23*g7 - _b27*g1 - _b28*y14 - _b39*y6 - _b41*y10 - _b46*y3 + g10*w17*y9 + g11*w19*y1 - g12*w20*y13 + g13*w20*y12 + g14*w19*y4 + g15*w21*y7 + g3*w14*y2 + g6*w17*y5 + g8*w16*y0,
        -_b22*y9 - _b26*y6 - _b34*g10 - _b37*g5 - _b43*g14 - _b44*g1 - _b46*y4 - _b47*y11 + g11*w20*y13 + g12*w19*y1 + g3*w15*y14 + g4*w14*y2 + g6*w18*y15 + g7*w17*y5 + g8*w17*y10 + g9*w16*y0,
        -_b22*y10 - _b23*g5 - _b36*y9 - _b37*g6 - _b48*y13 - _b49*y12 - _b50*y14 - _b51*y4 + g10*w16*y0 + g12*w20*y11 + g13*w19*y1 + g14*w19*y2 + g15*w21*y5 + g4*w14*y3 + g7*w17*y6 + g9*w17*y8,
        -_b52*y11 - _b53*y8 - _b54*y6 - _b55*g4 - _b56*y3 - _b57*g7 - _b58*g9 + g10*w26*y12 + g11*w27*y0 - g12*w28*y10 + g13*w28*y9 + g14*w28*y7 + g15*w29*y4 + g3*w23*y5 + g6*w25*y2 + g8*w25*y1,
        -_b52*y12 - _b53*y9 - _b54*y7 - _b56*y4 - _b59*y11 - _b60*y8 - _b61*y6 - _b62*y3 + g11*w28*y10 + g12*w27*y0 + g3*w24*y15 + g4*w23*y5 + g6*w26*y14 + g7*w25*y2 + g8*w26*y13 + g9*w25*y1,
        -_b52*y13 - _b53*y10 - _b55*g2 - _b57*g5 - _b63*y9 - _b64*y7 - _b65*g6 - _b66*g8 + g10*w25*y1 + g12*w28*y8 + g13*w27*y0 + g14*w28*y5 + g15*w29*y2 + g4*w23*y6 + g7*w25*y3 + g9*w26*y11,
        _b52*y14 - _b54*y10 - _b55*g1 + _b58*g5 + _b60*y5 - _b62*y1 + _b63*y7 + _b64*y9 - _b65*g8 - _b66*g6 - _b67*y6 - _b68*y8 - g10*w25*y2 + g14*w27*y0 + g7*w26*y11 + g9*w25*y3,
        -_b69*y15 - _b70*y4 - _b71*y2 - _b72*y1 - _b73*y12 - _b74*y10 - _b75*y8 - _b76*y6 + g1*w31*y14 + g10*w32*y5 + g12*w33*y3 + g15*w34*y0 + g2*w31*y13 + g4*w31*y11 + g6*w32*y9 + g8*w32*y7,
        _b0*x0 + _b77*w6 + _b78*w16 + _b79*w27 + _b80*w27 + _b81*w27 + _b82*w27 + _b83*w6 + _b84*w6 + _b85*w6 + _b86*w16 + _b87*w16 + _b88*w16 + _b89*w16 + _b90*w16 + g15*w34*x15,
        _b1*x0 - _b14*x2 + _b24*x8 + _b42*x9 - _b46*x5 - _b51*x6 + _b6*x1 - _b62*x14 - _b72*x15 + _b91*x13 + _b92*x10 - _b93*g6 - _b94*g7 - g4*w14*x7 + g8*w25*x11 + g9*w25*x12,
        -_b24*x6 - _b30*x5 - _b42*x7 - _b6*x2 - _b71*x15 - _b91*x14 - _b93*g8 - _b94*g9 + g14*w19*x10 + g15*w29*x13 + g2*w1*x0 + g3*w14*x8 + g4*w14*x9 + g5*w8*x1 + g6*w25*x11 + g7*w25*x12,
        -_b30*x6 - _b46*x8 - _b56*x11 - _b6*x3 - _b62*x12 - _b92*x7 - _b94*g10 + g11*w19*x5 + g12*w33*x15 - g14*w19*x9 + g3*w1*x0 + g4*w14*x10 + g6*w8*x1 + g7*w25*x13 + g8*w8*x2 + g9*w25*x14,
        -_b30*x7 - _b46*x9 - _b51*x10 - _b56*x12 - _b6*x4 - _b70*x15 + g10*w8*x3 + g12*w19*x5 + g13*w19*x6 + g14*w19*x8 + g15*w29*x11 + g4*w1*x0 - g6*w25*x13 + g7*w8*x1 - g8*w25*x14 + g9*w8*x2,
        _b22*x5 + _b26*x10 + _b3*x0 + _b35*x9 + _b36*x6 + _b40*x8 + _b41*x7 + _b60*x14 + _b61*x13 + _b64*x11 + _b68*x12 + _b8*x3 + _b9*x4 + _b95*x2 + g10*w32*x15 + g2*w7*x1,
        _b12*x1 + _b18*x4 + _b22*x6 - _b26*x9 + _b35*x10 - _b36*x5 + _b38*x7 - _b39*x8 + _b4*x0 - _b54*x11 - _b61*x12 - _b67*x14 + _b68*x13 - _b76*x15 - _b8*x2 + _b95*x3,
        _b13*x1 - _b18*x3 + _b22*x7 + _b26*x8 - _b38*x6 - _b39*x9 - _b40*x10 - _b41*x5 + _b5*x0 - _b54*x12 + _b61*x11 + _b63*x14 - _b64*x13 - _b9*x2 + _b95*x4 + g8*w32*x15,
        -_b12*x2 - _b22*x8 - _b38*x9 - _b40*x5 - _b53*x11 - _b60*x12 - _b68*x14 - _b75*x15 + g11*w10*x1 + g12*w28*x13 + g14*w10*x4 + g15*w21*x7 + g2*w7*x3 + g5*w17*x6 + g8*w2*x0 + g9*w17*x10,
        -_b13*x2 - _b19*x3 - _b22*x9 - _b26*x6 - _b35*x5 - _b36*x10 - _b53*x12 - _b63*x13 + g10*w17*x8 + g12*w10*x1 + g13*w28*x11 + g2*w7*x4 + g3*w23*x14 + g5*w17*x7 + g6*w32*x15 + g9*w2*x0,
        -_b13*x3 - _b22*x10 - _b35*x6 - _b41*x8 - _b53*x13 - _b54*x14 - _b67*x11 - _b74*x15 + g10*w2*x0 + g11*w28*x12 + g13*w10*x1 + g14*w10*x2 + g15*w21*x5 + g3*w7*x4 + g6*w17*x7 + g8*w17*x9,
        -_b11*x4 - _b32*x7 - _b47*x9 - _b48*x8 - _b50*x6 - _b52*x11 - _b59*x12 - _b96*x3 + g11*w3*x0 + g12*w20*x10 + g3*w15*x5 + g4*w31*x15 + g6*w9*x2 + g7*w26*x14 + g8*w9*x1 + g9*w26*x13,
        -_b48*x9 - _b49*x10 - _b50*x7 - _b52*x12 - _b73*x15 - _b96*x4 - _b97*g8 + g10*w26*x11 + g12*w3*x0 + g13*w20*x8 + g14*w20*x6 + g15*w12*x3 + g4*w15*x5 - g6*w26*x14 + g7*w9*x2 + g9*w9*x1,
        -_b100*g9 - _b11*x2 - _b32*x5 - _b45*x7 - _b48*x10 - _b52*x13 - _b98*x8 - _b99*g6 + g10*w9*x1 + g11*w20*x9 + g13*w3*x0 + g2*w31*x15 + g4*w15*x6 + g5*w26*x14 + g7*w9*x3 + g8*w26*x12,
        -_b100*g7 + _b11*x1 + _b2*x0 - _b28*x8 + _b45*x9 - _b47*x5 - _b49*x7 - _b50*x10 + _b52*x14 - _b7*x2 - _b97*g5 + _b98*x6 - _b99*g8 + g1*w31*x15 + g6*w26*x12 + g9*w9*x3,
        -_b101*w24 - _b102*w11 - _b103*w24 - _b104*w24 - _b105*w18 - _b106*w18 - _b107*w18 - _b69*x15 + g10*w18*x5 + g11*w11*x4 + g13*w11*x2 + g14*w11*x1 + g15*w4*x0 + g3*w24*x12 + g6*w18*x9 + g8*w18*x7,
        g0*x0*y0,
        g1*x0*y1 + g2*x0*y2 + g3*x0*y3 + g4*x0*y4,
        g10*x0*y10 + g5*x0*y5 + g6*x0*y6 + g7*x0*y7 + g8*x0*y8 + g9*x0*y9,
        g11*x0*y11 + g12*x0*y12 + g13*x0*y13 + g14*x0*y14,
        g15*x0*y15,
        g0*(x1*y1 - x2*y2 - x3*y3 - x4*y4),
        _b77*y0 + _b83*y0 + _b84*y0 + _b85*y0,
        g1*(x2*y5 + x3*y6 + x4*y7) + g2*(x1*y5 + x3*y8 + x4*y9) + g3*(x1*y6 - x2*y8 + x4*y10) + g4*(x1*y7 - x2*y9 - x3*y10),
        g10*(x3*y4 - x4*y3) + g5*(x1*y2 - x2*y1) + g6*(x1*y3 - x3*y1) + g7*(x1*y4 - x4*y1) + g8*(x2*y3 - x3*y2) + g9*(x2*y4 - x4*y2),
        g10*(x1*y13 - x2*y14) + g5*(-x3*y11 - x4*y12) + g6*(x2*y11 - x4*y13) + g7*(x2*y12 + x3*y13) + g8*(x1*y11 - x4*y14) + g9*(x1*y12 + x3*y14),
        g11*(x1*y8 - x2*y6 + x3*y5) + g12*(x1*y9 - x2*y7 + x4*y5) + g13*(x1*y10 - x3*y7 + x4*y6) + g14*(x2*y10 - x3*y9 + x4*y8),
        -_b102*y15 + g11*x4*y15 + g13*x2*y15 + g14*x1*y15,
        g15*(x1*y14 - x2*y13 + x3*y12 - x4*y11),
        g0*(-x10*y10 + x5*y5 + x6*y6 + x7*y7 - x8*y8 - x9*y9),
        g1*(-x5*y2 - x6*y3 - x7*y4) + g2*(-x5*y1 - x8*y3 - x9*y4) + g3*(-x10*y4 - x6*y1 + x8*y2) + g4*(x10*y3 - x7*y1 + x9*y2),
        g1*(-x10*y13 - x8*y11 - x9*y12) + g2*(-x10*y14 - x6*y11 - x7*y12) + g3*(x5*y11 - x7*y13 + x9*y14) + g4*(x5*y12 + x6*y13 - x8*y14),
        _b78*y0 + _b86*y0 + _b87*y0 + _b88*y0 + _b89*y0 + _b90*y0,
        g10*(-x6*y7 + x7*y6 + x8*y9 - x9*y8) + g5*(x6*y8 + x7*y9 - x8*y6 - x9*y7) + g6*(-x10*y7 - x5*y8 + x7*y10 + x8*y5) + g7*(x10*y6 - x5*y9 - x6*y10 + x9*y5) + g8*(-x10*y9 - x5*y6 + x6*y5 + x9*y10) + g9*(x10*y8 - x5*y7 + x7*y5 - x8*y10),
        -_b105*y15 - _b106*y15 - _b107*y15 + g10*x5*y15 + g6*x9*y15 + g8*x7*y15,
        g11*(x5*y3 - x6*y2 + x8*y1) + g12*(x5*y4 - x7*y2 + x9*y1) + g13*(x10*y1 + x6*y4 - x7*y3) + g14*(x10*y2 + x8*y4 - x9*y3),
        g11*(-x10*y12 - x7*y14 + x9*y13) + g12*(x10*y11 + x6*y14 - x8*y13) + g13*(-x5*y14 + x8*y12 - x9*y11) + g14*(-x5*y13 + x6*y12 - x7*y11),
        g15*(x10*y5 + x5*y10 - x6*y9 + x7*y8 + x8*y7 - x9*y6),
        g0*(-x11*y11 - x12*y12 - x13*y13 + x14*y14),
        g1*(-x11*y8 - x12*y9 - x13*y10) + g2*(-x11*y6 - x12*y7 - x14*y10) + g3*(x11*y5 - x13*y7 + x14*y9) + g4*(x12*y5 + x13*y6 - x14*y8),
        -_b101*y15 - _b103*y15 - _b104*y15 + g3*x12*y15,
        g10*(x13*y1 - x14*y2) + g5*(-x11*y3 - x12*y4) + g6*(x11*y2 - x13*y4) + g7*(x12*y2 + x13*y3) + g8*(x11*y1 - x14*y4) + g9*(x12*y1 + x14*y3),
        g10*(x11*y12 - x12*y11) + g5*(-x13*y14 + x14*y13) + g6*(x12*y14 - x14*y12) + g7*(-x11*y14 + x14*y11) + g8*(x12*y13 - x13*y12) + g9*(-x11*y13 + x13*y11),
        _b79*y0 + _b80*y0 + _b81*y0 + _b82*y0,
        g11*(x12*y10 - x13*y9 + x14*y7) + g12*(-x11*y10 + x13*y8 - x14*y6) + g13*(x11*y9 - x12*y8 + x14*y5) + g14*(x11*y7 - x12*y6 + x13*y5),
        g15*(x11*y4 - x12*y3 + x13*y2 - x14*y1),
        -g0*x15*y15,
        g1*x15*y14 + g2*x15*y13 - g3*x15*y12 + g4*x15*y11,
        g10*x15*y5 - g5*x15*y10 + g6*x15*y9 - g7*x15*y8 + g8*x15*y7 - g9*x15*y6,
        -g11*x15*y4 + g12*x15*y3 - g13*x15*y2 - g14*x15*y1,
        g15*x15*y0,
    )

def wgp(x, y, w):
    """Elementwise weighted GP: (..., 16) x (..., 16) x (..., 35) -> (..., 16).
    Broadcasting over leading dims follows torch semantics (the fc layer is
    wgp(x[:, None], y[:, None], w[None]) summed over the input-feature axis)."""
    o = _wgp_fwd(*x.unbind(-1), *y.unbind(-1), *w.unbind(-1))
    return torch.stack(torch.broadcast_tensors(*o), dim=-1)


def wgp_grads(x, y, w, go):
    """Generated gradients of `(go * wgp(x, y, w)).sum()` for SAME-SHAPE operands
    (no broadcasting): returns (gx, gy, gw) with shapes of (x, y, w)."""
    outs = _wgp_grad(*x.unbind(-1), *y.unbind(-1), *w.unbind(-1), *go.unbind(-1))
    outs = torch.broadcast_tensors(*outs)
    gx = torch.stack(outs[0:16], dim=-1)
    gy = torch.stack(outs[16:32], dim=-1)
    gw = torch.stack(outs[32:67], dim=-1)
    return gx, gy, gw
