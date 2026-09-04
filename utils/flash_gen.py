"""FLASH PLAN v2, Step 2: generate the Cl(1,3) weighted-GP reference module.

    python utils/flash_gen.py        # (re)writes experiments/baselines/cgenn/flash_ref_p1m3.py

Extends flash-kingdon's codegen recipe (github.com/tBuLi/flash-kingdon, README: build
the weighted geometric product symbolically, derive the gradient by symbolic
differentiation, CSE, emit flat arithmetic) from O(2)/O(3) to SO(1,3), with two
repo-specific hardenings:

- TERMS ARE SOURCED FROM KINGDON'S OWN Cl(1,3) PRODUCTS (`Algebra(1, 3)` blade
  products, the tool the port is built on and cites) and then ASSERTED, sign by sign
  and path by path, against this repo's reference tables (`CliffordAlgebra.cayley`,
  `sparse_gp_tables`) before a single line is emitted. Step 1 proved the two agree
  globally; the generator re-proves it at every term it uses. Two levels: per term,
  that (right blade, sign) is kingdon's; and globally, that the assembled weighted
  forward equals kingdon's own `sum_p w_p * <x_a y_c>_b` over the grade triples --
  the second is what pins the WEIGHT each term is multiplied by, which the first
  cannot see (`_kingdon_weighted_forward`).
- WEIGHT ORDER IS THE REPO'S COMPACT-PATH ORDER (`geometric_product_paths.nonzero()`,
  35 paths for the Lorentz metric) so generated kernels are checkpoint-compatible
  with every existing sparse-GP weight tensor.

The emitted functions are flat scalar arithmetic only (+, -, *), one assignment per
CSE symbol -- flash-clifford's kernel-body style (ops/fc_p3m0.py) -- so Step 3
transcribes them into `triton.jit` bodies mechanically. The emitted module also
carries torch wrappers (unbind/stack) used by the Step-2 gates and later as the
CPU/parity twin of the Triton kernels.
"""

import importlib.metadata
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sympy
import torch

OUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "experiments", "baselines", "cgenn", "flash_ref_p1m3.py",
)
NB, NPATH = 16, 35


def _repo_tables():
    from experiments.baselines.cgenn.cliffordalgebra import CliffordAlgebra, sparse_gp_tables

    alg = CliffordAlgebra((1.0, -1.0, -1.0, -1.0))
    pidx = alg.geometric_product_paths.nonzero().T.contiguous()
    spath, spval, _ = sparse_gp_tables(alg, pidx)
    assert pidx.shape[1] == NPATH, f"expected {NPATH} Lorentz paths, got {pidx.shape[1]}"
    return alg, pidx, spath.long(), spval, alg.gp_k_idx.long()


def _kingdon_terms():
    """{(i, j): (k, sign)} from kingdon's Cl(1,3) blade products: e_i * e_k has a
    single output blade j with coefficient sign (the quasigroup property)."""
    from kingdon import Algebra

    theirs = Algebra(1, 3)
    names = list(theirs.canon2bin)
    blades = theirs.blades
    terms = {}
    for i, ni in enumerate(names):
        for k, nk in enumerate(names):
            prod = blades[ni] * blades[nk]
            coeffs = {key: float(v) for key, v in zip(prod.keys(), prod.values()) if float(v) != 0}
            assert len(coeffs) == 1, f"non-quasigroup product {ni}*{nk}"
            (bin_key, sign), = coeffs.items()
            j = names.index(theirs.bin2canon[bin_key])
            terms[(i, j)] = (k, int(sign))
    return terms


def _kingdon_weighted_forward(pidx, x, y, w):
    """The whole weighted forward, reassembled from kingdon's grade projections.

    `_kingdon_terms` checks each term's (right blade, sign); nothing checked WHICH WEIGHT
    a term is multiplied by. That comes from `spath[i, j]`, the only generator input taken
    from the repo on faith, and a permuted one still satisfies the per-term assert and the
    all-35-paths assert while silently mixing up the compact weight vector.

    This closes it without giving up checkpoint compatibility, because the two things are
    separable: `pidx` (which grade triples exist, in which ORDER) stays the repo's, since
    that order is what every stored sparse-GP weight tensor is indexed by; the ASSIGNMENT
    of blade pairs to those triples is pure algebra, so it is rebuilt here as
    `sum_p w_p * <x_<g_left> y_<g_right>>_<g_out>` and never reads `spath` at all.

    `pidx` rows are the cayley's [left, OUT, right] orientation, not [left, right, out].
    """
    from kingdon import Algebra

    theirs = Algebra(1, 3)
    names = list(theirs.canon2bin)
    kx, ky = theirs.multivector(name="x"), theirs.multivector(name="y")
    # kingdon names a coefficient after its blade ('e12' -> 'x12', scalar 'e' -> 'x'); the
    # blade order is gated as the identity permutation in test_kingdon_conventions.py.
    rename = {}
    for i, name in enumerate(names):
        rename[sympy.Symbol(f"x{name[1:]}")] = x[i]
        rename[sympy.Symbol(f"y{name[1:]}")] = y[i]

    fwd = [sympy.Integer(0)] * NB
    for p in range(NPATH):
        g_left, g_out, g_right = (int(v) for v in pidx[:, p])
        proj = (kx.grade(g_left) * ky.grade(g_right)).grade(g_out)
        for blade, coeff in zip(proj.keys(), proj.values()):
            j = names.index(theirs.bin2canon[blade])
            fwd[j] = fwd[j] + w[p] * sympy.sympify(coeff).xreplace(rename)
    return fwd


def _build_expressions():
    alg, pidx, spath, spval, kidx = _repo_tables()
    kd = _kingdon_terms()

    x = sympy.symbols(f"x0:{NB}")
    y = sympy.symbols(f"y0:{NB}")
    w = sympy.symbols(f"w0:{NPATH}")
    g = sympy.symbols(f"g0:{NB}")

    fwd = [sympy.Integer(0)] * NB
    used_paths = set()
    for i in range(NB):
        for j in range(NB):
            sign = int(spval[i, j])
            if sign == 0:
                continue
            k = int(kidx[i, j])
            # the hardening: every term the repo tables prescribe must be exactly the
            # term kingdon's product algebra produces -- (right blade, sign) both.
            assert kd[(i, j)] == (k, sign), (
                f"term (i={i}, j={j}): repo says (k={k}, sign={sign}), "
                f"kingdon says {kd[(i, j)]}")
            p = int(spath[i, j])
            used_paths.add(p)
            fwd[j] = fwd[j] + sign * w[p] * x[i] * y[k]
    assert used_paths == set(range(NPATH)), "not every compact path is exercised"
    assert sum(len(e.args) for e in fwd) == 256, "expected 256 forward terms total"

    # weight placement (see `_kingdon_weighted_forward`): the per-term assert above is the
    # localized one and fires first with the offending (i, j); this one is global and is
    # the only check that `spath` puts each term under the right compact weight.
    king = _kingdon_weighted_forward(pidx, x, y, w)
    mismatched = [j for j in range(NB) if sympy.expand(fwd[j] - king[j]) != 0]
    assert not mismatched, (
        f"weighted forward disagrees with kingdon at output blades {mismatched}: the "
        f"repo's spath assigns those terms a different compact weight than the grade "
        f"triple in geometric_product_paths does"
    )

    loss = sum(gj * fj for gj, fj in zip(g, fwd))
    gx = [sympy.diff(loss, xi) for xi in x]
    gy = [sympy.diff(loss, yi) for yi in y]
    gw = [sympy.diff(loss, wp) for wp in w]
    return (x, y, w, g), fwd, gx, gy, gw


def _emit_function(name, args, exprs, prefix):
    repl, reduced = sympy.cse(exprs, symbols=sympy.numbered_symbols(prefix))
    lines = [f"def {name}({', '.join(map(str, args))}):"]
    for sym, val in repl:
        lines.append(f"    {sym} = {val}")
    body = ",\n        ".join(str(e) for e in reduced)
    lines.append(f"    return (\n        {body},\n    )")
    return "\n".join(lines), len(repl)


HEADER = '''"""Cl(1,3) weighted geometric product -- GENERATED, DO NOT EDIT.

Generator: utils/flash_gen.py (FLASH PLAN v2 step 2), terms sourced from kingdon
{kver} (`Algebra(1, 3)` blade products; MIT, arXiv:2503.10451) and asserted against
this repo's `CliffordAlgebra` cayley + `sparse_gp_tables` at generation time; the
expression assembly, differentiation and CSE run in sympy {sver} (a stated deviation
from kingdon's own compile() printer pipeline -- sourcing terms from the product
algebra keeps kingdon the mathematical authority while the repo controls weight
order and emission style). Weight order = the repo's 35-entry compact-path order:
checkpoint-compatible with every sparse-GP weight tensor. Flat arithmetic bodies
(flash-clifford kernel style) so step 3 wraps them with `triton.jit` directly
(flash-kingdon's trick); the torch wrappers below are the CPU reference / parity
twin. Gates: tests/internal/test_flash_ref_p1m3.py.
"""
# generated-with: kingdon={kver} sympy={sver}

import torch


'''

WRAPPERS = '''

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
'''


def main():
    kver = importlib.metadata.version("kingdon")
    sver = importlib.metadata.version("sympy")
    args, fwd, gx, gy, gw = _build_expressions()
    x, y, w, g = args

    fwd_src, n_fwd = _emit_function("_wgp_fwd", (*x, *y, *w), fwd, "_f")
    grad_src, n_grad = _emit_function("_wgp_grad", (*x, *y, *w, *g), (*gx, *gy, *gw), "_b")

    src = HEADER.format(kver=kver, sver=sver) + fwd_src + "\n\n" + grad_src + WRAPPERS
    with open(OUT_PATH, "w") as fh:
        fh.write(src)
    print(f"wrote {OUT_PATH}")
    print(f"  forward: 256 terms, {n_fwd} CSE temps; grad: 67 outputs, {n_grad} CSE temps")


if __name__ == "__main__":
    main()
