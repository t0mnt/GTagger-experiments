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
  globally; the generator re-proves it at every term it uses.
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
    return alg, spath.long(), spval, alg.gp_k_idx.long()


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


def _build_expressions():
    alg, spath, spval, kidx = _repo_tables()
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
this repo's `CliffordAlgebra` cayley + `sparse_gp_tables` at generation time.
Weight order = the repo's 35-entry compact-path order: checkpoint-compatible with
every sparse-GP weight tensor. Flat arithmetic bodies (flash-clifford kernel style)
so step 3 transcribes them into `triton.jit` mechanically; the torch wrappers below
are the CPU reference / parity twin. Gates: tests/internal/test_flash_ref_p1m3.py.
"""

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
    args, fwd, gx, gy, gw = _build_expressions()
    x, y, w, g = args

    fwd_src, n_fwd = _emit_function("_wgp_fwd", (*x, *y, *w), fwd, "_f")
    grad_src, n_grad = _emit_function("_wgp_grad", (*x, *y, *w, *g), (*gx, *gy, *gw), "_b")

    src = HEADER.format(kver=kver) + fwd_src + "\n\n" + grad_src + WRAPPERS
    with open(OUT_PATH, "w") as fh:
        fh.write(src)
    print(f"wrote {OUT_PATH}")
    print(f"  forward: 256 terms, {n_fwd} CSE temps; grad: 67 outputs, {n_grad} CSE temps")


if __name__ == "__main__":
    main()
