"""FLASH PLAN v2, Step 1 gate: kingdon's Cl(1,3) speaks OUR CliffordAlgebra's language.

Everything the flash port generates flows through kingdon's symbolic algebra, so before
one expression is trusted, the conventions must be machine-checked against the
repo's own reference — `CliffordAlgebra((1,-1,-1,-1))`, whose cayley the whole CGENN
family (and every BIT/TOL gate) is built on. Checked here:

- version pin: generated code is only trusted from the pinned kingdon;
- signature: [+1,-1,-1,-1] in kingdon's Algebra(1, 3);
- blade order: both sides are short-lex (grade, then lexicographic) with the vector
  map kingdon e_{k+1} <-> our index k, i.e. the IDENTITY permutation on blades;
- the full multiplication table: all 16x16 blade products, coefficient-exact against
  our cayley in its [left, out, right] orientation (256 nonzeros — the quasigroup
  property the sparse tables rely on).

Skips (not fails) when kingdon is absent, so environments without the flash-port
toolchain keep a green suite; the flash steps themselves require it installed.
"""

import itertools

import pytest
import torch

kingdon = pytest.importorskip("kingdon")

from experiments.baselines.cgenn.cliffordalgebra import CliffordAlgebra

KINGDON_PIN = "2.1.1"  # bump deliberately, with this file's gates green under the new pin


def test_kingdon_version_is_pinned():
    import importlib.metadata as im

    got = im.version("kingdon")
    assert got == KINGDON_PIN, (
        f"kingdon {got} != pinned {KINGDON_PIN}. Generated kernels are only trusted "
        f"from the pinned version -- re-run the full Step 1+2 gates before bumping.")


@pytest.fixture(scope="module")
def algebras():
    ours = CliffordAlgebra((1.0, -1.0, -1.0, -1.0))
    theirs = kingdon.Algebra(1, 3)
    return ours, theirs


def test_signature_matches(algebras):
    _, theirs = algebras
    assert list(theirs.signature) == [1, -1, -1, -1]


def test_blade_order_is_shortlex_with_identity_mapping(algebras):
    """kingdon's canonical order must equal ours under e_{k+1} <-> vector k."""
    _, theirs = algebras
    names = list(theirs.canon2bin)
    assert len(names) == 16
    # rebuild the expected short-lex order from OUR convention (powerset by grade,
    # lexicographic within grade, vectors 0..3 -> kingdon names 1..4)
    expected = ["e" + "".join(str(i + 1) for i in combo) if combo else "e"
                for r in range(5)
                for combo in itertools.combinations(range(4), r)]
    assert names == expected, (
        f"kingdon blade order diverged from short-lex:\n  got      {names}\n"
        f"  expected {expected}\nThe identity blade mapping the generator assumes "
        f"does not hold -- a permutation table is needed before ANY generated code.")


def _kingdon_cayley(theirs):
    """(16,16,16) table in OUR [left, out, right] orientation, from blade products."""
    names = list(theirs.canon2bin)
    blades = theirs.blades
    table = torch.zeros(16, 16, 16, dtype=torch.float64)
    for a, na in enumerate(names):
        for b, nb in enumerate(names):
            prod = blades[na] * blades[nb]
            coeffs = dict(zip(prod.keys(), (float(v) for v in prod.values())))
            for c, nc in enumerate(names):
                table[a, c, b] = coeffs.get(theirs.canon2bin[nc], 0.0)
    return table


def test_full_multiplication_table_identity(algebras):
    """All 4096 cayley entries coefficient-exact: ours[left, out, right] equals the
    kingdon product table under the identity blade mapping. This is the license to
    generate: every sign in every generated kernel traces back to this equality."""
    ours, theirs = algebras
    reference = ours.cayley.to(torch.float64)
    generated = _kingdon_cayley(theirs)
    assert torch.equal(reference, generated), (
        f"cayley mismatch at {int((reference != generated).sum())} entries -- "
        f"conventions differ; STOP and derive the conversion before generating.")
    # and the quasigroup property both sparse tables and the flash kernels rely on:
    # every (left, right) pair lands on exactly ONE output blade
    assert int((reference != 0).sum()) == 256
    assert ((reference != 0).sum(dim=1) == 1).all()
