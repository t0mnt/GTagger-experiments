"""The find_lr transcription rule, pinned to its live cases (rewritten 2026-08-16).

History this file must keep failing-loudly on: the first version of this test suite
PINNED THE BUG -- `test_coherent_pair_takes_steepest` asserted that PNPT's reading
(steepest 3.08e-05, bracket 1.17e-04, "coherent" at 3.8x) resolves to steepest, and
that endorsement cost the row 0.32pp accuracy / 40% background rejection against the
bracket-class value 1e-3 that matches the external weaver reference. Ratio-coherence
cannot see curve-pinning when the curve drags both statistics low; the rule is now
gated on the `pinned` curve-shape flag, and every case below states which flag it
carries and why. The detector itself is tested on synthetic curves at the bottom.
"""

import numpy as np

from utils.find_lr import PINNED_DECADES, suggest_lr, transcribe_lr


# ---------------------------------------------------------------- decision table


def test_pnpt_incident_reading_takes_bracket():
    # THE 0.32pp incident (2026-08-16): PNPT bs=512 read steepest 3.08e-05 with
    # bracket 1.17e-04 -- hybrid curves are pinned (slope plateau over decades), so
    # the old "coherent -> steepest" endorsement was the bug. Bracket now.
    lr, why = transcribe_lr(3.08e-05, 1.17e-04, interior_min=True, pinned=True)
    assert lr == 1.17e-04 and "loss-min/10" in why
    assert "confirm" in why  # the rerun caveat: hybrid brackets vary run-to-run


def test_pnpt_second_reading_takes_bracket_matching_weaver():
    # The same model's other finder reading: bracket ~1e-3 -- the value that scored
    # 0.9414 / rej 1771 and matched weaver's 0.9417. Both readings resolve to the
    # bracket; the run-to-run bracket spread is what the printed caveat covers.
    lr, _ = transcribe_lr(3.0e-05, 1.0e-03, interior_min=True, pinned=True)
    assert lr == 1.0e-03


def test_plaingraphtrans_takes_bracket():
    # PlainGraphTrans bs=512: steepest 3.08e-05 curve-pinned (read 4e-05 even at
    # bs=2048 -- the plateau does not move), bracket 4.52e-04 with interior minimum.
    lr, why = transcribe_lr(3.08e-05, 4.52e-04, interior_min=True, pinned=True)
    assert lr == 4.52e-04 and "loss-min/10" in why


def test_pinned_without_interior_min_refuses():
    # ParticleNet 2026-08-15 anomaly: 2.21e-04 vs 8.19e-03, no interior minimum, and
    # steepest 6-8x below its own nine-rerun envelope -- a shifted curve IS pinning.
    # Nothing anchored -> no recipe. (The 3-rerun anomaly protocol in the docs is
    # what separates this from the healthy no-interior ParticleNet runs; a single
    # curve cannot.)
    lr, why = transcribe_lr(2.21e-04, 8.19e-03, interior_min=False, pinned=True)
    assert lr is None and "NO RECIPE" in why


def test_particlenet_nine_rerun_pattern_keeps_steepest():
    # The nine-rerun evidence, correctly scoped: a DISTINCT slope peak at 1.32e-3
    # with the bracket 20x above (2.64e-2, trough hugging the divergence). The
    # bracket is the outlier there; steepest matched the reproduced optimum.
    lr, why = transcribe_lr(1.32e-03, 2.64e-02, interior_min=True, pinned=False)
    assert lr == 1.32e-03 and "steepest" in why


def test_distinct_no_interior_keeps_steepest():
    # ParticleNet's no-interior branch: bracket unanchored by definition, distinct
    # peak -> steepest. (The OLD rule returned None here for ratio > 10x, which
    # would have voided the nine good reruns it was founded on -- ratio 20-70x.)
    lr, why = transcribe_lr(1.91e-03, 1.39e-01, interior_min=False, pinned=False)
    assert lr == 1.91e-03 and "steepest" in why


def test_distinct_interior_coherent_takes_bracket():
    # Both anchored and agreeing within the bar: the bracket is the recipe value
    # the classic range test transcribes; steepest is its lower bound.
    lr, why = transcribe_lr(1.5e-03, 8.0e-03, interior_min=True, pinned=False)
    assert lr == 8.0e-03 and "loss-min/10" in why


def test_bracket_below_distinct_steepest_is_no_bracket():
    # ratio < 1: a "bracket" under a distinct slope peak brackets nothing.
    lr, why = transcribe_lr(1.0e-03, 5.0e-04, interior_min=True, pinned=True)
    assert lr == 5.0e-04  # pinned still defers to the anchored bracket
    lr, why = transcribe_lr(1.0e-03, 5.0e-04, interior_min=True, pinned=False)
    assert lr == 1.0e-03 and "no upper bracket" in why


# ---------------------------------------------------- the pinned detector itself


def _sweep(loss_of_loglr, n=300, lo=1e-7, hi=1e1):
    lrs = np.geomspace(lo, hi, n)
    return lrs, np.array([loss_of_loglr(np.log10(lr)) for lr in lrs])


def test_detector_flags_hybrid_plateau_curve():
    # GT-hybrid shape: near-constant shallow descent over ~3.5 decades (the plateau
    # that pinned steepest at 3e-5), easing into an interior trough, then blow-up.
    def loss(x):  # x = log10(lr)
        if x < -6.0:
            return 0.70
        if x < -2.5:
            return 0.70 - 0.10 * (x + 6.0)  # long shallow fall: -0.10 / decade
        if x < -2.0:
            return 0.35 - 0.05 * (x + 2.5)  # easing toward the trough
        return 0.325 + 0.8 * (x + 2.0)  # divergence

    lrs, losses = _sweep(loss)
    steepest, _, _, _, interior, pinned, decades = suggest_lr(lrs, losses, 10, 5)
    assert interior, "trough sits well inside the sweep"
    assert pinned and decades > PINNED_DECADES, (
        f"a {decades:.1f}-decade half-max slope region must flag as pinned")
    assert steepest < 3e-3  # steepest lives somewhere in the plateau; never a recipe here


def test_detector_passes_concentrated_fall():
    # ParticleNet shape: flat, one concentrated fall spanning well under 1.5 decades
    # around 1e-3, shallow tail into a late minimum, then blow-up.
    def loss(x):
        if x < -3.5:
            return 0.70
        if x < -2.5:
            return 0.70 - 0.40 * (x + 3.5)  # concentrated fall: -0.40 / decade
        if x < -0.7:
            return 0.30 - 0.01 * (x + 2.5)  # long shallow tail
        return 0.282 + 0.8 * (x + 0.7)

    lrs, losses = _sweep(loss)
    steepest, _, _, _, _, pinned, decades = suggest_lr(lrs, losses, 10, 5)
    assert not pinned and decades <= PINNED_DECADES, (
        f"a concentrated fall ({decades:.1f} decades) must NOT flag as pinned")
    assert 1e-4 < steepest < 1e-2, "steepest tracks the concentrated fall"


def test_pnpt_measured_width_classifies_pinned():
    """The real PNPT bs=512 curve measured its half-max slope region at EXACTLY 1.5
    decades (H100, 2026-08-16) and the first threshold (1.5 EXCLUSIVE) classified it
    'distinct' -- re-emitting the incident's 3.08e-05 through the bracket-outlier
    branch. Pin the constant and the inclusiveness: 1.5 must classify as pinned,
    and a curve sitting exactly ON the threshold must too."""
    assert PINNED_DECADES <= 1.5, "PNPT's measured 1.5-decade plateau must flag pinned"
    lr, why = transcribe_lr(3.08e-05, 1.14e-03, interior_min=True, pinned=1.5 >= PINNED_DECADES)
    assert lr == 1.14e-03 and "loss-min/10" in why


def test_detector_degenerate_curve_is_pinned():
    # fewer points than the gradient window needs -> no certified peak -> pinned
    # (main() refuses < 3 points already; this pins the helper's own conservatism)
    lrs, losses = np.array([1e-5]), np.array([0.7])
    *_, pinned, decades = suggest_lr(lrs, losses, 10, 5)
    assert pinned and np.isnan(decades)
