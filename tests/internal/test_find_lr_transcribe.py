"""The find_lr transcription rule, pinned to its three live cases (2026-08-15).

The banner's TRANSCRIBE line exists because a three-number banner got transcribed
wrong under queue pressure; this test pins the pure decision function to the exact
readings that motivated it, so a future tweak that flips any of them fails loudly.
"""

from utils.find_lr import transcribe_lr


def test_coherent_pair_takes_steepest():
    # ParticleNetParTGraphTrans bs=512: 3.08e-05 vs 1.17e-04 (3.8x) -> steepest
    lr, why = transcribe_lr(3.08e-05, 1.17e-04, interior_min=True)
    assert lr == 3.08e-05 and "steepest" in why


def test_rule_fires_with_interior_min_takes_bracket():
    # PlainGraphTrans bs=512: 3.08e-05 vs 4.52e-04 (14.7x, interior min) -> bracket
    lr, why = transcribe_lr(3.08e-05, 4.52e-04, interior_min=True)
    assert lr == 4.52e-04 and "loss-min/10" in why


def test_rule_fires_without_interior_min_refuses():
    # ParticleNet 2026-08-15: 2.21e-04 vs 8.19e-03 (37x, NO interior min) -> no recipe
    lr, why = transcribe_lr(2.21e-04, 8.19e-03, interior_min=False)
    assert lr is None and "NO RECIPE" in why


def test_no_interior_min_but_coherent_still_steepest():
    # a coherent pair does not need the interior minimum: steepest stands on its own
    lr, _ = transcribe_lr(1.5e-03, 8.0e-03, interior_min=False)
    assert lr == 1.5e-03


def test_bracket_below_steepest_is_coherent():
    # ratio < 1 (bracket under steepest) is trivially coherent -> steepest
    lr, _ = transcribe_lr(1.0e-03, 5.0e-04, interior_min=True)
    assert lr == 1.0e-03
