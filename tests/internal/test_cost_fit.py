import math

from utils.cost_fit import fit


def test_recovers_a_line():
    fixed, marginal, r2 = fit([32, 128, 512, 2048], [10 + 0.7 * b for b in (32, 128, 512, 2048)])
    assert math.isclose(fixed, 10, rel_tol=1e-9) and math.isclose(marginal, 0.7, rel_tol=1e-9)
    assert math.isclose(r2, 1.0, abs_tol=1e-12)


def test_rotorch_hulls_numbers_reproduce_their_doc():
    # rotorch docs/benchmark.rst: cgenn uncompiled, fixed 7.8 ms, marginal 715 us/sample
    fixed, marginal, r2 = fit([32, 128, 512, 2048], [39.9, 101.1, 359.6, 1475.4])
    assert abs(fixed - 7.8) < 0.5 and abs(marginal * 1e3 - 715) < 5 and r2 > 0.997
