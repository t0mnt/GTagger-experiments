"""No NEW CPU-GPU synchronization on the per-forward path.

Rationale is upstream's, from the lgatr/lloca author on their torch.compile workflow:

    "My torch.compile-improvement-workflow is to first use torch.profiler to find all
     CPU-GPU synchronizations (you need a GPU for that) and fix them (this is the main
     timing win usually), and afterwards look for rewrites to enable fast kernels
     everywhere instead of 4x4 or so kernels."

A sync costs more than the stall itself: it stops CPU run-ahead, so the queue drains and
the GPU idles between kernels. That hurts most in exactly this repo's shape -- many small
kernels -- which is the same reason the author's second step matters here.

Finding them needs a GPU. Catching NEW ones does not: every sync has a syntactic signature,
and the cheap ones to police are a tensor used as a python truth value (`if t.any()`,
`assert t.all()`) and an explicit host read (`.item()`, `.cpu()`, `.tolist()`, `.numpy()`).
This test is the static half, so a GPU profiling session starts from a short list instead of
rediscovering the same sites.

Census as of 2026-08-12, over the compiled campaign posture:
  - ONE per-forward sync, allowlisted below with its fix.
  - the data-dependent `.nonzero()` in ParT's PairEmbed is on the EAGER pair path only;
    `compiled_dense=True` routes the compiled posture to the dense twin, which has none.
  - the only `.item()` in model code is inside that warning's f-string, behind the same
    guard plus a once-per-process latch, so it is not a per-step read.
"""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
# MODEL code only: the nets, the wrappers and the per-step embedding. The experiment
# DRIVERS (experiment.py, jetclassexperiment.py, toptagxlexperiment.py) are deliberately
# out of scope -- they hold evaluation and metric logging, where reading results back to
# the host is the entire point and a sync costs nothing because training has stopped.
# Scoping by file rather than by function name is on purpose: the one real finding lives in
# `jet_frames`, which is called from forward but is not named forward, so a name-based
# scope would have missed exactly the site this test exists for.
SCANNED = sorted(
    [*(REPO / "experiments" / "baselines").glob("*.py"),
     REPO / "experiments" / "tagging" / "wrappers.py",
     REPO / "experiments" / "tagging" / "embedding.py"]
)

# A tensor used as a python bool, or read back to the host. Both force the queue to drain.
PATTERNS = [
    (re.compile(r"^\s*(?:if|elif|while|assert)\b.*\.\s*(?:any|all)\s*\(\s*\)"), "tensor truth value"),
    (re.compile(r"\.item\s*\(\s*\)"), "host read: .item()"),
    (re.compile(r"\.tolist\s*\(\s*\)"), "host read: .tolist()"),
    (re.compile(r"\.numpy\s*\(\s*\)"), "host read: .numpy()"),
]

# Known and accepted, keyed on the source line so it cannot silently move or multiply.
# Each entry states WHY it is tolerated and what the fix would be, because "allowlisted"
# without a fix is just a suppressed finding.
SYNC_OK = {
    "if bad.any():": (
        "experiments/tagging/wrappers.py",
        "jet_frames' degenerate-reference fallback, per forward on the two GraphTrans "
        "wrappers that set compute_jet_frames (ParticleNetParT, Plain). FIX, and it is "
        "bit-identical: the fallback below it is already branchless "
        "(`torch.where(bad[:,None,None], eye, trafo)`, which returns `trafo` unchanged when "
        "`bad` is all-False), so hoisting it out of the `if` removes the sync without "
        "touching arithmetic -- the `if` then only guards the once-only warning, which can "
        "be bounded to the first N steps. NOT applied: the campaign has started and the gain "
        "is unmeasured. Post-campaign, and measure it with the profiler recipe in "
        "docs/cgenn-compile.md first."
    ),
    'f"(max |L eta L^T - eta| = {dev.max().item():.2e}); using the identity "': (
        "experiments/tagging/wrappers.py",
        "the diagnostic inside that same warning. Sits behind BOTH `if bad.any()` and the "
        "`_jet_frames_degenerate_warned` latch, so it runs at most once per process -- a "
        "once-only host read is not a per-step sync and needs no fix. Listed so the count "
        "above stays honest rather than by widening the pattern to miss it."
    ),
}


def _hits(path):
    out = []
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        code = line.split("#", 1)[0]
        if not code.strip():
            continue
        for pattern, kind in PATTERNS:
            if pattern.search(code):
                out.append((lineno, code.strip(), kind))
    return out


@pytest.mark.parametrize("path", SCANNED, ids=lambda p: p.name)
def test_no_new_device_sync_on_the_forward_path(path):
    rel = str(path.relative_to(REPO))
    unexpected = [
        f"{rel}:{lineno}  [{kind}]  {code}"
        for lineno, code, kind in _hits(path)
        if not (code in SYNC_OK and SYNC_OK[code][0] == rel)
    ]
    assert not unexpected, (
        "new CPU-GPU synchronization on the per-forward path:\n  "
        + "\n  ".join(unexpected)
        + "\n\nA tensor used as a python bool, or read back to the host, drains the queue and "
        "stops CPU run-ahead. Rewrite it branchlessly (torch.where / masked arithmetic), or "
        "add it to SYNC_OK with the reason and the fix."
    )


def test_allowlist_entries_are_all_still_present():
    """An allowlist entry whose site is gone is a stale waiver — delete it, do not carry it."""
    live = {code for path in SCANNED for _, code, _ in _hits(path)}
    stale = sorted(set(SYNC_OK) - live)
    assert not stale, f"SYNC_OK entries no longer in the source: {stale}"
