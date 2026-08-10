"""GPU-class device bugs, caught on a CPU-only runner.

Motivation (docs/cgenn-compile.md, final audit): the CGENN hybrid's
``CliffordAlgebra.b()`` built its blade-index tensors on the DEFAULT device (CPU --
nothing in this repo calls ``torch.set_default_device``) and used them to index a
GPU-resident buffer. Bit-identical and free on a CPU runner, a per-forward host->device
transfer on a GPU. Every gate in the repo runs on CPU, so nothing could see it, and the
fix reached only one of two duplicated copies of the class.

Two complementary nets, neither of which needs a GPU:

1. ``test_no_device_implicit_tensor_in_forward`` (dynamic) intercepts torch's tensor
   factories during a REAL forward and fails on any call that omits ``device=``,
   reporting the exact call site. Zero false positives -- it sees only executed code.
2. ``test_no_default_device_tensor_creation`` (static) is the cheap always-on scan of
   the same property, so a new offender fails even in code the mini batch doesn't reach.

Both are proven non-vacuous: reverting the ``b()`` fix makes each of them fail on
exactly that line, which also demonstrates that ``b()`` really is on the live forward
path (via ``MVLayerNorm -> norm()``).

A fake-CUDA forward under ``FakeTensorMode`` was tried FIRST and abandoned, recorded
here so nobody retries it: fake tensors do enforce device agreement (verified: binary
ops, matmul and index_select all raise ``FakeTensorDeviceMismatchError``), but they
cannot traverse these models at all -- ``to_dense_batch`` needs real counts and raises
``GuardOnDataDependentSymNode`` even with a ``ShapeEnv``. Note also that advanced
indexing (``gpu[cpu_idx]``) is legal in torch and raises nowhere, which is precisely
why the b() bug degraded performance rather than crashing.
"""

import ast
import os

import pytest
import torch

from pathlib import Path

# SELF-CONTAINED ON PURPOSE. This file is KEEP-permanently (cleanup.md) while
# test_cgenn_compile.py and test_nonequi_compile.py are port instruments scheduled for
# deletion. Importing REPO/_fixed_batch/_build from them would turn the wipe into a
# collection ERROR for the one device test that is supposed to outlive the port, so the
# three helpers are inlined here instead. They are small and stable; keeping a copy is
# cheaper than coupling a permanent test to a temporary one.
REPO = Path(__file__).resolve().parents[2]


def _fixed_batch(exp):
    torch.manual_seed(1)
    return next(iter(exp.train_loader))


def _build(model, float64, extra_overrides=()):
    import logging.handlers  # noqa: F401
    import hydra
    import experiments.logger
    from experiments.tagging.experiment import TopTaggingExperiment

    experiments.logger.LOGGER.disabled = True
    overrides = ["save=false", "training.batchsize=4", "data.dataset=mini",
                 f"model={model}", f"use_float64={'true' if float64 else 'false'}",
                 *extra_overrides]
    with hydra.initialize_config_dir(config_dir=str(REPO / "config_quick"), version_base=None):
        cfg = hydra.compose(config_name="toptagging", overrides=overrides)
    torch.manual_seed(0)
    exp = TopTaggingExperiment(cfg)
    exp._init()
    exp.init_physics()
    exp.init_model()
    exp.init_data()
    exp._init_dataloader()
    exp._init_loss()
    exp.model.eval()
    return exp

RUN_SLOW = os.environ.get("CGENN_COMPILE_GATES") == "1"

MODELS = [
    "tag_ParT", "tag_particlenet", "tag_transformer", "tag_MIParT",
    "tag_PlainGraphTrans", "tag_PlainGraphGPS",
    "tag_ParticleNetParTGraphTrans", "tag_ParticleNetParTGraphGPS",
    "tag_cgenn", "tag_lorentznet",
    "tag_CGENNLGATrGraphTrans", "tag_CGENNLGATrGraphGPS",
    "tag_LorentzNetLGATrSlimGraphTrans", "tag_LorentzNetLGATrSlimGraphGPS",
]


_TRACED_FACTORIES = ["tensor", "zeros", "ones", "empty", "arange", "eye", "full",
                     "rand", "randn", "randint"]


@pytest.mark.skipif(not RUN_SLOW, reason="device hygiene sweep: set CGENN_COMPILE_GATES=1")
@pytest.mark.parametrize("model", MODELS)
def test_no_device_implicit_tensor_in_forward(model):
    """DYNAMIC: run a real forward and record every tensor factory called WITHOUT
    ``device=``, with its call site.

    Strictly better than the static scan for precision -- it sees only code that
    actually executes, so unreachable helpers and __init__-time table building cannot
    produce false positives, and it needs no allowlist. (A fake-CUDA forward was tried
    first and abandoned: FakeTensorMode cannot traverse these models at all, because
    `to_dense_batch` needs real counts and raises GuardOnDataDependentSymNode.)

    Construction is deliberately excluded by clearing the record after the model is
    built: __init__-time tensors become buffers/parameters that `.to(device)` moves.
    """
    import traceback

    hits, orig = {}, {n: getattr(torch, n) for n in _TRACED_FACTORIES}

    def make(fn):
        def wrapper(*a, **k):
            if k.get("device") is None:
                # attribute ONLY when our code is the DIRECT creator: stack[-1] is this
                # wrapper, so stack[-2] is the immediate caller. Walking further up
                # blames our call site for a third-party library's internal allocation
                # (e.g. xformers' BlockDiagonalMask.from_seqlens builds its own tensors
                # from a python list; placement is xformers' business, not ours).
                stack = traceback.extract_stack()
                fr = stack[-2] if len(stack) >= 2 else None
                if fr and "/experiments/" in fr.filename and "/test_" not in fr.filename:
                    rel = fr.filename.split("GTagger-experiments/")[-1]
                    hits.setdefault(f"{rel}:{fr.lineno}", (fr.line or "").strip())
            return fn(*a, **k)
        return wrapper

    for n in _TRACED_FACTORIES:
        setattr(torch, n, make(orig[n]))
    try:
        exp = _build(model, float64=False)
        data = _fixed_batch(exp)
        hits.clear()  # measure the FORWARD only, not construction
        with torch.no_grad():
            exp._get_ypred_and_label(data.clone())
    finally:
        for n in _TRACED_FACTORIES:
            setattr(torch, n, orig[n])

    print(f"GATE-DEVICE[{model}] per-forward device-implicit factory calls = {len(hits)}")
    assert not hits, (
        f"{model}: tensor(s) created per-forward without device= (CPU by default -- a "
        f"host->device transfer every forward on GPU, or a mismatch):\n  "
        + "\n  ".join(f"{k}  |  {v}" for k, v in sorted(hits.items())))


# Tensor factories that silently default to CPU when `device=` is omitted.
# The `*_like` family is deliberately EXCLUDED: it inherits device (and dtype) from its
# argument, so those calls are device-correct by construction.
_FACTORIES = {"tensor", "zeros", "ones", "empty", "arange", "eye", "full", "linspace",
              "rand", "randn", "randint"}

# Enclosing functions exempt from the per-forward rule, each with the reason it cannot
# run inside a forward. Verified by call-site inspection, not assumed.
_NON_FORWARD_FNS = {
    "construct_gmt": "called only from CliffordAlgebra.__init__ (blade table build)",
    "_compute_lookup": "__init__-time path-index lookup",
    "sparse_gp_tables": "__init__-time sparse table build",
    "output_blades": "no call site in the repo",
    "random": "equivariance-test utility; no forward call site",
    "random_vector": "equivariance-test utility; reached only via versor/rotor",
    "versor": "equivariance-test utility; no forward call site",
    "rotor": "equivariance-test utility; no forward call site",
    "geometric_product_paths": "__init__-time grade-path table",
}
# files whose tensors must all be device-explicit (model code that runs per forward)
_SCANNED = [
    "experiments/baselines/CGENNLGATrGraphTransHybrid.py",
    "experiments/baselines/cgenn/cliffordalgebra.py",
    "experiments/baselines/cgennlgatrgraphgps.py",
    "experiments/baselines/particletransformer.py",
    "experiments/baselines/particlenettransformer.py",
    "experiments/baselines/particlenetpartgraphgps.py",
    "experiments/baselines/plaingraphtrans.py",
    "experiments/baselines/plaingraphgps.py",
    "experiments/baselines/lorentznet.py",
    "experiments/baselines/lorentznetlgatrslimgraphtrans.py",
    "experiments/baselines/lorentznetlgatrslimgraphgps.py",
]


def _factory_calls_without_device(path):
    """(line, source) for torch.<factory>(...) calls lacking a device= kwarg."""
    src = (REPO / path).read_text()
    tree = ast.parse(src)
    lines = src.splitlines()
    # map: function def line ranges for __init__, which is allowed to build on CPU
    exempt = {"__init__", "reset_parameters", *_NON_FORWARD_FNS}
    init_ranges = [(n.lineno, n.end_lineno) for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and n.name in exempt]
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if not (isinstance(f, ast.Attribute) and f.attr in _FACTORIES
                and isinstance(f.value, ast.Name) and f.value.id == "torch"):
            continue
        if any(kw.arg == "device" for kw in node.keywords):
            continue
        if any(lo <= node.lineno <= hi for lo, hi in init_ranges):
            continue  # __init__-time: becomes a buffer/param, moved by .to()
        out.append((node.lineno, lines[node.lineno - 1].strip()))
    return out


def test_no_default_device_tensor_creation():
    """No per-forward tensor factory may omit `device=` in model code.

    Nothing sets a default device, so an omitted `device=` means CPU regardless of
    where the model lives -- at best a silent host->device transfer every forward, at
    worst a device-mismatch crash on GPU. __init__-time construction is exempt (those
    become buffers/parameters that `.to()` moves).
    """
    offenders = []
    for path in _SCANNED:
        for lineno, src in _factory_calls_without_device(path):
            offenders.append(f"{path}:{lineno}: {src}")
    assert not offenders, (
        "per-forward tensor creation without device= (CPU by default, so this is a "
        "host->device transfer or a mismatch on GPU):\n  " + "\n  ".join(offenders))


# Tensors held as PLAIN ATTRIBUTES (not buffers/parameters) that are nonetheless safe,
# each with the reason it cannot cause a device fault. Verified by call-site inspection.
# The bar for an entry here is: never used as a tensor operand in a forward -- only its
# shape, a count, or python ints derived from it.
_UNMOVABLE_OK = {
    ("CliffordAlgebra", "grades"):
        "__init__ only: grades_list (python ints), n_subspaces (len), subspaces (comb), "
        "and _grade_to_slice, which reduces it to int slice bounds",
    ("CliffordAlgebra", "geometric_product_paths"):
        "__init__ only: .nonzero() -> the _path_idx BUFFER, .sum() -> a count, .size()",
    ("FullyConnectedSteerableGeometricProductLayer", "product_paths"):
        "copy of the above; same three init-time uses (.nonzero()/.sum()/.size())",
    ("SteerableGeometricProductLayer", "product_paths"):
        "copy of the above; same three init-time uses (.nonzero()/.sum()/.size())",
}


@pytest.mark.skipif(not RUN_SLOW, reason="device hygiene sweep: set CGENN_COMPILE_GATES=1")
@pytest.mark.parametrize("model", MODELS)
def test_no_unmovable_tensor_attributes(model):
    """STRUCTURAL: no module may hold a forward-used tensor that `.to(device)` cannot move.

    The third net in this file, and the one that would have caught the bug the other two
    missed. `nn.Module.to()` walks `_buffers` and `_parameters`. A tensor that reaches
    `__dict__` by any other route -- a plain `self.x = tensor`, or a
    `functools.cached_property` (whose `__get__` writes to `instance.__dict__` directly,
    bypassing `nn.Module.__setattr__`) -- is invisible to it and stays on CPU forever.

    What this caught, on `main` as much as on this branch: `CliffordAlgebra`'s
    `_alpha/_beta/_gamma_signs` were cached properties, materialized during `__init__`, so
    every instance carried CPU sign vectors unconditionally. `mvlayernorm -> norm -> q ->
    b -> beta` does `signs * mv` on the live forward path, so tag_cgenn and both CGENN
    hybrids raised "Expected all tensors to be on the same device" at the first forward on
    a GPU -- while passing every CPU gate in this repo, including the other two nets here.
    The dynamic net only watches `torch.<factory>` calls (these came from `torch.pow`); the
    static scan exempts `__init__` (which is exactly where they were built). Neither could
    see it. This one is structural, so it needs no GPU and no forward.

    Fixed by registering them as non-persistent buffers -- values bit-identical,
    `state_dict` unchanged.
    """
    exp = _build(model, float64=False)
    offenders = []
    for mod in exp.model.modules():
        held = {n for n, _ in mod.named_buffers(recurse=False)}
        held |= {n for n, _ in mod.named_parameters(recurse=False)}
        for name, val in vars(mod).items():
            if not isinstance(val, torch.Tensor) or name in held:
                continue
            if (type(mod).__name__, name) in _UNMOVABLE_OK:
                continue
            offenders.append(f"{type(mod).__name__}.{name} {tuple(val.shape)}")

    print(f"GATE-UNMOVABLE[{model}] unregistered tensor attributes = {len(offenders)}")
    assert not offenders, (
        f"{model}: tensor(s) held where `.to(device)` cannot reach them -- a plain "
        f"attribute or a functools.cached_property:\n  " + "\n  ".join(sorted(set(offenders)))
        + "\n\nRegister them with `register_buffer(..., persistent=False)` if they are "
          "derived constants, or add them to _UNMOVABLE_OK with the call-site reason they "
          "can never be a tensor operand in a forward.")
