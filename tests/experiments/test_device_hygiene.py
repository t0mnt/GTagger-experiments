"""GPU-class device bugs, caught on a CPU-only runner.

Motivation (docs/cgenn-compile.md, final audit): the CGENN hybrid's
``CliffordAlgebra.b()`` built its blade-index tensors on the DEFAULT device (CPU --
nothing in this repo calls ``torch.set_default_device``) and used them to index a
GPU-resident buffer. Bit-identical and free on a CPU runner, a per-forward host->device
transfer on a GPU. Every gate in the repo runs on CPU, so nothing could see it, and the
fix reached only one of two duplicated copies of the class.

Four complementary nets, none of which needs a GPU. Two watch how tensors are CREATED,
one watches where they are STORED, and one runs the arithmetic somewhere other than CPU:

1. ``test_no_device_implicit_tensor_in_forward`` (dynamic) intercepts torch's tensor
   factories during a REAL forward and fails on any call that omits ``device=``,
   reporting the exact call site. Zero false positives -- it sees only executed code.
2. ``test_no_default_device_tensor_creation`` (static) is the cheap always-on scan of
   the same property, so a new offender fails even in code the mini batch doesn't reach.

   Both are proven non-vacuous: reverting the ``b()`` fix makes each of them fail on
   exactly that line, which also demonstrates that ``b()`` really is on the live forward
   path (via ``MVLayerNorm -> norm()``).

3. ``test_no_unmovable_tensor_attributes`` (structural) walks every module's ``__dict__``
   for tensors that are not in ``_buffers``/``_parameters`` -- the ones ``.to(device)``
   cannot reach. It needs no forward at all, and it is the one that caught the
   sign-vector bug that made tag_cgenn unable to run on a GPU. It also walks CONTAINERS,
   because ``grade_to_index`` was a plain LIST of tensors and a bare-tensor scan walked
   straight past it. Nets 1 and 2 are blind to this class by construction: these tensors
   are built once in ``__init__``, which both of them exempt.

4. ``test_meta_device_forward`` runs the real nets on the ``meta`` device, which moves
   exactly what ``.to("cuda")`` moves and leaves exactly what it leaves. The only net
   that executes the model's arithmetic off-CPU, and it costs nothing (meta tensors carry
   no storage). Scoped to the CGENN family, where all three unmovable-tensor bugs lived.

Net 4 is the one that made a fake-CUDA forward unnecessary. ``FakeTensorMode`` was tried
first and does enforce device agreement (verified: binary ops, matmul and index_select all
raise ``FakeTensorDeviceMismatchError``), but it cannot traverse these models at all --
``to_dense_batch`` needs real counts and raises ``GuardOnDataDependentSymNode`` even with a
``ShapeEnv``. Net 4 sidesteps that by running the WRAPPER for real on CPU and handing the
net its captured arguments, which is why it works where FakeTensorMode did not. Do not
re-attempt the FakeTensorMode route; net 4 already covers what it was for.

Note that advanced indexing (``gpu[cpu_idx]``) is legal in torch and raises nowhere, which
is precisely why the b() bug degraded performance rather than crashing -- and equally why
net 4 sees only the crash class. Nets 1-3 are what cover the silent-cost class.
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
}
# files whose tensors must all be device-explicit (model code that runs per forward)
_SCANNED = [
    "experiments/baselines/CGENNLGATrGraphTransHybrid.py",
    "experiments/baselines/cgenn/cliffordalgebra.py",
    "experiments/baselines/cgenn/sparse_gp.py",
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
    # Both are plain aliases of `algebra.geometric_product_paths`, taken at init (gp.py:29,
    # fcgp.py:37). That buffer IS registered (non-persistent) on CliffordAlgebra, but the
    # alias does NOT follow it: `.to(device)` rebinds `_buffers[key]` to a NEW tensor and
    # leaves the alias pointing at the original CPU one. So these are stale after a move,
    # and safe anyway for the reason at the top of this dict -- never a tensor OPERAND.
    # `.nonzero()` and `.sum()` run at init; `.size()` runs in the forward, inside
    # `_get_weight`, but returns python ints and never touches the device.
    ("FullyConnectedSteerableGeometricProductLayer", "product_paths"):
        "alias of the algebra's geometric_product_paths buffer; .nonzero()/.sum() at init "
        "and .size() in _get_weight -- shape only, never an operand",
    ("SteerableGeometricProductLayer", "product_paths"):
        "alias of the algebra's geometric_product_paths buffer; .nonzero()/.sum() at init "
        "and .size() in _get_weight -- shape only, never an operand",
}


# nn.Module's own bookkeeping lives in __dict__ too (_parameters, _buffers, _modules, ...).
# Those hold the REGISTERED tensors -- the ones `.to()` does move -- so scanning them would
# flag every model. Derived from a bare Module so it tracks torch rather than a hardcoded list.
_MODULE_INTERNALS = set(vars(torch.nn.Module()))


def _tensors_in(val, _depth=0):
    """Tensors reachable from a plain attribute -- directly, or inside a list/tuple/dict.

    Depth-limited: these are config-ish containers, not arbitrary object graphs.
    """
    if isinstance(val, torch.Tensor):
        return [val]
    if _depth >= 3:
        return []
    if isinstance(val, (list, tuple, set)):
        return [t for v in val for t in _tensors_in(v, _depth + 1)]
    if isinstance(val, dict):
        return [t for v in val.values() for t in _tensors_in(v, _depth + 1)]
    return []


@pytest.mark.skipif(not RUN_SLOW, reason="device hygiene sweep: set CGENN_COMPILE_GATES=1")
@pytest.mark.parametrize("model", MODELS)
def test_no_unmovable_tensor_attributes(model):
    """STRUCTURAL: no module may hold a forward-used tensor that `.to(device)` cannot move.

    The THIRD net in this file, and the one that would have caught the bug the other two
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
            if (name in held or name in _MODULE_INTERNALS
                    or (type(mod).__name__, name) in _UNMOVABLE_OK):
                continue
            # CONTAINERS TOO, not just bare tensors. CliffordAlgebra.grade_to_index was a
            # LIST of index tensors -- `.to(device)` cannot move it, and `norms()` (live via
            # MVSiLU and the normalization layer) used it to index CUDA sign buffers, so every
            # forward copied the index host->device. A bare-tensor scan walked straight past
            # it, which is exactly how it survived the first version of this test.
            found = _tensors_in(val)
            if found:
                shapes = ", ".join(str(tuple(s.shape)) for s in found[:4])
                offenders.append(f"{type(mod).__name__}.{name} "
                                 f"[{type(val).__name__}: {shapes}]")

    print(f"GATE-UNMOVABLE[{model}] unregistered tensor attributes = {len(offenders)}")
    assert not offenders, (
        f"{model}: tensor(s) held where `.to(device)` cannot reach them -- a plain "
        f"attribute or a functools.cached_property:\n  " + "\n  ".join(sorted(set(offenders)))
        + "\n\nRegister them with `register_buffer(..., persistent=False)` if they are "
          "derived constants, or add them to _UNMOVABLE_OK with the call-site reason they "
          "can never be a tensor operand in a forward.")


# ---------------------------------------------------------------- meta-device forward
# The FOURTH net, and the only one that runs the model's real arithmetic on a device that
# is not CPU. `.to("meta")` moves EXACTLY what `.to("cuda")` moves -- parameters and
# buffers -- and leaves behind EXACTLY what `.to("cuda")` leaves behind: plain __dict__
# tensors, cached_property values, tensors inside lists and inside plain (non-Module)
# objects. Mixing the two then raises "Tensor on device cpu is not on the expected device
# meta!", which is the same failure the GPU gives, on a CPU-only runner and with no data
# movement at all (meta tensors carry no storage, so the forward is ~free).
#
# This is what the file's header records FakeTensorMode failing to do. It works here for
# one reason: it is applied to the NET, with the wrapper's data-dependent preprocessing
# (`pair.nonzero`, `to_dense_batch`) already executed for real on CPU and its outputs
# handed in. Same trick the BREAKS gate uses to get a cold model its own arguments.
#
# Scope is the CGENN family: all three unmovable-tensor bugs lived in CliffordAlgebra, it
# is shared by all three, and this is where a fourth would land. Extending it needs only
# a model whose net can be called with captured tensor arguments.
#
# Known blind spot, stated so nobody over-trusts a green run: `matmul` does NOT raise on a
# meta/cpu mix (measured -- it promotes silently), and advanced indexing `meta[cpu_idx]`
# does not raise either, exactly as `cuda[cpu_idx]` does not. So this catches the CRASH
# class (pointwise ops against an unmoved constant -- all three bugs found here) and not
# the SILENT-COST class, which is what the other two nets in this file are for.
META_MODELS = ["tag_cgenn", "tag_CGENNLGATrGraphTrans", "tag_CGENNLGATrGraphGPS"]


def _to_meta(v):
    if torch.is_tensor(v):
        return v.to("meta")
    if isinstance(v, (list, tuple)):
        return type(v)(_to_meta(x) for x in v)
    if isinstance(v, dict):
        return {k: _to_meta(x) for k, x in v.items()}
    return v


def _capture_net_args(exp):
    """Run one real CPU forward and keep the arguments the wrapper passed to the net."""
    captured = {}
    orig = exp.model.net.forward

    def spy(*a, **k):
        captured["a"], captured["k"] = a, k
        return orig(*a, **k)

    exp.model.net.forward = spy
    try:
        with torch.no_grad():
            exp._get_ypred_and_label(_fixed_batch(exp))
    finally:
        exp.model.net.forward = orig
    return captured["a"], captured["k"]


@pytest.mark.parametrize("model", META_MODELS)
def test_meta_device_forward(model):
    """A forward on a non-CPU device completes -- i.e. nothing the model uses stayed behind.

    Non-vacuity is asserted, not assumed. Every CliffordAlgebra buffer is unregistered in
    turn -- moved out of `_buffers` into `__dict__` as a CPU tensor, which is exactly the
    state `functools.cached_property` leaves you in -- and the forward re-run. At least one
    must fail, or this gate is watching nothing. A gate for a bug class that only appears
    on hardware the CI does not have is worth exactly what its proof of failure is worth.

    The list it prints is the useful by-product: those are the algebra tables that reach a
    device-checked op on the live forward path. Two things it taught, both of which the
    first version of this test got wrong by hardcoding one buffer on one algebra:

      * `_alpha_signs` is NOT among them -- `alpha()` is not reached from this forward, so
        pinning the self-check to it passed vacuously.
      * `tag_CGENNLGATrGraphTrans` holds TWO CliffordAlgebra instances (`net.algebra`,
        serving `mv_bridge`, and `net.cgenn.algebra`, serving the CGENN block). Only the
        second one's buffers are device-checked operands; `mv_bridge` reaches its algebra
        through embed/get_grade, which are index and slice ops. So every instance is swept,
        and the assertion is over the model, not over one of them.

    That second point is the documented blind spot in the flesh: an unmovable tensor on
    `net.algebra` would be a silent per-forward host-to-device copy that this gate cannot
    see. `test_no_unmovable_tensor_attributes` is structural and covers both instances.
    """
    exp = _build(model, float64=False)
    args, kwargs = _capture_net_args(exp)
    net = exp.model.net.to("meta")
    with torch.no_grad():
        out = net(*_to_meta(args), **_to_meta(kwargs))
    assert out.device.type == "meta"
    print(f"GATE-META[{model}] forward completed on a non-CPU device")

    algebras = [(n, m) for n, m in net.named_modules()
                if type(m).__name__ == "CliffordAlgebra"]
    assert algebras, f"{model}: no CliffordAlgebra to mutate -- self-check void"
    live = []
    for path, algebra in algebras:
        for name in [n for n, _ in algebra.named_buffers(recurse=False)]:
            stashed = algebra._buffers.pop(name)
            # built fresh, not copied: `.to("cpu")` on a meta tensor raises (no storage)
            algebra.__dict__[name] = torch.ones(stashed.shape, dtype=stashed.dtype,
                                                device="cpu")
            try:
                with torch.no_grad():
                    net(*_to_meta(args), **_to_meta(kwargs))
            except RuntimeError as e:
                if "device" in str(e):
                    live.append(f"{path}.{name}" if path else name)
            finally:
                del algebra.__dict__[name]
                algebra._buffers[name] = stashed
    print(f"GATE-META[{model}] self-check: unregistering any of {live} fails the forward")
    assert live, (
        f"{model}: no CliffordAlgebra buffer, left on CPU, breaks the meta forward -- so "
        f"this gate would pass with every one of them unmovable. Watching nothing.")
